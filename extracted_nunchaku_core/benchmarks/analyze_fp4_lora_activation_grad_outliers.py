from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from typing import Any

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import (  # noqa: E402
    FP4LoRAConfig,
    NunchakuFP4LoRALinear,
    convert_linear_to_fp4_lora,
    iter_fp4_lora_modules,
)


class TinyBlock(torch.nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(hidden, hidden, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.q_proj(x)) + self.k_proj(x))


class TinyModel(torch.nn.Module):
    def __init__(self, hidden: int, layers: int, dtype: torch.dtype):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(hidden, dtype) for _ in range(layers)])
        self.lm_head = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class ChannelStats:
    def __init__(self, channels: int):
        self.channels = int(channels)
        self.rows = 0
        self.sum_abs = torch.zeros(channels, dtype=torch.float64)
        self.sum_sq = torch.zeros(channels, dtype=torch.float64)
        self.max_abs = torch.zeros(channels, dtype=torch.float64)

    def update(self, x: torch.Tensor) -> None:
        x2d = x.detach().reshape(-1, x.shape[-1]).float().abs().cpu()
        if x2d.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x2d.shape[1]}")
        self.rows += x2d.shape[0]
        self.sum_abs += x2d.double().sum(dim=0)
        self.sum_sq += x2d.double().square().sum(dim=0)
        self.max_abs = torch.maximum(self.max_abs, x2d.double().amax(dim=0))

    def summarize(self, topk: int) -> dict[str, Any]:
        if self.rows <= 0:
            return {"rows": 0, "channels": self.channels}
        mean_abs = self.sum_abs / float(self.rows)
        rms = torch.sqrt(self.sum_sq / float(self.rows))
        topk = min(int(topk), self.channels)
        top_values, top_indices = torch.topk(self.max_abs, k=topk)
        median = torch.quantile(self.max_abs, 0.50).clamp_min(1e-12)
        q90 = torch.quantile(self.max_abs, 0.90)
        q99 = torch.quantile(self.max_abs, 0.99)
        return {
            "rows": int(self.rows),
            "channels": int(self.channels),
            "mean_abs_mean": float(mean_abs.mean().item()),
            "mean_abs_max": float(mean_abs.max().item()),
            "rms_mean": float(rms.mean().item()),
            "rms_max": float(rms.max().item()),
            "channel_absmax_q50": float(median.item()),
            "channel_absmax_q90": float(q90.item()),
            "channel_absmax_q99": float(q99.item()),
            "channel_absmax_max": float(self.max_abs.max().item()),
            "channel_absmax_max_over_median": float((self.max_abs.max() / median).item()),
            "channel_absmax_q99_over_median": float((q99 / median).item()),
            "top_channels": [
                {"index": int(idx.item()), "absmax": float(value.item())}
                for idx, value in zip(top_indices, top_values, strict=False)
            ],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze FP4 LoRA activation/gradient outlier correlation.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--override-rank", type=int, default=64)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--target-modules", type=str, default="q_proj,down_proj")
    p.add_argument("--sensitive-module", type=str, default="layers.1.down_proj")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--rank-bump-threshold", type=float, default=6.0)
    p.add_argument("--smooth-correlation-threshold", type=float, default=0.6)
    p.add_argument("--inject-outliers", action="store_true")
    p.add_argument("--outlier-channel", type=int, default=0)
    p.add_argument("--outlier-scale", type=float, default=24.0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.numel() != b.numel() or a.numel() < 2:
        return None
    ra = rankdata(a.double())
    rb = rankdata(b.double())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.linalg.vector_norm(ra) * torch.linalg.vector_norm(rb)
    if float(denom.item()) == 0.0:
        return None
    return float((ra @ rb / denom).item())


def attach_hooks(
    model: torch.nn.Module,
) -> tuple[dict[str, dict[str, ChannelStats]], list[torch.utils.hooks.RemovableHandle]]:
    stats: dict[str, dict[str, ChannelStats]] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    for name, module in iter_fp4_lora_modules(model):
        stats[name] = {
            "activation": ChannelStats(module.in_features),
            "grad_output": ChannelStats(module.out_features),
        }

        def pre_hook(_module, inputs, full_name=name):
            stats[full_name]["activation"].update(inputs[0])

        def bwd_hook(_module, _grad_input, grad_output, full_name=name):
            if grad_output and grad_output[0] is not None:
                stats[full_name]["grad_output"].update(grad_output[0])

        hooks.append(module.register_forward_pre_hook(pre_hook))
        hooks.append(module.register_full_backward_hook(bwd_hook))

    return stats, hooks


def recommendation(
    act_summary: dict[str, Any],
    grad_summary: dict[str, Any],
    corr: float | None,
    rank_bump_threshold: float,
    smooth_correlation_threshold: float,
) -> dict[str, Any]:
    act_score = float(act_summary.get("channel_absmax_max_over_median", 0.0))
    grad_score = float(grad_summary.get("channel_absmax_max_over_median", 0.0))
    needs_rank_bump = act_score >= rank_bump_threshold or grad_score >= rank_bump_threshold
    smooth_candidate = corr is not None and corr >= smooth_correlation_threshold
    return {
        "rank_bump": bool(needs_rank_bump),
        "smooth_bwd_candidate": bool(smooth_candidate),
        "reason": (
            "activation/grad outlier score is high"
            if needs_rank_bump
            else "outlier score below rank-bump threshold"
        ),
        "smooth_note": (
            "activation and grad-output channel ranks are aligned"
            if smooth_candidate
            else "do not infer static backward smooth from activation stats yet"
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)

    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)
    model = TinyModel(args.hidden, args.layers, dtype=dtype).cuda()

    cfg = FP4LoRAConfig(
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        init="zero",
        frozen_residual_rank=32,
        frozen_residual_init="residual_svd",
        fuse_lora_dx=True,
        cache_fused_lora_dx=True,
    )
    override_cfg = replace(cfg, rank=args.override_rank)
    target_modules = tuple(item.strip() for item in args.target_modules.split(",") if item.strip())
    model, replaced = convert_linear_to_fp4_lora(
        model,
        cfg,
        target_modules=target_modules,
        exclude_modules=("lm_head",),
        config_overrides={args.sensitive_module: override_cfg},
    )
    if not replaced:
        raise RuntimeError("No Linear modules were converted to FP4 LoRA")

    stats, hooks = attach_hooks(model)
    try:
        for step in range(args.steps):
            x = torch.randn(args.batch, args.hidden, device="cuda", dtype=dtype)
            if args.inject_outliers:
                channel = args.outlier_channel % args.hidden
                x[:, channel] = x[:, channel] * float(args.outlier_scale)
            x.requires_grad_(True)
            y = model(x)
            loss = y.float().square().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
    finally:
        for hook in hooks:
            hook.remove()

    records = []
    for name, module_stats in stats.items():
        act_summary = module_stats["activation"].summarize(args.topk)
        grad_summary = module_stats["grad_output"].summarize(args.topk)
        corr = spearman(module_stats["activation"].max_abs, module_stats["grad_output"].max_abs)
        rec = recommendation(
            act_summary=act_summary,
            grad_summary=grad_summary,
            corr=corr,
            rank_bump_threshold=args.rank_bump_threshold,
            smooth_correlation_threshold=args.smooth_correlation_threshold,
        )
        module = dict(iter_fp4_lora_modules(model))[name]
        records.append(
            {
                "module": name,
                "kind": name.split(".")[-1],
                "in_features": module.in_features,
                "out_features": module.out_features,
                "requested_rank": module.requested_rank,
                "effective_rank": module.rank,
                "activation": act_summary,
                "grad_output": grad_summary,
                "spearman_activation_absmax_vs_grad_output_absmax": corr,
                "recommendation": rec,
            }
        )

    rank_bump_candidates = [
        {
            "module": record["module"],
            "kind": record["kind"],
            "requested_rank": record["requested_rank"],
            "suggested_rank": max(record["effective_rank"], args.override_rank),
            "activation_score": record["activation"].get("channel_absmax_max_over_median"),
            "grad_output_score": record["grad_output"].get("channel_absmax_max_over_median"),
        }
        for record in records
        if record["recommendation"]["rank_bump"]
    ]
    smooth_candidates = [
        {
            "module": record["module"],
            "kind": record["kind"],
            "spearman": record["spearman_activation_absmax_vs_grad_output_absmax"],
        }
        for record in records
        if record["recommendation"]["smooth_bwd_candidate"]
    ]
    payload = {
        "experiment": "fp4_lora_activation_grad_outliers",
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "layers": args.layers,
            "steps": args.steps,
            "rank": args.rank,
            "override_rank": args.override_rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "target_modules": list(target_modules),
            "sensitive_module": args.sensitive_module,
            "inject_outliers": args.inject_outliers,
            "outlier_channel": args.outlier_channel,
            "outlier_scale": args.outlier_scale,
        },
        "thresholds": {
            "rank_bump_threshold": args.rank_bump_threshold,
            "smooth_correlation_threshold": args.smooth_correlation_threshold,
        },
        "replaced": replaced,
        "module_records": sorted(records, key=lambda item: item["module"]),
        "summary": {
            "rank_bump_candidates": rank_bump_candidates,
            "smooth_bwd_candidates": smooth_candidates,
        },
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"fp4_lora_activation_grad_outliers_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_fp4_lora_activation_grad_outliers.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {latest_path}")


if __name__ == "__main__":
    main()
