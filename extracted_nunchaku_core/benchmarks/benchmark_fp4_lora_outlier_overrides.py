from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import (  # noqa: E402
    FP4LoRAConfig,
    convert_linear_to_fp4_lora,
    fp4_lora_config_overrides_from_outlier_report,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark FP4 LoRA outlier-driven config override overhead.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--override-rank", type=int, default=64)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--frozen-residual-rank", type=int, default=32)
    p.add_argument("--frozen-residual-init", choices=["none", "residual_svd"], default="residual_svd")
    p.add_argument("--fuse-lora-dx", action="store_true", default=True)
    p.add_argument("--no-fuse-lora-dx", dest="fuse_lora_dx", action="store_false")
    p.add_argument("--cache-fused-lora-dx", action="store_true", default=True)
    p.add_argument("--no-cache-fused-lora-dx", dest="cache_fused_lora_dx", action="store_false")
    p.add_argument("--outlier-json", type=str, default="results/latest_fp4_lora_activation_grad_outliers.json")
    p.add_argument("--synthetic-module", type=str, default="layers.0.q_proj")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))
    return float(sum(values) / len(values))


def zero_grads(module: torch.nn.Module, x: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    x.grad = None


def train_step_ms(module: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int) -> float:
    def fn() -> None:
        zero_grads(module, x)
        y = module(x)
        loss = y.float().square().mean()
        loss.backward()

    ms = time_cuda(fn, warmup=warmup, iters=iters)
    zero_grads(module, x)
    return ms


def load_or_synthetic_report(path: str, synthetic_module: str, override_rank: int) -> Mapping:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidates = data.get("summary", {}).get("rank_bump_candidates", [])
        if candidates:
            return data
    return {
        "summary": {
            "rank_bump_candidates": [
                {
                    "module": synthetic_module,
                    "suggested_rank": int(override_rank),
                    "source": "synthetic_fallback",
                }
            ]
        }
    }


def build_model(
    dense_state: dict[str, torch.Tensor],
    hidden: int,
    layers: int,
    dtype: torch.dtype,
    cfg: FP4LoRAConfig,
    overrides: Mapping[str, FP4LoRAConfig] | None,
) -> torch.nn.Module:
    model = TinyModel(hidden, layers, dtype=dtype).cuda()
    model.load_state_dict(dense_state)
    model, _ = convert_linear_to_fp4_lora(
        model,
        cfg,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        config_overrides=overrides,
    )
    return model


def rank_summary(model: torch.nn.Module) -> dict[str, int]:
    return {name: child.requested_rank for name, child in iter_fp4_lora_modules(model)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)

    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)
    dense = TinyModel(args.hidden, args.layers, dtype=dtype).cuda()
    dense_state = {key: value.detach().clone() for key, value in dense.state_dict().items()}

    cfg = FP4LoRAConfig(
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        init="zero",
        frozen_residual_rank=args.frozen_residual_rank,
        frozen_residual_init=args.frozen_residual_init,
        fuse_lora_dx=args.fuse_lora_dx,
        cache_fused_lora_dx=args.cache_fused_lora_dx,
    )
    report = load_or_synthetic_report(args.outlier_json, args.synthetic_module, args.override_rank)
    overrides = fp4_lora_config_overrides_from_outlier_report(
        report,
        cfg,
        min_rank=args.rank,
        max_rank=args.override_rank,
        force_init="zero",
        disable_fuse_frozen_residual_dx=True,
    )

    base_model = build_model(dense_state, args.hidden, args.layers, dtype, cfg, overrides=None)
    override_model = build_model(dense_state, args.hidden, args.layers, dtype, cfg, overrides=overrides)
    x_base = torch.randn(args.batch, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
    x_override = x_base.detach().clone().requires_grad_(True)

    base_ms = train_step_ms(base_model, x_base, args.warmup, args.iters)
    override_ms = train_step_ms(override_model, x_override, args.warmup, args.iters)

    payload = {
        "experiment": "fp4_lora_outlier_override_overhead",
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "layers": args.layers,
            "rank": args.rank,
            "override_rank": args.override_rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "frozen_residual_rank": args.frozen_residual_rank,
            "frozen_residual_init": args.frozen_residual_init,
            "fuse_lora_dx": args.fuse_lora_dx,
            "cache_fused_lora_dx": args.cache_fused_lora_dx,
        },
        "outlier_json": args.outlier_json,
        "overrides": {
            name: {
                "rank": override.rank,
                "init": override.init,
                "fuse_frozen_residual_dx": override.fuse_frozen_residual_dx,
            }
            for name, override in overrides.items()
        },
        "rank_summary": {
            "base": rank_summary(base_model),
            "override": rank_summary(override_model),
        },
        "latency_ms": {
            "base_train_step": base_ms,
            "override_train_step": override_ms,
        },
        "overhead": {
            "override_over_base": override_ms / base_ms,
            "extra_ms": override_ms - base_ms,
        },
        "checks": {
            "has_overrides": bool(overrides),
            "latency_positive": base_ms > 0 and override_ms > 0,
        },
    }
    payload["all_passed"] = bool(all(payload["checks"].values()))

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"fp4_lora_outlier_override_overhead_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_fp4_lora_outlier_override_overhead.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {latest_path}")


if __name__ == "__main__":
    main()
