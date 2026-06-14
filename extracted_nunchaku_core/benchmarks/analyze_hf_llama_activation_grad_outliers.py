from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from typing import Any

import torch
from torch import nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark_hf_llama_fp4_lora_finetuning import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_ID,
    build_batch_from_stream,
    dtype_from_name,
    ensure_model_downloaded,
    load_tokenizer,
    load_wikitext_token_stream,
    set_hf_mirror,
)
from native_fp4 import DEFAULT_FP4_LORA_TARGET_MODULES  # noqa: E402


FP4_GROUP_SIZE = 16
FP4_QMAX = 6.0
DEFAULT_EXCLUDE_MODULES = ("lm_head",)


class TensorOutlierStats:
    def __init__(self, channels: int, *, group_size: int, topk: int):
        self.channels = int(channels)
        self.group_size = int(group_size)
        self.topk = int(topk)
        self.rows = 0
        self.sum_abs = torch.zeros(channels, dtype=torch.float64)
        self.sum_sq = torch.zeros(channels, dtype=torch.float64)
        self.max_abs = torch.zeros(channels, dtype=torch.float64)
        self.block_count = 0
        self.block_dominance_sum = 0.0
        self.block_dominance_sq_sum = 0.0
        self.block_dominance_max = 0.0
        self.block_resolution_sum = 0.0
        self.block_resolution_sq_sum = 0.0
        self.block_resolution_max = 0.0
        self._top_block_dominance = torch.empty(0, dtype=torch.float64)

    def update(self, tensor: torch.Tensor) -> None:
        x = tensor.detach().reshape(-1, tensor.shape[-1]).float().abs().cpu()
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        rows = int(x.shape[0])
        if rows <= 0:
            return

        x64 = x.double()
        self.rows += rows
        self.sum_abs += x64.sum(dim=0)
        self.sum_sq += x64.square().sum(dim=0)
        self.max_abs = torch.maximum(self.max_abs, x64.amax(dim=0))

        pad = (-self.channels) % self.group_size
        if pad:
            x64 = torch.nn.functional.pad(x64, (0, pad))
        blocks = x64.reshape(rows, -1, self.group_size)
        block_absmax = blocks.amax(dim=-1)
        block_mean_abs = blocks.mean(dim=-1).clamp_min(1e-12)
        block_dominance = block_absmax / block_mean_abs
        block_resolution = block_absmax / (FP4_QMAX * block_mean_abs)
        self._update_block_metric(block_dominance, block_resolution)

    def _update_block_metric(self, dominance: torch.Tensor, resolution: torch.Tensor) -> None:
        dom = dominance.reshape(-1).double()
        res = resolution.reshape(-1).double()
        if dom.numel() == 0:
            return

        self.block_count += int(dom.numel())
        self.block_dominance_sum += float(dom.sum().item())
        self.block_dominance_sq_sum += float(dom.square().sum().item())
        self.block_dominance_max = max(self.block_dominance_max, float(dom.max().item()))
        self.block_resolution_sum += float(res.sum().item())
        self.block_resolution_sq_sum += float(res.square().sum().item())
        self.block_resolution_max = max(self.block_resolution_max, float(res.max().item()))

        keep = min(self.topk, dom.numel() + self._top_block_dominance.numel())
        if keep > 0:
            merged = torch.cat((self._top_block_dominance, dom.cpu()))
            self._top_block_dominance = torch.topk(merged, k=keep).values

    def summarize(self) -> dict[str, Any]:
        if self.rows <= 0:
            return {"rows": 0, "channels": self.channels, "group_size": self.group_size}

        mean_abs = self.sum_abs / float(self.rows)
        rms = torch.sqrt(self.sum_sq / float(self.rows))
        channel_median = torch.quantile(self.max_abs, 0.50).clamp_min(1e-12)
        channel_q90 = torch.quantile(self.max_abs, 0.90)
        channel_q99 = torch.quantile(self.max_abs, 0.99)
        channel_max = self.max_abs.max()
        topk = min(self.topk, self.channels)
        top_values, top_indices = torch.topk(self.max_abs, k=topk)

        block_dominance_mean = self.block_dominance_sum / max(self.block_count, 1)
        block_resolution_mean = self.block_resolution_sum / max(self.block_count, 1)
        block_dominance_var = max(self.block_dominance_sq_sum / max(self.block_count, 1) - block_dominance_mean**2, 0.0)
        block_resolution_var = max(self.block_resolution_sq_sum / max(self.block_count, 1) - block_resolution_mean**2, 0.0)

        return {
            "rows": int(self.rows),
            "channels": int(self.channels),
            "group_size": int(self.group_size),
            "fp4_qmax": float(FP4_QMAX),
            "mean_abs_mean": float(mean_abs.mean().item()),
            "mean_abs_max": float(mean_abs.max().item()),
            "rms_mean": float(rms.mean().item()),
            "rms_max": float(rms.max().item()),
            "channel_absmax_q50": float(channel_median.item()),
            "channel_absmax_q90": float(channel_q90.item()),
            "channel_absmax_q99": float(channel_q99.item()),
            "channel_absmax_max": float(channel_max.item()),
            "channel_absmax_max_over_median": float((channel_max / channel_median).item()),
            "channel_absmax_q99_over_median": float((channel_q99 / channel_median).item()),
            "block_count": int(self.block_count),
            "block_absmax_over_mean_abs_mean": float(block_dominance_mean),
            "block_absmax_over_mean_abs_std": float(block_dominance_var**0.5),
            "block_absmax_over_mean_abs_max": float(self.block_dominance_max),
            "block_resolution_pressure_mean": float(block_resolution_mean),
            "block_resolution_pressure_std": float(block_resolution_var**0.5),
            "block_resolution_pressure_max": float(self.block_resolution_max),
            "top_channels": [
                {"index": int(idx.item()), "absmax": float(value.item())}
                for idx, value in zip(top_indices, top_values, strict=False)
            ],
            "top_block_absmax_over_mean_abs": [float(value.item()) for value in self._top_block_dominance],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect real HF/Llama activation and grad-output outlier stats for FP4 LoRA policies."
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--dataset-max-docs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dataset-offset-tokens", type=int, default=0)
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_FP4_LORA_TARGET_MODULES))
    parser.add_argument("--exclude-modules", nargs="+", default=list(DEFAULT_EXCLUDE_MODULES))
    parser.add_argument("--linear-prefix", type=str, default="model.layers.")
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--replace-layer-start", type=int, default=None)
    parser.add_argument("--replace-layer-end", type=int, default=None)
    parser.add_argument("--replace-name-substrings", nargs="*", default=None)
    parser.add_argument("--max-modules", type=int, default=0)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--suggested-rank", type=int, default=64)
    parser.add_argument("--rank-bump-threshold", type=float, default=8.0)
    parser.add_argument("--keep-dense-threshold", type=float, default=24.0)
    parser.add_argument("--block-pressure-threshold", type=float, default=8.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def load_model(model_dir: str, *, dtype: torch.dtype, attn_implementation: str | None):
    from transformers import AutoModelForCausalLM

    gc.collect()
    torch.cuda.empty_cache()
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": {"": "cuda:0"},
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def cleanup_model(model: nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def module_name_matches(full_name: str, child_name: str, patterns: tuple[str, ...]) -> bool:
    return any(full_name == pattern or child_name == pattern or full_name.endswith(f".{pattern}") for pattern in patterns)


def extract_layer_index(full_name: str, linear_prefix: str) -> int | None:
    if not full_name.startswith(linear_prefix):
        return None
    match = re.match(rf"^{re.escape(linear_prefix)}(\d+)\.", full_name)
    if match is None:
        return None
    return int(match.group(1))


def layer_is_selected(
    full_name: str,
    linear_prefix: str,
    layer_start: int | None,
    layer_end: int | None,
) -> bool:
    layer_idx = extract_layer_index(full_name, linear_prefix)
    if layer_idx is None:
        return True
    if layer_start is not None and layer_idx < layer_start:
        return False
    if layer_end is not None and layer_idx >= layer_end:
        return False
    return True


def name_is_selected(full_name: str, name_substrings: list[str] | None) -> bool:
    if not name_substrings:
        return True
    return any(substring in full_name for substring in name_substrings)


def effective_exclude_modules(args: argparse.Namespace) -> tuple[str, ...]:
    excludes = tuple(args.exclude_modules or ())
    if args.include_lm_head:
        excludes = tuple(name for name in excludes if name != "lm_head")
    return excludes


def select_linear_modules(model: nn.Module, args: argparse.Namespace) -> list[tuple[str, nn.Linear]]:
    targets = tuple(args.target_modules or ())
    excludes = effective_exclude_modules(args)
    selected: list[tuple[str, nn.Linear]] = []
    for full_name, child in model.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        child_name = full_name.rsplit(".", 1)[-1]
        is_target = module_name_matches(full_name, child_name, targets)
        if args.include_lm_head and full_name == "lm_head":
            is_target = True
        is_excluded = module_name_matches(full_name, child_name, excludes)
        if (
            is_target
            and not is_excluded
            and layer_is_selected(full_name, args.linear_prefix, args.replace_layer_start, args.replace_layer_end)
            and name_is_selected(full_name, args.replace_name_substrings)
        ):
            selected.append((full_name, child))
            if args.max_modules > 0 and len(selected) >= args.max_modules:
                break
    return selected


def enable_input_grads(model: nn.Module) -> torch.utils.hooks.RemovableHandle | None:
    for param in model.parameters():
        param.requires_grad_(False)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return None

    embeddings = model.get_input_embeddings()

    def require_grad_hook(_module, _inputs, output):
        output.requires_grad_(True)

    return embeddings.register_forward_hook(require_grad_hook)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.numel() != b.numel() or a.numel() < 2:
        return None
    order_a = torch.argsort(a.double())
    order_b = torch.argsort(b.double())
    ranks_a = torch.empty_like(order_a, dtype=torch.float64)
    ranks_b = torch.empty_like(order_b, dtype=torch.float64)
    ranks_a[order_a] = torch.arange(a.numel(), dtype=torch.float64)
    ranks_b[order_b] = torch.arange(b.numel(), dtype=torch.float64)
    ranks_a = ranks_a - ranks_a.mean()
    ranks_b = ranks_b - ranks_b.mean()
    denom = torch.linalg.vector_norm(ranks_a) * torch.linalg.vector_norm(ranks_b)
    if float(denom.item()) == 0.0:
        return None
    return float((ranks_a @ ranks_b / denom).item())


def attach_hooks(
    selected: list[tuple[str, nn.Linear]],
    *,
    group_size: int,
    topk: int,
    collect_grad: bool,
) -> tuple[dict[str, dict[str, TensorOutlierStats]], list[torch.utils.hooks.RemovableHandle]]:
    stats: dict[str, dict[str, TensorOutlierStats]] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    for full_name, module in selected:
        stats[full_name] = {
            "activation": TensorOutlierStats(module.in_features, group_size=group_size, topk=topk),
            "grad_output": TensorOutlierStats(module.out_features, group_size=group_size, topk=topk),
        }

        def pre_hook(_module, inputs, name=full_name):
            if inputs and inputs[0] is not None:
                stats[name]["activation"].update(inputs[0])

        hooks.append(module.register_forward_pre_hook(pre_hook))

        if collect_grad:

            def bwd_hook(_module, _grad_input, grad_output, name=full_name):
                if grad_output and grad_output[0] is not None:
                    stats[name]["grad_output"].update(grad_output[0])

            hooks.append(module.register_full_backward_hook(bwd_hook))

    return stats, hooks


def score_summary(summary: dict[str, Any]) -> float:
    channel_score = float(summary.get("channel_absmax_max_over_median", 0.0))
    block_score = float(summary.get("block_absmax_over_mean_abs_max", 0.0))
    return max(channel_score, block_score)


def build_recommendation(
    activation: dict[str, Any],
    grad_output: dict[str, Any],
    *,
    rank_bump_threshold: float,
    keep_dense_threshold: float,
    block_pressure_threshold: float,
) -> dict[str, Any]:
    activation_score = score_summary(activation)
    grad_output_score = score_summary(grad_output)
    block_pressure = max(
        float(activation.get("block_absmax_over_mean_abs_max", 0.0)),
        float(grad_output.get("block_absmax_over_mean_abs_max", 0.0)),
    )
    keep_dense = activation_score >= keep_dense_threshold or grad_output_score >= keep_dense_threshold
    rank_bump = keep_dense or activation_score >= rank_bump_threshold or grad_output_score >= rank_bump_threshold
    block_pressure_candidate = block_pressure >= block_pressure_threshold
    return {
        "rank_bump": bool(rank_bump),
        "keep_dense_candidate": bool(keep_dense),
        "block_pressure_candidate": bool(block_pressure_candidate),
        "activation_score": float(activation_score),
        "grad_output_score": float(grad_output_score),
        "block_pressure_score": float(block_pressure),
        "reason": (
            "severe activation/grad-output outlier; consider keeping dense or using residual/high-rank policy"
            if keep_dense
            else "activation/grad-output outlier score exceeds rank-bump threshold"
            if rank_bump
            else "outlier score below rank-bump threshold"
        ),
    }


def run_probe(model: nn.Module, batch: dict[str, torch.Tensor], *, backward: bool) -> float:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
    )
    loss = outputs.loss
    if backward:
        loss.backward()
        model.zero_grad(set_to_none=True)
    return float(loss.detach().item())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    torch.manual_seed(args.seed)
    set_hf_mirror(args.hf_endpoint)
    os.makedirs(args.results_dir, exist_ok=True)

    dtype = dtype_from_name(args.dtype)
    model_dir = ensure_model_downloaded(args.model_id, args.model_dir)
    tokenizer = load_tokenizer(model_dir)
    token_stream = load_wikitext_token_stream(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_max_docs=args.dataset_max_docs,
    )
    model = load_model(model_dir, dtype=dtype, attn_implementation=args.attn_implementation)
    selected = select_linear_modules(model, args)
    if not selected:
        cleanup_model(model)
        raise RuntimeError("No nn.Linear modules selected for outlier analysis")

    input_grad_hook = None if args.forward_only else enable_input_grads(model)
    stats, hooks = attach_hooks(
        selected,
        group_size=FP4_GROUP_SIZE,
        topk=args.topk,
        collect_grad=not args.forward_only,
    )

    losses: list[float] = []
    try:
        for step in range(args.steps):
            offset = args.dataset_offset_tokens + step * args.batch_size * args.seq_len
            batch = build_batch_from_stream(
                token_stream,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                offset_tokens=offset,
            )
            losses.append(run_probe(model, batch, backward=not args.forward_only))
    finally:
        for hook in hooks:
            hook.remove()
        if input_grad_hook is not None:
            input_grad_hook.remove()

    records: list[dict[str, Any]] = []
    selected_lookup = dict(selected)
    for module_name in stats:
        module = selected_lookup[module_name]
        activation = stats[module_name]["activation"].summarize()
        grad_output = stats[module_name]["grad_output"].summarize()
        corr = None
        if grad_output.get("rows", 0):
            corr = spearman(stats[module_name]["activation"].max_abs, stats[module_name]["grad_output"].max_abs)
        recommendation = build_recommendation(
            activation,
            grad_output,
            rank_bump_threshold=args.rank_bump_threshold,
            keep_dense_threshold=args.keep_dense_threshold,
            block_pressure_threshold=args.block_pressure_threshold,
        )
        records.append(
            {
                "module": module_name,
                "kind": module_name.rsplit(".", 1)[-1],
                "layer_idx": extract_layer_index(module_name, args.linear_prefix),
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "num_params": int(module.weight.numel() + (0 if module.bias is None else module.bias.numel())),
                "activation": activation,
                "grad_output": grad_output,
                "spearman_activation_absmax_vs_grad_output_absmax": corr,
                "recommendation": recommendation,
            }
        )

    rank_bump_candidates = [
        {
            "module": record["module"],
            "kind": record["kind"],
            "layer_idx": record["layer_idx"],
            "suggested_rank": int(max(args.rank, args.suggested_rank)),
            "activation_score": record["recommendation"]["activation_score"],
            "grad_output_score": record["recommendation"]["grad_output_score"],
            "block_pressure_score": record["recommendation"]["block_pressure_score"],
        }
        for record in records
        if record["recommendation"]["rank_bump"]
    ]
    keep_dense_candidates = [
        {
            "module": record["module"],
            "kind": record["kind"],
            "layer_idx": record["layer_idx"],
            "activation_score": record["recommendation"]["activation_score"],
            "grad_output_score": record["recommendation"]["grad_output_score"],
            "block_pressure_score": record["recommendation"]["block_pressure_score"],
        }
        for record in records
        if record["recommendation"]["keep_dense_candidate"]
    ]
    sorted_records = sorted(
        records,
        key=lambda item: max(item["recommendation"]["activation_score"], item["recommendation"]["grad_output_score"]),
        reverse=True,
    )

    payload = {
        "experiment": "hf_llama_activation_grad_outliers",
        "model_id": args.model_id,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "max_docs": args.dataset_max_docs,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "steps": args.steps,
            "offset_tokens": args.dataset_offset_tokens,
        },
        "selection": {
            "target_modules": args.target_modules,
            "exclude_modules": list(effective_exclude_modules(args)),
            "linear_prefix": args.linear_prefix,
            "include_lm_head": args.include_lm_head,
            "replace_layer_start": args.replace_layer_start,
            "replace_layer_end": args.replace_layer_end,
            "replace_name_substrings": args.replace_name_substrings,
            "max_modules": args.max_modules,
        },
        "thresholds": {
            "rank": args.rank,
            "suggested_rank": args.suggested_rank,
            "rank_bump_threshold": args.rank_bump_threshold,
            "keep_dense_threshold": args.keep_dense_threshold,
            "block_pressure_threshold": args.block_pressure_threshold,
            "fp4_group_size": FP4_GROUP_SIZE,
            "fp4_qmax": FP4_QMAX,
        },
        "losses": losses,
        "forward_only": bool(args.forward_only),
        "module_records": sorted(records, key=lambda item: item["module"]),
        "summary": {
            "rank_bump_candidates": rank_bump_candidates,
            "keep_dense_candidates": keep_dense_candidates,
            "most_outlier_modules": sorted_records[: min(args.topk, len(sorted_records))],
        },
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"hf_llama_activation_grad_outliers_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_hf_llama_activation_grad_outliers.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {latest_path}")

    cleanup_model(model)


if __name__ == "__main__":
    main()
