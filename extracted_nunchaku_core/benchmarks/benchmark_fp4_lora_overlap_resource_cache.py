from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import native_fp4.training as training_impl  # noqa: E402
from native_fp4 import NunchakuFP4LoRALinear  # noqa: E402


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    da = a.float()
    db = b.float()
    diff = (da - db).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "rel_l2": float((da - db).norm().item() / (db.norm().item() + 1e-12)),
    }


def zero_grads(module: torch.nn.Module, x: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    x.grad = None


def make_module(
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    rank: int,
    lowrank_dtype: torch.dtype,
    overlap_lora_grad_min_rows: int,
) -> NunchakuFP4LoRALinear:
    module = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=rank,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        train_bias=True,
        cache_lora_act=True,
        fuse_lora_dx=True,
        cache_fused_lora_dx=True,
        overlap_lora_grad=True,
        overlap_lora_grad_min_rows=overlap_lora_grad_min_rows,
        zero_lora_up_fast_path=True,
    )
    module.refresh_fused_lora_dx_cache()
    return module


def sync_lora(dst: NunchakuFP4LoRALinear, src: NunchakuFP4LoRALinear) -> None:
    with torch.no_grad():
        dst.lora_down.copy_(src.lora_down)
        dst.lora_up.copy_(src.lora_up)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)
    dst.clear_lora_up_zero_fast_path()
    dst.refresh_fused_lora_dx_cache()


def train_step(module: NunchakuFP4LoRALinear, x: torch.Tensor, dy: torch.Tensor) -> None:
    zero_grads(module, x)
    y = module(x)
    loss = (y.float() * dy.float()).sum()
    loss.backward()


def time_cuda_and_wall(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    cuda_ms: list[float] = []
    wall_start = time.perf_counter()
    for _ in range(iters):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms.append(start_event.elapsed_time(end_event))
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / float(iters)
    return {
        "cuda_event_ms": float(sum(cuda_ms) / len(cuda_ms)),
        "wall_ms": float(wall_ms),
    }


def grad_snapshot(module: NunchakuFP4LoRALinear, x: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    train_step(module, x, dy)
    out = {
        "dx": x.grad.detach().clone(),
        "d_lora_down": module.lora_down.grad.detach().clone(),
        "d_lora_up": module.lora_up.grad.detach().clone(),
    }
    if module.bias is not None and module.bias.grad is not None:
        out["d_bias"] = module.bias.grad.detach().clone()
    zero_grads(module, x)
    return out


def count_overlap_resources(module: NunchakuFP4LoRALinear) -> int:
    return len(getattr(module.fp4_backward, "_lora_overlap_resources_by_key", {}))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate reusable CUDA stream/event resources for FP4 LoRA overlap backward")
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--overlap-lora-grad-min-rows", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)
    device = "cuda"

    weight = torch.randn(args.out_features, args.in_features, device=device, dtype=dtype) / (args.in_features**0.5)
    bias = torch.randn(args.out_features, device=device, dtype=dtype) * 0.01
    x_cached = torch.randn(args.m, args.in_features, device=device, dtype=dtype, requires_grad=True)
    x_uncached = x_cached.detach().clone().requires_grad_(True)
    dy = torch.randn(args.m, args.out_features, device=device, dtype=dtype)

    cached = make_module(
        weight,
        bias,
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
    )
    uncached = make_module(
        weight,
        bias,
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
    )
    sync_lora(uncached, cached)

    original_resource_cache_flag = training_impl._ENABLE_OVERLAP_RESOURCE_CACHE
    training_impl._ENABLE_OVERLAP_RESOURCE_CACHE = True
    try:
        cached_metrics = time_cuda_and_wall(lambda: train_step(cached, x_cached, dy), args.warmup, args.iters)
        cached_resource_count = count_overlap_resources(cached)
        cached_grads = grad_snapshot(cached, x_cached, dy)

        training_impl._ENABLE_OVERLAP_RESOURCE_CACHE = False
        uncached_metrics = time_cuda_and_wall(lambda: train_step(uncached, x_uncached, dy), args.warmup, args.iters)
        uncached_grads = grad_snapshot(uncached, x_uncached, dy)
    finally:
        training_impl._ENABLE_OVERLAP_RESOURCE_CACHE = original_resource_cache_flag

    errors = {name: tensor_error(cached_grads[name], uncached_grads[name]) for name in cached_grads}
    payload = {
        "experiment": "fp4_lora_overlap_resource_cache",
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "overlap_lora_grad_min_rows": args.overlap_lora_grad_min_rows,
        },
        "latency_ms": {
            "cached": cached_metrics,
            "uncached_stream_event_allocation": uncached_metrics,
        },
        "speedups": {
            "wall_cached_vs_uncached": uncached_metrics["wall_ms"] / cached_metrics["wall_ms"],
            "cuda_event_cached_vs_uncached": uncached_metrics["cuda_event_ms"] / cached_metrics["cuda_event_ms"],
        },
        "resource_cache": {
            "cached_entries_after_warmup": cached_resource_count,
            "uncached_entries_after_warmup": count_overlap_resources(uncached),
        },
        "errors_cached_vs_uncached": errors,
        "checks": {
            "dx_rel_l2_lt_5e-4": errors["dx"]["rel_l2"] < 5e-4,
            "lora_grad_rel_l2_lt_1e-6": all(
                errors[name]["rel_l2"] < 1e-6 for name in ("d_lora_down", "d_lora_up")
            ),
            "bias_grad_rel_l2_lt_1e-6": "d_bias" not in errors or errors["d_bias"]["rel_l2"] < 1e-6,
            "cached_resource_created": cached_resource_count > 0,
            "uncached_resource_not_retained": count_overlap_resources(uncached) == 0,
        },
    }
    payload["all_passed"] = all(payload["checks"].values())

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "latest_fp4_lora_overlap_resource_cache.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
