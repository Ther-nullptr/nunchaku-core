from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import NunchakuFP4LoRALinear  # noqa: E402


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
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


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    da = a.float()
    db = b.float()
    diff = (da - db).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "rel_l2": float((da - db).norm().item() / (db.norm().item() + 1e-12)),
    }


def train_step(module: NunchakuFP4LoRALinear, x: torch.Tensor, dy: torch.Tensor) -> None:
    zero_grads(module, x)
    y = module(x)
    loss = (y.float() * dy.float()).sum()
    loss.backward()


def grad_accum_step(
    module: NunchakuFP4LoRALinear,
    x: torch.Tensor,
    dy: torch.Tensor,
    steps: int,
) -> None:
    zero_grads(module, x)
    for _ in range(steps):
        y = module(x)
        loss = (y.float() * dy.float()).sum()
        loss.backward()


def make_module(args: argparse.Namespace, weight: torch.Tensor, bias: torch.Tensor, policy: str) -> NunchakuFP4LoRALinear:
    return NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lowrank_dtype=dtype_from_name(args.lowrank_dtype),
        init="gaussian",
        train_bias=False,
        cache_lora_act=True,
        fuse_lora_dx=True,
        cache_fused_lora_dx=True,
        backward_weight_policy=policy,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        overlap_lora_grad=args.overlap_lora_grad,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark FP4 backward qweight repack vs opt-in cache policy.")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--reuse-fused-dy-up-for-d-lora-down", action="store_true")
    p.add_argument("--overlap-lora-grad", action="store_true")
    p.add_argument("--overlap-lora-grad-min-rows", type=int, default=4096)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)

    dtype = dtype_from_name(args.dtype)
    if args.reuse_fused_dy_up_for_d_lora_down and dtype != dtype_from_name(args.lowrank_dtype):
        raise ValueError("--reuse-fused-dy-up-for-d-lora-down requires dtype == lowrank_dtype")

    x_repack = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype, requires_grad=True)
    x_cache = x_repack.detach().clone().requires_grad_(True)
    dy = torch.randn(args.m, args.out_features, device="cuda", dtype=dtype)
    weight = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)
    bias = torch.randn(args.out_features, device="cuda", dtype=dtype)

    repack = make_module(args, weight, bias, "repack")
    cache = make_module(args, weight, bias, "cache")
    with torch.no_grad():
        cache.lora_down.copy_(repack.lora_down)
        cache.lora_up.copy_(repack.lora_up)
    repack.refresh_fused_lora_dx_cache()
    cache.refresh_fused_lora_dx_cache()
    cache.refresh_backward_weight_cache()

    train_step(repack, x_repack, dy)
    train_step(cache, x_cache, dy)
    errors = {
        "x_grad_cache_vs_repack": tensor_error(x_cache.grad, x_repack.grad),
        "lora_down_grad_cache_vs_repack": tensor_error(cache.lora_down.grad, repack.lora_down.grad),
        "lora_up_grad_cache_vs_repack": tensor_error(cache.lora_up.grad, repack.lora_up.grad),
    }

    def repack_train_fn() -> None:
        train_step(repack, x_repack, dy)

    def cache_train_fn() -> None:
        train_step(cache, x_cache, dy)

    def repack_grad_accum_fn() -> None:
        grad_accum_step(repack, x_repack, dy, args.grad_accum_steps)

    def cache_grad_accum_fn() -> None:
        grad_accum_step(cache, x_cache, dy, args.grad_accum_steps)

    def repack_only_fn() -> torch.Tensor:
        return repack.fp4_backward.repack_qweight_for_backward()

    def cache_refresh_fn() -> torch.Tensor | None:
        cache.clear_backward_weight_cache()
        return cache.refresh_backward_weight_cache()

    def cache_hit_fn() -> torch.Tensor:
        return cache.fp4_backward.repack_qweight_for_backward()

    repack_only_ms = time_cuda(repack_only_fn, args.warmup, args.iters)
    cache_refresh_ms = time_cuda(cache_refresh_fn, args.warmup, args.iters)
    cache_hit_ms = time_cuda(cache_hit_fn, args.warmup, args.iters)
    repack_train_ms = time_cuda(repack_train_fn, args.warmup, args.iters)
    cache_train_ms = time_cuda(cache_train_fn, args.warmup, args.iters)
    repack_grad_accum_ms = time_cuda(repack_grad_accum_fn, args.warmup, args.iters)
    cache_grad_accum_ms = time_cuda(cache_grad_accum_fn, args.warmup, args.iters)
    zero_grads(repack, x_repack)
    zero_grads(cache, x_cache)

    cached_qweight = cache.fp4_backward._cached_qweight_bwd
    cached_qweight_bytes = 0 if cached_qweight is None else cached_qweight.numel() * cached_qweight.element_size()
    dense_weight_bytes = weight.numel() * weight.element_size()
    forward_qweight_bytes = repack.fp4_forward.qweight.numel() * repack.fp4_forward.qweight.element_size()
    x_grad_tol = 5e-6
    checks = {
        "cache_policy_has_cached_qweight": cached_qweight is not None,
        "repack_policy_has_no_cached_qweight": repack.fp4_backward._cached_qweight_bwd is None,
        "x_grad_rel_l2_lt_5e-6": errors["x_grad_cache_vs_repack"]["rel_l2"] < x_grad_tol,
        "lora_down_grad_rel_l2_lt_1e-6": errors["lora_down_grad_cache_vs_repack"]["rel_l2"] < 1e-6,
        "lora_up_grad_rel_l2_lt_1e-6": errors["lora_up_grad_cache_vs_repack"]["rel_l2"] < 1e-6,
        "latencies_positive": all(
            value > 0
            for value in (
                repack_only_ms,
                cache_refresh_ms,
                cache_hit_ms,
                repack_train_ms,
                cache_train_ms,
                repack_grad_accum_ms,
                cache_grad_accum_ms,
            )
        ),
    }

    payload = {
        "experiment": "fp4_lora_backward_weight_policy",
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "effective_rank": repack.rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "reuse_fused_dy_up_for_d_lora_down": args.reuse_fused_dy_up_for_d_lora_down,
            "overlap_lora_grad": args.overlap_lora_grad,
            "overlap_lora_grad_min_rows": args.overlap_lora_grad_min_rows,
            "grad_accum_steps": args.grad_accum_steps,
        },
        "bytes": {
            "dense_weight": dense_weight_bytes,
            "forward_qweight": forward_qweight_bytes,
            "cached_backward_qweight": cached_qweight_bytes,
            "cached_backward_qweight_vs_dense_weight": cached_qweight_bytes / dense_weight_bytes,
            "cached_backward_qweight_vs_forward_qweight": cached_qweight_bytes / forward_qweight_bytes,
        },
        "latency_ms": {
            "repack_only": repack_only_ms,
            "cache_refresh": cache_refresh_ms,
            "cache_hit": cache_hit_ms,
            "repack_train_step": repack_train_ms,
            "cache_train_step": cache_train_ms,
            "repack_grad_accum_total": repack_grad_accum_ms,
            "cache_grad_accum_total": cache_grad_accum_ms,
            "repack_grad_accum_per_step": repack_grad_accum_ms / args.grad_accum_steps,
            "cache_grad_accum_per_step": cache_grad_accum_ms / args.grad_accum_steps,
        },
        "speedups": {
            "cache_train_step_vs_repack": repack_train_ms / cache_train_ms,
            "cache_grad_accum_total_vs_repack": repack_grad_accum_ms / cache_grad_accum_ms,
            "cache_hit_vs_repack_only": repack_only_ms / cache_hit_ms,
        },
        "tolerances": {
            "x_grad_policy_rel_l2": x_grad_tol,
            "lora_grad_policy_rel_l2": 1e-6,
        },
        "errors": errors,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stamped = os.path.join(args.results_dir, f"fp4_lora_backward_weight_policy_{stamp}.json")
    latest = os.path.join(args.results_dir, "latest_fp4_lora_backward_weight_policy.json")
    for path in (stamped, latest):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {latest}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
