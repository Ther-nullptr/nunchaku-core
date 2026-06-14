from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

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


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    da = a.float()
    db = b.float()
    diff = (da - db).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "rel_l2": float((da - db).norm().item() / (db.norm().item() + 1e-12)),
    }


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values: list[float] = []
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


def make_module(
    *,
    weight: torch.Tensor,
    bias: torch.Tensor,
    rank: int,
    lowrank_dtype: torch.dtype,
    frozen_residual_rank: int,
    frozen_residual_init: str,
    fuse_lowrank_forward: bool,
    fuse_lora_dx: bool,
    cache_fused_lora_dx: bool,
    overlap_lora_grad: bool,
    overlap_lora_grad_min_rows: int,
    zero_lora_up_fast_path: bool,
) -> NunchakuFP4LoRALinear:
    return NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=rank,
        lowrank_dtype=lowrank_dtype,
        init="zero",
        frozen_residual_rank=frozen_residual_rank,
        frozen_residual_init=frozen_residual_init,  # type: ignore[arg-type]
        train_bias=True,
        cache_lora_act=True,
        fuse_lowrank_forward=fuse_lowrank_forward,
        fuse_lora_dx=fuse_lora_dx,
        cache_fused_lora_dx=cache_fused_lora_dx,
        overlap_lora_grad=overlap_lora_grad,
        overlap_lora_grad_min_rows=overlap_lora_grad_min_rows,
        zero_lora_up_fast_path=zero_lora_up_fast_path,
    )


def sync_lora(dst: NunchakuFP4LoRALinear, src: NunchakuFP4LoRALinear) -> None:
    with torch.no_grad():
        dst.lora_down.copy_(src.lora_down)
        dst.lora_up.copy_(src.lora_up)
        if dst.has_frozen_residual and src.has_frozen_residual:
            dst.frozen_residual_down.copy_(src.frozen_residual_down)
            dst.frozen_residual_up.copy_(src.frozen_residual_up)
    dst.clear_lora_up_zero_fast_path()
    if dst.zero_lora_up_fast_path and bool(torch.count_nonzero(dst.lora_up) == 0):
        # This path is used once during benchmark setup, not in the hot path.
        dst.mark_lora_up_zero_fast_path()


def refresh_optional_caches(module: NunchakuFP4LoRALinear) -> dict[str, int]:
    module.refresh_fused_lora_forward_cache()
    module.refresh_fused_lora_dx_cache()
    return {
        "forward_cache_present": int(
            module._cached_lora_down_fwd_packed is not None and module._cached_lora_up_fwd_packed is not None
        ),
        "dx_cache_present": int(
            module._cached_lora_down_bwd_packed is not None and module._cached_lora_up_bwd_packed is not None
        ),
    }


def benchmark_forward(module: NunchakuFP4LoRALinear, x: torch.Tensor, warmup: int, iters: int) -> float:
    def fn() -> None:
        y = module(x)
        _ = y.float().sum()

    return time_cuda(fn, warmup, iters)


def benchmark_train_step(
    module: NunchakuFP4LoRALinear,
    x: torch.Tensor,
    dy: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    def fn() -> None:
        zero_grads(module, x)
        y = module(x)
        loss = (y.float() * dy.float()).sum()
        loss.backward()

    ms = time_cuda(fn, warmup, iters)
    zero_grads(module, x)
    return ms


def run_step(
    module: NunchakuFP4LoRALinear,
    x_base: torch.Tensor,
    dy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    x = x_base.detach().clone().requires_grad_(True)
    zero_grads(module, x)
    y = module(x)
    loss = (y.float() * dy.float()).sum()
    loss.backward()
    out = {
        "y": y.detach(),
        "dx": x.grad.detach().clone(),
        "d_lora_down": module.lora_down.grad.detach().clone(),
        "d_lora_up": module.lora_up.grad.detach().clone(),
        "d_bias": module.bias.grad.detach().clone() if isinstance(module.bias, torch.nn.Parameter) else torch.empty(0),
    }
    zero_grads(module, x)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark zero-init LoRA-up fast path for FP4 LoRA training.")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--frozen-residual-rank", type=int, default=0)
    p.add_argument("--frozen-residual-init", choices=["none", "residual_svd"], default="none")
    p.add_argument("--fuse-lowrank-forward", action="store_true")
    p.add_argument("--fuse-lora-dx", action="store_true")
    p.add_argument("--cache-fused-lora-dx", action="store_true")
    p.add_argument("--overlap-lora-grad", action="store_true")
    p.add_argument("--overlap-lora-grad-min-rows", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.cache_fused_lora_dx and not args.fuse_lora_dx:
        raise ValueError("--cache-fused-lora-dx requires --fuse-lora-dx")
    if args.overlap_lora_grad and not (args.fuse_lora_dx and args.cache_fused_lora_dx):
        raise ValueError("--overlap-lora-grad requires --fuse-lora-dx --cache-fused-lora-dx")

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)

    x = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(args.m, args.out_features, device="cuda", dtype=dtype)
    weight = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)
    bias = torch.randn(args.out_features, device="cuda", dtype=dtype)

    fast = make_module(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        frozen_residual_rank=args.frozen_residual_rank,
        frozen_residual_init=args.frozen_residual_init,
        fuse_lowrank_forward=args.fuse_lowrank_forward,
        fuse_lora_dx=args.fuse_lora_dx,
        cache_fused_lora_dx=args.cache_fused_lora_dx,
        overlap_lora_grad=args.overlap_lora_grad,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
        zero_lora_up_fast_path=True,
    )
    baseline = make_module(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        frozen_residual_rank=args.frozen_residual_rank,
        frozen_residual_init=args.frozen_residual_init,
        fuse_lowrank_forward=args.fuse_lowrank_forward,
        fuse_lora_dx=args.fuse_lora_dx,
        cache_fused_lora_dx=args.cache_fused_lora_dx,
        overlap_lora_grad=args.overlap_lora_grad,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
        zero_lora_up_fast_path=False,
    )
    sync_lora(baseline, fast)

    fast_initial_cache = refresh_optional_caches(fast)
    baseline_initial_cache = refresh_optional_caches(baseline)
    fast_active_before = fast._lora_up_zero_fast_path_active()
    baseline_active_before = baseline._lora_up_zero_fast_path_active()

    ref = run_step(baseline, x, dy)
    opt = run_step(fast, x, dy)
    with torch.no_grad():
        x_lr = x.detach().reshape(-1, args.in_features).to(lowrank_dtype)
        dy_lr = dy.reshape(-1, args.out_features).to(lowrank_dtype)
        lora_act_exact = torch.matmul(x_lr, fast.lora_down.detach().to(lowrank_dtype).t())
        d_lora_up_exact = torch.matmul(dy_lr.t(), lora_act_exact).mul(fast.scaling).to(fast.lora_up.dtype)

    fast_forward_ms = benchmark_forward(fast, x, args.warmup, args.iters)
    baseline_forward_ms = benchmark_forward(baseline, x, args.warmup, args.iters)
    fast_train_step_ms = benchmark_train_step(fast, x, dy, args.warmup, args.iters)
    baseline_train_step_ms = benchmark_train_step(baseline, x, dy, args.warmup, args.iters)

    with torch.no_grad():
        fast.lora_up.add_(1e-4)
    fast_active_after_lora_up_update = fast._lora_up_zero_fast_path_active()

    payload: dict[str, Any] = {
        "experiment": "fp4_lora_zero_lora_up_fast_path",
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "frozen_residual_rank": args.frozen_residual_rank,
            "frozen_residual_init": args.frozen_residual_init,
            "fuse_lowrank_forward": args.fuse_lowrank_forward,
            "fuse_lora_dx": args.fuse_lora_dx,
            "cache_fused_lora_dx": args.cache_fused_lora_dx,
            "overlap_lora_grad": args.overlap_lora_grad,
            "overlap_lora_grad_min_rows": args.overlap_lora_grad_min_rows,
        },
        "zero_fast_path_state": {
            "fast_active_before_update": fast_active_before,
            "baseline_active_before_update": baseline_active_before,
            "fast_active_after_lora_up_update": fast_active_after_lora_up_update,
            "fast_initial_cache": fast_initial_cache,
            "baseline_initial_cache": baseline_initial_cache,
        },
        "latency_ms": {
            "baseline_forward_train_graph": baseline_forward_ms,
            "fast_forward_train_graph": fast_forward_ms,
            "baseline_train_step": baseline_train_step_ms,
            "fast_train_step": fast_train_step_ms,
        },
        "speedups": {
            "forward_train_graph": baseline_forward_ms / fast_forward_ms,
            "train_step": baseline_train_step_ms / fast_train_step_ms,
        },
        "errors": {
            "forward": tensor_error(opt["y"], ref["y"]),
            "dx": tensor_error(opt["dx"], ref["dx"]),
            "d_lora_down": tensor_error(opt["d_lora_down"], ref["d_lora_down"]),
            "d_lora_up": tensor_error(opt["d_lora_up"], ref["d_lora_up"]),
            "d_lora_up_fast_vs_exact": tensor_error(opt["d_lora_up"], d_lora_up_exact),
            "d_lora_up_baseline_vs_exact": tensor_error(ref["d_lora_up"], d_lora_up_exact),
            "d_bias": tensor_error(opt["d_bias"], ref["d_bias"]),
        },
    }
    payload["checks"] = {
        "fast_path_active_before_update": bool(fast_active_before),
        "baseline_fast_path_disabled": not bool(baseline_active_before),
        "fast_path_disabled_after_lora_up_update": not bool(fast_active_after_lora_up_update),
        "forward_rel_l2_lt_1e-6": payload["errors"]["forward"]["rel_l2"] < 1e-6,
        "dx_rel_l2_lt_5e-4": payload["errors"]["dx"]["rel_l2"] < 5e-4,
        "d_lora_down_rel_l2_lt_1e-6": payload["errors"]["d_lora_down"]["rel_l2"] < 1e-6,
        "d_lora_up_fast_vs_exact_rel_l2_lt_1e-6": (
            payload["errors"]["d_lora_up_fast_vs_exact"]["rel_l2"] < 1e-6
        ),
        "d_bias_rel_l2_lt_1e-6": payload["errors"]["d_bias"]["rel_l2"] < 1e-6,
    }
    payload["all_passed"] = bool(all(payload["checks"].values()))

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    latest_path = os.path.join(args.results_dir, "latest_fp4_lora_zero_fast_path.json")
    stamped_path = os.path.join(args.results_dir, f"fp4_lora_zero_fast_path_{stamp}.json")
    for path in (latest_path, stamped_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {latest_path}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
