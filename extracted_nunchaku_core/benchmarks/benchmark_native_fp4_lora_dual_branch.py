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

from native_fp4 import NunchakuFP4LoRALinear  # noqa: E402


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


def train_step_ms(
    module: torch.nn.Module,
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

    ms = time_cuda(fn, warmup=warmup, iters=iters)
    zero_grads(module, x)
    return ms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=2048)
    p.add_argument("--in-features", type=int, default=2048)
    p.add_argument("--out-features", type=int, default=2048)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--frozen-residual-rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--fuse-lowrank-forward", action="store_true")
    p.add_argument("--overlap-lora-grad-min-rows", type=int, default=4096)
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

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    lowrank_dtype = dtype
    x_task = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype, requires_grad=True)
    x_dense = x_task.detach().clone().requires_grad_(True)
    x_overlap = x_dense.detach().clone().requires_grad_(True)
    x_fused = x_dense.detach().clone().requires_grad_(True)
    dy = torch.randn(args.m, args.out_features, device="cuda", dtype=dtype)
    weight = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)
    bias = torch.randn(args.out_features, device="cuda", dtype=dtype)

    task_only = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        cache_lora_act=True,
        fuse_lowrank_forward=args.fuse_lowrank_forward,
        fuse_lora_dx=True,
        cache_fused_lora_dx=True,
    )
    dense_residual_dx = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        frozen_residual_rank=args.frozen_residual_rank,
        frozen_residual_init="residual_svd",
        cache_lora_act=True,
        fuse_lowrank_forward=args.fuse_lowrank_forward,
        fuse_lora_dx=True,
        fuse_frozen_residual_dx=False,
        cache_fused_lora_dx=True,
    )
    dense_residual_dx_overlap = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        frozen_residual_rank=args.frozen_residual_rank,
        frozen_residual_init="residual_svd",
        cache_lora_act=True,
        fuse_lowrank_forward=args.fuse_lowrank_forward,
        fuse_lora_dx=True,
        fuse_frozen_residual_dx=False,
        cache_fused_lora_dx=True,
        overlap_lora_grad=True,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
    )
    fused_residual_dx = None
    if dtype == torch.float16:
        fused_residual_dx = NunchakuFP4LoRALinear(
            weight=weight,
            bias=bias,
            rank=args.rank,
            lowrank_dtype=lowrank_dtype,
            init="gaussian",
            frozen_residual_rank=args.frozen_residual_rank,
            frozen_residual_init="residual_svd",
            cache_lora_act=True,
            fuse_lowrank_forward=args.fuse_lowrank_forward,
            fuse_lora_dx=True,
            fuse_frozen_residual_dx=True,
            cache_fused_lora_dx=True,
        )
    with torch.no_grad():
        task_only.lora_down.copy_(dense_residual_dx.lora_down)
        task_only.lora_up.copy_(dense_residual_dx.lora_up)
        dense_residual_dx_overlap.lora_down.copy_(dense_residual_dx.lora_down)
        dense_residual_dx_overlap.lora_up.copy_(dense_residual_dx.lora_up)
        dense_residual_dx_overlap.frozen_residual_down.copy_(dense_residual_dx.frozen_residual_down)
        dense_residual_dx_overlap.frozen_residual_up.copy_(dense_residual_dx.frozen_residual_up)
        if fused_residual_dx is not None:
            fused_residual_dx.lora_down.copy_(dense_residual_dx.lora_down)
            fused_residual_dx.lora_up.copy_(dense_residual_dx.lora_up)
            fused_residual_dx.frozen_residual_down.copy_(dense_residual_dx.frozen_residual_down)
            fused_residual_dx.frozen_residual_up.copy_(dense_residual_dx.frozen_residual_up)

    task_only.refresh_fused_lora_dx_cache()
    dense_residual_dx.refresh_fused_lora_dx_cache()
    dense_residual_dx_overlap.refresh_fused_lora_dx_cache()
    if fused_residual_dx is not None:
        fused_residual_dx.refresh_fused_lora_dx_cache()

    y_dense = dense_residual_dx(x_dense)
    loss_dense = (y_dense.float() * dy.float()).sum()
    loss_dense.backward()
    y_overlap = dense_residual_dx_overlap(x_overlap)
    loss_overlap = (y_overlap.float() * dy.float()).sum()
    loss_overlap.backward()

    errors = {
        "overlap_residual_dx": {
            "forward": tensor_error(y_overlap, y_dense),
            "dx": tensor_error(x_overlap.grad, x_dense.grad),
            "lora_down_grad": tensor_error(
                dense_residual_dx_overlap.lora_down.grad,
                dense_residual_dx.lora_down.grad,
            ),
            "lora_up_grad": tensor_error(
                dense_residual_dx_overlap.lora_up.grad,
                dense_residual_dx.lora_up.grad,
            ),
        }
    }
    if fused_residual_dx is not None:
        y_fused = fused_residual_dx(x_fused)
        loss_fused = (y_fused.float() * dy.float()).sum()
        loss_fused.backward()

        errors["fused_residual_dx"] = {
            "forward": tensor_error(y_fused, y_dense),
            "dx": tensor_error(x_fused.grad, x_dense.grad),
            "lora_down_grad": tensor_error(fused_residual_dx.lora_down.grad, dense_residual_dx.lora_down.grad),
            "lora_up_grad": tensor_error(fused_residual_dx.lora_up.grad, dense_residual_dx.lora_up.grad),
        }
    zero_grads(dense_residual_dx, x_dense)
    zero_grads(dense_residual_dx_overlap, x_overlap)
    if fused_residual_dx is not None:
        zero_grads(fused_residual_dx, x_fused)

    task_only_train_step_ms = train_step_ms(task_only, x_task, dy, args.warmup, args.iters)
    dense_residual_dx_train_step_ms = train_step_ms(dense_residual_dx, x_dense, dy, args.warmup, args.iters)
    overlap_residual_dx_train_step_ms = train_step_ms(
        dense_residual_dx_overlap,
        x_overlap,
        dy,
        args.warmup,
        args.iters,
    )
    fused_residual_dx_train_step_ms = None
    if fused_residual_dx is not None:
        fused_residual_dx_train_step_ms = train_step_ms(fused_residual_dx, x_fused, dy, args.warmup, args.iters)

    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "effective_rank": dense_residual_dx.rank,
            "frozen_residual_rank": args.frozen_residual_rank,
            "effective_frozen_residual_rank": dense_residual_dx.frozen_residual_rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.dtype,
            "fuse_lowrank_forward": args.fuse_lowrank_forward,
            "overlap_residual_dx_available": True,
            "overlap_lora_grad_min_rows": args.overlap_lora_grad_min_rows,
            "fused_residual_dx_available": fused_residual_dx is not None,
        },
        "latency_ms": {
            "task_only_fused_dx_train_step": task_only_train_step_ms,
            "dual_branch_dense_residual_dx_train_step": dense_residual_dx_train_step_ms,
            "dual_branch_dense_residual_overlap_train_step": overlap_residual_dx_train_step_ms,
            "dual_branch_fused_residual_dx_train_step": fused_residual_dx_train_step_ms,
        },
        "speedups": {
            "dual_dense_residual_dx_over_task_only_fused_dx_train_step": (
                dense_residual_dx_train_step_ms / task_only_train_step_ms
            ),
            "fused_residual_dx_vs_dense_residual_dx_train_step": (
                None
                if fused_residual_dx_train_step_ms is None
                else dense_residual_dx_train_step_ms / fused_residual_dx_train_step_ms
            ),
            "overlap_residual_dx_vs_dense_residual_dx_train_step": (
                dense_residual_dx_train_step_ms / overlap_residual_dx_train_step_ms
            ),
        },
        "errors": errors,
        "checks": {},
    }
    overlap_errors = errors["overlap_residual_dx"]
    lora_up_grad_tol = 5e-4 if args.fuse_lowrank_forward else 1e-6
    payload["checks"].update(
        {
            "overlap_forward_rel_l2_lt_tol": overlap_errors["forward"]["rel_l2"]
            < (5e-4 if args.fuse_lowrank_forward else 1e-6),
            "overlap_dx_rel_l2_lt_5e-4": overlap_errors["dx"]["rel_l2"] < 5e-4,
            "overlap_lora_down_grad_rel_l2_lt_1e-6": overlap_errors["lora_down_grad"]["rel_l2"] < 1e-6,
            "overlap_lora_up_grad_rel_l2_lt_tol": overlap_errors["lora_up_grad"]["rel_l2"] < lora_up_grad_tol,
        }
    )
    if fused_residual_dx is None:
        payload["checks"]["fused_residual_dx_unavailable_for_bf16"] = args.dtype == "bf16"
    else:
        fused_errors = errors["fused_residual_dx"]
        payload["checks"].update(
            {
                "fused_forward_rel_l2_lt_tol": fused_errors["forward"]["rel_l2"]
                < (5e-4 if args.fuse_lowrank_forward else 1e-6),
                "fused_dx_rel_l2_lt_5e-4": fused_errors["dx"]["rel_l2"] < 5e-4,
                "fused_lora_down_grad_rel_l2_lt_1e-6": fused_errors["lora_down_grad"]["rel_l2"] < 1e-6,
                "fused_lora_up_grad_rel_l2_lt_tol": fused_errors["lora_up_grad"]["rel_l2"] < lora_up_grad_tol,
            }
        )
    payload["all_passed"] = bool(all(payload["checks"].values()))

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp4_lora_dual_branch_{args.dtype}_{stamp}.json")
    latest_path = os.path.join(args.results_dir, f"latest_native_fp4_lora_dual_branch_{args.dtype}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
