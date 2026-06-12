from __future__ import annotations

import argparse
import json
import os
import sys

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=257)
    p.add_argument("--in-features", type=int, default=3072)
    p.add_argument("--out-features", type=int, default=3584)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache-lora-act", action="store_true")
    p.add_argument("--fuse-lora-dx", action="store_true")
    p.add_argument("--cache-fused-lora-dx", action="store_true")
    p.add_argument("--reuse-fused-dy-up-for-d-lora-down", action="store_true")
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    lowrank_dtype = torch.float16 if args.lowrank_dtype == "fp16" else torch.bfloat16

    x = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(args.m, args.out_features, device="cuda", dtype=dtype)
    weight = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)
    bias = torch.randn(args.out_features, device="cuda", dtype=dtype)

    op = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        train_bias=True,
        cache_lora_act=not args.no_cache_lora_act,
        fuse_lora_dx=args.fuse_lora_dx,
        cache_fused_lora_dx=args.cache_fused_lora_dx,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
    )

    cache_refresh_check = True
    if args.cache_fused_lora_dx:
        op.refresh_fused_lora_dx_cache()
        old_down_version = op._cached_lora_down_version
        old_up_version = op._cached_lora_up_version
        with torch.no_grad():
            op.lora_down.add_(1e-4)
            op.lora_up.add_(1e-4)
        cache_refresh_check = old_down_version != op.lora_down._version and old_up_version != op.lora_up._version

    y = op(x)
    loss = (y.float() * dy.float()).sum()
    loss.backward()
    if args.cache_fused_lora_dx:
        cache_refresh_check = bool(
            cache_refresh_check
            and op._cached_lora_down_version == op.lora_down._version
            and op._cached_lora_up_version == op.lora_up._version
        )

    with torch.no_grad():
        x2d = x.detach().reshape(-1, op.in_features)
        dy2d = dy.reshape(-1, op.out_features)
        x_lr = x2d.to(lowrank_dtype)
        dy_lr = dy2d.to(lowrank_dtype)
        down_lr = op.lora_down.detach().to(lowrank_dtype)
        up_lr = op.lora_up.detach().to(lowrank_dtype)

        y_main = op.fp4_forward(x.detach())
        lora_act = torch.matmul(x_lr, down_lr.t())
        y_lora = torch.matmul(lora_act, up_lr.t()).mul(op.scaling).to(dtype)
        y_ref = y_main + y_lora.reshape_as(y_main) + op.bias.detach().to(dtype)

        dy_up = torch.matmul(dy_lr, up_lr)
        dx_ref = op.fp4_backward(dy) + torch.matmul(dy_up, down_lr).mul(op.scaling).reshape_as(x).to(dtype)
        d_up_ref = torch.matmul(dy_lr.t(), lora_act).mul(op.scaling).to(op.lora_up.dtype)
        d_down_ref = torch.matmul(dy_up.t(), x_lr).mul(op.scaling).to(op.lora_down.dtype)
        d_bias_ref = dy2d.sum(dim=0).to(op.bias.dtype)

    errors = {
        "forward_vs_manual": tensor_error(y, y_ref),
        "dx_vs_manual": tensor_error(x.grad, dx_ref),
        "lora_up_grad_vs_manual": tensor_error(op.lora_up.grad, d_up_ref),
        "lora_down_grad_vs_manual": tensor_error(op.lora_down.grad, d_down_ref),
        "bias_grad_vs_manual": tensor_error(op.bias.grad, d_bias_ref),
    }
    grad_tol = 5e-4 if args.fuse_lora_dx else 1e-6
    checks = {
        "forward_rel_l2_lt_1e-6": errors["forward_vs_manual"]["rel_l2"] < 1e-6,
        "dx_rel_l2_lt_5e-4": errors["dx_vs_manual"]["rel_l2"] < 5e-4,
        "lora_up_grad_rel_l2_lt_1e-6": errors["lora_up_grad_vs_manual"]["rel_l2"] < 1e-6,
        "lora_down_grad_rel_l2_lt_tol": errors["lora_down_grad_vs_manual"]["rel_l2"] < grad_tol,
        "bias_grad_rel_l2_lt_1e-6": errors["bias_grad_vs_manual"]["rel_l2"] < 1e-6,
        "all_finite": bool(
            torch.isfinite(y).all()
            and torch.isfinite(x.grad).all()
            and torch.isfinite(op.lora_up.grad).all()
            and torch.isfinite(op.lora_down.grad).all()
        ),
        "cache_refresh_after_param_update": cache_refresh_check,
    }

    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "effective_rank": op.rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "cache_lora_act": not args.no_cache_lora_act,
            "fuse_lora_dx": args.fuse_lora_dx,
            "cache_fused_lora_dx": args.cache_fused_lora_dx,
            "reuse_fused_dy_up_for_d_lora_down": args.reuse_fused_dy_up_for_d_lora_down,
        },
        "errors": errors,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "latest_native_fp4_lora_training_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
