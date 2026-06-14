from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import NunchakuFP4LoRALinear, dequantize_fp4_activation, fp4_activation_cache_lora_down_grad  # noqa: E402
from native_fp4.operators import pad_tensor  # noqa: E402


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
    p.add_argument("--frozen-residual-rank", type=int, default=0)
    p.add_argument("--frozen-residual-init", choices=["none", "residual_svd"], default="none")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--init", choices=["zero", "gaussian", "residual_svd"], default="gaussian")
    p.add_argument("--residual-svd-method", choices=["full_svd", "svd_lowrank"], default="full_svd")
    p.add_argument("--residual-svd-lowrank-oversample", type=int, default=8)
    p.add_argument("--residual-svd-lowrank-niter", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache-lora-act", action="store_true")
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--fuse-lowrank-forward", action="store_true")
    p.add_argument("--fuse-lora-dx", action="store_true")
    p.add_argument("--fuse-frozen-residual-dx", action="store_true")
    p.add_argument("--cache-fused-lora-dx", action="store_true")
    p.add_argument("--backward-weight-policy", choices=["repack", "cache"], default="repack")
    p.add_argument("--reuse-fused-dy-up-for-d-lora-down", action="store_true")
    p.add_argument("--overlap-lora-grad", action="store_true")
    p.add_argument("--overlap-lora-grad-min-rows", type=int, default=4096)
    p.add_argument("--fp4-activation-cache-d-lora-down", action="store_true")
    p.add_argument("--fp4-activation-cache-d-lora-down-backend", choices=["fused", "dequant_gemm"], default="fused")
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
        init=args.init,
        frozen_residual_rank=args.frozen_residual_rank,
        frozen_residual_init=args.frozen_residual_init,
        residual_svd_method=args.residual_svd_method,
        residual_svd_lowrank_oversample=args.residual_svd_lowrank_oversample,
        residual_svd_lowrank_niter=args.residual_svd_lowrank_niter,
        train_bias=True,
        cache_lora_act=not args.no_cache_lora_act,
        activation_checkpoint=args.activation_checkpoint,
        fuse_lowrank_forward=args.fuse_lowrank_forward,
        fuse_lora_dx=args.fuse_lora_dx,
        fuse_frozen_residual_dx=args.fuse_frozen_residual_dx,
        cache_fused_lora_dx=args.cache_fused_lora_dx,
        backward_weight_policy=args.backward_weight_policy,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        overlap_lora_grad=args.overlap_lora_grad,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
        fp4_activation_cache_d_lora_down=args.fp4_activation_cache_d_lora_down,
        fp4_activation_cache_d_lora_down_backend=args.fp4_activation_cache_d_lora_down_backend,
    )

    cache_refresh_check = True
    backward_weight_cache_check = True
    if args.backward_weight_policy == "cache":
        op.refresh_backward_weight_cache()
        backward_weight_cache_check = op.fp4_backward._cached_qweight_bwd is not None
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
        lowrank_out = y_lora
        separate_lowrank_out = y_lora if args.fuse_lowrank_forward else None
        native_fused_forward = bool(args.fuse_lowrank_forward and dtype == lowrank_dtype)
        sequential_y_ref = None

        dy_up = torch.matmul(dy_lr, up_lr)
        dx_ref = op.fp4_backward(dy) + torch.matmul(dy_up, down_lr).mul(op.scaling).reshape_as(x).to(dtype)
        if op.has_frozen_residual:
            residual_down_lr = op.frozen_residual_down.detach().to(lowrank_dtype)
            residual_up_lr = op.frozen_residual_up.detach().to(lowrank_dtype)
            residual_act = torch.matmul(x_lr, residual_down_lr.t())
            y_residual = torch.matmul(residual_act, residual_up_lr.t()).mul(op.frozen_residual_scaling)
            separate_lowrank_out = y_lora + y_residual.to(dtype)
            if native_fused_forward:
                sequential_y_ref = y_main + y_lora.reshape_as(y_main)
                sequential_y_ref = sequential_y_ref + y_residual.to(dtype).reshape_as(y_main)
                lowrank_out = y_lora
            elif args.fuse_lowrank_forward:
                combined_down = torch.cat((down_lr, residual_down_lr), dim=0)
                combined_up = torch.cat(
                    (
                        up_lr.mul(float(op.scaling)),
                        residual_up_lr.mul(float(op.frozen_residual_scaling)),
                    ),
                    dim=1,
                )
                combined_act = torch.matmul(x_lr, combined_down.t())
                lora_act = combined_act[:, : op.rank]
                lowrank_out = torch.matmul(combined_act, combined_up.t()).to(dtype)
            else:
                lowrank_out = separate_lowrank_out
            dy_residual_up = torch.matmul(dy_lr, residual_up_lr)
            dx_residual = torch.matmul(dy_residual_up, residual_down_lr).mul(op.frozen_residual_scaling)
            dx_ref = dx_ref + dx_residual.reshape_as(x).to(dtype)
        y_ref = sequential_y_ref if sequential_y_ref is not None else y_main + lowrank_out.reshape_as(y_main)
        separate_y_ref = None
        if separate_lowrank_out is not None:
            separate_y_ref = (
                sequential_y_ref
                if sequential_y_ref is not None
                else y_main + separate_lowrank_out.reshape_as(y_main)
            )
        y_ref = y_ref + op.bias.detach().to(dtype)
        if separate_y_ref is not None:
            separate_y_ref = separate_y_ref + op.bias.detach().to(dtype)
        d_up_ref = torch.matmul(dy_lr.t(), lora_act).mul(op.scaling).to(op.lora_up.dtype)
        d_down_exact_ref = torch.matmul(dy_up.t(), x_lr).mul(op.scaling).to(op.lora_down.dtype)
        d_down_ref = d_down_exact_ref
        if args.fp4_activation_cache_d_lora_down:
            x2d_fp4 = x2d
            if op.fp4_forward.k_pad != op.in_features:
                x2d_fp4 = pad_tensor(x2d_fp4, divisor=op.fp4_forward.k_pad, dim=1)
            qact, ascales = op.fp4_forward.quantize_input(x2d_fp4)
            if args.fp4_activation_cache_d_lora_down_backend == "fused":
                d_down_ref = fp4_activation_cache_lora_down_grad(
                    qact,
                    ascales,
                    dy_up.contiguous(),
                    in_features=op.in_features,
                )
            else:
                x_hat_pad, _ = dequantize_fp4_activation(qact, ascales, dtype=lowrank_dtype, return_scales=False)
                x_hat = x_hat_pad[: x2d.shape[0], : op.in_features].contiguous()
                d_down_ref = torch.matmul(dy_up.t(), x_hat)
            d_down_ref = d_down_ref.mul(op.scaling).to(op.lora_down.dtype)
        d_bias_ref = dy2d.sum(dim=0).to(op.bias.dtype)

    errors = {
        "forward_vs_manual": tensor_error(y, y_ref),
        "dx_vs_manual": tensor_error(x.grad, dx_ref),
        "lora_up_grad_vs_manual": tensor_error(op.lora_up.grad, d_up_ref),
        "lora_down_grad_vs_manual": tensor_error(op.lora_down.grad, d_down_ref),
        "bias_grad_vs_manual": tensor_error(op.bias.grad, d_bias_ref),
    }
    if args.fp4_activation_cache_d_lora_down:
        errors["lora_down_grad_fp4_cache_vs_exact_manual"] = tensor_error(d_down_ref, d_down_exact_ref)
    if args.fuse_lowrank_forward and separate_y_ref is not None:
        errors["forward_vs_separate_lowrank_manual"] = tensor_error(y, separate_y_ref)
    grad_tol = 5e-4 if args.fuse_lora_dx else 1e-6
    forward_tol = 5e-4 if native_fused_forward else 1e-6
    lora_up_grad_tol = 5e-4 if native_fused_forward and not args.no_cache_lora_act else 1e-6
    checks = {
        "forward_rel_l2_lt_tol": errors["forward_vs_manual"]["rel_l2"] < forward_tol,
        "dx_rel_l2_lt_5e-4": errors["dx_vs_manual"]["rel_l2"] < 5e-4,
        "lora_up_grad_rel_l2_lt_tol": errors["lora_up_grad_vs_manual"]["rel_l2"] < lora_up_grad_tol,
        "lora_down_grad_rel_l2_lt_tol": errors["lora_down_grad_vs_manual"]["rel_l2"] < grad_tol,
        "bias_grad_rel_l2_lt_1e-6": errors["bias_grad_vs_manual"]["rel_l2"] < 1e-6,
        "all_finite": bool(
            torch.isfinite(y).all()
            and torch.isfinite(x.grad).all()
            and torch.isfinite(op.lora_up.grad).all()
            and torch.isfinite(op.lora_down.grad).all()
            and (not op.has_frozen_residual or bool(torch.isfinite(op.frozen_residual_down).all()))
            and (not op.has_frozen_residual or bool(torch.isfinite(op.frozen_residual_up).all()))
        ),
        "frozen_residual_is_buffer": not any(
            name.startswith("frozen_residual") for name, _ in op.named_parameters()
        ),
        "cache_refresh_after_param_update": cache_refresh_check,
        "backward_weight_cache_policy_matches": op.backward_weight_policy == args.backward_weight_policy,
        "backward_weight_cache_state_matches": backward_weight_cache_check,
    }
    if "forward_vs_separate_lowrank_manual" in errors:
        checks["fused_forward_separate_formula_rel_l2_lt_5e-4"] = (
            errors["forward_vs_separate_lowrank_manual"]["rel_l2"] < 5e-4
        )

    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "effective_rank": op.rank,
            "frozen_residual_rank": args.frozen_residual_rank,
            "effective_frozen_residual_rank": op.frozen_residual_rank,
            "frozen_residual_init": args.frozen_residual_init,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "init": args.init,
            "residual_svd_method": args.residual_svd_method,
            "residual_svd_lowrank_oversample": args.residual_svd_lowrank_oversample,
            "residual_svd_lowrank_niter": args.residual_svd_lowrank_niter,
            "cache_lora_act": not args.no_cache_lora_act,
            "activation_checkpoint": args.activation_checkpoint,
            "fuse_lowrank_forward": args.fuse_lowrank_forward,
            "fuse_lora_dx": args.fuse_lora_dx,
            "fuse_frozen_residual_dx": args.fuse_frozen_residual_dx,
            "cache_fused_lora_dx": args.cache_fused_lora_dx,
            "backward_weight_policy": args.backward_weight_policy,
            "reuse_fused_dy_up_for_d_lora_down": args.reuse_fused_dy_up_for_d_lora_down,
            "overlap_lora_grad": args.overlap_lora_grad,
            "overlap_lora_grad_min_rows": args.overlap_lora_grad_min_rows,
            "fp4_activation_cache_d_lora_down": args.fp4_activation_cache_d_lora_down,
            "fp4_activation_cache_d_lora_down_backend": args.fp4_activation_cache_d_lora_down_backend,
            "native_fused_forward": native_fused_forward,
        },
        "tolerances": {
            "forward_rel_l2": forward_tol,
            "fused_forward_separate_formula_rel_l2": 5e-4,
            "lora_up_grad_rel_l2": lora_up_grad_tol,
            "lora_down_grad_rel_l2": grad_tol,
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
