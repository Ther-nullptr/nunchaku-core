from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import (  # noqa: E402
    NunchakuFP4LoRALinear,
    fp4_lora_parameter_groups,
    register_fp4_lora_cache_refresh_hook,
)


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


def param_l2_delta(before: torch.Tensor, after: torch.Tensor) -> float:
    return float((after.detach().float() - before.float()).norm().item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate that a frozen FP4 backbone plus frozen residual-SVD compensation "
            "can fine-tune a zero-init BF16/FP16 LoRA branch on a synthetic low-rank target."
        )
    )
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--in-features", type=int, default=512)
    p.add_argument("--out-features", type=int, default=768)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--target-rank", type=int, default=8)
    p.add_argument("--frozen-residual-rank", type=int, default=32)
    p.add_argument("--frozen-residual-init", choices=["none", "residual_svd"], default="residual_svd")
    p.add_argument("--residual-svd-method", choices=["full_svd", "svd_lowrank"], default="svd_lowrank")
    p.add_argument("--residual-svd-lowrank-oversample", type=int, default=8)
    p.add_argument("--residual-svd-lowrank-niter", type=int, default=2)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adam-eps", type=float, default=1e-4)
    p.add_argument("--loss-threshold", type=float, default=0.35)
    p.add_argument("--target-base", choices=["fp4_initial", "dense"], default="fp4_initial")
    p.add_argument("--input-std", type=float, default=1.0)
    p.add_argument("--weight-std", type=float, default=0.02)
    p.add_argument("--bias-std", type=float, default=0.02)
    p.add_argument("--teacher-std", type=float, default=0.02)
    p.add_argument("--teacher-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-bias", action="store_true")
    p.add_argument("--no-fuse-lora-dx", dest="fuse_lora_dx", action="store_false")
    p.add_argument("--no-cache-fused-lora-dx", dest="cache_fused_lora_dx", action="store_false")
    p.add_argument("--no-cache-refresh-hook", dest="cache_refresh_hook", action="store_false")
    p.add_argument("--results-dir", type=str, default="results")
    p.set_defaults(fuse_lora_dx=True, cache_fused_lora_dx=True, cache_refresh_hook=True)
    return p.parse_args()


def make_teacher_delta(
    x: torch.Tensor,
    *,
    out_features: int,
    target_rank: int,
    lowrank_dtype: torch.dtype,
    teacher_std: float,
    teacher_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x_lr = x.detach().reshape(-1, x.shape[-1]).to(lowrank_dtype)
    teacher_down = torch.randn(
        target_rank,
        x.shape[-1],
        device=x.device,
        dtype=lowrank_dtype,
    ).mul(float(teacher_std))
    teacher_up = torch.randn(
        out_features,
        target_rank,
        device=x.device,
        dtype=lowrank_dtype,
    ).mul(float(teacher_std))
    act = torch.matmul(x_lr, teacher_down.t())
    delta = torch.matmul(act, teacher_up.t()).mul(float(teacher_scale))
    return delta.reshape(*x.shape[:-1], out_features).to(x.dtype), {
        "teacher_down": teacher_down,
        "teacher_up": teacher_up,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.target_rank <= 0:
        raise ValueError("--target-rank must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)
    lowrank_dtype = dtype_from_name(args.lowrank_dtype)

    x = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype).mul(float(args.input_std))
    x.requires_grad_(True)
    weight = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype).mul(float(args.weight_std))
    bias = torch.randn(args.out_features, device="cuda", dtype=dtype).mul(float(args.bias_std))
    frozen_residual_rank = 0 if args.frozen_residual_init == "none" else args.frozen_residual_rank

    module = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        init="zero",
        frozen_residual_rank=frozen_residual_rank,
        frozen_residual_init=args.frozen_residual_init,
        residual_svd_method=args.residual_svd_method,
        residual_svd_lowrank_oversample=args.residual_svd_lowrank_oversample,
        residual_svd_lowrank_niter=args.residual_svd_lowrank_niter,
        train_bias=args.train_bias,
        fuse_lora_dx=args.fuse_lora_dx,
        cache_fused_lora_dx=args.cache_fused_lora_dx,
    ).train()

    with torch.no_grad():
        initial_y = module(x.detach()).detach()
        dense_y = F.linear(x.detach(), weight, bias).detach()
        base_y = initial_y if args.target_base == "fp4_initial" else dense_y
        target_delta, teacher_factors = make_teacher_delta(
            x,
            out_features=args.out_features,
            target_rank=args.target_rank,
            lowrank_dtype=lowrank_dtype,
            teacher_std=args.teacher_std,
            teacher_scale=args.teacher_scale,
        )
        target = (base_y.float() + target_delta.float()).to(dtype).detach()
        frozen_down_before = (
            module.frozen_residual_down.detach().clone() if module.has_frozen_residual else torch.empty(0, device="cuda")
        )
        frozen_up_before = (
            module.frozen_residual_up.detach().clone() if module.has_frozen_residual else torch.empty(0, device="cuda")
        )
        lora_down_before = module.lora_down.detach().clone()
        lora_up_before = module.lora_up.detach().clone()

    param_groups = fp4_lora_parameter_groups(
        module,
        train_bias=args.train_bias,
        lora_weight_decay=args.weight_decay,
        bias_weight_decay=0.0,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=args.adam_eps)
    hook = None
    if args.fuse_lora_dx and args.cache_fused_lora_dx and args.cache_refresh_hook:
        hook = register_fp4_lora_cache_refresh_hook(optimizer, module)

    losses: list[float] = []
    grad_norms: dict[str, list[float]] = {"lora_down": [], "lora_up": []}
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        y = module(x)
        loss = F.mse_loss(y.float(), target.float())
        loss.backward()
        losses.append(float(loss.detach().item()))
        grad_norms["lora_down"].append(float(module.lora_down.grad.detach().float().norm().item()))
        grad_norms["lora_up"].append(float(module.lora_up.grad.detach().float().norm().item()))
        optimizer.step()

    torch.cuda.synchronize()
    with torch.no_grad():
        final_y = module(x.detach()).detach()
        final_loss = float(F.mse_loss(final_y.float(), target.float()).item())
        initial_loss = losses[0]
        best_loss = min(losses + [final_loss])
        pred_delta = final_y.float() - initial_y.float()
        target_delta_from_initial = target.float() - initial_y.float()

        frozen_down_after = module.frozen_residual_down.detach() if module.has_frozen_residual else frozen_down_before
        frozen_up_after = module.frozen_residual_up.detach() if module.has_frozen_residual else frozen_up_before
        frozen_unchanged = bool(
            torch.equal(frozen_down_before, frozen_down_after)
            and torch.equal(frozen_up_before, frozen_up_after)
        )
        lora_down_delta = param_l2_delta(lora_down_before, module.lora_down)
        lora_up_delta = param_l2_delta(lora_up_before, module.lora_up)

    loss_ratio = final_loss / (initial_loss + 1e-12)
    grad_finite = bool(
        torch.isfinite(module.lora_down.grad).all()
        and torch.isfinite(module.lora_up.grad).all()
        and x.grad is not None
        and torch.isfinite(x.grad).all()
    )
    if args.train_bias and isinstance(module.bias, torch.nn.Parameter):
        grad_finite = bool(grad_finite and module.bias.grad is not None and torch.isfinite(module.bias.grad).all())

    named_trainable = [name for name, param in module.named_parameters() if param.requires_grad]
    expected_trainable = {"lora_down", "lora_up"}
    if args.train_bias:
        expected_trainable.add("bias")

    checks = {
        "loss_decreased": final_loss < initial_loss,
        "loss_ratio_below_threshold": loss_ratio < args.loss_threshold,
        "all_losses_finite": all(torch.isfinite(torch.tensor(losses + [final_loss])).tolist()),
        "grads_finite": grad_finite,
        "lora_params_changed": lora_down_delta > 0.0 and lora_up_delta > 0.0,
        "frozen_residual_unchanged": frozen_unchanged,
        "only_expected_params_trainable": set(named_trainable) == expected_trainable,
    }
    if hook is not None:
        checks["cache_refresh_hook_ran"] = hook.last_refresh_count > 0

    payload: dict[str, Any] = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "effective_rank": module.rank,
            "target_rank": args.target_rank,
            "frozen_residual_rank": args.frozen_residual_rank,
            "effective_requested_frozen_residual_rank": frozen_residual_rank,
            "effective_frozen_residual_rank": module.frozen_residual_rank,
            "frozen_residual_init": args.frozen_residual_init,
            "residual_svd_method": args.residual_svd_method,
            "residual_svd_lowrank_oversample": args.residual_svd_lowrank_oversample,
            "residual_svd_lowrank_niter": args.residual_svd_lowrank_niter,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
        },
        "train": {
            "steps": args.steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "adam_eps": args.adam_eps,
            "loss_threshold": args.loss_threshold,
            "target_base": args.target_base,
            "train_bias": args.train_bias,
            "fuse_lora_dx": args.fuse_lora_dx,
            "cache_fused_lora_dx": args.cache_fused_lora_dx,
            "cache_refresh_hook": args.cache_refresh_hook,
        },
        "init": {
            "input_std": args.input_std,
            "weight_std": args.weight_std,
            "bias_std": args.bias_std,
            "teacher_std": args.teacher_std,
            "teacher_scale": args.teacher_scale,
            "seed": args.seed,
        },
        "loss": {
            "initial": initial_loss,
            "final": final_loss,
            "best": best_loss,
            "final_over_initial": loss_ratio,
            "reduction": 1.0 - loss_ratio,
            "curve": losses + [final_loss],
        },
        "errors": {
            "initial_fp4_vs_dense_base": tensor_error(initial_y, dense_y),
            "final_vs_target": tensor_error(final_y, target),
            "predicted_delta_vs_target_delta_from_initial": tensor_error(pred_delta, target_delta_from_initial),
        },
        "params": {
            "trainable": named_trainable,
            "lora_down_l2_delta": lora_down_delta,
            "lora_up_l2_delta": lora_up_delta,
            "teacher_down_norm": float(teacher_factors["teacher_down"].float().norm().item()),
            "teacher_up_norm": float(teacher_factors["teacher_up"].float().norm().item()),
            "last_grad_norms": {
                "lora_down": grad_norms["lora_down"][-1],
                "lora_up": grad_norms["lora_up"][-1],
            },
        },
        "checks": checks,
    }
    payload["all_passed"] = bool(all(checks.values()))

    if hook is not None:
        hook.remove()

    os.makedirs(args.results_dir, exist_ok=True)
    latest = os.path.join(args.results_dir, "latest_fp4_lora_finetune_convergence.json")
    stamped = os.path.join(
        args.results_dir,
        f"fp4_lora_finetune_convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    for path in (latest, stamped):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {latest}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
