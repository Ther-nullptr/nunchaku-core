from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import NunchakuFP4LoRALinear  # noqa: E402


class StackedFP4LoRA(torch.nn.Module):
    def __init__(
        self,
        *,
        layers: int,
        hidden: int,
        rank: int,
        dtype: torch.dtype,
        lowrank_dtype: torch.dtype,
        checkpoint_scope: str,
        cache_lora_act: bool,
        fuse_lora_dx: bool,
        cache_fused_lora_dx: bool,
        frozen_residual_rank: int,
        frozen_residual_init: str,
        intermediate_activation: str,
    ):
        super().__init__()
        if intermediate_activation not in ("none", "silu"):
            raise ValueError("intermediate_activation must be 'none' or 'silu'")
        if checkpoint_scope not in ("none", "module", "stack"):
            raise ValueError("checkpoint_scope must be 'none', 'module', or 'stack'")
        self.intermediate_activation = intermediate_activation
        self.checkpoint_scope = checkpoint_scope
        self.layers = torch.nn.ModuleList()
        for _ in range(layers):
            weight = torch.randn(hidden, hidden, device="cuda", dtype=dtype)
            bias = torch.randn(hidden, device="cuda", dtype=dtype)
            self.layers.append(
                NunchakuFP4LoRALinear(
                    weight=weight,
                    bias=bias,
                    rank=rank,
                    lowrank_dtype=lowrank_dtype,
                    init="gaussian" if frozen_residual_init == "none" else "zero",
                    frozen_residual_rank=frozen_residual_rank,
                    frozen_residual_init=frozen_residual_init,
                    train_bias=False,
                    cache_lora_act=cache_lora_act,
                    activation_checkpoint=checkpoint_scope == "module",
                    fuse_lora_dx=fuse_lora_dx,
                    cache_fused_lora_dx=cache_fused_lora_dx,
                )
            )

    def refresh_fused_lora_dx_caches(self) -> None:
        for layer in self.layers:
            if layer.fuse_lora_dx and layer.cache_fused_lora_dx:
                layer.refresh_fused_lora_dx_cache()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if self.intermediate_activation == "silu" and index + 1 != len(self.layers):
                x = F.silu(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_scope == "stack" and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, x, use_reentrant=False, preserve_rng_state=False)
        return self._forward_impl(x)


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    da = a.float()
    db = b.float()
    diff = (da - db).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "rel_l2": float((da - db).norm().item() / (db.norm().item() + 1e-12)),
    }


def zero_grads(model: torch.nn.Module, x: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    x.grad = None


def train_step(model: torch.nn.Module, x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    zero_grads(model, x)
    y = model(x)
    loss = (y.float() * dy.float()).sum()
    loss.backward()
    return y


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


def measure_peak_delta(fn) -> tuple[int, int, int]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return int(peak - baseline), int(baseline), int(peak)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure FP4 LoRA activation checkpoint memory/latency tradeoff.")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--frozen-residual-rank", type=int, default=0)
    p.add_argument("--frozen-residual-init", choices=["none", "residual_svd"], default="none")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--no-cache-lora-act", action="store_true")
    p.add_argument("--intermediate-activation", choices=["none", "silu"], default="none")
    p.add_argument("--fuse-lora-dx", action="store_true")
    p.add_argument("--cache-fused-lora-dx", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def make_model(args: argparse.Namespace, checkpoint_scope: str) -> StackedFP4LoRA:
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    lowrank_dtype = torch.float16 if args.lowrank_dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed)
    model = StackedFP4LoRA(
        layers=args.layers,
        hidden=args.hidden,
        rank=args.rank,
        dtype=dtype,
        lowrank_dtype=lowrank_dtype,
        checkpoint_scope=checkpoint_scope,
        cache_lora_act=not args.no_cache_lora_act,
        fuse_lora_dx=args.fuse_lora_dx,
        cache_fused_lora_dx=args.cache_fused_lora_dx,
        frozen_residual_rank=args.frozen_residual_rank,
        frozen_residual_init=args.frozen_residual_init,
        intermediate_activation=args.intermediate_activation,
    )
    model.train()
    model.refresh_fused_lora_dx_caches()
    return model


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed + 1)
    x = torch.randn(args.batch, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(args.batch, args.hidden, device="cuda", dtype=dtype)
    return x, dy


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    base = make_model(args, checkpoint_scope="none")
    x_base, dy = make_inputs(args)

    y_base = train_step(base, x_base, dy)
    errors = {}
    for scope in ("module", "stack"):
        ckpt = make_model(args, checkpoint_scope=scope)
        ckpt.load_state_dict(base.state_dict(), strict=True)
        ckpt.refresh_fused_lora_dx_caches()
        x_ckpt = x_base.detach().clone().requires_grad_(True)
        y_ckpt = train_step(ckpt, x_ckpt, dy)
        errors[scope] = {
            "forward": tensor_error(y_ckpt, y_base),
            "dx": tensor_error(x_ckpt.grad, x_base.grad),
            "first_lora_down_grad": tensor_error(ckpt.layers[0].lora_down.grad, base.layers[0].lora_down.grad),
            "first_lora_up_grad": tensor_error(ckpt.layers[0].lora_up.grad, base.layers[0].lora_up.grad),
            "last_lora_down_grad": tensor_error(ckpt.layers[-1].lora_down.grad, base.layers[-1].lora_down.grad),
            "last_lora_up_grad": tensor_error(ckpt.layers[-1].lora_up.grad, base.layers[-1].lora_up.grad),
        }
        del ckpt, x_ckpt, y_ckpt
    del base, x_base, dy, y_base
    torch.cuda.empty_cache()

    def run_variant(checkpoint_scope: str) -> tuple[float, tuple[int, int, int]]:
        model = make_model(args, checkpoint_scope=checkpoint_scope)
        x, dy = make_inputs(args)

        def fn() -> None:
            train_step(model, x, dy)

        peak = measure_peak_delta(fn)
        latency = time_cuda(fn, warmup=args.warmup, iters=args.iters)
        zero_grads(model, x)
        del model, x, dy
        torch.cuda.empty_cache()
        return latency, peak

    base_ms, base_peak = run_variant("none")
    module_ms, module_peak = run_variant("module")
    stack_ms, stack_peak = run_variant("stack")
    module_memory_reduction = 1.0 - (float(module_peak[0]) / float(base_peak[0])) if base_peak[0] > 0 else None
    stack_memory_reduction = 1.0 - (float(stack_peak[0]) / float(base_peak[0])) if base_peak[0] > 0 else None

    payload = {
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "layers": args.layers,
            "rank": args.rank,
            "frozen_residual_rank": args.frozen_residual_rank,
            "frozen_residual_init": args.frozen_residual_init,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "cache_lora_act": not args.no_cache_lora_act,
            "intermediate_activation": args.intermediate_activation,
            "fuse_lora_dx": args.fuse_lora_dx,
            "cache_fused_lora_dx": args.cache_fused_lora_dx,
        },
        "latency_ms": {
            "no_activation_checkpoint_train_step": base_ms,
            "module_activation_checkpoint_train_step": module_ms,
            "stack_activation_checkpoint_train_step": stack_ms,
        },
        "peak_memory_bytes": {
            "no_activation_checkpoint_delta": base_peak[0],
            "module_activation_checkpoint_delta": module_peak[0],
            "stack_activation_checkpoint_delta": stack_peak[0],
            "no_activation_checkpoint_baseline": base_peak[1],
            "module_activation_checkpoint_baseline": module_peak[1],
            "stack_activation_checkpoint_baseline": stack_peak[1],
            "no_activation_checkpoint_peak": base_peak[2],
            "module_activation_checkpoint_peak": module_peak[2],
            "stack_activation_checkpoint_peak": stack_peak[2],
        },
        "derived": {
            "module_activation_checkpoint_latency_over_no_checkpoint": module_ms / base_ms,
            "module_activation_checkpoint_speed_vs_no_checkpoint": base_ms / module_ms,
            "module_activation_checkpoint_peak_delta_reduction": module_memory_reduction,
            "stack_activation_checkpoint_latency_over_no_checkpoint": stack_ms / base_ms,
            "stack_activation_checkpoint_speed_vs_no_checkpoint": base_ms / stack_ms,
            "stack_activation_checkpoint_peak_delta_reduction": stack_memory_reduction,
        },
        "errors": errors,
        "checks": {},
    }
    for scope, scope_errors in errors.items():
        payload["checks"].update(
            {
                f"{scope}_forward_rel_l2_lt_1e-6": scope_errors["forward"]["rel_l2"] < 1e-6,
                f"{scope}_dx_rel_l2_lt_1e-6": scope_errors["dx"]["rel_l2"] < 1e-6,
                f"{scope}_first_lora_down_grad_rel_l2_lt_1e-6": (
                    scope_errors["first_lora_down_grad"]["rel_l2"] < 1e-6
                ),
                f"{scope}_first_lora_up_grad_rel_l2_lt_1e-6": (
                    scope_errors["first_lora_up_grad"]["rel_l2"] < 1e-6
                ),
                f"{scope}_last_lora_down_grad_rel_l2_lt_1e-6": (
                    scope_errors["last_lora_down_grad"]["rel_l2"] < 1e-6
                ),
                f"{scope}_last_lora_up_grad_rel_l2_lt_1e-6": (
                    scope_errors["last_lora_up_grad"]["rel_l2"] < 1e-6
                ),
            }
        )
    payload["all_passed"] = bool(all(payload["checks"].values()))

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"fp4_lora_activation_checkpoint_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_fp4_lora_activation_checkpoint.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
