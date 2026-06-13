from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import NunchakuFP4LoRALinear  # noqa: E402


class DenseLoRALinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: int,
        lora_alpha: float,
        lowrank_dtype: torch.dtype,
        train_bias: bool,
    ):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.rank = rank
        self.scaling = float(lora_alpha) / float(rank)
        self.lowrank_dtype = lowrank_dtype
        self.register_buffer("weight", weight.detach().contiguous(), persistent=True)
        if bias is None:
            self.register_parameter("bias", None)
        elif train_bias:
            self.bias = torch.nn.Parameter(bias.detach().contiguous())
        else:
            self.register_buffer("bias", bias.detach().contiguous(), persistent=True)
        self.lora_down = torch.nn.Parameter(torch.empty(rank, self.in_features, device=weight.device, dtype=lowrank_dtype))
        self.lora_up = torch.nn.Parameter(torch.empty(self.out_features, rank, device=weight.device, dtype=lowrank_dtype))
        torch.nn.init.normal_(self.lora_down, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.lora_up, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        x_lr = x.reshape(-1, self.in_features).to(self.lowrank_dtype)
        lora_act = torch.matmul(x_lr, self.lora_down.t())
        lora_out = torch.matmul(lora_act, self.lora_up.t()).mul(self.scaling)
        return y + lora_out.to(y.dtype).reshape(*x.shape[:-1], self.out_features)


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ms.append(start.elapsed_time(end))
    return float(sum(ms) / len(ms))


def zero_grads(module: torch.nn.Module, x: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    x.grad = None


def benchmark_forward(
    module: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    track_grad: bool,
) -> float:
    def fn() -> None:
        y = module(x)
        # Materialize a scalar dependency so graph-building work cannot be skipped.
        _ = y.float().sum()

    if track_grad:
        return time_cuda(fn, warmup=warmup, iters=iters)
    with torch.no_grad():
        return time_cuda(fn, warmup=warmup, iters=iters)


def benchmark_train_step(
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


def benchmark_grad_accum(
    module: torch.nn.Module,
    x: torch.Tensor,
    dy: torch.Tensor,
    warmup: int,
    iters: int,
    steps: int,
    refresh_fused_lora_dx_cache: bool = False,
) -> float:
    if steps <= 0:
        raise ValueError("steps must be positive")

    def fn() -> None:
        zero_grads(module, x)
        if refresh_fused_lora_dx_cache:
            module.clear_fused_lora_dx_cache()
            module.refresh_fused_lora_dx_cache()
        for _ in range(steps):
            y = module(x)
            loss = (y.float() * dy.float()).sum()
            loss.backward()

    ms = time_cuda(fn, warmup=warmup, iters=iters)
    zero_grads(module, x)
    return ms


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
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--train-bias", action="store_true")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
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
    can_reuse_fused_dy_up = dtype == torch.float16 and lowrank_dtype == torch.float16

    fp4_cached = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        train_bias=args.train_bias,
        cache_lora_act=True,
    )
    fp4_recompute = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        train_bias=args.train_bias,
        cache_lora_act=False,
    )
    fp4_cached_fused_dx = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        train_bias=args.train_bias,
        cache_lora_act=True,
        fuse_lora_dx=True,
    )
    fp4_recompute_fused_dx = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        train_bias=args.train_bias,
        cache_lora_act=False,
        fuse_lora_dx=True,
    )
    fp4_cached_fused_dx_cached_pack = NunchakuFP4LoRALinear(
        weight=weight,
        bias=bias,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        lowrank_dtype=lowrank_dtype,
        init="gaussian",
        train_bias=args.train_bias,
        cache_lora_act=True,
        fuse_lora_dx=True,
        cache_fused_lora_dx=True,
    )
    fp4_cached_fused_dx_cached_pack_reuse_dy_up = None
    fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap = None
    if can_reuse_fused_dy_up:
        fp4_cached_fused_dx_cached_pack_reuse_dy_up = NunchakuFP4LoRALinear(
            weight=weight,
            bias=bias,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            lowrank_dtype=lowrank_dtype,
            init="gaussian",
            train_bias=args.train_bias,
            cache_lora_act=True,
            fuse_lora_dx=True,
            cache_fused_lora_dx=True,
            reuse_fused_dy_up_for_d_lora_down=True,
        )
        fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap = NunchakuFP4LoRALinear(
            weight=weight,
            bias=bias,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            lowrank_dtype=lowrank_dtype,
            init="gaussian",
            train_bias=args.train_bias,
            cache_lora_act=True,
            fuse_lora_dx=True,
            cache_fused_lora_dx=True,
            reuse_fused_dy_up_for_d_lora_down=True,
            overlap_lora_grad=True,
        )
    dense = DenseLoRALinear(
        weight=weight,
        bias=bias,
        rank=fp4_cached.rank,
        lora_alpha=fp4_cached.lora_alpha,
        lowrank_dtype=lowrank_dtype,
        train_bias=args.train_bias,
    )

    with torch.no_grad():
        modules_to_sync = [
            fp4_recompute,
            fp4_cached_fused_dx,
            fp4_recompute_fused_dx,
            fp4_cached_fused_dx_cached_pack,
            dense,
        ]
        if fp4_cached_fused_dx_cached_pack_reuse_dy_up is not None:
            modules_to_sync.append(fp4_cached_fused_dx_cached_pack_reuse_dy_up)
        if fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap is not None:
            modules_to_sync.append(fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap)
        for module in modules_to_sync:
            module.lora_down.copy_(fp4_cached.lora_down)
            module.lora_up.copy_(fp4_cached.lora_up)
        if args.train_bias:
            for module in modules_to_sync:
                module.bias.copy_(fp4_cached.bias)

    with torch.no_grad():
        y_cached = fp4_cached(x)
        y_recompute = fp4_recompute(x)
        y_dense = dense(x)
        forward_cache_vs_recompute = tensor_error(y_cached, y_recompute)
        forward_fp4_vs_dense = tensor_error(y_cached, y_dense)

    dense_forward_inference_ms = benchmark_forward(dense, x, args.warmup, args.iters, track_grad=False)
    fp4_cached_forward_inference_ms = benchmark_forward(fp4_cached, x, args.warmup, args.iters, track_grad=False)
    fp4_recompute_forward_inference_ms = benchmark_forward(fp4_recompute, x, args.warmup, args.iters, track_grad=False)

    dense_forward_train_graph_ms = benchmark_forward(dense, x, args.warmup, args.iters, track_grad=True)
    fp4_cached_forward_train_graph_ms = benchmark_forward(fp4_cached, x, args.warmup, args.iters, track_grad=True)
    fp4_recompute_forward_train_graph_ms = benchmark_forward(fp4_recompute, x, args.warmup, args.iters, track_grad=True)

    dense_train_step_ms = benchmark_train_step(dense, x, dy, args.warmup, args.iters)
    fp4_cached_train_step_ms = benchmark_train_step(fp4_cached, x, dy, args.warmup, args.iters)
    fp4_recompute_train_step_ms = benchmark_train_step(fp4_recompute, x, dy, args.warmup, args.iters)
    fp4_cached_fused_dx_train_step_ms = benchmark_train_step(
        fp4_cached_fused_dx, x, dy, args.warmup, args.iters
    )
    fp4_recompute_fused_dx_train_step_ms = benchmark_train_step(
        fp4_recompute_fused_dx, x, dy, args.warmup, args.iters
    )

    def refresh_fused_lora_dx_cache_fn() -> None:
        fp4_cached_fused_dx_cached_pack.clear_fused_lora_dx_cache()
        fp4_cached_fused_dx_cached_pack.refresh_fused_lora_dx_cache()

    refresh_fused_lora_dx_cache_ms = time_cuda(
        refresh_fused_lora_dx_cache_fn, warmup=args.warmup, iters=args.iters
    )
    fp4_cached_fused_dx_cached_pack_train_step_ms = benchmark_train_step(
        fp4_cached_fused_dx_cached_pack, x, dy, args.warmup, args.iters
    )
    fp4_cached_fused_dx_cached_pack_reuse_dy_up_train_step_ms = None
    if fp4_cached_fused_dx_cached_pack_reuse_dy_up is not None:
        fp4_cached_fused_dx_cached_pack_reuse_dy_up_train_step_ms = benchmark_train_step(
            fp4_cached_fused_dx_cached_pack_reuse_dy_up, x, dy, args.warmup, args.iters
        )
    fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_train_step_ms = None
    if fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap is not None:
        fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_train_step_ms = benchmark_train_step(
            fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap, x, dy, args.warmup, args.iters
        )
    dense_grad_accum_total_ms = benchmark_grad_accum(
        dense, x, dy, args.warmup, args.iters, args.grad_accum_steps
    )
    fp4_cached_fused_dx_grad_accum_total_ms = benchmark_grad_accum(
        fp4_cached_fused_dx, x, dy, args.warmup, args.iters, args.grad_accum_steps
    )
    fp4_cached_fused_dx_cached_pack_grad_accum_total_ms = benchmark_grad_accum(
        fp4_cached_fused_dx_cached_pack, x, dy, args.warmup, args.iters, args.grad_accum_steps
    )
    fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_total_ms = None
    if fp4_cached_fused_dx_cached_pack_reuse_dy_up is not None:
        fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_total_ms = benchmark_grad_accum(
            fp4_cached_fused_dx_cached_pack_reuse_dy_up, x, dy, args.warmup, args.iters, args.grad_accum_steps
        )
    fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_grad_accum_total_ms = None
    if fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap is not None:
        fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_grad_accum_total_ms = benchmark_grad_accum(
            fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap,
            x,
            dy,
            args.warmup,
            args.iters,
            args.grad_accum_steps,
        )
    fp4_cached_fused_dx_cached_pack_refresh_grad_accum_total_ms = benchmark_grad_accum(
        fp4_cached_fused_dx_cached_pack,
        x,
        dy,
        args.warmup,
        args.iters,
        args.grad_accum_steps,
        refresh_fused_lora_dx_cache=True,
    )

    reuse_train_step_ms = fp4_cached_fused_dx_cached_pack_reuse_dy_up_train_step_ms
    reuse_grad_accum_total_ms = fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_total_ms
    overlap_train_step_ms = fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_train_step_ms
    overlap_grad_accum_total_ms = fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_grad_accum_total_ms
    reuse_grad_accum_per_micro_step_ms = (
        None if reuse_grad_accum_total_ms is None else reuse_grad_accum_total_ms / args.grad_accum_steps
    )
    overlap_grad_accum_per_micro_step_ms = (
        None if overlap_grad_accum_total_ms is None else overlap_grad_accum_total_ms / args.grad_accum_steps
    )
    reuse_backward_estimate_ms = (
        None if reuse_train_step_ms is None else reuse_train_step_ms - fp4_cached_forward_train_graph_ms
    )
    overlap_backward_estimate_ms = (
        None if overlap_train_step_ms is None else overlap_train_step_ms - fp4_cached_forward_train_graph_ms
    )

    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "effective_rank": fp4_cached.rank,
            "lora_alpha": fp4_cached.lora_alpha,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "train_bias": args.train_bias,
            "grad_accum_steps": args.grad_accum_steps,
            "can_reuse_fused_dy_up": can_reuse_fused_dy_up,
        },
        "latency_ms": {
            "dense_forward_inference": dense_forward_inference_ms,
            "fp4_cached_forward_inference": fp4_cached_forward_inference_ms,
            "fp4_recompute_forward_inference": fp4_recompute_forward_inference_ms,
            "dense_forward_train_graph": dense_forward_train_graph_ms,
            "fp4_cached_forward_train_graph": fp4_cached_forward_train_graph_ms,
            "fp4_recompute_forward_train_graph": fp4_recompute_forward_train_graph_ms,
            "dense_train_step": dense_train_step_ms,
            "fp4_cached_train_step": fp4_cached_train_step_ms,
            "fp4_recompute_train_step": fp4_recompute_train_step_ms,
            "fp4_cached_fused_dx_train_step": fp4_cached_fused_dx_train_step_ms,
            "fp4_recompute_fused_dx_train_step": fp4_recompute_fused_dx_train_step_ms,
            "fp4_cached_fused_dx_cached_pack_train_step": fp4_cached_fused_dx_cached_pack_train_step_ms,
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_train_step": (
                fp4_cached_fused_dx_cached_pack_reuse_dy_up_train_step_ms
            ),
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_train_step": (
                fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_train_step_ms
            ),
            "refresh_fused_lora_dx_cache": refresh_fused_lora_dx_cache_ms,
            "fp4_cached_fused_dx_cached_pack_plus_refresh_train_step": fp4_cached_fused_dx_cached_pack_train_step_ms
            + refresh_fused_lora_dx_cache_ms,
            "dense_grad_accum_total": dense_grad_accum_total_ms,
            "fp4_cached_fused_dx_grad_accum_total": fp4_cached_fused_dx_grad_accum_total_ms,
            "fp4_cached_fused_dx_cached_pack_grad_accum_total": fp4_cached_fused_dx_cached_pack_grad_accum_total_ms,
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_total": (
                fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_total_ms
            ),
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_grad_accum_total": (
                fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_grad_accum_total_ms
            ),
            "fp4_cached_fused_dx_cached_pack_refresh_grad_accum_total": (
                fp4_cached_fused_dx_cached_pack_refresh_grad_accum_total_ms
            ),
            "dense_grad_accum_per_micro_step": dense_grad_accum_total_ms / args.grad_accum_steps,
            "fp4_cached_fused_dx_grad_accum_per_micro_step": fp4_cached_fused_dx_grad_accum_total_ms
            / args.grad_accum_steps,
            "fp4_cached_fused_dx_cached_pack_grad_accum_per_micro_step": (
                fp4_cached_fused_dx_cached_pack_grad_accum_total_ms / args.grad_accum_steps
            ),
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_per_micro_step": (
                reuse_grad_accum_per_micro_step_ms
            ),
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_grad_accum_per_micro_step": (
                overlap_grad_accum_per_micro_step_ms
            ),
            "fp4_cached_fused_dx_cached_pack_refresh_grad_accum_per_micro_step": (
                fp4_cached_fused_dx_cached_pack_refresh_grad_accum_total_ms / args.grad_accum_steps
            ),
            "dense_backward_estimate": dense_train_step_ms - dense_forward_train_graph_ms,
            "fp4_cached_backward_estimate": fp4_cached_train_step_ms - fp4_cached_forward_train_graph_ms,
            "fp4_recompute_backward_estimate": fp4_recompute_train_step_ms - fp4_recompute_forward_train_graph_ms,
            "fp4_cached_fused_dx_backward_estimate": fp4_cached_fused_dx_train_step_ms
            - fp4_cached_forward_train_graph_ms,
            "fp4_recompute_fused_dx_backward_estimate": fp4_recompute_fused_dx_train_step_ms
            - fp4_recompute_forward_train_graph_ms,
            "fp4_cached_fused_dx_cached_pack_backward_estimate": fp4_cached_fused_dx_cached_pack_train_step_ms
            - fp4_cached_forward_train_graph_ms,
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_backward_estimate": reuse_backward_estimate_ms,
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_backward_estimate": overlap_backward_estimate_ms,
        },
        "speedups": {
            "fp4_cached_forward_inference_vs_dense": dense_forward_inference_ms / fp4_cached_forward_inference_ms,
            "fp4_recompute_forward_inference_vs_dense": dense_forward_inference_ms
            / fp4_recompute_forward_inference_ms,
            "fp4_cached_forward_train_graph_vs_dense": dense_forward_train_graph_ms
            / fp4_cached_forward_train_graph_ms,
            "fp4_recompute_forward_train_graph_vs_dense": dense_forward_train_graph_ms
            / fp4_recompute_forward_train_graph_ms,
            "fp4_cached_train_step_vs_dense": dense_train_step_ms / fp4_cached_train_step_ms,
            "fp4_recompute_train_step_vs_dense": dense_train_step_ms / fp4_recompute_train_step_ms,
            "fp4_cached_fused_dx_train_step_vs_dense": dense_train_step_ms / fp4_cached_fused_dx_train_step_ms,
            "fp4_recompute_fused_dx_train_step_vs_dense": dense_train_step_ms
            / fp4_recompute_fused_dx_train_step_ms,
            "fp4_cached_fused_dx_cached_pack_train_step_vs_dense": dense_train_step_ms
            / fp4_cached_fused_dx_cached_pack_train_step_ms,
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_train_step_vs_dense": (
                None if reuse_train_step_ms is None else dense_train_step_ms / reuse_train_step_ms
            ),
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_train_step_vs_dense": (
                None if overlap_train_step_ms is None else dense_train_step_ms / overlap_train_step_ms
            ),
            "fp4_cached_fused_dx_cached_pack_plus_refresh_train_step_vs_dense": dense_train_step_ms
            / (fp4_cached_fused_dx_cached_pack_train_step_ms + refresh_fused_lora_dx_cache_ms),
            "fp4_cached_fused_dx_grad_accum_vs_dense": dense_grad_accum_total_ms
            / fp4_cached_fused_dx_grad_accum_total_ms,
            "fp4_cached_fused_dx_cached_pack_grad_accum_vs_dense": dense_grad_accum_total_ms
            / fp4_cached_fused_dx_cached_pack_grad_accum_total_ms,
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_vs_dense": (
                None if reuse_grad_accum_total_ms is None else dense_grad_accum_total_ms / reuse_grad_accum_total_ms
            ),
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_grad_accum_vs_dense": (
                None
                if overlap_grad_accum_total_ms is None
                else dense_grad_accum_total_ms / overlap_grad_accum_total_ms
            ),
            "fp4_cached_fused_dx_cached_pack_refresh_grad_accum_vs_dense": dense_grad_accum_total_ms
            / fp4_cached_fused_dx_cached_pack_refresh_grad_accum_total_ms,
            "fp4_cached_backward_estimate_vs_dense": (dense_train_step_ms - dense_forward_train_graph_ms)
            / (fp4_cached_train_step_ms - fp4_cached_forward_train_graph_ms),
            "fp4_recompute_backward_estimate_vs_dense": (dense_train_step_ms - dense_forward_train_graph_ms)
            / (fp4_recompute_train_step_ms - fp4_recompute_forward_train_graph_ms),
            "fp4_cached_fused_dx_backward_estimate_vs_dense": (dense_train_step_ms - dense_forward_train_graph_ms)
            / (fp4_cached_fused_dx_train_step_ms - fp4_cached_forward_train_graph_ms),
            "fp4_recompute_fused_dx_backward_estimate_vs_dense": (
                dense_train_step_ms - dense_forward_train_graph_ms
            )
            / (fp4_recompute_fused_dx_train_step_ms - fp4_recompute_forward_train_graph_ms),
            "fp4_cached_fused_dx_cached_pack_backward_estimate_vs_dense": (
                dense_train_step_ms - dense_forward_train_graph_ms
            )
            / (fp4_cached_fused_dx_cached_pack_train_step_ms - fp4_cached_forward_train_graph_ms),
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_backward_estimate_vs_dense": (
                None
                if reuse_backward_estimate_ms is None
                else (dense_train_step_ms - dense_forward_train_graph_ms) / reuse_backward_estimate_ms
            ),
            "fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_backward_estimate_vs_dense": (
                None
                if overlap_backward_estimate_ms is None
                else (dense_train_step_ms - dense_forward_train_graph_ms) / overlap_backward_estimate_ms
            ),
            "cache_vs_recompute_train_step": fp4_recompute_train_step_ms / fp4_cached_train_step_ms,
            "fused_dx_cached_vs_dense_dx_cached_train_step": fp4_cached_train_step_ms
            / fp4_cached_fused_dx_train_step_ms,
            "fused_dx_cached_pack_vs_dynamic_pack_train_step": fp4_cached_fused_dx_train_step_ms
            / fp4_cached_fused_dx_cached_pack_train_step_ms,
            "fused_dx_cached_pack_reuse_dy_up_vs_cached_pack_train_step": (
                None
                if reuse_train_step_ms is None
                else fp4_cached_fused_dx_cached_pack_train_step_ms / reuse_train_step_ms
            ),
            "fused_dx_cached_pack_reuse_overlap_vs_reuse_train_step": (
                None if overlap_train_step_ms is None else reuse_train_step_ms / overlap_train_step_ms
            ),
            "fused_dx_cached_pack_plus_refresh_vs_dynamic_pack_train_step": fp4_cached_fused_dx_train_step_ms
            / (fp4_cached_fused_dx_cached_pack_train_step_ms + refresh_fused_lora_dx_cache_ms),
            "fused_dx_cached_pack_vs_dynamic_pack_grad_accum": fp4_cached_fused_dx_grad_accum_total_ms
            / fp4_cached_fused_dx_cached_pack_grad_accum_total_ms,
            "fused_dx_cached_pack_reuse_dy_up_vs_cached_pack_grad_accum": (
                None
                if reuse_grad_accum_total_ms is None
                else fp4_cached_fused_dx_cached_pack_grad_accum_total_ms / reuse_grad_accum_total_ms
            ),
            "fused_dx_cached_pack_reuse_overlap_vs_reuse_grad_accum": (
                None
                if overlap_grad_accum_total_ms is None
                else reuse_grad_accum_total_ms / overlap_grad_accum_total_ms
            ),
            "fused_dx_cached_pack_refresh_vs_dynamic_pack_grad_accum": fp4_cached_fused_dx_grad_accum_total_ms
            / fp4_cached_fused_dx_cached_pack_refresh_grad_accum_total_ms,
            "fused_dx_recompute_vs_dense_dx_recompute_train_step": fp4_recompute_train_step_ms
            / fp4_recompute_fused_dx_train_step_ms,
        },
        "errors": {
            "forward_cache_vs_recompute": forward_cache_vs_recompute,
            "forward_fp4_vs_dense": forward_fp4_vs_dense,
        },
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp4_lora_training_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_native_fp4_lora_training.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
