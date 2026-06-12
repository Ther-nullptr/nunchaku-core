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
from native_fp4.training import _fused_lora_dx  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
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
        train_bias=False,
        cache_lora_act=True,
        fuse_lora_dx=True,
        cache_fused_lora_dx=True,
    )

    x2d = x.detach().reshape(-1, op.in_features)
    dy2d = dy.reshape(-1, op.out_features)
    x_lr = x2d.to(lowrank_dtype)
    dy_lr = dy2d.to(lowrank_dtype)
    down_lr = op.lora_down.detach().to(lowrank_dtype)
    up_lr = op.lora_up.detach().to(lowrank_dtype)
    lora_act = torch.matmul(x_lr, down_lr.t()).contiguous()
    dy_up = torch.matmul(dy_lr, up_lr).contiguous()
    op.refresh_fused_lora_dx_cache()
    packed_lora_dx = op._get_fused_lora_dx_cache()

    def forward_train_graph_fn() -> torch.Tensor:
        return op(x)

    def refresh_lora_pack_fn() -> None:
        op.clear_fused_lora_dx_cache()
        op.refresh_fused_lora_dx_cache()

    def repack_backbone_fn() -> torch.Tensor:
        return op.fp4_backward.repack_qweight_for_backward()

    def fp4_dx_main_fn() -> torch.Tensor:
        return op.fp4_backward(dy)

    def fused_dx_dynamic_pack_fn() -> torch.Tensor:
        return _fused_lora_dx(
            dy=dy,
            lora_down=op.lora_down,
            lora_up=op.lora_up,
            fp4_backward_op=op.fp4_backward,
            scaling=op.scaling,
            lowrank_dtype=op.lowrank_dtype,
            in_features=op.in_features,
            out_features=op.out_features,
            packed_lora_dx=None,
        )

    def fused_dx_cached_pack_fn() -> torch.Tensor:
        return _fused_lora_dx(
            dy=dy,
            lora_down=op.lora_down,
            lora_up=op.lora_up,
            fp4_backward_op=op.fp4_backward,
            scaling=op.scaling,
            lowrank_dtype=op.lowrank_dtype,
            in_features=op.in_features,
            out_features=op.out_features,
            packed_lora_dx=packed_lora_dx,
        )

    def dy_up_fn() -> torch.Tensor:
        return torch.matmul(dy_lr, up_lr)

    def dense_dx_lora_fn() -> torch.Tensor:
        return torch.matmul(dy_up, down_lr).mul(op.scaling)

    def d_lora_up_fn() -> torch.Tensor:
        return torch.matmul(dy_lr.t(), lora_act).mul(op.scaling)

    def d_lora_down_fn() -> torch.Tensor:
        return torch.matmul(dy_up.t(), x_lr).mul(op.scaling)

    def dense_lora_grad_pair_fn() -> tuple[torch.Tensor, torch.Tensor]:
        return d_lora_up_fn(), d_lora_down_fn()

    def full_backward_cached_pack_fn() -> None:
        op.zero_grad(set_to_none=True)
        x.grad = None
        y = op(x)
        loss = (y.float() * dy.float()).sum()
        loss.backward()

    latency = {
        "forward_train_graph": time_cuda(forward_train_graph_fn, args.warmup, args.iters),
        "refresh_lora_dx_pack": time_cuda(refresh_lora_pack_fn, args.warmup, args.iters),
        "repack_backbone": time_cuda(repack_backbone_fn, args.warmup, args.iters),
        "fp4_dx_main": time_cuda(fp4_dx_main_fn, args.warmup, args.iters),
        "fused_dx_dynamic_pack": time_cuda(fused_dx_dynamic_pack_fn, args.warmup, args.iters),
        "fused_dx_cached_pack": time_cuda(fused_dx_cached_pack_fn, args.warmup, args.iters),
        "dy_up": time_cuda(dy_up_fn, args.warmup, args.iters),
        "dense_dx_lora": time_cuda(dense_dx_lora_fn, args.warmup, args.iters),
        "d_lora_up": time_cuda(d_lora_up_fn, args.warmup, args.iters),
        "d_lora_down": time_cuda(d_lora_down_fn, args.warmup, args.iters),
        "dense_lora_grad_pair": time_cuda(dense_lora_grad_pair_fn, args.warmup, args.iters),
        "full_backward_cached_pack": time_cuda(full_backward_cached_pack_fn, args.warmup, args.iters),
    }

    backward = latency["full_backward_cached_pack"] - latency["forward_train_graph"]
    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "effective_rank": op.rank,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
        },
        "latency_ms": latency,
        "derived": {
            "full_backward_minus_forward": backward,
            "cached_pack_over_dynamic_pack": latency["fused_dx_dynamic_pack"] / latency["fused_dx_cached_pack"],
            "lora_pack_over_cached_fused_dx": latency["refresh_lora_dx_pack"] / latency["fused_dx_cached_pack"],
            "dy_up_over_backward": latency["dy_up"] / backward,
            "d_lora_up_over_backward": latency["d_lora_up"] / backward,
            "d_lora_down_over_backward": latency["d_lora_down"] / backward,
            "dense_lora_grad_pair_over_backward": latency["dense_lora_grad_pair"] / backward,
            "fused_dx_cached_pack_over_backward": latency["fused_dx_cached_pack"] / backward,
        },
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp4_lora_training_breakdown_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_native_fp4_lora_training_breakdown.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
