from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch


def parse_int_list(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("expected at least one integer")
    return out


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


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


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm()
    denom = b.float().norm().clamp_min(1e-12)
    return float((diff / denom).item())


def benchmark_one(
    *,
    m: int,
    in_features: int,
    out_features: int,
    rank: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    x = torch.randn(m, in_features, device="cuda", dtype=dtype)
    dy = torch.randn(m, out_features, device="cuda", dtype=dtype)
    lora_down = torch.randn(rank, in_features, device="cuda", dtype=dtype)
    lora_up = torch.randn(out_features, rank, device="cuda", dtype=dtype)
    scaling = 1.0 / max(rank, 1)

    lora_act = torch.matmul(x, lora_down.t()).contiguous()
    dy_up = torch.matmul(dy, lora_up).contiguous()
    torch.cuda.synchronize()

    def lora_act_build_fn() -> torch.Tensor:
        return torch.matmul(x, lora_down.t())

    def dy_up_fn() -> torch.Tensor:
        return torch.matmul(dy, lora_up)

    def d_lora_up_fn() -> torch.Tensor:
        return torch.matmul(dy.t(), lora_act).mul(scaling)

    def d_lora_down_fn() -> torch.Tensor:
        return torch.matmul(dy_up.t(), x).mul(scaling)

    def grad_pair_reuse_dy_up_fn() -> tuple[torch.Tensor, torch.Tensor]:
        d_lora_up = torch.matmul(dy.t(), lora_act).mul(scaling)
        d_lora_down = torch.matmul(dy_up.t(), x).mul(scaling)
        return d_lora_down, d_lora_up

    def grad_pair_sequential_fn() -> tuple[torch.Tensor, torch.Tensor]:
        dy_up_local = torch.matmul(dy, lora_up)
        d_lora_up = torch.matmul(dy.t(), lora_act).mul(scaling)
        d_lora_down = torch.matmul(dy_up_local.t(), x).mul(scaling)
        return d_lora_down, d_lora_up

    def grad_pair_overlap_fn() -> tuple[torch.Tensor, torch.Tensor]:
        current_stream = torch.cuda.current_stream(device=dy.device)
        up_stream = torch.cuda.Stream(device=dy.device)
        down_stream = torch.cuda.Stream(device=dy.device)

        up_stream.wait_stream(current_stream)
        with torch.cuda.stream(up_stream):
            d_lora_up = torch.matmul(dy.t(), lora_act).mul(scaling)

        down_stream.wait_stream(current_stream)
        with torch.cuda.stream(down_stream):
            dy_up_local = torch.matmul(dy, lora_up)
            d_lora_down = torch.matmul(dy_up_local.t(), x).mul(scaling)

        current_stream.wait_stream(up_stream)
        current_stream.wait_stream(down_stream)
        return d_lora_down, d_lora_up

    ref_down, ref_up = grad_pair_sequential_fn()
    overlap_down, overlap_up = grad_pair_overlap_fn()
    torch.cuda.synchronize()

    latency = {
        "lora_act_build": time_cuda(lora_act_build_fn, warmup, iters),
        "dy_up": time_cuda(dy_up_fn, warmup, iters),
        "d_lora_up": time_cuda(d_lora_up_fn, warmup, iters),
        "d_lora_down_reuse_dy_up": time_cuda(d_lora_down_fn, warmup, iters),
        "grad_pair_reuse_dy_up": time_cuda(grad_pair_reuse_dy_up_fn, warmup, iters),
        "grad_pair_sequential": time_cuda(grad_pair_sequential_fn, warmup, iters),
        "grad_pair_overlap": time_cuda(grad_pair_overlap_fn, warmup, iters),
    }

    return {
        "shape": {
            "m": m,
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "latency_ms": latency,
        "derived": {
            "overlap_speedup_vs_sequential": latency["grad_pair_sequential"] / latency["grad_pair_overlap"],
            "reuse_dy_up_speedup_vs_sequential": latency["grad_pair_sequential"]
            / latency["grad_pair_reuse_dy_up"],
            "dy_up_share_of_sequential": latency["dy_up"] / latency["grad_pair_sequential"],
            "d_lora_up_share_of_sequential": latency["d_lora_up"] / latency["grad_pair_sequential"],
            "d_lora_down_reuse_share_of_sequential": latency["d_lora_down_reuse_dy_up"]
            / latency["grad_pair_sequential"],
        },
        "correctness": {
            "overlap_d_lora_down_rel_l2": rel_l2(overlap_down, ref_down),
            "overlap_d_lora_up_rel_l2": rel_l2(overlap_up, ref_up),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark exact FP4 LoRA low-rank gradient sub-graphs.")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--ranks", type=str, default="16,32,64,128")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = dtype_from_name(args.dtype)
    ranks = parse_int_list(args.ranks)
    results = [
        benchmark_one(
            m=args.m,
            in_features=args.in_features,
            out_features=args.out_features,
            rank=rank,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed + i,
        )
        for i, rank in enumerate(ranks)
    ]

    payload = {
        "experiment": "fp4_lora_lowrank_grad",
        "notes": {
            "grad_pair_sequential": "dy_up=dY@B, dB=dY^T@lora_act, dA=dy_up^T@X on the current stream",
            "grad_pair_overlap": "dB overlaps with the dy_up+dA chain on a second CUDA stream",
            "grad_pair_reuse_dy_up": "measures only dA+dB when dy_up is already available from fused dX",
        },
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "dtype": args.dtype,
            "ranks": ranks,
        },
        "results": results,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    latest_path = os.path.join(args.results_dir, "latest_fp4_lora_lowrank_grad.json")
    stamped_path = os.path.join(args.results_dir, f"fp4_lora_lowrank_grad_{stamp}.json")
    for path in (latest_path, stamped_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {latest_path}")


if __name__ == "__main__":
    main()
