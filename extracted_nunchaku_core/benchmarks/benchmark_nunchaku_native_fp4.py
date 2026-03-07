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

from native_fp4 import NunchakuFP4GemmOp, NunchakuFP4LowRankOp  # noqa: E402


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    ms_list = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ms_list.append(start.elapsed_time(end))

    return float(sum(ms_list) / len(ms_list))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    x = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype)
    w = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)
    b = torch.randn(args.out_features, device="cuda", dtype=dtype)

    fp4_gemm = NunchakuFP4GemmOp(weight=w, bias=b)
    fp4_hybrid = NunchakuFP4LowRankOp(weight=w, bias=b, rank=args.rank)

    def fp16_fn() -> torch.Tensor:
        return torch.matmul(x, w.t()) + b

    def fp4_fn() -> torch.Tensor:
        return fp4_gemm(x)

    def fp4_hybrid_fn() -> torch.Tensor:
        return fp4_hybrid(x)

    with torch.no_grad():
        y_ref = fp16_fn().float()
        y_fp4 = fp4_fn().float()
        y_hybrid = fp4_hybrid_fn().float()

    fp16_ms = time_cuda(fp16_fn, warmup=args.warmup, iters=args.iters)
    fp4_ms = time_cuda(fp4_fn, warmup=args.warmup, iters=args.iters)
    hybrid_ms = time_cuda(fp4_hybrid_fn, warmup=args.warmup, iters=args.iters)

    payload = {
        "m": args.m,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "rank": args.rank,
        "dtype": args.dtype,
        "fp16_ms": fp16_ms,
        "fp4_gemm_ms": fp4_ms,
        "fp4_hybrid_ms": hybrid_ms,
        "fp4_gemm_speedup_vs_fp16": fp16_ms / fp4_ms,
        "fp4_hybrid_speedup_vs_fp16": fp16_ms / hybrid_ms,
        "fp4_gemm_mae_vs_fp16": (y_fp4 - y_ref).abs().mean().item(),
        "fp4_hybrid_mae_vs_fp16": (y_hybrid - y_ref).abs().mean().item(),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp4_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_native_fp4.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
