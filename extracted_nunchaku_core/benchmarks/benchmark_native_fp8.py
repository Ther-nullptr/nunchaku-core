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

from native_fp8 import NunchakuFP8GemmOp  # noqa: E402


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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
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

    fp8_gemm = NunchakuFP8GemmOp(weight=w, bias=b)
    qx, scale_x = fp8_gemm.quantize_input(x)

    def fp16_fn() -> torch.Tensor:
        return torch.matmul(x, w.t()) + b

    def fp8_fn() -> torch.Tensor:
        return fp8_gemm(x)

    def fp8_prequantized_fn() -> torch.Tensor:
        return fp8_gemm.forward_prequantized(qx, scale_x)

    with torch.no_grad():
        y_ref = fp16_fn().float()
        y_fp8 = fp8_fn().float()
        y_fp8_preq = fp8_prequantized_fn().float()

    fp16_ms = time_cuda(fp16_fn, warmup=args.warmup, iters=args.iters)
    fp8_ms = time_cuda(fp8_fn, warmup=args.warmup, iters=args.iters)
    fp8_preq_ms = time_cuda(fp8_prequantized_fn, warmup=args.warmup, iters=args.iters)

    payload = {
        "m": args.m,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "dtype": args.dtype,
        "fp16_ms": fp16_ms,
        "fp8_gemm_ms": fp8_ms,
        "fp8_gemm_prequantized_ms": fp8_preq_ms,
        "fp8_gemm_speedup_vs_fp16": fp16_ms / fp8_ms,
        "fp8_gemm_prequantized_speedup_vs_fp16": fp16_ms / fp8_preq_ms,
        "fp8_gemm_mae_vs_fp16": (y_fp8 - y_ref).abs().mean().item(),
        "fp8_gemm_rel_l2_vs_fp16": ((y_fp8 - y_ref).norm() / y_ref.norm().clamp_min(1e-12)).item(),
        "fp8_gemm_prequantized_mae_vs_fp16": (y_fp8_preq - y_ref).abs().mean().item(),
        "fp8_gemm_prequantized_rel_l2_vs_fp16": ((y_fp8_preq - y_ref).norm() / y_ref.norm().clamp_min(1e-12)).item(),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp8_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_native_fp8.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to: {out_path}")


if __name__ == "__main__":
    main()
