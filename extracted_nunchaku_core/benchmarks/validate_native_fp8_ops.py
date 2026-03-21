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


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    da = a.float()
    db = b.float()
    diff = (da - db).abs()
    ref_norm = db.norm().item()
    diff_norm = (da - db).norm().item()
    rel_l2 = diff_norm / (ref_norm + 1e-12)
    return {
        "max_abs": float(diff.max().item()),
        "mae": float(diff.mean().item()),
        "rel_l2": float(rel_l2),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=333)
    p.add_argument("--in-features", type=int, default=4096)
    p.add_argument("--out-features", type=int, default=4096)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    x = torch.randn(args.m, args.in_features, device="cuda", dtype=dtype)
    w = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)
    b = torch.randn(args.out_features, device="cuda", dtype=dtype)

    op = NunchakuFP8GemmOp(weight=w, bias=b)

    y = op(x)
    y_ref = (x.float() @ w.float().t() + b.float()).to(dtype)

    qx, sx = op.quantize_input(x)
    y_manual = op.forward_prequantized(qx, sx)

    wrapper_vs_manual = tensor_error(y, y_manual)
    fp8_vs_fp16 = tensor_error(y, y_ref)

    checks = {
        "wrapper_exact_match": wrapper_vs_manual["max_abs"] == 0.0,
        "fp8_rel_l2_lt_0p08": fp8_vs_fp16["rel_l2"] < 0.08,
        "fp8_mae_lt_2p5": fp8_vs_fp16["mae"] < 2.5,
        "all_finite": bool(torch.isfinite(y).all().item()),
    }

    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "dtype": args.dtype,
        },
        "errors": {
            "wrapper_vs_manual": wrapper_vs_manual,
            "fp8_vs_fp16": fp8_vs_fp16,
        },
        "checks": checks,
        "all_passed": all(checks.values()),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp8_validation_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_native_fp8_validation.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved validation to: {out_path}")


if __name__ == "__main__":
    main()
