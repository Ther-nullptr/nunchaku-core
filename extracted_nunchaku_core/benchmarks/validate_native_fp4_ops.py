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

from native_fp4.operators import (  # noqa: E402
    _OPS,
    NunchakuFP4GemmOp,
    NunchakuFP4LowRankOp,
    pad_tensor,
    quantize_fp4_act_with_lora,
)


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
    p.add_argument("--in-features", type=int, default=3072)
    p.add_argument("--out-features", type=int, default=3584)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
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

    pure = NunchakuFP4GemmOp(weight=w, bias=b)
    hybrid = NunchakuFP4LowRankOp(weight=w, bias=b, rank=args.rank)

    # wrapper output
    y_pure = pure(x)
    y_hybrid = hybrid(x)

    # prepare padded input for manual-path validation
    x2d_src = x.reshape(-1, pure.in_features)
    x2d = x2d_src
    if pure.k_pad != pure.in_features:
        x2d = pad_tensor(x2d, divisor=pure.k_pad, dim=1)

    def run_fp4_kernel(
        qact: torch.Tensor,
        ascales: torch.Tensor,
        lora_act: torch.Tensor | None,
        lora_up: torch.Tensor | None,
        lora_scales: list[float],
    ) -> torch.Tensor:
        out_pad = torch.empty(qact.shape[0], hybrid.n_pad, dtype=hybrid.compute_dtype, device=qact.device)
        _OPS.gemm_w4a4(
            qact,
            hybrid.qweight,
            out_pad,
            None,
            ascales,
            hybrid.wscales,
            None,
            None,
            lora_act,
            lora_up,
            None,
            None,
            None,
            None,
            None,
            hybrid.bias_pad,
            None,
            None,
            None,
            False,
            lora_scales,
            False,
            True,
            1.0,
            None,
            None,
            None,
            None,
            0,
        )
        return out_pad

    # check 1: pure wrapper == manual pure path
    qact_p, asc_p = pure.quantize_input(x2d)
    y_pure_manual_pad = pure.forward_prequantized(qact_p, asc_p)
    y_pure_manual = y_pure_manual_pad[: x2d_src.shape[0], : pure.out_features].reshape_as(y_pure)
    pure_wrapper_err = tensor_error(y_pure, y_pure_manual)

    # check 2: hybrid wrapper ~= manual hybrid kernel call
    qact_h, asc_h, lora_act_h = hybrid.quantize_input_with_lora(x2d)
    y_hybrid_manual_pad = run_fp4_kernel(qact_h, asc_h, lora_act_h, hybrid.lora_up_packed, hybrid._lora_scales)
    y_hybrid_manual = y_hybrid_manual_pad[: x2d_src.shape[0], : hybrid.out_features].reshape_as(y_hybrid)
    hybrid_wrapper_err = tensor_error(y_hybrid, y_hybrid_manual)

    # check 3: zero-up invariant
    y_pure_same_quant_pad = run_fp4_kernel(qact_h, asc_h, None, None, [])
    zero_up = torch.zeros_like(hybrid.lora_up_packed)
    y_zero_up_pad = run_fp4_kernel(qact_h, asc_h, lora_act_h, zero_up, hybrid._lora_scales)
    zero_up_err = tensor_error(y_zero_up_pad, y_pure_same_quant_pad)

    # check 4: zero-down invariant
    qact_zd, asc_zd, lora_act_zd = quantize_fp4_act_with_lora(
        x2d,
        lora_down_packed=hybrid.dummy_down_packed,
        smooth=hybrid.smooth,
        pad_size=256,
    )
    y_pure_zd_pad = run_fp4_kernel(qact_zd, asc_zd, None, None, [])
    y_with_up_zd_pad = run_fp4_kernel(qact_zd, asc_zd, lora_act_zd, hybrid.lora_up_packed, hybrid._lora_scales)
    zero_down_err = tensor_error(y_with_up_zd_pad, y_pure_zd_pad)

    # reference fp16 error (quality context)
    y_ref = (x @ w.t() + b).float()
    pure_vs_fp16 = tensor_error(y_pure, y_ref)
    hybrid_vs_fp16 = tensor_error(y_hybrid, y_ref)

    # practical thresholds: wrapper is one extra quantize call so allow tiny drift.
    checks = {
        "pure_wrapper_rel_l2_lt_1e-7": pure_wrapper_err["rel_l2"] < 1e-7,
        "hybrid_wrapper_rel_l2_lt_2e-5": hybrid_wrapper_err["rel_l2"] < 2e-5,
        "zero_up_rel_l2_lt_2e-5": zero_up_err["rel_l2"] < 2e-5,
        "zero_down_rel_l2_lt_2e-5": zero_down_err["rel_l2"] < 2e-5,
    }

    payload = {
        "shape": {
            "m": args.m,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "dtype": args.dtype,
        },
        "errors": {
            "pure_wrapper_vs_manual": pure_wrapper_err,
            "hybrid_wrapper_vs_manual": hybrid_wrapper_err,
            "zero_up_invariant": zero_up_err,
            "zero_down_invariant": zero_down_err,
            "pure_vs_fp16": pure_vs_fp16,
            "hybrid_vs_fp16": hybrid_vs_fp16,
        },
        "checks": checks,
        "all_passed": all(checks.values()),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp4_validation_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_native_fp4_validation.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved validation to: {out_path}")


if __name__ == "__main__":
    main()
