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

from native_fp4 import NunchakuFP4BackwardDXOp, NunchakuFP4LowRankBackwardDXOp  # noqa: E402
from native_fp4.layout import (  # noqa: E402
    pack_fp4_weight_codes,
    pack_fp4_weight_scales,
    repack_fp4_weight_for_backward,
    unpack_fp4_weight_codes,
    unpack_fp4_weight_scales,
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
    dy = torch.randn(args.m, args.out_features, device="cuda", dtype=dtype)
    w = torch.randn(args.out_features, args.in_features, device="cuda", dtype=dtype)

    pure_dx = NunchakuFP4BackwardDXOp(weight=w)
    hybrid_dx = NunchakuFP4LowRankBackwardDXOp(weight=w, rank=args.rank, lowrank_dtype=dtype)

    qweight_roundtrip = pack_fp4_weight_codes(unpack_fp4_weight_codes(pure_dx.qweight))
    fwd_wscale_logical = unpack_fp4_weight_scales(pure_dx.wscales, pure_dx.n_pad, pure_dx.k_pad)
    wscale_roundtrip = pack_fp4_weight_scales(fwd_wscale_logical)
    qweight_bwd_ref = repack_fp4_weight_for_backward(
        qweight=pure_dx.qweight,
        packed_wscales=pure_dx.wscales,
        out_features=pure_dx.n_pad,
        in_features=pure_dx.k_pad,
        logical_scales_t=pure_dx.wscales_bwd_logical,
    )
    qweight_bwd_cuda = pure_dx.repack_qweight_for_backward()

    dx_ref = torch.matmul(dy, w)
    dx_pure = pure_dx(dy)
    pure_dx_err = tensor_error(dx_pure, dx_ref)

    dx_fused = hybrid_dx(dy)
    dx_unfused = hybrid_dx.forward_unfused(dy)
    hybrid_dx_internal_err = tensor_error(dx_fused, dx_unfused)

    lora_up = hybrid_dx.lora_up_dense[: hybrid_dx.out_features, : hybrid_dx.rank].to(dtype)
    lora_down = hybrid_dx.lora_down_dense[: hybrid_dx.rank, : hybrid_dx.in_features].to(dtype)
    dx_hybrid_ref = torch.matmul(dy, w) + torch.matmul(torch.matmul(dy.to(dtype), lora_up), lora_down)
    hybrid_dx_ref_err = tensor_error(dx_fused, dx_hybrid_ref)

    full_fused = hybrid_dx.backward_full(x, dy, fused_dx=True)
    full_unfused = hybrid_dx.backward_full(x, dy, fused_dx=False)
    forward_lora_cache = hybrid_dx.build_forward_lowrank_cache(x)
    full_shared_recompute = hybrid_dx.backward_full_shared(x, dy)
    full_shared_cached = hybrid_dx.backward_full_shared(x, dy, forward_lora_act=forward_lora_cache)
    full_shared_packed = hybrid_dx.backward_full_shared_packed(x, dy, forward_lora_act=forward_lora_cache)
    full_shared_packed_overlap = hybrid_dx.backward_full_shared_packed_overlap(x, dy, forward_lora_act=forward_lora_cache)
    full_shared_dual = hybrid_dx.backward_full_shared_dual(x, dy, forward_lora_act=forward_lora_cache)

    x_lr = x.to(dtype)
    dy_lr = dy.to(dtype)
    lora_act = torch.matmul(x_lr, lora_down.t())
    dy_up = torch.matmul(dy_lr, lora_up)
    _, _, packed_dy_up = hybrid_dx.quantize_grad_with_lora(
        dy if hybrid_dx.n_pad == hybrid_dx.out_features else torch.nn.functional.pad(dy, (0, hybrid_dx.n_pad - hybrid_dx.out_features))
    )
    dense_dy_up = hybrid_dx.decode_packed_lowrank_act(packed_dy_up)[: dy.shape[0], : hybrid_dx.rank].to(dtype)
    _, _, packed_dy_up_dual, dense_dy_up_dual = hybrid_dx.quantize_grad_with_lora_dual(
        dy if hybrid_dx.n_pad == hybrid_dx.out_features else torch.nn.functional.pad(dy, (0, hybrid_dx.n_pad - hybrid_dx.out_features))
    )
    dense_dy_up_dual = dense_dy_up_dual[: dy.shape[0], : hybrid_dx.rank].to(dtype)
    full_ref = {
        "dx": dx_hybrid_ref,
        "lora_up_grad": torch.matmul(dy_lr.t(), lora_act),
        "lora_down_grad": torch.matmul(dy_up.t(), x_lr),
    }

    checks = {
        "qweight_roundtrip_exact": bool(torch.equal(qweight_roundtrip, pure_dx.qweight)),
        "wscale_roundtrip_exact": bool(
            torch.equal(wscale_roundtrip.view(torch.uint8), pure_dx.wscales.view(torch.uint8))
        ),
        "qweight_bwd_cuda_matches_reference": bool(torch.equal(qweight_bwd_cuda, qweight_bwd_ref)),
        "hybrid_dx_fused_vs_unfused_rel_l2_lt_5e-4": hybrid_dx_internal_err["rel_l2"] < 5e-4,
        "full_up_rel_l2_lt_1e-5": tensor_error(full_fused["lora_up_grad"], full_ref["lora_up_grad"])["rel_l2"] < 1e-5,
        "full_down_rel_l2_lt_1e-5": tensor_error(full_fused["lora_down_grad"], full_ref["lora_down_grad"])["rel_l2"]
        < 1e-5,
        "full_fused_vs_unfused_dx_rel_l2_lt_5e-4": tensor_error(full_fused["dx"], full_unfused["dx"])["rel_l2"] < 5e-4,
        "full_shared_cached_dx_matches_fused_rel_l2_lt_5e-4": tensor_error(
            full_shared_cached["dx"], full_fused["dx"]
        )["rel_l2"]
        < 5e-4,
        "full_shared_cached_up_rel_l2_lt_1e-5": tensor_error(
            full_shared_cached["lora_up_grad"], full_ref["lora_up_grad"]
        )["rel_l2"]
        < 1e-5,
        "full_shared_cached_down_rel_l2_lt_5e-4": tensor_error(
            full_shared_cached["lora_down_grad"], full_ref["lora_down_grad"]
        )["rel_l2"]
        < 5e-4,
        "full_shared_cached_vs_recompute_dx_rel_l2_lt_5e-4": tensor_error(
            full_shared_cached["dx"], full_shared_recompute["dx"]
        )["rel_l2"]
        < 5e-4,
        "full_shared_cached_vs_recompute_down_rel_l2_lt_5e-4": tensor_error(
            full_shared_cached["lora_down_grad"], full_shared_recompute["lora_down_grad"]
        )["rel_l2"]
        < 5e-4,
        "decoded_dy_up_rel_l2_lt_5e-4": tensor_error(dense_dy_up, dy_up)["rel_l2"] < 5e-4,
        "full_shared_packed_dx_matches_fused_rel_l2_lt_5e-4": tensor_error(
            full_shared_packed["dx"], full_fused["dx"]
        )["rel_l2"]
        < 5e-4,
        "full_shared_packed_up_rel_l2_lt_1e-5": tensor_error(
            full_shared_packed["lora_up_grad"], full_ref["lora_up_grad"]
        )["rel_l2"]
        < 1e-5,
        "full_shared_packed_down_rel_l2_lt_5e-4": tensor_error(
            full_shared_packed["lora_down_grad"], full_ref["lora_down_grad"]
        )["rel_l2"]
        < 5e-4,
        "full_shared_packed_overlap_dx_matches_fused_rel_l2_lt_5e-4": tensor_error(
            full_shared_packed_overlap["dx"], full_fused["dx"]
        )["rel_l2"]
        < 5e-4,
        "full_shared_packed_overlap_up_rel_l2_lt_1e-5": tensor_error(
            full_shared_packed_overlap["lora_up_grad"], full_ref["lora_up_grad"]
        )["rel_l2"]
        < 1e-5,
        "full_shared_packed_overlap_down_rel_l2_lt_5e-4": tensor_error(
            full_shared_packed_overlap["lora_down_grad"], full_ref["lora_down_grad"]
        )["rel_l2"]
        < 5e-4,
        "dual_dy_up_rel_l2_lt_5e-4": tensor_error(dense_dy_up_dual, dy_up)["rel_l2"] < 5e-4,
        "dual_packed_matches_standard_rel_l2_lt_5e-4": tensor_error(
            packed_dy_up_dual.float(), packed_dy_up.float()
        )["rel_l2"]
        < 5e-4,
        "full_shared_dual_dx_matches_fused_rel_l2_lt_5e-4": tensor_error(
            full_shared_dual["dx"], full_fused["dx"]
        )["rel_l2"]
        < 5e-4,
        "full_shared_dual_up_rel_l2_lt_1e-5": tensor_error(
            full_shared_dual["lora_up_grad"], full_ref["lora_up_grad"]
        )["rel_l2"]
        < 1e-5,
        "full_shared_dual_down_rel_l2_lt_5e-4": tensor_error(
            full_shared_dual["lora_down_grad"], full_ref["lora_down_grad"]
        )["rel_l2"]
        < 5e-4,
        "pure_dx_all_finite": bool(torch.isfinite(dx_pure).all().item()),
        "hybrid_dx_all_finite": bool(torch.isfinite(dx_fused).all().item()),
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
            "pure_dx_vs_fp16": pure_dx_err,
            "hybrid_dx_vs_reference": hybrid_dx_ref_err,
            "hybrid_dx_fused_vs_unfused": hybrid_dx_internal_err,
            "qweight_bwd_cuda_vs_reference": tensor_error(qweight_bwd_cuda.float(), qweight_bwd_ref.float()),
            "full_dx_vs_reference": tensor_error(full_fused["dx"], full_ref["dx"]),
            "full_lora_up_grad_vs_reference": tensor_error(full_fused["lora_up_grad"], full_ref["lora_up_grad"]),
            "full_lora_down_grad_vs_reference": tensor_error(
                full_fused["lora_down_grad"], full_ref["lora_down_grad"]
            ),
            "full_dx_fused_vs_unfused": tensor_error(full_fused["dx"], full_unfused["dx"]),
            "full_shared_cached_dx_vs_reference": tensor_error(full_shared_cached["dx"], full_ref["dx"]),
            "full_shared_cached_up_vs_reference": tensor_error(
                full_shared_cached["lora_up_grad"], full_ref["lora_up_grad"]
            ),
            "full_shared_cached_down_vs_reference": tensor_error(
                full_shared_cached["lora_down_grad"], full_ref["lora_down_grad"]
            ),
            "full_shared_cached_dx_vs_fused": tensor_error(full_shared_cached["dx"], full_fused["dx"]),
            "full_shared_cached_up_vs_fused": tensor_error(
                full_shared_cached["lora_up_grad"], full_fused["lora_up_grad"]
            ),
            "full_shared_cached_down_vs_fused": tensor_error(
                full_shared_cached["lora_down_grad"], full_fused["lora_down_grad"]
            ),
            "decoded_dy_up_vs_reference": tensor_error(dense_dy_up, dy_up),
            "full_shared_packed_dx_vs_reference": tensor_error(full_shared_packed["dx"], full_ref["dx"]),
            "full_shared_packed_up_vs_reference": tensor_error(
                full_shared_packed["lora_up_grad"], full_ref["lora_up_grad"]
            ),
            "full_shared_packed_down_vs_reference": tensor_error(
                full_shared_packed["lora_down_grad"], full_ref["lora_down_grad"]
            ),
            "full_shared_packed_dx_vs_fused": tensor_error(full_shared_packed["dx"], full_fused["dx"]),
            "full_shared_packed_up_vs_fused": tensor_error(
                full_shared_packed["lora_up_grad"], full_fused["lora_up_grad"]
            ),
            "full_shared_packed_down_vs_fused": tensor_error(
                full_shared_packed["lora_down_grad"], full_fused["lora_down_grad"]
            ),
            "full_shared_packed_overlap_dx_vs_reference": tensor_error(
                full_shared_packed_overlap["dx"], full_ref["dx"]
            ),
            "full_shared_packed_overlap_up_vs_reference": tensor_error(
                full_shared_packed_overlap["lora_up_grad"], full_ref["lora_up_grad"]
            ),
            "full_shared_packed_overlap_down_vs_reference": tensor_error(
                full_shared_packed_overlap["lora_down_grad"], full_ref["lora_down_grad"]
            ),
            "full_shared_packed_overlap_dx_vs_fused": tensor_error(
                full_shared_packed_overlap["dx"], full_fused["dx"]
            ),
            "full_shared_packed_overlap_up_vs_fused": tensor_error(
                full_shared_packed_overlap["lora_up_grad"], full_fused["lora_up_grad"]
            ),
            "full_shared_packed_overlap_down_vs_fused": tensor_error(
                full_shared_packed_overlap["lora_down_grad"], full_fused["lora_down_grad"]
            ),
            "dual_dy_up_vs_reference": tensor_error(dense_dy_up_dual, dy_up),
            "dual_packed_vs_standard": tensor_error(packed_dy_up_dual.float(), packed_dy_up.float()),
            "full_shared_dual_dx_vs_reference": tensor_error(full_shared_dual["dx"], full_ref["dx"]),
            "full_shared_dual_up_vs_reference": tensor_error(
                full_shared_dual["lora_up_grad"], full_ref["lora_up_grad"]
            ),
            "full_shared_dual_down_vs_reference": tensor_error(
                full_shared_dual["lora_down_grad"], full_ref["lora_down_grad"]
            ),
            "full_shared_dual_dx_vs_fused": tensor_error(full_shared_dual["dx"], full_fused["dx"]),
            "full_shared_dual_up_vs_fused": tensor_error(
                full_shared_dual["lora_up_grad"], full_fused["lora_up_grad"]
            ),
            "full_shared_dual_down_vs_fused": tensor_error(
                full_shared_dual["lora_down_grad"], full_fused["lora_down_grad"]
            ),
        },
        "checks": checks,
        "all_passed": all(checks.values()),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"native_fp4_backward_validation_{stamp}.json")
    latest_path = os.path.join(args.results_dir, "latest_native_fp4_backward_validation.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved validation to: {out_path}")


if __name__ == "__main__":
    main()
