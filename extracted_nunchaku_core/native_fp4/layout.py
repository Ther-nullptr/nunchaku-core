from __future__ import annotations

import torch


WARP_N = 128
WARP_K = 64
FP4_GROUP_SIZE = 16
FP4_QMAX = 6.0

NUM_N_LANES = 8
NUM_K_LANES = 4
N_PACK_SIZE = 2
K_PACK_SIZE = 2
REG_N = 1
REG_K = 8
NUM_N_PACKS = WARP_N // (N_PACK_SIZE * NUM_N_LANES * REG_N)
NUM_K_PACKS = 1

FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _require_2d(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(tensor.shape)}")


def _require_fp4_weight_shape(name: str, rows: int, cols: int) -> None:
    if rows % WARP_N != 0:
        raise ValueError(f"{name} rows must be divisible by {WARP_N}, got {rows}")
    if cols % (WARP_K * 2) != 0:
        raise ValueError(f"{name} cols must be divisible by {WARP_K * 2}, got {cols}")


def _fp4_scale_row_order(device: torch.device) -> torch.Tensor:
    lane = torch.arange(32, device=device, dtype=torch.long)
    pack = torch.arange(4, device=device, dtype=torch.long)
    return ((lane % 4).unsqueeze(1) * 8 + (lane // 4).unsqueeze(1) + pack.unsqueeze(0) * 32).reshape(-1)


def unpack_fp4_weight_codes(qweight: torch.Tensor) -> torch.Tensor:
    _require_2d("qweight", qweight)
    if qweight.dtype not in (torch.uint8, torch.int8):
        raise ValueError(f"qweight must be uint8/int8, got {qweight.dtype}")

    n, packed_k = qweight.shape
    k = packed_k * 2
    _require_fp4_weight_shape("qweight", n, k)

    words = qweight.contiguous().view(torch.uint8).view(torch.int32).view(
        n // WARP_N,
        k // WARP_K,
        NUM_K_PACKS,
        NUM_N_PACKS,
        NUM_N_LANES,
        NUM_K_LANES,
        N_PACK_SIZE,
        K_PACK_SIZE,
        REG_N,
    )
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    codes = torch.bitwise_and(torch.bitwise_right_shift(words.unsqueeze(-1), shifts), 0xF).to(torch.uint8)
    codes = codes.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
    return codes.view(n, k)


def pack_fp4_weight_codes(codes: torch.Tensor) -> torch.Tensor:
    _require_2d("codes", codes)
    if codes.dtype != torch.uint8:
        codes = codes.to(torch.uint8)

    n, k = codes.shape
    _require_fp4_weight_shape("codes", n, k)

    structured = codes.contiguous().view(
        n // WARP_N,
        NUM_N_PACKS,
        N_PACK_SIZE,
        NUM_N_LANES,
        REG_N,
        k // WARP_K,
        NUM_K_PACKS,
        K_PACK_SIZE,
        NUM_K_LANES,
        REG_K,
    )
    packed = structured.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous().to(torch.int32)
    shifts = torch.arange(0, 32, 4, device=codes.device, dtype=torch.int32)
    words = torch.sum(torch.bitwise_left_shift(torch.bitwise_and(packed, 0xF), shifts), dim=-1, dtype=torch.int32)
    return words.view(torch.uint8).view(n, k // 2)


def unpack_fp4_weight_scales(packed_wscales: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    _require_2d("packed_wscales", packed_wscales)
    if packed_wscales.dtype != torch.float8_e4m3fn:
        raise ValueError(f"packed_wscales must be float8_e4m3fn, got {packed_wscales.dtype}")

    k_groups = in_features // FP4_GROUP_SIZE
    if packed_wscales.shape != (k_groups, out_features):
        raise ValueError(
            f"packed_wscales shape mismatch: expected {(k_groups, out_features)}, got {tuple(packed_wscales.shape)}"
        )
    _require_fp4_weight_shape("packed_wscales", out_features, in_features)

    n_blocks = out_features // WARP_N
    k_tiles = in_features // WARP_K
    row_order = _fp4_scale_row_order(packed_wscales.device)

    raw = packed_wscales.contiguous().view(torch.uint8).view(n_blocks, k_tiles, 32, 4, 4)
    dense_bytes = torch.empty((n_blocks, k_tiles, WARP_N, 4), dtype=torch.uint8, device=packed_wscales.device)
    dense_bytes[:, :, row_order, :] = raw.reshape(n_blocks, k_tiles, WARP_N, 4)
    logical_bytes = dense_bytes.permute(0, 2, 1, 3).contiguous().view(out_features, k_groups)

    logical_fp8 = torch.empty(logical_bytes.shape, dtype=torch.float8_e4m3fn, device=packed_wscales.device)
    logical_fp8.view(torch.uint8).copy_(logical_bytes)
    return logical_fp8.to(torch.float32)


def pack_fp4_weight_scales(logical_scales: torch.Tensor) -> torch.Tensor:
    _require_2d("logical_scales", logical_scales)
    if logical_scales.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        logical_scales = logical_scales.float()

    out_features, k_groups = logical_scales.shape
    in_features = k_groups * FP4_GROUP_SIZE
    _require_fp4_weight_shape("logical_scales", out_features, in_features)

    n_blocks = out_features // WARP_N
    k_tiles = in_features // WARP_K
    row_order = _fp4_scale_row_order(logical_scales.device)

    logical_fp8 = logical_scales.clamp(min=0, max=448).to(torch.float8_e4m3fn).contiguous()
    logical_bytes = logical_fp8.view(torch.uint8).view(n_blocks, WARP_N, k_tiles, 4).permute(0, 2, 1, 3).contiguous()
    raw = logical_bytes[:, :, row_order, :].view(n_blocks, k_tiles, 32, 4, 4)

    packed = torch.empty((k_groups, out_features), dtype=torch.float8_e4m3fn, device=logical_scales.device)
    packed.view(torch.uint8).copy_(raw.reshape_as(packed.view(torch.uint8)))
    return packed


def compute_fp4_logical_scales(weight: torch.Tensor) -> torch.Tensor:
    _require_2d("weight", weight)
    out_features, in_features = weight.shape
    if in_features % FP4_GROUP_SIZE != 0:
        raise ValueError(f"weight cols must be divisible by {FP4_GROUP_SIZE}, got {in_features}")

    scale = weight.abs().reshape(out_features, in_features // FP4_GROUP_SIZE, FP4_GROUP_SIZE).amax(dim=2)
    return (scale.float() / FP4_QMAX).clamp(max=448.0)


def quantize_fp4_codes_from_dense(weight: torch.Tensor, logical_scales: torch.Tensor) -> torch.Tensor:
    _require_2d("weight", weight)
    _require_2d("logical_scales", logical_scales)
    if logical_scales.shape != (weight.shape[0], weight.shape[1] // FP4_GROUP_SIZE):
        raise ValueError(
            "logical_scales shape mismatch: "
            f"expected {(weight.shape[0], weight.shape[1] // FP4_GROUP_SIZE)}, got {tuple(logical_scales.shape)}"
        )

    scale = logical_scales.to(weight.dtype)
    expanded_scale = scale.repeat_interleave(FP4_GROUP_SIZE, dim=1)
    safe_scale = torch.where(expanded_scale > 0, expanded_scale, torch.ones_like(expanded_scale))
    scaled = weight / safe_scale
    magnitude = scaled.abs()

    codes = (
        (magnitude >= 0.25).to(torch.uint8)
        + (magnitude >= 0.75).to(torch.uint8)
        + (magnitude >= 1.25).to(torch.uint8)
        + (magnitude >= 1.75).to(torch.uint8)
        + (magnitude >= 2.50).to(torch.uint8)
        + (magnitude >= 3.50).to(torch.uint8)
        + (magnitude >= 5.00).to(torch.uint8)
    )
    codes = torch.where(expanded_scale > 0, codes, torch.zeros_like(codes))
    return codes | ((scaled < 0).to(torch.uint8) << 3)


def dequantize_fp4_weight(
    qweight: torch.Tensor,
    packed_wscales: torch.Tensor,
    out_features: int,
    in_features: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = unpack_fp4_weight_codes(qweight)
    logical_scales = unpack_fp4_weight_scales(packed_wscales, out_features=out_features, in_features=in_features)
    lut = FP4_LUT.to(device=qweight.device, dtype=dtype)
    weight = lut[codes.long()] * logical_scales.to(dtype).repeat_interleave(FP4_GROUP_SIZE, dim=1)
    return weight, logical_scales


def build_backward_scales_from_forward_quant(
    qweight: torch.Tensor,
    packed_wscales: torch.Tensor,
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_hat, _ = dequantize_fp4_weight(
        qweight=qweight,
        packed_wscales=packed_wscales,
        out_features=out_features,
        in_features=in_features,
        dtype=torch.float16,
    )
    logical_scales_t = compute_fp4_logical_scales(weight_hat.t())
    return logical_scales_t, pack_fp4_weight_scales(logical_scales_t)


def repack_fp4_weight_for_backward(
    qweight: torch.Tensor,
    packed_wscales: torch.Tensor,
    out_features: int,
    in_features: int,
    logical_scales_t: torch.Tensor,
) -> torch.Tensor:
    weight_hat, _ = dequantize_fp4_weight(
        qweight=qweight,
        packed_wscales=packed_wscales,
        out_features=out_features,
        in_features=in_features,
        dtype=torch.float16,
    )
    codes_t = quantize_fp4_codes_from_dense(weight_hat.t().contiguous(), logical_scales_t.to(weight_hat.dtype))
    return pack_fp4_weight_codes(codes_t)
