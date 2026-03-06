from __future__ import annotations

import torch

from . import _int4_cuda


def quantize_int4_packed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise symmetric INT4 quantization on CUDA.

    Returns
    -------
    packed : torch.Tensor
        uint8 tensor with shape [M, K // 2].
    scales : torch.Tensor
        float32 tensor with shape [M].
    """

    if x.dim() != 2:
        raise ValueError(f"Expected 2D input [M, K], got shape={tuple(x.shape)}")
    return _int4_cuda.quantize_int4_packed(x.contiguous())


def unpack_int4_packed(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed INT4 values to int8 matrix with shape [M, K]."""

    if packed.dim() != 2:
        raise ValueError(f"Expected 2D packed tensor [M, K/2], got shape={tuple(packed.shape)}")
    return _int4_cuda.unpack_int4_packed(packed.contiguous())
