from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_QMAX = 448.0


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")


def _require_supported_dtype(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{name} dtype must be float16 or bfloat16, got {tensor.dtype}")


def _as_scalar_scale(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.tensor(float(scale.item()), device=device, dtype=torch.float32)


def quantize_fp8_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(x.shape)}")
    _require_cuda_tensor("x", x)
    _require_supported_dtype("x", x)

    scale = x.abs().amax().float().clamp_min(1e-4) / FP8_QMAX
    q = (x / scale.to(x.dtype)).to(FP8_DTYPE)
    return q.contiguous(), _as_scalar_scale(scale, x.device)


class NunchakuFP8GemmOp(torch.nn.Module):
    """Minimal FP8 GEMM wrapper backed by torch._scaled_mm on CUDA."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("torch._scaled_mm is required for native_fp8")
        _require_cuda_tensor("weight", weight)
        _require_supported_dtype("weight", weight)
        if weight.dim() != 2:
            raise ValueError("weight should have shape [out_features, in_features]")

        self.out_features, self.in_features = weight.shape
        self.compute_dtype = weight.dtype

        qweight, scale_w = quantize_fp8_per_tensor(weight.contiguous())
        self.register_buffer("qweight", qweight, persistent=True)
        self.register_buffer("scale_w", scale_w, persistent=True)

        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            _require_cuda_tensor("bias", bias)
            if bias.dim() != 1 or bias.numel() != self.out_features:
                raise ValueError("bias should have shape [out_features]")
            self.register_buffer("bias", bias.to(weight.dtype).contiguous(), persistent=True)

    def quantize_input(self, x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_fp8_per_tensor(x2d)

    def forward_prequantized(self, qx: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        if qx.dtype != FP8_DTYPE:
            raise ValueError(f"Expected qx dtype {FP8_DTYPE}, got {qx.dtype}")
        out = torch._scaled_mm(
            qx,
            self.qweight.t(),
            scale_a=scale_x,
            scale_b=self.scale_w,
            out_dtype=self.compute_dtype,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim = {self.in_features}, got {x.shape[-1]}")
        _require_cuda_tensor("x", x)
        _require_supported_dtype("x", x)

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features).contiguous()
        qx, scale_x = self.quantize_input(x2d)
        out = self.forward_prequantized(qx, scale_x)
        return out.reshape(*orig_shape[:-1], self.out_features)
