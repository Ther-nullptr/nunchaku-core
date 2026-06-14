from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import os
import sys
from typing import Literal

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_QMAX = 448.0
FP8Backend = Literal["auto", "torch", "deep_gemm"]


@dataclass(frozen=True)
class DeepGEMMStatus:
    importable: bool
    module_path: str | None
    version: str | None
    cuda_home: str | None
    cuobjdump_path: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")


def _require_supported_dtype(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{name} dtype must be float16 or bfloat16, got {tensor.dtype}")


def _as_scalar_scale(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.tensor(float(scale.item()), device=device, dtype=torch.float32)


def _align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    x = x.abs().float().clamp_min(1e-30)
    return torch.pow(torch.tensor(2.0, device=x.device, dtype=torch.float32), torch.ceil(torch.log2(x)))


def _deep_gemm_module(deep_gemm_path: str | None = None):
    path = deep_gemm_path or os.environ.get("NUNCHAKU_DEEPGEMM_PATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("deep_gemm")


def _cuda_home_for_deep_gemm() -> str | None:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        return cuda_home
    nvcc = next((p for p in os.environ.get("PATH", "").split(os.pathsep) if os.path.exists(os.path.join(p, "nvcc"))), None)
    return os.path.dirname(nvcc) if nvcc else None


def deep_gemm_status(deep_gemm_path: str | None = None) -> DeepGEMMStatus:
    cuda_home = _cuda_home_for_deep_gemm()
    cuobjdump_path = os.path.join(cuda_home, "bin", "cuobjdump") if cuda_home else None
    if cuobjdump_path and not os.path.exists(cuobjdump_path):
        cuobjdump_path = None

    try:
        module = _deep_gemm_module(deep_gemm_path)
    except Exception as exc:  # pragma: no cover - depends on optional external package
        return DeepGEMMStatus(
            importable=False,
            module_path=None,
            version=None,
            cuda_home=cuda_home,
            cuobjdump_path=cuobjdump_path,
            error=f"{type(exc).__name__}: {exc}",
        )

    has_fp8_gemm = hasattr(module, "fp8_gemm_nt")
    error = None
    if not has_fp8_gemm:
        error = "deep_gemm.fp8_gemm_nt is missing"
    elif cuobjdump_path is None:
        error = "CUDA_HOME/bin/cuobjdump is missing; DeepGEMM JIT cannot discover cubin symbols"

    return DeepGEMMStatus(
        importable=has_fp8_gemm,
        module_path=getattr(module, "__file__", None),
        version=getattr(module, "__version__", None),
        cuda_home=cuda_home,
        cuobjdump_path=cuobjdump_path,
        error=error,
    )


def quantize_fp8_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(x.shape)}")
    _require_cuda_tensor("x", x)
    _require_supported_dtype("x", x)

    scale = x.abs().amax().float().clamp_min(1e-4) / FP8_QMAX
    q = (x / scale.to(x.dtype)).to(FP8_DTYPE)
    return q.contiguous(), _as_scalar_scale(scale, x.device)


def quantize_fp8_per_token(
    x: torch.Tensor,
    *,
    gran_k: int = 128,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepGEMM-compatible 1D activation quantization: one scale per row/K block."""

    if x.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(x.shape)}")
    _require_cuda_tensor("x", x)
    _require_supported_dtype("x", x)

    m, k = x.shape
    k_pad = _align(k, gran_k)
    x_pad = torch.zeros((m, k_pad), dtype=x.dtype, device=x.device)
    x_pad[:, :k] = x
    x_view = x_pad.view(m, k_pad // gran_k, gran_k)
    scale = x_view.abs().float().amax(dim=2).clamp_min(1e-4) / FP8_QMAX
    scale = _ceil_to_ue8m0(scale) if use_ue8m0 else scale
    q = (x_view * (1.0 / scale.unsqueeze(2)).to(x.dtype)).to(FP8_DTYPE)
    return q.view(m, k_pad)[:, :k].contiguous(), scale.contiguous()


def quantize_fp8_per_block(
    x: torch.Tensor,
    *,
    gran_mn: int = 128,
    gran_k: int = 128,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepGEMM-compatible 2D weight quantization: one scale per MN/K block."""

    if x.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(x.shape)}")
    _require_cuda_tensor("x", x)
    _require_supported_dtype("x", x)

    mn, k = x.shape
    mn_pad = _align(mn, gran_mn)
    k_pad = _align(k, gran_k)
    x_pad = torch.zeros((mn_pad, k_pad), dtype=x.dtype, device=x.device)
    x_pad[:mn, :k] = x
    x_view = x_pad.view(mn_pad // gran_mn, gran_mn, k_pad // gran_k, gran_k)
    scale = x_view.abs().float().amax(dim=(1, 3)).clamp_min(1e-4) / FP8_QMAX
    scale = _ceil_to_ue8m0(scale) if use_ue8m0 else scale
    q = (x_view * (1.0 / scale[:, None, :, None]).to(x.dtype)).to(FP8_DTYPE)
    return q.view(mn_pad, k_pad)[:mn, :k].contiguous(), scale.contiguous()


class NunchakuFP8GemmOp(torch.nn.Module):
    """FP8 GEMM wrapper with a torch fallback and an optional DeepGEMM backend."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        backend: FP8Backend = "auto",
        deep_gemm_path: str | None = None,
        deep_gemm_use_ue8m0: bool = True,
        deep_gemm_gran_k: int = 128,
    ):
        super().__init__()
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("torch._scaled_mm is required for native_fp8")
        _require_cuda_tensor("weight", weight)
        _require_supported_dtype("weight", weight)
        if weight.dim() != 2:
            raise ValueError("weight should have shape [out_features, in_features]")
        if backend not in ("auto", "torch", "deep_gemm"):
            raise ValueError(f"Unknown FP8 backend: {backend}")
        if backend == "deep_gemm" and weight.dtype != torch.bfloat16:
            raise ValueError("DeepGEMM FP8 output path currently requires BF16 weights/outputs")

        self.out_features, self.in_features = weight.shape
        self.compute_dtype = weight.dtype
        self.requested_backend = backend
        self.selected_backend = "torch"
        self.last_backend = "none"
        self.deep_gemm_path = deep_gemm_path or ""
        self.deep_gemm_use_ue8m0 = bool(deep_gemm_use_ue8m0)
        self.deep_gemm_gran_k = int(deep_gemm_gran_k)
        self.deep_gemm_last_error: str | None = None
        self._deep_gemm = None
        self._deep_gemm_status = deep_gemm_status(deep_gemm_path)

        need_torch_fallback = backend in ("auto", "torch")
        use_deep_gemm = backend in ("auto", "deep_gemm") and weight.dtype == torch.bfloat16
        deep_gemm_ready = self._deep_gemm_status.importable and self._deep_gemm_status.cuobjdump_path is not None
        if use_deep_gemm and deep_gemm_ready:
            self._deep_gemm = _deep_gemm_module(deep_gemm_path)
            qweight_dg, scale_w_dg = quantize_fp8_per_block(
                weight.contiguous(),
                gran_mn=128,
                gran_k=self.deep_gemm_gran_k,
                use_ue8m0=self.deep_gemm_use_ue8m0,
            )
            self.register_buffer("qweight_deep_gemm", qweight_dg, persistent=True)
            self.register_buffer("scale_w_deep_gemm", scale_w_dg, persistent=True)
            self.selected_backend = "deep_gemm"
        else:
            if backend == "deep_gemm":
                raise RuntimeError(f"DeepGEMM backend is unavailable: {self._deep_gemm_status.error}")
            self.register_buffer("qweight_deep_gemm", None, persistent=True)
            self.register_buffer("scale_w_deep_gemm", None, persistent=True)

        if need_torch_fallback or self.selected_backend == "torch":
            qweight, scale_w = quantize_fp8_per_tensor(weight.contiguous())
        else:
            qweight, scale_w = None, None
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

    def quantize_input_deep_gemm(self, x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_fp8_per_token(
            x2d,
            gran_k=self.deep_gemm_gran_k,
            use_ue8m0=self.deep_gemm_use_ue8m0,
        )

    def forward_prequantized(self, qx: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        if self.qweight is None or self.scale_w is None:
            raise RuntimeError("Torch FP8 fallback buffers are not available for this operator")
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

    def forward_prequantized_deep_gemm(self, qx: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        if self._deep_gemm is None:
            raise RuntimeError("DeepGEMM backend is not available")
        if qx.dtype != FP8_DTYPE:
            raise ValueError(f"Expected qx dtype {FP8_DTYPE}, got {qx.dtype}")
        out = torch.empty((qx.shape[0], self.out_features), dtype=torch.bfloat16, device=qx.device)
        self._deep_gemm.fp8_gemm_nt(
            (qx, scale_x),
            (self.qweight_deep_gemm, self.scale_w_deep_gemm),
            out,
            disable_ue8m0_cast=not self.deep_gemm_use_ue8m0,
        )
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.to(self.compute_dtype)

    def backend_info(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "last_backend": self.last_backend,
            "deep_gemm_use_ue8m0": self.deep_gemm_use_ue8m0,
            "deep_gemm_gran_k": self.deep_gemm_gran_k,
            "deep_gemm_last_error": self.deep_gemm_last_error,
            "deep_gemm_status": self._deep_gemm_status.to_dict(),
        }

    def _forward_torch(self, x2d: torch.Tensor) -> torch.Tensor:
        qx, scale_x = self.quantize_input(x2d)
        out = self.forward_prequantized(qx, scale_x)
        self.last_backend = "torch"
        return out

    def _forward_deep_gemm(self, x2d: torch.Tensor) -> torch.Tensor:
        qx, scale_x = self.quantize_input_deep_gemm(x2d)
        out = self.forward_prequantized_deep_gemm(qx, scale_x)
        self.last_backend = "deep_gemm"
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input last dim = {self.in_features}, got {x.shape[-1]}")
        _require_cuda_tensor("x", x)
        _require_supported_dtype("x", x)

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features).contiguous()
        if self.selected_backend == "deep_gemm":
            try:
                out = self._forward_deep_gemm(x2d)
            except Exception as exc:
                self.deep_gemm_last_error = f"{type(exc).__name__}: {exc}"
                if self.requested_backend == "deep_gemm":
                    raise
                self.selected_backend = "torch"
                out = self._forward_torch(x2d)
        else:
            out = self._forward_torch(x2d)
        return out.reshape(*orig_shape[:-1], self.out_features)
