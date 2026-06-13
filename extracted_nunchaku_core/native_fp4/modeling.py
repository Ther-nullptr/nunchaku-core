from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

import torch

from .training import (
    DEFAULT_OVERLAP_LORA_GRAD_MIN_ROWS,
    FrozenResidualInitMode,
    LoRAInitMode,
    NunchakuFP4LoRALinear,
    ResidualSVDMethod,
)

FP4LoRAFinetuneMode = Literal["accuracy", "balanced", "throughput", "memory_saving"]
DEFAULT_FP4_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_FP4_LORA_EXCLUDE_MODULES = ("lm_head",)


@dataclass(frozen=True)
class FP4LoRAConfig:
    rank: int = 32
    lora_alpha: float | None = None
    lowrank_dtype: torch.dtype = torch.bfloat16
    init: LoRAInitMode = "zero"
    frozen_residual_rank: int = 0
    frozen_residual_init: FrozenResidualInitMode = "none"
    residual_svd_method: ResidualSVDMethod = "full_svd"
    residual_svd_lowrank_oversample: int = 8
    residual_svd_lowrank_niter: int = 2
    train_bias: bool = False
    cache_lora_act: bool = True
    activation_checkpoint: bool = False
    fuse_lowrank_forward: bool = False
    fuse_lora_dx: bool = False
    fuse_frozen_residual_dx: bool = False
    cache_fused_lora_dx: bool = False
    reuse_fused_dy_up_for_d_lora_down: bool = False
    overlap_lora_grad: bool = False
    overlap_lora_grad_min_rows: int = DEFAULT_OVERLAP_LORA_GRAD_MIN_ROWS
    fp4_activation_cache_d_lora_down: bool = False


@dataclass
class FP4LoRAPrepareResult:
    """Artifacts returned by ``prepare_fp4_lora_finetuning``."""

    model: torch.nn.Module
    config: FP4LoRAConfig
    replaced_modules: list[str]
    trainable_names: list[str]
    trainable_param_count: int
    optimizer_param_groups: list[dict[str, Any]]
    refreshed_cache_count: int
    target_modules: tuple[str, ...]
    exclude_modules: tuple[str, ...]

    def register_cache_refresh_hook(self, optimizer: torch.optim.Optimizer) -> FP4LoRACacheRefreshHook:
        return register_fp4_lora_cache_refresh_hook(optimizer, self.model)


def fp4_lora_finetune_config(
    *,
    mode: FP4LoRAFinetuneMode = "balanced",
    rank: int = 32,
    lora_alpha: float | None = None,
    dtype: torch.dtype = torch.bfloat16,
    lowrank_dtype: torch.dtype | None = None,
    use_frozen_residual: bool = True,
    frozen_residual_rank: int | None = None,
    residual_svd_method: ResidualSVDMethod | None = None,
    residual_svd_lowrank_oversample: int = 8,
    residual_svd_lowrank_niter: int = 2,
    train_bias: bool = False,
    cache_lora_act: bool = True,
    activation_checkpoint: bool = False,
    overlap_lora_grad_min_rows: int = DEFAULT_OVERLAP_LORA_GRAD_MIN_ROWS,
) -> FP4LoRAConfig:
    """Return a recommended FP4 LoRA fine-tuning config.

    Modes:
      - ``accuracy``: exact BF16/FP16 LoRA gradients and dense LoRA dX.
      - ``balanced``: exact LoRA gradients with fused cached LoRA dX and auto-gated overlap.
      - ``throughput``: balanced plus fused low-rank forward; FP16 also fuses frozen residual dX.
      - ``memory_saving``: balanced dX, but stores FP4 activation cache for approximate dA.

    The returned config follows the SVDQuant-inspired training policy from the
    project notes: freeze the residual-SVD quantization compensation and train a
    separate zero-init task LoRA branch.
    """

    if mode not in ("accuracy", "balanced", "throughput", "memory_saving"):
        raise ValueError("mode must be one of: accuracy, balanced, throughput, memory_saving")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("dtype must be torch.float16 or torch.bfloat16")
    if lowrank_dtype is None:
        lowrank_dtype = dtype
    if lowrank_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("lowrank_dtype must be torch.float16 or torch.bfloat16")
    if overlap_lora_grad_min_rows < 0:
        raise ValueError("overlap_lora_grad_min_rows must be non-negative")

    if frozen_residual_rank is None:
        effective_frozen_residual_rank = int(rank) if use_frozen_residual else 0
    else:
        effective_frozen_residual_rank = int(frozen_residual_rank)
    if not use_frozen_residual:
        effective_frozen_residual_rank = 0
    if effective_frozen_residual_rank < 0:
        raise ValueError("frozen_residual_rank must be non-negative")
    frozen_residual_init: FrozenResidualInitMode = (
        "residual_svd" if effective_frozen_residual_rank > 0 else "none"
    )

    if residual_svd_method is None:
        residual_svd_method = "full_svd" if mode == "accuracy" else "svd_lowrank"

    fuse_lora_dx = mode != "accuracy"
    cache_fused_lora_dx = fuse_lora_dx
    overlap_lora_grad = mode in ("balanced", "throughput")
    fp4_activation_cache_d_lora_down = mode == "memory_saving"
    if fp4_activation_cache_d_lora_down and not cache_lora_act:
        raise ValueError("mode='memory_saving' requires cache_lora_act=True")
    if fp4_activation_cache_d_lora_down:
        # FP4 activation-cache dA is an approximate gradient path and cannot
        # currently share the exact-overlap schedule.
        overlap_lora_grad = False
    fuse_lowrank_forward = mode == "throughput"
    fuse_frozen_residual_dx = (
        mode == "throughput"
        and effective_frozen_residual_rank > 0
        and dtype == torch.float16
        and lowrank_dtype == torch.float16
    )
    if fuse_frozen_residual_dx:
        # The exact overlap implementation keeps frozen residual dX on a dense
        # side stream. When residual dX is fused into the epilogue, use the
        # sequential cached fused-dX path.
        overlap_lora_grad = False

    return FP4LoRAConfig(
        rank=rank,
        lora_alpha=lora_alpha,
        lowrank_dtype=lowrank_dtype,
        init="zero",
        frozen_residual_rank=effective_frozen_residual_rank,
        frozen_residual_init=frozen_residual_init,
        residual_svd_method=residual_svd_method,
        residual_svd_lowrank_oversample=residual_svd_lowrank_oversample,
        residual_svd_lowrank_niter=residual_svd_lowrank_niter,
        train_bias=train_bias,
        cache_lora_act=cache_lora_act,
        activation_checkpoint=activation_checkpoint,
        fuse_lowrank_forward=fuse_lowrank_forward,
        fuse_lora_dx=fuse_lora_dx,
        fuse_frozen_residual_dx=fuse_frozen_residual_dx,
        cache_fused_lora_dx=cache_fused_lora_dx,
        reuse_fused_dy_up_for_d_lora_down=False,
        overlap_lora_grad=overlap_lora_grad,
        overlap_lora_grad_min_rows=overlap_lora_grad_min_rows,
        fp4_activation_cache_d_lora_down=fp4_activation_cache_d_lora_down,
    )


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return int(value)
    return ((int(value) + multiple - 1) // multiple) * multiple


def fp4_lora_config_overrides_from_outlier_report(
    report: Mapping[str, Any] | str,
    base_config: FP4LoRAConfig,
    *,
    rank_field: str = "suggested_rank",
    rank_multiple: int = 16,
    min_rank: int | None = None,
    max_rank: int | None = None,
    force_init: LoRAInitMode | None = None,
    disable_fuse_frozen_residual_dx: bool = False,
) -> dict[str, FP4LoRAConfig]:
    """Build per-module FP4 LoRA overrides from an outlier diagnostic report.

    The expected input is the JSON emitted by
    ``analyze_fp4_lora_activation_grad_outliers.py``. Only
    ``summary.rank_bump_candidates`` is consumed; unsupported fields are ignored.
    """

    if isinstance(report, str):
        with open(report, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    else:
        report_data = report

    candidates = report_data.get("summary", {}).get("rank_bump_candidates", [])
    overrides: dict[str, FP4LoRAConfig] = {}
    for item in candidates:
        module_name = item.get("module")
        if not module_name:
            continue
        rank = int(item.get(rank_field, base_config.rank))
        if min_rank is not None:
            rank = max(rank, int(min_rank))
        if max_rank is not None:
            rank = min(rank, int(max_rank))
        rank = _ceil_to_multiple(rank, rank_multiple)
        overrides[str(module_name)] = replace(
            base_config,
            rank=rank,
            init=base_config.init if force_init is None else force_init,
            fuse_frozen_residual_dx=False if disable_fuse_frozen_residual_dx else base_config.fuse_frozen_residual_dx,
        )
    return overrides


def _as_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(values)


def _name_matches(full_name: str, child_name: str, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return True
    for pattern in patterns:
        if full_name == pattern or child_name == pattern or full_name.endswith(f".{pattern}"):
            return True
    return False


def _name_excluded(full_name: str, child_name: str, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    return _name_matches(full_name, child_name, patterns)


def _select_config(
    base_config: FP4LoRAConfig,
    full_name: str,
    child_name: str,
    overrides: Mapping[str, FP4LoRAConfig] | None,
) -> FP4LoRAConfig:
    if not overrides:
        return base_config
    for pattern, override in overrides.items():
        if _name_matches(full_name, child_name, (pattern,)):
            return override
    return base_config


def convert_linear_to_fp4_lora(
    module: torch.nn.Module,
    config: FP4LoRAConfig | None = None,
    *,
    target_modules: Iterable[str] | None = None,
    exclude_modules: Iterable[str] | None = None,
    config_overrides: Mapping[str, FP4LoRAConfig] | None = None,
    inplace: bool = True,
) -> tuple[torch.nn.Module, list[str]]:
    """Replace selected CUDA Linear modules with NunchakuFP4LoRALinear.

    Matching uses the full module path, the child name, or a full-path suffix.
    For example, target_modules=("q_proj", "down_proj") matches
    "layers.0.self_attn.q_proj" and "layers.0.mlp.down_proj".
    ``config_overrides`` uses the same matching rules and the first matching
    override wins, which makes sensitive-layer policies deterministic.
    """

    cfg = FP4LoRAConfig() if config is None else config
    targets = _as_tuple(target_modules)
    excludes = _as_tuple(exclude_modules)
    root = module if inplace else copy.deepcopy(module)
    replaced: list[str] = []

    def visit(parent: torch.nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, torch.nn.Linear):
                if _name_matches(full_name, child_name, targets) and not _name_excluded(full_name, child_name, excludes):
                    child_cfg = _select_config(cfg, full_name, child_name, config_overrides)
                    if not child.weight.is_cuda:
                        raise ValueError(f"Linear module {full_name!r} must be on CUDA before FP4 LoRA conversion")
                    if child.weight.dtype not in (torch.float16, torch.bfloat16):
                        raise ValueError(f"Linear module {full_name!r} weight must be float16 or bfloat16")
                    fp4_lora = NunchakuFP4LoRALinear.from_linear(
                        child,
                        rank=child_cfg.rank,
                        lora_alpha=child_cfg.lora_alpha,
                        lowrank_dtype=child_cfg.lowrank_dtype,
                        init=child_cfg.init,
                        frozen_residual_rank=child_cfg.frozen_residual_rank,
                        frozen_residual_init=child_cfg.frozen_residual_init,
                        residual_svd_method=child_cfg.residual_svd_method,
                        residual_svd_lowrank_oversample=child_cfg.residual_svd_lowrank_oversample,
                        residual_svd_lowrank_niter=child_cfg.residual_svd_lowrank_niter,
                        train_bias=child_cfg.train_bias,
                        cache_lora_act=child_cfg.cache_lora_act,
                        activation_checkpoint=child_cfg.activation_checkpoint,
                        fuse_lowrank_forward=child_cfg.fuse_lowrank_forward,
                        fuse_lora_dx=child_cfg.fuse_lora_dx,
                        fuse_frozen_residual_dx=child_cfg.fuse_frozen_residual_dx,
                        cache_fused_lora_dx=child_cfg.cache_fused_lora_dx,
                        reuse_fused_dy_up_for_d_lora_down=child_cfg.reuse_fused_dy_up_for_d_lora_down,
                        overlap_lora_grad=child_cfg.overlap_lora_grad,
                        overlap_lora_grad_min_rows=child_cfg.overlap_lora_grad_min_rows,
                        fp4_activation_cache_d_lora_down=child_cfg.fp4_activation_cache_d_lora_down,
                    )
                    setattr(parent, child_name, fp4_lora)
                    replaced.append(full_name)
                    continue
            visit(child, full_name)

    visit(root, "")
    return root, replaced


def iter_fp4_lora_modules(module: torch.nn.Module) -> Iterator[tuple[str, NunchakuFP4LoRALinear]]:
    for name, child in module.named_modules():
        if isinstance(child, NunchakuFP4LoRALinear):
            yield name, child


def refresh_fused_lora_dx_caches(module: torch.nn.Module) -> int:
    count = 0
    for _, child in iter_fp4_lora_modules(module):
        if child.fuse_lora_dx and child.cache_fused_lora_dx:
            child.refresh_fused_lora_dx_cache()
            count += 1
    return count


def clear_fused_lora_dx_caches(module: torch.nn.Module) -> int:
    count = 0
    for _, child in iter_fp4_lora_modules(module):
        child.clear_fused_lora_dx_cache()
        count += 1
    return count


def iter_fp4_lora_named_parameters(
    module: torch.nn.Module,
    *,
    train_bias: bool = False,
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    """Yield LoRA-only trainable parameters from converted FP4 LoRA modules."""

    for module_name, child in iter_fp4_lora_modules(module):
        prefix = f"{module_name}." if module_name else ""
        yield f"{prefix}lora_down", child.lora_down
        yield f"{prefix}lora_up", child.lora_up
        bias = getattr(child, "bias", None)
        if train_bias and isinstance(bias, torch.nn.Parameter):
            yield f"{prefix}bias", bias


def fp4_lora_parameter_groups(
    module: torch.nn.Module,
    *,
    train_bias: bool = False,
    lora_weight_decay: float = 0.0,
    bias_weight_decay: float = 0.0,
    lr: float | None = None,
) -> list[dict[str, Any]]:
    """Return optimizer parameter groups containing only FP4 LoRA adapter params."""

    lora_params: list[torch.nn.Parameter] = []
    bias_params: list[torch.nn.Parameter] = []
    for name, param in iter_fp4_lora_named_parameters(module, train_bias=train_bias):
        if name.endswith(".bias") or name == "bias":
            bias_params.append(param)
        else:
            lora_params.append(param)

    groups: list[dict[str, Any]] = []
    if lora_params:
        group: dict[str, Any] = {"params": lora_params, "weight_decay": float(lora_weight_decay)}
        if lr is not None:
            group["lr"] = float(lr)
        groups.append(group)
    if bias_params:
        group = {"params": bias_params, "weight_decay": float(bias_weight_decay)}
        if lr is not None:
            group["lr"] = float(lr)
        groups.append(group)
    return groups


class FP4LoRACacheRefreshHook:
    """Removable optimizer post-step hook that eagerly refreshes fused LoRA caches."""

    def __init__(self, optimizer: torch.optim.Optimizer, module: torch.nn.Module):
        self.module = module
        self.last_refresh_count = 0
        self.handle = optimizer.register_step_post_hook(self._hook)

    def _hook(self, optimizer: torch.optim.Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del optimizer, args, kwargs
        self.last_refresh_count = refresh_fused_lora_dx_caches(self.module)

    def remove(self) -> None:
        self.handle.remove()


def register_fp4_lora_cache_refresh_hook(
    optimizer: torch.optim.Optimizer,
    module: torch.nn.Module,
) -> FP4LoRACacheRefreshHook:
    """Refresh packed LoRA dX caches after every optimizer step.

    The wrapped modules also have lazy version-based invalidation, so this hook
    is an eager refresh convenience for stable training-step latency.
    """

    return FP4LoRACacheRefreshHook(optimizer, module)


def fp4_lora_state_dict(
    module: torch.nn.Module,
    *,
    include_bias: bool = False,
    destination: str | torch.device | None = "cpu",
    clone: bool = True,
) -> dict[str, torch.Tensor]:
    """Return a LoRA-only state dict for converted FP4 LoRA modules."""

    out: dict[str, torch.Tensor] = {}
    for module_name, child in iter_fp4_lora_modules(module):
        prefix = f"{module_name}." if module_name else ""
        for key, tensor in (
            (f"{prefix}lora_down", child.lora_down),
            (f"{prefix}lora_up", child.lora_up),
        ):
            value = tensor.detach()
            if destination is not None:
                value = value.to(destination)
            if clone:
                value = value.clone()
            out[key] = value

        bias = getattr(child, "bias", None)
        if include_bias and isinstance(bias, torch.nn.Parameter):
            value = bias.detach()
            if destination is not None:
                value = value.to(destination)
            if clone:
                value = value.clone()
            out[f"{prefix}bias"] = value
    return out


def _export_tensor(
    tensor: torch.Tensor,
    destination: str | torch.device | None,
    clone: bool,
) -> torch.Tensor:
    value = tensor.detach()
    if destination is not None:
        value = value.to(destination)
    if clone:
        value = value.clone()
    return value


def _peft_base_name(module_name: str, prefix: str) -> str:
    clean_prefix = prefix.rstrip(".")
    if clean_prefix and module_name:
        return f"{clean_prefix}.{module_name}"
    return clean_prefix or module_name


def _peft_lora_keys(module_name: str, prefix: str, adapter_name: str | None) -> tuple[str, str]:
    base = _peft_base_name(module_name, prefix)
    if adapter_name:
        suffix_a = f"lora_A.{adapter_name}.weight"
        suffix_b = f"lora_B.{adapter_name}.weight"
    else:
        suffix_a = "lora_A.weight"
        suffix_b = "lora_B.weight"
    if not base:
        return suffix_a, suffix_b
    return f"{base}.{suffix_a}", f"{base}.{suffix_b}"


def fp4_lora_peft_state_dict(
    module: torch.nn.Module,
    *,
    adapter_name: str | None = "default",
    prefix: str = "",
    include_bias: bool = False,
    destination: str | torch.device | None = "cpu",
    clone: bool = True,
    trim_to_requested_rank: bool = False,
) -> dict[str, torch.Tensor]:
    """Return a PEFT-style LoRA adapter state dict.

    By default this exports the padded effective rank, which is exact for this
    kernel layout. ``trim_to_requested_rank=True`` emits the user-requested rank
    for ecosystem compatibility and should only be used when dropping padded
    tail channels is acceptable.
    """

    out: dict[str, torch.Tensor] = {}
    for module_name, child in iter_fp4_lora_modules(module):
        key_a, key_b = _peft_lora_keys(module_name, prefix, adapter_name)
        rank = child.requested_rank if trim_to_requested_rank else child.rank
        out[key_a] = _export_tensor(child.lora_down[:rank, :], destination, clone)
        out[key_b] = _export_tensor(child.lora_up[:, :rank], destination, clone)

        bias = getattr(child, "bias", None)
        if include_bias and isinstance(bias, torch.nn.Parameter):
            base = _peft_base_name(module_name, prefix)
            key = f"{base}.bias" if base else "bias"
            out[key] = _export_tensor(bias, destination, clone)
    return out


def load_fp4_lora_state_dict(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a LoRA-only state dict into converted FP4 LoRA modules.

    Returns (missing_keys, unexpected_keys). With strict=True, any mismatch
    raises ValueError after collecting the mismatch lists.
    """

    expected: dict[str, torch.nn.Parameter] = {}
    for module_name, child in iter_fp4_lora_modules(module):
        prefix = f"{module_name}." if module_name else ""
        expected[f"{prefix}lora_down"] = child.lora_down
        expected[f"{prefix}lora_up"] = child.lora_up
        bias = getattr(child, "bias", None)
        if isinstance(bias, torch.nn.Parameter):
            expected[f"{prefix}bias"] = bias

    missing = sorted(key for key in expected if key not in state_dict)
    unexpected = sorted(key for key in state_dict if key not in expected)
    if strict and (missing or unexpected):
        raise ValueError(f"FP4 LoRA state_dict mismatch: missing={missing}, unexpected={unexpected}")

    loaded = 0
    with torch.no_grad():
        for key, param in expected.items():
            if key not in state_dict:
                continue
            value = state_dict[key]
            if value.shape != param.shape:
                raise ValueError(f"Shape mismatch for {key}: expected {tuple(param.shape)}, got {tuple(value.shape)}")
            param.copy_(value.to(device=param.device, dtype=param.dtype))
            loaded += 1

    if loaded:
        clear_fused_lora_dx_caches(module)
    return missing, unexpected


def load_fp4_lora_peft_state_dict(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    adapter_name: str | None = "default",
    prefix: str = "",
    strict: bool = True,
    include_bias: bool = False,
    allow_padded_rank: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a PEFT-style LoRA adapter state dict into FP4 LoRA modules.

    Exact padded-rank tensors are copied directly. If ``allow_padded_rank`` is
    true, requested-rank tensors are copied into the leading channels and the
    padded tail is zeroed so stale random tail weights cannot affect outputs.
    """

    expected: dict[str, tuple[NunchakuFP4LoRALinear, str]] = {}
    for module_name, child in iter_fp4_lora_modules(module):
        key_a, key_b = _peft_lora_keys(module_name, prefix, adapter_name)
        expected[key_a] = (child, "lora_down")
        expected[key_b] = (child, "lora_up")
        bias = getattr(child, "bias", None)
        if include_bias and isinstance(bias, torch.nn.Parameter):
            base = _peft_base_name(module_name, prefix)
            expected[f"{base}.bias" if base else "bias"] = (child, "bias")

    missing = sorted(key for key in expected if key not in state_dict)
    unexpected = sorted(key for key in state_dict if key not in expected)
    if strict and (missing or unexpected):
        raise ValueError(f"FP4 LoRA PEFT state_dict mismatch: missing={missing}, unexpected={unexpected}")

    loaded = 0
    with torch.no_grad():
        for module_name, child in iter_fp4_lora_modules(module):
            key_a, key_b = _peft_lora_keys(module_name, prefix, adapter_name)
            if key_a in state_dict and key_b in state_dict:
                down = state_dict[key_a]
                up = state_dict[key_b]
                if down.dim() != 2 or up.dim() != 2:
                    raise ValueError(f"Expected 2D LoRA tensors for {module_name!r}")
                if down.shape[1] != child.in_features or up.shape[0] != child.out_features:
                    raise ValueError(
                        f"Shape mismatch for {module_name!r}: "
                        f"A={tuple(down.shape)}, B={tuple(up.shape)}, "
                        f"expected (*, {child.in_features}) and ({child.out_features}, *)"
                    )
                rank = down.shape[0]
                if up.shape[1] != rank:
                    raise ValueError(f"Rank mismatch for {module_name!r}: A rank {rank}, B rank {up.shape[1]}")
                if rank == child.rank:
                    child.lora_down.copy_(down.to(device=child.lora_down.device, dtype=child.lora_down.dtype))
                    child.lora_up.copy_(up.to(device=child.lora_up.device, dtype=child.lora_up.dtype))
                elif allow_padded_rank and 0 < rank <= child.rank:
                    child.lora_down.zero_()
                    child.lora_up.zero_()
                    child.lora_down[:rank, :].copy_(
                        down.to(device=child.lora_down.device, dtype=child.lora_down.dtype)
                    )
                    child.lora_up[:, :rank].copy_(up.to(device=child.lora_up.device, dtype=child.lora_up.dtype))
                else:
                    raise ValueError(f"Rank mismatch for {module_name!r}: expected {child.rank}, got {rank}")
                loaded += 2

            bias = getattr(child, "bias", None)
            if include_bias and isinstance(bias, torch.nn.Parameter):
                base = _peft_base_name(module_name, prefix)
                key = f"{base}.bias" if base else "bias"
                if key in state_dict:
                    value = state_dict[key]
                    if value.shape != bias.shape:
                        raise ValueError(
                            f"Shape mismatch for {key}: expected {tuple(bias.shape)}, got {tuple(value.shape)}"
                        )
                    bias.copy_(value.to(device=bias.device, dtype=bias.dtype))
                    loaded += 1

    if loaded:
        clear_fused_lora_dx_caches(module)
    return missing, unexpected


def freeze_non_fp4_lora_parameters(module: torch.nn.Module, train_bias: bool = False) -> list[str]:
    """Freeze everything except FP4 LoRA A/B and optionally trainable LoRA bias."""

    for param in module.parameters():
        param.requires_grad_(False)

    trainable: list[str] = []
    allowed = {name for name, _ in iter_fp4_lora_named_parameters(module, train_bias=train_bias)}
    for name, param in module.named_parameters():
        allow = name in allowed
        param.requires_grad_(allow)
        if allow:
            trainable.append(name)
    return trainable


def prepare_fp4_lora_finetuning(
    module: torch.nn.Module,
    config: FP4LoRAConfig | None = None,
    *,
    mode: FP4LoRAFinetuneMode = "balanced",
    rank: int = 32,
    lora_alpha: float | None = None,
    dtype: torch.dtype = torch.bfloat16,
    lowrank_dtype: torch.dtype | None = None,
    use_frozen_residual: bool = True,
    frozen_residual_rank: int | None = None,
    residual_svd_method: ResidualSVDMethod | None = None,
    residual_svd_lowrank_oversample: int = 8,
    residual_svd_lowrank_niter: int = 2,
    train_bias: bool = False,
    cache_lora_act: bool = True,
    activation_checkpoint: bool = False,
    overlap_lora_grad_min_rows: int = DEFAULT_OVERLAP_LORA_GRAD_MIN_ROWS,
    target_modules: Iterable[str] | None = DEFAULT_FP4_LORA_TARGET_MODULES,
    exclude_modules: Iterable[str] | None = DEFAULT_FP4_LORA_EXCLUDE_MODULES,
    config_overrides: Mapping[str, FP4LoRAConfig] | None = None,
    outlier_report: Mapping[str, Any] | str | None = None,
    outlier_rank_field: str = "suggested_rank",
    outlier_rank_multiple: int = 16,
    outlier_min_rank: int | None = None,
    outlier_max_rank: int | None = None,
    inplace: bool = True,
    refresh_caches: bool = True,
    lora_weight_decay: float = 0.0,
    bias_weight_decay: float = 0.0,
    lr: float | None = None,
) -> FP4LoRAPrepareResult:
    """Convert a model and prepare LoRA-only optimizer inputs for FP4 fine-tuning.

    This is the high-level training entry point: it creates a recommended config
    when one is not provided, replaces selected ``Linear`` modules, freezes all
    non-LoRA parameters, optionally refreshes fused dX caches, and returns
    LoRA-only optimizer parameter groups. The caller still owns optimizer and
    scheduler construction.
    """

    cfg = config
    if cfg is None:
        cfg = fp4_lora_finetune_config(
            mode=mode,
            rank=rank,
            lora_alpha=lora_alpha,
            dtype=dtype,
            lowrank_dtype=lowrank_dtype,
            use_frozen_residual=use_frozen_residual,
            frozen_residual_rank=frozen_residual_rank,
            residual_svd_method=residual_svd_method,
            residual_svd_lowrank_oversample=residual_svd_lowrank_oversample,
            residual_svd_lowrank_niter=residual_svd_lowrank_niter,
            train_bias=train_bias,
            cache_lora_act=cache_lora_act,
            activation_checkpoint=activation_checkpoint,
            overlap_lora_grad_min_rows=overlap_lora_grad_min_rows,
        )
    else:
        train_bias = cfg.train_bias

    merged_overrides: dict[str, FP4LoRAConfig] = {}
    if config_overrides:
        merged_overrides.update(config_overrides)
    if outlier_report is not None:
        auto_overrides = fp4_lora_config_overrides_from_outlier_report(
            outlier_report,
            cfg,
            rank_field=outlier_rank_field,
            rank_multiple=outlier_rank_multiple,
            min_rank=outlier_min_rank,
            max_rank=outlier_max_rank,
            force_init="zero",
            disable_fuse_frozen_residual_dx=True,
        )
        for key, value in auto_overrides.items():
            merged_overrides.setdefault(key, value)

    targets = _as_tuple(target_modules)
    excludes = _as_tuple(exclude_modules)
    prepared_model, replaced = convert_linear_to_fp4_lora(
        module,
        cfg,
        target_modules=targets,
        exclude_modules=excludes,
        config_overrides=merged_overrides or None,
        inplace=inplace,
    )
    trainable_names = freeze_non_fp4_lora_parameters(prepared_model, train_bias=train_bias)
    refreshed = refresh_fused_lora_dx_caches(prepared_model) if refresh_caches else 0
    param_groups = fp4_lora_parameter_groups(
        prepared_model,
        train_bias=train_bias,
        lora_weight_decay=lora_weight_decay,
        bias_weight_decay=bias_weight_decay,
        lr=lr,
    )
    trainable_param_count = int(sum(param.numel() for group in param_groups for param in group["params"]))

    return FP4LoRAPrepareResult(
        model=prepared_model,
        config=cfg,
        replaced_modules=replaced,
        trainable_names=trainable_names,
        trainable_param_count=trainable_param_count,
        optimizer_param_groups=param_groups,
        refreshed_cache_count=refreshed,
        target_modules=targets,
        exclude_modules=excludes,
    )
