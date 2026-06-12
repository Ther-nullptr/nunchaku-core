from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .training import LoRAInitMode, NunchakuFP4LoRALinear


@dataclass(frozen=True)
class FP4LoRAConfig:
    rank: int = 32
    lora_alpha: float | None = None
    lowrank_dtype: torch.dtype = torch.bfloat16
    init: LoRAInitMode = "zero"
    train_bias: bool = False
    cache_lora_act: bool = True
    fuse_lora_dx: bool = False
    cache_fused_lora_dx: bool = False
    reuse_fused_dy_up_for_d_lora_down: bool = False


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


def convert_linear_to_fp4_lora(
    module: torch.nn.Module,
    config: FP4LoRAConfig | None = None,
    *,
    target_modules: Iterable[str] | None = None,
    exclude_modules: Iterable[str] | None = None,
    inplace: bool = True,
) -> tuple[torch.nn.Module, list[str]]:
    """Replace selected CUDA Linear modules with NunchakuFP4LoRALinear.

    Matching uses the full module path, the child name, or a full-path suffix.
    For example, target_modules=("q_proj", "down_proj") matches
    "layers.0.self_attn.q_proj" and "layers.0.mlp.down_proj".
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
                    if not child.weight.is_cuda:
                        raise ValueError(f"Linear module {full_name!r} must be on CUDA before FP4 LoRA conversion")
                    if child.weight.dtype not in (torch.float16, torch.bfloat16):
                        raise ValueError(f"Linear module {full_name!r} weight must be float16 or bfloat16")
                    fp4_lora = NunchakuFP4LoRALinear.from_linear(
                        child,
                        rank=cfg.rank,
                        lora_alpha=cfg.lora_alpha,
                        lowrank_dtype=cfg.lowrank_dtype,
                        init=cfg.init,
                        train_bias=cfg.train_bias,
                        cache_lora_act=cfg.cache_lora_act,
                        fuse_lora_dx=cfg.fuse_lora_dx,
                        cache_fused_lora_dx=cfg.cache_fused_lora_dx,
                        reuse_fused_dy_up_for_d_lora_down=cfg.reuse_fused_dy_up_for_d_lora_down,
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
