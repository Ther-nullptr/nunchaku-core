from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from native_fp4 import (  # noqa: E402
    FP4LoRAConfig,
    NunchakuFP4LoRALinear,
    clear_fused_lora_dx_caches,
    convert_linear_to_fp4_lora,
    fp4_lora_config_overrides_from_outlier_report,
    fp4_lora_parameter_groups,
    fp4_lora_peft_state_dict,
    fp4_lora_state_dict,
    freeze_non_fp4_lora_parameters,
    iter_fp4_lora_named_parameters,
    iter_fp4_lora_modules,
    load_fp4_lora_peft_state_dict,
    load_fp4_lora_state_dict,
    register_fp4_lora_cache_refresh_hook,
    refresh_fused_lora_dx_caches,
)


class TinyBlock(torch.nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(hidden, hidden, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.q_proj(x)) + self.k_proj(x))


class TinyModel(torch.nn.Module):
    def __init__(self, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(hidden, dtype), TinyBlock(hidden, dtype)])
        self.lm_head = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--frozen-residual-rank", type=int, default=0)
    p.add_argument("--frozen-residual-init", choices=["none", "residual_svd"], default="none")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--lowrank-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--init", choices=["zero", "gaussian", "residual_svd"], default="gaussian")
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--fuse-lowrank-forward", action="store_true")
    p.add_argument("--fuse-lora-dx", action="store_true")
    p.add_argument("--fuse-frozen-residual-dx", action="store_true")
    p.add_argument("--cache-fused-lora-dx", action="store_true")
    p.add_argument("--reuse-fused-dy-up-for-d-lora-down", action="store_true")
    p.add_argument("--overlap-lora-grad", action="store_true")
    p.add_argument("--overlap-lora-grad-min-rows", type=int, default=4096)
    p.add_argument("--fp4-activation-cache-d-lora-down", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    lowrank_dtype = torch.float16 if args.lowrank_dtype == "fp16" else torch.bfloat16
    model = TinyModel(args.hidden, dtype).cuda()

    cfg = FP4LoRAConfig(
        rank=args.rank,
        lowrank_dtype=lowrank_dtype,
        init=args.init,
        frozen_residual_rank=args.frozen_residual_rank,
        frozen_residual_init=args.frozen_residual_init,
        train_bias=False,
        cache_lora_act=True,
        activation_checkpoint=args.activation_checkpoint,
        fuse_lowrank_forward=args.fuse_lowrank_forward,
        fuse_lora_dx=args.fuse_lora_dx,
        fuse_frozen_residual_dx=args.fuse_frozen_residual_dx,
        cache_fused_lora_dx=args.cache_fused_lora_dx,
        reuse_fused_dy_up_for_d_lora_down=args.reuse_fused_dy_up_for_d_lora_down,
        overlap_lora_grad=args.overlap_lora_grad,
        overlap_lora_grad_min_rows=args.overlap_lora_grad_min_rows,
        fp4_activation_cache_d_lora_down=args.fp4_activation_cache_d_lora_down,
    )
    override_rank = 24 if args.rank != 24 else 16
    override_cfg = replace(cfg, rank=override_rank, init="zero")
    config_overrides = {"layers.1.down_proj": override_cfg}
    auto_override_rank = 48 if args.rank != 48 else 64
    auto_overrides = fp4_lora_config_overrides_from_outlier_report(
        {
            "summary": {
                "rank_bump_candidates": [
                    {
                        "module": "layers.1.down_proj",
                        "suggested_rank": auto_override_rank,
                    }
                ]
            }
        },
        cfg,
        force_init="zero",
        disable_fuse_frozen_residual_dx=True,
    )
    model, replaced = convert_linear_to_fp4_lora(
        model,
        cfg,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        config_overrides=config_overrides,
    )
    trainable = freeze_non_fp4_lora_parameters(model)
    refreshed = refresh_fused_lora_dx_caches(model)
    cleared = clear_fused_lora_dx_caches(model)
    refreshed_after_clear = refresh_fused_lora_dx_caches(model)

    x = torch.randn(args.batch, args.hidden, device="cuda", dtype=dtype, requires_grad=True)
    y = model(x)
    loss = y.float().square().mean()
    loss.backward()

    fp4_modules = dict(iter_fp4_lora_modules(model))
    frozen_residual_modules = {
        name: child for name, child in fp4_modules.items() if child.has_frozen_residual
    }
    expected_frozen_residual_count = len(replaced) if args.frozen_residual_init != "none" else 0
    frozen_residual_params = {
        name for name, _ in model.named_parameters() if "frozen_residual" in name
    }
    frozen_residual_buffers_finite = all(
        bool(torch.isfinite(child.frozen_residual_down).all())
        and bool(torch.isfinite(child.frozen_residual_up).all())
        for child in frozen_residual_modules.values()
    )
    trainable_named = {name for name, param in model.named_parameters() if param.requires_grad}
    grad_named = {name for name, param in model.named_parameters() if param.grad is not None}
    expected_replaced = {
        "layers.0.q_proj",
        "layers.0.down_proj",
        "layers.1.q_proj",
        "layers.1.down_proj",
    }
    expected_adapter_keys = {f"{name}.{param}" for name in expected_replaced for param in ("lora_down", "lora_up")}
    expected_cache_count = len(replaced) if args.fuse_lora_dx and args.cache_fused_lora_dx else 0

    named_lora_params = dict(iter_fp4_lora_named_parameters(model))
    param_groups = fp4_lora_parameter_groups(model, lora_weight_decay=0.0)
    optimizer = torch.optim.AdamW(param_groups, lr=1e-3, eps=1e-4)
    optimizer_hook = register_fp4_lora_cache_refresh_hook(optimizer, model)
    pre_step_refreshed = refresh_fused_lora_dx_caches(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    hook_refresh_count = optimizer_hook.last_refresh_count
    optimizer_hook.remove()
    optimizer_param_ids = {id(param) for group in param_groups for param in group["params"]}
    expected_param_ids = {id(param) for param in named_lora_params.values()}
    optimizer_hook_caches_current = all(
        child._cached_lora_down_version == child.lora_down._version
        and child._cached_lora_up_version == child.lora_up._version
        and child._cached_lora_scaling == float(child.scaling)
        for _, child in iter_fp4_lora_modules(model)
        if child.fuse_lora_dx and child.cache_fused_lora_dx
    )

    adapter_state = fp4_lora_state_dict(model)
    adapter_state_finite = all(bool(torch.isfinite(value).all()) for value in adapter_state.values())
    model2 = TinyModel(args.hidden, dtype).cuda()
    model2, replaced2 = convert_linear_to_fp4_lora(
        model2,
        cfg,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        config_overrides=config_overrides,
    )
    pre_load_refreshed = refresh_fused_lora_dx_caches(model2)
    missing, unexpected = load_fp4_lora_state_dict(model2, adapter_state, strict=True)
    loaded_state = fp4_lora_state_dict(model2)
    loaded_matches = all(torch.equal(adapter_state[key], loaded_state[key]) for key in expected_adapter_keys)
    fp4_modules2 = dict(iter_fp4_lora_modules(model2))
    caches_cleared_after_load = all(
        child._cached_lora_down_bwd_packed is None and child._cached_lora_up_bwd_packed is None
        for _, child in iter_fp4_lora_modules(model2)
    )
    strict_mismatch_raises = False
    try:
        bad_state = dict(adapter_state)
        bad_state.pop(next(iter(expected_adapter_keys)))
        load_fp4_lora_state_dict(model2, bad_state, strict=True)
    except ValueError:
        strict_mismatch_raises = True

    model_auto = TinyModel(args.hidden, dtype).cuda()
    model_auto, replaced_auto = convert_linear_to_fp4_lora(
        model_auto,
        cfg,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        config_overrides=auto_overrides,
    )
    fp4_modules_auto = dict(iter_fp4_lora_modules(model_auto))

    expected_peft_keys = {
        f"{name}.{suffix}"
        for name in expected_replaced
        for suffix in ("lora_A.default.weight", "lora_B.default.weight")
    }
    peft_state = fp4_lora_peft_state_dict(model)
    peft_state_finite = all(bool(torch.isfinite(value).all()) for value in peft_state.values())
    model3 = TinyModel(args.hidden, dtype).cuda()
    model3, replaced3 = convert_linear_to_fp4_lora(
        model3,
        cfg,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        config_overrides=config_overrides,
    )
    peft_missing, peft_unexpected = load_fp4_lora_peft_state_dict(model3, peft_state, strict=True)
    peft_loaded_state = fp4_lora_peft_state_dict(model3)
    peft_loaded_matches = all(torch.equal(peft_state[key], peft_loaded_state[key]) for key in expected_peft_keys)

    peft_trimmed_state = fp4_lora_peft_state_dict(model, trim_to_requested_rank=True)
    model4 = TinyModel(args.hidden, dtype).cuda()
    model4, replaced4 = convert_linear_to_fp4_lora(
        model4,
        cfg,
        target_modules=("q_proj", "down_proj"),
        exclude_modules=("lm_head",),
        config_overrides=config_overrides,
    )
    peft_trimmed_missing, peft_trimmed_unexpected = load_fp4_lora_peft_state_dict(
        model4,
        peft_trimmed_state,
        strict=True,
    )
    fp4_modules4 = dict(iter_fp4_lora_modules(model4))
    peft_trimmed_shapes_match_requested_rank = all(
        peft_trimmed_state[f"{name}.lora_A.default.weight"].shape
        == (fp4_modules[name].requested_rank, fp4_modules[name].in_features)
        and peft_trimmed_state[f"{name}.lora_B.default.weight"].shape
        == (fp4_modules[name].out_features, fp4_modules[name].requested_rank)
        for name in expected_replaced
    )
    peft_trimmed_load_leading_matches = all(
        torch.equal(
            fp4_modules4[name].lora_down[: fp4_modules[name].requested_rank, :].detach().cpu(),
            fp4_modules[name].lora_down[: fp4_modules[name].requested_rank, :].detach().cpu(),
        )
        and torch.equal(
            fp4_modules4[name].lora_up[:, : fp4_modules[name].requested_rank].detach().cpu(),
            fp4_modules[name].lora_up[:, : fp4_modules[name].requested_rank].detach().cpu(),
        )
        for name in expected_replaced
    )
    peft_trimmed_load_tail_zero = all(
        (
            fp4_modules4[name].requested_rank == fp4_modules4[name].rank
            or (
                bool(torch.count_nonzero(fp4_modules4[name].lora_down[fp4_modules4[name].requested_rank :, :]) == 0)
                and bool(torch.count_nonzero(fp4_modules4[name].lora_up[:, fp4_modules4[name].requested_rank :]) == 0)
            )
        )
        for name in expected_replaced
    )

    checks = {
        "replaced_expected_modules": set(replaced) == expected_replaced,
        "second_model_replaced_expected_modules": set(replaced2) == expected_replaced,
        "peft_model_replaced_expected_modules": set(replaced3) == expected_replaced,
        "trimmed_peft_model_replaced_expected_modules": set(replaced4) == expected_replaced,
        "auto_override_model_replaced_expected_modules": set(replaced_auto) == expected_replaced,
        "lm_head_not_replaced": not isinstance(model.lm_head, NunchakuFP4LoRALinear),
        "all_replaced_are_fp4_lora": all(isinstance(fp4_modules[name], NunchakuFP4LoRALinear) for name in expected_replaced),
        "config_override_rank_applied": fp4_modules["layers.1.down_proj"].requested_rank == override_rank,
        "config_override_init_applied": fp4_modules["layers.1.down_proj"].init_mode == "zero",
        "base_configs_preserved": all(
            fp4_modules[name].requested_rank == args.rank and fp4_modules[name].init_mode == args.init
            for name in expected_replaced
            if name != "layers.1.down_proj"
        ),
        "second_model_config_override_rank_applied": fp4_modules2["layers.1.down_proj"].requested_rank == override_rank,
        "outlier_report_override_rank_applied": (
            fp4_modules_auto["layers.1.down_proj"].requested_rank == auto_override_rank
        ),
        "outlier_report_override_init_applied": fp4_modules_auto["layers.1.down_proj"].init_mode == "zero",
        "outlier_report_override_disabled_fused_residual_dx": (
            not fp4_modules_auto["layers.1.down_proj"].fuse_frozen_residual_dx
        ),
        "only_lora_trainable": trainable_named == set(trainable),
        "trainable_grads_present": set(trainable).issubset(grad_named),
        "x_grad_finite": bool(x.grad is not None and torch.isfinite(x.grad).all()),
        "output_finite": bool(torch.isfinite(y).all()),
        "cache_count_matches": refreshed == expected_cache_count,
        "clear_count_matches": cleared == len(replaced),
        "refresh_after_clear_matches": refreshed_after_clear == expected_cache_count,
        "frozen_residual_count_matches": len(frozen_residual_modules) == expected_frozen_residual_count,
        "frozen_residual_not_parameter": frozen_residual_params == set(),
        "frozen_residual_buffers_finite": frozen_residual_buffers_finite,
        "lora_named_parameters_match_trainable": set(named_lora_params) == set(trainable),
        "optimizer_param_groups_match_lora": optimizer_param_ids == expected_param_ids,
        "optimizer_pre_step_refresh_count_matches": pre_step_refreshed == expected_cache_count,
        "optimizer_hook_refresh_count_matches": hook_refresh_count == expected_cache_count,
        "optimizer_hook_caches_current": optimizer_hook_caches_current,
        "adapter_state_finite": adapter_state_finite,
        "adapter_state_keys_match": set(adapter_state) == expected_adapter_keys,
        "adapter_state_excludes_frozen_residual": not any("frozen_residual" in key for key in adapter_state),
        "adapter_state_load_missing_empty": missing == [],
        "adapter_state_load_unexpected_empty": unexpected == [],
        "adapter_state_loaded_matches": loaded_matches,
        "adapter_load_clears_cache": caches_cleared_after_load,
        "strict_mismatch_raises": strict_mismatch_raises,
        "peft_adapter_state_finite": peft_state_finite,
        "peft_adapter_state_keys_match": set(peft_state) == expected_peft_keys,
        "peft_adapter_state_excludes_frozen_residual": not any("frozen_residual" in key for key in peft_state),
        "peft_adapter_state_load_missing_empty": peft_missing == [],
        "peft_adapter_state_load_unexpected_empty": peft_unexpected == [],
        "peft_adapter_state_loaded_matches": peft_loaded_matches,
        "peft_trimmed_shapes_match_requested_rank": peft_trimmed_shapes_match_requested_rank,
        "peft_trimmed_load_missing_empty": peft_trimmed_missing == [],
        "peft_trimmed_load_unexpected_empty": peft_trimmed_unexpected == [],
        "peft_trimmed_load_leading_matches": peft_trimmed_load_leading_matches,
        "peft_trimmed_load_tail_zero": peft_trimmed_load_tail_zero,
    }
    payload = {
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "rank": args.rank,
            "frozen_residual_rank": args.frozen_residual_rank,
            "frozen_residual_init": args.frozen_residual_init,
            "dtype": args.dtype,
            "lowrank_dtype": args.lowrank_dtype,
            "init": args.init,
            "activation_checkpoint": args.activation_checkpoint,
            "fuse_lowrank_forward": args.fuse_lowrank_forward,
            "fuse_lora_dx": args.fuse_lora_dx,
            "fuse_frozen_residual_dx": args.fuse_frozen_residual_dx,
            "cache_fused_lora_dx": args.cache_fused_lora_dx,
            "reuse_fused_dy_up_for_d_lora_down": args.reuse_fused_dy_up_for_d_lora_down,
            "overlap_lora_grad": args.overlap_lora_grad,
            "overlap_lora_grad_min_rows": args.overlap_lora_grad_min_rows,
            "fp4_activation_cache_d_lora_down": args.fp4_activation_cache_d_lora_down,
        },
        "config_overrides": {
            "layers.1.down_proj": {
                "rank": override_rank,
                "init": "zero",
            }
        },
        "outlier_report_config_overrides": {
            name: {
                "rank": override.rank,
                "init": override.init,
                "fuse_frozen_residual_dx": override.fuse_frozen_residual_dx,
            }
            for name, override in auto_overrides.items()
        },
        "replaced": replaced,
        "trainable": trainable,
        "grad_named": sorted(grad_named),
        "frozen_residual_modules": sorted(frozen_residual_modules),
        "cache_counts": {
            "refreshed": refreshed,
            "cleared": cleared,
            "refreshed_after_clear": refreshed_after_clear,
            "pre_load_refreshed": pre_load_refreshed,
            "pre_step_refreshed": pre_step_refreshed,
            "optimizer_hook_refreshed": hook_refresh_count,
        },
        "adapter_state_keys": sorted(adapter_state),
        "peft_adapter_state_keys": sorted(peft_state),
        "peft_trimmed_adapter_shapes": {
            key: list(value.shape) for key, value in sorted(peft_trimmed_state.items())
        },
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "latest_native_fp4_lora_modeling_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
