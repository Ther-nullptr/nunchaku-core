# FP4 + BF16 LoRA 微调接口设计

## 目标

把当前已经独立出来的 Nunchaku FP4 推理/反向算子推进到可接入微调的接口层：

```text
y = FP4_GEMM(x, W0)
  + (x @ R_down.T) @ R_up.T
  + scaling * (x @ A.T) @ B.T
  + bias
```

- `W0` 是冻结的 FP4 backbone 权重。
- `R_down/R_up` 是可选 frozen BF16/FP16 residual branch，用来固定量化补偿。
- `A/B` 是可训练的 BF16/FP16 LoRA 权重。
- forward 的 FP4 主分支复用现有 `gemm_w4a4` CUDA kernel。
- backward 的 `dX_main = dY @ W0` 复用现有 transient repack + FP4 GEMM 路径。
- `dA/dB` 先用 PyTorch BF16/FP16 matmul，作为 P0 可训练接口；后续再把低秩梯度规约融合进 CUDA。

## 本轮调研结论

### Nunchaku / SVDQuant 路径

- 原始核心创新是 `4-bit backbone + 16-bit low-rank branch`。
- 当前 extracted core 已经具备：
  - `NunchakuFP4GemmOp`：纯 FP4 forward。
  - `NunchakuFP4LowRankOp`：FP4 + low-rank fused forward。
  - `NunchakuFP4BackwardDXOp`：纯 FP4 backward dX。
  - `NunchakuFP4LowRankBackwardDXOp`：混合 backward dX 和多种 full backward 消融。
- 现有 full backward 更像固定 residual low-rank 分支的算子验证；微调需要显式暴露 trainable LoRA 参数。

### KernelWiki / contest solution

- NVFP4/FP4 的高性能路径依赖 Blackwell 原生 FP4 MMA，以及每 16 个元素一组的细粒度 scale。
- DeepGEMM / `tcgen05.mma` 资料更偏 SM100/B200 数据中心 Blackwell；本机 RTX 5090 路径此前可编译的是 Nunchaku 手写 `mma.sync ... mxf4nvf4` 路径。
- contest workflow 强调每个候选实现都要留下 task contract、正确性验证、benchmark 证据和保留/回滚决策。

### personal-vault 里的 FP4 finetuning 需求

- 目标不是训练 FP4 backbone，而是冻结量化 backbone，只训练 LoRA。
- 关键公式：

```text
forward:  y  = x @ Q4(W0).T + x @ A.T @ B.T
backward: dX = dY @ Q4(W0)   + dY @ B @ A
          dB = dY.T @ (x @ A.T)
          dA = (dY @ B).T @ x
```

推荐的微调形态是 personal-vault 里“方案 B”：

```text
forward: y = x @ Q4(W0).T + x @ R_down.T @ R_up.T + scaling * x @ A.T @ B.T
```

其中 `R_down/R_up` 冻结，用 `W - dequant(Q4(W))` 的低秩 SVD 初始化，持续承担量化误差补偿；`A/B` 是 zero-init task LoRA，只负责下游任务适配。

- 主要工程风险：
  - backward 不能常驻保存第二份 transposed FP4 packed weight，否则压缩权重内存接近翻倍。
  - 如果完全不缓存 forward LoRA activation，`dB` 需要重算 `x @ A.T`。
  - outlier 激活仍会限制 FP4 主分支精度，需要训练时的 scale/smooth 策略继续单独研究。

## 已落地接口

新增：

- `native_fp4.training.NunchakuFP4LoRALinear`
- `native_fp4.modeling.FP4LoRAConfig`
- `native_fp4.modeling.convert_linear_to_fp4_lora`
- `native_fp4.modeling.freeze_non_fp4_lora_parameters`
- `native_fp4.modeling.iter_fp4_lora_named_parameters`
- `native_fp4.modeling.fp4_lora_parameter_groups`
- `native_fp4.modeling.register_fp4_lora_cache_refresh_hook`
- `native_fp4.modeling.fp4_lora_state_dict`
- `native_fp4.modeling.load_fp4_lora_state_dict`
- `native_fp4.modeling.refresh_fused_lora_forward_caches`
- `native_fp4.modeling.clear_fused_lora_forward_caches`
- `native_fp4.modeling.refresh_fused_lora_dx_caches`
- `native_fp4.modeling.clear_fused_lora_dx_caches`
- `benchmarks/validate_native_fp4_lora_training.py`
- `benchmarks/validate_native_fp4_lora_modeling.py`

模块语义：

```python
from native_fp4 import NunchakuFP4LoRALinear

op = NunchakuFP4LoRALinear(
    weight=linear.weight,
    bias=linear.bias,
    rank=32,
    lowrank_dtype=torch.bfloat16,
    init="zero",
    frozen_residual_rank=32,
    frozen_residual_init="residual_svd",
)
y = op(x)
```

模型级替换示例：

```python
from dataclasses import replace

from native_fp4 import (
    FP4LoRAConfig,
    convert_linear_to_fp4_lora,
    fp4_lora_config_overrides_from_outlier_report,
    fp4_lora_finetune_config,
    fp4_lora_sensitivity_policy_from_report,
    prepare_fp4_lora_finetuning,
)

cfg = fp4_lora_finetune_config(
    mode="balanced",
    rank=32,
    dtype=torch.bfloat16,
    lowrank_dtype=torch.bfloat16,
)

sensitive_overrides = {
    # 同一模型内不同 projection 可用不同 LoRA/residual 策略。
    # 典型用途：down_proj 或 activation outlier 明显的完整模块路径。
    "layers.1.mlp.down_proj": replace(cfg, rank=64, fuse_frozen_residual_dx=False),
}

auto_overrides = fp4_lora_config_overrides_from_outlier_report(
    "results/latest_fp4_lora_activation_grad_outliers.json",
    cfg,
    force_init="zero",
    disable_fuse_frozen_residual_dx=True,
)
sensitive_overrides.update(auto_overrides)

prepared = prepare_fp4_lora_finetuning(
    model.cuda().to(torch.bfloat16),
    config=cfg,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    exclude_modules=("lm_head",),
    config_overrides=sensitive_overrides,
    sensitivity_report="results/llama_module_fp4_sensitivity_20260321_202421.json",
    sensitivity_rank_bump_ratio=1.05,
    sensitivity_exclude_ratio=10.0,
    sensitivity_rank_scale=2.0,
    lr=1e-4,
)
model = prepared.model
```

`sensitivity_report` 直接消费真实模型 module sensitivity scan 的 `module_records[].perplexity_ratio_vs_fp16`：超过 `sensitivity_exclude_ratio` 的 projection 保持 BF16/FP16，超过 `sensitivity_rank_bump_ratio` 的 projection 自动提高 LoRA rank。策略优先级是手写 `config_overrides` > activation/grad outlier report > module sensitivity report；LlamaForCausalLM 报告里的 `model.` 前缀会自动生成去前缀 alias，便于同一份报告复用到裸 decoder 子模块。

`fp4_lora_finetune_config(mode=...)` 是微调推荐入口，用来避免手动拼装不兼容开关：

| mode | 目标 | 关键策略 |
| --- | --- | --- |
| `accuracy` | 精度优先 / 调试 | `full_svd` frozen residual，dense LoRA dX，exact `dA/dB` |
| `balanced` | 默认推荐 | `svd_lowrank` frozen residual，fused cached LoRA dX，exact `dA/dB`，大 batch exact overlap |
| `throughput` | 速度消融 | fused low-rank forward；FP16 自动 fused frozen-residual dX，并关闭不兼容的 overlap |
| `memory_saving` | 显存压力模式 | 保存 FP4 activation cache 计算近似 `dA`，默认 fused backend，自动关闭 overlap |

`validate_fp4_lora_training_policies.py` 已验证这些预设能实际运行 forward/backward/optimizer step：BF16 四模式全部通过；FP16 `throughput` 覆盖 `fuse_frozen_residual_dx=True, overlap_lora_grad=False` 的自动规则；`memory_saving + fp4_activation_cache_d_lora_down_backend="dequant_gemm"` 也通过。验证脚本也覆盖 `backward_weight_policy="cache"`，确认 compressed backward qweight cache 预热后仍常驻。

`prepare_fp4_lora_finetuning` 是真实微调推荐入口：它包装 `convert_linear_to_fp4_lora + freeze_non_fp4_lora_parameters + refresh_fused_lora_forward_caches + refresh_fused_lora_dx_caches + fp4_lora_parameter_groups`，返回 `FP4LoRAPrepareResult`。验证脚本 `validate_fp4_lora_prepare.py` 覆盖了替换层、manual override 优先级、sensitivity 自动 rank bump/exclude、LoRA-only 冻结、optimizer 参数组、cache hook、cache summary 和一次 backward/optimizer step；BF16 balanced、FP16 throughput、BF16 memory_saving/dequant_gemm 均通过。`backward_weight_policy="cache"` 是显式 opt-in，prepare 会预热 compressed backward qweight 并报告 `refreshed_backward_weight_count`；`cache_summary` 记录当前实际常驻的 packed LoRA forward cache、packed LoRA dX cache、backward qweight cache 和相对 dense weight 的字节比例；native fused forward 会在 `prepare(..., refresh_caches=True)` 时预热 forward cache；optimizer hook 会刷新随 LoRA 参数变化的 packed forward/dX cache，并用 `last_fused_lora_forward_refresh_count`、`last_fused_lora_dx_refresh_count` 和 `last_backward_weight_cache_count` 区分三类状态，默认 `repack` 仍不常驻第二份 FP4 backbone。

`benchmark_fp4_lora_prepare_policies.py` 使用同一个 high-level prepare 入口构建 TinyTransformer，默认比较 dense LoRA baseline 与 `accuracy/balanced/throughput/memory_saving_fused/memory_saving_dequant_gemm`，并把 optimizer step 与 cache refresh hook 计入 train-step latency；输出 `latest_fp4_lora_prepare_policies.json`，用于模型级 preset 速度、峰值显存、cache summary、初始 forward 误差和相对 dense LoRA speedup 消融。

追加 `--include-reuse-policies` 后，benchmark 会为支持的 `balanced/throughput` preset 增加 `*_reuse_dy_up` 记录，并在 JSON 中写出 `reuse_fused_dy_up_for_d_lora_down`。该策略要求 `dtype == lowrank_dtype`；如果 frozen residual 开启，高层 config 会自动关闭 reuse-based overlap，避免当前不支持的 frozen-residual overlap 组合。需要看 reuse+overlap 上限时应同时使用 `--no-frozen-residual`。RTX 5090 短测显示，TinyTransformer 小 M 形状下 `balanced_reuse_dy_up` 相对 `balanced` 为 `0.968x`，而 4096 单层 kernel benchmark 中 BF16 reuse/reuse+overlap 分别为 `1.014x/1.032x`，因此该项是形状相关的 opt-in 消融，不作为默认 preset。

Adapter checkpoint 示例：

```python
from native_fp4 import (
    fp4_lora_peft_state_dict,
    fp4_lora_state_dict,
    load_fp4_lora_peft_state_dict,
    load_fp4_lora_state_dict,
)

adapter_state = fp4_lora_state_dict(model)
missing, unexpected = load_fp4_lora_state_dict(model, adapter_state, strict=True)

# PEFT 风格 key：module.lora_A.default.weight / module.lora_B.default.weight。
peft_state = fp4_lora_peft_state_dict(model)
peft_trimmed_state = fp4_lora_peft_state_dict(model, trim_to_requested_rank=True)
missing, unexpected = load_fp4_lora_peft_state_dict(model, peft_state, strict=True)
```

checkpoint 边界：

- `fp4_lora_state_dict` 只导出 `lora_down/lora_up` 和可选 trainable bias。
- `fp4_lora_peft_state_dict` 导出 PEFT 风格 `lora_A/lora_B`；默认导出 padded effective rank 以保证数值无损。
- `trim_to_requested_rank=True` 会裁到用户请求的 rank；加载时 padded tail 清零，适合需要原始 rank 的外部生态，但会丢弃 tail rank。
- 不导出 `qweight/wscales/wscales_bwd_*` 等 frozen FP4 backbone buffers。
- 不导出 `frozen_residual_down/frozen_residual_up`；它们属于 quantized base model 的 frozen compensation branch，不属于 task adapter。
- `load_fp4_lora_state_dict` 和 `load_fp4_lora_peft_state_dict` 加载后都会清空 packed LoRA forward/dX cache，避免 adapter 参数与 cache 不一致。

Optimizer 示例：

```python
from native_fp4 import fp4_lora_parameter_groups, register_fp4_lora_cache_refresh_hook

optimizer = torch.optim.AdamW(fp4_lora_parameter_groups(model), lr=1e-4, eps=1e-4)
cache_hook = register_fp4_lora_cache_refresh_hook(optimizer, model)
```

optimizer 边界：

- `fp4_lora_parameter_groups` 只返回 LoRA A/B，`train_bias=True` 时额外包含 trainable bias。
- 如果 LoRA 参数是 FP16，AdamW 建议显式设置 `eps=1e-4` 或使用带 FP32 master weight 的 optimizer，避免默认 `1e-8` 在 FP16 下数值过小。
- `register_fp4_lora_cache_refresh_hook` 使用 PyTorch optimizer post-step hook，不替换 optimizer 类型，因此不影响 scheduler 使用原始 optimizer。
- `NunchakuFP4LoRALinear` 本身已有参数 `_version` lazy invalidation；post-step hook 是 eager refresh，用于减少下一次 forward/backward 的 cache refresh 抖动。

匹配规则：

- `target_modules=None`：替换所有 `torch.nn.Linear`。
- `target_modules=("q_proj",)`：匹配完整路径、子模块名或完整路径后缀。
- `exclude_modules` 使用同样的匹配规则，优先排除。
- `config_overrides` 使用同样匹配规则，第一条匹配生效；用于对敏感层单独提高 rank、切换 init、关闭实验 fusion 或改 residual branch。

参数：

- `weight`：CUDA 上的 FP16/BF16 dense 权重，构造时量化为 frozen FP4 backbone。
- `bias`：可选。默认作为 frozen buffer；`train_bias=True` 时作为可训练参数。
- `rank`：LoRA rank，会向上补齐到 16 的倍数，便于后续复用 Nunchaku 低秩 packed layout。
- `lora_alpha`：默认等于补齐后的 rank，因此默认 `scaling=1`。
- `lowrank_dtype`：`torch.bfloat16` 或 `torch.float16`。
- `init`：
  - `zero`：标准 LoRA 零效果初始化，`A` Kaiming，`B` 为 0。
  - `gaussian`：`A/B` 都用小方差正态，用于 correctness/压力测试。
  - `residual_svd`：用 `W0 - dequant(Q4(W0))` 的低秩 SVD 初始化 LoRA，贴近 SVDQuant residual branch。
- `frozen_residual_rank/frozen_residual_init`：
  - `0/"none"`：只使用 trainable task LoRA。
  - `rank/"residual_svd"`：额外构造 frozen residual branch，推荐与 `init="zero"` 搭配，避免训练破坏量化补偿。
- `residual_svd_method`：
  - `full_svd`：默认，初始化误差最低。
  - `svd_lowrank`：使用 `torch.svd_lowrank`，通过 `residual_svd_lowrank_oversample/niter` 控制随机低秩 SVD；适合大模型批量转换。
- `cache_lora_act`：是否保存 forward 的 `x @ A.T`，避免 backward 计算 `dB` 时重算。
- `activation_checkpoint`：逐 `NunchakuFP4LoRALinear` 的局部 checkpoint。它只省该算子内部 saved tensors；要显著降低多层输入 activation，应该在 transformer block/segment 外层做 checkpoint。
- `fuse_lowrank_forward`：opt-in forward 消融选项。当 `lowrank_dtype == weight dtype` 时走 Nunchaku 原生 `quantize_w4a4_act_fuse_lora + gemm_w4a4` low-rank epilogue，把 trainable task LoRA forward 并入 FP4 主分支；有 frozen residual 时，residual 仍作为 dense side branch 追加。该路径会改变低秩分支的累加/量化侧调度，验证脚本用 `native_fused_forward` 标记并用 `5e-4` rel_l2 tolerance 报告相对当前 BF16/FP16 调度公式的差异；默认关闭。
- `zero_lora_up_fast_path`：默认开启的 zero-init 首步优化。仅当 `init="zero"` 后 `lora_up` 的 parameter version 仍匹配初始化零张量时触发；forward 跳过 LoRA out/native low-rank epilogue，backward 跳过 LoRA dX 和 `dA`，只计算 `dB=dY.T@(x@A.T)`。初始 cache refresh 不生成 packed LoRA forward/dX cache；`optimizer.step()` 或 adapter load 改变版本后自动回到常规路径。若 `overlap_lora_grad=True` 且行数达到门槛，则 zero-up backward 会把 FP4 main dX、`dB` 和可选 frozen residual dX 分流到多 CUDA stream。
- `fuse_frozen_residual_dx`：FP16-only 实验选项，把 task LoRA 和 frozen residual 的 dX 合并为同一个 packed low-rank epilogue。BF16 下同一路径目前 `dX` rel_l2 约 `2.1e-3`，不作为默认。
- `target_modules/exclude_modules/config_overrides`：模型级替换时用于控制哪些 Linear 进入 FP4 LoRA，以及每个 projection 的 rank/init/fusion/residual 策略。

## 当前 backward 边界

P0 backward 的计算拆分如下：

```text
dX_main = NunchakuFP4BackwardDXOp(dY)
dy_res  = dY @ R_up
res_dX  = dy_res @ R_down
dy_up   = dY @ B
dX      = dX_main + res_dX + scaling * dy_up @ A
dB      = scaling * dY.T @ lora_act
dA      = scaling * dy_up.T @ x
```

其中：

- `dX_main` 调 CUDA FP4 kernel。
- `dX_residual/dX_lora/dA/dB` 暂时调 PyTorch matmul。
- 当前没有训练 FP4 backbone，也不会产生 backbone `dW`。
- frozen residual branch 是 buffer，不进入 optimizer 参数组，也不进入 LoRA-only adapter checkpoint。
- 当前默认不预存 backward packed weight；沿用 transient CUDA repack，符合“不让常驻内存乘 2”的约束。
- `NunchakuFP4LoRALinear` 会让 forward/backward op 共享同一份 resident `qweight/wscales`，只额外保存 backward scale metadata。

## 当前性能证据

新增 benchmark：

```bash
conda run -n triton python benchmarks/benchmark_native_fp4_lora_training.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --warmup 5 \
  --iters 10 \
  --grad-accum-steps 4 \
  --backward-weight-policy repack
```

RTX 5090 短测，`M=N=K=4096, rank=32`：

| dtype | dense train step ms | FP4 dense-dX step ms | FP4 fused-dX dynamic-pack ms | FP4 fused-dX cached-pack ms | cached+refresh ms | cached-pack speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.0159 | 0.9464 | 0.9148 | 0.8917 | 0.9058 | 2.261x |
| FP16 | 1.7440 | 0.9009 | 0.8816 | 0.8708 | 0.8841 | 2.003x |

Gradient accumulation 短测，`grad_accum_steps=4, warmup=5, iters=10`：

| dtype | dense per micro-step ms | FP4 dynamic-pack per micro-step ms | FP4 cached-pack per micro-step ms | cached-pack vs dense |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 3.5701 | 1.5405 | 1.4255 | 2.504x |
| FP16 | 3.5335 | 1.4621 | 1.5103 | 2.340x |

结论：

- P0/P1 接口已经能在典型 4096 线性层上给出约 `1.9x-2.2x` 的训练 step 加速。
- `fuse_lora_dx=True`：将 `dX_lora = (dY @ B) @ A` 的第二段放进 FP4 dX epilogue。
- `cache_fused_lora_dx=True`：只缓存 LoRA packed A/B，额外内存约 `rank * (in + out)`，不缓存第二份 FP4 backbone；参数 version 变化时自动刷新。
- `zero_lora_up_fast_path=True`：zero-init task LoRA 首步只保留 `dB`，跳过 LoRA out、LoRA dX、`dA` 和初始 packed LoRA cache。RTX 5090 4096/rank32 BF16 短测中，fused dX cached-pack train step `0.9058 -> 0.7439ms`（`1.218x`），throughput fused-forward+fused-DX `0.8768 -> 0.7994ms`（`1.097x`）。打开 zero-up overlap 后，fused dX cached-pack `0.8371 -> 0.7254ms`（`1.154x` vs disabled fast path），throughput fused-forward+fused-DX `0.7713 -> 0.7260ms`（`1.062x`）。`lora_up` 更新后 fast path 自动失效，post-step hook 再刷新 packed cache。
- `backward_weight_policy="repack"`：默认每次 backward transient repack `W^T` 的 packed FP4 权重，只预存转置后的 scale，不常驻第二份 backbone。`benchmark_native_fp4_lora_training.py --backward-weight-policy cache` 会把该策略接入所有训练变体并报告 `backward_weight_cache_bytes`；`"cache"` 是 memory-budget opt-in，常驻一份 compressed backward qweight。RTX 5090 4096/rank32 BF16 短测中 train step `1.056x`、4-step accumulation `1.050x` vs repack，额外 cache 为 dense BF16 weight 的 `25%`。
- `fp4_activation_cache_d_lora_down=True`：forward 保存主分支已有 `qact + ascales` 而不是 BF16/FP16 `x`。`fp4_activation_cache_min_rows=0` 保持旧行为；设成大于 0 后，低于该 flattened row 数的 forward 自动回 exact saved-x `dA`，用于避免短序列/小 batch 的近似误差和额外开销。`fp4_activation_cache_d_lora_down_backend="fused"` 直接用 fused CUDA kernel 从 FP4 cache 算 `dA`，避免 dense `x_hat`；`"dequant_gemm"` 先反量化出 dense `x_hat` 再用 torch GEMM，当前更快但 transient 显存更高。这是显存/近似训练模式，要求 `cache_lora_act=True`，当前不支持 `overlap_lora_grad` 或 `reuse_fused_dy_up_for_d_lora_down`。
- BF16 单步下 cached-pack fused dX 相比 dynamic-pack fused dX 快 `1.026x`；每步刷新 cache 后仍快 `1.010x`。FP16 单步下 cached-pack 约 `1.012x`，每步刷新后基本持平。
- Gradient accumulation 会摊薄 cache refresh 开销；accumulation benchmark 对测量顺序更敏感，默认策略仍应以真实训练循环为准。
- 为保证训练梯度精度，默认 `dA` 仍使用 dense `dY @ B`。`reuse_fused_dy_up_for_d_lora_down=True` 是 opt-in：FP16 复用 decoded packed `dY @ B`，`dA` rel_l2 约 `3.35e-5`；BF16 复用 dual quantize 的 dense `dy_up` 输出，`dA` rel_l2 为 `0`。
- 4096/rank32 BF16 短测中，reuse 相对 cached-pack 单步 `1.014x`，reuse+overlap 相对 reuse 单步 `1.032x`，backward estimate 相对 dense `2.086x`。FP16 两次短测中，单步相对 cached-pack 约 `0.968x-1.018x`，梯度累积 per micro-step 约 `1.016x-1.036x`。
- `overlap_lora_grad=True` 要求同时打开 `fuse_lora_dx=True` 和 `cache_fused_lora_dx=True`；实现上用多 CUDA stream 重叠 transient FP4 repack、fused dX、`dB` GEMM 和 `dA` GEMM。
- `overlap_lora_grad_min_rows=4096` 是默认 auto gate：小于该 flattened row 数时自动回落到 sequential cached fused-dX 路径，避免 2048 形状上多 stream 调度变慢；设为 `0` 可强制 always-overlap 做消融。
- `NUNCHAKU_FP4_LORA_CACHE_OVERLAP_RESOURCES=1` 是 stream/event 资源复用消融开关，默认关闭。RTX 5090 上 1024 forced-overlap 约 `1.01x`，4096 主形状约 `0.974x`，说明复用 Python `Stream/Event` 对象不是稳定收益；默认仍在 helper 内临时创建资源。
- BF16 exact overlap 默认不复用 packed 近似 `dY @ B`，`dA` 仍走 dense `dY @ B`。Correctness：`validate_native_fp4_lora_training.py --m 257 --in-features 3072 --out-features 3584 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --overlap-lora-grad --overlap-lora-grad-min-rows 0` 通过，`dX` rel_l2 `1.55e-4`，`dA` rel_l2 `0`。如果同时打开 `reuse_fused_dy_up_for_d_lora_down=True` 且不使用 frozen residual，BF16 使用 dual dense `dy_up`，强制 overlap correctness 同样通过，`dA` rel_l2 `0`。
- `benchmark_native_fp4_lora_training_breakdown.py` 现在额外报告 `dy_read_model` 和 overlap 子图 correctness。4096/rank32 BF16 短测中，exact current 约等价于 3 次读 `dY`，reuse 路径约 2 次读 `dY`，理想大融合 kernel 才能降到 1 次；exact overlap 子图 `0.2595ms`、cached fused dX 单独 `0.2158ms`、`dA/dB` exact rel_l2 为 0，reuse `dA` rel_l2 约 `5.75e-4`。
- BF16 FP4 activation-cache `dA` 已接入 `NunchakuFP4LoRALinear`：`validate_native_fp4_lora_training.py --m 129 --in-features 512 --out-features 768 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --fp4-activation-cache-d-lora-down` 通过；`--fp4-activation-cache-d-lora-down-backend fused/dequant_gemm` 两条路径均通过。`dA` 对齐 FP4-cache reference，FP4-cache reference 相对 exact saved-x `dA` rel_l2 约 `9.7e-2`。
- BF16 frozen residual exact overlap 已支持：task LoRA dX 走 fused epilogue，frozen residual dX 保持 dense side stream；`validate_native_fp4_lora_training.py --m 129 --in-features 512 --out-features 768 --rank 32 --frozen-residual-rank 32 --frozen-residual-init residual_svd --init zero --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --overlap-lora-grad` 通过，forward/dX/LoRA A/B/bias grad rel_l2 全为 `0`。
- `residual_svd_method="svd_lowrank"` 已接入单层和模型级接口：`validate_native_fp4_lora_training.py --frozen-residual-init residual_svd --init zero --residual-svd-method svd_lowrank ...` 和 `validate_native_fp4_lora_modeling.py --frozen-residual-init residual_svd --init zero --residual-svd-method svd_lowrank ...` 均通过。
- trainable `init="residual_svd"` 使用 dense LoRA dX 时 correctness 通过；若同时打开 `fuse_lora_dx=True`，BF16 下较大的 residual factors 会把 fused LoRA dX 近似误差放大到约 `2e-3`，因此推荐把 residual_svd 作为 frozen residual branch，task LoRA 仍 zero-init。
- 4096/rank32 BF16 短测，`benchmark_native_fp4_lora_training.py --warmup 10 --iters 30 --grad-accum-steps 4`：cached-pack `0.9314 ms`，exact overlap `0.8830 ms`；gradient accumulation per micro-step `0.9594 -> 0.8956 ms`。单步 `1.055x` vs cached-pack，grad accumulation `1.071x` vs cached-pack。
- FP16 reuse+overlap 路径同时打开 `reuse_fused_dy_up_for_d_lora_down=True`，复用 decoded packed `dY @ B`。Correctness：`dX` rel_l2 `1.81e-5`，`dA` rel_l2 `3.56e-5`；4096/rank32 短测中单步 `1.008x` vs reuse，grad accumulation `1.037x` vs reuse。
- reuse-based overlap 仍不支持 frozen residual branch；高层 `fp4_lora_finetune_config` 遇到 reuse + frozen residual 会自动关闭 overlap，保留顺序 dense residual dX。
- 保存 forward `lora_act` 对大形状有小幅收益，约 `3%-4%`；是否默认缓存要结合训练显存预算决定。

初始化消融，RTX 5090 BF16，`benchmark_fp4_lora_initialization.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --warmup 5 --iters 10`：

| policy | residual SVD method | forward rel_l2 vs dense | error reduction vs zero | construct s | train step ms |
| --- | --- | ---: | ---: | ---: | ---: |
| FP4 + zero LoRA | none | 1.4377 | 1.00x | 0.0834 | 0.4203 |
| trainable residual_svd LoRA | full_svd | 1.3943 | 1.031x | 0.1628 | 0.3021 |
| frozen residual_svd + zero LoRA | full_svd | 1.3943 | 1.031x | 0.1547 | 0.3190 |
| trainable residual_svd LoRA | svd_lowrank | 1.4013 | 1.026x | 0.0720 | 0.2654 |
| frozen residual_svd + zero LoRA | svd_lowrank | 1.4013 | 1.026x | 0.0042 | 0.3188 |

单层微调收敛验证，RTX 5090 BF16，`validate_fp4_lora_finetune_convergence.py` 默认配置：

| target base | initial loss | final loss | final / initial | final vs target rel_l2 | fitted delta rel_l2 | checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP4 initial + teacher low-rank delta | 1.0753e-2 | 2.9362e-6 | 2.7307e-4 | 3.5572e-3 | 1.6525e-2 | pass |

这个实验对应 personal-vault 中的“单层微调 loss 收敛曲线”待办。默认 `target_base=fp4_initial`，目标是初始 `FP4 + frozen residual` 输出加 teacher low-rank delta，避免把高秩量化误差混进 task LoRA 的低秩拟合目标。验证通过项包括 loss 显著下降、LoRA A/B 参数发生更新、梯度 finite、frozen residual buffer 不变、只有预期参数可训练，以及动态 packed cache 的 optimizer post-step refresh hook 已运行。

## 后续优化路线

### Breakdown 证据

新增 breakdown benchmark：

```bash
conda run -n triton python benchmarks/benchmark_native_fp4_lora_training_breakdown.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --backward-weight-policy repack \
  --warmup 5 \
  --iters 10
```

新增 low-rank grad 子图 benchmark：

```bash
conda run -n triton python benchmarks/benchmark_fp4_lora_lowrank_grad.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --ranks 16,32,64,128 \
  --dtype bf16 \
  --warmup 5 \
  --iters 10
```

新增 FP4 dX pipeline benchmark：

```bash
conda run -n triton python benchmarks/benchmark_fp4_dx_pipeline.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --warmup 5 \
  --iters 10
```

RTX 5090 短测，`M=N=K=4096, rank=32`：

| dtype | backward estimate ms | fused dX cached-pack ms | dense LoRA grad pair ms | LoRA grad share | LoRA pack refresh ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.6195 | 0.2152 | 0.0761 | 12.3% | 0.0388 |
| FP16 | 0.6467 | 0.2275 | 0.0396 | 6.1% | 0.0128 |

低秩梯度子图短测，`M=N=K=4096, rank=32`：

| dtype | sequential dA+dB ms | reuse existing dy_up ms | reuse speedup | two-stream overlap ms | overlap vs sequential |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.0855 | 0.0623 | 1.37x | 0.0928 | 0.92x |
| FP16 | 0.0536 | 0.0390 | 1.37x | 0.0748 | 0.72x |

`dB=dY.T@lora_act` 的 scalar-reduction CUDA 原型已加入 `native_fp4.lora_up_grad` 作为消融，但不进默认训练路径。4096/rank32 BF16 短测为 `0.488ms` vs torch GEMM `0.0295ms`（`0.06x`），FP16 为 `0.493ms` vs `0.0226ms`（`0.046x`）。结论是普通 block reduction 明显不如 cuBLAS/Tensor Core；低秩梯度优化不应继续沿这个方向做。

FP4 dX pipeline 短测，`M=N=K=4096, rank=32`：

| dtype | full dX ms | quantize dY ms | repack W^T ms | prequantized GEMM ms | cached-qweight upper bound | fused LoRA dX ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.1837 | 0.0365 | 0.0390 | 0.1068 | 1.33x | 0.2157 |
| FP16 | 0.1858 | 0.0360 | 0.0391 | 0.1072 | 1.34x | 0.2281 |

直接结论：

- `dA/dB` 目前不是最大瓶颈；低秩梯度专用 kernel 的收益上限有限。
- 单独给 `dA/dB` 加 CUDA stream overlap 在 5090 上反而变慢；后续低秩梯度 kernel 应优先围绕复用/消除 `dy_up` 中间量设计。
- FP4 dX 主路径中 prequantized GEMM 占比最大，`dY` quantize 和 transient repack 各约 20%。预存 `W^T` 的 cached-qweight 消融上界只有约 `1.33x-1.34x`，且会带来第二份 backbone 内存，不作为默认方案。
- `benchmark_native_fp4_lora_training_breakdown.py --backward-weight-policy cache` 会保留强制 transient `repack_backbone` 指标，并额外报告 `backward_qweight_policy_access`、`refresh_backward_qweight_cache` 和常驻 qweight 字节，用于判断 cache hit 与预热成本。
- LoRA pack refresh 已经换成 native CUDA layout pack；相对旧 PyTorch `pad + permute + contiguous` 路径，4096/rank32 短测约减少一半。

### Native FP4 backward repack micro-optimization

新增微优化：

- `csrc/fp4_repack_cuda.cu`
- 将每个 32-bit 输出 word 内重复使用的 backward scale load、zero check 和固定 forward scale group 计算移到 8 元素循环外。
- 不保存第二份 transposed FP4 backbone；仍然每次 backward transient repack。
- 高层训练接口新增 `backward_weight_policy`：默认 `"repack"` 保持上述行为；`"cache"` 显式常驻 compressed backward qweight，用于量化 repack 开销上限，不作为默认策略。
- `validate_native_fp4_backward.py --m 256 --in-features 4096 --out-features 4096 --rank 32 --dtype fp16` 通过，`qweight_bwd_cuda_matches_reference=true`，repack 输出 bitwise exact。

RTX 5090 短测，`benchmark_native_fp4_lora_training_breakdown.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10`：

| metric | before ms | after ms | speedup |
| --- | ---: | ---: | ---: |
| `repack_backbone` | 0.0424 | 0.0391 | 1.084x |
| `fp4_dx_main` | 0.1868 | 0.1839 | 1.016x |
| `fused_dx_cached_pack` | 0.2175 | 0.2139 | 1.017x |
| `full_backward_minus_forward` | 0.6703 | 0.6561 | 1.022x |

### Native LoRA pack

新增：

- `csrc/fp4_lora_pack_cuda.cu`
- `_fp4_native_cuda.pack_lowrank_weight`
- `benchmarks/validate_native_fp4_lora_pack.py`

作用：

- 直接把 BF16/FP16 LoRA A/B 写成 Nunchaku fused dX epilogue 使用的 packed layout。
- 替代原来的 PyTorch `pad + view + permute + contiguous` refresh 路径。
- 不缓存第二份 transposed FP4 backbone，只降低 trainable LoRA packed cache 的刷新成本。

RTX 5090 短测，`validate_native_fp4_lora_pack.py --dtype bf16/fp16 --warmup 20 --iters 100`：

| dtype | typical native pack ms | typical torch pack ms | native speedup |
| --- | ---: | ---: | ---: |
| BF16 | 0.0058-0.0063 | 0.0195-0.0212 | 3.3x-3.6x |
| FP16 | 0.0058-0.0060 | 0.0196-0.0212 | 3.3x-3.6x |

P1：optimizer post-step eager refresh 接口已落地；后续需要在多层真实模型里继续测量它对 step latency 抖动和梯度累积的收益。

P2：BF16 下 packed `dY @ B` decode 精度问题已有可用绕法：`reuse_fused_dy_up_for_d_lora_down=True` 在 BF16 使用 dual quantize 输出 dense `dy_up`，`dA` 与手写 BF16 matmul 对齐为 `rel_l2=0`；FP16 仍可用 decoded packed 路径作为小误差 opt-in。后续只需继续评估真实训练 loop 里的收益和是否值得默认开启。

P3：把 `dA/dB` 的低秩 GEMM 改成小 rank 专用 CUDA kernel，减少 PyTorch kernel launch 和中间张量开销。`benchmark_fp4_lora_lowrank_grad.py` 已给出基线：rank32 下复用已有 `dy_up` 可让低秩梯度 pair 快约 `1.37x`，但单独双 stream overlap 会退化；新增 scalar-reduction `lora_up_grad` 原型在 rank32 BF16/FP16 都显著慢于 torch GEMM。因此下一版 kernel 不应做普通 block reduction，而应尝试把 `dy_up=dY@B` 与 `dA=dy_up^T@X` 的中间量复用/压缩、做 CUTLASS grouped/specialized small-N GEMM，或把 `dB` 规约融合进 FP4 dX 的 `dY` 读取路径。

P4：加入 activation cache policy：

- `save_bf16`：保存 BF16 `x` 和 `lora_act`，速度优先。
- `recompute_lora_act`：只保存 `x`，少存一个 `[M, rank]`。
- `save_fp4_cache`：保存 forward FP4 主分支已经生成的 `qact + ascales`，不保存 BF16/FP16 `x`。这能显著省 activation cache，但 `dA=(dY@B).T@x` 会变成近似梯度。当前训练接口同时提供 `dequant_gemm` 和 `fused` 两个 backend：前者会重新物化 dense `x_hat`、速度更好；后者避免中间张量、峰值显存更低但仍需继续优化。
- `checkpoint`：进一步重算上游 activation，面向长序列微调。实测逐 Linear checkpoint 只省内部 `lora_act`，收益很小；应按 transformer block/segment 做 checkpoint。

RTX 5090 短测，`benchmark_fp4_lora_activation_checkpoint.py --batch 512 --hidden 1024 --layers 4 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --warmup 5 --iters 10`：

| checkpoint scope | intermediate activation | train step ms | peak delta reduction | conclusion |
| --- | --- | ---: | ---: | --- |
| module | none | 2.0313 | 1.2% | 逐 Linear checkpoint 基本只省内部 `lora_act`，不推荐默认开启 |
| stack/block | none | 1.6118 | 9.9% | 能省跨层输入 activation，但要重算整段 forward |
| module | silu | 2.0873 | 1.0% | 非线性本身仍保存激活，逐 Linear checkpoint 收益更低 |
| stack/block | silu | 1.6642 | 7.6% | 推荐作为长序列/多层微调的显存模式，按 block 粒度打开 |

正确性：`module` 和 `stack` checkpoint 的 forward、`dX`、首尾 LoRA A/B 梯度与 no-checkpoint reference 均为 `rel_l2=0`。

`save_fp4_cache` 消融已落地：

- 新增 `native_fp4.layout.dequantize_fp4_activation(qact, ascales, return_scales=False)`；CUDA kernel 直接按 Nunchaku `uint4` activation layout 和 FP8 scale layout 反量化，和 Torch fallback 对齐到 `rel_l2=0`。
- 新增 `native_fp4.fp4_activation_cache_lora_down_grad(qact, ascales, dy_up, in_features)`；CUDA kernel 直接从 native FP4 activation cache 计算 `dA`，避免 backward 物化 dense `x_hat`。
- `NunchakuFP4LoRALinear(..., fp4_activation_cache_d_lora_down_backend=...)` 已支持 `"fused"` 与 `"dequant_gemm"` 两种 backend；`FP4LoRAConfig`、`fp4_lora_finetune_config` 和 `prepare_fp4_lora_finetuning` 同步暴露该字段。
- 新增 `benchmark_fp4_lora_activation_cache_policy.py`，比较 saved BF16/FP16 `x` 与 FP4 activation cache 对 `dA` 的显存、速度和精度影响。
- 新增 `benchmark_fp4_lora_saved_tensors.py`，用 `torch.autograd.graph.saved_tensors_hooks` 直接检查训练 wrapper 的 autograd context，确认 `fp4_activation_cache_d_lora_down=True` 不再保存 BF16/FP16 `x`。
- `NunchakuFP4LoRALinear(fp4_activation_cache_d_lora_down=True)` 已接入该模式：forward 不把 BF16/FP16 `x` 存入 autograd context，而是保存 `qact + ascales`；`dB` 仍依赖 `cache_lora_act=True` 保存的 LoRA activation。`fp4_activation_cache_min_rows` 可在运行时按 row 数门控，小 row 数自动保存原始 `x` 并走 exact `dA`。

RTX 5090 BF16 短测：

| shape | saved x cache | FP4 cache | memory reduction | x_hat rel_l2 | dA rel_l2 | saved-x dA ms | FP4 dequant+dA ms | fused dA ms | fused vs dequant rel_l2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048^2, rank32 | 8.00 MiB | 2.25 MiB | 3.56x | 9.78e-2 | 9.82e-2 | 0.0248 | 0.0680 | 0.1126 | 2.86e-3 |
| 4096^2, rank32 | 32.00 MiB | 9.00 MiB | 3.56x | 9.78e-2 | 9.68e-2 | 0.0366 | 0.2018 | 0.3060 | 7.84e-5 |
| 2048^2, rank64 | 8.00 MiB | 2.25 MiB | 3.56x | 9.78e-2 | 9.82e-2 | 0.0276 | 0.0677 | 0.3206 | 3.38e-5 |
| 4096^2, rank64 | 32.00 MiB | 9.00 MiB | 3.56x | 9.78e-2 | 9.79e-2 | 0.0539 | 0.2201 | 0.9876 | 8.02e-5 |
| 2048^2, rank128 | 8.00 MiB | 2.25 MiB | 3.56x | 9.78e-2 | 9.79e-2 | 0.0209 | 0.0641 | 0.5633 | 2.63e-3 |
| 4096^2, rank128 | 32.00 MiB | 9.00 MiB | 3.56x | 9.78e-2 | 9.79e-2 | 0.0354 | 0.2017 | 1.8640 | 2.63e-3 |
| 2048^2, rank256 | 8.00 MiB | 2.25 MiB | 3.56x | 9.78e-2 | 9.79e-2 | 0.0510 | 0.0924 | 1.0643 | 4.45e-5 |
| 4096^2, rank256 | 32.00 MiB | 9.00 MiB | 3.56x | 9.78e-2 | 9.79e-2 | 0.0684 | 0.2264 | 3.6840 | 2.87e-3 |
| 2048^2, rank512 | 8.00 MiB | 2.25 MiB | 3.56x | 9.78e-2 | 9.79e-2 | 0.0484 | 0.0893 | 2.0542 | 2.86e-3 |
| 4096^2, rank512 | 32.00 MiB | 9.00 MiB | 3.56x | 9.78e-2 | 9.78e-2 | 0.1084 | 0.2694 | 7.2254 | 7.43e-5 |

训练接口接入后的短测，`benchmark_native_fp4_lora_training.py --warmup 5 --iters 10 --grad-accum-steps 4`，BF16，`fuse_lora_dx=True, cache_fused_lora_dx=True`：

| shape | exact cached-pack step ms | FP4-cache dA step ms | FP4-cache / exact | exact grad-accum per micro-step ms | FP4-cache grad-accum per micro-step ms | saved x -> FP4 cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048^2, rank32 | 0.2635 | 0.3624 | 0.73x | 0.2409 | 0.3443 | 8.00 MiB -> 2.25 MiB |
| 4096^2, rank32 | 0.9499 | 1.3145 | 0.72x | 0.9724 | 1.3182 | 32.00 MiB -> 9.00 MiB |

训练接口实际 saved tensor 短测，`benchmark_fp4_lora_saved_tensors.py --warmup 5 --iters 10`，BF16，`fuse_lora_dx=True, cache_fused_lora_dx=True`：

| shape | exact activation context | FP4-cache activation context | context reduction | exact all saved | FP4-cache all saved | all-saved reduction | FP4-cache / exact step | dA rel_l2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048^2, rank32 | 8.125 MiB | 2.375 MiB | 3.42x | 8.375 MiB | 2.625 MiB | 3.19x | 0.71x | 9.83e-2 |
| 4096^2, rank32 | 32.25 MiB | 9.25 MiB | 3.49x | 32.75 MiB | 9.75 MiB | 3.36x | 0.77x | 9.78e-2 |

这里 `activation context = saved_x/qact/ascales + saved_lora_act`，`all saved` 额外包含 LoRA A/B 等权重引用。这个口径比只看 `x -> qact+ascales` 更接近 PyTorch autograd 真实保存量。

结论：

- FP4 cache 的显存收益明确，`qact + ascales` 约为 saved BF16/FP16 `x` 的 `28.1%`；训练接口实测 activation context 缩减约 `3.42x-3.49x`，all saved tensors 缩减约 `3.19x-3.36x`。
- 直接用 FP4-dequant activation 计算 `dA` 会带来约 `1e-1` rel_l2 梯度误差；如果要保持 LoRA 梯度精确，默认仍应使用 `save_bf16` 或重算 `x` 来源。
- 当前 naive CUDA dequant 仍要物化 dense `x_hat`，4096 形状比 saved-x `dA` 慢约 `5.1x`。
- fused `dA` 原型避免了 dense `x_hat`，rank<=32 当前使用 `kVec=4,rVec=16,threads=128`，rank<=512 使用 `kVec=3,rVec=32,threads=128`，rank>512 仍回落 `kVec=2,rVec=16`。本轮候选记录见 `docs/fp4_kernel_research_notes.md`：rank32 的 4096 fused `dA` 从约 `0.391ms` 降到 `0.306ms`，约 `1.28x`；rank64 从约 `1.50ms` 降到 `0.99ms`，约 `1.52x`；rank128 从约 `3.34ms` 降到 `1.86ms`，约 `1.79x`；rank256 从约 `5.72ms` 降到 `3.68ms`，约 `1.55x`；rank512 从约 `11.54ms` 降到 `7.23ms`，约 `1.60x`。
- 它仍慢于 `dequant + GEMM`，4096/rank32 约 `0.58x`，4096/rank64 约 `0.22x`，4096/rank128 约 `0.11x`，4096/rank256 约 `0.06x`，4096/rank512 约 `0.04x`。本轮 5090 复测 4096/rank64：`dequant_gemm=0.2142ms`，`fused=0.9916ms`。下一步应把 FP4 decode staging 和 reduction 改成更 tensor-core/GEMM 友好的分块。
- 接入训练 wrapper 后，该模式用约 `0.77x` exact cached-pack step 速度换取 saved `x` 约 `3.56x` 缓存压缩；它当前是显存压力模式，不是性能模式。
- 即便 fused `dA` 继续优化，它对齐的仍是 `dequant(qact, ascales)` 近似路径，不能消除 FP4 activation cache 自身带来的 `dA` 精度损失，因此只适合作为显存/近似训练模式。

P5：dual-branch residual/task LoRA 初始化已落地。FP16 下 `fuse_frozen_residual_dx=True` 可以把 frozen residual dX 与 task LoRA dX 一并打包进 fused epilogue；BF16 下该 packed residual dX 路径误差偏大，默认仍保留 residual dense dX。BF16 exact overlap 支持 dual-branch，但 residual dX 保持 dense side stream。

P5.1：`fuse_lowrank_forward=True` 已作为 opt-in forward 消融路径加入。当前实现：

- task LoRA branch：当 `lowrank_dtype == weight dtype` 时，直接复用 Nunchaku 原生 `quantize_w4a4_act_fuse_lora` 生成 packed LoRA activation，再用 `gemm_w4a4` low-rank epilogue 输出 `FP4(W0) + scale * LoRA(x)`。
- frozen residual branch：如果存在 frozen residual，仍保持 dense side branch，在 native `FP4+task LoRA` 输出后追加；不把 frozen residual 打进同一个 forward epilogue。

由于 native 路径在量化 kernel 内生成 task LoRA activation，累加顺序和精确 `x @ A.T @ B.T` 不同，验证脚本用 `native_fused_forward` 标记该路径，并以 `5e-4` rel_l2 tolerance 报告相对当前 BF16/FP16 调度公式的误差。

RTX 5090 短测，`benchmark_native_fp4_lora_training.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10 --grad-accum-steps 4`：

| metric | default FP4+dense LoRA | native fused forward | result |
| --- | ---: | ---: | --- |
| forward inference ms | 0.3458 | 0.2723 | `1.270x` vs default |
| forward train graph ms | 0.3462 | 0.2711 | `1.277x` vs default |
| train step ms | 0.9963 | 0.9268 | `1.075x` vs default |
| forward rel_l2 vs default | - | `1.55e-4` | within `5e-4` |
| forward cache pack ms | - | 0.0232 | LoRA A/B packed cache refresh |

RTX 5090 短测，`benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --warmup 10 --iters 30`：

| path | dtype | train step ms | result |
| --- | --- | ---: | --- |
| task LoRA only, fused dX | bf16 | 0.2997 | task-only reference |
| dual branch, residual dense dX | bf16 | 0.3077 | `1.027x` overhead vs task-only；BF16 fused residual dX 暂不启用 |
| dual branch, residual exact overlap auto | bf16 | 0.3086 | `0.997x` vs dense residual；默认门槛回落，不再退化 |
| dual branch, residual forced overlap | bf16 | 0.4699 | `0.666x` vs dense residual；`--overlap-lora-grad-min-rows 0` 消融 |
| task LoRA only, fused dX | fp16 | 0.2712 | task-only reference |
| dual branch, residual dense dX | fp16 | 0.3128 | `1.153x` overhead vs task-only |
| dual branch, residual exact overlap auto | fp16 | 0.3115 | `1.004x` vs dense residual；默认门槛回落 |
| dual branch, residual fused dX | fp16 | 0.3130 | `0.999x` vs residual dense dX，`dX` rel_l2 `3.80e-4` |

RTX 5090 短测，`benchmark_native_fp4_lora_dual_branch.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --frozen-residual-rank 32 --dtype bf16 --warmup 10 --iters 30`：

| path | dtype | train step ms | result |
| --- | --- | ---: | --- |
| task LoRA only, fused dX | bf16 | 0.9590 | task-only reference |
| dual branch, residual dense dX | bf16 | 1.1883 | `1.239x` overhead vs task-only |
| dual branch, residual exact overlap auto | bf16 | 1.0961 | `1.084x` vs dense residual，`dX` rel_l2 `9.08e-7` |

`fuse_lowrank_forward=True` 消融，RTX 5090 同形状 `warmup=5,iters=10`：

| path | dtype | train step ms | result |
| --- | --- | ---: | --- |
| dual branch, residual dense dX | bf16 | 1.2076 | default reference |
| dual branch, residual dense dX + native task forward | bf16 | 1.1048 | `1.093x` vs default dual |
| dual branch, residual exact overlap auto | bf16 | 1.1005 | default overlap reference |
| dual branch, residual exact overlap auto + native task forward | bf16 | 1.0122 | `1.087x` vs default overlap |

因此当前仍不把 `fuse_lowrank_forward` 设为 balanced 默认，但 throughput preset 会打开它。native task forward 对 forward 和 4096 dual-branch train step 都有明确收益；代价是额外一份 packed LoRA forward cache，以及 task LoRA activation/`dB` 约 `1e-4` 级 rel_l2 差异。实际训练 step 的主要剩余瓶颈仍偏向 dX 主路径、LoRA 参数梯度和 residual dX 融合。

P5.2：zero-init `lora_up` 首步 fast path 已落地。实现要点：

- `NunchakuFP4LoRALinear` 记录初始化后 `lora_up._version`，只在 version 匹配时启用 fast path；不在热路径做 `count_nonzero`。
- fast path 下 forward 只算 FP4 main 和可选 frozen residual，仍按需保存精确 dense `lora_act=x@A.T` 供 `dB` 使用；backward 只算 FP4 main dX、可选 residual dX、`dB`，并返回零 `dA`。
- 若 `overlap_lora_grad=True` 且 rows >= `overlap_lora_grad_min_rows`，zero-up backward 使用专用 overlap helper 并行 FP4 main dX、`dB` 和 residual dX；不依赖 packed LoRA dX cache，也不新增 resident memory。
- overlap helper 的 stream/event 资源复用已做消融并默认关闭：4096 主形状下 cache 约 `0.974x`，不值得作为默认路径。
- `refresh_fused_lora_forward_caches` 和 `refresh_fused_lora_dx_caches` 在初始 zero-up 状态不生成 packed LoRA cache，`prepare(..., refresh_caches=True)` 的 cache summary 因此真实反映 resident bytes 为 0；optimizer post-step hook 会在 `lora_up` 更新后恢复常规 cache refresh。
- `load_fp4_lora_state_dict` / `load_fp4_lora_peft_state_dict` 会清除 zero-up 标记，避免外部 adapter 被误判为初始化零状态。
- Correctness 覆盖 active zero-up、fused dX、fused forward、FP4 activation-cache dA、frozen residual 和非 zero-init gaussian 回归。新增 `benchmark_fp4_lora_zero_fast_path.py` 输出 `latest_fp4_lora_zero_fast_path.json`。

P6：加入 outlier-aware FP4 训练策略：

- `analyze_fp4_lora_activation_grad_outliers.py` 已落地 activation / grad-output 通道统计。
- `summary.rank_bump_candidates` 可用 `fp4_lora_config_overrides_from_outlier_report` 直接转成 `config_overrides`，对敏感 projection 单独提高 rank 或调整 residual/task LoRA 策略。
- Llama module sensitivity scan 可用 `fp4_lora_sensitivity_policy_from_report` 或 `prepare_fp4_lora_finetuning(..., sensitivity_report=...)` 直接转成 rank bump / BF16 exclude 策略。
- `summary.smooth_bwd_candidates` 用 Spearman rank correlation 判断 activation outlier 是否能代理 backward `dY` outlier；当前只作为诊断，不直接改 kernel `smooth_bwd`。
- 对真正不适合 FP4 的 projection，仍用 `exclude_modules` 保留 BF16。

示例：

```bash
python benchmarks/analyze_fp4_lora_activation_grad_outliers.py \
  --batch 4 \
  --hidden 128 \
  --layers 2 \
  --steps 2 \
  --rank 32 \
  --override-rank 64 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --inject-outliers \
  --outlier-channel 0 \
  --outlier-scale 16
```

生成 overrides：

```python
from native_fp4 import fp4_lora_config_overrides_from_outlier_report

config_overrides = fp4_lora_config_overrides_from_outlier_report(
    "results/latest_fp4_lora_activation_grad_outliers.json",
    cfg,
    force_init="zero",
    disable_fuse_frozen_residual_dx=True,
)
```

测 rank bump 策略的训练开销：

```bash
python benchmarks/benchmark_fp4_lora_outlier_overrides.py \
  --batch 4 \
  --hidden 128 \
  --layers 2 \
  --rank 32 \
  --override-rank 64 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --warmup 3 \
  --iters 5
```

关注 `latency_ms.base_train_step`、`latency_ms.override_train_step` 和 `overhead.override_over_base`。

## 验证命令

```bash
cd /home/wyj24/projects/nunchaku/extracted_nunchaku_core
conda run -n triton python benchmarks/validate_native_fp4_lora_training.py \
  --m 257 \
  --in-features 3072 \
  --out-features 3584 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16

conda run -n triton python benchmarks/validate_native_fp4_lora_training.py \
  --m 129 \
  --in-features 512 \
  --out-features 768 \
  --rank 32 \
  --frozen-residual-rank 32 \
  --frozen-residual-init residual_svd \
  --init zero \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx

conda run -n triton python benchmarks/validate_native_fp4_lora_training.py \
  --m 129 \
  --in-features 512 \
  --out-features 768 \
  --rank 32 \
  --frozen-residual-rank 32 \
  --frozen-residual-init residual_svd \
  --init zero \
  --dtype fp16 \
  --lowrank-dtype fp16 \
  --fuse-lora-dx \
  --fuse-frozen-residual-dx \
  --cache-fused-lora-dx

conda run -n triton python benchmarks/validate_fp4_lora_training_policies.py

conda run -n triton python benchmarks/validate_fp4_lora_training_policies.py \
  --dtype fp16 \
  --lowrank-dtype fp16 \
  --modes throughput \
  --steps 2

conda run -n triton python benchmarks/validate_fp4_lora_prepare.py

conda run -n triton python benchmarks/validate_fp4_lora_prepare.py \
  --dtype fp16 \
  --lowrank-dtype fp16 \
  --mode throughput \
  --batch 4 \
  --hidden 128

conda run -n triton python benchmarks/validate_fp4_lora_finetune_convergence.py
```

验证项：

如果要验证 backward 重算 `x @ A.T` 的路径，在命令末尾追加 `--no-cache-lora-act`；如果要验证 fused dX 路径，追加 `--fuse-lora-dx`；如果要验证 packed LoRA dX cache，追加 `--cache-fused-lora-dx`。

- forward wrapper 是否等价于手写 `FP4 main + frozen residual branch + LoRA dense branch`。
- `dX` 是否等价于 `FP4 backward dX + frozen residual dX + LoRA dense dX`。
- `dA/dB/bias` 是否等价于手写 BF16/FP16 matmul 梯度。
