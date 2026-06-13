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
)

cfg = FP4LoRAConfig(
    rank=32,
    lowrank_dtype=torch.bfloat16,
    init="zero",
    frozen_residual_rank=32,
    frozen_residual_init="residual_svd",
    fuse_lora_dx=True,
    cache_fused_lora_dx=True,
    overlap_lora_grad=True,
    overlap_lora_grad_min_rows=4096,
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

model, replaced = convert_linear_to_fp4_lora(
    model.cuda().to(torch.bfloat16),
    cfg,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    exclude_modules=("lm_head",),
    config_overrides=sensitive_overrides,
)
```

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
- `load_fp4_lora_state_dict` 和 `load_fp4_lora_peft_state_dict` 加载后都会清空 packed LoRA dX cache，避免 adapter 参数与 cache 不一致。

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
- `cache_lora_act`：是否保存 forward 的 `x @ A.T`，避免 backward 计算 `dB` 时重算。
- `activation_checkpoint`：逐 `NunchakuFP4LoRALinear` 的局部 checkpoint。它只省该算子内部 saved tensors；要显著降低多层输入 activation，应该在 transformer block/segment 外层做 checkpoint。
- `fuse_lowrank_forward`：dual-branch opt-in 实验选项，把 task LoRA forward 与 frozen residual forward 合成一次拼接 low-rank GEMM；减少 launch/GEMM 次数，但会改变浮点归约顺序。验证脚本 strict 检查当前调度，并额外报告相对“两支分开计算”公式的 rel_l2；默认关闭。
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
  --grad-accum-steps 4
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
- BF16 单步下 cached-pack fused dX 相比 dynamic-pack fused dX 快 `1.026x`；每步刷新 cache 后仍快 `1.010x`。FP16 单步下 cached-pack 约 `1.012x`，每步刷新后基本持平。
- Gradient accumulation 会摊薄 cache refresh 开销；accumulation benchmark 对测量顺序更敏感，默认策略仍应以真实训练循环为准。
- 为保证训练梯度精度，默认 `dA` 仍使用 dense `dY @ B`；BF16 下复用 fused dX 产生的 packed `dY @ B` 会给 `dA` 带来约 `3.36e-3` rel_l2，不作为默认梯度来源。
- FP16 下提供 `reuse_fused_dy_up_for_d_lora_down=True` 实验选项，复用 packed `dY @ B` 可通过 correctness，`dA` rel_l2 约 `3.35e-5`；性能收益较小且有噪声，4096/rank32 短测中单步约 `0.968x-1.018x`，梯度累积 per micro-step 约 `1.016x-1.036x`。
- `overlap_lora_grad=True` 要求同时打开 `fuse_lora_dx=True` 和 `cache_fused_lora_dx=True`；实现上用多 CUDA stream 重叠 transient FP4 repack、fused dX、`dB` GEMM 和 `dA` GEMM。
- `overlap_lora_grad_min_rows=4096` 是默认 auto gate：小于该 flattened row 数时自动回落到 sequential cached fused-dX 路径，避免 2048 形状上多 stream 调度变慢；设为 `0` 可强制 always-overlap 做消融。
- BF16 exact overlap 不复用 packed 近似 `dY @ B`，`dA` 仍走 dense `dY @ B`。Correctness：`validate_native_fp4_lora_training.py --m 257 --in-features 3072 --out-features 3584 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --overlap-lora-grad --overlap-lora-grad-min-rows 0` 通过，`dX` rel_l2 `1.55e-4`，`dA` rel_l2 `0`。
- BF16 frozen residual exact overlap 已支持：task LoRA dX 走 fused epilogue，frozen residual dX 保持 dense side stream；`validate_native_fp4_lora_training.py --m 129 --in-features 512 --out-features 768 --rank 32 --frozen-residual-rank 32 --frozen-residual-init residual_svd --init zero --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --overlap-lora-grad` 通过，forward/dX/LoRA A/B/bias grad rel_l2 全为 `0`。
- 4096/rank32 BF16 短测，`benchmark_native_fp4_lora_training.py --warmup 10 --iters 30 --grad-accum-steps 4`：cached-pack `0.9314 ms`，exact overlap `0.8830 ms`；gradient accumulation per micro-step `0.9594 -> 0.8956 ms`。单步 `1.055x` vs cached-pack，grad accumulation `1.071x` vs cached-pack。
- FP16 reuse+overlap 路径同时打开 `reuse_fused_dy_up_for_d_lora_down=True`，复用 decoded packed `dY @ B`。Correctness：`dX` rel_l2 `1.81e-5`，`dA` rel_l2 `3.56e-5`；4096/rank32 短测中单步 `1.008x` vs reuse，grad accumulation `1.037x` vs reuse。
- reuse-based overlap 仍不支持 frozen residual branch；dual-branch 默认使用 exact overlap。
- 保存 forward `lora_act` 对大形状有小幅收益，约 `3%-4%`；是否默认缓存要结合训练显存预算决定。

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
  --warmup 5 \
  --iters 10
```

RTX 5090 短测，`M=N=K=4096, rank=32`：

| dtype | backward estimate ms | fused dX cached-pack ms | dense LoRA grad pair ms | LoRA grad share | LoRA pack refresh ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.6195 | 0.2152 | 0.0761 | 12.3% | 0.0388 |
| FP16 | 0.6467 | 0.2275 | 0.0396 | 6.1% | 0.0128 |

直接结论：

- `dA/dB` 目前不是最大瓶颈；低秩梯度专用 kernel 的收益上限有限。
- 更值得优先看 FP4 dX 主路径，包括 `dY` quantize、backbone repack、fused dX epilogue 的调度和重叠。
- LoRA pack refresh 已经换成 native CUDA layout pack；相对旧 PyTorch `pad + permute + contiguous` 路径，4096/rank32 短测约减少一半。

### Native FP4 backward repack micro-optimization

新增微优化：

- `csrc/fp4_repack_cuda.cu`
- 将每个 32-bit 输出 word 内重复使用的 backward scale load、zero check 和固定 forward scale group 计算移到 8 元素循环外。
- 不保存第二份 transposed FP4 backbone；仍然每次 backward transient repack。
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

P2：继续研究 BF16 下 packed `dY @ B` 复用的精度问题；当前仅 FP16 opt-in，BF16 必须继续用 dense `dY @ B` 保护 `dA`。

P3：把 `dA/dB` 的低秩 GEMM 改成小 rank 专用 CUDA kernel，减少 PyTorch kernel launch 和中间张量开销。

P4：加入 activation cache policy：

- `save_bf16`：保存 BF16 `x` 和 `lora_act`，速度优先。
- `recompute_lora_act`：只保存 `x`，少存一个 `[M, rank]`。
- `save_fp4_cache`：保存 forward FP4 主分支已经生成的 `qact + ascales`，不保存 BF16/FP16 `x`。这能显著省 activation cache，但 `dA=(dY@B).T@x` 会变成近似梯度。当前同时提供 naive dequant 路径和 fused `dA` 原型，前者会重新物化 dense `x_hat`，后者避免中间张量但仍需继续优化。
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
- 新增 `benchmark_fp4_lora_activation_cache_policy.py`，比较 saved BF16/FP16 `x` 与 FP4 activation cache 对 `dA` 的显存、速度和精度影响。

RTX 5090 BF16 短测：

| shape | saved x cache | FP4 cache | memory reduction | x_hat rel_l2 | dA rel_l2 | saved-x dA ms | FP4 dequant+dA ms | fused dA ms | fused vs dequant rel_l2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048^2, rank32 | 8.00 MiB | 2.25 MiB | 3.56x | 9.78e-2 | 9.82e-2 | 0.0280 | 0.0689 | 0.1584 | 2.86e-3 |
| 4096^2, rank32 | 32.00 MiB | 9.00 MiB | 3.56x | 9.78e-2 | 9.68e-2 | 0.0387 | 0.1977 | 0.5062 | 7.84e-5 |

结论：

- FP4 cache 的显存收益明确，`qact + ascales` 约为 saved BF16/FP16 `x` 的 `28.1%`。
- 直接用 FP4-dequant activation 计算 `dA` 会带来约 `1e-1` rel_l2 梯度误差；如果要保持 LoRA 梯度精确，默认仍应使用 `save_bf16` 或重算 `x` 来源。
- 当前 naive CUDA dequant 仍要物化 dense `x_hat`，4096 形状比 saved-x `dA` 慢约 `5.1x`。
- fused `dA` 原型避免了 dense `x_hat`，但当前 rank-tiled 标量 reduction 仍慢于 `dequant + GEMM`，4096 形状约 `0.39x`；下一步应把 FP4 decode staging 和 reduction 改成更 tensor-core/GEMM 友好的分块。
- 即便 fused `dA` 继续优化，它对齐的仍是 `dequant(qact, ascales)` 近似路径，不能消除 FP4 activation cache 自身带来的 `dA` 精度损失，因此只适合作为显存/近似训练模式。

P5：dual-branch residual/task LoRA 初始化已落地。FP16 下 `fuse_frozen_residual_dx=True` 可以把 frozen residual dX 与 task LoRA dX 一并打包进 fused epilogue；BF16 下该 packed residual dX 路径误差偏大，默认仍保留 residual dense dX。BF16 exact overlap 支持 dual-branch，但 residual dX 保持 dense side stream。

P5.1：`fuse_lowrank_forward=True` 已作为 opt-in forward 消融路径加入。该路径把 `x @ A.T @ B.T` 与 `x @ R_down.T @ R_up.T` 拼成一次 low-rank GEMM，预期降低 dual-branch forward overhead；由于归约顺序不同，验证脚本对当前调度保持严格 `1e-6`，同时用 `5e-4` rel_l2 tolerance 报告相对“两支分开计算”公式的差异。

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

`fuse_lowrank_forward=True` 消融，RTX 5090 同形状 `warmup=10,iters=30`：

| path | dtype | train step ms | result |
| --- | --- | ---: | --- |
| dual branch, residual dense dX | bf16 | 0.3122 | default reference |
| dual branch, residual dense dX + fused lowrank forward | bf16 | 0.3092 | `1.010x` vs default dual，接近噪声 |
| dual branch, residual dense dX | fp16 | 0.2999 | default reference |
| dual branch, residual dense dX + fused lowrank forward | fp16 | 0.3400 | slower，`0.882x` vs default dual |
| dual branch, residual fused dX | fp16 | 0.2791 | default fused dX reference |
| dual branch, residual fused dX + fused lowrank forward | fp16 | 0.2797 | essentially tied，`0.998x` vs default fused dX |

因此当前不把 `fuse_lowrank_forward` 设为默认。它的价值主要是量化 dual-branch forward 合并的上限；实际训练 step 的瓶颈仍更偏向 dX 主路径和 residual dX 融合。

P6：加入 outlier-aware FP4 训练策略：

- `analyze_fp4_lora_activation_grad_outliers.py` 已落地 activation / grad-output 通道统计。
- `summary.rank_bump_candidates` 可用 `fp4_lora_config_overrides_from_outlier_report` 直接转成 `config_overrides`，对敏感 projection 单独提高 rank 或调整 residual/task LoRA 策略。
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
```

验证项：

如果要验证 backward 重算 `x @ A.T` 的路径，在命令末尾追加 `--no-cache-lora-act`；如果要验证 fused dX 路径，追加 `--fuse-lora-dx`；如果要验证 packed LoRA dX cache，追加 `--cache-fused-lora-dx`。

- forward wrapper 是否等价于手写 `FP4 main + frozen residual branch + LoRA dense branch`。
- `dX` 是否等价于 `FP4 backward dX + frozen residual dX + LoRA dense dX`。
- `dA/dB/bias` 是否等价于手写 BF16/FP16 matmul 梯度。
