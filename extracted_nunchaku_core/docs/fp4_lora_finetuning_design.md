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
from native_fp4 import FP4LoRAConfig, convert_linear_to_fp4_lora

cfg = FP4LoRAConfig(
    rank=32,
    lowrank_dtype=torch.bfloat16,
    init="zero",
    frozen_residual_rank=32,
    frozen_residual_init="residual_svd",
    fuse_lora_dx=True,
    cache_fused_lora_dx=True,
)

model, replaced = convert_linear_to_fp4_lora(
    model.cuda().to(torch.bfloat16),
    cfg,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"),
    exclude_modules=("lm_head",),
)
```

Adapter checkpoint 示例：

```python
from native_fp4 import fp4_lora_state_dict, load_fp4_lora_state_dict

adapter_state = fp4_lora_state_dict(model)
missing, unexpected = load_fp4_lora_state_dict(model, adapter_state, strict=True)
```

checkpoint 边界：

- `fp4_lora_state_dict` 只导出 `lora_down/lora_up` 和可选 trainable bias。
- 不导出 `qweight/wscales/wscales_bwd_*` 等 frozen FP4 backbone buffers。
- 不导出 `frozen_residual_down/frozen_residual_up`；它们属于 quantized base model 的 frozen compensation branch，不属于 task adapter。
- `load_fp4_lora_state_dict` 加载后会清空 packed LoRA dX cache，避免 adapter 参数与 cache 不一致。

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
- `fuse_frozen_residual_dx`：FP16-only 实验选项，把 task LoRA 和 frozen residual 的 dX 合并为同一个 packed low-rank epilogue。BF16 下同一路径目前 `dX` rel_l2 约 `2.1e-3`，不作为默认。
- `target_modules/exclude_modules`：模型级替换时用于控制哪些 Linear 进入 FP4 LoRA。

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
- `checkpoint`：进一步重算上游 activation，面向长序列微调。

P5：dual-branch residual/task LoRA 初始化已落地。FP16 下 `fuse_frozen_residual_dx=True` 可以把 frozen residual dX 与 task LoRA dX 一并打包进 fused epilogue；BF16 下该路径误差偏大，默认仍保留 residual dense dX。

RTX 5090 短测，`benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --warmup 5 --iters 10`：

| path | train step ms | result |
| --- | ---: | --- |
| task LoRA fused + residual dense dX | 0.3480 | baseline |
| task LoRA + residual fused dX | 0.3038 | `1.146x`, `dX` rel_l2 `3.81e-4` |

P6：加入 outlier-aware FP4 训练策略：

- activation smooth / online scale 统计。
- 对敏感 projection 保留 BF16。
- residual LoRA 初始化和 task LoRA 初始化分离后的 scale/outlier 策略。

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
