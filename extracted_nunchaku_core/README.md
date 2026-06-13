# Extracted Nunchaku Core

这个目录是从 Nunchaku 中独立整理出来的实验库，目标是把你关心的几条路径单独拿出来，方便直接做实验：

- 原生 FP4 GEMM
- FP4 + 16-bit 低秩分支混合算子
- 原生 FP8 GEMM
- fused / unfused 低秩分支消融
- FP4 backward `dX`
- 完整 LoRA backward 的多种优化版本

当前实验重点是 **RTX 5090 / Blackwell 原生 FP4 路径**。

论文：

- https://arxiv.org/abs/2411.05007

## 1. 目录说明

- `fp4_backend/`
  - 独立整理后的原生 FP4 CUDA 后端
- `csrc/`
  - PyTorch extension 入口，以及补充的 repack / decode CUDA 实现
- `native_fp4/`
  - Python 封装，主要实验接口都在这里
- `native_fp8/`
  - 最小 FP8 GEMM Python 封装，后端使用 CUDA/cuBLASLt 的 `torch._scaled_mm`
- `benchmarks/`
  - benchmark 和 validation 脚本
- `results/`
  - 实验结果 JSON

总结文档：

- [OPTIMIZATION_SUMMARY.md](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/OPTIMIZATION_SUMMARY.md)
- [OPTIMIZATION_SUMMARY_ZH.md](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/OPTIMIZATION_SUMMARY_ZH.md)

## 2. 你应该在哪个目录执行命令

下面所有命令都默认你已经进入：

```bash
cd /home/wyj24/projects/nunchaku/extracted_nunchaku_core
```

这很重要。

因为 benchmark 默认把结果写到相对路径 `results/`。如果你在别的目录执行，结果会落到错误的位置。

## 3. 环境准备

推荐直接使用你现在已经在用的环境：

```bash
conda activate triton
```

如果需要显式指定 CUDA：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

## 4. 编译扩展

第一次实验，先编译：

```bash
python setup.py build_ext --inplace
```

如果你更想安装成可导入包，也可以：

```bash
pip install -e .
```

编译成功后，关键扩展包括：

- `nunchaku_core._int4_cuda`
- `nunchaku_core._fp4_native_cuda`

`native_fp8/` 不依赖新的自定义 `.so`，只要求当前 PyTorch 版本支持：

- `torch.float8_e4m3fn`
- `torch._scaled_mm`

## 5. 最小导入检查

先确认 Python 封装能正常导入：

```bash
python -c "from native_fp4 import NunchakuFP4GemmOp, NunchakuFP4LowRankOp, NunchakuFP4BackwardDXOp, NunchakuFP4LowRankBackwardDXOp, NunchakuFP4LoRALinear; from native_fp8 import NunchakuFP8GemmOp; print('import ok')"
```

如果这里失败，不要急着跑 benchmark，先回去重编译。

## 5.1 Native FP8 最小验证

FP8 当前是最小可用版本：

- 数据格式：`float8_e4m3fn`
- 输出类型：跟权重一致（`fp16` 或 `bf16`）
- 后端：`torch._scaled_mm`
- 当前量化方式：`per-tensor scale`

先做 correctness：

```bash
python benchmarks/validate_native_fp8_ops.py \
  --m 333 \
  --in-features 4096 \
  --out-features 4096 \
  --dtype bf16
```

结果会写到：

- `results/latest_native_fp8_validation.json`

重点字段：

- `all_passed`
- `wrapper_vs_manual`
- `fp8_vs_fp16`

再做 benchmark：

```bash
python benchmarks/benchmark_native_fp8.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --dtype fp16 \
  --warmup 20 \
  --iters 50
```

结果会写到：

- `results/latest_native_fp8.json`

重点字段：

- `fp8_gemm_ms`
- `fp8_gemm_prequantized_ms`
- `fp8_gemm_speedup_vs_fp16`
- `fp8_gemm_prequantized_speedup_vs_fp16`

说明：

- `fp8_gemm_ms`：在线量化 + FP8 GEMM 的端到端时间
- `fp8_gemm_prequantized_ms`：只测 FP8 GEMM 本体，不含输入量化

## 6. 先做 correctness，再做 benchmark

建议按下面顺序做实验：

1. forward correctness
2. forward benchmark
3. fused / unfused 消融
4. backward correctness
5. backward benchmark

这样一旦出错，更容易定位。

## 7. Forward correctness

验证前向纯 FP4 和 FP4 + low-rank 封装是否正确：

```bash
python benchmarks/validate_native_fp4_ops.py \
  --m 333 \
  --in-features 3072 \
  --out-features 3584 \
  --rank 32 \
  --dtype fp16
```

结果会写到：

- `results/latest_native_fp4_validation.json`

你重点看：

- `all_passed`
- `pure_wrapper_vs_manual`
- `hybrid_wrapper_vs_manual`
- `zero_up_invariant`
- `zero_down_invariant`

## 8. Forward benchmark

### 8.1 纯 FP4 GEMM + hybrid forward

```bash
python benchmarks/benchmark_nunchaku_native_fp4.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype fp16 \
  --warmup 20 \
  --iters 50
```

结果会写到：

- `results/latest_native_fp4.json`

核心字段：

- `fp16_ms`
- `fp4_gemm_ms`
- `fp4_hybrid_ms`
- `fp4_gemm_speedup_vs_fp16`
- `fp4_hybrid_speedup_vs_fp16`

如果你在当前机器上复现实验，典型结果大致应接近：

- 纯 FP4 GEMM：约 `4.6x` vs FP16
- FP4 + low-rank：约 `3.5x` vs FP16

### 8.2 fused / unfused 低秩分支消融

```bash
python benchmarks/benchmark_fp4_bf16_fusion_ablation.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype fp16 \
  --lowrank-dtype bf16 \
  --warmup 20 \
  --iters 50
```

结果会写到：

- `results/latest_fp4_bf16_fusion_ablation.json`

重点字段：

- `fp4_bf16_fused_ms`
- `fp4_bf16_unfused_ms`
- `fused_speedup_vs_fp16`
- `unfused_speedup_vs_fp16`
- `unfused_over_fused`

这个实验主要回答一个问题：

- “低秩分支和 FP4 主分支做融合，到底快多少？”

## 9. Backward correctness

验证 backward `dX`、完整 low-rank backward，以及 repack / packed reuse 路径的数值正确性：

```bash
python benchmarks/validate_native_fp4_backward.py \
  --m 256 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype fp16
```

结果会写到：

- `results/latest_native_fp4_backward_validation.json`

重点字段：

- `all_passed`
- `qweight_bwd_cuda_matches_reference`
- `full_shared_packed_dx_matches_fused_rel_l2_lt_5e-4`
- `full_shared_packed_overlap_dx_matches_fused_rel_l2_lt_5e-4`
- `full_shared_packed_overlap_up_rel_l2_lt_1e-5`
- `full_shared_packed_overlap_down_rel_l2_lt_5e-4`

如果只是确认“这套 backward 当前能不能用”，先看：

- `all_passed == true`

## 10. Backward benchmark

完整 backward benchmark：

```bash
python benchmarks/benchmark_native_fp4_backward.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype fp16 \
  --warmup 10 \
  --iters 20
```

结果会写到：

- `results/latest_native_fp4_backward.json`

重点字段很多，建议优先看：

- `fp16_dx_ms`
- `fp4_dx_ms`
- `fp4_dx_hybrid_unfused_ms`
- `fp4_dx_hybrid_fused_ms`
- `fp16_full_backward_ms`
- `fp4_full_backward_unfused_ms`
- `fp4_full_backward_fused_ms`
- `fp4_full_backward_shared_cached_ms`
- `fp4_full_backward_shared_packed_ms`
- `fp4_full_backward_shared_packed_overlap_ms`
- `fp4_full_backward_shared_dual_ms`

如果你想快速判断“当前最优 full backward 是哪条路径”，直接看：

- `fp4_full_backward_shared_packed_overlap_ms`

当前这条路径是已经测出来的最优版本。

## 10.1 FP4 LoRA training correctness

如果你要把 frozen FP4 backbone + trainable BF16/FP16 LoRA 接到微调流程，先验证新的训练接口：

```bash
python benchmarks/validate_native_fp4_lora_training.py \
  --m 257 \
  --in-features 3072 \
  --out-features 3584 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16
```

结果会写到：

- `results/latest_native_fp4_lora_training_validation.json`

重点字段：

- `all_passed`
- `forward_vs_manual`
- `dx_vs_manual`
- `lora_up_grad_vs_manual`
- `lora_down_grad_vs_manual`

这个脚本验证的是 `NunchakuFP4LoRALinear` 与手写公式一致：

```text
y  = FP4_GEMM(x, W0) + scaling * (x @ A.T) @ B.T + bias
dX = FP4_dX(dY, W0) + scaling * (dY @ B) @ A
dB = scaling * dY.T @ (x @ A.T)
dA = scaling * (dY @ B).T @ x
```

详细设计见：

- [docs/fp4_lora_finetuning_design.md](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/docs/fp4_lora_finetuning_design.md)

如果要验证 backward 重算 `x @ A.T` 的路径，在命令末尾追加 `--no-cache-lora-act`。如果要验证 fused dX 路径，追加 `--fuse-lora-dx`；如果要验证 packed LoRA dX cache，追加 `--cache-fused-lora-dx`。

## 10.2 FP4 LoRA training benchmark

对比 frozen dense BF16/FP16 LoRA Linear 与 `NunchakuFP4LoRALinear` 的 forward、train step 和估算 backward：

```bash
python benchmarks/benchmark_native_fp4_lora_training.py \
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

结果会写到：

- `results/latest_native_fp4_lora_training.json`

重点字段：

- `latency_ms.dense_train_step`
- `latency_ms.fp4_cached_train_step`
- `latency_ms.fp4_recompute_train_step`
- `latency_ms.fp4_cached_fused_dx_train_step`
- `latency_ms.fp4_cached_fused_dx_cached_pack_train_step`
- `latency_ms.fp4_cached_fused_dx_cached_pack_reuse_dy_up_train_step`
- `latency_ms.refresh_fused_lora_dx_cache`
- `latency_ms.fp4_cached_fused_dx_cached_pack_grad_accum_per_micro_step`
- `latency_ms.fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_per_micro_step`
- `speedups.fp4_cached_train_step_vs_dense`
- `speedups.fp4_cached_fused_dx_train_step_vs_dense`
- `speedups.fp4_cached_fused_dx_cached_pack_train_step_vs_dense`
- `speedups.fp4_cached_fused_dx_cached_pack_plus_refresh_train_step_vs_dense`
- `speedups.fp4_cached_fused_dx_cached_pack_grad_accum_vs_dense`
- `speedups.fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_vs_dense`
- `speedups.fp4_cached_backward_estimate_vs_dense`
- `speedups.fp4_cached_fused_dx_backward_estimate_vs_dense`
- `speedups.fused_dx_cached_pack_vs_dynamic_pack_train_step`
- `speedups.fused_dx_cached_pack_reuse_dy_up_vs_cached_pack_train_step`
- `speedups.fused_dx_cached_pack_plus_refresh_vs_dynamic_pack_train_step`
- `speedups.fused_dx_cached_pack_vs_dynamic_pack_grad_accum`

当前 RTX 5090 短测结果，形状 `M=N=K=4096, rank=32, warmup=5, iters=10`：

| dtype | dense train step ms | FP4 dense-dX step ms | FP4 fused-dX dynamic-pack ms | FP4 fused-dX cached-pack ms | cached+refresh ms | cached-pack speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.0159 | 0.9464 | 0.9148 | 0.8917 | 0.9058 | 2.261x |
| FP16 | 1.7440 | 0.9009 | 0.8816 | 0.8708 | 0.8841 | 2.003x |

Gradient accumulation 短测，`grad_accum_steps=4, warmup=5, iters=10`：

| dtype | dense per micro-step ms | FP4 dynamic-pack per micro-step ms | FP4 cached-pack per micro-step ms | cached-pack vs dense |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 3.5701 | 1.5405 | 1.4255 | 2.504x |
| FP16 | 3.5335 | 1.4621 | 1.5103 | 2.340x |

说明：

- `backward estimate = train_step - train_graph_forward`，用于判断 backward 优化方向，不是单独 CUDA event 包住 backward 的精确拆分。
- `fuse_lora_dx=True` 会把 `dX_lora = (dY @ B) @ A` 的第二段并入 FP4 dX epilogue，但 LoRA 参数梯度仍用 dense BF16/FP16 matmul 保精度。
- `fuse_lowrank_forward=True` 是 dual-branch opt-in 实验选项：把 task LoRA forward 和 frozen residual forward 合成一次拼接 low-rank GEMM。它减少 launch/GEMM 次数，但会改变低秩分支的浮点归约顺序；验证脚本会 strict 检查当前调度，并额外用 `5e-4` rel_l2 tolerance 报告它相对“两支分开计算”公式的差异。默认关闭。
- `cache_fused_lora_dx=True` 只缓存 LoRA packed A/B，不缓存第二份 FP4 backbone；参数 version 变化时会自动刷新。
- `reuse_fused_dy_up_for_d_lora_down=True` 是 FP16-only 实验选项：复用 fused dX quantize kernel 产生的 packed `dY @ B`，decode 后用于 `dA = (dY @ B).T @ X`，避免额外 dense `dY @ B` matmul。
- `activation_checkpoint=True` 是逐 `NunchakuFP4LoRALinear` 的局部 checkpoint，只能省该算子内部的 `lora_act` 等 saved tensors；真正要省多层输入 activation，应在 transformer block/segment 外层用 `torch.utils.checkpoint`。
- BF16 单步下 cached-pack fused dX 相比 dynamic-pack fused dX 训练 step 快 `1.026x`；每步刷新 cache 后仍快 `1.010x`。FP16 单步下 cached-pack 约 `1.012x`，每步刷新后基本持平。
- Gradient accumulation 会摊薄 cache refresh 开销；accumulation 数字对测量顺序更敏感，建议看多轮结果再定默认策略。
- `forward_fp4_vs_dense` 的误差是 FP4 量化相对 dense full precision 权重的误差，不是 wrapper correctness；wrapper correctness 请看 `validate_native_fp4_lora_training.py`。

FP16 packed `dY @ B` 复用消融：

```bash
python benchmarks/validate_native_fp4_lora_training.py \
  --m 257 \
  --in-features 3072 \
  --out-features 3584 \
  --rank 32 \
  --dtype fp16 \
  --lowrank-dtype fp16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx \
  --reuse-fused-dy-up-for-d-lora-down
```

RTX 5090 上该路径 correctness 通过，`d_lora_down` rel_l2 约 `3.35e-5`。BF16 下同一复用方式会把 `d_lora_down` rel_l2 放大到约 `3.36e-3`，因此构造函数会拒绝 BF16 weight/LoRA 打开此选项。

性能上它是噪声敏感的小优化：FP16 `M=N=K=4096, rank=32` 两次短测中，单步相对 cached-pack 约 `0.968x-1.018x`，gradient accumulation per micro-step 约 `1.016x-1.036x`。建议只在实际训练循环里确认收益后启用。

## 10.3 FP4 LoRA training backward breakdown

如果要判断下一步该优化哪一块，跑 breakdown：

```bash
python benchmarks/benchmark_native_fp4_lora_training_breakdown.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --warmup 5 \
  --iters 10
```

结果会写到：

- `results/latest_native_fp4_lora_training_breakdown.json`

RTX 5090 短测，`M=N=K=4096, rank=32`：

| dtype | backward estimate ms | fused dX cached-pack ms | dense LoRA grad pair ms | LoRA grad share | LoRA pack refresh ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.6195 | 0.2152 | 0.0761 | 12.3% | 0.0388 |
| FP16 | 0.6467 | 0.2275 | 0.0396 | 6.1% | 0.0128 |

本轮 repack micro-optimization 后，RTX 5090 同形状 BF16 短测：

| metric | before ms | after ms | speedup |
| --- | ---: | ---: | ---: |
| `repack_backbone` | 0.0424 | 0.0391 | 1.084x |
| `fp4_dx_main` | 0.1868 | 0.1839 | 1.016x |
| `fused_dx_cached_pack` | 0.2175 | 0.2139 | 1.017x |
| `full_backward_minus_forward` | 0.6703 | 0.6561 | 1.022x |

这个优化只把 repack kernel 内每个输出 word 重复使用的 backward scale load、zero check 和固定 scale-group 计算移到 8 元素循环外；不保存第二份 transposed FP4 backbone，`qweight_bwd_cuda_matches_reference` 仍保持 bitwise exact。

结论：

- `dA/dB` 目前不是最大瓶颈；先写专用低秩梯度 kernel 的收益上限有限。
- 下一步更应该看 FP4 dX 主路径，包括 `dY` quantize、backbone repack、fused dX epilogue 的调度和重叠。
- LoRA pack refresh 已经换成 native CUDA layout pack；相对旧 PyTorch `pad + permute + contiguous` 路径，4096/rank32 短测约减少一半。

## 10.4 FP4 LoRA native pack validation

`cache_fused_lora_dx=True` 需要把 trainable LoRA A/B 转成 Nunchaku fused dX epilogue 使用的 packed layout。当前默认走 CUDA `pack_lowrank_weight`，不缓存第二份 FP4 backbone。

验证 CUDA pack 与原 PyTorch layout 参考实现 bitwise 一致，并测量单次 pack 开销：

```bash
python benchmarks/validate_native_fp4_lora_pack.py \
  --dtype bf16 \
  --warmup 20 \
  --iters 100
```

结果会写到：

- `results/latest_native_fp4_lora_pack_validation.json`

RTX 5090 短测，默认 shapes：

| dtype | typical native pack ms | typical torch pack ms | native speedup |
| --- | ---: | ---: | ---: |
| BF16 | 0.0058-0.0063 | 0.0195-0.0212 | 3.3x-3.6x |
| FP16 | 0.0058-0.0060 | 0.0196-0.0212 | 3.3x-3.6x |

## 10.5 FP4 LoRA model conversion

真实模型微调不应该手工逐层替换。`native_fp4.modeling` 提供了模型级工具：

```python
from dataclasses import replace

from native_fp4 import (
    FP4LoRAConfig,
    convert_linear_to_fp4_lora,
    fp4_lora_config_overrides_from_outlier_report,
    fp4_lora_parameter_groups,
    fp4_lora_peft_state_dict,
    fp4_lora_state_dict,
    freeze_non_fp4_lora_parameters,
    load_fp4_lora_peft_state_dict,
    load_fp4_lora_state_dict,
    register_fp4_lora_cache_refresh_hook,
    refresh_fused_lora_dx_caches,
)

cfg = FP4LoRAConfig(
    rank=32,
    lowrank_dtype=torch.bfloat16,
    # 推荐微调形态：冻结 residual_svd 量化补偿，只训练 zero-init task LoRA。
    init="zero",
    frozen_residual_rank=32,
    frozen_residual_init="residual_svd",
    # Optional: fuse task LoRA + frozen residual forward low-rank GEMMs.
    fuse_lowrank_forward=False,
    fuse_lora_dx=True,
    # FP16-only experimental: fuse frozen residual dX into the same epilogue.
    fuse_frozen_residual_dx=False,
    cache_fused_lora_dx=True,
)

sensitive_overrides = {
    # 完整路径、子模块名和完整路径后缀都可匹配；第一条匹配生效。
    # 用于 down_proj/outlier 层等敏感模块：可单独加 rank、改 init 或关闭实验 fusion。
    "layers.1.mlp.down_proj": replace(cfg, rank=64, fuse_frozen_residual_dx=False),
}

# 如果已经跑过 outlier 诊断，也可以直接从 JSON 生成 overrides。
auto_overrides = fp4_lora_config_overrides_from_outlier_report(
    "results/latest_fp4_lora_activation_grad_outliers.json",
    cfg,
    force_init="zero",
    disable_fuse_frozen_residual_dx=True,
)
sensitive_overrides.update(auto_overrides)

model, replaced = convert_linear_to_fp4_lora(
    model,
    cfg,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    exclude_modules=("lm_head",),
    config_overrides=sensitive_overrides,
)
trainable = freeze_non_fp4_lora_parameters(model)
refresh_fused_lora_dx_caches(model)

optimizer = torch.optim.AdamW(fp4_lora_parameter_groups(model), lr=1e-4, eps=1e-4)
cache_hook = register_fp4_lora_cache_refresh_hook(optimizer, model)

# 保存/加载时只处理 LoRA adapter，不保存 FP4 backbone buffers。
adapter_state = fp4_lora_state_dict(model)
load_fp4_lora_state_dict(model, adapter_state)

# PEFT 风格导出：module.lora_A.default.weight / module.lora_B.default.weight。
# 默认导出 padded effective rank，数值完全等价；如外部生态要求原始 rank，可裁剪导出。
peft_state = fp4_lora_peft_state_dict(model)
peft_trimmed_state = fp4_lora_peft_state_dict(model, trim_to_requested_rank=True)
load_fp4_lora_peft_state_dict(model, peft_state)
```

如果只想跑单分支 task LoRA，把 `frozen_residual_rank=0` 且 `frozen_residual_init="none"`。
如果已经通过 sensitivity scan 发现某些完整模块路径不适合 FP4，可用 `exclude_modules` 保持 BF16；如果只是需要更强补偿能力，优先用 `config_overrides` 对这些模块单独提高 rank 或调整 residual/task LoRA 策略。

验证批量替换、冻结参数、cache refresh/clear 和 backward：

```bash
python benchmarks/validate_native_fp4_lora_modeling.py \
  --batch 8 \
  --hidden 256 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx
```

验证 dual-branch 微调形态：

```bash
python benchmarks/validate_native_fp4_lora_modeling.py \
  --batch 4 \
  --hidden 128 \
  --rank 32 \
  --frozen-residual-rank 32 \
  --frozen-residual-init residual_svd \
  --init zero \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx
```

验证结果：

- `results/latest_native_fp4_lora_modeling_validation.json`
- BF16 fused cached 路径：替换 4 个目标 Linear，`lm_head` 保持 dense，只有 LoRA A/B 可训练，optimizer 参数组、post-step cache refresh hook、LoRA-only state_dict strict load 和 backward 均通过。
- `config_overrides` 路径：验证 `layers.1.down_proj` 可独立覆盖 rank/init，第二个模型 strict load 同样按该策略构造。
- PEFT adapter 路径：验证 `lora_A/lora_B` exact round-trip；`trim_to_requested_rank=True` 会按 requested rank 导出，加载时 padded tail 清零。
- dual-branch 路径：`frozen_residual_*` 是 frozen buffer，不进入 optimizer 参数组，也不进入 LoRA-only adapter checkpoint。
- `fuse_lowrank_forward=True` 可用于测试 forward 低秩分支合并收益；它只改变 low-rank forward 的调度和归约顺序，不改变默认数学公式。
- FP16 下可打开 `fuse_frozen_residual_dx=True`，把 task LoRA 和 frozen residual 的 dX 一并打包进 fused epilogue；BF16 下该路径目前误差偏大，默认关闭。

## 10.6 FP4 LoRA activation / grad outlier 诊断

如果要把 sensitivity scan 的结论转成 `config_overrides`，先看每个 FP4 LoRA Linear 的输入 activation 和 backward `dY` 通道 outlier：

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

输出：

- `results/latest_fp4_lora_activation_grad_outliers.json`
- `summary.rank_bump_candidates`：建议用 `config_overrides` 单独提高 rank 的模块。
- `summary.smooth_bwd_candidates`：activation/grad-output 通道 rank correlation 足够高的模块。当前只作为诊断信号；直接改 `smooth_bwd` 需要同步权重量化/反缩放补偿，不能把它当成无损开关。

把诊断结果转成模型转换策略：

```python
from native_fp4 import fp4_lora_config_overrides_from_outlier_report

config_overrides = fp4_lora_config_overrides_from_outlier_report(
    "results/latest_fp4_lora_activation_grad_outliers.json",
    cfg,
    force_init="zero",
    disable_fuse_frozen_residual_dx=True,
)
```

## 10.7 Outlier-driven overrides 开销 benchmark

提高敏感模块 rank 会增加 LoRA 分支计算。用下面脚本量化 base config 和 outlier-driven overrides 的 train-step 开销：

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

输出：

- `results/latest_fp4_lora_outlier_override_overhead.json`
- `latency_ms.base_train_step`
- `latency_ms.override_train_step`
- `overhead.override_over_base`

## 10.8 FP4 LoRA activation checkpoint 消融

用于测量逐 Linear checkpoint 与 block/segment checkpoint 的显存/速度权衡：

```bash
python benchmarks/benchmark_fp4_lora_activation_checkpoint.py \
  --batch 512 \
  --hidden 1024 \
  --layers 4 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx \
  --intermediate-activation silu \
  --warmup 5 \
  --iters 10
```

输出：

- `results/latest_fp4_lora_activation_checkpoint.json`
- `latency_ms.no_activation_checkpoint_train_step`
- `latency_ms.module_activation_checkpoint_train_step`
- `latency_ms.stack_activation_checkpoint_train_step`
- `peak_memory_bytes.*_delta`
- `derived.*_peak_delta_reduction`

RTX 5090 短测，`batch=512, hidden=1024, layers=4, rank=32, BF16, fused dX cached pack`：

| checkpoint scope | intermediate activation | train step ms | peak delta reduction | conclusion |
| --- | --- | ---: | ---: | --- |
| module | none | 2.0313 | 1.2% | 逐 Linear checkpoint 基本只省内部 `lora_act`，不推荐默认开启 |
| stack/block | none | 1.6118 | 9.9% | 能省跨层输入 activation，但要重算整段 forward |
| module | silu | 2.0873 | 1.0% | 非线性本身仍保存激活，逐 Linear checkpoint 收益更低 |
| stack/block | silu | 1.6642 | 7.6% | 推荐作为长序列/多层微调的显存模式，按 block 粒度打开 |

正确性：`module` 和 `stack` checkpoint 的 forward、`dX`、首尾 LoRA A/B 梯度与 no-checkpoint reference 均为 `rel_l2=0`。

## 11. 建议的完整实验顺序

直接按下面执行即可：

```bash
cd /home/wyj24/projects/nunchaku/extracted_nunchaku_core
conda activate triton
python setup.py build_ext --inplace
python benchmarks/validate_native_fp4_ops.py --m 333 --in-features 3072 --out-features 3584 --rank 32 --dtype fp16
python benchmarks/benchmark_nunchaku_native_fp4.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype fp16 --warmup 20 --iters 50
python benchmarks/benchmark_fp4_bf16_fusion_ablation.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype fp16 --lowrank-dtype bf16 --warmup 20 --iters 50
python benchmarks/validate_native_fp4_backward.py --m 256 --in-features 4096 --out-features 4096 --rank 32 --dtype fp16
python benchmarks/benchmark_native_fp4_backward.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype fp16 --warmup 10 --iters 20
python benchmarks/validate_native_fp4_lora_training.py --m 257 --in-features 3072 --out-features 3584 --rank 32 --dtype bf16 --lowrank-dtype bf16
python benchmarks/validate_native_fp4_lora_training.py --m 129 --in-features 512 --out-features 768 --rank 32 --frozen-residual-rank 32 --frozen-residual-init residual_svd --init zero --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx
python benchmarks/validate_native_fp4_lora_training.py --m 129 --in-features 512 --out-features 768 --rank 32 --frozen-residual-rank 32 --frozen-residual-init residual_svd --init zero --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --fuse-lowrank-forward
python benchmarks/validate_native_fp4_lora_training.py --m 129 --in-features 512 --out-features 768 --rank 32 --frozen-residual-rank 32 --frozen-residual-init residual_svd --init zero --dtype fp16 --lowrank-dtype fp16 --fuse-lora-dx --fuse-frozen-residual-dx --cache-fused-lora-dx
python benchmarks/validate_native_fp4_lora_pack.py --dtype bf16 --warmup 20 --iters 100
python benchmarks/validate_native_fp4_lora_modeling.py --batch 8 --hidden 256 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx
python benchmarks/analyze_fp4_lora_activation_grad_outliers.py --batch 4 --hidden 128 --layers 2 --steps 2 --rank 32 --override-rank 64 --dtype bf16 --lowrank-dtype bf16 --inject-outliers --outlier-channel 0 --outlier-scale 16
python benchmarks/benchmark_fp4_lora_outlier_overrides.py --batch 4 --hidden 128 --layers 2 --rank 32 --override-rank 64 --dtype bf16 --lowrank-dtype bf16 --warmup 3 --iters 5
python benchmarks/benchmark_fp4_lora_activation_checkpoint.py --batch 512 --hidden 1024 --layers 4 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --intermediate-activation silu --warmup 5 --iters 10
python benchmarks/benchmark_native_fp4_lora_training.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10
python benchmarks/benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --dtype bf16 --warmup 10 --iters 30
python benchmarks/benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --dtype bf16 --warmup 10 --iters 30 --fuse-lowrank-forward
python benchmarks/benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --dtype fp16 --warmup 10 --iters 30
python benchmarks/benchmark_native_fp4_lora_training_breakdown.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10
```

RTX 5090 上 `benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --warmup 10 --iters 30`：

| path | dtype | train step ms | note |
| --- | --- | ---: | --- |
| task LoRA only, fused dX | bf16 | 0.2586 | task-only reference |
| dual branch, residual dense dX | bf16 | 0.3367 | `1.302x` overhead vs task-only；BF16 fused residual dX 暂不启用 |
| task LoRA only, fused dX | fp16 | 0.2575 | task-only reference |
| dual branch, residual dense dX | fp16 | 0.3046 | `1.183x` overhead vs task-only |
| dual branch, residual fused dX | fp16 | 0.2765 | `1.101x` vs residual dense dX，`dX` rel_l2 `3.80e-4` |

`fuse_lowrank_forward=True` 消融，RTX 5090 同形状 `warmup=10,iters=30`：

| path | dtype | train step ms | result |
| --- | --- | ---: | --- |
| dual branch, residual dense dX | bf16 | 0.3122 | default reference |
| dual branch, residual dense dX + fused lowrank forward | bf16 | 0.3092 | `1.010x` vs default dual，接近噪声 |
| dual branch, residual dense dX | fp16 | 0.2999 | default reference |
| dual branch, residual dense dX + fused lowrank forward | fp16 | 0.3400 | slower，`0.882x` vs default dual |
| dual branch, residual fused dX | fp16 | 0.2791 | default fused dX reference |
| dual branch, residual fused dX + fused lowrank forward | fp16 | 0.2797 | essentially tied，`0.998x` vs default fused dX |

结论：当前 fused low-rank forward 只作为消融保留，不建议默认打开。它减少了一次低秩 branch 的 matmul/launch，但拼接 rank 后的 GEMM 形状不一定更优，训练 step 收益在 BF16 下接近噪声，在 FP16 dense residual dX 路径反而变慢。

## 12. 主要 Python 接口

可以直接在 Python 里使用的类：

- `native_fp4.NunchakuFP4GemmOp`
  - 纯 FP4 GEMM
- `native_fp4.NunchakuFP4LowRankOp`
  - 前向 FP4 + 16-bit low-rank 混合算子
- `native_fp4.NunchakuFP4LowRankUnfusedOp`
  - 前向 unfused 低秩分支消融
- `native_fp4.NunchakuFP4BackwardDXOp`
  - backward 纯 FP4 `dX`
- `native_fp4.NunchakuFP4LowRankBackwardDXOp`
  - backward 混合算子和 full backward 多种路径
- `native_fp4.NunchakuFP4LoRALinear`
  - frozen FP4 backbone + 可选 frozen residual low-rank + trainable BF16/FP16 task LoRA 微调接口
- `native_fp4.FP4LoRAConfig`
  - 批量替换 Linear 时使用的配置对象，支持 `frozen_residual_rank/init`、`activation_checkpoint` 和 FP16-only `fuse_frozen_residual_dx`
- `native_fp4.convert_linear_to_fp4_lora`
  - 按完整路径/后缀/子模块名匹配并替换 `torch.nn.Linear`
- `native_fp4.fp4_lora_config_overrides_from_outlier_report`
  - 从 outlier 诊断 JSON 自动生成 `config_overrides`
- `native_fp4.freeze_non_fp4_lora_parameters`
  - 冻结非 LoRA 参数，只保留 LoRA A/B 可训练
- `native_fp4.iter_fp4_lora_named_parameters`
  - 枚举 LoRA-only 参数，默认不包含 bias
- `native_fp4.fp4_lora_parameter_groups`
  - 生成只含 LoRA adapter 的 optimizer 参数组
- `native_fp4.register_fp4_lora_cache_refresh_hook`
  - 在 `optimizer.step()` 后 eager refresh fused dX packed LoRA cache
- `native_fp4.fp4_lora_state_dict`
  - 导出 LoRA-only adapter checkpoint，不包含 FP4 backbone buffers
- `native_fp4.load_fp4_lora_state_dict`
  - strict 加载 LoRA-only adapter checkpoint，并清空 packed LoRA cache
- `native_fp4.fp4_lora_peft_state_dict`
  - 导出 PEFT 风格 `lora_A/lora_B` adapter checkpoint，默认保留 padded effective rank 以保证无损
- `native_fp4.load_fp4_lora_peft_state_dict`
  - 加载 PEFT 风格 adapter checkpoint，支持 requested-rank 输入并清零 padded tail
- `native_fp4.refresh_fused_lora_dx_caches`
  - optimizer step 后显式刷新 fused dX packed LoRA cache
- `native_fp4.clear_fused_lora_dx_caches`
  - 清空模型内所有 FP4 LoRA cache

## 13. 结果文件说明

默认结果都写到 `results/`：

- `latest_native_fp4.json`
  - 前向 FP4 / hybrid benchmark
- `latest_native_fp4_validation.json`
  - 前向 correctness
- `latest_fp4_bf16_fusion_ablation.json`
  - fused / unfused 消融
- `latest_native_fp4_backward.json`
  - backward benchmark
- `latest_native_fp4_backward_validation.json`
  - backward correctness
- `latest_native_fp4_lora_training_validation.json`
  - FP4 LoRA training wrapper correctness
- `latest_native_fp4_lora_training.json`
  - FP4 LoRA training benchmark
- `latest_native_fp4_lora_dual_branch_bf16.json`
  - dual-branch BF16 residual dense dX benchmark
- `latest_native_fp4_lora_dual_branch_fp16.json`
  - dual-branch FP16 fused residual dX benchmark
- `latest_native_fp4_lora_modeling_validation.json`
  - 模型级 Linear 替换、参数冻结和 cache 管理验证
- `latest_fp4_lora_activation_grad_outliers.json`
  - FP4 LoRA activation / grad-output outlier 诊断和 rank/smooth 建议
- `latest_fp4_lora_outlier_override_overhead.json`
  - outlier-driven `config_overrides` 相对 base config 的 train-step 开销

另外还会生成带时间戳的快照 JSON，方便保留历史实验结果。

## 14. 常见注意事项

### 14.1 一定在本目录里跑

建议始终在下面目录执行：

```bash
cd /home/wyj24/projects/nunchaku/extracted_nunchaku_core
```

否则 `results/` 会写到别的地方。

### 14.2 当前 README 的实验重点是 FP4，不是旧的 INT4

这个仓库里还保留了早期 INT4 路径和相关脚本，但你现在在 5090 上，优先关注：

- `benchmark_nunchaku_native_fp4.py`
- `validate_native_fp4_ops.py`
- `benchmark_fp4_bf16_fusion_ablation.py`
- `validate_native_fp4_backward.py`
- `benchmark_native_fp4_backward.py`

### 14.3 backward repack 是瞬时的，不是常驻双份权重

当前 backward 设计刻意避免永久保存第二份 `qweight_bwd`。

这意味着：

- 常驻内存不会因为 backward 权重复制而近似翻倍
- 但 benchmark 里会包含 transient repack 开销

### 14.4 `shared_dual` 不是当前最快路径

虽然 `shared_dual` 看起来减少了一步 decode，但目前它比：

- `shared_packed`
- `shared_packed_overlap`

都更慢。

如果你的目标是“跑最快的一版 full backward”，优先看：

- `shared_packed_overlap`

## 15. 如果你只想快速看当前最好结果

先跑：

```bash
python benchmarks/benchmark_nunchaku_native_fp4.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype fp16 --warmup 20 --iters 50
python benchmarks/benchmark_native_fp4_backward.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype fp16 --warmup 10 --iters 20
```

然后看：

- `results/latest_native_fp4.json`
- `results/latest_native_fp4_backward.json`

目前可参考的典型量级：

- 前向纯 FP4 GEMM：约 `4.6x` vs FP16
- 前向 hybrid：约 `3.5x` vs FP16
- backward 纯 FP4 `dX`：约 `3.2x` vs FP16
- 最优 full backward：约 `2.8x` vs FP16

## 16. 想看更完整的实验总结

如果你不仅想跑，还想看已经做过哪些优化与收益，请看：

- [OPTIMIZATION_SUMMARY.md](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/OPTIMIZATION_SUMMARY.md)
- [OPTIMIZATION_SUMMARY_ZH.md](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/OPTIMIZATION_SUMMARY_ZH.md)
