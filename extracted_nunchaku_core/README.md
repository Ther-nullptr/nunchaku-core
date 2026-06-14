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
  - FP8 GEMM Python 封装，默认使用 PyTorch `torch._scaled_mm`，也可以探测并尝试 DeepGEMM 后端
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

`native_fp8/` 的默认后端不依赖新的自定义 `.so`，只要求当前 PyTorch 版本支持：

- `torch.float8_e4m3fn`
- `torch._scaled_mm`

DeepGEMM 后端是可选路径：

- 设置 `NUNCHAKU_DEEPGEMM_PATH=/home/wyj24/projects/DeepGEMM`，或者在脚本里传 `--deep-gemm-path /home/wyj24/projects/DeepGEMM`
- DeepGEMM JIT 还需要 `CUDA_HOME/bin/cuobjdump`
- 在当前 RTX 5090 / SM120 环境下，DeepGEMM 的 SM100 kernel 可能因为 `tcgen05.*` 指令不被 `sm_120a` 目标接受而编译失败；`--backend auto` 会记录错误并回退到 PyTorch FP8

## 5. 最小导入检查

先确认 Python 封装能正常导入：

```bash
python -c "from native_fp4 import NunchakuFP4GemmOp, NunchakuFP4LowRankOp, NunchakuFP4BackwardDXOp, NunchakuFP4LowRankBackwardDXOp, NunchakuFP4LoRALinear; from native_fp8 import NunchakuFP8GemmOp; print('import ok')"
```

如果这里失败，不要急着跑 benchmark，先回去重编译。

## 5.1 Native FP8 最小验证

FP8 当前提供两个后端：

- 数据格式：`float8_e4m3fn`
- 输出类型：跟权重一致（`fp16` 或 `bf16`）
- `torch` 后端：`torch._scaled_mm`，per-tensor scale
- `deep_gemm` 后端：DeepGEMM 风格 A per-token scale + B 128x128 block scale，当前仅用于 BF16 输出
- `auto` 后端：优先尝试 DeepGEMM，失败时回退到 `torch`

先做 correctness：

```bash
python benchmarks/validate_native_fp8_ops.py \
  --m 333 \
  --in-features 4096 \
  --out-features 4096 \
  --dtype bf16 \
  --backend auto
```

如果想显式探测本地 DeepGEMM：

```bash
NUNCHAKU_DEEPGEMM_PATH=/home/wyj24/projects/DeepGEMM \
python benchmarks/validate_native_fp8_ops.py \
  --m 333 \
  --in-features 4096 \
  --out-features 4096 \
  --dtype bf16 \
  --backend auto \
  --deep-gemm-path /home/wyj24/projects/DeepGEMM
```

结果会写到：

- `results/latest_native_fp8_validation.json`

重点字段：

- `all_passed`
- `backend`
- `wrapper_vs_manual`
- `fp8_vs_fp16`

再做 benchmark：

```bash
python benchmarks/benchmark_native_fp8.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --dtype fp16 \
  --backend torch \
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
- `backend`

说明：

- `fp8_gemm_ms`：在线量化 + FP8 GEMM 的端到端时间
- `fp8_gemm_prequantized_ms`：只测 FP8 GEMM 本体，不含输入量化
- `backend.last_backend`：实际执行的后端，判断 DeepGEMM 是否真的跑起来时看这个字段

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
  --grad-accum-steps 4 \
  --backward-weight-policy repack
```

结果会写到：

- `results/latest_native_fp4_lora_training.json`

重点字段：

- `latency_ms.dense_train_step`
- `latency_ms.fp4_cached_train_step`
- `latency_ms.fp4_recompute_train_step`
- `latency_ms.fp4_cached_fused_dx_train_step`
- `latency_ms.fp4_cached_fused_dx_cached_pack_train_step`
- `latency_ms.fp4_cached_fused_dx_cached_pack_fp4_act_cache_d_lora_down_train_step`
- `latency_ms.fp4_cached_fused_dx_cached_pack_reuse_dy_up_train_step`
- `latency_ms.refresh_fused_lora_dx_cache`
- `latency_ms.fp4_cached_fused_dx_cached_pack_grad_accum_per_micro_step`
- `latency_ms.fp4_cached_fused_dx_cached_pack_fp4_act_cache_d_lora_down_grad_accum_per_micro_step`
- `latency_ms.fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_per_micro_step`
- `activation_cache_bytes.fp4_cache_reduction_vs_saved_x`
- `backward_weight_cache_bytes.cached_backward_qweight_vs_dense_weight`
- `speedups.fp4_cached_train_step_vs_dense`
- `speedups.fp4_cached_fused_dx_train_step_vs_dense`
- `speedups.fp4_cached_fused_dx_cached_pack_train_step_vs_dense`
- `speedups.fp4_act_cache_d_lora_down_train_step_vs_cached_pack_exact`
- `speedups.fp4_cached_fused_dx_cached_pack_plus_refresh_train_step_vs_dense`
- `speedups.fp4_cached_fused_dx_cached_pack_grad_accum_vs_dense`
- `speedups.fp4_act_cache_d_lora_down_grad_accum_vs_cached_pack_exact`
- `speedups.fp4_cached_fused_dx_cached_pack_reuse_dy_up_grad_accum_vs_dense`
- `speedups.fp4_cached_fused_dx_cached_pack_reuse_dy_up_overlap_grad_accum_vs_dense`
- `speedups.fp4_cached_backward_estimate_vs_dense`
- `speedups.fp4_cached_fused_dx_backward_estimate_vs_dense`
- `speedups.fused_dx_cached_pack_vs_dynamic_pack_train_step`
- `speedups.fused_dx_cached_pack_reuse_dy_up_vs_cached_pack_train_step`
- `speedups.fused_dx_cached_pack_reuse_overlap_vs_reuse_train_step`
- `speedups.fused_dx_cached_pack_plus_refresh_vs_dynamic_pack_train_step`
- `speedups.fused_dx_cached_pack_vs_dynamic_pack_grad_accum`

`--backward-weight-policy cache` 会在所有 FP4 LoRA 变体里常驻一份 compressed backward qweight，用于量化去掉 transient repack 后的训练 step 上限；默认 `repack` 不额外常驻第二份 backbone。

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

FP4 activation-cache `dA` 接入训练接口后的 BF16 短测，`fuse_lora_dx=True, cache_fused_lora_dx=True, fp4_activation_cache_d_lora_down=True`：

| shape | saved x bytes | FP4 act cache bytes | cache reduction | exact cached-pack step ms | FP4-cache dA step ms | FP4-cache / exact | dA FP4-cache vs exact rel_l2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048^2, rank32 | 8.00 MiB | 2.25 MiB | 3.56x | 0.2635 | 0.3624 | 0.73x | ~9.8e-2 |
| 4096^2, rank32 | 32.00 MiB | 9.00 MiB | 3.56x | 0.9499 | 1.3145 | 0.72x | ~9.7e-2 |

说明：

- `backward estimate = train_step - train_graph_forward`，用于判断 backward 优化方向，不是单独 CUDA event 包住 backward 的精确拆分。
- `fuse_lora_dx=True` 会把 `dX_lora = (dY @ B) @ A` 的第二段并入 FP4 dX epilogue，但 LoRA 参数梯度仍用 dense BF16/FP16 matmul 保精度。
- `fuse_lowrank_forward=True` 是 opt-in forward 消融选项：当 `lowrank_dtype == weight dtype` 时走 Nunchaku 原生 `quantize_w4a4_act_fuse_lora + gemm_w4a4` low-rank epilogue，把 trainable task LoRA forward 并入 FP4 主分支；如果有 frozen residual，它仍作为 dense side branch 追加。验证脚本用 `native_fused_forward` 标记原生路径，并用 `5e-4` rel_l2 tolerance 报告它相对当前 BF16/FP16 调度公式的差异。默认关闭。
- `cache_fused_lora_dx=True` 只缓存 LoRA packed A/B，不缓存第二份 FP4 backbone；参数 version 变化时会自动刷新。
- `zero_lora_up_fast_path=True` 是默认开启的 zero-init 首步优化：当 `init="zero"` 且 `lora_up` 的版本仍等于初始化后的零张量版本时，forward 跳过 LoRA out / native low-rank epilogue，backward 跳过 LoRA dX 和 `dA`，只保留 `dB=dY.T@(x@A.T)`；同时初始 `refresh_fused_lora_forward_caches/refresh_fused_lora_dx_caches` 不生成 packed LoRA cache。`optimizer.step()` 或 adapter load 改变 `lora_up` 后该 fast path 自动失效，post-step hook 再刷新 packed cache。若 `overlap_lora_grad=True` 且行数达到门槛，zero-up backward 会把 FP4 main dX、`dB` 和可选 residual dX 放到多 stream 并行。
- `backward_weight_policy="repack"` 是默认策略：每次 backward transient repack 出 `W^T` 的 packed FP4 权重，只预存转置后的 scale，不额外常驻第二份 backbone。`"cache"` 是显式 opt-in：常驻一份 compressed backward qweight，用显存换掉 repack 开销。RTX 5090 4096/rank32 BF16 短测中，cache train step 相对 repack 为 `1.056x`，4-step accumulation 为 `1.050x`；额外 qweight 为 dense BF16 权重的 `25%`、forward qweight 的 `1.0x`。
- `fp4_activation_cache_d_lora_down=True` 是显存/近似训练模式：forward 保存主分支已有的 `qact + ascales` 而不是 BF16/FP16 `x`。`fp4_activation_cache_d_lora_down_backend="fused"` 是默认值，直接用 fused CUDA kernel 从 FP4 cache 算 `dA`，避免 backward 临时物化 dense `x_hat`；`"dequant_gemm"` 会先反量化出 dense `x_hat` 再走 torch GEMM，通常更快但有额外 transient 显存。该模式要求 `cache_lora_act=True`，当前不支持 `overlap_lora_grad` 或 `reuse_fused_dy_up_for_d_lora_down`。
- `reuse_fused_dy_up_for_d_lora_down=True` 是 opt-in 实验选项：复用 fused dX quantize kernel 产生的 `dY @ B`，避免额外 dense `dY @ B` matmul。FP16 走 packed decode，有小量 `dA` 误差；BF16 走 dual dense `dy_up` 输出，保持 `dA` 与手写 BF16 matmul 对齐。
- `overlap_lora_grad=True` 要求同时打开 `fuse_lora_dx=True` 和 `cache_fused_lora_dx=True`，用多 CUDA stream 重叠 transient FP4 repack、fused dX、`dB` GEMM 和 `dA` GEMM；exact overlap 支持 frozen residual branch，但 residual dX 保持 dense 计算。
- `overlap_lora_grad_min_rows=4096` 是默认 auto gate：小于该 flattened row 数时自动回落到 sequential cached fused-dX 路径，避免 2048 形状上多 stream 调度变慢；传 `--overlap-lora-grad-min-rows 0` 可强制 always-overlap 做消融。
- `NUNCHAKU_FP4_LORA_CACHE_OVERLAP_RESOURCES=1` 是 stream/event 资源复用消融开关，默认关闭。RTX 5090 短测中 1024 forced-overlap 约 `1.01x`，4096 主形状约 `0.974x`，因此不作为默认优化；后续若要做 CUDA Graph/多 step capture 再重新评估。
- BF16 下 `reuse_fused_dy_up_for_d_lora_down=True` 使用 dual quantize 输出 dense `dy_up`，不复用 packed 近似中间量；`fuse_frozen_residual_dx=True` 仍不用于 BF16。
- BF16 下 `fp4_activation_cache_d_lora_down=True` 可省 saved `x` 约 `3.56x`，但 `dA` 相对 exact saved-x 约 `1e-1` rel_l2，且当前 fused dA kernel 仍慢；4096 train step 只有 exact cached-pack 的约 `0.72x`。它是显存压力模式，不是默认性能模式；如果显存允许临时 dense `x_hat`，可把 backend 切到 `"dequant_gemm"` 做速度消融。
- FP16 下如果同时打开 `reuse_fused_dy_up_for_d_lora_down=True`，`overlap_lora_grad=True` 会复用 decoded packed `dY @ B`，这是更快但有小量 `dA` 误差的实验路径；BF16 同一开关使用 dual dense `dy_up`，`dA` rel_l2 为 0。
- reuse-based overlap 目前不支持 frozen residual branch；高层 `fp4_lora_finetune_config` 遇到 reuse + frozen residual 会自动关闭 overlap。
- `activation_checkpoint=True` 是逐 `NunchakuFP4LoRALinear` 的局部 checkpoint，只能省该算子内部的 `lora_act` 等 saved tensors；真正要省多层输入 activation，应在 transformer block/segment 外层用 `torch.utils.checkpoint`。
- BF16 单步下 cached-pack fused dX 相比 dynamic-pack fused dX 训练 step 快 `1.026x`；每步刷新 cache 后仍快 `1.010x`。FP16 单步下 cached-pack 约 `1.012x`，每步刷新后基本持平。
- Gradient accumulation 会摊薄 cache refresh 开销；accumulation 数字对测量顺序更敏感，建议看多轮结果再定默认策略。
- `forward_fp4_vs_dense` 的误差是 FP4 量化相对 dense full precision 权重的误差，不是 wrapper correctness；wrapper correctness 请看 `validate_native_fp4_lora_training.py`。

Zero-init LoRA-up fast path 消融：

```bash
python benchmarks/benchmark_fp4_lora_zero_fast_path.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx \
  --overlap-lora-grad \
  --warmup 5 \
  --iters 20
```

如果要测试 throughput preset 对应的 native forward epilogue：

```bash
python benchmarks/benchmark_fp4_lora_zero_fast_path.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lowrank-forward \
  --fuse-lora-dx \
  --cache-fused-lora-dx \
  --overlap-lora-grad \
  --warmup 5 \
  --iters 20
```

RTX 5090 BF16 4096/rank32 短测：

| config | baseline train step ms | zero-up fast train step ms | speedup | initial packed cache |
| --- | ---: | ---: | ---: | --- |
| fused dX cached-pack | 0.9058 | 0.7439 | 1.218x | fast path `0/0` vs baseline forward/dX `1/1` |
| throughput fused forward + fused dX | 0.8768 | 0.7994 | 1.097x | fast path `0/0` vs baseline forward/dX `1/1` |
| fused dX cached-pack + zero-up overlap | 0.8371 | 0.7254 | 1.154x | fast path `0/0` vs baseline forward/dX `1/1` |
| throughput fused forward + fused dX + zero-up overlap | 0.7713 | 0.7260 | 1.062x | fast path `0/0` vs baseline forward/dX `1/1` |

`benchmark_fp4_lora_zero_fast_path.py` 同时报告 `fast_active_after_lora_up_update=false`，确认 `lora_up` 一旦更新就不会继续跳过 LoRA 分支。throughput 组合里 disabled baseline 的 native fused-forward 会产生近似 `lora_act`，因此 `d_lora_up_baseline_vs_exact` 约 `7.1e-4`；fast path 使用 dense `x@A.T`，`d_lora_up_fast_vs_exact` 为 0。
单看 zero-up fast path 自身，`overlap_lora_grad=True` 把 fused-dX 形状 `0.7439 -> 0.7254ms`（`1.025x`），throughput 形状 `0.7994 -> 0.7260ms`（`1.101x`）。带 frozen residual 的 throughput overlap 短测为 `1.0146 -> 0.9908ms`（`1.024x`），收益较小但 correctness 通过。

Fused `dY @ B` 复用消融：

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

RTX 5090 correctness：

- FP16 decoded packed `dy_up`：`d_lora_down` rel_l2 约 `3.35e-5`。
- BF16 dual dense `dy_up`：`d_lora_down` rel_l2 为 `0`；强制 reuse+overlap correctness 也通过。

性能上它是噪声敏感的小优化。4096/rank32 BF16 短测中，reuse 相对 cached-pack 单步 `1.014x`，reuse+overlap 相对 reuse 单步 `1.032x`，backward estimate 相对 dense `2.086x`。FP16 两次短测中，单步相对 cached-pack 约 `0.968x-1.018x`，gradient accumulation per micro-step 约 `1.016x-1.036x`。建议只在实际训练循环里确认收益后启用。

BF16 exact overlap 消融：

```bash
python benchmarks/validate_native_fp4_lora_training.py \
  --m 257 \
  --in-features 3072 \
  --out-features 3584 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --fuse-lora-dx \
  --cache-fused-lora-dx \
  --overlap-lora-grad \
  --overlap-lora-grad-min-rows 0
```

RTX 5090 correctness：`dX` rel_l2 `1.55e-4`，`d_lora_down` rel_l2 `0`。

4096/rank32 BF16 短测，`benchmark_native_fp4_lora_training.py --warmup 10 --iters 30 --grad-accum-steps 4`：

| path | train step ms | backward estimate ms | grad accum per micro-step ms | speedup vs dense step |
| --- | ---: | ---: | ---: | ---: |
| cached-pack | 0.9314 | 0.5980 | 0.9594 | 2.168x |
| exact overlap | 0.8830 | 0.5496 | 0.8956 | 2.287x |

当前结论：BF16 exact overlap 不降低 LoRA 梯度精度。默认 `overlap_lora_grad_min_rows=4096` 下，2048 形状自动回落、避免 forced overlap 退化；4096 形状单步相对 cached-pack 快 `1.055x`，4-step gradient accumulation 中快 `1.071x`。如果同时打开 `reuse_fused_dy_up_for_d_lora_down=True` 且不使用 frozen residual，BF16 会用 dual dense `dy_up` 保持 `dA` 精确。

FP16 overlap 消融：

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
  --reuse-fused-dy-up-for-d-lora-down \
  --overlap-lora-grad
```

RTX 5090 correctness：`dX` rel_l2 `1.81e-5`，`d_lora_down` rel_l2 `3.36e-5`。

4096/rank32 FP16 短测，`benchmark_native_fp4_lora_training.py --warmup 10 --iters 30 --grad-accum-steps 4`：

| path | train step ms | backward estimate ms | grad accum per micro-step ms | speedup vs dense step |
| --- | ---: | ---: | ---: | ---: |
| cached-pack | 0.9221 | 0.5824 | 0.9443 | 1.943x |
| reuse packed `dY @ B` | 0.8984 | 0.5587 | 0.9296 | 1.994x |
| reuse + overlap | 0.8909 | 0.5512 | 0.8962 | 2.011x |

当前结论：overlap 单步收益较小，`1.008x` vs reuse；在 4-step gradient accumulation 中收益约 `1.037x` vs reuse。它适合作为真实训练 loop 里的 opt-in latency-smoothing/overlap 选项，不作为默认路径。

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
  --backward-weight-policy repack \
  --warmup 5 \
  --iters 10
```

结果会写到：

- `results/latest_native_fp4_lora_training_breakdown.json`

如果要单独判断 `dA/dB` 低秩梯度子图是否值得写专用 kernel，跑：

```bash
python benchmarks/benchmark_fp4_lora_lowrank_grad.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --ranks 16,32,64,128 \
  --dtype bf16 \
  --warmup 5 \
  --iters 10
```

结果会写到：

- `results/latest_fp4_lora_lowrank_grad.json`

如果要判断 overlap helper 的 Python-side stream/event 分配是否值得缓存，跑：

```bash
python benchmarks/benchmark_fp4_lora_overlap_resource_cache.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --overlap-lora-grad-min-rows 4096 \
  --warmup 10 \
  --iters 50
```

结果会写到：

- `results/latest_fp4_lora_overlap_resource_cache.json`

当前 5090 结论：缓存 stream/event 资源对 4096 主形状是轻微负收益（约 `0.974x`），所以默认保持每次 helper 内创建资源；该脚本只作为消融工具。

如果要拆开 FP4 dX 主路径，跑：

```bash
python benchmarks/benchmark_fp4_dx_pipeline.py \
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

- `results/latest_fp4_dx_pipeline.json`

RTX 5090 短测，`M=N=K=4096, rank=32`：

| dtype | backward estimate ms | fused dX cached-pack ms | dense LoRA grad pair ms | LoRA grad share | LoRA pack refresh ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.6195 | 0.2152 | 0.0761 | 12.3% | 0.0388 |
| FP16 | 0.6467 | 0.2275 | 0.0396 | 6.1% | 0.0128 |

追加 `--backward-weight-policy cache` 时，breakdown 会同时报告 `latency_ms.backward_qweight_policy_access`、`latency_ms.refresh_backward_qweight_cache` 和 `backward_weight_cache_bytes.cached_backward_qweight_vs_dense_weight`，用于区分 transient repack、cache hit 和 cache 预热成本。

低秩梯度子图短测，`M=N=K=4096, rank=32`：

| dtype | sequential dA+dB ms | reuse existing dy_up ms | reuse speedup | two-stream overlap ms | overlap vs sequential |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.0855 | 0.0623 | 1.37x | 0.0928 | 0.92x |
| FP16 | 0.0536 | 0.0390 | 1.37x | 0.0748 | 0.72x |

FP4 dX pipeline 短测，`M=N=K=4096, rank=32`：

| dtype | full dX ms | quantize dY ms | repack W^T ms | prequantized GEMM ms | cached-qweight upper bound | fused LoRA dX ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.1837 | 0.0365 | 0.0390 | 0.1068 | 1.33x | 0.2157 |
| FP16 | 0.1858 | 0.0360 | 0.0391 | 0.1072 | 1.34x | 0.2281 |

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
- 单独给 `dA/dB` 加 CUDA stream overlap 在 5090 上反而变慢；后续低秩梯度 kernel 要优先围绕复用/消除 `dy_up` 中间量设计。
- FP4 dX 主路径中 prequantized GEMM 占比最大，`dY` quantize 和 transient repack 各约 20%。预存 `W^T` 的 cached-qweight 消融上界只有约 `1.33x-1.34x`，且会带来第二份 backbone 内存，不作为默认方案。
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
    fp4_lora_finetune_config,
    fp4_lora_parameter_groups,
    fp4_lora_peft_state_dict,
    fp4_lora_sensitivity_policy_from_report,
    fp4_lora_state_dict,
    freeze_non_fp4_lora_parameters,
    load_fp4_lora_peft_state_dict,
    load_fp4_lora_state_dict,
    prepare_fp4_lora_finetuning,
    register_fp4_lora_cache_refresh_hook,
    refresh_fused_lora_forward_caches,
    refresh_fused_lora_dx_caches,
)

cfg = fp4_lora_finetune_config(
    mode="balanced",
    rank=32,
    dtype=torch.bfloat16,
    lowrank_dtype=torch.bfloat16,
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

prepared = prepare_fp4_lora_finetuning(
    model,
    config=cfg,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    exclude_modules=("lm_head",),
    config_overrides=sensitive_overrides,
    # 可直接读取 Llama module sensitivity scan；手写 config_overrides 优先级最高。
    sensitivity_report="results/llama_module_fp4_sensitivity_20260321_202421.json",
    sensitivity_rank_bump_ratio=1.05,
    sensitivity_exclude_ratio=10.0,
    sensitivity_rank_scale=2.0,
    lr=1e-4,
)
model = prepared.model

optimizer = torch.optim.AdamW(prepared.optimizer_param_groups, eps=1e-4)
cache_hook = prepared.register_cache_refresh_hook(optimizer)

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
如果已经通过 sensitivity scan 发现某些完整模块路径不适合 FP4，可以把 JSON 直接传给 `prepare_fp4_lora_finetuning(..., sensitivity_report=...)`：超过 `sensitivity_exclude_ratio` 的模块自动加入 `exclude_modules` 保持 BF16/FP16，超过 `sensitivity_rank_bump_ratio` 但未被排除的模块自动提高 LoRA rank。`model.` 前缀会自动补一个去前缀 alias，因此同一份 LlamaForCausalLM 报告也可用于裸 `model.model` 子模块。手写 `config_overrides` 优先级最高，其次 activation/grad outlier 报告，最后是 sensitivity 报告。

`fp4_lora_finetune_config` 提供四种预设，均采用“frozen residual_svd 量化补偿 + zero-init task LoRA”的推荐形态：

| mode | 用途 | 关键开关 |
| --- | --- | --- |
| `accuracy` | 精度优先 / 调试 | `full_svd` 初始化，LoRA dX 走 dense BF16/FP16，关闭 overlap |
| `balanced` | 默认推荐 | `svd_lowrank`，fused cached LoRA dX，exact `dA/dB`，大 batch 自动 overlap |
| `throughput` | 速度消融 | 在 `balanced` 基础上打开 fused low-rank forward；FP16 会自动 fused frozen-residual dX 并关闭 overlap |
| `memory_saving` | 显存压力模式 | fused cached LoRA dX，但用 FP4 activation cache 计算近似 `dA`，默认 fused `dA` backend，自动关闭 overlap |

`memory_saving` 的 `dA` backend 可显式选择：

```python
cfg = fp4_lora_finetune_config(
    mode="memory_saving",
    fp4_activation_cache_d_lora_down_backend="dequant_gemm",  # 或默认 "fused"
)
```

选择原则：`"fused"` 不物化 dense `x_hat`，峰值显存更低；`"dequant_gemm"` 会临时物化 `x_hat`，但当前在 5090 上通常比 fused 原型更快。

如果要 opt-in 复用 fused dX quantize kernel 里的 `dy_up`，可以显式打开：

```python
cfg = fp4_lora_finetune_config(
    mode="balanced",
    dtype=torch.bfloat16,
    lowrank_dtype=torch.bfloat16,
    reuse_fused_dy_up_for_d_lora_down=True,
)
```

该开关要求 `dtype == lowrank_dtype`，且不支持 `memory_saving`。如果同时使用 frozen residual，高层 config 会自动关闭 reuse-based overlap，保留顺序 dense residual dX 路径。

验证这些预设是否能实际跑 forward/backward/optimizer step：

```bash
python benchmarks/validate_fp4_lora_training_policies.py
python benchmarks/validate_fp4_lora_training_policies.py --dtype fp16 --lowrank-dtype fp16 --modes throughput --steps 2
python benchmarks/validate_fp4_lora_training_policies.py --modes balanced memory_saving --backward-weight-policy cache --steps 2
```

模型级性能/显存消融使用高层 `prepare_fp4_lora_finetuning` 入口，而不是手写替换：

```bash
python benchmarks/benchmark_fp4_lora_prepare_policies.py \
  --batch 8 \
  --hidden 256 \
  --layers 2 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --warmup 3 \
  --iters 5
```

输出 `results/latest_fp4_lora_prepare_policies.json`，默认比较 dense LoRA baseline 与 `accuracy/balanced/throughput/memory_saving_fused/memory_saving_dequant_gemm`，并报告每个 FP4 preset 的 `latency_ms.train_step_with_optimizer`、`throughput.samples_per_second`、`peak_memory_bytes.train_step_delta`、`cache_summary.total_cache_bytes`、`initial_forward_vs_dense`、相对 `balanced` 的 speedup 和 `relative_to_dense_lora.train_step_speedup`。

如果要测试第二份 compressed backward qweight 的上限收益，追加 `--backward-weight-policy cache`。默认仍是 `repack`，避免常驻第二份 backbone；cache 策略会在 `prepare_fp4_lora_finetuning(..., refresh_caches=True)` 时预热，并在结果中报告 `refreshed_backward_weight_count`。

如果要把 BF16/FP16 `dy_up` 复用纳入模型级消融，追加 `--include-reuse-policies`：

```bash
python benchmarks/benchmark_fp4_lora_prepare_policies.py \
  --batch 8 \
  --hidden 256 \
  --layers 2 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --modes balanced \
  --include-reuse-policies \
  --no-frozen-residual \
  --warmup 3 \
  --iters 5
```

该选项会为 `balanced/throughput` 增加 `*_reuse_dy_up` 记录；当 frozen residual 开启时，高层 config 会自动关闭 reuse-based overlap，因此用 `--no-frozen-residual` 可以观察 reuse+overlap 的上限。RTX 5090 短测中，TinyTransformer 默认形状 `batch=8, hidden=256, layers=2` 的 `balanced_reuse_dy_up` 相对 `balanced` 为 `0.968x`，说明小 M 场景额外同步/调度开销会压过收益；4096 单层 benchmark 中 BF16 reuse 与 reuse+overlap 分别为 `1.014x` 和 `1.032x`。因此该策略保持 opt-in，应按真实训练形状决定是否启用。

RTX 5090 验证结果：

- BF16 `accuracy/balanced/throughput/memory_saving`：全部通过，LoRA A/B 更新、frozen residual 不变、optimizer cache hook 按需运行。
- FP16 `throughput`：自动配置 `fuse_frozen_residual_dx=True` 且 `overlap_lora_grad=False`，通过。

`prepare_fp4_lora_finetuning` 是推荐的真实微调入口，会一次性完成：

- 按 `target_modules/exclude_modules/config_overrides/outlier_report/sensitivity_report` 替换模型 Linear。
- 冻结所有非 LoRA 参数，只保留 LoRA A/B 和可选 bias 可训练。
- 按需刷新 native fused forward cache 和 fused dX cache。
- zero-init task LoRA 的首步不会预热 packed LoRA forward/dX cache；`optimizer.step()` 后版本失效，cache refresh hook 会按需生成新 cache。
- opt-in `backward_weight_policy="cache"` 时预热 compressed backward qweight；默认 `repack` 不额外常驻第二份 backbone。
- 返回 `FP4LoRAPrepareResult.cache_summary`，记录当前实际常驻的 packed LoRA forward cache、packed LoRA dX cache、backward qweight cache 和相对 dense weight 的字节比例；native fused forward 会在 `prepare(..., refresh_caches=True)` 时预热 forward cache。
- 返回 LoRA-only `optimizer_param_groups`，可直接传给 AdamW/ZeRO/FSDP 外层 optimizer。
- 返回 `FP4LoRAPrepareResult.register_cache_refresh_hook(optimizer)`，用于 optimizer step 后 eager refresh packed LoRA forward/dX cache；`hook.last_fused_lora_forward_refresh_count` 和 `hook.last_fused_lora_dx_refresh_count` 分别记录两类动态 cache，`hook.last_backward_weight_cache_count` 只报告静态 backward qweight cache 是否常驻，不在每步重复 repack。

验证 prepare 接口：

```bash
python benchmarks/validate_fp4_lora_prepare.py
python benchmarks/validate_fp4_lora_prepare.py --dtype fp16 --lowrank-dtype fp16 --mode throughput --batch 4 --hidden 128
python benchmarks/validate_fp4_lora_prepare.py --mode memory_saving --batch 4 --hidden 128 --fp4-activation-cache-d-lora-down-backend dequant_gemm
```

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
- `fuse_lowrank_forward=True` 可用于测试 native forward epilogue 收益；它会改变 task LoRA activation 的生成路径和低秩分支归约顺序，验证脚本按 `5e-4` rel_l2 tolerance 报告该近似。
- FP16 下可打开 `fuse_frozen_residual_dx=True`，把 task LoRA 和 frozen residual 的 dX 一并打包进 fused epilogue；BF16 下该路径目前误差偏大，默认关闭。

### 10.5.1 residual_svd 初始化消融

比较 `zero`、trainable `residual_svd`、以及推荐的 frozen `residual_svd + zero task LoRA` 初始化策略：

```bash
python benchmarks/benchmark_fp4_lora_initialization.py \
  --m 2048 \
  --in-features 2048 \
  --out-features 2048 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --warmup 5 \
  --iters 10
```

输出：

- `results/latest_fp4_lora_initialization.json`
- `policies.*.forward_vs_dense`
- `policies.*.construct_s`
- `derived.*_error_reduction_vs_zero`
- `derived.frozen_lowrank_construct_speedup_vs_full`

RTX 5090 BF16 `2048^2, rank32, weight_std=0.02` 短测：

| policy | residual SVD method | forward rel_l2 vs dense | error reduction vs zero | construct s | train step ms |
| --- | --- | ---: | ---: | ---: | ---: |
| FP4 + zero LoRA | none | 1.4377 | 1.00x | 0.0834 | 0.4203 |
| trainable residual_svd LoRA | full_svd | 1.3943 | 1.031x | 0.1628 | 0.3021 |
| frozen residual_svd + zero LoRA | full_svd | 1.3943 | 1.031x | 0.1547 | 0.3190 |
| trainable residual_svd LoRA | svd_lowrank | 1.4013 | 1.026x | 0.0720 | 0.2654 |
| frozen residual_svd + zero LoRA | svd_lowrank | 1.4013 | 1.026x | 0.0042 | 0.3188 |

结论：

- `residual_svd` 初始化能降低初始 FP4 量化误差；随机权重短测中收益约 `2.6%-3.1%` rel_l2，真实 outlier 层需要结合 sensitivity scan 单独评估。
- `svd_lowrank` 保留了大部分误差收益，同时显著降低 frozen residual 构造开销；适合大模型批量替换时先用作默认。
- trainable `init="residual_svd"` 如果搭配 `fuse_lora_dx=True`，BF16 下较大的 residual factors 会放大 fused LoRA dX 近似误差；需要精确梯度时用 dense LoRA dX，或采用推荐的 `frozen_residual_init="residual_svd" + init="zero"`。

### 10.5.2 单层微调收敛验证

验证推荐训练形态是否真的可优化：冻结 FP4 backbone 和 frozen `residual_svd` 量化补偿，只训练 zero-init task LoRA。默认 `target_base=fp4_initial`，即 teacher 目标为初始 `FP4 + frozen residual` 输出再叠加一个低秩 task delta；这样不会要求 LoRA 去拟合高秩 FP4 量化误差，验证点集中在 LoRA 梯度、optimizer 和 fused dX cache 路径。

```bash
python benchmarks/validate_fp4_lora_finetune_convergence.py
```

输出：

- `results/latest_fp4_lora_finetune_convergence.json`
- `loss.initial`
- `loss.final`
- `loss.final_over_initial`
- `errors.final_vs_target`
- `errors.predicted_delta_vs_target_delta_from_initial`
- `checks.*`

RTX 5090 BF16 默认短测，`m=256, in=512, out=768, rank=32, target_rank=8, steps=80`：

| target base | initial loss | final loss | final / initial | final vs target rel_l2 | fitted delta rel_l2 | checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP4 initial + teacher low-rank delta | 1.0753e-2 | 2.9362e-6 | 2.7307e-4 | 3.5572e-3 | 1.6525e-2 | pass |

验证项：

- loss 显著下降，默认要求 `final / initial < 0.35`。
- `lora_down/lora_up` 均发生更新，梯度和 loss 全部 finite。
- `frozen_residual_down/up` 保持不变，确认 residual branch 是 frozen buffer。
- 只有 LoRA A/B 可训练；打开 `--train-bias` 时才额外训练 bias。
- 默认动态 packed cache 的 optimizer post-step refresh hook 会运行。

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

## 10.9 FP4 LoRA activation cache policy 消融

如果你想判断“能不能不保存 BF16/FP16 `x`，改存 forward 已有的 FP4 activation cache”，跑：

```bash
python benchmarks/benchmark_fp4_lora_activation_cache_policy.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --warmup 10 \
  --iters 30
```

输出：

- `results/latest_fp4_lora_activation_cache_policy.json`
- `cache_bytes.fp4_qact_plus_fp8_ascales_padded`
- `derived.fp4_cache_reduction_vs_unpadded_x`
- `latency_ms.fp4_cache_dequant_only`
- `latency_ms.fp4_cache_dequant_plus_d_lora_down`
- `latency_ms.fp4_cache_fused_d_lora_down`
- `implementation.module_default_fp4_activation_cache_d_lora_down_backend`
- `implementation.fastest_measured_fp4_activation_cache_d_lora_down_backend`
- `implementation.fp4_cache_fused_d_lora_down`
- `errors.d_lora_down_fp4_cache_vs_saved_x`
- `errors.d_lora_down_fp4_cache_fused_vs_dequant_gemm`

这个脚本复用 native forward quantize 生成的 `qact/ascales`，再比较三条 `dA` 路径：

```text
dA_ref          = (dY @ B).T @ x
dA_fp4_cache    = (dY @ B).T @ dequant(qact, ascales)
dA_fp4_fused    = fused((dY @ B).T, qact, ascales)
```

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

当前结论：

- `qact + ascales` 的缓存体积是 BF16/FP16 `x` 的约 `28.1%`，理论显存节省明确。
- 但 `dA` 直接用 FP4-dequant `x_hat` 会引入约 `1e-1` rel_l2 的 LoRA A 梯度误差，不适合作为默认精度路径。
- naive `dequant -> dense x_hat -> GEMM` 需要在 backward 重新物化 dense `x_hat`，4096 形状比直接用 saved BF16 `x` 慢约 `5.1x`。
- 已加入 `fp4_activation_cache_lora_down_grad` fused CUDA 原型，避免 dense `x_hat` 中间张量；rank<=32 使用 `kVec=4,rVec=16,threads=128`，rank<=512 使用 `kVec=3,rVec=32,threads=128`，rank>512 回落 `kVec=2,rVec=16`。tile sweep 中 rank32 的 4096 fused `dA` 从约 `0.391ms` 降到 `0.306ms`，约 `1.28x`；rank64 从约 `1.50ms` 降到 `0.99ms`，约 `1.52x`；rank128 从约 `3.34ms` 降到 `1.86ms`，约 `1.79x`；rank256 从约 `5.72ms` 降到 `3.68ms`，约 `1.55x`；rank512 从约 `11.54ms` 降到 `7.23ms`，约 `1.60x`；候选记录见 [docs/fp4_kernel_research_notes.md](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/docs/fp4_kernel_research_notes.md)。
- 这个 fused 原型仍慢于 `dequant + GEMM`。本轮 5090 复测 4096/rank64：saved-x `dA` `0.0550ms`，`dequant_gemm` `0.2142ms`，fused `0.9916ms`；`implementation.fastest_measured_fp4_activation_cache_d_lora_down_backend="dequant_gemm"`。后续要继续优化 decode staging/reduction 或改成 tensor-core 友好的分块。
- 精度上 fused 原型对齐的是 `dequant(qact, ascales)` 近似路径，不解决 FP4 activation cache 本身带来的约 `1e-1` `dA` 误差。因此它仍应作为显存模式或近似训练消融，而非默认精度路径。

训练接口接入后的实际 autograd saved tensor 测量：

```bash
python benchmarks/benchmark_fp4_lora_saved_tensors.py \
  --m 4096 \
  --in-features 4096 \
  --out-features 4096 \
  --rank 32 \
  --dtype bf16 \
  --lowrank-dtype bf16 \
  --warmup 5 \
  --iters 10
```

可选加 `--fp4-activation-cache-d-lora-down-backend dequant_gemm` 测“临时物化 dense `x_hat` + GEMM”的训练 wrapper 开销；默认 `fused` 测最低 transient memory 路径。

输出：

- `results/latest_fp4_lora_saved_tensors.json`
- `shape.fp4_activation_cache_d_lora_down_backend`
- `saved_tensors.exact_cached_pack`
- `saved_tensors.fp4_activation_cache_d_lora_down`
- `saved_bytes.activation_context_reduction`
- `saved_bytes.all_saved_tensors_reduction`
- `speedups.fp4_activation_cache_d_lora_down_vs_exact_cached_pack`
- `errors.d_lora_down_vs_exact`

这个脚本用 `torch.autograd.graph.saved_tensors_hooks` 只包住 `module(x)`，直接检查 `_FP4LoRALinearFunction` 的 `ctx.save_for_backward`，不会把 loss backward 的额外临时保存算进去。

RTX 5090 BF16 短测：

| shape | exact activation context | FP4-cache activation context | context reduction | exact all saved | FP4-cache all saved | all-saved reduction | FP4-cache / exact step | dA rel_l2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048^2, rank32 | 8.125 MiB | 2.375 MiB | 3.42x | 8.375 MiB | 2.625 MiB | 3.19x | 0.71x | 9.83e-2 |
| 4096^2, rank32 | 32.25 MiB | 9.25 MiB | 3.49x | 32.75 MiB | 9.75 MiB | 3.36x | 0.77x | 9.78e-2 |

这里的 `activation context` 只统计 `saved_x/qact/ascales + saved_lora_act`；`all saved` 还包含 LoRA A/B 等权重引用。结论是：训练 wrapper 里真实 autograd context 也能拿到约 `3.4x` activation-cache 缩减，但当前近似 `dA` 路径通常仍慢于 exact cached-pack，所以仍定位为显存压力模式；4096 形状在 rank32 fast path 后从约 `0.74x` 提升到约 `0.77x`。

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
python benchmarks/validate_native_fp4_lora_training.py --m 129 --in-features 512 --out-features 768 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --backward-weight-policy cache
python benchmarks/validate_native_fp4_lora_pack.py --dtype bf16 --warmup 20 --iters 100
python benchmarks/validate_native_fp4_lora_modeling.py --batch 8 --hidden 256 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx
python benchmarks/validate_native_fp4_lora_modeling.py --batch 4 --hidden 128 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --fp4-activation-cache-d-lora-down --fp4-activation-cache-d-lora-down-backend dequant_gemm
python benchmarks/validate_fp4_lora_training_policies.py
python benchmarks/validate_fp4_lora_prepare.py
python benchmarks/validate_fp4_lora_prepare.py --dtype bf16 --lowrank-dtype bf16 --mode balanced --backward-weight-policy cache
python benchmarks/benchmark_fp4_lora_prepare_policies.py --batch 8 --hidden 256 --layers 2 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 3 --iters 5
python benchmarks/benchmark_fp4_lora_prepare_policies.py --batch 8 --hidden 256 --layers 2 --rank 32 --dtype bf16 --lowrank-dtype bf16 --modes balanced --backward-weight-policy cache --warmup 3 --iters 5
python benchmarks/benchmark_fp4_lora_prepare_policies.py --batch 8 --hidden 256 --layers 2 --rank 32 --dtype bf16 --lowrank-dtype bf16 --modes balanced --include-reuse-policies --no-frozen-residual --warmup 3 --iters 5
python benchmarks/benchmark_fp4_lora_backward_weight_policy.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10
python benchmarks/benchmark_fp4_lora_initialization.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10
python benchmarks/validate_fp4_lora_finetune_convergence.py
python benchmarks/analyze_fp4_lora_activation_grad_outliers.py --batch 4 --hidden 128 --layers 2 --steps 2 --rank 32 --override-rank 64 --dtype bf16 --lowrank-dtype bf16 --inject-outliers --outlier-channel 0 --outlier-scale 16
python benchmarks/benchmark_fp4_lora_outlier_overrides.py --batch 4 --hidden 128 --layers 2 --rank 32 --override-rank 64 --dtype bf16 --lowrank-dtype bf16 --warmup 3 --iters 5
python benchmarks/benchmark_fp4_lora_activation_checkpoint.py --batch 512 --hidden 1024 --layers 4 --rank 32 --dtype bf16 --lowrank-dtype bf16 --fuse-lora-dx --cache-fused-lora-dx --intermediate-activation silu --warmup 5 --iters 10
python benchmarks/benchmark_fp4_lora_activation_cache_policy.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 10 --iters 30
python benchmarks/benchmark_fp4_lora_saved_tensors.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10
python benchmarks/benchmark_native_fp4_lora_training.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10
python benchmarks/benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --dtype bf16 --warmup 10 --iters 30
python benchmarks/benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --dtype bf16 --warmup 10 --iters 30 --fuse-lowrank-forward
python benchmarks/benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --dtype fp16 --warmup 10 --iters 30
python benchmarks/benchmark_native_fp4_lora_training_breakdown.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10
```

RTX 5090 上 `benchmark_native_fp4_lora_dual_branch.py --m 2048 --in-features 2048 --out-features 2048 --rank 32 --frozen-residual-rank 32 --warmup 10 --iters 30`：

| path | dtype | train step ms | note |
| --- | --- | ---: | --- |
| task LoRA only, fused dX | bf16 | 0.2997 | task-only reference |
| dual branch, residual dense dX | bf16 | 0.3077 | `1.027x` overhead vs task-only；BF16 fused residual dX 暂不启用 |
| dual branch, residual exact overlap auto | bf16 | 0.3086 | `0.997x` vs dense residual；默认门槛回落，不再退化 |
| dual branch, residual forced overlap | bf16 | 0.4699 | `0.666x` vs dense residual；`--overlap-lora-grad-min-rows 0` 消融 |
| task LoRA only, fused dX | fp16 | 0.2712 | task-only reference |
| dual branch, residual dense dX | fp16 | 0.3128 | `1.153x` overhead vs task-only |
| dual branch, residual exact overlap auto | fp16 | 0.3115 | `1.004x` vs dense residual；默认门槛回落 |
| dual branch, residual fused dX | fp16 | 0.3130 | `0.999x` vs residual dense dX，`dX` rel_l2 `3.80e-4` |

RTX 5090 上 `benchmark_native_fp4_lora_dual_branch.py --m 4096 --in-features 4096 --out-features 4096 --rank 32 --frozen-residual-rank 32 --dtype bf16 --warmup 10 --iters 30`：

| path | dtype | train step ms | note |
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

task-branch native fused forward 消融，RTX 5090 上：

```bash
conda run -n triton python benchmarks/benchmark_native_fp4_lora_training.py \
  --m 4096 --in-features 4096 --out-features 4096 --rank 32 \
  --dtype bf16 --lowrank-dtype bf16 --warmup 5 --iters 10 --grad-accum-steps 4
```

| metric | default FP4+dense LoRA | native fused forward | result |
| --- | ---: | ---: | --- |
| forward inference ms | 0.3458 | 0.2723 | `1.270x` vs default |
| forward train graph ms | 0.3462 | 0.2711 | `1.277x` vs default |
| train step ms | 0.9963 | 0.9268 | `1.075x` vs default |
| forward rel_l2 vs default | - | `1.55e-4` | within `5e-4` |
| forward cache pack ms | - | 0.0232 | LoRA A/B packed cache refresh |

结论：native task forward 对 forward 和 4096 dual-branch train step 都有明确收益，但它相对当前 BF16/FP16 调度公式有约 `1e-4` 级别 rel_l2 差异，并需要额外一份 packed LoRA forward cache，因此仍作为 opt-in；balanced 默认关闭，throughput preset 打开。实际训练 step 的主要剩余瓶颈更偏向 dX 主路径、LoRA 参数梯度和 residual dX 融合。

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
  - frozen FP4 backbone + 可选 frozen residual low-rank + trainable BF16/FP16 task LoRA 微调接口；支持 opt-in `fuse_lowrank_forward` 原生 forward epilogue 消融、`fp4_activation_cache_d_lora_down` 显存/近似训练模式，并可通过 `fp4_activation_cache_d_lora_down_backend` 选择 `fused` 或 `dequant_gemm`
- `native_fp4.dequantize_fp4_activation`
  - 反解 native `qact/ascales` activation layout；`return_scales=False` 时走 CUDA fast path，供 FP4 activation cache 消融和后续 fused `dA` kernel 使用
- `native_fp4.fp4_activation_cache_lora_down_grad`
  - 直接从 native FP4 activation cache 计算 LoRA `dA` 的 fused CUDA 原型；rank<=32 使用 `kVec=4,rVec=16,threads=128` fast path，rank<=512 使用 `kVec=3,rVec=32,threads=128` fast path，rank>512 回落 `kVec=2,rVec=16`；用于显存/近似训练消融，当前不建议默认开启
- `native_fp4.FP4LoRAConfig`
  - 批量替换 Linear 时使用的配置对象，支持 `frozen_residual_rank/init`、`residual_svd_method`、`activation_checkpoint`、`reuse_fused_dy_up_for_d_lora_down`、`zero_lora_up_fast_path`、`fp4_activation_cache_d_lora_down`、`fp4_activation_cache_d_lora_down_backend` 和 FP16-only `fuse_frozen_residual_dx`
- `native_fp4.convert_linear_to_fp4_lora`
  - 按完整路径/后缀/子模块名匹配并替换 `torch.nn.Linear`
- `native_fp4.fp4_lora_config_overrides_from_outlier_report`
  - 从 outlier 诊断 JSON 自动生成 `config_overrides`
- `native_fp4.fp4_lora_sensitivity_policy_from_report`
  - 从真实模型 module sensitivity JSON 生成 rank bump `config_overrides` 和 BF16/FP16 `exclude_modules` 策略
- `native_fp4.freeze_non_fp4_lora_parameters`
  - 冻结非 LoRA 参数，只保留 LoRA A/B 可训练
- `native_fp4.iter_fp4_lora_named_parameters`
  - 枚举 LoRA-only 参数，默认不包含 bias
- `native_fp4.fp4_lora_parameter_groups`
  - 生成只含 LoRA adapter 的 optimizer 参数组
- `native_fp4.register_fp4_lora_cache_refresh_hook`
  - 在 `optimizer.step()` 后 eager refresh native fused forward 和 fused dX packed LoRA cache
- `native_fp4.fp4_lora_state_dict`
  - 导出 LoRA-only adapter checkpoint，不包含 FP4 backbone buffers
- `native_fp4.load_fp4_lora_state_dict`
  - strict 加载 LoRA-only adapter checkpoint，并清空 packed LoRA forward/dX cache
- `native_fp4.fp4_lora_peft_state_dict`
  - 导出 PEFT 风格 `lora_A/lora_B` adapter checkpoint，默认保留 padded effective rank 以保证无损
- `native_fp4.load_fp4_lora_peft_state_dict`
  - 加载 PEFT 风格 adapter checkpoint，支持 requested-rank 输入并清零 padded tail
- `native_fp4.refresh_fused_lora_forward_caches`
  - optimizer step 后显式刷新 native fused forward packed LoRA cache
- `native_fp4.clear_fused_lora_forward_caches`
  - 清空模型内所有 native fused forward packed LoRA cache
- `native_fp4.refresh_fused_lora_dx_caches`
  - optimizer step 后显式刷新 fused dX packed LoRA cache
- `native_fp4.clear_fused_lora_dx_caches`
  - 清空模型内所有 fused dX packed LoRA cache

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
- `latest_fp4_dx_pipeline.json`
  - FP4 dX 主路径拆解 benchmark，分离 `dY` 量化、transient repack、prequantized GEMM 和 fused LoRA epilogue
- `latest_native_fp4_backward_validation.json`
  - backward correctness
- `latest_native_fp4_lora_training_validation.json`
  - FP4 LoRA training wrapper correctness
- `latest_native_fp4_lora_training.json`
  - FP4 LoRA training benchmark
- `latest_fp4_lora_lowrank_grad.json`
  - FP4 LoRA 低秩梯度子图 benchmark，拆分 `dy_up/dA/dB`、复用 `dy_up` 和双 stream overlap
- `latest_fp4_lora_overlap_resource_cache.json`
  - overlap backward 中 stream/event 资源复用的消融，当前默认关闭该 cache
- `latest_fp4_lora_zero_fast_path.json`
  - zero-init `lora_up` 首步 fast path 的 correctness、cache skip 和 train-step speedup
- `latest_native_fp4_lora_dual_branch_bf16.json`
  - dual-branch BF16 residual dense dX benchmark
- `latest_native_fp4_lora_dual_branch_fp16.json`
  - dual-branch FP16 fused residual dX benchmark
- `latest_native_fp4_lora_modeling_validation.json`
  - 模型级 Linear 替换、参数冻结和 cache 管理验证
- `latest_fp4_lora_training_policies_validation.json`
  - `accuracy/balanced/throughput/memory_saving` 四种 FP4 LoRA 微调预设的 forward/backward/optimizer step 验证
- `latest_fp4_lora_prepare_validation.json`
  - 高层 `prepare_fp4_lora_finetuning` 接口的模型替换、冻结、optimizer 参数组和 cache hook 验证
- `latest_fp4_lora_prepare_policies.json`
  - 高层 `prepare_fp4_lora_finetuning` preset 相对 dense LoRA baseline 的模型级 train step、optimizer/cache hook、peak memory 和 forward 误差消融
- `latest_fp4_lora_initialization.json`
  - FP4 LoRA `zero`、trainable `residual_svd`、frozen `residual_svd` 初始化策略和 `full_svd/svd_lowrank` 后端消融
- `latest_fp4_lora_finetune_convergence.json`
  - frozen FP4 backbone + frozen residual_svd + zero-init task LoRA 的单层微调 loss 收敛验证
- `latest_fp4_lora_activation_grad_outliers.json`
  - FP4 LoRA activation / grad-output outlier 诊断和 rank/smooth 建议
- `latest_fp4_lora_outlier_override_overhead.json`
  - outlier-driven `config_overrides` 相对 base config 的 train-step 开销
- `latest_fp4_lora_activation_checkpoint.json`
  - activation checkpoint 显存/速度消融
- `latest_fp4_lora_activation_cache_policy.json`
  - FP4 activation cache 替代 saved BF16/FP16 `x` 的显存、速度、fused `dA` 原型和 `dA` 精度消融
- `latest_fp4_lora_saved_tensors.json`
  - FP4 activation-cache `dA` 训练接口的实际 autograd saved tensor、速度和精度消融

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
