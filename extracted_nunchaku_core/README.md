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
python -c "from native_fp4 import NunchakuFP4GemmOp, NunchakuFP4LowRankOp, NunchakuFP4BackwardDXOp, NunchakuFP4LowRankBackwardDXOp; from native_fp8 import NunchakuFP8GemmOp; print('import ok')"
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
```

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
