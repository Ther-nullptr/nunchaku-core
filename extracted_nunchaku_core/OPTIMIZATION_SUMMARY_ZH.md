# Nunchaku Core 优化总结

## 1. 工作范围

本文总结 `extracted_nunchaku_core` 中已经完成的优化工作，重点包括：

- Nunchaku 原生 FP4 后端的独立抽取
- 前向纯 FP4 GEMM 与 FP4 + 16-bit 低秩分支混合算子的整理
- 低秩分支 fused / unfused 消融实验
- FP4 反向传播算子的设计、实现与逐步优化

当前实现面向 **NVIDIA GeForce RTX 5090**，走的是 Blackwell 原生 FP4 MMA 指令路径。

## 2. 实验环境

- GPU：`NVIDIA GeForce RTX 5090`
- Driver：`590.44.01`
- PyTorch：`2.9.1+cu128`
- CUDA runtime：`12.8`
- Conda 环境：`triton`

## 3. 交付内容

独立整理后的库位于：

- [extracted_nunchaku_core](/home/wyj24/projects/nunchaku/extracted_nunchaku_core)

核心目录和脚本：

- CUDA 后端：
  - [fp4_backend](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/fp4_backend)
- 原生扩展入口：
  - [fp4_native_ops.cpp](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/csrc/fp4_native_ops.cpp)
- Python 封装：
  - [operators.py](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/native_fp4/operators.py)
- 前向 benchmark：
  - [benchmark_native_fp4.py](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/benchmarks/benchmark_native_fp4.py)
- 反向 benchmark：
  - [benchmark_native_fp4_backward.py](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/benchmarks/benchmark_native_fp4_backward.py)
- 反向正确性验证：
  - [validate_native_fp4_backward.py](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/benchmarks/validate_native_fp4_backward.py)

## 4. 主要优化项

### 4.1 独立抽取原生 FP4 后端

目标：

- 把 Nunchaku 原仓库中的 FP4 CUDA 后端独立整理到 `core` 下
- 不再依赖原仓库运行时布局和原始 `.so`
- 让 `extracted_nunchaku_core` 具备真正的即插即用能力

已完成内容：

- 把 CUDA 后端整理到 `fp4_backend`
- 补齐 standalone build、pybind 和 Python 包装
- 适配 RTX 5090 / Blackwell 的 FP4 构建路径

结果：

- `extracted_nunchaku_core` 现在可以在本目录下独立编译和使用

### 4.2 前向纯 FP4 GEMM 抽取

目标：

- 只保留 FP4 主分支
- 提供独立的 PyTorch 封装
- 对比 PyTorch FP16 GEMM 的速度

结果来自：

- [latest_native_fp4.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4.json)

测量值：

- `fp16_ms = 0.6466 ms`
- `fp4_gemm_ms = 0.1391 ms`
- 相对 FP16 加速比：`4.648x`

结论：

- 纯 FP4 GEMM 是当前前向路径里最快的核心算子

### 4.3 前向 FP4 + 16-bit 低秩分支混合算子

目标：

- 提取论文核心思想：
  - 主分支使用 FP4
  - 残差分支使用 16-bit 低秩分支

结果来自：

- [latest_native_fp4.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4.json)

测量值：

- `fp4_hybrid_ms = 0.1831 ms`
- 相对 FP16 加速比：`3.532x`

结论：

- 混合算子比纯 FP4 稍慢，这是低秩分支额外计算带来的必然开销
- 但相对 FP16 基线仍然保持明显加速

### 4.4 低秩分支融合消融：fused vs unfused

目标：

- 验证“把 FP4 主分支和 BF16 低秩分支做融合”到底值不值
- 对比 naive unfused 实现方式的开销

结果来自：

- [latest_fp4_bf16_fusion_ablation.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_fp4_bf16_fusion_ablation.json)

测量值：

- `fp4_only_ms = 0.1379 ms`
- `fp4_bf16_fused_ms = 0.1810 ms`
- `fp4_bf16_unfused_ms = 0.2939 ms`
- fused 相对 FP16：`3.553x`
- unfused 相对 FP16：`2.188x`
- `unfused / fused = 1.624x`

结论：

- naive unfused 方案比 fused 方案慢约 **62.4%**
- 这说明论文里“融合低秩分支能显著减少开销”的结论在当前 5090 / FP4 路径下是成立的

数值精度方面：

- `fused_mae_vs_fp16 = 73.0114`
- `unfused_mae_vs_fp16 = 73.0114`
- `unfused_vs_fused_mae = 0.0325`

说明：

- fused 和 unfused 的数值表现几乎一致
- 差异主要是性能问题，不是精度问题

### 4.5 反向 dX：采用 transient repack，而不是永久存两份权重

目标：

- 支持 `dX = dY @ W^T` 的 FP4 backward 路径
- 避免常驻保存第二份 `qweight_bwd`

设计选择：

- 不预存第二份 FP4 压缩权重
- 仅保存 backward 需要的逻辑 scale
- 在 backward 时临时生成 `qweight_bwd`

为什么必须这样做：

- Blackwell FP4 的压缩布局不能直接通过“转置 view”复用给 `W^T`
- 如果同时常驻保存 forward 和 backward 两份 packed weight，压缩权重常驻内存会近似翻倍

结果：

- 当前实现避免了常驻双份压缩权重
- backward 仍能直接复用原生 FP4 GEMM 路径

### 4.6 CUDA repack kernel

目标：

- 用 CUDA 替换掉 Python reference repack

关键文件：

- [fp4_repack_cuda.cu](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/csrc/fp4_repack_cuda.cu)

结果来自：

- [latest_native_fp4_backward.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4_backward.json)

测量值：

- `repack_python_ms = 1.4378 ms`
- `repack_ms = 0.04136 ms`
- `repack_speedup_vs_python = 34.76x`

结论：

- Python repack 对 backward 来说开销过大，无法接受
- CUDA repack 把 repack 开销压到纯 FP4 `dX` 的约 `22.1%`

### 4.7 融合后的 backward dX

目标：

- 实现 `dX = dY @ (W^T + B^T A^T)` 的融合版本

结果来自：

- [latest_native_fp4_backward.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4_backward.json)

测量值：

- `fp16_dx_ms = 0.6078 ms`
- `fp4_dx_ms = 0.1874 ms`
- `fp4_dx_hybrid_unfused_ms = 0.2630 ms`
- `fp4_dx_hybrid_fused_ms = 0.2280 ms`

速度对比：

- 纯 FP4 backward `dX` 相对 FP16：`3.244x`
- hybrid unfused `dX` 相对 FP16：`2.312x`
- hybrid fused `dX` 相对 FP16：`2.666x`

结论：

- low-rank 分支 fusion 在 backward `dX` 上同样有效
- fused hybrid `dX` 比 unfused hybrid `dX` 快约 **13.3%**

### 4.8 全量 backward 的逐步优化

已实现的 full backward 版本：

1. `fp4_full_backward_unfused`
2. `fp4_full_backward_fused`
3. `fp4_full_backward_shared_recompute`
4. `fp4_full_backward_shared_cached`
5. `fp4_full_backward_shared_packed`
6. `fp4_full_backward_shared_dual`
7. `fp4_full_backward_shared_packed_overlap`

结果来自：

- [latest_native_fp4_backward.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4_backward.json)

| 方案 | 时延 (ms) | 相对 FP16 加速比 |
| --- | ---: | ---: |
| FP16 full backward | 0.7520 | 1.000x |
| FP4 full backward unfused | 0.3492 | 2.154x |
| FP4 full backward fused | 0.3117 | 2.413x |
| shared recompute | 0.3071 | 2.449x |
| shared cached | 0.2917 | 2.578x |
| shared packed | 0.2770 | 2.715x |
| shared dual | 0.2956 | 2.544x |
| shared packed overlap | 0.2673 | 2.813x |

每一版的意义：

- `shared_recompute`
  - 复用 fused `dX`
  - 但低秩中间量仍然重算

- `shared_cached`
  - 复用前向缓存 `x @ A`
  - 少一次低秩重算

- `shared_packed`
  - 复用 quantize 阶段产生的 packed `dY @ A`
  - 通过 decode 得到 `dB` 所需的 dense 低秩激活
  - 避免重复计算 dense `dY @ A`

- `shared_dual`
  - 在 quantize 阶段同时输出 packed 和 dense 两份 low-rank 激活
  - 目标是省掉 decode

- `shared_packed_overlap`
  - 数值路径沿用 `shared_packed`
  - 通过多 stream 调度，把 repack、`d_up`、`d_down` 和 fused `dX` 并发

优化收益：

- fused full backward 相对 unfused：
  - `0.3492 -> 0.3117 ms`
  - 快约 **10.7%**

- shared cached 相对 fused：
  - `0.3117 -> 0.2917 ms`
  - 快约 **6.4%**

- shared packed 相对 fused：
  - `0.3117 -> 0.2770 ms`
  - 快约 **11.1%**

- shared packed overlap 相对 shared packed：
  - `0.2770 -> 0.2673 ms`
  - 快约 **3.5%**

- shared packed overlap 相对 fused：
  - `0.3117 -> 0.2673 ms`
  - 快约 **14.2%**

### 4.9 dual-output quantize 消融

目标：

- 检查“直接在 quantize 阶段输出 dense `dY @ A`”是否一定比 packed + decode 更快

结果：

- `shared_dual_ms = 0.2956 ms`
- `shared_packed_ms = 0.2770 ms`

结论：

- 当前实现下，`shared_dual` 反而比 `shared_packed` 更慢
- 原因不是功能错误，而是 quantize 内部多了一份 dense 全局归约和写回
- 这份额外写回成本大于单独 decode kernel 的成本

这是一个很重要的负结果：

- “减少 kernel 数量”不等于一定更快
- 真正决定性能的是中间结果的写回方式和全局归约成本

## 5. 数值正确性

结果文件：

- [latest_native_fp4_backward_validation.json](/home/wyj24/projects/nunchaku/extracted_nunchaku_core/results/latest_native_fp4_backward_validation.json)

结论：

- `all_passed = true`

关键检查项：

- `qweight_bwd_cuda_matches_reference = true`
- `full_shared_packed_overlap_dx_matches_fused_rel_l2_lt_5e-4 = true`
- `full_shared_packed_overlap_up_rel_l2_lt_1e-5 = true`
- `full_shared_packed_overlap_down_rel_l2_lt_5e-4 = true`

代表性误差：

- packed low-rank decode vs dense reference：
  - `rel_l2 = 3.252e-4`

- 当前最优 full backward 路径 `shared_packed_overlap`：
  - `dx vs fused rel_l2 = 4.236e-6`
  - `lora_down_grad vs reference rel_l2 = 4.221e-4`

结论：

- 当前所有保留的优化路径都满足既定精度阈值

## 6. 关键设计决策

### 6.1 明确避免的方案

- 不常驻保存第二份 FP4 backward packed weight
- 不依赖原仓库运行时 `.so`
- 不依赖 FlashInfer 来执行原生 FP4 GEMM

### 6.2 当前最优技术路线

当前效果最好的 full backward 方案是：

1. backward 使用 transient CUDA repack 生成 `W^T` 需要的 packed weight
2. `dX` 走 fused FP4 主分支
3. `dA` 复用前向 cache
4. `dB` 复用 packed low-rank activation，再 decode
5. 通过多 CUDA stream 对 repack、`dX`、`dA`、`dB` 做 overlap

## 7. 关键 commit 时间线

- `eb7c8da` `extract: vendor standalone fp4 cuda backend under core`
- `140dbfd` `benchmark: refresh standalone fp4 speed and correctness reports`
- `379f7ae` `ablation: add unfused fp4+bf16 path and fusion benchmark`
- `be37b48` `benchmark: record fp4-bf16 fusion ablation results`
- `92c5e7b` `backward: add fp4 transient repack dx operators and benchmarks`
- `0e7fb0f` `backward: add cuda fp4 repack kernel and refresh benchmarks`
- `fdef60a` `backward: add cached full fp4 backward path and benchmarks`
- `3fba0bc` `backward: decode packed lowrank activations for dy reuse`
- `cc1f93a` `backward: add dual-output quantize ablation for dy reuse`
- `25f16b2` `backward: overlap repack with packed lowrank branches`

## 8. 当前最佳结果

前向：

- 纯 FP4 GEMM：**4.648x** vs FP16
- FP4 + 16-bit low-rank hybrid：**3.532x** vs FP16

反向：

- 纯 FP4 `dX`：**3.244x** vs FP16
- fused hybrid `dX`：**2.666x** vs FP16
- 最优 full backward `shared_packed_overlap`：**2.813x** vs FP16

## 9. 后续仍可继续推进的方向

目前最大的剩余优化点是：

- 把 `d_up = dY^T @ (x @ A)` 也并进同一次 `dY` 消费

当前状态是：

- `shared_packed_overlap` 已经做到较强的调度级融合
- 但 `d_up` 仍然是单独发射的一次 GEMM

所以目前的实现已经实现了明显的全局优化，但还不是最终意义上的单次 `dY` 全融合大 backward kernel。
