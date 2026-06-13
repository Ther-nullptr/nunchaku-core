#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockM = 256;
constexpr int kWarpM = 32;
constexpr int kWarpK = 64;
constexpr int kNumWarps = 8;
constexpr int kWarpMTiles = 2;
constexpr int kWarpSize = 32;
constexpr int kFP4GroupSize = 16;

__device__ __forceinline__ float decode_fp4(uint8_t code) {
    constexpr float mag_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = mag_lut[code & 0x7];
    return (code & 0x8) ? -value : value;
}

__device__ __forceinline__ float decode_e4m3fn(uint8_t byte) {
    const int sign = byte >> 7;
    const int exp = (byte >> 3) & 0xF;
    const int mant = byte & 0x7;

    float value;
    if (exp == 0) {
        value = static_cast<float>(mant) * 0x1.0p-9f;
    } else {
        value = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
    }
    return sign ? -value : value;
}

template <typename scalar_t>
__global__ void dequantize_fp4_activation_kernel(
    const uint8_t* __restrict__ qact,
    const uint8_t* __restrict__ ascales,
    scalar_t* __restrict__ output,
    int rows,
    int cols,
    int k_tiles,
    int64_t total) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    const int row = static_cast<int>(index / cols);
    const int col = static_cast<int>(index - static_cast<int64_t>(row) * cols);

    const int m_block = row / kBlockM;
    const int row_in_block = row - m_block * kBlockM;
    const int warp = row_in_block / kWarpM;
    const int row_in_warp = row_in_block - warp * kWarpM;
    const int tile_m = row_in_warp / 16;
    const int row_in_tile = row_in_warp - tile_m * 16;
    const int row_high = row_in_tile / 8;
    const int row_low = row_in_tile - row_high * 8;

    const int k_tile = col / kWarpK;
    const int col_in_tile = col - k_tile * kWarpK;
    const int frag = col_in_tile / 8;
    const int pair = (col_in_tile & 0x7) >> 1;
    const int nibble = col_in_tile & 0x1;

    const int q_lane = row_low * 4 + (frag & 0x3);
    const int q_byte_in_uint4 = row_high * 4 + ((frag >= 4) ? 8 : 0) + pair;
    int64_t q_uint4_index = m_block;
    q_uint4_index = q_uint4_index * k_tiles + k_tile;
    q_uint4_index = q_uint4_index * kNumWarps + warp;
    q_uint4_index = q_uint4_index * kWarpMTiles + tile_m;
    q_uint4_index = q_uint4_index * kWarpSize + q_lane;

    const uint8_t packed = qact[q_uint4_index * 16 + q_byte_in_uint4];
    const uint8_t code = static_cast<uint8_t>((packed >> (nibble * 4)) & 0xF);

    const int scale_lane = row_low * 4 + tile_m * 2 + row_high;
    const int scale_group = col_in_tile / kFP4GroupSize;
    int64_t scale_index = m_block;
    scale_index = scale_index * k_tiles + k_tile;
    scale_index = scale_index * kNumWarps + warp;
    scale_index = scale_index * kWarpSize + scale_lane;
    scale_index = scale_index * 4 + scale_group;

    const float value = decode_fp4(code) * decode_e4m3fn(ascales[scale_index]);
    output[index] = static_cast<scalar_t>(value);
}

__device__ __forceinline__ float load_dequantized_fp4_activation(
    const uint8_t* __restrict__ qact,
    const uint8_t* __restrict__ ascales,
    int row,
    int col,
    int k_tiles) {
    const int m_block = row / kBlockM;
    const int row_in_block = row - m_block * kBlockM;
    const int warp = row_in_block / kWarpM;
    const int row_in_warp = row_in_block - warp * kWarpM;
    const int tile_m = row_in_warp / 16;
    const int row_in_tile = row_in_warp - tile_m * 16;
    const int row_high = row_in_tile / 8;
    const int row_low = row_in_tile - row_high * 8;

    const int k_tile = col / kWarpK;
    const int col_in_tile = col - k_tile * kWarpK;
    const int frag = col_in_tile / 8;
    const int pair = (col_in_tile & 0x7) >> 1;
    const int nibble = col_in_tile & 0x1;

    const int q_lane = row_low * 4 + (frag & 0x3);
    const int q_byte_in_uint4 = row_high * 4 + ((frag >= 4) ? 8 : 0) + pair;
    int64_t q_uint4_index = m_block;
    q_uint4_index = q_uint4_index * k_tiles + k_tile;
    q_uint4_index = q_uint4_index * kNumWarps + warp;
    q_uint4_index = q_uint4_index * kWarpMTiles + tile_m;
    q_uint4_index = q_uint4_index * kWarpSize + q_lane;

    const uint8_t packed = qact[q_uint4_index * 16 + q_byte_in_uint4];
    const uint8_t code = static_cast<uint8_t>((packed >> (nibble * 4)) & 0xF);

    const int scale_lane = row_low * 4 + tile_m * 2 + row_high;
    const int scale_group = col_in_tile / kFP4GroupSize;
    int64_t scale_index = m_block;
    scale_index = scale_index * k_tiles + k_tile;
    scale_index = scale_index * kNumWarps + warp;
    scale_index = scale_index * kWarpSize + scale_lane;
    scale_index = scale_index * 4 + scale_group;

    return decode_fp4(code) * decode_e4m3fn(ascales[scale_index]);
}

template <typename scalar_t, int kVec, int rVec, int kThreads>
__global__ void fp4_activation_cache_lora_down_grad_tiled_kernel(
    const uint8_t* __restrict__ qact,
    const uint8_t* __restrict__ ascales,
    const scalar_t* __restrict__ dy_up,
    scalar_t* __restrict__ output,
    int rows,
    int cols,
    int rank,
    int k_tiles) {
    static_assert(kThreads == 128 || kThreads == 256);
    const int col_base = blockIdx.x * kVec;
    const int rank_base = blockIdx.y * rVec;
    const int tid = threadIdx.x;

    float accum[rVec][kVec];
#pragma unroll
    for (int r = 0; r < rVec; ++r) {
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
            accum[r][k] = 0.0f;
        }
    }

    for (int row = tid; row < rows; row += blockDim.x) {
        float x_values[kVec];
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
            const int col = col_base + k;
            x_values[k] = col < cols ? load_dequantized_fp4_activation(qact, ascales, row, col, k_tiles) : 0.0f;
        }

        float dy_values[rVec];
#pragma unroll
        for (int r = 0; r < rVec; ++r) {
            const int rank_idx = rank_base + r;
            dy_values[r] =
                rank_idx < rank ? static_cast<float>(dy_up[static_cast<int64_t>(row) * rank + rank_idx]) : 0.0f;
        }

#pragma unroll
        for (int r = 0; r < rVec; ++r) {
#pragma unroll
            for (int k = 0; k < kVec; ++k) {
                accum[r][k] += dy_values[r] * x_values[k];
            }
        }
    }

    __shared__ float smem[kThreads * kVec * rVec];
#pragma unroll
    for (int r = 0; r < rVec; ++r) {
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
            smem[(r * kVec + k) * blockDim.x + tid] = accum[r][k];
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int r = 0; r < rVec; ++r) {
#pragma unroll
                for (int k = 0; k < kVec; ++k) {
                    const int offset = (r * kVec + k) * blockDim.x + tid;
                    smem[offset] += smem[offset + stride];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
#pragma unroll
        for (int r = 0; r < rVec; ++r) {
            const int rank_idx = rank_base + r;
            if (rank_idx < rank) {
#pragma unroll
                for (int k = 0; k < kVec; ++k) {
                    const int col = col_base + k;
                    if (col < cols) {
                        output[static_cast<int64_t>(rank_idx) * cols + col] =
                            static_cast<scalar_t>(smem[(r * kVec + k) * blockDim.x]);
                    }
                }
            }
        }
    }
}

} // namespace

namespace nunchaku_core::ops {

void dequantize_fp4_activation_cuda(torch::Tensor qact, torch::Tensor ascales, torch::Tensor output) {
    const c10::cuda::CUDAGuard device_guard(qact.device());

    const int rows = static_cast<int>(qact.size(0));
    const int cols = static_cast<int>(qact.size(1) * 2);
    TORCH_CHECK(rows % kBlockM == 0, "qact rows must be divisible by 256");
    TORCH_CHECK(cols % kWarpK == 0, "qact cols must be divisible by 64");
    TORCH_CHECK(output.size(0) == rows && output.size(1) == cols, "output shape mismatch");
    TORCH_CHECK(ascales.numel() == rows * cols / kFP4GroupSize, "ascales numel mismatch");

    const int k_tiles = cols / kWarpK;
    const int64_t total = static_cast<int64_t>(rows) * cols;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        output.scalar_type(),
        "dequantize_fp4_activation_cuda",
        [&] {
            dequantize_fp4_activation_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                qact.data_ptr<uint8_t>(),
                reinterpret_cast<const uint8_t*>(ascales.data_ptr()),
                output.data_ptr<scalar_t>(),
                rows,
                cols,
                k_tiles,
                total);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp4_activation_cache_lora_down_grad_cuda(
    torch::Tensor qact,
    torch::Tensor ascales,
    torch::Tensor dy_up,
    torch::Tensor output) {
    const c10::cuda::CUDAGuard device_guard(qact.device());

    const int padded_rows = static_cast<int>(qact.size(0));
    const int padded_cols = static_cast<int>(qact.size(1) * 2);
    const int rows = static_cast<int>(dy_up.size(0));
    const int rank = static_cast<int>(dy_up.size(1));
    const int cols = static_cast<int>(output.size(1));
    TORCH_CHECK(padded_rows % kBlockM == 0, "qact rows must be divisible by 256");
    TORCH_CHECK(padded_cols % kWarpK == 0, "qact cols must be divisible by 64");
    TORCH_CHECK(rows <= padded_rows, "dy_up rows must be <= qact padded rows");
    TORCH_CHECK(cols <= padded_cols, "output cols must be <= qact padded cols");
    TORCH_CHECK(output.size(0) == rank, "output rows must match dy_up rank");
    TORCH_CHECK(ascales.numel() == padded_rows * padded_cols / kFP4GroupSize, "ascales numel mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();

    if (rank <= 32) {
        // Rank-32 LoRA is the common finetuning case. kVec=3 reduces CTA count
        // without changing rank tiling and stays within the 48KB static smem cap.
        constexpr int kVec = 3;
        constexpr int rVec = 16;
        constexpr int kThreads = 256;
        const dim3 blocks((cols + kVec - 1) / kVec, (rank + rVec - 1) / rVec);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf,
            at::kBFloat16,
            output.scalar_type(),
            "fp4_activation_cache_lora_down_grad_cuda_rank32",
            [&] {
                fp4_activation_cache_lora_down_grad_tiled_kernel<scalar_t, kVec, rVec, kThreads>
                    <<<blocks, kThreads, 0, stream>>>(
                        qact.data_ptr<uint8_t>(),
                        reinterpret_cast<const uint8_t*>(ascales.data_ptr()),
                        dy_up.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        rows,
                        cols,
                        rank,
                        static_cast<int>(padded_cols / kWarpK));
            });
    } else if (rank <= 256) {
        // Higher-rank sensitive projections still benefit from a wider rank tile.
        // Keep the thread count low so the reduction fits under 48KB static smem.
        constexpr int kVec = 3;
        constexpr int rVec = 32;
        constexpr int kThreads = 128;
        const dim3 blocks((cols + kVec - 1) / kVec, (rank + rVec - 1) / rVec);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf,
            at::kBFloat16,
            output.scalar_type(),
            "fp4_activation_cache_lora_down_grad_cuda_rank256",
            [&] {
                fp4_activation_cache_lora_down_grad_tiled_kernel<scalar_t, kVec, rVec, kThreads>
                    <<<blocks, kThreads, 0, stream>>>(
                        qact.data_ptr<uint8_t>(),
                        reinterpret_cast<const uint8_t*>(ascales.data_ptr()),
                        dy_up.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        rows,
                        cols,
                        rank,
                        static_cast<int>(padded_cols / kWarpK));
            });
    } else {
        constexpr int kVec = 2;
        constexpr int rVec = 16;
        constexpr int kThreads = 256;
        const dim3 blocks((cols + kVec - 1) / kVec, (rank + rVec - 1) / rVec);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf,
            at::kBFloat16,
            output.scalar_type(),
            "fp4_activation_cache_lora_down_grad_cuda",
            [&] {
                fp4_activation_cache_lora_down_grad_tiled_kernel<scalar_t, kVec, rVec, kThreads>
                    <<<blocks, kThreads, 0, stream>>>(
                        qact.data_ptr<uint8_t>(),
                        reinterpret_cast<const uint8_t*>(ascales.data_ptr()),
                        dy_up.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        rows,
                        cols,
                        rank,
                        static_cast<int>(padded_cols / kWarpK));
            });
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace nunchaku_core::ops
