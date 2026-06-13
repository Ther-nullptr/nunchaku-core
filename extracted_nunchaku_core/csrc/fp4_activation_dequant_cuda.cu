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

} // namespace nunchaku_core::ops
