#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWarpN = 128;
constexpr int kWarpK = 64;
constexpr int kGroupSize = 16;
constexpr int kNumNPacks = 8;
constexpr int kNumNLanes = 8;
constexpr int kNumKLanes = 4;
constexpr int kNPackSize = 2;
constexpr int kKPackSize = 2;

__device__ __forceinline__ float decode_fp4(uint8_t code) {
    constexpr float mag_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = mag_lut[code & 0x7];
    return (code & 0x8) ? -value : value;
}

__device__ __forceinline__ half decode_fp4_half(uint8_t code) {
    return __float2half_rn(decode_fp4(code));
}

__device__ __forceinline__ uint8_t quantize_fp4_from_half(half scaled) {
    const float magnitude = fabsf(__half2float(scaled));
    uint8_t code = 0;
    code += static_cast<uint8_t>(magnitude >= 0.25f);
    code += static_cast<uint8_t>(magnitude >= 0.75f);
    code += static_cast<uint8_t>(magnitude >= 1.25f);
    code += static_cast<uint8_t>(magnitude >= 1.75f);
    code += static_cast<uint8_t>(magnitude >= 2.50f);
    code += static_cast<uint8_t>(magnitude >= 3.50f);
    code += static_cast<uint8_t>(magnitude >= 5.00f);
    return static_cast<uint8_t>(code | ((__hlt(scaled, __float2half_rn(0.0f))) ? 0x8 : 0x0));
}

__device__ __forceinline__ uint8_t load_fp4_code(
    const uint8_t* __restrict__ qweight,
    int src_k_tiles,
    int row,
    int col) {
    const int n_block = row >> 7;
    const int row_local = row & (kWarpN - 1);
    const int n_pack = row_local >> 4;
    const int n_pack_size = (row_local >> 3) & 0x1;
    const int n_lane = row_local & 0x7;

    const int k_tile = col >> 6;
    const int col_local = col & (kWarpK - 1);
    const int k_pack_size = col_local >> 5;
    const int k_lane = (col_local >> 3) & 0x3;
    const int reg_k = col_local & 0x7;

    int64_t word_index = n_block;
    word_index = word_index * src_k_tiles + k_tile;
    word_index = word_index * kNumNPacks + n_pack;
    word_index = word_index * kNumNLanes + n_lane;
    word_index = word_index * kNumKLanes + k_lane;
    word_index = word_index * kNPackSize + n_pack_size;
    word_index = word_index * kKPackSize + k_pack_size;

    const uint8_t packed_byte = qweight[word_index * 4 + (reg_k >> 1)];
    return static_cast<uint8_t>((packed_byte >> ((reg_k & 1) * 4)) & 0xF);
}

__global__ void fp4_repack_backward_kernel(
    const uint8_t* __restrict__ qweight_fwd,
    const half* __restrict__ fwd_scales,
    const half* __restrict__ bwd_scales,
    uint32_t* __restrict__ qweight_bwd_words,
    int src_k_tiles,
    int dst_k_tiles,
    int fwd_groups,
    int bwd_groups,
    int64_t total_words) {
    const int64_t word_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word_index >= total_words) {
        return;
    }

    int64_t tmp = word_index;
    const int k_pack_size = static_cast<int>(tmp & 0x1);
    tmp >>= 1;
    const int n_pack_size = static_cast<int>(tmp & 0x1);
    tmp >>= 1;
    const int k_lane = static_cast<int>(tmp & 0x3);
    tmp >>= 2;
    const int n_lane = static_cast<int>(tmp & 0x7);
    tmp >>= 3;
    const int n_pack = static_cast<int>(tmp & 0x7);
    tmp >>= 3;
    const int k_tile = static_cast<int>(tmp % dst_k_tiles);
    const int n_block = static_cast<int>(tmp / dst_k_tiles);

    const int dst_row = (n_block << 7) + (n_pack << 4) + (n_pack_size << 3) + n_lane;
    const int dst_col_base = (k_tile << 6) + (k_pack_size << 5) + (k_lane << 3);

    uint32_t out_word = 0;

    #pragma unroll
    for (int reg_k = 0; reg_k < 8; ++reg_k) {
        const int dst_col = dst_col_base + reg_k;
        const int src_row = dst_col;
        const int src_col = dst_row;

        const uint8_t src_code = load_fp4_code(qweight_fwd, src_k_tiles, src_row, src_col);
        const half src_scale = fwd_scales[static_cast<int64_t>(src_row) * fwd_groups + (src_col >> 4)];
        const half dst_scale = bwd_scales[static_cast<int64_t>(dst_row) * bwd_groups + (dst_col >> 4)];

        uint8_t dst_code = 0;
        if (__hgt(dst_scale, __float2half_rn(0.0f)) && __hgt(src_scale, __float2half_rn(0.0f))) {
            const half value = __hmul(decode_fp4_half(src_code), src_scale);
            const half scaled = __hdiv(value, dst_scale);
            dst_code = quantize_fp4_from_half(scaled);
        }

        out_word |= static_cast<uint32_t>(dst_code & 0xF) << (reg_k * 4);
    }

    qweight_bwd_words[word_index] = out_word;
}

} // namespace

namespace nunchaku_core::ops {

torch::Tensor fp4_repack_backward_cuda(
    torch::Tensor qweight,
    torch::Tensor fwd_scales_logical,
    torch::Tensor bwd_scales_logical) {
    const c10::cuda::CUDAGuard device_guard(qweight.device());

    const int64_t src_rows = qweight.size(0);
    const int64_t src_cols = qweight.size(1) * 2;
    TORCH_CHECK(src_rows % kWarpN == 0, "qweight rows must be divisible by 128");
    TORCH_CHECK(src_cols % (kWarpK * 2) == 0, "qweight cols must be divisible by 128");

    TORCH_CHECK(
        fwd_scales_logical.size(0) == src_rows && fwd_scales_logical.size(1) == src_cols / kGroupSize,
        "fwd_scales_logical shape mismatch");
    TORCH_CHECK(
        bwd_scales_logical.size(0) == src_cols && bwd_scales_logical.size(1) == src_rows / kGroupSize,
        "bwd_scales_logical shape mismatch");

    auto output = torch::empty({src_cols, src_rows / 2}, torch::dtype(torch::kUInt8).device(qweight.device()));

    constexpr int threads = 256;
    const int64_t total_words = output.numel() / 4;
    const int blocks = static_cast<int>((total_words + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    fp4_repack_backward_kernel<<<blocks, threads, 0, stream>>>(
        qweight.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(fwd_scales_logical.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bwd_scales_logical.data_ptr<at::Half>()),
        reinterpret_cast<uint32_t*>(output.data_ptr<uint8_t>()),
        static_cast<int>(src_cols / kWarpK),
        static_cast<int>(src_rows / kWarpK),
        static_cast<int>(src_cols / kGroupSize),
        static_cast<int>(src_rows / kGroupSize),
        total_words);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

} // namespace nunchaku_core::ops
