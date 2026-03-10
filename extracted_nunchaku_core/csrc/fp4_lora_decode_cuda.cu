#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kBlockM = 256;
constexpr int kWarpM = 32;
constexpr int kWarpR = 16;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = 8;
constexpr int kLoraMTiles = 2;

template<typename out_t>
__device__ __forceinline__ out_t cast_out(float value);

template<>
__device__ __forceinline__ half cast_out<half>(float value) {
    return __float2half_rn(value);
}

template<>
__device__ __forceinline__ __nv_bfloat16 cast_out<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template<typename out_t>
__global__ void decode_lora_act_kernel(
    const float* __restrict__ packed,
    out_t* __restrict__ dense,
    int m_pad,
    int rank) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m_pad * rank;
    if (idx >= total) {
        return;
    }

    const int row = idx / rank;
    const int col = idx - row * rank;

    const int bm = row / kBlockM;
    const int row_in_block = row % kBlockM;
    const int warp_id = row_in_block / kWarpM;
    const int row_in_warp = row_in_block % kWarpM;
    const int m_tile = row_in_warp / 16;
    const int row_in_tile = row_in_warp % 16;

    const int r_tile = col / kWarpR;
    const int col_in_tile = col % kWarpR;

    const int lane_row = row_in_tile % 8;
    const int lane_col_group = (col_in_tile % 8) / 2;
    const int lane_id = lane_row * 4 + lane_col_group;

    const bool upper_rows = row_in_tile >= 8;
    const bool upper_cols = col_in_tile >= 8;
    const int pair_col = col_in_tile & 1;

    int data_idx = pair_col;
    if (upper_rows) {
        data_idx += 2;
    }
    if (upper_cols) {
        data_idx += 4;
    }

    const int rank_tiles = rank / kWarpR;
    const int tile_stride = kLoraMTiles * 8 * kWarpSize;
    const int64_t src_index =
        (((static_cast<int64_t>(bm) * rank_tiles + r_tile) * kNumWarps + warp_id) * tile_stride) +
        m_tile * 8 * kWarpSize + data_idx * kWarpSize + lane_id;

    dense[idx] = cast_out<out_t>(packed[src_index]);
}

} // namespace

namespace nunchaku_core::ops {

void decode_lora_act_cuda(torch::Tensor packed_lora_act, torch::Tensor dense_lora_act) {
    const c10::cuda::CUDAGuard device_guard(packed_lora_act.device());

    const int m_pad = static_cast<int>(packed_lora_act.size(0));
    const int rank = static_cast<int>(packed_lora_act.size(1));

    TORCH_CHECK(m_pad % kBlockM == 0, "packed_lora_act rows must be divisible by 256");
    TORCH_CHECK(rank % kWarpR == 0, "packed_lora_act cols must be divisible by 16");

    constexpr int threads = 256;
    const int total = m_pad * rank;
    const int blocks = (total + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();

    switch (dense_lora_act.scalar_type()) {
        case torch::kHalf:
            decode_lora_act_kernel<half><<<blocks, threads, 0, stream>>>(
                packed_lora_act.data_ptr<float>(),
                reinterpret_cast<half*>(dense_lora_act.data_ptr<at::Half>()),
                m_pad,
                rank);
            break;
        case torch::kBFloat16:
            decode_lora_act_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                packed_lora_act.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(dense_lora_act.data_ptr<at::BFloat16>()),
                m_pad,
                rank);
            break;
        default:
            TORCH_CHECK(false, "dense_lora_act dtype must be float16 or bfloat16");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace nunchaku_core::ops
