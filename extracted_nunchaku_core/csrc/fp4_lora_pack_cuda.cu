#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kFragN = 16;
constexpr int kFragK = 16;

template<typename scalar_t>
__global__ void pack_lowrank_weight_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int rows,
    int cols,
    int r_frags,
    bool down,
    int64_t total) {
    const int64_t out_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_index >= total) {
        return;
    }

    int64_t tmp = out_index;
    const int lane_k = static_cast<int>(tmp & 0x1);
    tmp >>= 1;
    const int k_pack_size = static_cast<int>(tmp & 0x1);
    tmp >>= 1;
    const int n_pack_size = static_cast<int>(tmp & 0x1);
    tmp >>= 1;
    const int k_lane = static_cast<int>(tmp & 0x3);
    tmp >>= 2;
    const int n_lane = static_cast<int>(tmp & 0x7);
    tmp >>= 3;
    const int r_frag = static_cast<int>(tmp % r_frags);
    const int c_frag = static_cast<int>(tmp / r_frags);

    const int row_local = n_pack_size * 8 + n_lane;
    const int col_local = k_pack_size * 8 + k_lane * 2 + lane_k;

    int src_row;
    int src_col;
    if (down) {
        src_row = r_frag * kFragN + row_local;
        src_col = c_frag * kFragK + col_local;
    } else {
        src_row = c_frag * kFragN + row_local;
        src_col = r_frag * kFragK + col_local;
    }

    scalar_t value = scalar_t(0);
    if (src_row < rows && src_col < cols) {
        value = input[static_cast<int64_t>(src_row) * cols + src_col];
    }
    output[out_index] = value;
}

} // namespace

namespace nunchaku_core::ops {

torch::Tensor pack_lowrank_weight_cuda(torch::Tensor weight, bool down) {
    const c10::cuda::CUDAGuard device_guard(weight.device());

    const int rows = static_cast<int>(weight.size(0));
    const int cols = static_cast<int>(weight.size(1));
    const int rows_pad = ((rows + kFragN - 1) / kFragN) * kFragN;
    const int cols_pad = ((cols + kFragK - 1) / kFragK) * kFragK;

    const int out_rows = down ? cols_pad : rows_pad;
    const int out_cols = down ? rows_pad : cols_pad;
    const int r_frags = out_cols / kFragK;

    auto output = torch::empty({out_rows, out_cols}, weight.options());
    const int64_t total = output.numel();

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        weight.scalar_type(),
        "pack_lowrank_weight_cuda",
        [&] {
            pack_lowrank_weight_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                rows,
                cols,
                r_frags,
                down,
                total);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

} // namespace nunchaku_core::ops
