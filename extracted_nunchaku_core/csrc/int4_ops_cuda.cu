#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr float kEps = 1e-8f;

template<typename scalar_t>
__global__ void quantize_int4_packed_kernel(
    const scalar_t* __restrict__ input,
    uint8_t* __restrict__ packed,
    float* __restrict__ scales,
    int K,
    int packed_K) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float sdata[];

    float local_max = 0.0f;
    const scalar_t* row_ptr = input + static_cast<int64_t>(row) * K;

    for (int k = tid; k < K; k += blockDim.x) {
        const float v = static_cast<float>(row_ptr[k]);
        local_max = fmaxf(local_max, fabsf(v));
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float scale = fmaxf(sdata[0] / 7.0f, kEps);
    if (tid == 0) {
        scales[row] = scale;
    }
    __syncthreads();

    const float inv_scale = 1.0f / scale;
    uint8_t* packed_row = packed + static_cast<int64_t>(row) * packed_K;

    for (int j = tid; j < packed_K; j += blockDim.x) {
        const int k0 = j * 2;
        const int k1 = k0 + 1;

        int q0 = __float2int_rn(static_cast<float>(row_ptr[k0]) * inv_scale);
        int q1 = __float2int_rn(static_cast<float>(row_ptr[k1]) * inv_scale);

        q0 = max(-8, min(7, q0));
        q1 = max(-8, min(7, q1));

        const uint8_t low = static_cast<uint8_t>(q0 & 0x0F);
        const uint8_t high = static_cast<uint8_t>(q1 & 0x0F);
        packed_row[j] = static_cast<uint8_t>(low | (high << 4));
    }
}

__global__ void unpack_int4_packed_kernel(
    const uint8_t* __restrict__ packed,
    int8_t* __restrict__ output,
    int K,
    int packed_K,
    int total_pairs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) {
        return;
    }

    const int row = idx / packed_K;
    const int pair_col = idx - row * packed_K;

    const uint8_t byte = packed[idx];
    int q0 = static_cast<int>(byte & 0x0F);
    int q1 = static_cast<int>((byte >> 4) & 0x0F);

    if (q0 >= 8) {
        q0 -= 16;
    }
    if (q1 >= 8) {
        q1 -= 16;
    }

    const int base = row * K + pair_col * 2;
    output[base] = static_cast<int8_t>(q0);
    output[base + 1] = static_cast<int8_t>(q1);
}

} // namespace

std::vector<torch::Tensor> quantize_int4_packed_cuda(torch::Tensor input) {
    const c10::cuda::CUDAGuard device_guard(input.device());

    const auto M = input.size(0);
    const auto K = input.size(1);

    TORCH_CHECK(K % 2 == 0, "K must be even for INT4 packing");

    auto packed = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(input.device()));
    auto scales = torch::empty({M}, torch::dtype(torch::kFloat32).device(input.device()));

    constexpr int threads = 256;
    const int shared_mem = threads * static_cast<int>(sizeof(float));

    auto stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        input.scalar_type(),
        "quantize_int4_packed_cuda",
        [&] {
            quantize_int4_packed_kernel<scalar_t><<<M, threads, shared_mem, stream>>>(
                input.data_ptr<scalar_t>(),
                packed.data_ptr<uint8_t>(),
                scales.data_ptr<float>(),
                static_cast<int>(K),
                static_cast<int>(K / 2));
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {packed, scales};
}

torch::Tensor unpack_int4_packed_cuda(torch::Tensor packed) {
    const c10::cuda::CUDAGuard device_guard(packed.device());

    const auto M = packed.size(0);
    const auto packed_K = packed.size(1);
    const auto K = packed_K * 2;

    auto output = torch::empty({M, K}, torch::dtype(torch::kInt8).device(packed.device()));

    const int total_pairs = static_cast<int>(M * packed_K);
    constexpr int threads = 256;
    const int blocks = (total_pairs + threads - 1) / threads;

    auto stream = at::cuda::getDefaultCUDAStream();

    unpack_int4_packed_kernel<<<blocks, threads, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        static_cast<int>(K),
        static_cast<int>(packed_K),
        total_pairs);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
