#include <torch/extension.h>

std::vector<torch::Tensor> quantize_int4_packed_cuda(torch::Tensor input);
torch::Tensor unpack_int4_packed_cuda(torch::Tensor packed);

std::vector<torch::Tensor> quantize_int4_packed(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
    TORCH_CHECK(
        input.scalar_type() == torch::kHalf || input.scalar_type() == torch::kBFloat16 || input.scalar_type() == torch::kFloat,
        "input dtype must be float16, bfloat16 or float32");
    return quantize_int4_packed_cuda(input);
}

torch::Tensor unpack_int4_packed(torch::Tensor packed) {
    TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D [M, K/2]");
    TORCH_CHECK(packed.scalar_type() == torch::kByte, "packed dtype must be uint8");
    return unpack_int4_packed_cuda(packed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_int4_packed", &quantize_int4_packed, "Quantize to packed INT4 (CUDA)");
    m.def("unpack_int4_packed", &unpack_int4_packed, "Unpack INT4 to int8 matrix (CUDA)");
}
