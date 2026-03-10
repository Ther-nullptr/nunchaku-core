#include <torch/extension.h>
#include <optional>
#include <vector>

#include "interop/torch.h"
#include "kernels/zgemm/zgemm.h"

namespace nunchaku_core::ops {

torch::Tensor fp4_repack_backward_cuda(
    torch::Tensor qweight,
    torch::Tensor fwd_scales_logical,
    torch::Tensor bwd_scales_logical);
void decode_lora_act_cuda(torch::Tensor packed_lora_act, torch::Tensor dense_lora_act);

void gemm_w4a4(
    std::optional<torch::Tensor> act,
    std::optional<torch::Tensor> wgt,
    std::optional<torch::Tensor> out,
    std::optional<torch::Tensor> qout,
    std::optional<torch::Tensor> ascales,
    std::optional<torch::Tensor> wscales,
    std::optional<torch::Tensor> oscales,
    std::optional<torch::Tensor> poolout,
    std::optional<torch::Tensor> lora_act_in,
    std::optional<torch::Tensor> lora_up,
    std::optional<torch::Tensor> lora_down,
    std::optional<torch::Tensor> lora_act_out,
    std::optional<torch::Tensor> norm_q,
    std::optional<torch::Tensor> norm_k,
    std::optional<torch::Tensor> rotary_emb,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> smooth_factor,
    std::optional<torch::Tensor> out_vk,
    std::optional<torch::Tensor> out_linearattn,
    bool act_unsigned,
    std::vector<float> lora_scales,
    bool fuse_silu,
    bool fp4,
    float alpha,
    std::optional<torch::Tensor> wcscales,
    std::optional<torch::Tensor> out_q,
    std::optional<torch::Tensor> out_k,
    std::optional<torch::Tensor> out_v,
    int attn_tokens) {
    TorchOpContext ctx;

    auto getTensor = [](std::optional<torch::Tensor>& t) {
        return t.has_value() ? from_torch(t.value()) : Tensor{};
    };

    nunchaku::kernels::gemm_w4a4(
        getTensor(act),
        getTensor(wgt),
        getTensor(out),
        getTensor(qout),
        getTensor(ascales),
        getTensor(wscales),
        getTensor(oscales),
        getTensor(poolout),
        getTensor(lora_act_in),
        getTensor(lora_up),
        getTensor(lora_down),
        getTensor(lora_act_out),
        getTensor(norm_q),
        getTensor(norm_k),
        getTensor(rotary_emb),
        getTensor(bias),
        getTensor(smooth_factor),
        getTensor(out_vk),
        getTensor(out_linearattn),
        act_unsigned,
        lora_scales,
        fuse_silu,
        fp4,
        alpha,
        getTensor(wcscales),
        getTensor(out_q),
        getTensor(out_k),
        getTensor(out_v),
        attn_tokens);
}

void quantize_w4a4_act_fuse_lora(std::optional<torch::Tensor> input,
                                 std::optional<torch::Tensor> output,
                                 std::optional<torch::Tensor> oscales,
                                 std::optional<torch::Tensor> lora_down,
                                 std::optional<torch::Tensor> lora_act_out,
                                 std::optional<torch::Tensor> smooth,
                                 bool fuse_glu,
                                 bool fp4) {
    TorchOpContext ctx;

    auto getTensor = [](std::optional<torch::Tensor>& t) {
        return t.has_value() ? from_torch(t.value()) : Tensor{};
    };

    nunchaku::kernels::quantize_w4a4_act_fuse_lora(
        getTensor(input),
        getTensor(output),
        getTensor(oscales),
        getTensor(lora_down),
        getTensor(lora_act_out),
        getTensor(smooth),
        fuse_glu,
        fp4);
}

torch::Tensor fp4_repack_backward(
    torch::Tensor qweight,
    torch::Tensor fwd_scales_logical,
    torch::Tensor bwd_scales_logical) {
    TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
    TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D [N, K/2]");
    TORCH_CHECK(qweight.scalar_type() == torch::kUInt8, "qweight dtype must be uint8");

    TORCH_CHECK(fwd_scales_logical.is_cuda(), "fwd_scales_logical must be a CUDA tensor");
    TORCH_CHECK(fwd_scales_logical.is_contiguous(), "fwd_scales_logical must be contiguous");
    TORCH_CHECK(fwd_scales_logical.dim() == 2, "fwd_scales_logical must be 2D [N, K/16]");
    TORCH_CHECK(fwd_scales_logical.scalar_type() == torch::kHalf, "fwd_scales_logical dtype must be float16");

    TORCH_CHECK(bwd_scales_logical.is_cuda(), "bwd_scales_logical must be a CUDA tensor");
    TORCH_CHECK(bwd_scales_logical.is_contiguous(), "bwd_scales_logical must be contiguous");
    TORCH_CHECK(bwd_scales_logical.dim() == 2, "bwd_scales_logical must be 2D [K, N/16]");
    TORCH_CHECK(bwd_scales_logical.scalar_type() == torch::kHalf, "bwd_scales_logical dtype must be float16");

    return fp4_repack_backward_cuda(qweight, fwd_scales_logical, bwd_scales_logical);
}

void decode_lora_act(torch::Tensor packed_lora_act, torch::Tensor dense_lora_act) {
    TORCH_CHECK(packed_lora_act.is_cuda(), "packed_lora_act must be a CUDA tensor");
    TORCH_CHECK(packed_lora_act.is_contiguous(), "packed_lora_act must be contiguous");
    TORCH_CHECK(packed_lora_act.dim() == 2, "packed_lora_act must be 2D [M_pad, rank]");
    TORCH_CHECK(packed_lora_act.scalar_type() == torch::kFloat, "packed_lora_act dtype must be float32");

    TORCH_CHECK(dense_lora_act.is_cuda(), "dense_lora_act must be a CUDA tensor");
    TORCH_CHECK(dense_lora_act.is_contiguous(), "dense_lora_act must be contiguous");
    TORCH_CHECK(dense_lora_act.dim() == 2, "dense_lora_act must be 2D [M_pad, rank]");
    TORCH_CHECK(
        dense_lora_act.scalar_type() == torch::kHalf || dense_lora_act.scalar_type() == torch::kBFloat16,
        "dense_lora_act dtype must be float16 or bfloat16");
    TORCH_CHECK(
        packed_lora_act.sizes() == dense_lora_act.sizes(),
        "packed_lora_act and dense_lora_act must have the same shape");

    decode_lora_act_cuda(packed_lora_act, dense_lora_act);
}

} // namespace nunchaku_core::ops

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_w4a4", nunchaku_core::ops::gemm_w4a4);
    m.def("quantize_w4a4_act_fuse_lora", nunchaku_core::ops::quantize_w4a4_act_fuse_lora);
    m.def("fp4_repack_backward", nunchaku_core::ops::fp4_repack_backward);
    m.def("decode_lora_act", nunchaku_core::ops::decode_lora_act);
}
