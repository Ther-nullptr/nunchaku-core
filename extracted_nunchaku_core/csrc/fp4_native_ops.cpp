#include <torch/extension.h>
#include <optional>
#include <vector>

#include "interop/torch.h"
#include "kernels/zgemm/zgemm.h"

namespace nunchaku_core::ops {

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

} // namespace nunchaku_core::ops

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_w4a4", nunchaku_core::ops::gemm_w4a4);
    m.def("quantize_w4a4_act_fuse_lora", nunchaku_core::ops::quantize_w4a4_act_fuse_lora);
}
