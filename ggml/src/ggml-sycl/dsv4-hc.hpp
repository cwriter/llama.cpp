#ifndef GGML_SYCL_DSV4_HC_HPP
#define GGML_SYCL_DSV4_HC_HPP

#include "common.hpp"

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// The per-stream gate that feeds dsv4_hc_post is built by scale -> sigmoid -> scale over an
// (hc, n_tokens) tensor - a handful of floats per token, for three kernel launches. When the
// graph has that shape, hc_post reads the chain's input instead and applies it inline.
struct ggml_sycl_dsv4_hc_post_gate {
    const ggml_tensor * src = nullptr;  // stands in for dst->src[2]
    float               s0  = 1.0f;     // first scale:  s0*v + b0
    float               b0  = 0.0f;
    float               s1  = 1.0f;     // second scale: s1*sigmoid(...) + b1
    float               b1  = 0.0f;
};

void ggml_sycl_op_dsv4_hc_post_fused_gate(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                          const ggml_sycl_dsv4_hc_post_gate & gate);

#endif // GGML_SYCL_DSV4_HC_HPP
