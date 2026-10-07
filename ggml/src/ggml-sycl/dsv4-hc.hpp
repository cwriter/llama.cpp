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

// The hc_pre gate is up(silu(scale(lo))) with a reordered Q8_0 up weight, built by three launches
// per hyper-connection block. With GGML_SYCL_FUSE_HC_PRE one kernel does the scale, the unary, the
// mat-vec and the gated hc_pre: a sub-group owns the hc weight rows of one embedding element.
struct ggml_sycl_dsv4_hc_pre_up_match {
    ggml_tensor * scale = nullptr;  // SCALE(lo), the first node of the span
    ggml_tensor * unary = nullptr;
    ggml_tensor * up    = nullptr;  // MUL_MAT(w_up, unary)
    ggml_tensor * dst   = nullptr;  // DSV4_HC_PRE, the only node of the span that is written
    int           last  = -1;       // graph index of dst
};

// structural match at the SCALE node `node_idx`; the weight layout is checked by the caller.
// any_tokens drops the token limit: graph_optimize must keep the same alloc deps for every
// ubatch size, or the graph topology changes between ubatches and forces a re-plan.
bool ggml_sycl_match_dsv4_hc_pre_up(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_dsv4_hc_pre_up_match & match,
                                    bool any_tokens = false);

// the up weight must already be in the reordered Q8_0 layout
void ggml_sycl_op_dsv4_hc_pre_up_fused(ggml_backend_sycl_context & ctx, const ggml_sycl_dsv4_hc_pre_up_match & match);

#endif // GGML_SYCL_DSV4_HC_HPP
