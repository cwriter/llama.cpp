#ifndef GGML_SYCL_HC_FFN_TAIL_HPP
#define GGML_SYCL_HC_FFN_TAIL_HPP

#include "common.hpp"
#include "dsv4-hc.hpp"

// The gated hc_post + norm fusion that starts at SCALE node i: SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST
// -> RMS_NORM -> MUL. Shared by the compute loop, graph_optimize and the FFN tail below.
bool ggml_sycl_hc_post_norm_at(const ggml_cgraph * cgraph, int i);

// The FFN output that such an hc_post combines (qwen4exp ffn_out): the MoE weighted sum of the
// expert outputs, plus the shared expert times its sigmoid gate. The hc_post + norm kernel computes
// it per element instead of reading it, so the reduction and the add never launch.
struct ggml_sycl_hc_ffn_tail {
    ggml_sycl_dsv4_hc_post_x x;
    const ggml_tensor *      nodes[2 * GGML_SYCL_MOE_REDUCE_MAX_EXPERTS + 2];
    int                      n_nodes = 0;
};

// the tail of the hc_post + norm fusion at SCALE node i, if it has one
bool ggml_sycl_match_hc_ffn_tail(const ggml_cgraph * cgraph, int i, ggml_sycl_hc_ffn_tail & tail);

// 1 when node_idx is a node of such a tail
int ggml_sycl_hc_ffn_tail_absorbs(const ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_HC_FFN_TAIL_HPP
