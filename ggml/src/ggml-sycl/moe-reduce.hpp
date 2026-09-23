#ifndef GGML_SYCL_MOE_REDUCE_HPP
#define GGML_SYCL_MOE_REDUCE_HPP

#include "common.hpp"

// The tail of every MoE layer, as llama-graph.cpp builds it:
//
//     experts = ggml_mul(experts, weights);                       // 1 node
//     cur[i]  = ggml_view_2d(experts, n_embd, n_tokens, ...);     // n_expert_used nodes
//     out     = ((cur[0] + cur[1]) + cur[2]) + ...;               // n_expert_used - 1 nodes
//
// The views are metadata and the adds are a left-leaning chain over one buffer, so the whole
// thing is a weighted sum along the expert axis and fits in one kernel. Matching it needs
// ggml_can_fuse_subgraph() rather than ggml_can_fuse(): the views are listed as part of the
// span instead of being stepped over, which is also what proves nothing outside the span
// reads them.
struct ggml_sycl_moe_reduce_match {
    const ggml_tensor * experts    = nullptr;  // the mul's wide operand, read instead of its result
    const ggml_tensor * weights    = nullptr;  // the mul's per-(expert, token) operand
    ggml_tensor *       dst        = nullptr;  // the last add, the only output of the span
    int                 node_count = 0;
};

bool ggml_sycl_match_moe_weighted_reduction(const ggml_cgraph * cgraph, int node_idx,
                                            ggml_sycl_moe_reduce_match & match);

void ggml_sycl_op_moe_weighted_reduction(ggml_backend_sycl_context & ctx,
                                         const ggml_sycl_moe_reduce_match & match);

#endif  // GGML_SYCL_MOE_REDUCE_HPP
