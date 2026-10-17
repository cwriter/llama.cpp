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

// 1 when node_idx is a node of a matched reduction span other than its output: the mul, the
// views and the inner adds are never written, so fusion_absorbs reports them and ggml-alloc gives
// them no buffer (the mul alone is the full [n_embd, n_expert_used, n_tokens] expert output).
// graph_optimize already keeps the experts and weights live until the span's output.
int ggml_sycl_moe_reduce_absorbs(const ggml_cgraph * cgraph, int node_idx);

// GGML_SYCL_MOE_DOWN_REDUCE: the expert down projection (MUL_MAT_ID at node_idx) whose only reader is
// such a reduction is never materialized. The tiled prefill path adds weight * row into the
// reduction's output from its GEMM; every other path writes the experts to pool scratch and reduces
// from there. The MUL_MAT_ID is reported through fusion_absorbs, so the [n_embd, n_expert_used,
// n_tokens] expert output gets no buffer; its sources stay live through the absorbed chain until the
// reduction's output. rm is the reduction it feeds.
bool ggml_sycl_match_moe_down_reduce(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_moe_reduce_match & rm);
int  ggml_sycl_moe_down_reduce_absorbs(const ggml_cgraph * cgraph, int node_idx);
// true when the reduction rm (its mul at mul_idx) was already computed by the down projection
bool ggml_sycl_moe_reduce_by_down(const ggml_cgraph * cgraph, const ggml_sycl_moe_reduce_match & rm, int mul_idx);

#endif  // GGML_SYCL_MOE_REDUCE_HPP
