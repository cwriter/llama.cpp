#include "moe-reduce.hpp"

#include "ggml-impl.h"

#include <vector>

// one work-item per (token, embedding column); the expert axis is walked inside the item, so
// the intermediate weighted buffer is never written and never read back
static void moe_weighted_reduction_f32_sycl(const float * experts, const float * weights, float * dst,
                                            const int64_t n_embd, const int64_t n_tokens,
                                            const int n_expert_used, const int64_t stride_expert,
                                            const int64_t stride_token, const int64_t stride_w_expert,
                                            const int64_t stride_w_token, const int64_t stride_dst_token,
                                            queue_ptr stream) {
    constexpr int block = 256;
    const size_t  cols  = (size_t) ((n_embd + block - 1) / block);

    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) n_tokens, cols * block), sycl::range<2>(1, block)),
        [=](sycl::nd_item<2> item) {
            const int64_t token = (int64_t) item.get_global_id(0);
            const int64_t col   = (int64_t) item.get_global_id(1);
            if (col >= n_embd) {
                return;
            }

            const float * e = experts + token * stride_token + col;
            const float * w = weights + token * stride_w_token;

            float sum = 0.0f;
            for (int expert = 0; expert < n_expert_used; ++expert) {
                sum += e[expert * stride_expert] * w[expert * stride_w_expert];
            }
            dst[token * stride_dst_token + col] = sum;
        });
}

bool ggml_sycl_match_moe_weighted_reduction(const ggml_cgraph * cgraph, int node_idx,
                                            ggml_sycl_moe_reduce_match & match) {
    if (!g_ggml_sycl_enable_fusion || !(g_ggml_sycl_fuse_types & GGML_SYCL_FUSE_MOE_REDUCE)) {
        return false;
    }

    const ggml_tensor * mul = cgraph->nodes[node_idx];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul)) {
        return false;
    }

    // one operand is [n_embd, n_expert_used, n_tokens], the other the [1, n_expert_used, n_tokens]
    // routing weight. Only that shape is handled; anything else falls through to the plain ops.
    const auto is_weights = [mul](const ggml_tensor * t) {
        return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == 1 &&
               t->ne[1] == mul->ne[1] && t->ne[2] == mul->ne[2] && t->ne[3] == mul->ne[3];
    };
    const ggml_tensor * experts = nullptr;
    const ggml_tensor * weights = nullptr;
    if (is_weights(mul->src[1])) {
        experts = mul->src[0];
        weights = mul->src[1];
    } else if (is_weights(mul->src[0])) {
        experts = mul->src[1];
        weights = mul->src[0];
    } else {
        return false;
    }
    if (experts->type != GGML_TYPE_F32 || !ggml_is_contiguous(experts) ||
        !ggml_are_same_shape(experts, mul)) {
        return false;
    }

    const int     n_expert_used = (int) mul->ne[1];
    const int64_t n_tokens      = mul->ne[2] * mul->ne[3];
    // a single expert leaves no add chain to absorb, and the span must fit ggml_can_fuse_subgraph.
    // an empty span (a ubatch with no outputs) still matches: graph_optimize then adds the same
    // alloc deps, so the graph keeps one topology and ggml-alloc keeps its plan. Compute skips
    // empty nodes before it tries any fusion.
    if (n_expert_used < 2 || n_expert_used > GGML_SYCL_MOE_REDUCE_MAX_EXPERTS || n_tokens < 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used;  // the mul, one view per expert, n-1 adds
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    // walk the span, checking each view addresses exactly its expert slice and each add extends
    // the left-leaning chain, then hand the whole thing to the subgraph matcher
    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;

    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int                 n_adds   = 0;

    for (int offset = 1; offset < node_count; ++offset) {
        const ggml_tensor * node = cgraph->nodes[node_idx + offset];
        ops[offset] = node->op;

        if (node->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || node->src[0] != mul || node->view_src != mul ||
                node->type != GGML_TYPE_F32 || node->ne[0] != mul->ne[0] || node->ne[1] != n_tokens ||
                node->ne[2] != 1 || node->ne[3] != 1 || node->nb[0] != mul->nb[0] ||
                node->nb[1] != mul->nb[2] || node->view_offs != (size_t) expert * mul->nb[1]) {
                return false;
            }
            views.push_back(node);
            continue;
        }

        if (node->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = (n_adds == 0) ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (node->src[0] != lhs || node->src[1] != rhs || node->type != GGML_TYPE_F32) {
            return false;
        }
        previous = node;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != mul->ne[0] || previous->ne[1] != n_tokens ||
        previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    // the last add is the only tensor the rest of the graph may read: this is what makes skipping
    // the mul's own output and every view legal
    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts    = experts;
    match.weights    = weights;
    match.dst        = cgraph->nodes[output_idx];
    match.node_count = node_count;
    return true;
}

int ggml_sycl_moe_reduce_absorbs(const ggml_cgraph * cgraph, int node_idx) {
    static const bool on = ggml_sycl_get_env("GGML_SYCL_MOE_REDUCE_ABSORB", 1) != 0;
    const ggml_tensor * n = cgraph->nodes[node_idx];
    if (!on || (n->op != GGML_OP_MUL && n->op != GGML_OP_VIEW && n->op != GGML_OP_ADD)) {
        return 0;
    }
    // the span starts at its mul and is 2 * n_expert_used nodes long
    for (int j = node_idx; j >= 0 && j > node_idx - 2 * GGML_SYCL_MOE_REDUCE_MAX_EXPERTS; --j) {
        if (cgraph->nodes[j]->op != GGML_OP_MUL) {
            continue;
        }
        ggml_sycl_moe_reduce_match m;
        if (ggml_sycl_match_moe_weighted_reduction(cgraph, j, m)) {
            return node_idx < j + m.node_count - 1 ? 1 : 0;
        }
    }
    return 0;
}

static constexpr int GGML_SYCL_MOE_DOWN_WINDOW = 16;

static int32_t moe_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

bool ggml_sycl_match_moe_down_reduce(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_moe_reduce_match & rm) {
    static const bool on = ggml_sycl_get_env("GGML_SYCL_MOE_DOWN_REDUCE", 1) != 0;
    const ggml_tensor * d = cgraph->nodes[node_idx];
    if (!on || d->op != GGML_OP_MUL_MAT_ID || d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || d->view_src ||
        (d->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) || !(d->flags & GGML_TENSOR_FLAG_COMPUTE) ||
        moe_use_count(cgraph, d) != 1) {
        return false;
    }
    const int end = std::min(cgraph->n_nodes, node_idx + GGML_SYCL_MOE_DOWN_WINDOW + 1);
    for (int j = node_idx + 1; j < end; ++j) {
        if (cgraph->nodes[j]->op != GGML_OP_MUL) {
            continue;
        }
        if (ggml_sycl_match_moe_weighted_reduction(cgraph, j, rm)) {
            return rm.experts == d;
        }
    }
    return false;
}

int ggml_sycl_moe_down_reduce_absorbs(const ggml_cgraph * cgraph, int node_idx) {
    ggml_sycl_moe_reduce_match rm;
    return cgraph->nodes[node_idx]->op == GGML_OP_MUL_MAT_ID && ggml_sycl_match_moe_down_reduce(cgraph, node_idx, rm) ? 1 : 0;
}

bool ggml_sycl_moe_reduce_by_down(const ggml_cgraph * cgraph, const ggml_sycl_moe_reduce_match & rm, int mul_idx) {
    if (!rm.experts || rm.experts->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    for (int j = mul_idx - 1; j >= 0 && j >= mul_idx - GGML_SYCL_MOE_DOWN_WINDOW; --j) {
        if (cgraph->nodes[j] == rm.experts) {
            ggml_sycl_moe_reduce_match m;
            return ggml_sycl_match_moe_down_reduce(cgraph, j, m);
        }
    }
    return false;
}

void ggml_sycl_op_moe_weighted_reduction(ggml_backend_sycl_context & ctx,
                                         const ggml_sycl_moe_reduce_match & match) {
    scope_op_debug_print scope_dbg_print(__func__, match.dst, /*num_src=*/0);

    const ggml_tensor * experts = match.experts;
    const ggml_tensor * weights = match.weights;
    ggml_tensor *       dst     = match.dst;

    const int64_t n_embd        = experts->ne[0];
    const int     n_expert_used = (int) experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    moe_weighted_reduction_f32_sycl(
        (const float *) experts->data, (const float *) weights->data, (float *) dst->data,
        n_embd, n_tokens, n_expert_used,
        (int64_t) (experts->nb[1] / sizeof(float)), (int64_t) (experts->nb[2] / sizeof(float)),
        (int64_t) (weights->nb[1] / sizeof(float)), (int64_t) (weights->nb[2] / sizeof(float)),
        (int64_t) (dst->nb[1] / sizeof(float)), ctx.stream());
}
