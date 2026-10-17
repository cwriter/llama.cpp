#include "hc-ffn-tail.hpp"
#include "fusion.hpp"
#include "moe-reduce.hpp"

#include "ggml-impl.h"

// Every qwen4exp FFN block ends in
//   moe_out = MUL(experts, weights) -> per-expert VIEWs -> ADD chain      (moe_weighted_reduction)
//   ffn_out = ADD(moe_out, MUL(shexp, sigmoid(shared gate)))              (mul + add)
//   hc_post(ffn_out, residual, gate) -> RMS_NORM -> MUL                   (hc_post + norm fusion)
// and the hc_post + norm kernel reads ffn_out once per element. It can as well compute it there:
// the weighted sum over the few experts plus the gated shared expert, from the tensors the reduction
// and the add read. The reduction span, the gated MUL and the ADD are reported through
// fusion_absorbs and never launch; graph_optimize keeps their inputs live until the norm output.
// Every test is structural, so the answer is the same before and after allocation.

static constexpr int SYCL_HC_FFN_TAIL_WINDOW = 96;

static int32_t hc_ffn_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

static bool hc_ffn_private(const ggml_cgraph * cgraph, const ggml_tensor * t, int32_t uses) {
    return (t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) == 0 &&
           (t->flags & GGML_TENSOR_FLAG_COMPUTE) && hc_ffn_use_count(cgraph, t) == uses;
}

static int hc_ffn_find(const ggml_cgraph * cgraph, const ggml_tensor * t, int before) {
    for (int j = before - 1; j >= std::max(0, before - SYCL_HC_FFN_TAIL_WINDOW); --j) {
        if (cgraph->nodes[j] == t) {
            return j;
        }
    }
    return -1;
}

bool ggml_sycl_hc_post_norm_at(const ggml_cgraph * cgraph, int i) {
    if (i + 5 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    if (n[0]->op != GGML_OP_SCALE || n[1]->op != GGML_OP_UNARY || n[4]->op != GGML_OP_RMS_NORM ||
        n[5]->op != GGML_OP_MUL) {
        return false;
    }
    if (!ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST },
                            { GGML_UNARY_OP_SIGMOID })) {
        return false;
    }
    if (!ggml_sycl_dsv4_hc_post_norm_ok(n[3], n[4], n[5]) || n[0]->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    return ggml_can_fuse_subgraph(cgraph, i,
        { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST, GGML_OP_RMS_NORM, GGML_OP_MUL },
        { i + 3, i + 5 });
}

bool ggml_sycl_match_hc_ffn_tail(const ggml_cgraph * cgraph, int i, ggml_sycl_hc_ffn_tail & tail) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_hc_ffn_tail || !ggml_sycl_hc_post_norm_at(cgraph, i)) {
        return false;
    }
    const ggml_tensor * hp       = cgraph->nodes[i + 3];
    const ggml_tensor * x        = hp->src[0];
    const int64_t       n_embd   = hp->ne[0];
    const int64_t       n_tokens = hp->ne[2];
    if (!x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != n_embd ||
        ggml_nrows(x) != n_tokens || !hc_ffn_private(cgraph, x, 1)) {
        return false;
    }

    tail         = {};
    tail.n_nodes = 0;

    // ffn_out = ADD(moe_out, MUL(shexp, sgate)), or moe_out itself without a shared expert
    const ggml_tensor * moe_out = x;
    const ggml_tensor * gated   = nullptr;
    // (the reduction's own last add has two VIEW/ADD operands, never a MUL)
    if (x->op == GGML_OP_ADD && x->src[0] && x->src[1] &&
        (x->src[0]->op == GGML_OP_MUL || x->src[1]->op == GGML_OP_MUL)) {
        const ggml_tensor * a = x->src[0];
        const ggml_tensor * b = x->src[1];
        if (!ggml_are_same_shape(a, x) || !ggml_are_same_shape(b, x)) {
            return false;
        }
        moe_out = b->op == GGML_OP_MUL ? a : b;
        gated   = b->op == GGML_OP_MUL ? b : a;
        if (gated->op != GGML_OP_MUL || gated->type != GGML_TYPE_F32 || !hc_ffn_private(cgraph, gated, 1) ||
            !hc_ffn_private(cgraph, moe_out, 1)) {
            return false;
        }
        const ggml_tensor * s0 = gated->src[0];
        const ggml_tensor * s1 = gated->src[1];
        const ggml_tensor * sh = s0 && s0->ne[0] == n_embd ? s0 : s1;
        const ggml_tensor * sg = sh == s0 ? s1 : s0;
        if (!sh || !sg || sh->type != GGML_TYPE_F32 || sg->type != GGML_TYPE_F32 || !ggml_are_same_shape(sh, x) ||
            !ggml_is_contiguous(sh) || sg->ne[0] != 1 || ggml_nelements(sg) != n_tokens || sh->nb[1] % 16 != 0) {
            return false;
        }
        tail.x.shexp = sh;
        tail.x.sgate = sg;
    }

    // moe_out must be the last add of a weighted reduction
    const ggml_tensor * t = moe_out;
    while (t && t->op == GGML_OP_ADD) {
        t = t->src[0];
    }
    if (!t || t->op != GGML_OP_VIEW || !t->src[0]) {
        return false;
    }
    const int j = hc_ffn_find(cgraph, t->src[0], i);
    ggml_sycl_moe_reduce_match rm;
    if (j < 0 || !ggml_sycl_match_moe_weighted_reduction(cgraph, j, rm) || rm.dst != moe_out) {
        return false;
    }
    const ggml_tensor * ex = rm.experts;
    if (ex->ne[0] != n_embd || ex->ne[2] * ex->ne[3] != n_tokens || ex->nb[1] % 16 != 0 || ex->nb[2] % 16 != 0) {
        return false;
    }
    tail.x.experts = ex;
    tail.x.weights = rm.weights;

    // the nodes that never launch, all in the window before the hc_post fusion
    for (int k = 0; k < rm.node_count; ++k) {
        tail.nodes[tail.n_nodes++] = cgraph->nodes[j + k];
    }
    int at_gated = -1;
    if (gated) {
        at_gated = hc_ffn_find(cgraph, gated, i);
        if (at_gated < 0 || hc_ffn_find(cgraph, x, i) < 0) {
            return false;
        }
        tail.nodes[tail.n_nodes++] = gated;
        tail.nodes[tail.n_nodes++] = x;
    }

    // the inputs are now read at the hc_post fusion, later than the graph read them
    if (ggml_sycl_written_between(cgraph, rm.experts, j, i) || ggml_sycl_written_between(cgraph, rm.weights, j, i)) {
        return false;
    }
    if (gated && (ggml_sycl_written_between(cgraph, tail.x.shexp, at_gated, i) ||
                  ggml_sycl_written_between(cgraph, tail.x.sgate, at_gated, i))) {
        return false;
    }
    return true;
}

int ggml_sycl_hc_ffn_tail_absorbs(const ggml_cgraph * cgraph, int node_idx) {
    // 2: a diagnostic mode, the kernel computes the tail but its nodes still run
    if (!g_ggml_sycl_enable_fusion || g_ggml_sycl_fuse_hc_ffn_tail != 1) {
        return 0;
    }
    const ggml_tensor * n = cgraph->nodes[node_idx];
    if ((n->op != GGML_OP_MUL && n->op != GGML_OP_VIEW && n->op != GGML_OP_ADD) || n->type != GGML_TYPE_F32) {
        return 0;
    }
    const int end = std::min(cgraph->n_nodes, node_idx + SYCL_HC_FFN_TAIL_WINDOW + 1);
    for (int i = node_idx + 1; i < end; ++i) {
        if (cgraph->nodes[i]->op != GGML_OP_SCALE || i + 3 >= cgraph->n_nodes ||
            cgraph->nodes[i + 3]->op != GGML_OP_DSV4_HC_POST) {
            continue;
        }
        ggml_sycl_hc_ffn_tail tail;
        if (!ggml_sycl_match_hc_ffn_tail(cgraph, i, tail)) {
            continue;
        }
        for (int k = 0; k < tail.n_nodes; ++k) {
            if (tail.nodes[k] == n) {
                return 1;
            }
        }
    }
    return 0;
}
