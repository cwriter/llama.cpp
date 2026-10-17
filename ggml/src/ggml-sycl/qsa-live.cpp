#include "qsa-live.hpp"
#include "fusion.hpp"

#include "ggml-impl.h"

#include <cstring>

// The qwen4exp indexer turns its selection into the list that flash attention reads with 16
// nodes on a few hundred floats, one launch each (build_indexer):
//   live_pool = CLAMP(SCALE(top_score)), REPEAT-ed over the kpool cells of each pool
//   live_tail = CLAMP(SCALE(CAST(tail_idxs)))
//   live      = CONCAT(live_pool, live_tail)
//   dump      = SCALE(CUMSUM(FILL(live, 1)))
//   out       = CAST_I32(ADD(MUL(SUB(CAST_F32(sel_idx), dump), live), dump))
// Every slot depends only on its own inputs and its index, so one kernel at the final cast
// computes all of it. The 15 nodes before it are reported through fusion_absorbs: ggml-alloc
// reserves nothing for them, and keeps sel_idx, top_score and tail_idxs live until the cast.
// Every test below is structural, so the answer is the same before and after allocation.

// the chain is 15 nodes plus whatever the graph interleaves (the top_score gather)
static constexpr int SYCL_QSA_LIVE_WINDOW = 32;
static constexpr int SYCL_QSA_LIVE_NODES  = 15;

struct qsa_live_chain {
    const ggml_tensor * sel;    // i32 [n_sel, n_rows]: the cell of every slot
    const ggml_tensor * score;  // f32 [1, n_top, n_rows]: one per pool, shared by its kpool slots
    const ggml_tensor * tail;   // i32 [n_tail, n_rows]
    int64_t             n_sel, n_rows, n_top, kpool, n_tail;
    float               s_pool, b_pool, lo_pool, hi_pool;
    float               s_tail, b_tail, lo_tail, hi_tail;
    float               s_dump, b_dump;
    const ggml_tensor * nodes[SYCL_QSA_LIVE_NODES];
};

static int32_t qsa_live_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

static bool qsa_live_private(const ggml_cgraph * cgraph, const ggml_tensor * t, int32_t uses) {
    return (t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) == 0 &&
           (t->flags & GGML_TENSOR_FLAG_COMPUTE) && qsa_live_use_count(cgraph, t) == uses;
}

static bool qsa_live_f32(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

static bool qsa_live_op(const ggml_tensor * t, ggml_op op) {
    return qsa_live_f32(t) && t->op == op;
}

// ggml_cast: a CPY into a fresh tensor of the same shape
static bool qsa_live_cast(const ggml_tensor * t, ggml_type from, ggml_type to) {
    return t && t->op == GGML_OP_CPY && t->type == to && t->src[1] == t && t->src[0] && t->src[0]->type == from &&
           ggml_is_contiguous(t) && ggml_is_contiguous(t->src[0]) && ggml_are_same_shape(t, t->src[0]);
}

static void qsa_live_params(const ggml_tensor * t, float & a, float & b) {
    memcpy(&a, (const float *) t->op_params + 0, sizeof(float));
    memcpy(&b, (const float *) t->op_params + 1, sizeof(float));
}

// The structural test from the final cast back through src[] only, then every chain node must sit
// in the window before it, and nothing in between may write the inputs the kernel reads late.
static bool qsa_live_match(const ggml_cgraph * cgraph, int i_out, qsa_live_chain & c) {
    const ggml_tensor * out = cgraph->nodes[i_out];
    if (!qsa_live_cast(out, GGML_TYPE_F32, GGML_TYPE_I32) || out->ne[2] != 1 || out->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * add = out->src[0];
    if (!qsa_live_op(add, GGML_OP_ADD)) {
        return false;
    }
    const ggml_tensor * mul  = add->src[0];
    const ggml_tensor * dump = add->src[1];
    if (!qsa_live_op(mul, GGML_OP_MUL) || !qsa_live_op(dump, GGML_OP_SCALE)) {
        return false;
    }
    const ggml_tensor * sub  = mul->src[0];
    const ggml_tensor * live = mul->src[1];
    if (!qsa_live_op(sub, GGML_OP_SUB) || sub->src[1] != dump || !qsa_live_op(live, GGML_OP_CONCAT)) {
        return false;
    }
    const ggml_tensor * csel = sub->src[0];
    if (!qsa_live_cast(csel, GGML_TYPE_I32, GGML_TYPE_F32)) {
        return false;
    }
    for (const ggml_tensor * t : { add, mul, dump, sub, live, csel }) {
        if (!ggml_are_same_shape(t, out)) {
            return false;
        }
    }

    // dump = SCALE(CUMSUM(FILL(live, 1))): slot s gets (s + 1)*s_dump + b_dump
    const ggml_tensor * cs = dump->src[0];
    if (!qsa_live_op(cs, GGML_OP_CUMSUM) || !ggml_are_same_shape(cs, out)) {
        return false;
    }
    const ggml_tensor * fl = cs->src[0];
    if (!qsa_live_op(fl, GGML_OP_FILL) || fl->src[0] != live || !ggml_are_same_shape(fl, out) ||
        ggml_get_op_params_f32(fl, 0) != 1.0f) {
        return false;
    }

    // live = CONCAT(RESHAPE(REPEAT(CLAMP(SCALE(top_score)))), CLAMP(SCALE(CAST_F32(tail_idxs))))
    if (ggml_get_op_params_i32(live, 0) != 0) {
        return false;
    }
    const ggml_tensor * lp = live->src[0];
    const ggml_tensor * lt = live->src[1];
    if (!lp || lp->op != GGML_OP_RESHAPE || !qsa_live_f32(lp) || !qsa_live_op(lt, GGML_OP_CLAMP)) {
        return false;
    }
    const ggml_tensor * rp = lp->src[0];
    if (!qsa_live_op(rp, GGML_OP_REPEAT) || rp->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * cl_pool = rp->src[0];
    if (!qsa_live_op(cl_pool, GGML_OP_CLAMP) || cl_pool->ne[0] != 1 || cl_pool->ne[1] != rp->ne[1] ||
        cl_pool->ne[2] != rp->ne[2] || cl_pool->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * sc_pool = cl_pool->src[0];
    if (!qsa_live_op(sc_pool, GGML_OP_SCALE) || !ggml_are_same_shape(sc_pool, cl_pool)) {
        return false;
    }
    const ggml_tensor * score = sc_pool->src[0];
    if (!qsa_live_f32(score) || !ggml_are_same_shape(score, sc_pool)) {
        return false;
    }
    const ggml_tensor * sc_tail = lt->src[0];
    if (!qsa_live_op(sc_tail, GGML_OP_SCALE) || !ggml_are_same_shape(sc_tail, lt)) {
        return false;
    }
    const ggml_tensor * ctail = sc_tail->src[0];
    if (!qsa_live_cast(ctail, GGML_TYPE_I32, GGML_TYPE_F32) || !ggml_are_same_shape(ctail, lt)) {
        return false;
    }

    c.n_sel  = out->ne[0];
    c.n_rows = out->ne[1];
    c.kpool  = rp->ne[0];
    c.n_top  = rp->ne[1];
    c.n_tail = lt->ne[0];
    if (rp->ne[2] != c.n_rows || lp->ne[0] != c.kpool * c.n_top || lp->ne[1] != c.n_rows || lp->ne[2] != 1 ||
        lp->ne[3] != 1 || lt->ne[1] != c.n_rows || lt->ne[2] != 1 || lt->ne[3] != 1 ||
        lp->ne[0] + c.n_tail != c.n_sel) {
        return false;
    }
    // cumsum of ones and the casts are exact only below 2^24
    if (c.n_sel >= (1 << 24) || ggml_nelements(out) > INT32_MAX) {
        return false;
    }

    // every node the kernel replaces, with the uses it has inside the chain alone; a cast is its
    // own src[1], which counts as one more
    const ggml_tensor * nodes[SYCL_QSA_LIVE_NODES] = { csel, sc_pool, cl_pool, rp, lp, ctail, sc_tail, lt,
                                                       live, fl, cs, dump, sub, mul, add };
    const int32_t uses[SYCL_QSA_LIVE_NODES]        = { 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1 };
    for (int k = 0; k < SYCL_QSA_LIVE_NODES; ++k) {
        if (!qsa_live_private(cgraph, nodes[k], uses[k])) {
            return false;
        }
        c.nodes[k] = nodes[k];
    }

    // all of them in the window before the cast
    int       at[SYCL_QSA_LIVE_NODES];
    const int lo = std::max(0, i_out - SYCL_QSA_LIVE_WINDOW);
    for (int k = 0; k < SYCL_QSA_LIVE_NODES; ++k) {
        at[k] = -1;
        for (int j = i_out - 1; j >= lo; --j) {
            if (cgraph->nodes[j] == nodes[k]) {
                at[k] = j;
                break;
            }
        }
        if (at[k] < 0) {
            return false;
        }
    }
    c.sel   = csel->src[0];
    c.score = score;
    c.tail  = ctail->src[0];
    // the inputs are read at the cast instead of where the graph read them (the casts and the
    // pool scale); the top scores are usually produced inside the window, after the first cast
    if (ggml_sycl_written_between(cgraph, c.sel, at[0], i_out) ||
        ggml_sycl_written_between(cgraph, c.score, at[1], i_out) ||
        ggml_sycl_written_between(cgraph, c.tail, at[5], i_out)) {
        return false;
    }

    qsa_live_params(sc_pool, c.s_pool, c.b_pool);
    qsa_live_params(cl_pool, c.lo_pool, c.hi_pool);
    qsa_live_params(sc_tail, c.s_tail, c.b_tail);
    qsa_live_params(lt, c.lo_tail, c.hi_tail);
    qsa_live_params(dump, c.s_dump, c.b_dump);
    return true;
}

// the final cast whose chain holds node i, or -1; shared by fusion_absorbs and the compute loop
static int qsa_live_reader_of(const ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_live) {
        return -1;
    }
    const ggml_tensor * n = cgraph->nodes[i];
    switch (n->op) {
        case GGML_OP_CPY:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_REPEAT:
        case GGML_OP_RESHAPE:
        case GGML_OP_CONCAT:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_ADD:
            break;
        default:
            return -1;
    }
    if (n->type != GGML_TYPE_F32) {
        return -1;
    }
    const int end = std::min(cgraph->n_nodes, i + SYCL_QSA_LIVE_WINDOW + 1);
    for (int j = i + 1; j < end; ++j) {
        const ggml_tensor * o = cgraph->nodes[j];
        if (o->op != GGML_OP_CPY || o->type != GGML_TYPE_I32 || !o->src[0] || o->src[0]->op != GGML_OP_ADD) {
            continue;
        }
        qsa_live_chain c;
        if (!qsa_live_match(cgraph, j, c)) {
            continue;
        }
        for (const ggml_tensor * t : c.nodes) {
            if (t == n) {
                return j;
            }
        }
    }
    return -1;
}

int ggml_sycl_qsa_live_absorbs(const ggml_cgraph * cgraph, int node_idx) {
    return qsa_live_reader_of(cgraph, node_idx) >= 0 ? 1 : 0;
}

bool ggml_sycl_qsa_live(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_live) {
        return false;
    }
    qsa_live_chain c;
    if (!qsa_live_match(cgraph, i, c)) {
        return false;
    }
    ggml_tensor * out = cgraph->nodes[i];
    scope_op_debug_print scope_dbg_print(__func__, out, 0, " (QSA live list)");

    const int32_t * sel   = (const int32_t *) c.sel->data;
    const float *   score = (const float *) c.score->data;
    const int32_t * tail  = (const int32_t *) c.tail->data;
    int32_t *       dst   = (int32_t *) out->data;

    const int     n_sel  = (int) c.n_sel;
    const int     n_pool = (int) (c.kpool * c.n_top);
    const int     kpool  = (int) c.kpool;
    const int     n_top  = (int) c.n_top;
    const int     n_tail = (int) c.n_tail;
    const int     n      = (int) (c.n_sel * c.n_rows);
    const float   s_pool = c.s_pool, b_pool = c.b_pool, lo_pool = c.lo_pool, hi_pool = c.hi_pool;
    const float   s_tail = c.s_tail, b_tail = c.b_tail, lo_tail = c.lo_tail, hi_tail = c.hi_tail;
    const float   s_dump = c.s_dump, b_dump = c.b_dump;
    constexpr int wg     = 256;

    ctx.stream()->parallel_for(sycl::nd_range<1>(((n + wg - 1) / wg) * wg, wg), [=](sycl::nd_item<1> it) {
        // the graph rounds after every op, so no contraction across them
#pragma clang fp contract(off)
        const int i = (int) it.get_global_id(0);
        if (i >= n) {
            return;
        }
        const int s = i % n_sel;
        const int r = i / n_sel;
        float live;
        if (s < n_pool) {
            const float x = sycl::fma(score[r * n_top + s / kpool], s_pool, b_pool);
            live          = x < lo_pool ? lo_pool : (x > hi_pool ? hi_pool : x);
        } else {
            const float x = sycl::fma((float) tail[r * n_tail + s - n_pool], s_tail, b_tail);
            live          = x < lo_tail ? lo_tail : (x > hi_tail ? hi_tail : x);
        }
        const float dump = sycl::fma((float) (s + 1), s_dump, b_dump);
        const float d    = (float) sel[i] - dump;
        const float m    = d * live;
        dst[i]           = (int32_t) (m + dump);
    });
    return true;
}
