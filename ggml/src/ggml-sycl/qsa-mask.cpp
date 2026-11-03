#include "qsa-mask.hpp"
#include "kq-mask-bits.hpp"

#include "ggml-impl.h"

#include "common.hpp"
#include "fattn-qsa.hpp"
#include "top-k-mask.hpp"

#include <cmath>
#include <initializer_list>

// QSA attends only to the cells that the indexer selected. The qwen4exp graph says that with
//   REPEAT(FILL(seed, -inf)) -> SET_ROWS(0 at sel) -> VIEW -> ADD(kq_mask) -> FLASH_ATTN_EXT
// where the REPEAT has one row more than the cache for the n_kv sentinel of padded blocks. That
// is two more full [n_kv, n_tps] masks, 256 MiB each at 128k context. Because -inf + x == -inf
// and 0 + x == x, the chain is exactly
//   out[c, t] = selected(c, t) ? kq_mask[c, t] : -inf
// so flash attention takes the selection list and the causal mask instead (fattn-qsa.cpp), and
// every node of the chain is reported through fusion_absorbs: ggml-alloc reserves nothing for
// them, and keeps the list and the mask live until the flash attention node releases the chain.
// Every test below is structural, so the answer is the same before and after allocation.
//
// The DSA graph (llama-graph.cpp, build_attn with top_k) says the same with plain fills,
//   FILL(kq_mask, -inf) -> SET_ROWS(FILL(0) at top_k) -> VIEW -> ADD(kq_mask) -> FLASH_ATTN_EXT
// and, when the backend offers GGML_OP_TOP_K_MASK, in one node,
//   TOP_K_MASK(kq_mask, top_k) -> FLASH_ATTN_EXT
// which is the same selection without scores, blocks or extra cells; both run the same way.
//
// The k-pool models keep only cells of the token's own sequence, so they need no causal mask:
//   qwen4exp:  REPEAT(FILL(-inf)) -> SET_ROWS(0 at sel) -> VIEW (drops the dump rows) -> CONT -> FLASH_ATTN_EXT
//   fused:     TOP_K_MASK(FILL(0), top_k, score, blocks, extra) -> FLASH_ATTN_EXT
// Every listed cell is visible there, so flash attention gets an all-visible mask. A TOP_K_MASK with
// scores, blocks or extra cells has its cell list built first (one I32 per pick cell and extra cell,
// -1 for a dead pick), which is what the scatter would have read.

// the chain sits right before its flash attention node; a chain node further away is left to run
static constexpr int SYCL_QSA_MASK_WINDOW = 48;
static constexpr int SYCL_QSA_MASK_MAX    = 16;

struct qsa_mask_chain {
    const ggml_tensor * mask;  // the causal kq_mask, or null: every listed cell is visible
    ggml_tensor         idx;   // the selection list seen as [width, n_tps, n_stream]
    const ggml_tensor * gen;   // a TOP_K_MASK whose list is built at run time into idx, or null
    const ggml_tensor * add;   // the dense mask the graph would build
    const ggml_tensor * nodes[SYCL_QSA_MASK_MAX];
    int                 n_nodes;
};

// ggml_node_get_use_count() keyed by tensor, so the chain can be walked through src[] alone
static int32_t qsa_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

static bool qsa_private(const ggml_cgraph * cgraph, const ggml_tensor * t, int32_t uses) {
    return t && (t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) == 0 &&
           (t->flags & GGML_TENSOR_FLAG_COMPUTE) && qsa_use_count(cgraph, t) == uses;
}

// a REPEAT of a FILL with this value, or that FILL alone, possibly seen through a view; returns
// the REPEAT, or the FILL when there is none
static const ggml_tensor * qsa_filled(const ggml_tensor * t, float value) {
    const ggml_tensor * rp   = t->view_src ? t->view_src : t;
    const ggml_tensor * fill = rp->op == GGML_OP_FILL ? rp : rp->op == GGML_OP_REPEAT ? rp->src[0] : nullptr;
    if (!fill || fill->op != GGML_OP_FILL || fill->view_src) {
        return nullptr;
    }
    const float v = ggml_get_op_params_f32(fill, 0);
    if (std::isinf(value) ? !(std::isinf(v) && v < 0.0f) : v != value) {
        return nullptr;
    }
    return rp;
}

static void qsa_add_node(qsa_mask_chain & c, const ggml_tensor * t) {
    for (int i = 0; i < c.n_nodes; ++i) {
        if (c.nodes[i] == t) {
            return;
        }
    }
    GGML_ASSERT(c.n_nodes < SYCL_QSA_MASK_MAX);
    c.nodes[c.n_nodes++] = t;
}

// Adds the nodes that only build the dense mask (all used once, by the chain), and the scalars the
// fills are seeded from when only the chain reads them.
static bool qsa_add_private(const ggml_cgraph * cgraph, qsa_mask_chain & c,
                            std::initializer_list<const ggml_tensor *> nodes) {
    for (const ggml_tensor * t : nodes) {
        if (!qsa_private(cgraph, t, 1)) {
            return false;
        }
        qsa_add_node(c, t);
    }
    // the fills are seeded from a scalar that only the chain reads (one element of the list, cast
    // to the mask type), so that the dense mask is built after the list; those nodes go too
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < c.n_nodes; ++i) {
            const ggml_tensor * s = c.nodes[i]->src[0];
            if (!s || s->op == GGML_OP_NONE || ggml_nelements(s) != 1) {
                continue;
            }
            int32_t uses = 0;
            for (int k = 0; k < c.n_nodes; ++k) {
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    uses += c.nodes[k]->src[j] == s;
                }
            }
            if (qsa_private(cgraph, s, uses)) {
                qsa_add_node(c, s);
            }
        }
    }
    return true;
}

// TOP_K_MASK keeps mask[c, i] for the cells c of the live picks and the extra cells of row i, and
// -inf elsewhere. The mask is either the causal mask at the flash attention shape, or a private
// FILL(0) (nothing hidden). Without scores, blocks or extra cells top_k is the list as it stands.
static bool qsa_top_k_mask_chain(const ggml_cgraph * cgraph, const ggml_tensor * fa, const ggml_tensor * tk,
                                 qsa_mask_chain & c) {
    const ggml_tensor * fm = fa->src[3];  // what flash attention reads, possibly a reshape of tk
    if (tk->view_src || !ggml_is_contiguous(tk) || tk->type != GGML_TYPE_F16 || tk->ne[0] != fm->ne[0] ||
        ggml_nelements(tk) != ggml_nelements(fm) || !qsa_private(cgraph, tk, 1)) {
        return false;
    }
    const ggml_tensor * mask   = tk->src[0];
    const ggml_tensor * top_k  = tk->src[1];
    const ggml_tensor * score  = tk->src[2];
    const ggml_tensor * blocks = tk->src[3];
    const ggml_tensor * extra  = tk->src[4];
    if (!mask || !top_k || top_k->type != GGML_TYPE_I32 || top_k->nb[0] != sizeof(int32_t) || top_k->ne[0] < 1) {
        return false;
    }

    c.n_nodes = 0;
    c.add     = tk;
    c.gen     = nullptr;
    qsa_add_node(c, tk);

    if (mask->op == GGML_OP_FILL && !mask->view_src && ggml_get_op_params_f32(mask, 0) == 0.0f &&
        mask->type == tk->type) {
        // the row every pick reads is all zeros: nothing is hidden, and the fill is never built
        if (!qsa_add_private(cgraph, c, { mask })) {
            return false;
        }
        c.mask = nullptr;
    } else if (mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask) && ggml_are_same_shape(mask, fm) &&
               mask->ne[0] <= INT32_MAX) {
        c.mask = mask;
    } else {
        return false;
    }

    const int64_t n_tps    = fm->ne[1];
    const int64_t n_stream = fm->ne[3];
    if (!score && !blocks && !extra && top_k->ne[1] == n_tps && top_k->ne[2] == 1 && top_k->ne[3] == n_stream &&
        top_k->ne[0] <= INT32_MAX) {
        // [n_top, n_tps, 1, n_stream] seen as [width, n_tps, n_stream]
        c.idx       = *top_k;
        c.idx.ne[2] = top_k->ne[3];
        c.idx.ne[3] = 1;
        c.idx.nb[2] = top_k->nb[3];
        c.idx.nb[3] = top_k->nb[3] * top_k->ne[3];
        return true;
    }

    // the list is built into pool scratch before flash attention runs, one row per row of tk
    const int64_t width = top_k->ne[0] * (blocks ? blocks->ne[0] : 1) + (extra ? extra->ne[0] : 0);
    if (width < 1 || width > INT32_MAX) {
        return false;
    }
    c.gen       = tk;
    c.idx       = *top_k;
    c.idx.op    = GGML_OP_NONE;
    c.idx.data  = nullptr;
    c.idx.ne[0] = width;
    c.idx.ne[1] = n_tps;
    c.idx.ne[2] = n_stream;
    c.idx.ne[3] = 1;
    c.idx.nb[0] = sizeof(int32_t);
    c.idx.nb[1] = sizeof(int32_t) * width;
    c.idx.nb[2] = c.idx.nb[1] * n_tps;
    c.idx.nb[3] = c.idx.nb[2] * n_stream;
    return true;
}

// qwen4exp's selection without a causal mask: the scatter rows hold n_kv cells and then one dump
// cell per slot, a VIEW keeps the first n_kv of each row, and a CONT makes that the dense mask
static bool qsa_cont_chain(const ggml_cgraph * cgraph, const ggml_tensor * fa, const ggml_tensor * ct,
                           qsa_mask_chain & c) {
    const ggml_tensor * fm = fa->src[3];
    if (ct->view_src || !ggml_is_contiguous(ct) || ct->type != fm->type || ct->ne[0] != fm->ne[0] ||
        ggml_nelements(ct) != ggml_nelements(fm) || !qsa_private(cgraph, ct, 1)) {
        return false;
    }
    const ggml_tensor * rv = ct->src[0];
    if (!rv || rv->op != GGML_OP_VIEW || rv->view_offs != 0 || ggml_nelements(rv) != ggml_nelements(ct) ||
        rv->ne[0] != ct->ne[0]) {
        return false;
    }
    const ggml_tensor * sr = rv->src[0];
    if (!sr || sr->op != GGML_OP_SET_ROWS) {
        return false;
    }
    const ggml_tensor * fz = sr->src[0];
    const ggml_tensor * iv = sr->src[1];
    const ggml_tensor * mv = sr->src[2];
    if (!fz || !iv || !mv) {
        return false;
    }
    const ggml_tensor * rp_inf  = qsa_filled(mv, -INFINITY);
    const ggml_tensor * rp_zero = qsa_filled(fz, 0.0f);
    if (!rp_inf || !rp_zero || sr->view_src != rp_inf || rv->view_src != rp_inf) {
        return false;
    }
    if (rp_inf->type != fm->type || !ggml_is_contiguous(rp_inf)) {
        return false;
    }

    const int64_t n_kv     = fm->ne[0];
    const int64_t n_tps    = fm->ne[1];
    const int64_t n_stream = fm->ne[3];
    const int64_t n_rows   = n_tps * n_stream;
    const int64_t width    = iv->ne[0];
    const size_t  ts       = ggml_type_size(fm->type);

    // mv is [1, n_cells, n_rows] with n_cells >= n_kv; rv reads the first n_kv cells of every row
    if (mv->ne[0] != 1 || mv->ne[1] < n_kv || mv->ne[1] * mv->ne[2] * mv->ne[3] != ggml_nelements(rp_inf) ||
        mv->ne[2] * mv->ne[3] != n_rows || !ggml_is_contiguous(mv) || mv->view_offs != 0) {
        return false;
    }
    const size_t row = (size_t) mv->ne[1] * ts;
    if (rv->nb[0] != ts || rv->nb[1] != row || rv->ne[1] * rv->ne[2] * rv->ne[3] != n_rows ||
        (rv->ne[2] > 1 && rv->nb[2] != row * rv->ne[1]) ||
        (rv->ne[3] > 1 && rv->nb[3] != row * rv->ne[1] * rv->ne[2])) {
        return false;
    }
    if (iv->type != GGML_TYPE_I32 || iv->nb[0] != sizeof(int32_t) || width < 1 || width > INT32_MAX) {
        return false;
    }
    if (iv->ne[1] * iv->ne[2] != n_rows || iv->ne[3] != 1 || iv->nb[2] != iv->nb[1] * iv->ne[1]) {
        return false;
    }
    if (fz->ne[0] != 1 || fz->ne[1] != width || fz->ne[2] * fz->ne[3] != n_rows) {
        return false;
    }

    c.mask    = nullptr;
    c.add     = ct;
    c.gen     = nullptr;
    c.n_nodes = 0;

    c.idx       = *iv;
    c.idx.ne[1] = n_tps;
    c.idx.ne[2] = n_stream;
    c.idx.nb[2] = iv->nb[1] * n_tps;
    c.idx.nb[3] = c.idx.nb[2] * n_stream;

    const ggml_tensor * fill_inf  = rp_inf->op == GGML_OP_REPEAT ? rp_inf->src[0] : rp_inf;
    const ggml_tensor * fill_zero = rp_zero->op == GGML_OP_REPEAT ? rp_zero->src[0] : rp_zero;
    return qsa_add_private(cgraph, c, { ct, rv, sr, mv, rp_inf, fill_inf, fz, rp_zero, fill_zero });
}

// The whole structural test, from the flash attention node back through src[] only.
static bool qsa_mask_chain_from_fa(const ggml_cgraph * cgraph, const ggml_tensor * fa, qsa_mask_chain & c) {
    if (!fa || fa->op != GGML_OP_FLASH_ATTN_EXT || !fa->src[3] || !ggml_sycl_qsa_sparse_fa_supported(fa)) {
        return false;
    }

    // the mask may reach flash attention through a reshape
    const ggml_tensor * ad = fa->src[3];
    if (ad->op == GGML_OP_RESHAPE) {
        ad = ad->src[0];
    }
    if (ad && ad->op == GGML_OP_TOP_K_MASK) {
        return qsa_top_k_mask_chain(cgraph, fa, ad, c);
    }
    if (ad && ad->op == GGML_OP_CONT) {
        return qsa_cont_chain(cgraph, fa, ad, c);
    }
    if (!ad || ad->op != GGML_OP_ADD || ad->view_src || !ggml_is_contiguous(ad) || !qsa_private(cgraph, ad, 1)) {
        return false;
    }

    const ggml_tensor * rv   = ad->src[0];  // the selection rows, n_kv cells each
    const ggml_tensor * mask = ad->src[1];
    if (!rv || !mask || rv->op != GGML_OP_VIEW || !qsa_private(cgraph, rv, 1)) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) || mask->ne[2] != 1 || mask->ne[0] > INT32_MAX) {
        return false;
    }
    if (ad->type != mask->type || !ggml_are_same_shape(ad, mask) || !ggml_are_same_shape(rv, mask)) {
        return false;
    }

    const ggml_tensor * sr = rv->src[0];
    if (!sr || sr->op != GGML_OP_SET_ROWS || !qsa_private(cgraph, sr, 1)) {
        return false;
    }
    const ggml_tensor * fz = sr->src[0];  // the zeros it writes
    const ggml_tensor * iv = sr->src[1];  // the selection list
    const ggml_tensor * mv = sr->src[2];  // the -inf rows seen as rows of one cell
    if (!fz || !iv || !mv) {
        return false;
    }

    const ggml_tensor * rp_inf  = qsa_filled(mv, -INFINITY);
    const ggml_tensor * rp_zero = qsa_filled(fz, 0.0f);
    if (!rp_inf || !rp_zero || sr->view_src != rp_inf || rv->view_src != rp_inf || rv->view_offs != 0) {
        return false;
    }
    if (rp_inf->type != mask->type || !ggml_is_contiguous(rp_inf)) {
        return false;
    }

    const int64_t n_kv     = mask->ne[0];
    const int64_t n_tps    = mask->ne[1];
    const int64_t n_stream = mask->ne[3];
    const int64_t n_rows   = n_tps * n_stream;
    const int64_t width    = iv->ne[0];
    const size_t  ts       = ggml_type_size(mask->type);

    // mv is [1, n_cells, n_rows] with n_cells >= n_kv; rv reads the first n_kv cells of every row
    if (mv->ne[0] != 1 || mv->ne[1] < n_kv || mv->ne[1] * mv->ne[2] * mv->ne[3] != ggml_nelements(rp_inf)) {
        return false;
    }
    if (mv->ne[2] * mv->ne[3] != n_rows || !ggml_is_contiguous(mv) || mv->view_offs != 0) {
        return false;
    }
    const size_t row = (size_t) mv->ne[1] * ts;
    if (rv->nb[0] != ts || rv->nb[1] != row || rv->nb[2] != row * rv->ne[1] || rv->nb[3] != row * rv->ne[1] * rv->ne[2]) {
        return false;
    }

    if (iv->type != GGML_TYPE_I32 || iv->nb[0] != sizeof(int32_t) || width < 1 || width > INT32_MAX) {
        return false;
    }
    if (iv->ne[1] * iv->ne[2] != n_rows || iv->ne[3] != 1 || iv->nb[2] != iv->nb[1] * iv->ne[1]) {
        return false;
    }
    if (fz->ne[0] != 1 || fz->ne[1] != width || fz->ne[2] * fz->ne[3] != n_rows) {
        return false;
    }

    c.mask    = mask;
    c.add     = ad;
    c.gen     = nullptr;
    c.n_nodes = 0;

    c.idx       = *iv;
    c.idx.ne[1] = n_tps;
    c.idx.ne[2] = n_stream;
    c.idx.nb[2] = iv->nb[1] * n_tps;
    c.idx.nb[3] = c.idx.nb[2] * n_stream;

    // the nodes that only build the dense mask. A plain FILL is its own fill: its source only gives
    // the shape (for DSA the causal mask itself, which stays live as the chain's mask)
    const ggml_tensor * fill_inf  = rp_inf->op == GGML_OP_REPEAT ? rp_inf->src[0] : rp_inf;
    const ggml_tensor * fill_zero = rp_zero->op == GGML_OP_REPEAT ? rp_zero->src[0] : rp_zero;
    return qsa_add_private(cgraph, c, { ad, rv, sr, mv, rp_inf, fill_inf, fz, rp_zero, fill_zero });
}

// The flash attention node whose chain holds node i, or -1. Shared by fusion_absorbs and the
// compute loop, so both see the same chains.
static int qsa_mask_reader_of(const ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_fa_mask) {
        return -1;
    }
    const ggml_tensor * n = cgraph->nodes[i];
    switch (n->op) {
        case GGML_OP_ADD:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_SET_ROWS:
        case GGML_OP_REPEAT:
        case GGML_OP_FILL:
        case GGML_OP_TOP_K_MASK:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
            break;
        default:
            return -1;
    }
    const int end = std::min(cgraph->n_nodes, i + SYCL_QSA_MASK_WINDOW);
    for (int j = i + 1; j < end; ++j) {
        if (cgraph->nodes[j]->op != GGML_OP_FLASH_ATTN_EXT) {
            continue;
        }
        qsa_mask_chain c;
        if (!qsa_mask_chain_from_fa(cgraph, cgraph->nodes[j], c)) {
            continue;
        }
        for (int k = 0; k < c.n_nodes; ++k) {
            if (c.nodes[k] == n) {
                return j;
            }
        }
    }
    return -1;
}

int ggml_sycl_qsa_mask_absorbs(const ggml_cgraph * cgraph, int node_idx) {
    return qsa_mask_reader_of(cgraph, node_idx) >= 0 ? 1 : 0;
}

// The cell list of a TOP_K_MASK, one row per row of tk: slot i*n_blk + k is cell k of pick i (-1 for
// a pick that is out of range or dead), then the extra cells. Out-of-range cells select nothing.
static void qsa_top_k_list(ggml_backend_sycl_context & ctx, const ggml_tensor * tk, int32_t * list, int64_t width) {
    const ggml_tensor * top_k  = tk->src[1];
    const ggml_tensor * score  = tk->src[2];
    const ggml_tensor * blocks = tk->src[3];
    const ggml_tensor * extra  = tk->src[4];

    const int64_t n_kv   = tk->src[0]->ne[0];
    const int64_t ne1    = tk->ne[1];
    const int64_t ne2    = tk->ne[2];
    const int64_t n_rows = ggml_nrows(tk);
    const int64_t n_top  = top_k->ne[0];
    const int64_t n_blk  = blocks ? blocks->ne[0] : 1;
    const int64_t n_pick = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    const int64_t n_cell = n_top * n_blk;

    const size_t sk0 = top_k->nb[0], sk1 = top_k->nb[1], sk2 = top_k->nb[2], sk3 = top_k->nb[3];
    const size_t ss0 = score ? score->nb[0] : 0, ss1 = score ? score->nb[1] : 0;
    const size_t ss2 = score ? score->nb[2] : 0, ss3 = score ? score->nb[3] : 0;
    const size_t sb0 = blocks ? blocks->nb[0] : 0, sb1 = blocks ? blocks->nb[1] : 0;
    const size_t se0 = extra ? extra->nb[0] : 0, se1 = extra ? extra->nb[1] : 0;
    const size_t se2 = extra ? extra->nb[2] : 0, se3 = extra ? extra->nb[3] : 0;

    // after the tiled indexer the score tensor holds its staging; the picks' scores are kept apart
    const float * picked   = score ? ggml_sycl_indexer_picked(ctx, top_k) : nullptr;
    const char *  score_d  = score && !picked ? (const char *) score->data : nullptr;
    const char *  top_k_d  = (const char *) top_k->data;
    const char *  blocks_d = blocks ? (const char *) blocks->data : nullptr;
    const char *  extra_d  = extra ? (const char *) extra->data : nullptr;

    constexpr int WG = 256;
    ctx.stream()->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(n_rows, ((width + WG - 1) / WG) * WG), sycl::range<2>(1, WG)),
        [=](sycl::nd_item<2> it) {
            const int64_t w = (int64_t) it.get_global_id(1);
            if (w >= width) {
                return;
            }
            const int64_t r  = (int64_t) it.get_global_id(0);
            const int64_t i1 = r % ne1;
            const int64_t i2 = (r / ne1) % ne2;
            const int64_t i3 = r / (ne1 * ne2);
            int32_t       c  = -1;
            if (w < n_cell) {
                const int64_t i = w / n_blk;
                const int64_t k = w - i * n_blk;
                const int32_t p = *(const int32_t *) (top_k_d + i * sk0 + i1 * sk1 + i2 * sk2 + i3 * sk3);
                bool live = p >= 0 && p < n_pick;
                if (live && picked) {
                    live = picked[r * n_top + i] > -INFINITY;
                } else if (live && score_d) {
                    live = *(const float *) (score_d + p * ss0 + i1 * ss1 + i2 * ss2 + i3 * ss3) > -INFINITY;
                }
                if (live) {
                    c = blocks_d ? *(const int32_t *) (blocks_d + k * sb0 + p * sb1) : p;
                }
            } else {
                c = *(const int32_t *) (extra_d + (w - n_cell) * se0 + i1 * se1 + i2 * se2 + i3 * se3);
            }
            list[r * width + w] = c;
        });
}

// A mask that hides nothing, at the flash attention mask's shape: packed (one bit per cell) when the
// backend packs masks, else dense f16 zeros
static const ggml_tensor * qsa_visible_mask(ggml_backend_sycl_context & ctx, const ggml_tensor * fm,
                                            ggml_sycl_pool_alloc<char> & buf, ggml_tensor & m,
                                            ggml_tensor_extra_gpu & extra) {
    m           = *fm;
    m.op        = GGML_OP_NONE;
    m.view_src  = nullptr;
    m.view_offs = 0;
    m.buffer    = nullptr;
    for (auto & s : m.src) {
        s = nullptr;
    }
    extra                                   = {};
    extra.optimized_feature.layout.kind     = GGML_SYCL_LAYOUT_MASK_BITS;
    extra.optimized_feature.layout.type     = GGML_TYPE_F16;
    extra.optimized_feature.layout.span     = 32;
    m.extra                                 = &extra;
    const int64_t n_rows = ggml_nrows(fm);
    if (ggml_sycl_kq_mask_is_bits(&m)) {
        const size_t bytes = ggml_sycl_kq_mask_row_bytes(fm->ne[0]) * n_rows;
        m.data             = buf.alloc(ctx.pool(), bytes);
        SYCL_CHECK(CHECK_TRY_ERROR(ctx.stream()->memset(m.data, 0xFF, bytes)));
    } else {
        // the packed form is switched off: +0 is all zero bits in f16
        m.extra            = nullptr;
        const size_t bytes = ggml_nbytes(fm);
        m.data             = buf.alloc(ctx.pool(), bytes);
        SYCL_CHECK(CHECK_TRY_ERROR(ctx.stream()->memset(m.data, 0, bytes)));
    }
    return &m;
}

bool ggml_sycl_qsa_fa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_fa_mask) {
        return false;
    }
    qsa_mask_chain c;
    if (!qsa_mask_chain_from_fa(cgraph, cgraph->nodes[i], c)) {
        return false;
    }
    // the dense mask was never built only if its ADD was absorbed, i.e. found from here
    int i_add = -1;
    for (int j = i - 1; j >= 0 && j >= i - SYCL_QSA_MASK_WINDOW; --j) {
        if (cgraph->nodes[j] == c.add) {
            i_add = j;
            break;
        }
    }
    if (i_add < 0 || qsa_mask_reader_of(cgraph, i_add) != i) {
        return false;
    }

    if (ggml_sycl_kq_mask_is_bits(c.mask)) {
        ggml_sycl_kq_mask_taught(ctx, c.add);
    }

    // scratch for the list and the mask, released after flash attention on the same in-order queue
    ggml_sycl_pool_alloc<int32_t> list(ctx.pool());
    if (c.gen) {
        c.idx.data = list.alloc((size_t) c.idx.ne[0] * c.idx.ne[1] * c.idx.ne[2]);
        qsa_top_k_list(ctx, c.gen, (int32_t *) c.idx.data, c.idx.ne[0]);
    }
    ggml_sycl_pool_alloc<char> visible(ctx.pool());
    ggml_tensor                visible_t;
    ggml_tensor_extra_gpu      visible_x;
    const ggml_tensor *        mask = c.mask;
    if (!mask) {
        mask = qsa_visible_mask(ctx, cgraph->nodes[i]->src[3], visible, visible_t, visible_x);
    }
    ggml_sycl_qsa_sparse_fa(ctx, cgraph->nodes[i], mask, &c.idx);
    return true;
}
