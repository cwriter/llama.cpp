#include "qsa-mask.hpp"
#include "kq-mask-bits.hpp"

#include "ggml-impl.h"

#include "common.hpp"
#include "fattn-qsa.hpp"

#include <cmath>

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

// the chain sits right before its flash attention node; a chain node further away is left to run
static constexpr int SYCL_QSA_MASK_WINDOW = 48;
static constexpr int SYCL_QSA_MASK_MAX    = 16;

struct qsa_mask_chain {
    const ggml_tensor * mask;  // the causal kq_mask
    ggml_tensor         idx;   // the selection list seen as [width, n_tps, n_stream]
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

// a REPEAT of a FILL with this value, possibly seen through a view; returns the REPEAT
static const ggml_tensor * qsa_filled(const ggml_tensor * t, float value) {
    const ggml_tensor * rp = t->view_src ? t->view_src : t;
    if (rp->op != GGML_OP_REPEAT || !rp->src[0] || rp->src[0]->op != GGML_OP_FILL) {
        return nullptr;
    }
    const float v = ggml_get_op_params_f32(rp->src[0], 0);
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
    c.n_nodes = 0;

    c.idx       = *iv;
    c.idx.ne[1] = n_tps;
    c.idx.ne[2] = n_stream;
    c.idx.nb[2] = iv->nb[1] * n_tps;
    c.idx.nb[3] = c.idx.nb[2] * n_stream;

    // the nodes that only build the dense mask
    const ggml_tensor * fill_inf  = rp_inf->src[0];
    const ggml_tensor * fill_zero = rp_zero->src[0];
    for (const ggml_tensor * t : { ad, rv, sr, mv, rp_inf, fill_inf, fz, rp_zero, fill_zero }) {
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
    ggml_sycl_qsa_sparse_fa(ctx, cgraph->nodes[i], c.mask, &c.idx);
    return true;
}
