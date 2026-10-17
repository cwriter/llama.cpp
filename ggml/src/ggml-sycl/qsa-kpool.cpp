#include "qsa-kpool.hpp"
#include "fusion.hpp"
#include "kv-soa.hpp"
#include "rope-pair.hpp"

#include "ggml-impl.h"

#include <cstring>

// The qwen4exp QSA indexer keeps one q8_0 cache row per cell, raw key | pooled key, and every
// ubatch runs two short chains on it (build_indexer), one launch per node, on 128-float keys:
//
//   write: SET_ROWS(cache, VIEW(RESHAPE(CONCAT(k_raw, FILL(0)))), k_idxs)
//   pool:  rows   = RESHAPE(GET_ROWS(cache raw halves, new_pool_idxs))       [D, kpool, n_new]
//          pooled = ADD(...ADD(CONT(VIEW(rows, 0)), VIEW(rows, 1))..., VIEW(rows, kpool-1))
//          pooled = ROPE_MULTI(RESHAPE(MUL(RMS_NORM(SCALE(pooled)), w)), new_pool_pos)
//          SET_ROWS(cache pooled halves, RESHAPE(pooled), new_pool_rep)
//
// The write becomes one kernel at its SET_ROWS (3 launches -> 1), the pool one work-group per pooled
// key at the second SET_ROWS (8 launches -> 1). The other nodes of each chain are reported through
// fusion_absorbs: ggml-alloc reserves nothing for them and keeps their inputs live until the SET_ROWS.
// Every test below is structural, so the answer is the same before and after allocation.

static constexpr int SYCL_QSA_KPOOL_WINDOW   = 40;
static constexpr int SYCL_QSA_KPOOL_MAX_POOL = 8;
static constexpr int SYCL_QSA_KPOOL_NODES    = 2 * SYCL_QSA_KPOOL_MAX_POOL + 10;

struct qsa_kpool_chain {
    bool                pool;  // the pool chain, else the write chain
    const ggml_tensor * nodes[SYCL_QSA_KPOOL_NODES];
    int                 at[SYCL_QSA_KPOOL_NODES];
    int                 n_nodes;

    // write: rows of x (ne[0] wide), then the fill value up to the row width
    const ggml_tensor * x;
    float               fill;

    // pool
    const ggml_tensor * gather;  // the GET_ROWS
    const ggml_tensor * mul;     // norm * w
    const ggml_tensor * rope;
    int                 kpool;
};

static int32_t qsa_kpool_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

static bool qsa_kpool_private(const ggml_cgraph * cgraph, const ggml_tensor * t, int32_t uses) {
    return (t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) == 0 &&
           (t->flags & GGML_TENSOR_FLAG_COMPUTE) && qsa_kpool_use_count(cgraph, t) == uses;
}

static bool qsa_kpool_f32(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

static void qsa_kpool_add(qsa_kpool_chain & c, const ggml_tensor * t) {
    GGML_ASSERT(c.n_nodes < SYCL_QSA_KPOOL_NODES);
    c.nodes[c.n_nodes++] = t;
}

// the destination of a fused SET_ROWS: plain q8_0 rows of whole blocks, one index per source row
static bool qsa_kpool_dst_ok(const ggml_tensor * sr, int64_t width, int64_t n_rows) {
    const ggml_tensor * idx = sr->src[1];
    if (sr->type != GGML_TYPE_Q8_0 || ggml_sycl_kv_is_soa(sr) || (sr->view_src && ggml_sycl_kv_is_soa(sr->view_src))) {
        return false;
    }
    if (sr->ne[0] != width || width % QK8_0 != 0 || sr->nb[0] != sizeof(block_q8_0) || sr->ne[2] != 1 || sr->ne[3] != 1) {
        return false;
    }
    if (!idx || (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32) || !ggml_is_contiguous(idx) ||
        ggml_nelements(idx) != n_rows) {
        return false;
    }
    return true;
}

// the write chain from its SET_ROWS
static bool qsa_kpool_match_write(const ggml_cgraph * cgraph, int i, qsa_kpool_chain & c) {
    const ggml_tensor * sr = cgraph->nodes[i];
    const ggml_tensor * vw = sr->src[0];
    if (!vw || vw->op != GGML_OP_VIEW || vw->view_offs != 0 || !qsa_kpool_f32(vw)) {
        return false;
    }
    const ggml_tensor * rs = vw->src[0];
    if (!rs || rs->op != GGML_OP_RESHAPE || !qsa_kpool_f32(rs)) {
        return false;
    }
    const ggml_tensor * cc = rs->src[0];
    if (!cc || cc->op != GGML_OP_CONCAT || !qsa_kpool_f32(cc) || ggml_get_op_params_i32(cc, 0) != 0) {
        return false;
    }
    const ggml_tensor * x  = cc->src[0];
    const ggml_tensor * fl = cc->src[1];
    if (!qsa_kpool_f32(x) || !fl || fl->op != GGML_OP_FILL || !qsa_kpool_f32(fl) || !ggml_are_same_shape(x, fl)) {
        return false;
    }
    // [width, n_rows] all the way: the view and reshape only regroup whole rows
    const int64_t width  = cc->ne[0];
    const int64_t n_rows = ggml_nrows(cc);
    if (x->ne[0] >= width || vw->ne[0] != width || ggml_nrows(vw) != n_rows || !qsa_kpool_dst_ok(sr, width, n_rows)) {
        return false;
    }
    if (n_rows > INT32_MAX / width) {
        return false;
    }

    c.pool    = false;
    c.n_nodes = 0;
    for (const ggml_tensor * t : { fl, cc, rs, vw }) {
        if (!qsa_kpool_private(cgraph, t, 1)) {
            return false;
        }
        qsa_kpool_add(c, t);
    }
    c.x    = x;
    c.fill = ggml_get_op_params_f32(fl, 0);
    return true;
}

// the pool chain from its SET_ROWS
static bool qsa_kpool_match_pool(const ggml_cgraph * cgraph, int i, qsa_kpool_chain & c) {
    const ggml_tensor * sr = cgraph->nodes[i];
    const ggml_tensor * pn = sr->src[0];
    if (!pn || pn->op != GGML_OP_RESHAPE || !qsa_kpool_f32(pn)) {
        return false;
    }
    const ggml_tensor * ro = pn->src[0];
    if (!ro || ro->op != GGML_OP_ROPE || !qsa_kpool_f32(ro)) {
        return false;
    }
    // a NEOX or multi-section rope over [D, 1, n_new]: positions are 1 or 4 per pooled key
    if (!ggml_sycl_rope_pair_supported(ro) || ro->ne[1] != 1 || ro->ne[3] != 1) {
        return false;
    }
    const int           n_pos_per = (ggml_get_op_params_i32(ro, 2) & GGML_ROPE_TYPE_MROPE) ? 4 : 1;
    const ggml_tensor * pos       = ro->src[1];
    if (!pos || pos->type != GGML_TYPE_I32 || !ggml_is_contiguous(pos) || ggml_nelements(pos) != n_pos_per * ro->ne[2]) {
        return false;
    }
    if (ro->src[2] && !qsa_kpool_f32(ro->src[2])) {
        return false;
    }
    const ggml_tensor * r3 = ro->src[0];
    if (!r3 || r3->op != GGML_OP_RESHAPE || !qsa_kpool_f32(r3)) {
        return false;
    }
    const ggml_tensor * mul = r3->src[0];
    if (!mul || mul->op != GGML_OP_MUL || !qsa_kpool_f32(mul)) {
        return false;
    }
    const ggml_tensor * rn = mul->src[0];
    const ggml_tensor * w  = mul->src[1];
    if (!rn || rn->op != GGML_OP_RMS_NORM || !qsa_kpool_f32(rn) || !qsa_kpool_f32(w) ||
        ggml_nelements(w) != rn->ne[0] || !ggml_are_same_shape(mul, rn)) {
        return false;
    }
    const ggml_tensor * sc = rn->src[0];
    if (!sc || sc->op != GGML_OP_SCALE || !qsa_kpool_f32(sc) || !ggml_are_same_shape(sc, rn)) {
        return false;
    }

    const int64_t D     = sc->ne[0];
    const int64_t n_new = ggml_nrows(sc);
    if (ro->ne[0] != D || ro->ne[2] != n_new || pn->ne[0] != D || ggml_nrows(pn) != n_new || D > 1024 ||
        !qsa_kpool_dst_ok(sr, D, n_new)) {
        return false;
    }

    // the sum: ADD(...ADD(CONT(v0), v1)..., v[k-1]), every v a row-slice view of one RESHAPE(GET_ROWS)
    const ggml_tensor * slices[SYCL_QSA_KPOOL_MAX_POOL];
    const ggml_tensor * adds[SYCL_QSA_KPOOL_MAX_POOL];
    int                 n_add = 0;
    const ggml_tensor * t     = sc->src[0];
    while (t && t->op == GGML_OP_ADD) {
        if (n_add + 1 >= SYCL_QSA_KPOOL_MAX_POOL || !qsa_kpool_f32(t) || !ggml_are_same_shape(t, sc)) {
            return false;
        }
        adds[n_add]     = t;
        slices[++n_add] = t->src[1];
        t               = t->src[0];
    }
    const ggml_tensor * ct = t;
    if (!ct || ct->op != GGML_OP_CONT || !qsa_kpool_f32(ct) || !ggml_are_same_shape(ct, sc) || n_add < 1) {
        return false;
    }
    const int kpool = n_add + 1;
    // adds[0] is the last add; slices[1 + k] was read by adds[k], so slice m (m >= 1) sits at slices[kpool - m]
    const ggml_tensor * views[SYCL_QSA_KPOOL_MAX_POOL];
    views[0] = ct->src[0];
    for (int m = 1; m < kpool; ++m) {
        views[m] = slices[kpool - m];
    }
    const ggml_tensor * rows = views[0] ? views[0]->src[0] : nullptr;
    if (!rows || rows->op != GGML_OP_RESHAPE || rows->type != GGML_TYPE_F32 || !ggml_is_contiguous(rows) ||
        rows->ne[0] != D || rows->ne[1] != kpool || rows->ne[2] != n_new || rows->ne[3] != 1) {
        return false;
    }
    for (int m = 0; m < kpool; ++m) {
        const ggml_tensor * v = views[m];
        if (!v || v->op != GGML_OP_VIEW || v->src[0] != rows || v->type != GGML_TYPE_F32 || v->ne[0] != D ||
            v->ne[1] != n_new || v->ne[2] != 1 || v->ne[3] != 1 || v->nb[0] != sizeof(float) ||
            v->nb[1] != rows->nb[2] || v->view_offs != rows->view_offs + (size_t) m * rows->nb[1]) {
            return false;
        }
    }
    const ggml_tensor * gr = rows->src[0];
    if (!gr || gr->op != GGML_OP_GET_ROWS || !qsa_kpool_f32(gr) || gr->ne[0] != D || gr->ne[1] != kpool * n_new ||
        ggml_nrows(gr) != kpool * n_new) {
        return false;
    }
    const ggml_tensor * kg  = gr->src[0];
    const ggml_tensor * idx = gr->src[1];
    if (!kg || kg->type != GGML_TYPE_Q8_0 || kg->ne[0] != D || kg->nb[0] != sizeof(block_q8_0) || kg->ne[2] != 1 ||
        kg->ne[3] != 1 || ggml_sycl_kv_is_soa(kg) || (kg->view_src && ggml_sycl_kv_is_soa(kg->view_src))) {
        return false;
    }
    if (!idx || idx->type != GGML_TYPE_I32 || !ggml_is_contiguous(idx) || ggml_nelements(idx) != kpool * n_new) {
        return false;
    }

    c.pool    = true;
    c.n_nodes = 0;
    if (!qsa_kpool_private(cgraph, gr, 1) || !qsa_kpool_private(cgraph, rows, kpool)) {
        return false;
    }
    qsa_kpool_add(c, gr);
    qsa_kpool_add(c, rows);
    for (int m = 0; m < kpool; ++m) {
        if (!qsa_kpool_private(cgraph, views[m], 1)) {
            return false;
        }
        qsa_kpool_add(c, views[m]);
    }
    if (!qsa_kpool_private(cgraph, ct, 1)) {
        return false;
    }
    qsa_kpool_add(c, ct);
    for (int k = 0; k < n_add; ++k) {
        if (!qsa_kpool_private(cgraph, adds[k], 1)) {
            return false;
        }
        qsa_kpool_add(c, adds[k]);
    }
    for (const ggml_tensor * n : { sc, rn, mul, r3, ro, pn }) {
        if (!qsa_kpool_private(cgraph, n, 1)) {
            return false;
        }
        qsa_kpool_add(c, n);
    }
    c.gather = gr;
    c.mul    = mul;
    c.rope   = ro;
    c.kpool  = kpool;
    return true;
}

// both chains: the structure, then every chain node in the window before the SET_ROWS, and nothing
// in between writing what the kernel now reads later than the graph did
static bool qsa_kpool_match(const ggml_cgraph * cgraph, int i, qsa_kpool_chain & c) {
    const ggml_tensor * sr = cgraph->nodes[i];
    if (sr->op != GGML_OP_SET_ROWS || sr->type != GGML_TYPE_Q8_0 || !sr->src[0]) {
        return false;
    }
    if (!(sr->src[0]->op == GGML_OP_VIEW ? qsa_kpool_match_write(cgraph, i, c) : qsa_kpool_match_pool(cgraph, i, c))) {
        return false;
    }
    const int lo = std::max(0, i - SYCL_QSA_KPOOL_WINDOW);
    for (int k = 0; k < c.n_nodes; ++k) {
        c.at[k] = -1;
        for (int j = i - 1; j >= lo; --j) {
            if (cgraph->nodes[j] == c.nodes[k]) {
                c.at[k] = j;
                break;
            }
        }
        if (c.at[k] < 0) {
            return false;
        }
    }
    if (!c.pool) {
        // x was read by the CONCAT (nodes[1])
        return !ggml_sycl_written_between(cgraph, c.x, c.at[1], i);
    }
    // the cache rows were read by the GET_ROWS (nodes[0]); the rest were read further down
    const ggml_tensor * gr = c.gather;
    const ggml_tensor * ro = c.rope;
    const int           at_mul  = c.at[c.n_nodes - 4];
    const int           at_rope = c.at[c.n_nodes - 2];
    return !ggml_sycl_written_between(cgraph, gr->src[0], c.at[0], i) &&
           !ggml_sycl_written_between(cgraph, gr->src[1], c.at[0], i) &&
           !ggml_sycl_written_between(cgraph, c.mul->src[1], at_mul, i) &&
           !ggml_sycl_written_between(cgraph, ro->src[1], at_rope, i) &&
           !(ro->src[2] && ggml_sycl_written_between(cgraph, ro->src[2], at_rope, i));
}

// the SET_ROWS whose chain holds node i, or -1; shared by fusion_absorbs and the compute loop
static int qsa_kpool_reader_of(const ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_kpool) {
        return -1;
    }
    const ggml_tensor * n = cgraph->nodes[i];
    switch (n->op) {
        case GGML_OP_FILL:
        case GGML_OP_CONCAT:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_GET_ROWS:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_SCALE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_MUL:
        case GGML_OP_ROPE:
            break;
        default:
            return -1;
    }
    if (n->type != GGML_TYPE_F32) {
        return -1;
    }
    const int end = std::min(cgraph->n_nodes, i + SYCL_QSA_KPOOL_WINDOW + 1);
    for (int j = i + 1; j < end; ++j) {
        const ggml_tensor * s = cgraph->nodes[j];
        if (s->op != GGML_OP_SET_ROWS || s->type != GGML_TYPE_Q8_0) {
            continue;
        }
        qsa_kpool_chain c;
        if (!qsa_kpool_match(cgraph, j, c)) {
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

int ggml_sycl_qsa_kpool_absorbs(const ggml_cgraph * cgraph, int node_idx) {
    return qsa_kpool_reader_of(cgraph, node_idx) >= 0 ? 1 : 0;
}

// one q8_0 block, rounded as cpy_blck_f32_q8_0 does it
static inline void qsa_kpool_store_block(const float * v, char * blk) {
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
        amax = sycl::fmax(amax, sycl::fabs(v[j]));
    }
    const float d  = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    *(sycl::half *) blk = (sycl::half) d;
    int8_t * qs = (int8_t *) (blk + sizeof(sycl::half));
    for (int j = 0; j < QK8_0; ++j) {
        qs[j] = (int8_t) sycl::round(v[j] * id);
    }
}

template <typename TIdx>
static void qsa_kpool_write(const qsa_kpool_chain & c, const ggml_tensor * sr, dpct::queue_ptr stream) {
    const float * x       = (const float *) c.x->data;
    const TIdx *  idx     = (const TIdx *) sr->src[1]->data;
    char *        dst     = (char *) sr->data;
    const int     width   = (int) sr->ne[0];
    const int     nx      = (int) c.x->ne[0];
    const int     n_rows  = (int) ggml_nelements(sr->src[1]);
    const size_t  sx      = c.x->nb[1] / sizeof(float);
    const size_t  row     = sr->nb[1];
    const float   fill    = c.fill;
    const int     n_blk   = width / QK8_0;
    const int     n       = n_rows * n_blk;
    constexpr int wg      = 64;

    // one work-item per output block
    stream->parallel_for(sycl::nd_range<1>(((n + wg - 1) / wg) * wg, wg), [=](sycl::nd_item<1> it) {
        const int i = (int) it.get_global_id(0);
        if (i >= n) {
            return;
        }
        const int r = i / n_blk;
        const int b = i % n_blk;
        float     v[QK8_0];
        for (int j = 0; j < QK8_0; ++j) {
            const int e = b * QK8_0 + j;
            v[j]        = e < nx ? x[r * sx + e] : fill;
        }
        qsa_kpool_store_block(v, dst + (size_t) idx[r] * row + (size_t) b * sizeof(block_q8_0));
    });
}

template <typename TIdx>
static void qsa_kpool_pool(const qsa_kpool_chain & c, const ggml_tensor * sr, dpct::queue_ptr stream) {
    const ggml_tensor * gr  = c.gather;
    const ggml_tensor * ro  = c.rope;
    const ggml_tensor * sc  = c.mul->src[0]->src[0];
    const ggml_tensor * rn  = c.mul->src[0];

    const char *    cache   = (const char *) gr->src[0]->data;
    const size_t    crow    = gr->src[0]->nb[1];
    const int32_t * gidx    = (const int32_t *) gr->src[1]->data;
    const float *   w       = (const float *) c.mul->src[1]->data;
    const int32_t * pos     = (const int32_t *) ro->src[1]->data;
    const float *   ff      = ro->src[2] ? (const float *) ro->src[2]->data : nullptr;
    const TIdx *    rep     = (const TIdx *) sr->src[1]->data;
    char *          dst     = (char *) sr->data;
    const size_t    drow    = sr->nb[1];
    const int       D       = (int) sc->ne[0];
    const int       n_new   = (int) ggml_nrows(sc);
    const int       kpool   = c.kpool;

    float s_scale, b_scale, eps;
    memcpy(&s_scale, (const float *) sc->op_params + 0, sizeof(float));
    memcpy(&b_scale, (const float *) sc->op_params + 1, sizeof(float));
    memcpy(&eps, rn->op_params, sizeof(float));

    const ggml_sycl_rope_pair_params p = ggml_sycl_rope_pair_init(ro);

    // one work-group per pooled key, one work-item per element
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> xs(sycl::range<1>(D), cgh);
        sycl::local_accessor<float, 1> ys(sycl::range<1>(D), cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) n_new * D, D), [=](sycl::nd_item<1> it) {
            const int t  = (int) it.get_local_id(0);
            const int ip = (int) it.get_group(0);
            const int b  = t / QK8_0;

            // the members' raw keys, summed in the graph's order, then the scale
            float acc = 0.0f;
            for (int m = 0; m < kpool; ++m) {
                const char * blk = cache + (size_t) gidx[ip * kpool + m] * crow + (size_t) b * sizeof(block_q8_0);
                const float  d   = (float) *(const sycl::half *) blk;
                const float  v   = ((const int8_t *) (blk + sizeof(sycl::half)))[t % QK8_0] * d;
                acc              = m == 0 ? v : acc + v;
            }
            const float x = acc * s_scale + b_scale;

            // rms norm * w
            const float sum   = sycl::reduce_over_group(it.get_group(), x * x, sycl::plus<float>());
            const float scale = sycl::rsqrt(sum / D + eps);
            xs[t]             = x * scale * w[t];
            sycl::group_barrier(it.get_group());

            // rope: the work-item of the first channel of a rotated pair writes both
            const int n_dims = p.n_dims;
            const int n_offs = p.n_offs;
            if (t < n_offs || t >= n_offs + n_dims) {
                ys[t] = xs[t];
            } else if (t < n_offs + n_dims / 2) {
                float cos_theta, sin_theta;
                ggml_sycl_rope_pair_cs(p, pos, n_new, ip, t - n_offs, ff, cos_theta, sin_theta);
                const float x0       = xs[t];
                const float x1       = xs[t + n_dims / 2];
                ys[t]                = x0 * cos_theta - x1 * sin_theta;
                ys[t + n_dims / 2]   = x0 * sin_theta + x1 * cos_theta;
            }
            sycl::group_barrier(it.get_group());

            if (t % QK8_0 == 0) {
                float v[QK8_0];
                for (int j = 0; j < QK8_0; ++j) {
                    v[j] = ys[t + j];
                }
                qsa_kpool_store_block(v, dst + (size_t) rep[ip] * drow + (size_t) b * sizeof(block_q8_0));
            }
        });
    });
}

bool ggml_sycl_qsa_kpool(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_kpool) {
        return false;
    }
    qsa_kpool_chain c;
    if (!qsa_kpool_match(cgraph, i, c)) {
        return false;
    }
    ggml_tensor * sr = cgraph->nodes[i];
    scope_op_debug_print scope_dbg_print(__func__, sr, 2, c.pool ? " (QSA key pool)" : " (QSA key write)");

    const bool i64 = sr->src[1]->type == GGML_TYPE_I64;
    if (c.pool) {
        i64 ? qsa_kpool_pool<int64_t>(c, sr, ctx.stream()) : qsa_kpool_pool<int32_t>(c, sr, ctx.stream());
    } else {
        i64 ? qsa_kpool_write<int64_t>(c, sr, ctx.stream()) : qsa_kpool_write<int32_t>(c, sr, ctx.stream());
    }
    return true;
}
