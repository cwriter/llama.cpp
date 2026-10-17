#include "norm-rope.hpp"
#include "rope-pair.hpp"
#include "kv-soa.hpp"

#include "ggml-impl.h"

// Attention layers norm q and k per head and rope them right after (qwen4exp: Q, K and the QSA
// indexer q): RMS_NORM -> MUL(w) -> ROPE, three launches on a few KB. One work-group per head row
// computes the norm in registers, keeps the scaled row in local memory, and writes the rotated row.

static bool ggml_sycl_rms_norm_rope_ok(const ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_norm_rope) {
        return false;
    }
    if (!ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE })) {
        return false;
    }
    const ggml_tensor * rn  = cgraph->nodes[i];
    const ggml_tensor * mul = cgraph->nodes[i + 1];
    const ggml_tensor * ro  = cgraph->nodes[i + 2];
    const ggml_tensor * x   = rn->src[0];
    const ggml_tensor * w   = mul->src[0] == rn ? mul->src[1] : mul->src[0];

    if (x->type != GGML_TYPE_F32 || rn->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        ro->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || x->ne[3] != 1) {
        return false;
    }
    // the weight is one row, broadcast over every head and token
    if (!w || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != x->ne[0] ||
        !ggml_are_same_shape(mul, rn)) {
        return false;
    }
    const int64_t D = x->ne[0];
    if (D > 1024 || D % 2 != 0 || ro->src[0] != mul || !ggml_is_contiguous(ro) || !ggml_are_same_shape(ro, mul) ||
        !ggml_sycl_rope_pair_supported(ro)) {
        return false;
    }
    const ggml_tensor * pos       = ro->src[1];
    const int           n_pos_per = (ggml_get_op_params_i32(ro, 2) & GGML_ROPE_TYPE_MROPE) ? 4 : 1;
    if (!pos || pos->type != GGML_TYPE_I32 || !ggml_is_contiguous(pos) || ggml_nelements(pos) != n_pos_per * ro->ne[2]) {
        return false;
    }
    if (ro->src[2] && (ro->src[2]->type != GGML_TYPE_F32 || !ggml_is_contiguous(ro->src[2]))) {
        return false;
    }
    return ggml_nrows(ro) <= INT32_MAX;
}

static int32_t norm_rope_uses(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

static bool norm_rope_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW;
}

// the first non-view node after `from` whose source (through single-use views) is t, or -1
static int norm_rope_next_reader(const ggml_cgraph * cgraph, int from, const ggml_tensor * t, const ggml_tensor ** src) {
    for (int j = from; j < cgraph->n_nodes && j < from + 8; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (norm_rope_is_view(n)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * v = n->src[s];
            const ggml_tensor * root = v;
            while (root && root != t && norm_rope_is_view(root) && norm_rope_uses(cgraph, root) == 1) {
                root = root->src[0];
            }
            if (v && root == t) {
                *src = v;
                return j;
            }
        }
        return -1;
    }
    return -1;
}

// After the rope (GGML_SYCL_FUSE_NORM_ROPE_ROT): the Hadamard rotation of every head row that ggml tags on a
// MUL_MAT (GGML_HINT_SRC0_IS_HADAMARD, see fwht.cpp), then optionally the q8_0 cache write of the rotated keys.
struct norm_rope_tail {
    ggml_tensor * rot  = nullptr;  // the MUL_MAT, written unless the cache write takes it
    ggml_tensor * sr   = nullptr;  // the SET_ROWS into a q8_0 cache
    int           last = -1;
};

static bool norm_rope_match_tail(const ggml_cgraph * cgraph, int i, norm_rope_tail & tl) {
    static const bool on = ggml_sycl_get_env("GGML_SYCL_FUSE_NORM_ROPE_ROT", 1) != 0;
    const ggml_tensor * ro = cgraph->nodes[i + 2];
    const int64_t       D  = ro->ne[0];
    if (!on || D < 64 || D > 1024 || (D & (D - 1)) != 0 || norm_rope_uses(cgraph, ro) != 1) {
        return false;
    }
    const ggml_tensor * v  = nullptr;
    const int           j  = norm_rope_next_reader(cgraph, i + 3, ro, &v);
    if (j < 0) {
        return false;
    }
    ggml_tensor * mm = cgraph->nodes[j];
    if (mm->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD || mm->src[1] != v ||
        mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || v->ne[0] != D || ggml_nrows(v) != ggml_nrows(ro) ||
        !ggml_are_same_shape(mm, v)) {
        return false;
    }
    tl.rot  = mm;
    tl.last = j;
    // the rotated rows written straight into the q8_0 cache, when that is all they are for
    if (norm_rope_uses(cgraph, mm) == 1 && D % QK8_0 == 0) {
        const ggml_tensor * sv = nullptr;
        const int           k  = norm_rope_next_reader(cgraph, j + 1, mm, &sv);
        ggml_tensor *       sr = k >= 0 ? cgraph->nodes[k] : nullptr;
        if (sr && sr->op == GGML_OP_SET_ROWS && sr->src[0] == sv && sr->type == GGML_TYPE_Q8_0 &&
            !ggml_sycl_kv_is_soa(sr) && !(sr->view_src && ggml_sycl_kv_is_soa(sr->view_src)) &&
            ggml_is_contiguous(sv) && sv->ne[0] == D * ro->ne[1] && ggml_nrows(sv) == ro->ne[2] &&
            sr->ne[0] == sv->ne[0] && sr->nb[0] == sizeof(block_q8_0) && sr->ne[2] == 1 && sr->ne[3] == 1 &&
            sr->src[1] && (sr->src[1]->type == GGML_TYPE_I64 || sr->src[1]->type == GGML_TYPE_I32) &&
            ggml_is_contiguous(sr->src[1]) && ggml_nelements(sr->src[1]) == ro->ne[2]) {
            tl.sr   = sr;
            tl.last = k;
        }
    }
    return true;
}

int ggml_sycl_rms_norm_rope_span(const ggml_cgraph * cgraph, int i, const ggml_tensor ** x, const ggml_tensor ** last) {
    if (!ggml_sycl_rms_norm_rope_ok(cgraph, i)) {
        return 0;
    }
    norm_rope_tail tl;
    const bool tail = norm_rope_match_tail(cgraph, i, tl);
    *x    = cgraph->nodes[i]->src[0];
    *last = tail ? (tl.sr ? tl.sr : tl.rot) : cgraph->nodes[i + 2];
    return tail ? tl.last - i + 1 : 3;
}

int ggml_sycl_rms_norm_rope(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!ggml_sycl_rms_norm_rope_ok(cgraph, i)) {
        return 0;
    }
    const ggml_tensor * rn  = cgraph->nodes[i];
    const ggml_tensor * mul = cgraph->nodes[i + 1];
    ggml_tensor *       ro  = cgraph->nodes[i + 2];
    const ggml_tensor * x   = rn->src[0];
    const ggml_tensor * w_t = mul->src[0] == rn ? mul->src[1] : mul->src[0];
    norm_rope_tail      tl;
    const bool          tail = norm_rope_match_tail(cgraph, i, tl);
    const int           mode = !tail ? 0 : (tl.sr ? 2 : 1);  // 0 rope, 1 + rotation, 2 + q8_0 cache write
    scope_op_debug_print scope_dbg_print(__func__, mode == 2 ? tl.sr : mode == 1 ? tl.rot : ro, 3,
                                         mode == 2 ? " (rms_norm * w -> rope -> hadamard -> q8_0 cache)" :
                                         mode == 1 ? " (rms_norm * w -> rope -> hadamard)" : " (rms_norm * w -> rope)");

    float eps;
    memcpy(&eps, rn->op_params, sizeof(float));
    const ggml_sycl_rope_pair_params p = ggml_sycl_rope_pair_init(ro);

    const float *   xd  = (const float *) x->data;
    const float *   w   = (const float *) w_t->data;
    const int32_t * pos = (const int32_t *) ro->src[1]->data;
    const float *   ff  = ro->src[2] ? (const float *) ro->src[2]->data : nullptr;

    const int    D   = (int) x->ne[0];
    const int    ne1 = (int) x->ne[1];
    const int    ne2 = (int) x->ne[2];
    const size_t sx1 = x->nb[1] / sizeof(float);
    const size_t sx2 = x->nb[2] / sizeof(float);
    // mode 0 writes the rope output, mode 1 the rotation's rows (one per head and token, contiguous)
    float *      dst = mode == 0 ? (float *) ro->data : mode == 1 ? (float *) tl.rot->data : nullptr;
    const size_t sd1 = mode == 0 ? ro->nb[1] / sizeof(float) : (size_t) D;
    const size_t sd2 = mode == 0 ? ro->nb[2] / sizeof(float) : (size_t) D * ne1;
    // mode 2: cache row idx[token], head i1 at blocks [i1 * D / 32, (i1 + 1) * D / 32)
    char *       cache = mode == 2 ? (char *) tl.sr->data : nullptr;
    const size_t crow  = mode == 2 ? tl.sr->nb[1] : 0;
    const void * cidx  = mode == 2 ? tl.sr->src[1]->data : nullptr;
    const bool   i64   = mode == 2 && tl.sr->src[1]->type == GGML_TYPE_I64;
    const float  hs    = 1.0f / sycl::sqrt((float) D);  // fwht.cpp scales by 1/sqrt(n) before the butterflies
    const int    n_rows = ne1 * ne2;

    ctx.stream()->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> ys(sycl::range<1>(D), cgh);
        sycl::local_accessor<float, 1> zs(sycl::range<1>(D), cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) n_rows * D, D), [=](sycl::nd_item<1> it) {
            const int t  = (int) it.get_local_id(0);
            const int r  = (int) it.get_group(0);
            const int i1 = r % ne1;
            const int i2 = r / ne1;

            const float v     = xd[i1 * sx1 + i2 * sx2 + t];
            const float sum   = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
            const float scale = sycl::rsqrt(sum / D + eps);
            ys[t]             = v * scale * w[t];
            sycl::group_barrier(it.get_group());

            // the rope, into dst (mode 0) or into zs for the rotation
            float *   o      = mode == 0 ? dst + i1 * sd1 + i2 * sd2 : &zs[0];
            const int n_dims = p.n_dims;
            const int n_offs = p.n_offs;
            if (t < n_offs || t >= n_offs + n_dims) {
                o[t] = ys[t];
            } else if (t < n_offs + n_dims / 2) {
                float cos_theta, sin_theta;
                ggml_sycl_rope_pair_cs(p, pos, ne2, i2, t - n_offs, ff, cos_theta, sin_theta);
                const float x0    = ys[t];
                const float x1    = ys[t + n_dims / 2];
                o[t]              = x0 * cos_theta - x1 * sin_theta;
                o[t + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            }
            if (mode == 0) {
                return;
            }
            sycl::group_barrier(it.get_group());

            // the Walsh-Hadamard butterflies of fwht.cpp, in its order: scale, then h = 1, 2, ..., D/2
            float h_v = zs[t] * hs;
            for (int h = 1; h < D; h <<= 1) {
                sycl::group_barrier(it.get_group());
                zs[t] = h_v;
                sycl::group_barrier(it.get_group());
                const float q = zs[t ^ h];
                h_v           = (t & h) == 0 ? h_v + q : q - h_v;
            }
            if (mode == 1) {
                dst[i1 * sd1 + i2 * sd2 + t] = h_v;
                return;
            }
            sycl::group_barrier(it.get_group());
            zs[t] = h_v;
            sycl::group_barrier(it.get_group());
            if (t % QK8_0 == 0) {
                // one q8_0 block, rounded as cpy_blck_f32_q8_0 does it
                float amax = 0.0f;
                for (int j = 0; j < QK8_0; ++j) {
                    amax = sycl::fmax(amax, sycl::fabs(zs[t + j]));
                }
                const float d  = amax / ((1 << 7) - 1);
                const float id = d ? 1.0f / d : 0.0f;
                const int64_t row = i64 ? ((const int64_t *) cidx)[i2] : ((const int32_t *) cidx)[i2];
                char * blk = cache + (size_t) row * crow + (size_t) ((i1 * D + t) / QK8_0) * sizeof(block_q8_0);
                *(sycl::half *) blk = (sycl::half) d;
                int8_t * qs = (int8_t *) (blk + sizeof(sycl::half));
                for (int j = 0; j < QK8_0; ++j) {
                    qs[j] = (int8_t) sycl::round(zs[t + j] * id);
                }
            }
        });
    });
    return mode == 0 ? 3 : tl.last - i + 1;
}
