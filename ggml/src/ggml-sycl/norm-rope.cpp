#include "norm-rope.hpp"
#include "rope-pair.hpp"

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

int ggml_sycl_rms_norm_rope(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!ggml_sycl_rms_norm_rope_ok(cgraph, i)) {
        return 0;
    }
    const ggml_tensor * rn  = cgraph->nodes[i];
    const ggml_tensor * mul = cgraph->nodes[i + 1];
    ggml_tensor *       ro  = cgraph->nodes[i + 2];
    const ggml_tensor * x   = rn->src[0];
    const ggml_tensor * w_t = mul->src[0] == rn ? mul->src[1] : mul->src[0];
    scope_op_debug_print scope_dbg_print(__func__, ro, 3, " (rms_norm * w -> rope)");

    float eps;
    memcpy(&eps, rn->op_params, sizeof(float));
    const ggml_sycl_rope_pair_params p = ggml_sycl_rope_pair_init(ro);

    const float *   xd  = (const float *) x->data;
    const float *   w   = (const float *) w_t->data;
    const int32_t * pos = (const int32_t *) ro->src[1]->data;
    const float *   ff  = ro->src[2] ? (const float *) ro->src[2]->data : nullptr;
    float *         dst = (float *) ro->data;

    const int    D   = (int) x->ne[0];
    const int    ne1 = (int) x->ne[1];
    const int    ne2 = (int) x->ne[2];
    const size_t sx1 = x->nb[1] / sizeof(float);
    const size_t sx2 = x->nb[2] / sizeof(float);
    const size_t sd1 = ro->nb[1] / sizeof(float);
    const size_t sd2 = ro->nb[2] / sizeof(float);
    const int    n_rows = ne1 * ne2;

    ctx.stream()->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> ys(sycl::range<1>(D), cgh);
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

            float *   o      = dst + i1 * sd1 + i2 * sd2;
            const int n_dims = p.n_dims;
            const int n_offs = p.n_offs;
            if (t < n_offs || t >= n_offs + n_dims) {
                o[t] = ys[t];
            } else if (t < n_offs + n_dims / 2) {
                float cos_theta, sin_theta;
                ggml_sycl_rope_pair_cs(p, pos, ne2, i2, t - n_offs, ff, cos_theta, sin_theta);
                const float x0   = ys[t];
                const float x1   = ys[t + n_dims / 2];
                o[t]             = x0 * cos_theta - x1 * sin_theta;
                o[t + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            }
        });
    });
    return 3;
}
