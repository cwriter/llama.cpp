#include "ggml-impl.h"
#include "dsv4-hc.hpp"
#include "element_wise.hpp"  // op_exp, for sigmoid parity with the unfused chain

#include <cmath>
#include <type_traits>

static constexpr int DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
static constexpr int dsv4_hc_pre_block_size = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <bool gated>
static void dsv4_hc_pre_f32_sycl(
        const float * x, const float * weights, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        float scale,
        queue_ptr stream) {
    const int64_t nr = n_embd * n_tokens;
    const int64_t num_blocks = (nr + dsv4_hc_pre_block_size - 1) / dsv4_hc_pre_block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * dsv4_hc_pre_block_size),
                          sycl::range<1>(dsv4_hc_pre_block_size)),
        [=](sycl::nd_item<1> item) {
            const int64_t ir = item.get_global_id(0);
            if (ir >= nr) {
                return;
            }

            const int64_t i0 = ir % n_embd;
            const int64_t it = ir / n_embd;

            float sum = 0.0f;
            for (int64_t ih = 0; ih < hc; ++ih) {
                const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
                float wv;
                if constexpr (gated) {
                    const float gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } else {
                    wv = weights[ih*sw0 + it*sw1];
                }
                sum += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * sum;
        });
}

static void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static void dsv4_hc_comb_f32_sycl(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr int comb_offset = 2*DSV4_HC;

    const int64_t block_size = 256;
    const int64_t num_blocks = (n_tokens + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item_ct1) {
            const int64_t it = item_ct1.get_global_id(0);

            if (it >= n_tokens) {
                return;
            }

            const float scale_comb = scale[2*ss0];
            float comb[DSV4_HC*DSV4_HC];

            for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
                float max = -INFINITY;
                for (int idst = 0; idst < DSV4_HC; ++idst) {
                    const int idx = idst + DSV4_HC*isrc;
                    const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                float sum = 0.0f;
                for (int idst = 0; idst < DSV4_HC; ++idst) {
                    const int idx = idst + DSV4_HC*isrc;
                    const float v = expf(comb[idx] - max);
                    comb[idx] = v;
                    sum += v;
                }

                const float inv_sum = 1.0f / sum;
                for (int idst = 0; idst < DSV4_HC; ++idst) {
                    const int idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_sum + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            for (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
                for (int idst = 0; idst < DSV4_HC; ++idst) {
                    const int idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
static constexpr int dsv4_hc_post_block_size = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
// fused_gate: post points at the input of a scale -> sigmoid -> scale chain rather than its
// result, and the three ops are applied here. g0/g1 carry (scale, bias) for the two scales.
template <bool has_comb, bool fused_gate>
static void dsv4_hc_post_f32_sycl(
        const float * x, const float * residual, const float * post, const float * comb, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        sycl::float2 g0, sycl::float2 g1,
        queue_ptr stream) {
    const int64_t nr = n_embd * hc * n_tokens;
    const int64_t block_size = dsv4_hc_post_block_size;
    const int64_t num_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            const int64_t ir = item.get_global_id(0);
            if (ir >= nr) {
                return;
            }

            const int64_t i0   = ir % n_embd;
            const int64_t idst = (ir / n_embd) % hc;
            const int64_t it   = ir / (n_embd * hc);

            float gate = post[idst*sp0 + it*sp1];
            if constexpr (fused_gate) {
                gate = g0[0] * gate + g0[1];
                gate = 1.0f / (1.0f + op_exp(-gate));  // exactly op_sigmoid()
                gate = g1[0] * gate + g1[1];
            }

            float sum = x[i0*sx0 + it*sx1] * gate;
            if constexpr (has_comb) {
                for (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } else {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        });
}

// dsv4_hc_post_f32_sycl() for rows that are contiguous and 16 B aligned: a work-item takes 4
// consecutive i0, so x and residual are read and dst written as one float4 each, and the 3D range
// replaces the three 64-bit divisions per element. Every element gets the same operations in the
// same order, so the result is bit for bit the same.
template <bool has_comb, bool fused_gate>
static void dsv4_hc_post_f32_vec4_sycl(
        const float * x, const float * residual, const float * post, const float * comb, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens, int wg,
        int64_t sx1,
        int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd1, int64_t sd2,
        sycl::float2 g0, sycl::float2 g1,
        queue_ptr stream) {
    const int64_t nv = n_embd / 4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(n_tokens, hc, nv), sycl::range<3>(1, 1, wg)),
        [=](sycl::nd_item<3> item) {
            const int64_t it   = item.get_global_id(0);
            const int64_t idst = item.get_global_id(1);
            const int64_t i0   = item.get_global_id(2) * 4;

            float gate = post[idst*sp0 + it*sp1];
            if constexpr (fused_gate) {
                gate = g0[0] * gate + g0[1];
                gate = 1.0f / (1.0f + op_exp(-gate));  // exactly op_sigmoid()
                gate = g1[0] * gate + g1[1];
            }

            const sycl::float4 xv = *(const sycl::float4 *) (x + i0 + it*sx1);
            sycl::float4 sum;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sum[j] = xv[j] * gate;
            }
            if constexpr (has_comb) {
                for (int64_t isrc = 0; isrc < hc; ++isrc) {
                    const sycl::float4 rv = *(const sycl::float4 *) (residual + i0 + isrc*sr1 + it*sr2);
                    const float        c  = comb[idst*sc0 + isrc*sc1 + it*sc2];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        sum[j] += rv[j] * c;
                    }
                }
            } else {
                const sycl::float4 rv = *(const sycl::float4 *) (residual + i0 + idst*sr1 + it*sr2);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sum[j] += rv[j];
                }
            }

            *(sycl::float4 *) (dst + i0 + idst*sd1 + it*sd2) = sum;
        });
}

// dsv4_hc_pre_f32_sycl() for contiguous, 16 B aligned rows: 4 consecutive i0 per work-item and a
// 2D range, so x (and a gated weight) are read as float4 and no 64-bit division is left. Same
// operations per element in the same order, so the result is bit for bit the same.
template <bool gated>
static void dsv4_hc_pre_f32_vec4_sycl(
        const float * x, const float * weights, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens, int wg,
        int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd1,
        float scale,
        queue_ptr stream) {
    const int64_t nv = n_embd / 4;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(n_tokens, nv), sycl::range<2>(1, wg)),
        [=](sycl::nd_item<2> item) {
            const int64_t it = item.get_global_id(0);
            const int64_t i0 = item.get_global_id(1) * 4;

            sycl::float4 sum(0.0f);
            for (int64_t ih = 0; ih < hc; ++ih) {
                const sycl::float4 xv = *(const sycl::float4 *) (x + i0 + ih*sx1 + it*sx2);
                if constexpr (gated) {
                    const sycl::float4 gv = *(const sycl::float4 *) (weights + i0 + ih*sw1 + it*sw2);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float wv = 1.0f / (1.0f + sycl::exp(-gv[j]));
                        sum[j] += xv[j] * wv;
                    }
                } else {
                    const float wv = weights[ih*sw0 + it*sw1];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        sum[j] += xv[j] * wv;
                    }
                }
            }

            sycl::float4 out;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[j] = scale * sum[j];
            }
            *(sycl::float4 *) (dst + i0 + it*sd1) = out;
        });
}

// work-group size for the float4 path, or 0 when a row is not contiguous and 16 B aligned
static int dsv4_hc_vec4_wg(const ggml_tensor * const * ts, int n, int64_t n_embd) {
    if (!(g_ggml_sycl_wide_loads & GGML_SYCL_WIDE_HC) || n_embd % 4 != 0) {
        return 0;
    }
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * t = ts[k];
        if (t->nb[0] != sizeof(float) || (uintptr_t) t->data % 16 != 0) {
            return 0;
        }
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return 0;
            }
        }
    }
    for (int wg : { 256, 128, 64, 32, 16 }) {
        if ((n_embd / 4) % wg == 0) {
            return wg;
        }
    }
    return 0;
}

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    const ggml_tensor * vec_ts[3] = { x, dst, weights };
    const int           wg        = dsv4_hc_vec4_wg(vec_ts, gated ? 3 : 2, n_embd);

    if (gated && wg > 0) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_vec4_sycl<true>(
                (const float *) x->data, (const float *) weights->data, (float *) dst->data,
                n_embd, hc, n_tokens, wg,
                nbx1 / sizeof(float), nbx2 / sizeof(float),
                nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
                nbd1 / sizeof(float),
                scale, stream);
    } else if (!gated && wg > 0) {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_vec4_sycl<false>(
                (const float *) x->data, (const float *) weights->data, (float *) dst->data,
                n_embd, hc, n_tokens, wg,
                nbx1 / sizeof(float), nbx2 / sizeof(float),
                nbw0 / sizeof(float), nbw1 / sizeof(float), /*sw2=*/ 0,
                nbd1 / sizeof(float),
                scale, stream);
    } else if (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<true>(
                (const float *) x->data, (const float *) weights->data, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
                nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
                nbd0 / sizeof(float), nbd1 / sizeof(float),
                scale, stream);
    } else {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<false>(
                (const float *) x->data, (const float *) weights->data, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
                nbw0 / sizeof(float), nbw1 / sizeof(float), /*sw2=*/ 0,
                nbd0 / sizeof(float), nbd1 / sizeof(float),
                scale, stream);
    }
}

void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter, stream);
}

// gate == nullptr keeps dst->src[2] as the gate; otherwise the scale -> sigmoid -> scale chain
// that produced it is folded into this launch and its three nodes are skipped by the caller.
static void dsv4_hc_post_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                              const ggml_sycl_dsv4_hc_post_gate * gate) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = gate ? gate->src : dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    size_t nbc0 = 0;
    size_t nbc1 = 0;
    size_t nbc2 = 0;
    if (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    // the folded chain is elementwise, so its input indexes exactly like the gate it replaces.
    // Only assert on the fused path: the plain path never carried this check and must keep
    // accepting whatever shapes it already did.
    if (gate) {
        GGML_ASSERT(post->ne[0] == hc && post->ne[1] == n_tokens);
    }

    const sycl::float2 g0(gate ? gate->s0 : 1.0f, gate ? gate->b0 : 0.0f);
    const sycl::float2 g1(gate ? gate->s1 : 1.0f, gate ? gate->b1 : 0.0f);

    queue_ptr stream = ctx.stream();

    const ggml_tensor * vec_ts[3] = { x, residual, dst };
    const int           wg        = dsv4_hc_vec4_wg(vec_ts, 3, n_embd);

    const auto launch = [&](auto has_comb, auto fused_gate) {
        if (wg > 0) {
            dsv4_hc_post_f32_vec4_sycl<decltype(has_comb)::value, decltype(fused_gate)::value>(
                (const float *) x->data, (const float *) residual->data,
                (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
                n_embd, hc, n_tokens, wg,
                nbx1 / sizeof(float),
                nbr1 / sizeof(float), nbr2 / sizeof(float),
                nbp0 / sizeof(float), nbp1 / sizeof(float),
                nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
                nbd1 / sizeof(float), nbd2 / sizeof(float),
                g0, g1, stream);
            return;
        }
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value, decltype(fused_gate)::value>(
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            g0, g1, stream);
    };

    if (comb && gate)   { launch(std::true_type{},  std::true_type{});  }
    else if (comb)      { launch(std::true_type{},  std::false_type{}); }
    else if (gate)      { launch(std::false_type{}, std::true_type{});  }
    else                { launch(std::false_type{}, std::false_type{}); }
}

void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    dsv4_hc_post_impl(ctx, dst, /*gate=*/nullptr);
}

void ggml_sycl_op_dsv4_hc_post_fused_gate(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                          const ggml_sycl_dsv4_hc_post_gate & gate) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    GGML_ASSERT(gate.src != nullptr);
    dsv4_hc_post_impl(ctx, dst, &gate);
}
