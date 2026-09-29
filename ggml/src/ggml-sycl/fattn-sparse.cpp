#include "kv-soa.hpp"
#include "kq-mask-bits.hpp"
#include "fattn.hpp"
#include "fattn-sparse.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

static constexpr int64_t SPARSE_FA_PAD       = 256;
static constexpr int64_t SPARSE_FA_MIN_RATIO = 2;
// a batch of up to this many queries (e.g. an MTP verify batch) shares one gathered set: the union of the rows
static constexpr int64_t SPARSE_FA_MAX_ROWS  = 8;
// the gathered mask has this many rows at least, so a tile kernel may read a whole ncols1 group
static constexpr int64_t SPARSE_FA_ROW_PAD   = 16;
static constexpr int     SPARSE_FA_WG        = 256;
// cells one work-group of the compaction owns: one 32-cell word per work-item
static constexpr int64_t SPARSE_FA_WG_CELLS  = 32 * SPARSE_FA_WG;

extern int g_ggml_sycl_enable_sparse_fa;
extern int g_ggml_sycl_debug_sparse_fa;
extern int g_ggml_sycl_sparse_fa_margin;

static int sparse_fa_enabled(void) {
    return g_ggml_sycl_enable_sparse_fa;
}

static int sparse_fa_debug(void) {
    return g_ggml_sycl_debug_sparse_fa;
}

// slack above n_kv_max; callers may exceed the hint by a few always-attended positions
static int sparse_fa_margin(void) {
    return g_ggml_sycl_sparse_fa_margin;
}

// bit b of word w: some query row sees cell 32*w + b
static __dpct_inline__ uint32_t sparse_fa_word(const sycl::half * mask, int64_t n_kv, int64_t n_rows, size_t s1,
                                               int64_t w) {
    uint32_t      bits = 0;
    const int64_t c0   = w * 32;
    if (c0 >= n_kv) {
        return 0;
    }
    const int nc = (int) sycl::min((int64_t) 32, n_kv - c0);
    for (int64_t r = 0; r < n_rows; ++r) {
        const sycl::half * row = mask + (size_t) r * s1 + c0;
        for (int b = 0; b < nc; ++b) {
            if (sycl::isfinite((float) row[b])) {
                bits |= 1u << b;
            }
        }
    }
    return bits;
}

// The union of the rows' visible cells, in ascending order, so the gathered set and the result
// do not depend on scheduling. Pass 1 counts per work-group, pass 2 writes at the prefix.
// count gets the uncapped total; only the first n_kv_g cells are written.
static void sparse_fa_compact_mask(sycl::queue * stream,
                                   const sycl::half * __restrict__ mask,
                                   int32_t * __restrict__ indices,
                                   int32_t * __restrict__ count,
                                   int32_t * __restrict__ wg_counts,
                                   const int64_t n_kv,
                                   const int64_t n_rows,
                                   const size_t  s1,
                                   const int64_t n_kv_g) {
    const int64_t n_wg  = (n_kv + SPARSE_FA_WG_CELLS - 1) / SPARSE_FA_WG_CELLS;
    const sycl::nd_range<1> range((size_t) n_wg * SPARSE_FA_WG, SPARSE_FA_WG);

    stream->parallel_for(range, [=](sycl::nd_item<1> it) {
        const int64_t w   = (int64_t) it.get_global_id(0);
        const int     cnt = sycl::popcount(sparse_fa_word(mask, n_kv, n_rows, s1, w));
        const int     tot = sycl::reduce_over_group(it.get_group(), cnt, sycl::plus<int>());
        if (it.get_local_id(0) == 0) {
            wg_counts[it.get_group(0)] = tot;
        }
    });

    stream->parallel_for(range, [=](sycl::nd_item<1> it) {
        const int64_t g = (int64_t) it.get_group(0);
        int64_t base = 0;
        for (int64_t j = 0; j < g; ++j) {
            base += wg_counts[j];
        }
        const int64_t w   = (int64_t) it.get_global_id(0);
        uint32_t      u   = sparse_fa_word(mask, n_kv, n_rows, s1, w);
        const int     cnt = sycl::popcount(u);
        int64_t       pos = base + sycl::exclusive_scan_over_group(it.get_group(), cnt, sycl::plus<int>());
        while (u) {
            if (pos < n_kv_g) {
                indices[pos] = (int32_t) (w * 32 + sycl::ctz(u));
            }
            ++pos;
            u &= u - 1;
        }
        if (g == n_wg - 1) {
            const int tot = sycl::reduce_over_group(it.get_group(), cnt, sycl::plus<int>());
            if (it.get_local_id(0) == 0) {
                *count = (int32_t) (base + tot);
            }
        }
    });
}

// Rows along ne[0] are contiguous for every type used as a KV cache, so this is
// a plain byte copy and needs no per-type code. Slots past the live cells are
// zeroed up to the next FATTN_KQ_STRIDE; the kernel never reads further.
static void sparse_fa_gather_rows(sycl::queue * stream,
                                  const uint8_t * __restrict__ src,
                                  uint8_t * __restrict__ dst,
                                  const int32_t * __restrict__ indices,
                                  const int32_t * __restrict__ count,
                                  const size_t row_size,
                                  const size_t src_nb1,
                                  const size_t src_nb2,
                                  const int64_t n_kv_g,
                                  const int64_t n_head) {
    GGML_ASSERT(row_size % sizeof(uint32_t) == 0);
    const size_t words = row_size / sizeof(uint32_t);

    stream->parallel_for(
        sycl::range<3>((size_t) n_head, (size_t) n_kv_g, words),
        [=](sycl::id<3> id) {
            const int64_t h    = (int64_t) id[0];
            const int64_t slot = (int64_t) id[1];
            const size_t  w    = id[2];

            uint32_t * dst_row =
                (uint32_t *) (dst + ((size_t) (h * n_kv_g + slot)) * row_size);

            const int64_t cnt = *count;
            if (slot >= cnt) {
                if (slot < GGML_PAD(cnt, SPARSE_FA_PAD)) {
                    dst_row[w] = 0;
                }
                return;
            }

            const uint32_t * src_row =
                (const uint32_t *) (src + (size_t) indices[slot] * src_nb1 +
                                    (size_t) h * src_nb2);
            dst_row[w] = src_row[w];
        });
}

// The same for a SoA-span q8_0 cache (kv-soa.hpp), written back in the canonical block order: the
// gathered copy is a plain tensor with no layout marker, so every kernel reads it as canonical.
// One work-item per q8_0 block.
static void sparse_fa_gather_rows_soa(sycl::queue * stream,
                                      const uint8_t * __restrict__ src,
                                      uint8_t * __restrict__ dst,
                                      const int32_t * __restrict__ indices,
                                      const int32_t * __restrict__ count,
                                      const int64_t ne0,
                                      const size_t src_nb1,
                                      const size_t src_nb2,
                                      const int64_t n_kv_g,
                                      const int64_t n_head) {
    GGML_ASSERT(ne0 % GGML_SYCL_KV_SOA_SPAN == 0);
    const int64_t nblk     = ne0 / QK8_0;
    const size_t  row_size = (size_t) nblk * sizeof(block_q8_0);

    stream->parallel_for(
        sycl::range<3>((size_t) n_head, (size_t) n_kv_g, (size_t) nblk),
        [=](sycl::id<3> id) {
            const int64_t h    = (int64_t) id[0];
            const int64_t slot = (int64_t) id[1];
            const int     b    = (int) id[2];

            block_q8_0 * out = (block_q8_0 *) (dst + (size_t) (h * n_kv_g + slot) * row_size) + b;

            const int64_t cnt = *count;
            if (slot >= cnt) {
                if (slot < GGML_PAD(cnt, SPARSE_FA_PAD)) {
                    out->d = sycl::half(0.0f);
                    for (int i = 0; i < QK8_0; ++i) {
                        out->qs[i] = 0;
                    }
                }
                return;
            }

            using A = ggml_sycl_q8_0_access<GGML_SYCL_LAYOUT_SOA_SPAN>;
            const char * row = (const char *) src + (size_t) indices[slot] * src_nb1 + (size_t) h * src_nb2;
            size_t off;
            int    iblk;
            ggml_sycl_q8_0_locate<GGML_SYCL_LAYOUT_SOA_SPAN>((int64_t) b * QK8_0, off, iblk);
            const int8_t * qs = A::qs(row + off, iblk);
            out->d = sycl::half(A::d(row + off, iblk));
            for (int i = 0; i < QK8_0; ++i) {
                out->qs[i] = qs[i];
            }
        });
}

// n_rows_g >= n_rows: the extra rows are -inf, so a tile kernel that reads a whole ncols1 group
// past the last query finds nothing visible there.
static void sparse_fa_gather_mask(sycl::queue * stream,
                                  const sycl::half * __restrict__ mask,
                                  sycl::half * __restrict__ mask_g,
                                  const int32_t * __restrict__ indices,
                                  const int32_t * __restrict__ count,
                                  const int64_t n_kv_g,
                                  const int64_t n_rows,
                                  const int64_t n_rows_g,
                                  const size_t mask_s1) {
    stream->parallel_for(
        sycl::range<2>((size_t) n_rows_g, (size_t) n_kv_g),
        [=](sycl::id<2> id) {
            const int64_t r    = (int64_t) id[0];
            const int64_t slot = (int64_t) id[1];

            sycl::half v = sycl::half(-INFINITY);
            if (r < n_rows && slot < (int64_t) *count) {
                v = mask[(size_t) r * mask_s1 + (size_t) indices[slot]];
            }
            mask_g[(size_t) r * n_kv_g + slot] = v;
        });
}

static bool sparse_fa_applicable(const ggml_tensor * dst, int64_t & n_kv_g_out) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!Q || !K || !V || !mask) {
        return false;
    }

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (n_kv_max <= 0) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    // decode and short batches (an MTP verify batch); prefill amortises the scan already
    if (Q->ne[1] < 1 || Q->ne[1] > SPARSE_FA_MAX_ROWS) {
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    // a packed mask (kq-mask-bits.hpp) is not f16 cells
    if (mask->type != GGML_TYPE_F16 || ggml_sycl_kq_mask_is_bits(mask)) {
        return false;
    }
    if (mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->nb[0] != sizeof(sycl::half)) {
        return false;
    }
    if (K->ne[2] != V->ne[2]) {
        return false;
    }

    // nb[1] may stride over heads (interleaved cache); only ne[0] must be contiguous
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }

    const size_t k_row = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row = ggml_row_size(V->type, V->ne[0]);
    if (k_row % sizeof(uint32_t) || v_row % sizeof(uint32_t)) {
        return false;
    }
    for (const ggml_tensor * t : { K, V }) {
        if (ggml_sycl_kv_is_soa(t) && t->ne[0] % GGML_SYCL_KV_SOA_SPAN != 0) {
            return false;
        }
    }

    // every row sees at most n_kv_max cells, so the union of the rows is bounded by their sum
    const int64_t n_sel  = std::min<int64_t>(K->ne[1], Q->ne[1] * (int64_t) n_kv_max);
    const int64_t n_kv_g = GGML_PAD(n_sel + sparse_fa_margin(), SPARSE_FA_PAD);
    if (n_kv_g * SPARSE_FA_MIN_RATIO > K->ne[1]) {
        return false;
    }

    n_kv_g_out = n_kv_g;
    return true;
}

bool ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    int64_t n_kv_g = 0;
    if (!sparse_fa_enabled() || !sparse_fa_applicable(dst, n_kv_g)) {
        return false;
    }

    ggml_tensor * Q    = dst->src[0];
    ggml_tensor * K    = dst->src[1];
    ggml_tensor * V    = dst->src[2];
    ggml_tensor * mask = dst->src[3];

    const int64_t n_kv     = K->ne[1];
    const int64_t n_head_k = K->ne[2];
    const int64_t n_rows_q = Q->ne[1];
    const int64_t n_rows_m = mask->ne[1];
    const int64_t n_rows_g = std::max<int64_t>(n_rows_m, SPARSE_FA_ROW_PAD);
    const size_t  mask_s1  = mask->nb[1] / sizeof(sycl::half);
    const int64_t n_wg     = (n_kv + SPARSE_FA_WG_CELLS - 1) / SPARSE_FA_WG_CELLS;

    const bool k_soa = ggml_sycl_kv_is_soa(K);
    const bool v_soa = ggml_sycl_kv_is_soa(V);

    const size_t k_row = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row = ggml_row_size(V->type, V->ne[0]);

    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t>    idx_alloc(ctx.pool(), (size_t) n_kv_g);
    ggml_sycl_pool_alloc<int32_t>    cnt_alloc(ctx.pool(), (size_t) n_wg + 1);
    ggml_sycl_pool_alloc<uint8_t>    k_alloc(ctx.pool(), (size_t) n_head_k * n_kv_g * k_row);
    ggml_sycl_pool_alloc<uint8_t>    v_alloc(ctx.pool(), (size_t) n_head_k * n_kv_g * v_row);
    ggml_sycl_pool_alloc<sycl::half> m_alloc(ctx.pool(), (size_t) n_rows_g * n_kv_g);

    int32_t *    d_idx  = idx_alloc.get();
    int32_t *    d_cnt  = cnt_alloc.get();
    int32_t *    d_wgc  = d_cnt + 1;
    uint8_t *    d_K    = k_alloc.get();
    uint8_t *    d_V    = v_alloc.get();
    sycl::half * d_mask = m_alloc.get();

    sparse_fa_compact_mask(stream, (const sycl::half *) mask->data, d_idx, d_cnt, d_wgc,
                           n_kv, n_rows_q, mask_s1, n_kv_g);

    if (k_soa) {
        sparse_fa_gather_rows_soa(stream, (const uint8_t *) K->data, d_K, d_idx, d_cnt,
                                  K->ne[0], K->nb[1], K->nb[2], n_kv_g, n_head_k);
    } else {
        sparse_fa_gather_rows(stream, (const uint8_t *) K->data, d_K, d_idx, d_cnt,
                              k_row, K->nb[1], K->nb[2], n_kv_g, n_head_k);
    }
    if (v_soa) {
        sparse_fa_gather_rows_soa(stream, (const uint8_t *) V->data, d_V, d_idx, d_cnt,
                                  V->ne[0], V->nb[1], V->nb[2], n_kv_g, n_head_k);
    } else {
        sparse_fa_gather_rows(stream, (const uint8_t *) V->data, d_V, d_idx, d_cnt,
                              v_row, V->nb[1], V->nb[2], n_kv_g, n_head_k);
    }

    sparse_fa_gather_mask(stream, (const sycl::half *) mask->data, d_mask,
                          d_idx, d_cnt, n_kv_g, n_rows_m, n_rows_g, mask_s1);

    if (sparse_fa_debug()) {
        int32_t h_cnt = 0;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(&h_cnt, d_cnt, sizeof(int32_t))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        fprintf(stderr, "[FA-SPARSE] n_kv=%lld n_q=%lld n_kv_max=%d n_kv_g=%lld finite=%d soa=%d%s\n",
                (long long) n_kv, (long long) n_rows_q, ggml_get_op_params_i32(dst, 4),
                (long long) n_kv_g, (int) h_cnt, (int) (k_soa || v_soa),
                h_cnt > (int32_t) n_kv_g ? "  OVERFLOW" : "");
    }

    // shallow copies retargeted at the gathered buffers; kernels are unchanged. The gathered
    // K/V are plain canonical tensors: no view, no layout marker.
    ggml_tensor K_g = *K;
    K_g.data      = d_K;
    K_g.ne[1]     = n_kv_g;
    K_g.nb[1]     = k_row;
    K_g.nb[2]     = (size_t) n_kv_g * k_row;
    K_g.nb[3]     = (size_t) n_head_k * n_kv_g * k_row;
    K_g.view_src  = nullptr;
    K_g.view_offs = 0;
    K_g.extra     = nullptr;

    ggml_tensor V_g = *V;
    V_g.data      = d_V;
    V_g.ne[1]     = n_kv_g;
    V_g.nb[1]     = v_row;
    V_g.nb[2]     = (size_t) n_kv_g * v_row;
    V_g.nb[3]     = (size_t) V->ne[2] * n_kv_g * v_row;
    V_g.view_src  = nullptr;
    V_g.view_offs = 0;
    V_g.extra     = nullptr;

    ggml_tensor M_g = *mask;
    M_g.data      = d_mask;
    M_g.ne[0]     = n_kv_g;
    M_g.nb[1]     = (size_t) n_kv_g * sizeof(sycl::half);
    M_g.nb[2]     = M_g.nb[1] * mask->ne[1];
    M_g.nb[3]     = M_g.nb[2];
    M_g.view_src  = nullptr;
    M_g.view_offs = 0;
    M_g.extra     = nullptr;

    ggml_tensor dst_g = *dst;
    dst_g.src[1] = &K_g;
    dst_g.src[2] = &V_g;
    dst_g.src[3] = &M_g;
    // not re-entering this path; the live cells come first, so the kernel may stop at the last one
    dst_g.op_params[4] = GGML_SYCL_FATTN_GATHERED;

    ggml_sycl_flash_attn_ext(ctx, &dst_g);

    return true;
}
