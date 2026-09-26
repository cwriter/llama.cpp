#include "fattn-qsa.hpp"
#include "fattn.hpp"
#include "kv-soa.hpp"

#include <oneapi/mkl.hpp>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// Flash attention for the QSA layers without the dense [n_kv, n_tps] mask. The graph says
//   out = flash_attn(q, k, v, (top-k select ? kq_mask : -inf))
// and here the FLASH_ATTN_EXT node gets the top-k list and the causal mask instead.
//
// Prefill (the chunked oneMKL kernel would take the node): neighbouring tokens select mostly the
// same cells, so a tile of T tokens gathers the UNION of its lists once, dequantizes those rows
// to f16, and runs the same XMX GEMMs the dense kernel runs over them, with an online softmax
// that keeps, per query row, only the cells that row's own list names. The union is a sorted
// cell list, so the result is deterministic. At a short context the union is the whole cache
// and this is the dense kernel with bigger query tiles; at a long context it is a fraction.
//
// Everything else (decode, short batches, sinks, softcap, ALiBi): the mask the graph wanted is
// written into pool scratch at its real size and the dense kernels run unchanged, so the output
// is bit for bit what the unfused graph gives, and no mask is reserved at the worst case.

static constexpr int QSA_UNION_T      = 64;    // query tokens per tile
static constexpr int QSA_UNION_CHUNK  = 8192;  // union cells per GEMM
// up to this many cells a tile's union is about the whole cache, so the dense oneMKL kernel
// takes the selection bits directly (fattn.hpp), with far fewer launches
static constexpr int QSA_UNION_MIN_KV = 8192;
static constexpr int QSA_WG          = 256;
static constexpr int QSA_PAD         = 8;     // a union chunk is padded to this, so rows load 16 bytes at a time

enum qsa_kv_kind {
    QSA_KV_F16,
    QSA_KV_Q8_0,
    QSA_KV_Q8_0_SOA,
};

using oneapi::mkl::transpose;
using oneapi::mkl::blas::column_major::gemm;

// ---------------------------------------------------------------------------------------------
// the mask, written at its real size, then the dense kernels
// ---------------------------------------------------------------------------------------------

static void qsa_dense_scratch_fa(ggml_backend_sycl_context & ctx, ggml_tensor * fa, const ggml_tensor * mask,
                                 const ggml_tensor * idx) {
    const int64_t n_kv   = mask->ne[0];
    const int64_t n_tps  = mask->ne[1];
    const int64_t n_rows = n_tps * mask->ne[3];
    const int64_t width  = idx->ne[0];
    const int64_t n_all  = n_rows * n_kv;

    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<sycl::half> scratch(ctx.pool(), (size_t) n_all);
    sycl::half *       dst   = scratch.get();
    const sycl::half * src   = (const sycl::half *) mask->data;
    const char *       idx_d = (const char *) idx->data;
    const size_t       i_nb1 = idx->nb[1];
    const size_t       i_nb2 = idx->nb[2];
    const sycl::half   ninf  = sycl::half(-INFINITY);

    stream->parallel_for(sycl::nd_range<1>(((n_all + QSA_WG - 1) / QSA_WG) * QSA_WG, QSA_WG), [=](sycl::nd_item<1> it) {
        const int64_t i = (int64_t) it.get_global_id(0);
        if (i < n_all) {
            dst[i] = ninf;
        }
    });
    // rows are contiguous in the mask (the chain matcher requires it), so row r is r * n_kv
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>(n_rows, ((width + QSA_WG - 1) / QSA_WG) * QSA_WG),
                                           sycl::range<2>(1, QSA_WG)), [=](sycl::nd_item<2> it) {
        const int64_t w = (int64_t) it.get_global_id(1);
        if (w >= width) {
            return;
        }
        const int64_t r = (int64_t) it.get_global_id(0);
        const int64_t t = r % n_tps;
        const int64_t s = r / n_tps;
        const int64_t c = *(const int32_t *) (idx_d + w * sizeof(int32_t) + t * i_nb1 + s * i_nb2);
        if (c >= 0 && c < n_kv) {
            dst[r * n_kv + c] = src[r * n_kv + c];
        }
    });

    ggml_tensor m = *mask;
    m.data     = dst;
    m.extra    = nullptr;
    m.view_src = nullptr;
    ggml_tensor node = *fa;
    node.src[3] = &m;
    ggml_sycl_flash_attn_ext(ctx, &node);
}

// ---------------------------------------------------------------------------------------------
// union tiles
// ---------------------------------------------------------------------------------------------

// One bit per (query row, cell): the cell is in the row's list and the causal mask lets the row
// see it. odd is set when a visible listed cell carries a finite mask value that is not zero;
// only then does the softmax read the mask.
static void k_qsa_bits(const char * idx, const char * mask, uint32_t * bits, int32_t * odd, int64_t n_kv,
                       int64_t n_tps, int64_t width, int64_t words, size_t i_nb1, size_t i_nb2, size_t m_nb1,
                       size_t m_nb3, const sycl::nd_item<2> & it) {
    const int64_t w = (int64_t) it.get_global_id(1);
    if (w >= width) {
        return;
    }
    const int64_t r = (int64_t) it.get_global_id(0);
    const int64_t t = r % n_tps;
    const int64_t s = r / n_tps;
    const int64_t c = *(const int32_t *) (idx + w * sizeof(int32_t) + t * i_nb1 + s * i_nb2);
    if (c < 0 || c >= n_kv) {
        return;
    }
    const float m = (float) ((const sycl::half *) (mask + t * m_nb1 + s * m_nb3))[c];
    if (!sycl::isfinite(m)) {
        return;
    }
    if (m != 0.0f) {
        *odd = 1;
    }
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(bits[r * words + (c >> 5)]);
    ref.fetch_or(1u << (c & 31));
}

// One work-group per tile: OR the rows of the tile, then write the set cells in ascending order
static void k_qsa_union(const uint32_t * bits, int32_t * ulist, int32_t * ucount, int64_t cap, int64_t n_tps,
                        int64_t words, int n_tiles, const sycl::nd_item<1> & it) {
    const auto wg   = it.get_group();
    const int  lid  = (int) it.get_local_linear_id();
    const int  tile = (int) it.get_group(0);
    const int  s    = tile / n_tiles;
    const int  t0   = (tile % n_tiles) * QSA_UNION_T;
    const int  t1   = (int) sycl::min((int64_t) t0 + QSA_UNION_T, n_tps);

    const uint32_t * rows = bits + ((int64_t) s * n_tps + t0) * words;
    int32_t *        out  = ulist + (int64_t) tile * cap;

    int carry = 0;
    for (int64_t wb = 0; wb < words; wb += QSA_WG) {
        const int64_t w = wb + lid;
        uint32_t      u = 0;
        if (w < words) {
            for (int t = 0; t < t1 - t0; ++t) {
                u |= rows[(int64_t) t * words + w];
            }
        }
        const int cnt = sycl::popcount(u);
        int       pos = carry + sycl::exclusive_scan_over_group(wg, cnt, sycl::plus<int>());
        carry += sycl::reduce_over_group(wg, cnt, sycl::plus<int>());
        while (u) {
            const int b  = sycl::ctz(u);
            out[pos++]   = (int32_t) (w * 32 + b);
            u &= u - 1;
        }
    }
    // the slots that pad the last chunk name no cell
    if (lid < QSA_PAD) {
        out[carry + lid] = -1;
    }
    if (lid == 0) {
        ucount[tile] = carry;
    }
}

// Q rows of one tile as scaled f16: row = head-in-group * tile length + token
static void k_qsa_pack_q(const char * Q, sycl::half * out, int D, int n_rows, int Tt, int t0, int h0, int s,
                         float scale, size_t q_nb1, size_t q_nb2, size_t q_nb3, const sycl::nd_item<1> & it) {
    const int64_t e   = (int64_t) it.get_global_id(0) * 4;
    const int     row = (int) (e / D);
    if (row >= n_rows) {
        return;
    }
    const int     d   = (int) (e - (int64_t) row * D);
    const int     hg  = row / Tt;
    const int     t   = t0 + row - hg * Tt;
    const float * q   = (const float *) (Q + (size_t) t * q_nb1 + (size_t) (h0 + hg) * q_nb2 + (size_t) s * q_nb3);
    const sycl::float4 v = *(const sycl::float4 *) (q + d) * scale;
    *(sycl::half4 *) (out + e) = v.convert<sycl::half>();
}

// 16 features of one K or V row, dequantized, as two 16-byte stores
template <int KIND>
static __dpct_inline__ void qsa_deq16(const char * row, int f0, sycl::half * out) {
    sycl::half8 * dst = (sycl::half8 *) out;
    if constexpr (KIND == QSA_KV_F16) {
        const sycl::half8 * src = (const sycl::half8 *) (row + (size_t) f0 * sizeof(sycl::half));
        dst[0] = src[0];
        dst[1] = src[1];
        return;
    }
    sycl::half8 lo, hi;
    if constexpr (KIND == QSA_KV_Q8_0_SOA) {
        // the span keeps its quants 16-byte aligned and its eight scales in one 16-byte word
        using A = ggml_sycl_q8_0_access<GGML_SYCL_LAYOUT_SOA_SPAN>;
        size_t off;
        int    iblk;
        ggml_sycl_q8_0_locate<GGML_SYCL_LAYOUT_SOA_SPAN>(f0, off, iblk);
        const sycl::char16 q  = *(const sycl::char16 *) (A::qs(row + off, iblk) + f0 % QK8_0);
        const sycl::uint4  sw = *(const sycl::uint4 *) (row + off + GGML_SYCL_KV_SOA_SPAN);
        // pick the block's scale with selects: a dynamic vector index would go through private memory
        const uint32_t     w  = (iblk >> 1) == 0 ? sw.x() : (iblk >> 1) == 1 ? sw.y() : (iblk >> 1) == 2 ? sw.z() : sw.w();
        const float        d  = (float) sycl::bit_cast<sycl::half>((uint16_t) ((iblk & 1) ? (w >> 16) : (w & 0xffff)));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            lo[i] = sycl::half(d * (float) q[i]);
            hi[i] = sycl::half(d * (float) q[i + 8]);
        }
    } else {
        // a canonical block is only 2-byte aligned, so its quants come in as eight 2-byte words
        using A = ggml_sycl_q8_0_access<GGML_SYCL_LAYOUT_CANONICAL>;
        const uint16_t * q = (const uint16_t *) (A::qs(row, f0 / QK8_0) + f0 % QK8_0);
        const float      d = A::d(row, f0 / QK8_0);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint16_t a = q[i];
            const uint16_t b = q[i + 4];
            lo[2 * i]     = sycl::half(d * (float) (int8_t) (a & 0xff));
            lo[2 * i + 1] = sycl::half(d * (float) (int8_t) (a >> 8));
            hi[2 * i]     = sycl::half(d * (float) (int8_t) (b & 0xff));
            hi[2 * i + 1] = sycl::half(d * (float) (int8_t) (b >> 8));
        }
    }
    dst[0] = lo;
    dst[1] = hi;
}

// the K rows (then the V rows) of one union chunk as dense f16 [n x D]
template <int KIND>
static void k_qsa_gather(const char * K, const char * V, size_t k_nb1, size_t v_nb1, const int32_t * cells,
                         sycl::half * Kf, sycl::half * Vf, int n_used, int n, int D, const sycl::nd_item<1> & it) {
    const int64_t e   = (int64_t) it.get_global_id(0);
    const int     fgs = D / 16;
    const int64_t per = (int64_t) n * fgs;
    if (e >= 2 * per) {
        return;
    }
    const bool    isv = e >= per;
    const int64_t x   = isv ? e - per : e;
    const int     i   = (int) (x / fgs);
    const int     f0  = (int) (x - (int64_t) i * fgs) * 16;
    sycl::half *  out = (isv ? Vf : Kf) + (size_t) i * D + f0;
    if (i >= n_used) {
        ((sycl::half8 *) out)[0] = sycl::half8(0.0f);
        ((sycl::half8 *) out)[1] = sycl::half8(0.0f);
        return;
    }
    const int    c   = cells[i];
    const char * row = isv ? V + (size_t) c * v_nb1 : K + (size_t) c * k_nb1;
    qsa_deq16<KIND>(row, f0, out);
}

// Online softmax over one union chunk, one work-group per query row (as fattn-mkl.cpp does it).
// Cells the row's own list does not name score -inf. ODD reads the causal mask value.
template <bool ODD>
static void k_qsa_softmax(const float * KQ, sycl::half * S, float * kmax, float * ksum, float * acc,
                          const int32_t * cells, const uint32_t * bits, const char * mask, int n, int D, int Tt,
                          int t0, int s, int64_t n_tps, int64_t words, size_t m_nb1, size_t m_nb3,
                          const sycl::nd_item<1> & it) {
    const int lid = (int) it.get_local_id(0);
    const int row = (int) it.get_group(0);
    const int t   = t0 + row % Tt;

    const float *      kq   = KQ + (int64_t) row * n;
    sycl::half *       srow = S + (int64_t) row * n;
    float *            vkq  = acc + (int64_t) row * D;
    const uint32_t *   sel  = bits + ((int64_t) s * n_tps + t) * words;
    const sycl::half * mrow = ODD ? (const sycl::half *) (mask + (size_t) t * m_nb1 + (size_t) s * m_nb3) : nullptr;

    // n is a multiple of QSA_PAD, so cells, scores and S come and go four at a time
    const int          n4 = n / 4;
    const sycl::int4 *   c4 = (const sycl::int4 *) cells;
    const sycl::float4 * k4 = (const sycl::float4 *) kq;
    sycl::half4 *        s4 = (sycl::half4 *) srow;

    auto score = [&](int c, float v) {
        if (c < 0 || ((sel[c >> 5] >> (c & 31)) & 1u) == 0) {
            return -INFINITY;
        }
        if constexpr (ODD) {
            v += (float) mrow[c];
        }
        return v;
    };

    float lmax = -INFINITY;
    for (int i = lid; i < n4; i += QSA_WG) {
        const sycl::int4   c = c4[i];
        const sycl::float4 v = k4[i];
        lmax = sycl::fmax(lmax, sycl::fmax(sycl::fmax(score(c.x(), v.x()), score(c.y(), v.y())),
                                           sycl::fmax(score(c.z(), v.z()), score(c.w(), v.w()))));
    }
    const float cmax    = sycl::reduce_over_group(it.get_group(), lmax, sycl::maximum<float>());
    const float old     = kmax[row];
    const float nmax    = sycl::fmax(old, cmax);
    const float rescale = (old == -INFINITY) ? 1.0f : sycl::native::exp(old - nmax);
    for (int v = lid; v < D; v += QSA_WG) {
        vkq[v] *= rescale;
    }
    auto num = [&](int c, float v) {
        const float sc = score(c, v);
        return sc == -INFINITY ? 0.0f : sycl::native::exp(sc - nmax);
    };
    float lsum = 0.0f;
    for (int i = lid; i < n4; i += QSA_WG) {
        const sycl::int4   c = c4[i];
        const sycl::float4 v = k4[i];
        const sycl::float4 e(num(c.x(), v.x()), num(c.y(), v.y()), num(c.z(), v.z()), num(c.w(), v.w()));
        s4[i] = e.convert<sycl::half>();
        lsum += (e.x() + e.y()) + (e.z() + e.w());
    }
    const float csum = sycl::reduce_over_group(it.get_group(), lsum, sycl::plus<float>());
    if (lid == 0) {
        ksum[row] = ksum[row] * rescale + csum;
        kmax[row] = nmax;
    }
}

// a row that saw no cell gives 0, as the CPU reference does
static void k_qsa_normalize(const float * acc, const float * ksum, float * dst, int D, int n_rows, int Tt, int t0,
                            int h0, int s, int64_t n_tps, int n_head, const sycl::nd_item<1> & it) {
    const int64_t e   = (int64_t) it.get_global_id(0) * 4;
    const int     row = (int) (e / D);
    if (row >= n_rows) {
        return;
    }
    const int     d   = (int) (e - (int64_t) row * D);
    const int     hg  = row / Tt;
    const int     t   = t0 + row - hg * Tt;
    const float   sum = ksum[row];
    const float   inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    float *       o   = dst + (((int64_t) s * n_tps + t) * n_head + h0 + hg) * D + d;
    *(sycl::float4 *) o = *(const sycl::float4 *) (acc + e) * inv;
}

struct qsa_union_args {
    const char *  Q;
    size_t        q_nb1, q_nb2, q_nb3;
    const char *  K;
    size_t        k_nb1, k_nb2, k_nb3;
    const char *  V;
    size_t        v_nb1, v_nb2, v_nb3;
    const char *  mask;
    size_t        m_nb1, m_nb3;
    const char *  idx;
    size_t        i_nb1, i_nb2;
    float *       dst;
    int64_t       n_kv, n_tps, n_stream, width;
    int           D, n_head, n_head_kv;
    float         scale;
};

// bits: [n_stream * n_tps, words], zeroed here
static void qsa_build_bits(dpct::queue_ptr stream, const qsa_union_args & a, uint32_t * bits, int32_t * odd) {
    const int64_t words = (a.n_kv + 31) / 32;
    const int64_t n_rows = a.n_tps * a.n_stream;
    stream->memset(bits, 0, (size_t) n_rows * words * sizeof(uint32_t));
    const char *  idx   = a.idx;
    const char *  mask  = a.mask;
    const int64_t n_kv = a.n_kv, n_tps = a.n_tps, width = a.width;
    const size_t  i_nb1 = a.i_nb1, i_nb2 = a.i_nb2, m_nb1 = a.m_nb1, m_nb3 = a.m_nb3;
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>(n_rows, ((width + QSA_WG - 1) / QSA_WG) * QSA_WG),
                                           sycl::range<2>(1, QSA_WG)), [=](sycl::nd_item<2> it) {
        k_qsa_bits(idx, mask, bits, odd, n_kv, n_tps, width, words, i_nb1, i_nb2, m_nb1, m_nb3, it);
    });
}

// the dense oneMKL kernel over the whole cache, keeping the cells the bits name and adding the mask
static void qsa_bits_mkl_fa(ggml_backend_sycl_context & ctx, ggml_tensor * fa, const ggml_tensor * mask,
                            const qsa_union_args & a) {
    const int64_t words = (a.n_kv + 31) / 32;
    ggml_sycl_pool_alloc<uint32_t> bits(ctx.pool(), (size_t) a.n_tps * a.n_stream * words);
    ggml_sycl_pool_alloc<int32_t>  odd(ctx.pool(), 1);
    qsa_build_bits(ctx.stream(), a, bits.get(), odd.get());

    ggml_tensor node = *fa;
    node.src[3]      = const_cast<ggml_tensor *>(mask);
    ggml_sycl_flash_attn_ext_mkl(ctx, &node, bits.get(), words, 1);
}

template <int KIND>
static void qsa_union_fa(ggml_backend_sycl_context & ctx, const qsa_union_args & a) {
    const int64_t words   = (a.n_kv + 31) / 32;
    const int64_t cap     = (a.n_kv + 2 * QSA_PAD - 1) / QSA_PAD * QSA_PAD;  // a tile's list, plus its sentinels
    const int     n_tiles = (int) ((a.n_tps + QSA_UNION_T - 1) / QSA_UNION_T);
    const int     tiles   = (int) a.n_stream * n_tiles;
    const int     gqa     = a.n_head / a.n_head_kv;
    const int     D       = a.D;
    const int64_t n_rows  = a.n_tps * a.n_stream;
    const int     rows_max = gqa * QSA_UNION_T;

    dpct::queue_ptr  stream = ctx.stream();
    ggml_sycl_pool & pool   = ctx.pool();

    ggml_sycl_pool_alloc<uint32_t>   bits(pool, (size_t) n_rows * words);
    ggml_sycl_pool_alloc<int32_t>    ulist(pool, (size_t) tiles * cap);
    ggml_sycl_pool_alloc<int32_t>    ucount(pool, (size_t) tiles + 1);  // counts, then the odd flag
    ggml_sycl_pool_alloc<sycl::half> Qf(pool, (size_t) rows_max * D);
    ggml_sycl_pool_alloc<float>      KQ(pool, (size_t) rows_max * QSA_UNION_CHUNK);
    ggml_sycl_pool_alloc<sycl::half> S(pool, (size_t) rows_max * QSA_UNION_CHUNK);
    ggml_sycl_pool_alloc<sycl::half> Kf(pool, (size_t) QSA_UNION_CHUNK * D);
    ggml_sycl_pool_alloc<sycl::half> Vf(pool, (size_t) QSA_UNION_CHUNK * D);
    ggml_sycl_pool_alloc<float>      acc(pool, (size_t) rows_max * D);
    ggml_sycl_pool_alloc<float>      kmax(pool, (size_t) rows_max);
    ggml_sycl_pool_alloc<float>      ksum(pool, (size_t) rows_max);

    uint32_t *   bits_p   = bits.get();
    int32_t *    ulist_p  = ulist.get();
    int32_t *    ucount_p = ucount.get();
    sycl::half * Qf_p     = Qf.get();
    float *      KQ_p     = KQ.get();
    sycl::half * S_p      = S.get();
    sycl::half * Kf_p     = Kf.get();
    sycl::half * Vf_p     = Vf.get();
    float *      acc_p    = acc.get();
    float *      kmax_p   = kmax.get();
    float *      ksum_p   = ksum.get();

    // 1. selection bits, then the union of every tile; the host needs the union sizes for the GEMMs
    stream->memset(ucount_p + tiles, 0, sizeof(int32_t));
    qsa_build_bits(stream, a, bits_p, ucount_p + tiles);
    {
        const int64_t n_tps = a.n_tps;
        stream->parallel_for(sycl::nd_range<1>((size_t) tiles * QSA_WG, QSA_WG), [=](sycl::nd_item<1> it) {
            k_qsa_union(bits_p, ulist_p, ucount_p, cap, n_tps, words, n_tiles, it);
        });
    }
    std::vector<int32_t> counts((size_t) tiles + 1);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(counts.data(), ucount_p, counts.size() * sizeof(int32_t)).wait()));
    const bool odd = counts[tiles] != 0;

    // 2. per stream, KV head and tile: the dense kernel's pipeline over the tile's union
    for (int s = 0; s < (int) a.n_stream; ++s) {
        for (int g = 0; g < a.n_head_kv; ++g) {
            const char * Kh = a.K + (size_t) g * a.k_nb2 + (size_t) s * a.k_nb3;
            const char * Vh = a.V + (size_t) g * a.v_nb2 + (size_t) s * a.v_nb3;
            const int    h0 = g * gqa;
            for (int tile = 0; tile < n_tiles; ++tile) {
                const int       t0    = tile * QSA_UNION_T;
                const int       Tt    = (int) std::min<int64_t>(QSA_UNION_T, a.n_tps - t0);
                const int       rows  = gqa * Tt;
                const int       u     = counts[s * n_tiles + tile];
                const int32_t * cells = ulist_p + (int64_t) (s * n_tiles + tile) * cap;

                {
                    const char * Q = a.Q;
                    const float  scale = a.scale;
                    const size_t q_nb1 = a.q_nb1, q_nb2 = a.q_nb2, q_nb3 = a.q_nb3;
                    const int64_t n_el = (int64_t) rows * D;
                    const size_t  gl4  = ((n_el / 4 + QSA_WG - 1) / QSA_WG) * QSA_WG;
                    stream->parallel_for(sycl::nd_range<1>(gl4, QSA_WG), [=](sycl::nd_item<1> it) {
                        k_qsa_pack_q(Q, Qf_p, D, rows, Tt, t0, h0, s, scale, q_nb1, q_nb2, q_nb3, it);
                    });
                    const size_t gl = ((n_el + QSA_WG - 1) / QSA_WG) * QSA_WG;
                    stream->parallel_for(sycl::nd_range<1>(gl, QSA_WG), [=](sycl::nd_item<1> it) {
                        const int64_t i = (int64_t) it.get_global_id(0);
                        if (i < n_el) {
                            acc_p[i] = 0.0f;
                        }
                        if (i < rows) {
                            kmax_p[i] = -INFINITY;
                            ksum_p[i] = 0.0f;
                        }
                    });
                }

                for (int j0 = 0; j0 < u; j0 += QSA_UNION_CHUNK) {
                    const int       n_used = std::min(QSA_UNION_CHUNK, u - j0);
                    const int       n      = (n_used + QSA_PAD - 1) / QSA_PAD * QSA_PAD;
                    const int32_t * cj     = cells + j0;
                    {
                        const size_t  k_nb1 = a.k_nb1, v_nb1 = a.v_nb1;
                        const int64_t items = 2 * (int64_t) n * (D / 16);
                        stream->parallel_for(sycl::nd_range<1>(((items + QSA_WG - 1) / QSA_WG) * QSA_WG, QSA_WG),
                                             [=](sycl::nd_item<1> it) {
                            k_qsa_gather<KIND>(Kh, Vh, k_nb1, v_nb1, cj, Kf_p, Vf_p, n_used, n, D, it);
                        });
                    }
                    // KQ[i, row] = sum_d Kf[i, d] * Qf[row, d]
                    gemm(*stream, transpose::trans, transpose::nontrans, n, rows, D, 1.0f, Kf_p, D, Qf_p, D, 0.0f,
                         KQ_p, n);
                    {
                        const char *  mask  = a.mask;
                        const int64_t n_tps = a.n_tps;
                        const size_t  m_nb1 = a.m_nb1, m_nb3 = a.m_nb3;
                        const sycl::nd_range<1> range((size_t) rows * QSA_WG, QSA_WG);
                        if (odd) {
                            stream->parallel_for(range, [=](sycl::nd_item<1> it) {
                                k_qsa_softmax<true>(KQ_p, S_p, kmax_p, ksum_p, acc_p, cj, bits_p, mask, n, D, Tt, t0,
                                                    s, n_tps, words, m_nb1, m_nb3, it);
                            });
                        } else {
                            stream->parallel_for(range, [=](sycl::nd_item<1> it) {
                                k_qsa_softmax<false>(KQ_p, S_p, kmax_p, ksum_p, acc_p, cj, bits_p, mask, n, D, Tt, t0,
                                                     s, n_tps, words, m_nb1, m_nb3, it);
                            });
                        }
                    }
                    // acc[row, d] += sum_i Vf[i, d] * S[row, i]; the softmax rescaled acc first
                    gemm(*stream, transpose::nontrans, transpose::nontrans, D, rows, n, 1.0f, Vf_p, D, S_p, n, 1.0f,
                         acc_p, D);
                }

                {
                    float *       dst    = a.dst;
                    const int64_t n_tps  = a.n_tps;
                    const int     n_head = a.n_head;
                    const int64_t n_el = (int64_t) rows * D;
                    stream->parallel_for(sycl::nd_range<1>(((n_el / 4 + QSA_WG - 1) / QSA_WG) * QSA_WG, QSA_WG),
                                         [=](sycl::nd_item<1> it) {
                        k_qsa_normalize(acc_p, ksum_p, dst, D, rows, Tt, t0, h0, s, n_tps, n_head, it);
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------

bool ggml_sycl_qsa_sparse_fa_supported(const ggml_tensor * fa) {
    if (!fa || fa->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * Q    = fa->src[0];
    const ggml_tensor * K    = fa->src[1];
    const ggml_tensor * V    = fa->src[2];
    const ggml_tensor * mask = fa->src[3];
    if (!Q || !K || !V || !mask) {
        return false;
    }
    // the scratch mask is written in the layout the graph gave the node
    if (mask->type != GGML_TYPE_F16 || mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] || mask->ne[2] != 1 ||
        mask->ne[3] != Q->ne[3] || !ggml_is_contiguous(mask)) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || fa->type != GGML_TYPE_F32) {
        return false;
    }
    return ggml_nelements(mask) <= INT32_MAX;
}

// The union kernel takes what the chunked oneMKL kernel would take, so it and mode 0 reach the
// same node with the same GEMMs; everything else keeps the dense kernels.
static bool qsa_union_applicable(const ggml_tensor * fa, int & kind) {
    const ggml_tensor * Q = fa->src[0];
    const ggml_tensor * K = fa->src[1];
    const ggml_tensor * V = fa->src[2];
    if (!ggml_sycl_fattn_picks_mkl(fa) || fa->src[4]) {
        return false;
    }
    const int64_t D = Q->ne[0];
    if (D % 32 != 0 || Q->nb[0] != sizeof(float) || !ggml_is_contiguous(fa)) {
        return false;
    }
    if (K->type != V->type || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0)) {
        return false;
    }
    if (K->ne[0] != D || V->ne[0] != D || K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3] || V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2]) {
        return false;
    }
    kind = QSA_KV_F16;
    if (K->type == GGML_TYPE_Q8_0) {
        const bool soa = ggml_sycl_kv_is_soa(K);
        if (soa != ggml_sycl_kv_is_soa(V)) {
            return false;
        }
        if (soa && D % GGML_SYCL_KV_SOA_SPAN != 0) {
            return false;
        }
        kind = soa ? QSA_KV_Q8_0_SOA : QSA_KV_Q8_0;
    }
    return (int64_t) (Q->ne[3] * Q->ne[1] * Q->ne[2]) <= INT32_MAX / D;
}

void ggml_sycl_qsa_sparse_fa(ggml_backend_sycl_context & ctx, ggml_tensor * fa, const ggml_tensor * mask,
                             const ggml_tensor * idx) {
    GGML_ASSERT(ggml_sycl_qsa_sparse_fa_supported(fa));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && ggml_are_same_shape(mask, fa->src[3]) && ggml_is_contiguous(mask));
    GGML_ASSERT(idx->type == GGML_TYPE_I32 && idx->nb[0] == sizeof(int32_t));
    GGML_ASSERT(idx->ne[1] == fa->src[0]->ne[1] && idx->ne[2] == fa->src[0]->ne[3]);
    GGML_ASSERT(mask->data && idx->data);

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    int kind = QSA_KV_F16;
    const bool take = qsa_union_applicable(fa, kind);

    static std::atomic<int> trace_left{ getenv("GGML_SYCL_QSA_MASK_TRACE") ?
                                        std::max(1, atoi(getenv("GGML_SYCL_QSA_MASK_TRACE"))) : 0 };
    const bool small = take && fa->src[1]->ne[1] <= QSA_UNION_MIN_KV;
    if (trace_left.fetch_sub(1) > 0) {
        fprintf(stderr, "[QSASPARSE] n_kv=%ld n_tps=%ld heads=%ld/%ld width=%ld path=%s kind=%d\n",
                (long) mask->ne[0], (long) mask->ne[1], (long) fa->src[0]->ne[2], (long) fa->src[1]->ne[2],
                (long) idx->ne[0], !take ? "dense" : small ? "bits" : "union", kind);
    }

    if (!take) {
        qsa_dense_scratch_fa(ctx, fa, mask, idx);
        return;
    }

    const ggml_tensor * Q = fa->src[0];
    const ggml_tensor * K = fa->src[1];
    const ggml_tensor * V = fa->src[2];

    qsa_union_args a = {};
    a.Q     = (const char *) Q->data;
    a.q_nb1 = Q->nb[1];
    a.q_nb2 = Q->nb[2];
    a.q_nb3 = Q->nb[3];
    a.K     = (const char *) K->data;
    a.k_nb1 = K->nb[1];
    a.k_nb2 = K->nb[2];
    a.k_nb3 = K->nb[3];
    a.V     = (const char *) V->data;
    a.v_nb1 = V->nb[1];
    a.v_nb2 = V->nb[2];
    a.v_nb3 = V->nb[3];
    a.mask  = (const char *) mask->data;
    a.m_nb1 = mask->nb[1];
    a.m_nb3 = mask->nb[3];
    a.idx   = (const char *) idx->data;
    a.i_nb1 = idx->nb[1];
    a.i_nb2 = idx->nb[2];
    a.dst   = (float *) fa->data;

    a.n_kv      = K->ne[1];
    a.n_tps     = Q->ne[1];
    a.n_stream  = Q->ne[3];
    a.width     = idx->ne[0];
    a.D         = (int) Q->ne[0];
    a.n_head    = (int) Q->ne[2];
    a.n_head_kv = (int) K->ne[2];
    memcpy(&a.scale, (const float *) fa->op_params + 0, sizeof(float));

    GGML_ASSERT(a.Q && a.K && a.V && a.dst);

    if (small) {
        qsa_bits_mkl_fa(ctx, fa, mask, a);
        return;
    }
    switch (kind) {
        case QSA_KV_F16:
            qsa_union_fa<QSA_KV_F16>(ctx, a);
            break;
        case QSA_KV_Q8_0:
            qsa_union_fa<QSA_KV_Q8_0>(ctx, a);
            break;
        default:
            qsa_union_fa<QSA_KV_Q8_0_SOA>(ctx, a);
            break;
    }
}
