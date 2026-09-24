#include "qsa-mask.hpp"
#include "kq-mask-bits.hpp"

#include "ggml-impl.h"

#include "common.hpp"
#include "fattn.hpp"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <vector>

// QSA attends only to the cells that the indexer picked. The model graph says that with
//   FILL(kq_mask, -inf) -> VIEW -> SET_ROWS(0 at top_k) -> VIEW -> ADD(kq_mask)
// which materialises a second full [n_kv, n_tps] mask, 256 MiB at 128k context, before the
// ADD writes the one that flash attention reads. Because -inf + x == -inf and 0 + x == x, the
// chain is exactly
//   out[c, t] = selected(c, t) ? kq_mask[c, t] : -inf
// so a fill plus a scatter writes the ADD's buffer alone. The FILL is reported through
// fusion_absorbs, so ggml-alloc reserves nothing for it. Every test below is structural, so
// the answer is the same before and after allocation.
//
// GGML_SYCL_FUSE_QSA_FA_MASK goes further and never builds the dense mask at all: the ADD is
// absorbed too, and the reader - only the chunked oneMKL kernel, which is the one we can teach -
// takes the causal mask plus a one-bit-per-cell selection bitmap. That drops 256 MiB from the
// reserve and cuts what the softmax reads per cell from 16 bits to 1.
//   mode 1: the bit is selection only, the kernel still adds the causal mask
//   mode 2: the bit is selection AND the causal mask, which the kernel then never reads
// The bitmap is built HERE, at the chain, and not at the reader: ggml-alloc releases the top-k
// list at the SET_ROWS node, so by the time the reader runs the list may already be gone.
//
// The set_rows payload is a separate FILL of zeros that the graph emits far from the chain,
// so it is reached through src[] and left alone; the fusion never reads it.

static constexpr int SYCL_QSA_MASK_SPAN = 4;

struct qsa_mask_chain {
    const ggml_tensor * fill;  // FILL -inf, the copy of the mask that the fusion never writes
    const ggml_tensor * mask;  // the causal kq_mask
    const ggml_tensor * idx;   // the top-k index list
    ggml_tensor *       add;   // the only node of the chain that keeps a buffer
};

// ggml_node_get_use_count() keyed by tensor, so the chain can be walked through src[] alone
static int32_t qsa_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

// The whole structural test, reached from the ADD through src[] only. Graph order never enters,
// which matters because the zeros FILL sits about 53 nodes before the rest of the chain.
static bool qsa_mask_chain_from_add(const ggml_cgraph * cgraph, ggml_tensor * ad, qsa_mask_chain * out) {
    if (!ad || ad->op != GGML_OP_ADD) {
        return false;
    }

    ggml_tensor * rv   = ad->src[0];  // back to the mask shape
    ggml_tensor * mask = ad->src[1];
    if (!rv || !mask || rv->op != GGML_OP_VIEW) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        return false;
    }
    if (mask->ne[2] != 1 || mask->ne[0] > INT32_MAX || !ggml_is_contiguous(mask)) {
        return false;
    }

    ggml_tensor * fm = rv->view_src;   // FILL -inf
    ggml_tensor * sr = rv->src[0];     // SET_ROWS
    if (!fm || !sr || fm->op != GGML_OP_FILL || sr->op != GGML_OP_SET_ROWS) {
        return false;
    }

    ggml_tensor * fz = sr->src[0];     // the zeros payload
    ggml_tensor * iv = sr->src[1];     // the top-k index list
    ggml_tensor * mv = sr->src[2];     // the mask copy seen as rows of one cell
    if (!fz || !iv || !mv || mv->op != GGML_OP_VIEW) {
        return false;
    }

    // the fill is fused away, so nothing outside the chain may read it
    if (fm->src[0] != mask || fm->view_src || (fm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(fm)) {
        return false;
    }
    if (fm->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (!ggml_is_contiguous(fm) || fm->type != mask->type || !ggml_are_same_shape(fm, mask)) {
        return false;
    }
    for (const ggml_tensor * t : { fm, mv, sr, rv }) {
        if (qsa_use_count(cgraph, t) != 1) {
            return false;
        }
    }

    // -inf everywhere plus 0 at the picked cells is a select; any other pair of values is not
    const float v_fill = ggml_get_op_params_f32(fm, 0);
    if (!std::isinf(v_fill) || v_fill > 0.0f) {
        return false;
    }
    if (fz->op != GGML_OP_FILL || ggml_get_op_params_f32(fz, 0) != 0.0f) {
        return false;
    }

    const int64_t n_kv     = mask->ne[0];
    const int64_t n_tps    = mask->ne[1];
    const int64_t n_stream = mask->ne[3];
    const int64_t width    = iv->ne[0];

    if (iv->type != GGML_TYPE_I32 || iv->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (width < 1 || iv->ne[1] != n_tps || iv->ne[2] != n_stream || iv->ne[3] != 1) {
        return false;
    }
    if (fz->ne[0] != 1 || fz->ne[1] != width || fz->ne[2] != n_tps || fz->ne[3] != n_stream) {
        return false;
    }

    if (mv->view_src != fm || mv->src[0] != fm || mv->view_offs != 0 || !ggml_is_contiguous(mv)) {
        return false;
    }
    if (mv->ne[0] != 1 || mv->ne[1] != n_kv || mv->ne[2] != n_tps || mv->ne[3] != n_stream) {
        return false;
    }
    if (sr->view_src != fm || sr->view_offs != 0 || !ggml_are_same_shape(sr, mv)) {
        return false;
    }
    if (rv->view_src != fm || rv->view_offs != 0 || !ggml_is_contiguous(rv)) {
        return false;
    }
    if (!ggml_are_same_shape(rv, mask)) {
        return false;
    }

    if (ad->type != mask->type || ad->view_src || ggml_is_empty(ad) || !ggml_is_contiguous(ad)) {
        return false;
    }
    if (!ggml_are_same_shape(ad, mask) || (ad->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }

    if (out) {
        *out = { fm, mask, iv, ad };
    }
    return true;
}

static bool ggml_sycl_qsa_mask_shape(const ggml_cgraph * cgraph, int i, qsa_mask_chain * out) {
    if (i < 0 || i + SYCL_QSA_MASK_SPAN >= cgraph->n_nodes) {
        return false;
    }

    qsa_mask_chain c;
    if (!qsa_mask_chain_from_add(cgraph, cgraph->nodes[i + SYCL_QSA_MASK_SPAN], &c)) {
        return false;
    }
    // the five nodes the model emits together, in the order it emits them
    const ggml_tensor * rv = c.add->src[0];
    const ggml_tensor * sr = rv->src[0];
    if (cgraph->nodes[i] != c.fill || cgraph->nodes[i + 1] != sr->src[2] ||
        cgraph->nodes[i + 2] != sr || cgraph->nodes[i + 3] != rv) {
        return false;
    }

    if (out) {
        *out = c;
    }
    return true;
}

// Everything the reader side needs except where the chain sits in the graph.
static bool qsa_fa_mask_core(const ggml_cgraph * cgraph, ggml_tensor * fa, qsa_mask_chain * out) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_fa_mask) {
        return false;
    }
    if (!fa || fa->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    ggml_tensor * ad = fa->src[3];
    if (!ad || ad->type != GGML_TYPE_F16 || qsa_use_count(cgraph, ad) != 1) {
        return false;
    }
    if (!qsa_mask_chain_from_add(cgraph, ad, out)) {
        return false;
    }
    // A packed causal mask is itself a bitmap, and this fusion would hand oneMKL a second one
    // with different semantics, so the two features must not combine.
    if (ggml_sycl_kq_mask_is_bits(ad) || ggml_sycl_kq_mask_is_bits(fa->src[3])) {
        return false;
    }
    // only the chunked oneMKL kernel can take a bitmap; every other kernel must keep the dense mask
    return ggml_sycl_fattn_picks_mkl(fa);
}

// The flash attention node that reads the chain starting at i_fill, or -1.
static int qsa_fa_mask_reader(const ggml_cgraph * cgraph, int i_fill) {
    qsa_mask_chain c;
    if (!g_ggml_sycl_fuse_qsa_fa_mask || !ggml_sycl_qsa_mask_shape(cgraph, i_fill, &c)) {
        return -1;
    }
    for (int j = i_fill + SYCL_QSA_MASK_SPAN + 1; j < cgraph->n_nodes; j++) {
        if (cgraph->nodes[j]->src[3] == c.add && qsa_fa_mask_core(cgraph, cgraph->nodes[j], nullptr)) {
            return j;
        }
    }
    return -1;
}

// The same relation seen from the reader: the chain must be in this graph, and before it.
static bool qsa_fa_mask_located(const ggml_cgraph * cgraph, int i_fa, qsa_mask_chain * out) {
    qsa_mask_chain c;
    if (!qsa_fa_mask_core(cgraph, cgraph->nodes[i_fa], &c)) {
        return false;
    }
    for (int j = i_fa - 1; j >= SYCL_QSA_MASK_SPAN; j--) {
        if (cgraph->nodes[j] == c.add) {
            if (!ggml_sycl_qsa_mask_shape(cgraph, j - SYCL_QSA_MASK_SPAN, nullptr)) {
                return false;
            }
            if (out) {
                *out = c;
            }
            return true;
        }
    }
    return false;
}

// The fused kernel writes the ADD's buffer at the FILL's position, so the fill AND the three
// views between them are reported: ggml-alloc then frees nothing inside the span (the top-k list
// and the zeros are released at the ADD, after its buffer is placed) and the ADD cannot land on
// memory the kernel still reads. When flash attention takes the index list the ADD goes too.
int ggml_sycl_qsa_mask_absorbs(const ggml_cgraph * cgraph, int node_idx) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }
    if (g_ggml_sycl_fuse_qsa_fa_mask) {
        if (qsa_fa_mask_reader(cgraph, node_idx) >= 0) {
            return SYCL_QSA_MASK_SPAN + 1;  // fill, three views, add
        }
    }
    if (!g_ggml_sycl_fuse_qsa_mask) {
        return 0;
    }
    return ggml_sycl_qsa_mask_shape(cgraph, node_idx, nullptr) ? SYCL_QSA_MASK_SPAN : 0;
}

template <typename T>
static void k_qsa_mask_drop(T * dst, int64_t n, T value, const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t) item.get_global_id(0);
    if (i < n) {
        dst[i] = value;
    }
}

// keep the cells that row (t, s) of the index list names; rows are contiguous in both tensors
template <typename T>
static void k_qsa_mask_keep(const T * mask, T * dst, const char * idx, int64_t n_kv, int64_t n_tps,
                            int64_t width, size_t nb1, size_t nb2, const sycl::nd_item<2> & item) {
    const int64_t w = (int64_t) item.get_global_id(1);
    if (w >= width) {
        return;
    }

    const int64_t r = (int64_t) item.get_global_id(0);
    const int64_t t = r % n_tps;
    const int64_t s = r / n_tps;

    const int64_t c = *(const int32_t *) (idx + w*sizeof(int32_t) + t*nb1 + s*nb2);
    if (c < 0 || c >= n_kv) {
        return;
    }

    dst[r*n_kv + c] = mask[r*n_kv + c];
}

template <typename T>
static void qsa_mask_launch(dpct::queue_ptr stream, const void * mask_v, void * dst_v, const char * idx,
                            int64_t n_all, int64_t n_kv, int64_t n_tps, int64_t n_rows, int64_t width,
                            size_t nb1, size_t nb2, float value) {
    constexpr int block = 256;

    const T * mask = (const T *) mask_v;
    T *       dst  = (T *) dst_v;
    const T   val  = static_cast<T>(value);

    stream->parallel_for(
        sycl::nd_range<1>(((n_all + block - 1) / block) * block, block),
        [=](sycl::nd_item<1> item) { k_qsa_mask_drop(dst, n_all, val, item); });

    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(n_rows, ((width + block - 1) / block) * block), sycl::range<2>(1, block)),
        [=](sycl::nd_item<2> item) { k_qsa_mask_keep(mask, dst, idx, n_kv, n_tps, width, nb1, nb2, item); });
}

// One bit per (row, kv cell), 1 keeps the cell. Built from the top-k list, so it touches width
// cells per row and not n_kv. fold also requires the causal mask to be finite, which is what
// lets the reader skip the dense mask entirely.
static void k_qsa_sel_bits(const char * mask, const char * idx, uint32_t * bits,
                           int64_t n_kv, int64_t n_tps, int64_t width, int64_t words,
                           size_t m_nb1, size_t m_nb3, size_t i_nb1, size_t i_nb2,
                           int fold, const sycl::nd_item<2> & item) {
    const int64_t w = (int64_t) item.get_global_id(1);
    if (w >= width) {
        return;
    }

    const int64_t r = (int64_t) item.get_global_id(0);
    const int64_t t = r % n_tps;
    const int64_t s = r / n_tps;

    const int64_t c = *(const int32_t *) (idx + w*sizeof(int32_t) + t*i_nb1 + s*i_nb2);
    if (c < 0 || c >= n_kv) {
        return;
    }
    if (fold) {
        const sycl::half m = ((const sycl::half *) (mask + t*m_nb1 + s*m_nb3))[c];
        if (!sycl::isfinite((float) m)) {
            return;
        }
    }

    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(bits[r*words + (c >> 5)]);
    ref.fetch_or(1u << (c & 31));
}


// counts cells whose causal mask is finite but not zero; mode 2 folds the mask into one bit and
// is only exact while this is zero
static void k_qsa_mask_oddballs(const char * mask, int64_t n_kv, size_t m_nb1, int32_t * count,
                                const sycl::nd_item<2> & item) {
    const int64_t c = (int64_t) item.get_global_id(1);
    if (c >= n_kv) {
        return;
    }
    const int64_t r = (int64_t) item.get_global_id(0);
    const float   m = (float) ((const sycl::half *) (mask + r*m_nb1))[c];
    if (m != 0.0f && sycl::isfinite(m)) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> ref(*count);
        ref.fetch_add(1);
    }
}

static void qsa_sel_stats(ggml_backend_sycl_context & ctx, const qsa_mask_chain & c, const uint32_t * bits,
                          int64_t n_kv, int64_t n_tps, int64_t n_rows, int64_t words) {
    dpct::queue_ptr stream = ctx.stream();

    ggml_sycl_pool_alloc<int32_t> cnt(ctx.pool(), 1);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(cnt.get(), 0, sizeof(int32_t))));
    {
        constexpr int wg = 256;
        const char *  md = (const char *) c.mask->data;
        const size_t  nb1 = c.mask->nb[1];
        int32_t *     cp  = cnt.get();
        const int64_t nkv = n_kv;
        stream->parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_rows, ((n_kv + wg - 1) / wg) * wg), sycl::range<2>(1, wg)),
            [=](sycl::nd_item<2> item) { k_qsa_mask_oddballs(md, nkv, nb1, cp, item); });
    }

    std::vector<uint32_t> h((size_t) n_rows * words);
    int32_t               oddballs = 0;
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(h.data(), bits, h.size()*sizeof(uint32_t))));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(&oddballs, cnt.get(), sizeof(int32_t))));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    fprintf(stderr, "[QSASTATS] n_kv=%ld n_tps=%ld rows=%ld mask_finite_nonzero=%d\n",
            (long) n_kv, (long) n_tps, (long) n_rows, (int) oddballs);

    std::vector<uint32_t> u((size_t) words);
    for (int64_t B : { (int64_t) 1, (int64_t) 8, (int64_t) 32, (int64_t) 64, (int64_t) 160, n_tps }) {
        if (B > n_rows) {
            continue;
        }
        double  ratio = 0.0, chunk_hit = 0.0;
        int64_t blocks = 0, chunks = 0, hit = 0;
        for (int64_t r0 = 0; r0 + B <= n_rows; r0 += B) {
            std::fill(u.begin(), u.end(), 0u);
            for (int64_t r = r0; r < r0 + B; r++) {
                for (int64_t w = 0; w < words; w++) {
                    u[w] |= h[(size_t) r*words + w];
                }
            }
            int64_t set = 0;
            for (int64_t w = 0; w < words; w++) {
                set += __builtin_popcount(u[w]);
            }
            ratio += (double) set / (double) n_kv;
            blocks++;
            for (int64_t w0 = 0; w0 < words; w0 += 256) {  // 256 words = one 8192 cell chunk
                bool any = false;
                for (int64_t w = w0; w < std::min(w0 + 256, words); w++) {
                    any |= u[w] != 0;
                }
                chunks++;
                hit += any;
            }
        }
        chunk_hit = chunks ? (double) hit / (double) chunks : 0.0;
        fprintf(stderr, "[QSASTATS] block=%-5ld union=%.4f  chunks_touched=%.4f  (%ld blocks)\n",
                (long) B, blocks ? ratio / blocks : 0.0, chunk_hit, (long) blocks);
    }
}

// Built at the chain, where the top-k list is still live, and handed to the reader through the
// context. The pool block is keyed by the ADD, which is the tensor the reader still points at.
static void qsa_sel_bits_build(ggml_backend_sycl_context & ctx, const ggml_tensor * fa, const qsa_mask_chain & c) {
    const int64_t n_kv   = fa->src[1]->ne[1];
    const int64_t n_tps  = c.mask->ne[1];
    const int64_t n_rows = n_tps * c.mask->ne[3];
    const int64_t width  = c.idx->ne[0];
    const int64_t words  = (n_kv + 31) / 32;

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    size_t     have = 0;
    uint32_t * bits = (uint32_t *) ctx.pool().alloc((size_t) n_rows * words * sizeof(uint32_t), &have);
    ctx.qsa_sel_put(c.add, bits, have);

    const char * mask_dd = (const char *) c.mask->data;
    const char * idx_dd  = (const char *) c.idx->data;
    GGML_ASSERT(mask_dd && idx_dd);

    const int fold = g_ggml_sycl_fuse_qsa_fa_mask >= 2;

    stream->memset(bits, 0, (size_t) n_rows * words * sizeof(uint32_t));

    constexpr int block = 256;
    const size_t  m_nb1 = c.mask->nb[1];
    const size_t  m_nb3 = c.mask->nb[3];
    const size_t  i_nb1 = c.idx->nb[1];
    const size_t  i_nb2 = c.idx->nb[2];

    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(n_rows, ((width + block - 1) / block) * block), sycl::range<2>(1, block)),
        [=](sycl::nd_item<2> item) {
            k_qsa_sel_bits(mask_dd, idx_dd, bits, n_kv, n_tps, width, words, m_nb1, m_nb3, i_nb1, i_nb2, fold, item);
        });

    // GGML_SYCL_QSA_SEL_STATS is the smallest n_kv worth reporting on, because the answer only
    // matters at a long context. It answers two questions off the live data: how much of the
    // context a block of neighbouring tokens selects between them (which is what decides whether
    // a tile of the flash attention loop could ever be skipped), and whether the causal mask
    // really only ever holds 0 or -inf (which is what mode 2 relies on).
    static const int    stats_min  = ggml_sycl_get_env("GGML_SYCL_QSA_SEL_STATS", 0);
    static std::atomic<int> stats_left{ 4 };
    if (stats_min > 0 && n_kv >= stats_min && stats_left.fetch_sub(1) > 0) {
        qsa_sel_stats(ctx, c, bits, n_kv, n_tps, n_rows, words);
    }
}

// Runs the chain matched by ggml_sycl_qsa_mask_shape(); returns the extra nodes consumed.
int ggml_sycl_fuse_qsa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    qsa_mask_chain c;
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }

    // the dense mask is never built when flash attention takes the bitmap; leave it the bits
    if (g_ggml_sycl_fuse_qsa_fa_mask) {
        const int i_fa = qsa_fa_mask_reader(cgraph, i);
        if (i_fa >= 0) {
            ggml_sycl_qsa_mask_shape(cgraph, i, &c);
            qsa_sel_bits_build(ctx, cgraph->nodes[i_fa], c);
            return SYCL_QSA_MASK_SPAN;
        }
    }

    if (!g_ggml_sycl_fuse_qsa_mask || !ggml_sycl_qsa_mask_shape(cgraph, i, &c)) {
        return 0;
    }

    const ggml_tensor * fm   = c.fill;
    const ggml_tensor * mask = c.mask;
    const ggml_tensor * idx  = c.idx;
    ggml_tensor *       dst  = c.add;

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_tps    = dst->ne[1];
    const int64_t n_stream = dst->ne[3];
    const int64_t width    = idx->ne[0];

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const char * idx_dd  = (const char *) idx->data;
    const void * mask_dd = mask->data;
    void *       dst_dd  = dst->data;

    // GGML_SYCL_QSA_MASK_TRACE gives the number of firings to report
    static std::atomic<int> trace_left{ getenv("GGML_SYCL_QSA_MASK_TRACE") ?
                                        std::max(1, atoi(getenv("GGML_SYCL_QSA_MASK_TRACE"))) : 0 };
    if (trace_left.fetch_sub(1) > 0) {
        fprintf(stderr, "[QSAMASK] n_kv=%ld n_tps=%ld n_stream=%ld width=%ld type=%s\n", (long) n_kv, (long) n_tps,
                (long) n_stream, (long) width, ggml_type_name(dst->type));
    }

    GGML_ASSERT(idx_dd && mask_dd && dst_dd);
    GGML_ASSERT(mask_dd != dst_dd);

    const int64_t n_all  = ggml_nelements(dst);
    const int64_t n_rows = n_tps * n_stream;
    const float   value  = ggml_get_op_params_f32(fm, 0);

    if (dst->type == GGML_TYPE_F16) {
        qsa_mask_launch<sycl::half>(stream, mask_dd, dst_dd, idx_dd, n_all, n_kv, n_tps, n_rows, width,
                                    idx->nb[1], idx->nb[2], value);
    } else {
        qsa_mask_launch<float>(stream, mask_dd, dst_dd, idx_dd, n_all, n_kv, n_tps, n_rows, width,
                               idx->nb[1], idx->nb[2], value);
    }

    return SYCL_QSA_MASK_SPAN;
}


bool ggml_sycl_qsa_fa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    qsa_mask_chain c;
    if (!qsa_fa_mask_located(cgraph, i, &c)) {
        return false;
    }

    // the same matcher decided the dense mask was never built, so a miss here means the absorb
    // side and the compute side disagreed - which must be loud, not a silent dense read
    uint32_t * bits = (uint32_t *) ctx.qsa_sel_find(c.add);
    GGML_ASSERT(bits && "QSA FA mask: no selection bitmap for this flash attention node");

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    // the reader gets the causal mask in place of the mask that was never built
    ggml_tensor fa = *cgraph->nodes[i];
    fa.src[3] = const_cast<ggml_tensor *>(c.mask);

    const int64_t words = (fa.src[1]->ne[1] + 31) / 32;

    static std::atomic<int> trace_left{ getenv("GGML_SYCL_QSA_MASK_TRACE") ?
                                        std::max(1, atoi(getenv("GGML_SYCL_QSA_MASK_TRACE"))) : 0 };
    if (trace_left.fetch_sub(1) > 0) {
        fprintf(stderr, "[QSAFA] n_kv=%ld n_tps=%ld width=%ld words=%ld mode=%d held=%zu high=%zu\n",
                (long) fa.src[1]->ne[1], (long) c.mask->ne[1], (long) c.idx->ne[0], (long) words,
                g_ggml_sycl_fuse_qsa_fa_mask, ctx.qsa_sel.size(), ctx.qsa_sel_high_water);
    }

    ggml_sycl_flash_attn_ext_mkl(ctx, &fa, bits, words, g_ggml_sycl_fuse_qsa_fa_mask >= 2 ? 2 : 1);

    ctx.qsa_sel_drop(c.add);
    return true;
}
