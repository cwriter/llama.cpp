#include "qsa-mask.hpp"

#include "ggml-impl.h"

#include "common.hpp"

#include <atomic>
#include <cmath>

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
// The set_rows payload is a separate FILL of zeros that the graph emits far from the chain,
// so it is reached through src[] and left alone; the fusion never reads it.

static constexpr int SYCL_QSA_MASK_SPAN = 4;

struct qsa_mask_chain {
    int i_fill;  // FILL -inf, the copy of the mask that the fusion never writes
    int i_out;   // the ADD, the only node of the chain that keeps a buffer
};

static bool ggml_sycl_qsa_mask_shape(const ggml_cgraph * cgraph, int i, qsa_mask_chain * out) {
    if (i + SYCL_QSA_MASK_SPAN >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * fm = cgraph->nodes[i];      // FILL -inf
    ggml_tensor * mv = cgraph->nodes[i + 1];  // the mask copy seen as rows of one cell
    ggml_tensor * sr = cgraph->nodes[i + 2];  // SET_ROWS
    ggml_tensor * rv = cgraph->nodes[i + 3];  // back to the mask shape
    ggml_tensor * ad = cgraph->nodes[i + 4];  // ADD

    if (fm->op != GGML_OP_FILL || mv->op != GGML_OP_VIEW || sr->op != GGML_OP_SET_ROWS ||
        rv->op != GGML_OP_VIEW || ad->op != GGML_OP_ADD) {
        return false;
    }

    const ggml_tensor * mask = fm->src[0];
    if (!mask || ad->src[0] != rv || ad->src[1] != mask) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        return false;
    }
    if (mask->ne[2] != 1 || mask->ne[0] > INT32_MAX || !ggml_is_contiguous(mask)) {
        return false;
    }

    // the fill is fused away, so nothing outside the chain may read it
    if (fm->view_src || (fm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(fm)) {
        return false;
    }
    if (fm->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (!ggml_is_contiguous(fm) || fm->type != mask->type || !ggml_are_same_shape(fm, mask)) {
        return false;
    }
    for (int n : { i, i + 1, i + 2, i + 3 }) {
        if (ggml_node_get_use_count(cgraph, n) != 1) {
            return false;
        }
    }

    // -inf everywhere plus 0 at the picked cells is a select; any other pair of values is not
    const float v_fill = ggml_get_op_params_f32(fm, 0);
    if (!std::isinf(v_fill) || v_fill > 0.0f) {
        return false;
    }

    const ggml_tensor * fz = sr->src[0];
    const ggml_tensor * iv = sr->src[1];
    if (!fz || !iv || sr->src[2] != mv) {
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
    if (rv->view_src != fm || rv->src[0] != sr || rv->view_offs != 0 || !ggml_is_contiguous(rv)) {
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
        *out = { i, i + SYCL_QSA_MASK_SPAN };
    }
    return true;
}

// Only the fill loses its buffer. The three views keep their normal bookkeeping, which is what
// makes ggml-alloc release the mask at the ADD instead of stranding it.
int ggml_sycl_qsa_mask_absorbs(const ggml_cgraph * cgraph, int node_idx) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_mask) {
        return 0;
    }
    return ggml_sycl_qsa_mask_shape(cgraph, node_idx, nullptr) ? 1 : 0;
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

// Runs the chain matched by ggml_sycl_qsa_mask_shape(); returns the extra nodes consumed.
int ggml_sycl_fuse_qsa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    qsa_mask_chain c;
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_mask || !ggml_sycl_qsa_mask_shape(cgraph, i, &c)) {
        return 0;
    }

    const ggml_tensor * fm   = cgraph->nodes[c.i_fill];
    const ggml_tensor * mask = fm->src[0];
    const ggml_tensor * idx  = cgraph->nodes[c.i_fill + 2]->src[1];
    ggml_tensor *       dst  = cgraph->nodes[c.i_out];

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

    return c.i_out - i;
}
