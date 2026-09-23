#include "kv-soa.hpp"

bool ggml_sycl_kv_soa_eligible(const ggml_tensor * t) {
    if (!g_ggml_sycl_kv_soa || t == nullptr || t->type != GGML_TYPE_Q8_0) {
        return false;
    }
    // must own its bytes and tile exactly into spans
    return t->view_src == nullptr && t->ne[0] % GGML_SYCL_KV_SOA_SPAN == 0 && t->ne[3] == 1;
}

bool ggml_sycl_kv_is_soa(const ggml_tensor * t) {
    if (!g_ggml_sycl_kv_soa || t == nullptr || t->type != GGML_TYPE_Q8_0) {
        return false;
    }
    const ggml_tensor * owner = t;
    while (owner->view_src) {
        owner = owner->view_src;
    }
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(owner->extra);
    if (!extra || extra->optimized_feature.layout.kind != GGML_SYCL_LAYOUT_SOA_SPAN) {
        return false;
    }
    // a view may slice rows, heads and streams, but never part of a span
    GGML_ASSERT(t->ne[0] % GGML_SYCL_KV_SOA_SPAN == 0 &&
                "SoA KV cache viewed with a partial span; see GGML_SYCL_KV_SOA_SPAN in kv-soa.hpp");
    return true;
}

void ggml_sycl_kv_soa_mark(ggml_tensor * t) {
    if (!ggml_sycl_kv_soa_eligible(t) || t->extra == nullptr) {
        return;
    }
    auto & layout = static_cast<ggml_tensor_extra_gpu *>(t->extra)->optimized_feature.layout;
    layout.kind   = GGML_SYCL_LAYOUT_SOA_SPAN;
    layout.type   = GGML_TYPE_Q8_0;
    layout.span   = GGML_SYCL_KV_SOA_SPAN;
}

// one work item per block; the span base comes from the view's own strides, so this serves the
// owning cache and any head/row/stream slice of it alike
static void kv_soa_dequant_f16(const char * __restrict__ vx, sycl::half * __restrict__ y,
                               const int64_t nblk_row, const int64_t ne1, const int64_t total,
                               const size_t nb1, const size_t nb2, const int64_t ne0,
                               const sycl::nd_item<1> & item) {
    using A = ggml_sycl_q8_0_access<GGML_SYCL_LAYOUT_SOA_SPAN>;

    const int64_t ib = (int64_t) item.get_global_linear_id();
    if (ib >= total) {
        return;
    }
    const int64_t slice = ib / nblk_row;             // which (i1, i2) row of the view
    const int64_t iblk  = ib - slice * nblk_row;     // block index inside that row
    const int64_t i1    = slice % ne1;
    const int64_t i2    = slice / ne1;

    size_t span_off;
    int    iblk_in_span;
    ggml_sycl_q8_0_locate<GGML_SYCL_LAYOUT_SOA_SPAN>(iblk * QK8_0, span_off, iblk_in_span);

    const char *   base = vx + i2 * nb2 + i1 * nb1 + span_off;
    const int8_t * qs   = A::qs(base, iblk_in_span);
    const float    d    = A::d(base, iblk_in_span);

    sycl::half * out = y + (i2 * ne1 + i1) * ne0 + iblk * QK8_0;
    for (int j = 0; j < QK8_0; ++j) {
        out[j] = (sycl::half) (d * (float) qs[j]);
    }
}

void ggml_sycl_kv_soa_to_fp16(const void * vx, sycl::half * y, int64_t ne0, int64_t ne1,
                              int64_t ne2, size_t nb1, size_t nb2, queue_ptr stream) {
    const int64_t nblk_row = ne0 / QK8_0;
    const int64_t total    = nblk_row * ne1 * ne2;

    constexpr int block = 256;
    const size_t  grid  = (size_t) ((total + block - 1) / block);

    stream->parallel_for(sycl::nd_range<1>(grid * block, block), [=](sycl::nd_item<1> item) {
        kv_soa_dequant_f16((const char *) vx, y, nblk_row, ne1, total, nb1, nb2, ne0, item);
    });
}
