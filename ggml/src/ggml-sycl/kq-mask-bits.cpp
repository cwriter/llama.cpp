#include "kq-mask-bits.hpp"

#include <cstring>

bool ggml_sycl_kq_mask_eligible(const ggml_tensor * t) {
    if (!g_ggml_sycl_kq_mask_bits || t == nullptr || t->type != GGML_TYPE_F16) {
        return false;
    }
    if (t->view_src != nullptr || !ggml_is_contiguous(t)) {
        return false;
    }
    // the scheduler names its copy "<backend>#<src name>#<n>", so match anywhere in the name
    if (strstr(t->name, "kq_mask") == nullptr) {
        return false;
    }
    // a packed row must fit inside the f16 row it replaces
    return t->ne[0] >= 32 && ggml_sycl_kq_mask_row_bytes(t->ne[0]) <= (size_t) t->ne[0] * sizeof(sycl::half);
}

bool ggml_sycl_kq_mask_is_bits(const ggml_tensor * t) {
    if (!g_ggml_sycl_kq_mask_bits || t == nullptr || t->type != GGML_TYPE_F16) {
        return false;
    }
    const ggml_tensor * owner = t;
    while (owner->view_src) {
        owner = owner->view_src;
    }
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(owner->extra);
    return extra && extra->optimized_feature.layout.kind == GGML_SYCL_LAYOUT_MASK_BITS;
}

void ggml_sycl_kq_mask_mark(ggml_tensor * t) {
    if (!ggml_sycl_kq_mask_eligible(t)) {
        return;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(t->extra);
    if (!extra) {
        return;
    }
    extra->optimized_feature.layout.kind = GGML_SYCL_LAYOUT_MASK_BITS;
    extra->optimized_feature.layout.type = GGML_TYPE_F16;
    extra->optimized_feature.layout.span = 32;
}

// -inf is the only value that masks; everything else the producer writes is a keep (it writes 0)
static inline bool kq_mask_keep(uint16_t h) {
    return h != 0xFC00u;  // f16 -inf
}

void ggml_sycl_kq_mask_pack(void * vdst, const void * vsrc, int64_t ne0, int64_t nrows) {
    const uint16_t * src   = (const uint16_t *) vsrc;
    char *           dst   = (char *) vdst;
    const size_t     rbyte = ggml_sycl_kq_mask_row_bytes(ne0);
    for (int64_t r = 0; r < nrows; ++r) {
        uint32_t *       out = (uint32_t *) (dst + (size_t) r * rbyte);
        const uint16_t * in  = src + (size_t) r * ne0;
        memset(out, 0, rbyte);
        for (int64_t c = 0; c < ne0; ++c) {
            if (kq_mask_keep(in[c])) {
                out[c >> 5] |= 1u << (c & 31);
            }
        }
    }
}

void ggml_sycl_kq_mask_unpack(void * vdst, const void * vsrc, int64_t ne0, int64_t nrows) {
    uint16_t *   dst   = (uint16_t *) vdst;
    const char * src   = (const char *) vsrc;
    const size_t rbyte = ggml_sycl_kq_mask_row_bytes(ne0);
    for (int64_t r = 0; r < nrows; ++r) {
        const uint32_t * in  = (const uint32_t *) (src + (size_t) r * rbyte);
        uint16_t *       out = dst + (size_t) r * ne0;
        for (int64_t c = 0; c < ne0; ++c) {
            out[c] = ((in[c >> 5] >> (c & 31)) & 1u) ? 0x0000u : 0xFC00u;
        }
    }
}

void ggml_sycl_kq_mask_to_f16(const void * vbits, sycl::half * y, int64_t ne0, int64_t nrows,
                              queue_ptr stream) {
    const size_t rbyte  = ggml_sycl_kq_mask_row_bytes(ne0);
    const char * bits   = (const char *) vbits;
    const int64_t total = ne0 * nrows;
    constexpr int WG    = 256;
    const int64_t ngrp  = (total + WG - 1) / WG;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ngrp * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
            const int64_t i = item.get_global_linear_id();
            if (i >= total) {
                return;
            }
            const int64_t r = i / ne0;
            const int64_t c = i - r * ne0;
            const uint32_t * in = (const uint32_t *) (bits + (size_t) r * rbyte);
            y[i] = ((in[c >> 5] >> (c & 31)) & 1u) ? (sycl::half) 0.0f
                                                   : (sycl::half) (-INFINITY);
        });
}
