#include "send-recv.hpp"

// The dequantizers of dequantize.hpp return dfloat, which is half in a GGML_SYCL_F16 build. These follow
// dequantize_row_* of ggml-quants.c operation for operation in f32, so a row expanded here matches the CPU.

static void recv_iq4_nl(const void * vx, float * y, int64_t nb, queue_ptr stream) {
    // one work-item per byte: values j and j + 16 of a block
    const int64_t n = nb * (QK4_NL / 2);
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const int64_t        g  = id[0];
        const int64_t        ib = g / (QK4_NL / 2);
        const int            j  = g % (QK4_NL / 2);
        const block_iq4_nl * x  = (const block_iq4_nl *) vx + ib;
        const float          d  = (float) x->d;
        const uint8_t        q  = x->qs[j];
        float *              yb = y + ib * QK4_NL;
        yb[j]              = d * kvalues_iq4nl[q & 0xf];
        yb[j + QK4_NL / 2] = d * kvalues_iq4nl[q >> 4];
    });
}

static void recv_iq4_xs(const void * vx, float * y, int64_t nb, queue_ptr stream) {
    // one work-item per byte: sub-block s of 32 values, values j and j + 16 of it
    const int64_t n = nb * (QK_K / 2);
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const int64_t        g  = id[0];
        const int64_t        i  = g / (QK_K / 2);
        const int            r  = g % (QK_K / 2);
        const int            s  = r / 16;
        const int            j  = r % 16;
        const block_iq4_xs * x  = (const block_iq4_xs *) vx + i;
        const float          d  = (float) x->d;
        const int            ls = ((x->scales_l[s / 2] >> 4 * (s % 2)) & 0xf) | (((x->scales_h >> 2 * s) & 3) << 4);
        const float          dl = d * (ls - 32);
        const uint8_t        q  = x->qs[16 * s + j];
        float *              yb = y + i * QK_K + 32 * s;
        yb[j]      = dl * kvalues_iq4nl[q & 0xf];
        yb[j + 16] = dl * kvalues_iq4nl[q >> 4];
    });
}

template <typename T> static void recv_widen(const void * vx, float * y, int64_t n, queue_ptr stream) {
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        y[id[0]] = (float) ((const T *) vx)[id[0]];
    });
}

bool ggml_sycl_recv_supported(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    if (ggml_get_op_params_i32(op, 0) != GGML_SEND_CODEC_ROWS || op->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(op) || ggml_nelements(src0) != ggml_nelements(op)) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

void ggml_sycl_op_recv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_sycl_recv_supported(dst));
    const ggml_tensor * src0   = dst->src[0];
    const void *        x      = src0->data;
    float *             y      = (float *) dst->data;
    const int64_t       n      = ggml_nelements(src0);
    queue_ptr           stream = ctx.stream();
    switch (src0->type) {
        case GGML_TYPE_IQ4_NL:
            recv_iq4_nl(x, y, n / QK4_NL, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            recv_iq4_xs(x, y, n / QK_K, stream);
            break;
        case GGML_TYPE_F16:
            recv_widen<sycl::half>(x, y, n, stream);
            break;
        case GGML_TYPE_BF16:
            recv_widen<sycl::ext::oneapi::bfloat16>(x, y, n, stream);
            break;
        default:
            GGML_ABORT("unsupported RECV type %s", ggml_type_name(src0->type));
    }
}
