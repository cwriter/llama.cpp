#include "send-recv.hpp"
#include "kq-mask-bits.hpp"

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
    const ggml_tensor * src0  = op->src[0];
    const int           codec = ggml_get_op_params_i32(op, 0);
    if (codec == GGML_SEND_CODEC_MASK_BITS) {
        return src0->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F16 && ggml_is_contiguous(src0) &&
               ggml_is_contiguous(op) && src0->ne[0] == (op->ne[0] + 31) / 32;
    }
    if (codec == GGML_SEND_CODEC_BF16) {
        return src0->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) &&
               ggml_is_contiguous(op) && ggml_nelements(src0) == ggml_nelements(op);
    }
    if (codec != GGML_SEND_CODEC_ROWS || op->type != GGML_TYPE_F32 ||
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
    if (ggml_get_op_params_i32(dst, 0) == GGML_SEND_CODEC_MASK_BITS) {
        // the package rows are the packed rows of kq-mask-bits.hpp: a mask the backend keeps as bits takes them as
        // they are, one that has to be dense (no bits, or a reader that cannot take them) is expanded
        const int64_t nrows = ggml_nrows(dst);
        if (ggml_sycl_kq_mask_is_bits(dst)) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx.stream()->memcpy(dst->data, dst->src[0]->data, ggml_nbytes(dst->src[0]))));
        } else {
            ggml_sycl_kq_mask_to_f16(dst->src[0]->data, (sycl::half *) dst->data, dst->ne[0], nrows, ctx.stream());
        }
        return;
    }
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

// value k of a package, as the RECV kernels above compute it
template <ggml_type type> static inline float recv_value(const void * vx, int64_t k) {
    if constexpr (type == GGML_TYPE_IQ4_NL) {
        const block_iq4_nl * x = (const block_iq4_nl *) vx + k / QK4_NL;
        const int            j = k % QK4_NL;
        const uint8_t        q = x->qs[j % (QK4_NL / 2)];
        return (float) x->d * kvalues_iq4nl[j < QK4_NL / 2 ? q & 0xf : q >> 4];
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const block_iq4_xs * x  = (const block_iq4_xs *) vx + k / QK_K;
        const int            r  = k % QK_K;
        const int            s  = r / 32;
        const int            j  = r % 32;
        const int            ls = ((x->scales_l[s / 2] >> 4 * (s % 2)) & 0xf) | (((x->scales_h >> 2 * s) & 3) << 4);
        const float          dl = (float) x->d * (ls - 32);
        const uint8_t        q  = x->qs[16 * s + j % 16];
        return dl * kvalues_iq4nl[j < 16 ? q & 0xf : q >> 4];
    } else if constexpr (type == GGML_TYPE_F16) {
        return (float) ((const sycl::half *) vx)[k];
    } else {
        return (float) ((const sycl::ext::oneapi::bfloat16 *) vx)[k];
    }
}

template <ggml_type type>
static void recv_repeat(const void * vx, float * y, const int64_t ne[4], const int64_t sne[4], queue_ptr stream) {
    const int64_t ne0 = ne[0], ne1 = ne[1], ne2 = ne[2];
    const int64_t s1 = sne[1], s2 = sne[2], s3 = sne[3];
    stream->parallel_for(sycl::range<1>(ne[0] * ne[1] * ne[2] * ne[3]), [=](sycl::id<1> id) {
        const int64_t k  = id[0];
        const int64_t i0 = k % ne0;
        const int64_t r  = k / ne0;
        const int64_t i1 = r % ne1;
        const int64_t i2 = (r / ne1) % ne2;
        const int64_t i3 = r / (ne1 * ne2);
        const int64_t sr = ((i3 % s3) * s2 + i2 % s2) * s1 + i1 % s1;
        y[k] = recv_value<type>(vx, sr * ne0 + i0);
    });
}

bool ggml_sycl_recv_repeat_fusable(const ggml_cgraph * cgraph, int i) {
    if (i + 1 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * recv = cgraph->nodes[i];
    const ggml_tensor * rep  = cgraph->nodes[i + 1];
    if (recv->op != GGML_OP_RECV || rep->op != GGML_OP_REPEAT || ggml_get_op_params_i32(recv, 1) != 1 ||
        !(recv->flags & GGML_TENSOR_FLAG_COMPUTE) || !(rep->flags & GGML_TENSOR_FLAG_COMPUTE) ||
        recv->src[0] == nullptr || !ggml_sycl_recv_supported(recv)) {
        return false;
    }
    const ggml_tensor * src = rep->src[0];
    if (src != recv && (src->view_src != recv || src->view_offs != 0)) {
        return false;
    }
    return src->type == GGML_TYPE_F32 && rep->type == GGML_TYPE_F32 && ggml_is_contiguous(src) &&
           ggml_is_contiguous(rep) && ggml_nelements(src) == ggml_nelements(recv) && rep->ne[0] == src->ne[0];
}

void ggml_sycl_op_recv_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * recv, ggml_tensor * repeat) {
    const ggml_tensor * pkg    = recv->src[0];
    const ggml_tensor * src    = repeat->src[0];
    const void *        x      = pkg->data;
    float *             y      = (float *) repeat->data;
    queue_ptr           stream = ctx.stream();
    switch (pkg->type) {
        case GGML_TYPE_IQ4_NL:
            recv_repeat<GGML_TYPE_IQ4_NL>(x, y, repeat->ne, src->ne, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            recv_repeat<GGML_TYPE_IQ4_XS>(x, y, repeat->ne, src->ne, stream);
            break;
        case GGML_TYPE_F16:
            recv_repeat<GGML_TYPE_F16>(x, y, repeat->ne, src->ne, stream);
            break;
        case GGML_TYPE_BF16:
            recv_repeat<GGML_TYPE_BF16>(x, y, repeat->ne, src->ne, stream);
            break;
        default:
            GGML_ABORT("unsupported RECV type %s", ggml_type_name(pkg->type));
    }
}

bool ggml_sycl_send_supported(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    if (ggml_get_op_params_i32(op, 0) == GGML_SEND_CODEC_MASK_BITS) {
        return (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32) && op->type == GGML_TYPE_I32 &&
               src0->nb[0] == ggml_type_size(src0->type) && ggml_is_contiguous(op);
    }
    return ggml_get_op_params_i32(op, 0) == GGML_SEND_CODEC_BF16 && op->src[0]->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_BF16 && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op) &&
           ggml_nelements(op->src[0]) == ggml_nelements(op);
}

// one work-item per package word: 32 values of one row, bit set where the value is 0
template <typename T>
static void send_mask_bits(const ggml_tensor * src, uint32_t * y, int64_t nw, queue_ptr stream) {
    const char *  x   = (const char *) src->data;
    const int64_t ne0 = src->ne[0], ne1 = src->ne[1], ne2 = src->ne[2];
    const size_t  nb1 = src->nb[1], nb2 = src->nb[2], nb3 = src->nb[3];
    const int64_t n   = nw * ggml_nrows(src);
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const int64_t w  = id[0] % nw;
        const int64_t r  = id[0] / nw;
        const int64_t i1 = r % ne1;
        const int64_t i2 = (r / ne1) % ne2;
        const int64_t i3 = r / (ne1 * ne2);
        const T *     row = (const T *) (x + i1 * nb1 + i2 * nb2 + i3 * nb3);
        uint32_t      word = 0;
        for (int k = 0; k < 32 && w * 32 + k < ne0; ++k) {
            word |= (uint32_t) ((float) row[w * 32 + k] == 0.0f) << k;
        }
        y[id[0]] = word;
    });
}

void ggml_sycl_op_send(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_sycl_send_supported(dst));
    if (ggml_get_op_params_i32(dst, 0) == GGML_SEND_CODEC_MASK_BITS) {
        if (dst->src[0]->type == GGML_TYPE_F16) {
            send_mask_bits<sycl::half>(dst->src[0], (uint32_t *) dst->data, dst->ne[0], ctx.stream());
        } else {
            send_mask_bits<float>(dst->src[0], (uint32_t *) dst->data, dst->ne[0], ctx.stream());
        }
        return;
    }
    const float *                    x = (const float *) dst->src[0]->data;
    sycl::ext::oneapi::bfloat16 *    y = (sycl::ext::oneapi::bfloat16 *) dst->data;
    // round to nearest even, as ggml_compute_fp32_to_bf16
    ctx.stream()->parallel_for(sycl::range<1>(ggml_nelements(dst)), [=](sycl::id<1> id) {
        y[id[0]] = sycl::ext::oneapi::bfloat16(x[id[0]]);
    });
}
