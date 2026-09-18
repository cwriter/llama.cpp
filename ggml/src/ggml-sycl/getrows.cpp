//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <atomic>
#include <type_traits>

#include "ggml-impl.h"
#include "common.hpp"
#include "dequantize.hpp"
#include "getrows.hpp"

// sycl::vec tops out at 16; a 16 wide float vector is one 64 byte cache line per work item
static constexpr int GET_ROWS_MAX_VEC_WIDTH = 16;

// Vectorising trades work items for wider messages, so only widen while the grid still has
// enough work items left to fill the device. Measured on 3x Arc Pro B60: 4 is the best of
// 1, 2, 4, 8, 16, 32, 64 (20.81 t/s; 1 picks width 16 and gives 20.52, 16+ falls back to
// scalar and gives 20.42).
static constexpr int GET_ROWS_WORK_GROUPS_PER_CU = 4;



template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

template<int qk, int qr, dequantize_kernel_f32_t dequantize_kernel, typename dst_t>
static void k_get_rows_f32(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk;
    const int iqs = (i00%qk)/qr;
    const int iybs = i00 - i00%qk;
    const int y_offset = qr == 1 ? 1 : qk/2;

    float v0;
    float v1;
    dequantize_kernel(src0_row, ib, iqs, v0, v1);

    dst_row[iybs + iqs + 0] = (dst_t) v0;
    dst_row[iybs + iqs + y_offset] = (dst_t) v1;
}

template<typename src0_t, typename dst_t, int vec_width>
static void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    if constexpr (vec_width > 1) {
        // one vec_width element message per work item instead of that many scalar ones; the
        // launcher only picks a width the row length, the row strides and the bases divide
        const sycl::vec<src0_t, vec_width> in =
            *(const sycl::vec<src0_t, vec_width> *) (src0_row + vec_width*i00);
        sycl::vec<dst_t, vec_width> out;
#pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            out[j] = (dst_t) in[j];
        }
        *(sycl::vec<dst_t, vec_width> *) (dst_row + vec_width*i00) = out;
    } else {
        dst_row[i00] = src0_row[i00];
    }
}

template <int qk, int qr, dequantize_kernel_t dq>
static void get_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <int qk, int qr, dequantize_kernel_f32_t dq>
static void get_rows_sycl_f32(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const void *src0_dd,
                              const int32_t *src1_dd, float *dst_dd,
                              queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows_f32<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <typename src0_t, typename dst_t>
static void get_rows_sycl_float(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const src0_t *src0_dd, const int32_t *src1_dd,
                                dst_t *dst_dd, queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    // a w element message per work item needs the row length, every row stride and both
    // bases to divide by w. take the widest that fits, down to one element per work item.
    const auto fits = [&](int w) {
        return ne00 % w == 0 &&
            nb1 % (w*sizeof(dst_t)) == 0 && nb2 % (w*sizeof(dst_t)) == 0 && nb3 % (w*sizeof(dst_t)) == 0 &&
            nb01 % (w*sizeof(src0_t)) == 0 && nb02 % (w*sizeof(src0_t)) == 0 && nb03 % (w*sizeof(src0_t)) == 0 &&
            ((uintptr_t) src0_dd) % (w*sizeof(src0_t)) == 0 &&
            ((uintptr_t) dst_dd)  % (w*sizeof(dst_t))  == 0;
    };
    const int64_t n_rows    = ne10 * ne11 * ne12;
    const int64_t min_items = (int64_t) ggml_sycl_info().devices[ctx.device].nsm * 16 *
                              GET_ROWS_WORK_GROUPS_PER_CU * SYCL_GET_ROWS_BLOCK_SIZE;
    int width = 1;
    for (int w : { 16, 8, 4, 2 }) {
        if (w <= GET_ROWS_MAX_VEC_WIDTH && fits(w) && (ne00 / w) * n_rows >= min_items) {
            width = w;
            break;
        }
    }
    const int64_t ne00_dispatch = ne00 / width;

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00_dispatch + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        auto launch = [&](auto width) {
            stream->parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_get_rows_float<src0_t, dst_t, decltype(width)::value>(
                        src0_dd, src1_dd, dst_dd, ne00_dispatch, ne12, s1, s2,
                        s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                });
        };

        switch (width) {
            case 16: launch(std::integral_constant<int, 16>{}); break;
            case  8: launch(std::integral_constant<int,  8>{}); break;
            case  4: launch(std::integral_constant<int,  4>{}); break;
            case  2: launch(std::integral_constant<int,  2>{}); break;
            default: launch(std::integral_constant<int,  1>{}); break;
        }
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32 );

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_size(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_size(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) dst->src[1]->data;
    /* TODO: Refactor and remove duplicates */
    switch (dst->src[0]->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::half *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_BF16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::ext::oneapi::bfloat16 *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_I32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const int32_t *)dst->src[0]->data,
            src1_i32, (int32_t *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q1_0:
            get_rows_sycl<QK1_0, 1, dequantize_q1_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_MXFP4:
            get_rows_sycl<QK_MXFP4, 2, dequantize_mxfp4>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_NVFP4:
            get_rows_sycl<QK_NVFP4, 1, dequantize_nvfp4>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_sycl<QK_K, 1, dequantize_iq2_xxs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_sycl<QK_K, 1, dequantize_iq2_xs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_sycl<QK_K, 1, dequantize_iq2_s>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_sycl<QK_K, 1, dequantize_iq3_xxs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_sycl<QK_K, 1, dequantize_iq1_s>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_sycl<QK_K, 1, dequantize_iq1_m>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_sycl<QK_K, 1, dequantize_iq3_s>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_sycl<QK4_NL, 1, dequantize_iq4_nl>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_sycl<QK_K, 1, dequantize_iq4_xs>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q2_K:
            get_rows_sycl_f32<QK_K, 1, dequantize_q2_K_f32>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q3_K:
            get_rows_sycl<QK_K, 1, dequantize_q3_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_0:
            get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_K:
            get_rows_sycl_f32<QK_K, 1, dequantize_q4_K_f32>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_K:
            get_rows_sycl_f32<QK_K, 1, dequantize_q5_K_f32>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q6_K:
            get_rows_sycl<QK_K, 1, dequantize_q6_K>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q8_0:
            get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        default:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}

// The QSA indexer gives every cell the score of its block. get_rows only gathers along ne1,
// so the model graph transposes the scores, gathers, and transposes back:
//   CONT(PERMUTE(score)) -> GET_ROWS -> PERMUTE -> CONT
// The whole chain is expanded[c, t] = score[idx[c], t]. One kernel writes that final layout
// directly, so neither transposed copy nor the gather output is ever written; the first CONT
// and the GET_ROWS are reported through fusion_absorbs so ggml-alloc reserves nothing for them.
// Every test below is structural, so the answer is the same before and after allocation.
bool ggml_sycl_can_fuse_qsa_gather(const ggml_cgraph * cgraph, int i) {
    if (!g_ggml_sycl_enable_fusion || !g_ggml_sycl_fuse_qsa_gather) {
        return false;
    }
    if (i + 3 >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * cont_in  = cgraph->nodes[i];
    ggml_tensor * rows     = cgraph->nodes[i + 1];
    ggml_tensor * perm_out = cgraph->nodes[i + 2];
    ggml_tensor * cont_out = cgraph->nodes[i + 3];

    if (cont_in->op != GGML_OP_CONT || rows->op != GGML_OP_GET_ROWS || perm_out->op != GGML_OP_PERMUTE ||
        cont_out->op != GGML_OP_CONT) {
        return false;
    }
    for (const ggml_tensor * n : { cont_in, rows, perm_out, cont_out }) {
        if (n->type != GGML_TYPE_F32 || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }

    // an empty node is skipped by the compute loop, which would leave an absorbed dst unwritten
    if (ggml_is_empty(cont_in) || ggml_is_empty(rows) || ggml_is_empty(cont_out)) {
        return false;
    }

    // the chain must be linear and only cont_out may leave it
    if (cont_in->view_src || rows->view_src || cont_out->view_src) {
        return false;
    }
    if ((cont_in->flags | rows->flags | perm_out->flags) & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 1) != 1 ||
        ggml_node_get_use_count(cgraph, i + 2) != 1) {
        return false;
    }
    if (rows->src[0] != cont_in || perm_out->src[0] != rows || perm_out->view_src != rows ||
        perm_out->view_offs != 0 || cont_out->src[0] != perm_out) {
        return false;
    }

    // the kernel reads the score through the permuted view, so only its strides matter
    const ggml_tensor * src = cont_in->src[0];
    const ggml_tensor * idx = rows->src[1];
    if (!src || !idx || src->type != GGML_TYPE_F32 || idx->type != GGML_TYPE_I32) {
        return false;
    }
    if (!ggml_are_same_shape(src, cont_in) || src->ne[2] != 1 || src->ne[3] != 1) {
        return false;
    }
    if (idx->ne[1] != 1 || idx->ne[2] != 1 || idx->ne[3] != 1 || idx->nb[0] != ggml_type_size(GGML_TYPE_I32)) {
        return false;
    }
    const size_t ts = ggml_type_size(GGML_TYPE_F32);
    if (src->nb[0] % ts != 0 || src->nb[1] % ts != 0) {
        return false;
    }

    // get_rows output, then the exact ne0/ne1 swap of it
    if (rows->ne[0] != cont_in->ne[0] || rows->ne[1] != idx->ne[0] || rows->ne[2] != 1 || rows->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(cont_in) || !ggml_is_contiguous(rows) || !ggml_is_contiguous(cont_out)) {
        return false;
    }
    if (perm_out->ne[0] != rows->ne[1] || perm_out->ne[1] != rows->ne[0] || perm_out->ne[2] != 1 ||
        perm_out->ne[3] != 1 || perm_out->nb[0] != rows->nb[1] || perm_out->nb[1] != rows->nb[0]) {
        return false;
    }
    if (!ggml_are_same_shape(cont_out, perm_out)) {
        return false;
    }

    return true;
}

template <int width>
static void k_qsa_gather(const char * src, const int32_t * idx, float * dst,
                         int64_t n_idx, size_t nb0, size_t nb1, const sycl::nd_item<2> & item) {
    const int64_t t = item.get_global_id(0);
    const int64_t c = (int64_t) item.get_global_id(1) * width;
    if (c >= n_idx) {
        return;
    }

    const char * row = src + t*nb0;
    float *      out = dst + t*n_idx + c;

    if constexpr (width > 1) {
        sycl::vec<float, width> v;
#pragma unroll
        for (int j = 0; j < width; ++j) {
            v[j] = *(const float *) (row + (int64_t) idx[c + j]*nb1);
        }
        *(sycl::vec<float, width> *) out = v;
    } else {
        *out = *(const float *) (row + (int64_t) idx[c]*nb1);
    }
}

// Runs the chain matched by ggml_sycl_can_fuse_qsa_gather(); returns the extra nodes consumed.
int ggml_sycl_fuse_qsa_gather(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (!ggml_sycl_can_fuse_qsa_gather(cgraph, i)) {
        return 0;
    }

    const ggml_tensor * src      = cgraph->nodes[i]->src[0];
    const ggml_tensor * idx      = cgraph->nodes[i + 1]->src[1];
    ggml_tensor *       cont_out = cgraph->nodes[i + 3];

    const int64_t n_idx  = cont_out->ne[0];
    const int64_t n_cols = cont_out->ne[1];

    // one 16 byte store per work item when the destination rows line up for it
    const int width = (n_idx % 4 == 0 && ((uintptr_t) cont_out->data) % 16 == 0) ? 4 : 1;

    constexpr int block = 256;
    const int64_t items = (n_idx + width - 1) / width;
    const sycl::range<2> local(1, block);
    const sycl::range<2> global(n_cols, ((items + block - 1) / block) * block);

    const char *    src_dd = (const char *) src->data;
    const int32_t * idx_dd = (const int32_t *) idx->data;
    float *         dst_dd = (float *) cont_out->data;
    const size_t    nb0    = src->nb[0];
    const size_t    nb1    = src->nb[1];

    // GGML_SYCL_QSA_GATHER_TRACE gives the number of firings to report
    static std::atomic<int> trace_left{ getenv("GGML_SYCL_QSA_GATHER_TRACE") ?
                                        std::max(1, atoi(getenv("GGML_SYCL_QSA_GATHER_TRACE"))) : 0 };
    if (trace_left.fetch_sub(1) > 0) {
        fprintf(stderr, "[QSAGATHER] n_idx=%ld n_cols=%ld width=%d nb0=%zu nb1=%zu src=%p idx=%p dst=%p cont_in=%p rows=%p\n",
                (long) n_idx, (long) n_cols, width, nb0, nb1, (const void *) src_dd, (const void *) idx_dd,
                (void *) dst_dd, (void *) cgraph->nodes[i]->data, (void *) cgraph->nodes[i + 1]->data);
    }

    GGML_ASSERT(src_dd && idx_dd && dst_dd);

    auto launch = [&](auto w) {
        ctx.stream()->parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            k_qsa_gather<decltype(w)::value>(src_dd, idx_dd, dst_dd, n_idx, nb0, nb1, item);
        });
    };
    if (width == 4) {
        launch(std::integral_constant<int, 4>{});
    } else {
        launch(std::integral_constant<int, 1>{});
    }

    return 3;
}
