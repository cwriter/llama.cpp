#include "binbcast.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "ggml.h"

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        int s00, int s01, int s02, int s03,
        int s10, int s11, int s12, int s13,
        const sycl::nd_item<3> &item_ct1) {
    const int i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    const int i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    const int i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        const int i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0*s00] : 0.0f, (float)src1_row[i10*s10]);
    }
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_unravel(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        int s00, int s01, int s02, int s03,
        int s10, int s11, int s12, int s13,
        const sycl::nd_item<3> &item_ct1) {

    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    const int i3 = i/(ne2*ne1*ne0);
    const int i2 = (i/(ne1*ne0)) % ne2;
    const int i1 = (i/ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    const int i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0*s00] : 0.0f, (float)src1_row[i10*s10]);
}


template<float (*bin_op)(const float, const float)>
struct bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    void operator()(const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd, const int64_t ne00,
                    const int64_t ne01, const int64_t ne02, const int64_t ne03, const int64_t ne10, const int64_t ne11,
                    const int64_t ne12, const int64_t ne13, const int64_t ne0, const int64_t ne1, const int64_t ne2,
                    const int64_t ne3, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                    const size_t nb10, const size_t nb11, const size_t nb12, const size_t nb13, const size_t nb0,
                    const size_t nb1, const size_t nb2, const size_t nb3, const bool src0_is_contiguous,
                    const bool src1_is_contiguous, const bool src0_is_permuted, const bool src1_is_permuted,
                    queue_ptr stream) {
        int nr0 = ne10 / ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        if (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            for (int i = 0; i < 4; i++) {
                if (nr[i] != 1) {
                    break;
                }
                if (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }
        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            // size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            size_t s00 = nb00 / sizeof(src0_t);
            size_t s01 = nb01 / sizeof(src0_t);
            size_t s02 = nb02 / sizeof(src0_t);
            size_t s03 = nb03 / sizeof(src0_t);

            GGML_UNUSED(s00);

            GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

            GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

            GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<unsigned int>(hne0, block_size);
            block_dims[1] = std::min<unsigned int>(
                ne1, block_size / (unsigned int)block_dims[2]);
            block_dims[0] = std::min(
                std::min<unsigned int>(
                    ne2 * ne3, block_size / (unsigned int)block_dims[2] /
                                   (unsigned int)block_dims[1]),
                64U);

            sycl::range<3> block_nums(
                (ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                (ne1 + block_dims[1] - 1) / block_dims[1],
                (hne0 + block_dims[2] - 1) / block_dims[2]);

            if (block_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                int block_num = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, block_num) *
                                              sycl::range<3>(1, 1, block_size),
                                          sycl::range<3>(1, 1, block_size)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s00, s01, s02,
                                s03, s10, s11, s12, s13, item_ct1);
                        });
                }
            } else {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1,
                                            ne2, ne3, ne10, ne11, ne12, ne13,
                                            s1, s2, s3, s00, s01, s02, s03, s10, s11, s12, s13,
                                            item_ct1);
                    });
            }
        }
    }
};

template <class op>
inline void ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::queue_ptr main_stream = ctx.stream();
    GGML_TENSOR_BINARY_OP_LOCALS

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((const float *) src0->data, (const float *) src1->data, (float *) dst->data, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((const sycl::half *) src0->data, (const sycl::half *) src1->data, (sycl::half *) dst->data, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((const sycl::half *) src0->data, (const float *) src1->data, (sycl::half *) dst->data, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((const int32_t *) src0->data, (const int32_t *) src1->data, (int32_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((const int16_t *) src0->data, (const int16_t *) src1->data, (int16_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((const sycl::ext::oneapi::bfloat16 *) src0->data, (const sycl::ext::oneapi::bfloat16 *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((const sycl::ext::oneapi::bfloat16 *) src0->data, (const float *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
#endif
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

inline void ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_add(ctx, dst);
}

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
static void k_bin_bcast3(const src0_t * src0, const src1_t * src1, const src2_t * src2, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        int ne20, int ne21, int ne22, int ne23,
        int s1, int s2, int s3,
        int s00, int s01, int s02, int s03,
        int s10, int s11, int s12, int s13,
        int s20, int s21, int s22, int s23,
        const sycl::nd_item<3> & item_ct1) {
    const int i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    const int i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    const int i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;
    const int i21 = i1 % ne21;
    const int i22 = i2 % ne22;
    const int i23 = i3 % ne23;

    const size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    const size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    const size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    const size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    const src2_t * src2_row = src2 + i_src2;
    dst_t *        dst_row  = dst + i_dst;

    for (int i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        const int   i10 = i0 % ne10;
        const int   i20 = i0 % ne20;
        const float acc = bin_op((float) src0_row[i0 * s00], (float) src1_row[i10 * s10]);
        dst_row[i0]     = (dst_t) bin_op(acc, (float) src2_row[i20 * s20]);
    }
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
static void k_bin_bcast3_unravel(const src0_t * src0, const src1_t * src1, const src2_t * src2, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        int ne20, int ne21, int ne22, int ne23,
        int s1, int s2, int s3,
        int s00, int s01, int s02, int s03,
        int s10, int s11, int s12, int s13,
        int s20, int s21, int s22, int s23,
        const sycl::nd_item<3> & item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    const int i3 = i / (ne2 * ne1 * ne0);
    const int i2 = (i / (ne1 * ne0)) % ne2;
    const int i1 = (i / ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;
    const int i21 = i1 % ne21;
    const int i22 = i2 % ne22;
    const int i23 = i3 % ne23;

    const size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    const size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    const size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    const size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    const int   i10 = i0 % ne10;
    const int   i20 = i0 % ne20;
    const float acc = bin_op((float) src0[i_src0 + i0 * s00], (float) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (float) src2[i_src2 + i20 * s20]);
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
static void launch_bin_bcast3(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * src2, ggml_tensor * dst) {
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    int nr1[4] = { (int) (ne10 / ne0), (int) (ne11 / ne1), (int) (ne12 / ne2), (int) (ne13 / ne3) };
    int nr2[4] = { (int) (ne20 / ne0), (int) (ne21 / ne1), (int) (ne22 / ne2), (int) (ne23 / ne3) };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };
    int64_t cne2[] = { ne20, ne21, ne22, ne23 };
    size_t  cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t  cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t  cnb1[] = { nb10, nb11, nb12, nb13 };
    size_t  cnb2[] = { nb20, nb21, nb22, nb23 };

    auto collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    const bool can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    if (can_collapse) {
        for (int i = 0; i < 4; i++) {
            if (nr1[i] != 1 || nr2[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb, cne);
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse_nb(cnb2, cne2);
                collapse(cne);
                collapse(cne0);
                collapse(cne1);
                collapse(cne2);
            }
        }
    }

    {
        int64_t ne0 = cne[0];
        int64_t ne1 = cne[1];
        int64_t ne2 = cne[2];
        int64_t ne3 = cne[3];

        int64_t ne10 = cne1[0];
        int64_t ne11 = cne1[1];
        int64_t ne12 = cne1[2];
        int64_t ne13 = cne1[3];

        int64_t ne20 = cne2[0];
        int64_t ne21 = cne2[1];
        int64_t ne22 = cne2[2];
        int64_t ne23 = cne2[3];

        size_t s1 = cnb[1] / sizeof(dst_t);
        size_t s2 = cnb[2] / sizeof(dst_t);
        size_t s3 = cnb[3] / sizeof(dst_t);

        size_t s00 = cnb0[0] / sizeof(src0_t);
        size_t s01 = cnb0[1] / sizeof(src0_t);
        size_t s02 = cnb0[2] / sizeof(src0_t);
        size_t s03 = cnb0[3] / sizeof(src0_t);

        size_t s10 = cnb1[0] / sizeof(src1_t);
        size_t s11 = cnb1[1] / sizeof(src1_t);
        size_t s12 = cnb1[2] / sizeof(src1_t);
        size_t s13 = cnb1[3] / sizeof(src1_t);

        size_t s20 = cnb2[0] / sizeof(src2_t);
        size_t s21 = cnb2[1] / sizeof(src2_t);
        size_t s22 = cnb2[2] / sizeof(src2_t);
        size_t s23 = cnb2[3] / sizeof(src2_t);

        GGML_ASSERT(cnb[0] % sizeof(dst_t) == 0 && cnb[1] % sizeof(dst_t) == 0 && cnb[2] % sizeof(dst_t) == 0 &&
                    cnb[3] % sizeof(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % sizeof(src0_t) == 0 && cnb0[1] % sizeof(src0_t) == 0 && cnb0[2] % sizeof(src0_t) == 0 &&
                    cnb0[3] % sizeof(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % sizeof(src1_t) == 0 && cnb1[1] % sizeof(src1_t) == 0 && cnb1[2] % sizeof(src1_t) == 0 &&
                    cnb1[3] % sizeof(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % sizeof(src2_t) == 0 && cnb2[1] % sizeof(src2_t) == 0 && cnb2[2] % sizeof(src2_t) == 0 &&
                    cnb2[3] % sizeof(src2_t) == 0);

        const src0_t * src0_dd = (const src0_t *) src0->data;
        const src1_t * src1_dd = (const src1_t *) src1->data;
        const src2_t * src2_dd = (const src2_t *) src2->data;
        dst_t *        dst_dd  = (dst_t *) dst->data;

        const int block_size = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> block_dims(1, 1, 1);
        block_dims[2] = std::min<unsigned int>(hne0, block_size);
        block_dims[1] = std::min<unsigned int>(ne1, block_size / (unsigned int) block_dims[2]);
        block_dims[0] = std::min(std::min<unsigned int>(ne2 * ne3,
                                                        block_size / (unsigned int) block_dims[2] /
                                                            (unsigned int) block_dims[1]),
                                 64U);

        sycl::range<3> block_nums((ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                                  (ne1 + block_dims[1] - 1) / block_dims[1],
                                  (hne0 + block_dims[2] - 1) / block_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        if (block_nums[0] > 65535) {
            int block_num = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, block_num) * sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, item_ct1);
                });
        } else {
            stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     k_bin_bcast3<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10,
                                                          ne11, ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00,
                                                          s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23,
                                                          item_ct1);
                                 });
        }
    }
}

void ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    const ggml_tensor * src0 = add0->src[0];
    const ggml_tensor * src1 = add0->src[1];
    const ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_add_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_add_kernel_supports(add0->type, src2->type, dst->type));

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_add, float, float, float, float>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, float, float, sycl::half>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, float, sycl::half>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, float, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_add, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_add, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#ifdef GGML_SYCL_HAS_BF16
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, float, float, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, float,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, float, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#endif
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}


struct ggml_sycl_add_n_operand {
    const float * data;
    size_t nb1;
    size_t nb2;
    size_t nb3;
};

template <int begin, int end, size_t n_operands>
static float add_n_f32_strided_sum(const std::array<ggml_sycl_add_n_operand, n_operands> & operands,
                                   uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
    if constexpr (end - begin == 1) {
        const char * base = (const char *) operands[begin].data;
        const float * src = (const float *) (base + i1 * operands[begin].nb1 +
            i2 * operands[begin].nb2 + i3 * operands[begin].nb3);
        return src[i0];
    } else {
        constexpr int mid = begin + (end - begin) / 2;
        return add_n_f32_strided_sum<begin, mid>(operands, i0, i1, i2, i3) +
               add_n_f32_strided_sum<mid, end>(operands, i0, i1, i2, i3);
    }
}

template <int n_operands>
static void k_add_n_f32(const std::array<ggml_sycl_add_n_operand, n_operands> operands, float * dst,
                        uint32_t nelements, const sycl::uint3 ne0_fd, const sycl::uint3 ne1_fd,
                        const sycl::uint3 ne2_fd,
                        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3, float scale, float bias,
                        const sycl::nd_item<1> & item) {
    const uint32_t index = item.get_global_linear_id();
    if (index >= nelements) {
        return;
    }
    const sycl::uint2 i0_rc = fast_div_modulo(index, ne0_fd);
    const sycl::uint2 i1_rc = fast_div_modulo(i0_rc.x(), ne1_fd);
    const sycl::uint2 i2_rc = fast_div_modulo(i1_rc.x(), ne2_fd);
    const uint32_t i0 = i0_rc.y();
    const uint32_t i1 = i1_rc.y();
    const uint32_t i2 = i2_rc.y();
    const uint32_t i3 = i2_rc.x();

    const float sum = add_n_f32_strided_sum<0, n_operands>(operands, i0, i1, i2, i3);
    char * dst_base = (char *) dst + i1 * dst_nb1 + i2 * dst_nb2 + i3 * dst_nb3;
    ((float *) dst_base)[i0] = sum * scale + bias;
}

template <int operand, size_t n_linear, size_t n_strided>
static float add_n_f32_grouped_load(const std::array<const float *, n_linear> & linear,
                                    const std::array<const float *, n_strided> & strided,
                                    uint32_t index, size_t strided_offset) {
    if constexpr (operand < n_linear) {
        return linear[operand][index];
    } else {
        const char * base = (const char *) strided[operand - n_linear];
        return *(const float *) (base + strided_offset);
    }
}

template <int begin, int end, size_t n_linear, size_t n_strided>
static float add_n_f32_grouped_sum(const std::array<const float *, n_linear> & linear,
                                   const std::array<const float *, n_strided> & strided,
                                   uint32_t index, size_t strided_offset) {
    if constexpr (end - begin == 1) {
        return add_n_f32_grouped_load<begin>(linear, strided, index, strided_offset);
    } else {
        constexpr int mid = begin + (end - begin) / 2;
        return add_n_f32_grouped_sum<begin, mid>(linear, strided, index, strided_offset) +
               add_n_f32_grouped_sum<mid, end>(linear, strided, index, strided_offset);
    }
}

template <int n_linear, int n_strided, bool bounds_check>
static void k_add_n_f32_grouped(const std::array<const float *, n_linear> linear,
                                const std::array<const float *, n_strided> strided, float * dst,
                                uint32_t nelements, const sycl::uint3 ne0_fd, const sycl::uint3 ne1_fd,
                                const sycl::uint3 ne2_fd, size_t nb1, size_t nb2, size_t nb3,
                                float scale, float bias, const sycl::nd_item<1> & item) {
    const uint32_t index = item.get_global_linear_id();
    if constexpr (bounds_check) {
        if (index >= nelements) {
            return;
        }
    }
    const sycl::uint2 i0_rc = fast_div_modulo(index, ne0_fd);
    const sycl::uint2 i1_rc = fast_div_modulo(i0_rc.x(), ne1_fd);
    const sycl::uint2 i2_rc = fast_div_modulo(i1_rc.x(), ne2_fd);
    const size_t offset = i0_rc.y() * sizeof(float) + i1_rc.y() * nb1 + i2_rc.y() * nb2 + i2_rc.x() * nb3;
    const float sum = add_n_f32_grouped_sum<0, n_linear + n_strided>(linear, strided, index, offset);
    dst[index] = sum * scale + bias;
}

template <int n_operands>
static void launch_add_n_f32_strided(ggml_backend_sycl_context & ctx,
                                     const std::array<const ggml_tensor *, 9> & tensors,
                                     ggml_tensor * dst, uint32_t nelements, float scale, float bias) {
    std::array<ggml_sycl_add_n_operand, n_operands> operands;
    for (int i = 0; i < n_operands; ++i) {
        operands[i] = { (const float *) tensors[i]->data, tensors[i]->nb[1], tensors[i]->nb[2], tensors[i]->nb[3] };
    }
    const sycl::uint3 ne0_fd = init_fastdiv_values((uint32_t) dst->ne[0]);
    const sycl::uint3 ne1_fd = init_fastdiv_values((uint32_t) dst->ne[1]);
    const sycl::uint3 ne2_fd = init_fastdiv_values((uint32_t) dst->ne[2]);
    // read the host tensor here: the kernel must not dereference it
    float * const dst_d   = (float *) dst->data;
    const size_t  dst_nb1 = dst->nb[1];
    const size_t  dst_nb2 = dst->nb[2];
    const size_t  dst_nb3 = dst->nb[3];
    constexpr int block_size = 256;
    const uint32_t global_size = ceil_div(nelements, (uint32_t) block_size) * block_size;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(global_size, block_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            k_add_n_f32<n_operands>(operands, dst_d, nelements, ne0_fd, ne1_fd, ne2_fd,
                                    dst_nb1, dst_nb2, dst_nb3, scale, bias, item);
        });
}

template <int n_linear, int n_strided, bool bounds_check>
static void launch_add_n_f32_grouped_impl(ggml_backend_sycl_context & ctx,
                                          const std::array<const ggml_tensor *, 9> & tensors,
                                          float * dst, uint32_t nelements, float scale, float bias) {
    std::array<const float *, n_linear> linear;
    std::array<const float *, n_strided> strided;
    for (int i = 0; i < n_linear; ++i) {
        linear[i] = (const float *) tensors[i]->data;
    }
    for (int i = 0; i < n_strided; ++i) {
        strided[i] = (const float *) tensors[n_linear + i]->data;
    }
    const ggml_tensor * layout = tensors[n_linear];
    const sycl::uint3 ne0_fd = init_fastdiv_values((uint32_t) layout->ne[0]);
    const sycl::uint3 ne1_fd = init_fastdiv_values((uint32_t) layout->ne[1]);
    const sycl::uint3 ne2_fd = init_fastdiv_values((uint32_t) layout->ne[2]);
    // read the host tensor here: the kernel must not dereference it
    const size_t nb1 = layout->nb[1];
    const size_t nb2 = layout->nb[2];
    const size_t nb3 = layout->nb[3];
    constexpr int block_size = 256;
    const uint32_t global_size = ceil_div(nelements, (uint32_t) block_size) * block_size;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(global_size, block_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            k_add_n_f32_grouped<n_linear, n_strided, bounds_check>(linear, strided, dst, nelements,
                                                                  ne0_fd, ne1_fd, ne2_fd, nb1, nb2, nb3,
                                                                  scale, bias, item);
        });
}

template <int n_linear, int n_strided>
static void launch_add_n_f32_grouped(ggml_backend_sycl_context & ctx,
                                     const std::array<const ggml_tensor *, 9> & tensors,
                                     float * dst, uint32_t nelements, float scale, float bias) {
    constexpr int block_size = 256;
    if (nelements % block_size == 0) {
        launch_add_n_f32_grouped_impl<n_linear, n_strided, false>(ctx, tensors, dst, nelements, scale, bias);
    } else {
        launch_add_n_f32_grouped_impl<n_linear, n_strided, true>(ctx, tensors, dst, nelements, scale, bias);
    }
}

template <int n_operands>
static void launch_add_n_f32_grouped(ggml_backend_sycl_context & ctx,
                                     const std::array<const ggml_tensor *, 9> & tensors,
                                     float * dst, uint32_t nelements, float scale, float bias, int n_linear) {
    if (n_linear == 0) {
        launch_add_n_f32_grouped<0, n_operands>(ctx, tensors, dst, nelements, scale, bias);
    } else {
        GGML_ASSERT(n_linear == 1);
        launch_add_n_f32_grouped<1, n_operands - 1>(ctx, tensors, dst, nelements, scale, bias);
    }
}

template <int begin, int end, size_t n_operands>
static float add_n_f32_contiguous_sum(const std::array<const float *, n_operands> & src, uint32_t index) {
    if constexpr (end - begin == 1) {
        return src[begin][index];
    } else {
        constexpr int mid = begin + (end - begin) / 2;
        return add_n_f32_contiguous_sum<begin, mid>(src, index) +
               add_n_f32_contiguous_sum<mid, end>(src, index);
    }
}

template <int n_operands>
static void launch_add_n_f32_contiguous(ggml_backend_sycl_context & ctx,
                                        const std::array<const ggml_tensor *, 9> & tensors,
                                        float * dst, uint32_t nelements, float scale, float bias) {
    std::array<const float *, n_operands> src;
    for (int i = 0; i < n_operands; ++i) {
        src[i] = (const float *) tensors[i]->data;
    }
    constexpr int block_size = 256;
    const uint32_t global_size = ceil_div(nelements, (uint32_t) block_size) * block_size;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(global_size, block_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const uint32_t index = item.get_global_linear_id();
            if (index >= nelements) {
                return;
            }
            const float sum = add_n_f32_contiguous_sum<0, n_operands>(src, index);
            dst[index] = sum * scale + bias;
        });
}

void ggml_sycl_op_add_n_fused(ggml_backend_sycl_context & ctx, ggml_tensor * const * nodes, int n_nodes,
                              ggml_tensor * scale_node) {
    GGML_ASSERT(n_nodes >= 1 && n_nodes <= 8);
    std::array<const ggml_tensor *, 9> tensors = {};
    tensors[0] = nodes[0]->src[0];
    tensors[1] = nodes[0]->src[1];
    for (int i = 1; i < n_nodes; ++i) {
        GGML_ASSERT(nodes[i]->src[0] == nodes[i - 1]);
        tensors[i + 1] = nodes[i]->src[1];
    }

    for (int i = 0; i <= n_nodes; ++i) {
        GGML_ASSERT(tensors[i]->type == GGML_TYPE_F32 && tensors[i]->nb[0] == sizeof(float));
    }
    ggml_tensor * dst = scale_node != nullptr ? scale_node : nodes[n_nodes - 1];
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    const int64_t nelements = ggml_nelements(dst);
    GGML_ASSERT(nelements < ((int64_t) 1 << 31));
    float scale = 1.0f;
    float bias = 0.0f;
    if (scale_node != nullptr) {
        GGML_ASSERT(scale_node->src[0] == nodes[n_nodes - 1]);
        memcpy(&scale, (const float *) scale_node->op_params + 0, sizeof(float));
        memcpy(&bias,  (const float *) scale_node->op_params + 1, sizeof(float));
    }

    bool contiguous = ggml_is_contiguous(dst);
    for (int i = 0; i <= n_nodes; ++i) {
        contiguous = contiguous && ggml_is_contiguous(tensors[i]);
    }
    if (contiguous) {
        switch (n_nodes + 1) {
            case 2: launch_add_n_f32_contiguous<2>(ctx, tensors, (float *) dst->data, nelements, scale, bias); break;
            case 3: launch_add_n_f32_contiguous<3>(ctx, tensors, (float *) dst->data, nelements, scale, bias); break;
            case 4: launch_add_n_f32_contiguous<4>(ctx, tensors, (float *) dst->data, nelements, scale, bias); break;
            case 5: launch_add_n_f32_contiguous<5>(ctx, tensors, (float *) dst->data, nelements, scale, bias); break;
            case 6: launch_add_n_f32_contiguous<6>(ctx, tensors, (float *) dst->data, nelements, scale, bias); break;
            case 7: launch_add_n_f32_contiguous<7>(ctx, tensors, (float *) dst->data, nelements, scale, bias); break;
            case 8: launch_add_n_f32_contiguous<8>(ctx, tensors, (float *) dst->data, nelements, scale, bias); break;
            case 9: launch_add_n_f32_contiguous<9>(ctx, tensors, (float *) dst->data, nelements, scale, bias); break;
            default: GGML_ABORT("invalid fused ADD operand count");
        }
        return;
    }

    int n_linear = 0;
    while (n_linear <= n_nodes && ggml_is_contiguous(tensors[n_linear])) {
        ++n_linear;
    }
    bool grouped = ggml_is_contiguous(dst) && n_linear <= 1;
    for (int i = n_linear + 1; grouped && i <= n_nodes; ++i) {
        grouped = ggml_are_same_stride(tensors[n_linear], tensors[i]);
    }
    if (grouped) {
        switch (n_nodes + 1) {
            case 2: launch_add_n_f32_grouped<2>(ctx, tensors, (float *) dst->data, nelements, scale, bias, n_linear); break;
            case 3: launch_add_n_f32_grouped<3>(ctx, tensors, (float *) dst->data, nelements, scale, bias, n_linear); break;
            case 4: launch_add_n_f32_grouped<4>(ctx, tensors, (float *) dst->data, nelements, scale, bias, n_linear); break;
            case 5: launch_add_n_f32_grouped<5>(ctx, tensors, (float *) dst->data, nelements, scale, bias, n_linear); break;
            case 6: launch_add_n_f32_grouped<6>(ctx, tensors, (float *) dst->data, nelements, scale, bias, n_linear); break;
            case 7: launch_add_n_f32_grouped<7>(ctx, tensors, (float *) dst->data, nelements, scale, bias, n_linear); break;
            case 8: launch_add_n_f32_grouped<8>(ctx, tensors, (float *) dst->data, nelements, scale, bias, n_linear); break;
            case 9: launch_add_n_f32_grouped<9>(ctx, tensors, (float *) dst->data, nelements, scale, bias, n_linear); break;
            default: GGML_ABORT("invalid fused ADD operand count");
        }
        return;
    }

    switch (n_nodes + 1) {
        case 2: launch_add_n_f32_strided<2>(ctx, tensors, dst, nelements, scale, bias); break;
        case 3: launch_add_n_f32_strided<3>(ctx, tensors, dst, nelements, scale, bias); break;
        case 4: launch_add_n_f32_strided<4>(ctx, tensors, dst, nelements, scale, bias); break;
        case 5: launch_add_n_f32_strided<5>(ctx, tensors, dst, nelements, scale, bias); break;
        case 6: launch_add_n_f32_strided<6>(ctx, tensors, dst, nelements, scale, bias); break;
        case 7: launch_add_n_f32_strided<7>(ctx, tensors, dst, nelements, scale, bias); break;
        case 8: launch_add_n_f32_strided<8>(ctx, tensors, dst, nelements, scale, bias); break;
        case 9: launch_add_n_f32_strided<9>(ctx, tensors, dst, nelements, scale, bias); break;
        default: GGML_ABORT("invalid fused ADD operand count");
    }
}
