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

#ifndef GGML_SYCL_DMMV_HPP
#define GGML_SYCL_DMMV_HPP

#include "common.hpp"


void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Reordered MoE expert mat-vec with the f32 activation (ESIMD). Returns false for an unsupported type.
bool ggml_sycl_mul_mat_vec_q_id_reorder_esimd(
    enum ggml_type src0_type, const void * vx_base, const float * y, const int32_t * ids_dev,
    float * dst_base, int ncols, int nrows, int n_experts_used, int n_tokens,
    size_t expert_weight_stride, size_t dst_row_stride, size_t src1_row_stride,
    size_t ids_token_stride, size_t dst_token_stride, size_t src1_token_stride,
    dpct::queue_ptr stream);

// Same, for the gate and up weights with the GLU folded in (IQ3_S, SWIGLU).
bool ggml_sycl_mul_mat_vec_q_id_reorder_glu_esimd(
    enum ggml_type src0_type, const void * vx_gate_base, const void * vx_up_base, const float * y,
    const int32_t * ids_dev, float * dst_base, int ncols, int nrows, int n_experts_used, int n_tokens,
    size_t expert_weight_stride, size_t dst_row_stride, size_t src1_row_stride,
    size_t ids_token_stride, size_t dst_token_stride, size_t src1_token_stride,
    ggml_glu_op glu_op, dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
