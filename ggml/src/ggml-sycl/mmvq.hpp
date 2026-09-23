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

#ifndef GGML_SYCL_MMVQ_HPP
#define GGML_SYCL_MMVQ_HPP

#include "common.hpp"

struct ggml_sycl_moe_route_order {
    const uint32_t * expert_offsets;
    const uint32_t * sorted_routes;
    int              n_experts;
    // Experts that actually have routes, compacted, with the count in *n_active. The
    // mat-vec grid is sized by n_active_max (a host-side bound) instead of by every
    // expert, so work-groups are not launched for experts no token selected.
    const uint32_t * active_experts;
    const uint32_t * n_active;
    int              n_active_max;
    int              n_routes;      // n_tokens * n_experts_used, for shape-based dispatch
};

// Up to this many mat-vecs that read one activation are fused into a single launch.
#define GGML_SYCL_MMVQ_MULTI_MAX 3

// MoE counterpart: several expert-weight tensors that share one activation, one id table and
// one geometry. Only the unordered mat-vec path takes more than one weight.
struct ggml_sycl_mmvq_moe_multi {
    const void * vx_base[GGML_SYCL_MMVQ_MULTI_MAX];
    float *      dst_base[GGML_SYCL_MMVQ_MULTI_MAX];
    int          n_mats;
};

void ggml_sycl_build_moe_route_order(
    const int32_t * ids_dev,
    size_t          ids_token_stride,
    int             n_experts,
    int             n_experts_used,
    int             n_tokens,
    uint32_t *      expert_counts,
    uint32_t *      expert_offsets,
    uint32_t *      expert_cursors,
    uint32_t *      sorted_routes,
    uint32_t *      active_experts,
    dpct::queue_ptr stream);

void ggml_sycl_op_mul_mat_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Requires standard (non-reorder) block layout for src0.
// Returns false if src0_type isn't handled; caller should fall back.
bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,             // start of stacked expert weights
    const void *       vy,                  // pre-quantized src1 (Q8_1)
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    int                n_tokens,
    size_t             expert_weight_stride, // bytes between experts in vx_base
    size_t             dst_row_stride,       // bytes between dst rows
    size_t             src1_row_stride,      // 0 = shared src1 within a token
    size_t             ids_token_stride,
    size_t             dst_token_stride,
    size_t             src1_token_stride,
    const ggml_sycl_moe_route_order * route_order,
    // extra weights sharing vy and ids_dev; nullptr or n_mats <= 1 means vx_base/dst_base only
    const ggml_sycl_mmvq_moe_multi * multi,
    dpct::queue_ptr    stream);

// True if the mat-vec MoE entry points handle src0_type, i.e. whether MUL_MAT_ID runs
// entirely on device. The reorder variant covers fewer types than the plain one.
bool ggml_sycl_mul_mat_vec_q_id_supports_type(enum ggml_type src0_type);
bool ggml_sycl_mul_mat_vec_q_id_reorder_supports_type(enum ggml_type src0_type);

// MoE mat-vec that folds a GLU over a gate/up weight pair, so neither intermediate is written
bool ggml_sycl_mul_mat_vec_q_id_reorder_glu(
    enum ggml_type src0_type, const void * vx_gate_base, const void * vx_up_base, const void * vy,
    const int32_t * ids_dev, float * dst_base, int ncols, int nrows, int n_experts_used, int n_tokens,
    size_t expert_weight_stride, size_t dst_row_stride, size_t src1_row_stride,
    size_t ids_token_stride, size_t dst_token_stride, size_t src1_token_stride,
    ggml_glu_op glu_op, dpct::queue_ptr stream);

// Reorder (SoA) variant of the fused MoE expert GEMV.
// vx_base: each expert slice (stride expert_weight_stride == src0->nb[2]) is a self-contained reorder/SoA layout.
// vy: src1 quantized with quantize_and_reorder_q8_1_soa (per-row SoA). Returns false if src0_type isn't handled.
bool ggml_sycl_mul_mat_vec_q_id_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    int                n_tokens,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    size_t             ids_token_stride,
    size_t             dst_token_stride,
    size_t             src1_token_stride,
    const ggml_sycl_moe_route_order * route_order,
    dpct::queue_ptr    stream);

// A group of mat-vecs that share one quantized activation. The weights are separate tensors
// with their own row counts, so the grid spans the concatenation of the output rows and each
// sub-group maps its row back to (weight, row) through row_begin.
struct ggml_sycl_mmvq_multi {
    const void * vx[GGML_SYCL_MMVQ_MULTI_MAX];
    float *      dst[GGML_SYCL_MMVQ_MULTI_MAX];
    int          nrows[GGML_SYCL_MMVQ_MULTI_MAX];
    int          row_begin[GGML_SYCL_MMVQ_MULTI_MAX];
    int          stride_col_dst[GGML_SYCL_MMVQ_MULTI_MAX];
    int          n_mats;
    int          nrows_total;
};

// True if the grouped reorder mat-vec below has a kernel for src0_type.
bool ggml_sycl_mul_mat_vec_q_multi_reorder_supports_type(enum ggml_type src0_type);

// Reorder (SoA) layout, one activation quantized once, one launch per group.
// All weights must share src0_type and ncols. Returns false if unhandled.
bool ggml_sycl_mul_mat_vec_q_multi_reorder(
    enum ggml_type               src0_type,
    const ggml_sycl_mmvq_multi & mats,
    const void *                 vy,
    int                          ncols,
    int                          ncols_dst,
    int                          stride_col_y_bytes,
    dpct::queue_ptr              stream);

// Fused dense-FFN GEMV: writes glu(gate . y, up . y) instead of the two mat-vec results.
// vx / vgate must share shape, stride and reorder layout. Returns false if unhandled.
bool ggml_sycl_mul_mat_vec_q_glu_reorder(
    enum ggml_type     src0_type,
    enum ggml_glu_op   glu_op,
    const void *       vx,
    const void *       vgate,
    const void *       vy,
    float *            dst,
    int                ncols,                // K, shared by both weights
    int                nrows,                // output rows, i.e. weight ne[1]
    int                ncols_dst,            // activation columns, 1..MMVQ_MAX_BATCH_SIZE
    int                stride_col_y_bytes,   // bytes between activation columns in vy
    int                stride_col_dst,       // floats between output columns in dst
    dpct::queue_ptr    stream);

#endif // GGML_SYCL_MMVQ_HPP
