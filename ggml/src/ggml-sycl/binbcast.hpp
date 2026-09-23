#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#include "common.hpp"


static __dpct_inline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __dpct_inline__ float op_add(const float a, const float b) {
    return a + b;
}

static __dpct_inline__ float op_sub(const float a, const float b) {
    return a - b;
}

static __dpct_inline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __dpct_inline__ float op_div(const float a, const float b) {
    return a / b;
}

void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1);
void ggml_sycl_op_mul_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * mul, ggml_tensor * add);

void ggml_sycl_op_add_n_fused(ggml_backend_sycl_context & ctx, ggml_tensor * const * nodes, int n_nodes,
                              ggml_tensor * scale = nullptr);

// Type combinations the standalone SYCL add() kernel can run. Fused ADD+ADD
// uses the same set; anything else falls back to two add() launches.
inline bool ggml_sycl_add_kernel_supports(enum ggml_type src0, enum ggml_type src1, enum ggml_type dst) {
    if (src0 == GGML_TYPE_F32 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F32) {
        return true;
    }
    if (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F16 && dst == GGML_TYPE_F16) {
        return true;
    }
    if (src0 == GGML_TYPE_F16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_F16) {
        return true;
    }
    if (src0 == GGML_TYPE_I32 && src1 == GGML_TYPE_I32 && dst == GGML_TYPE_I32) {
        return true;
    }
    if (src0 == GGML_TYPE_I16 && src1 == GGML_TYPE_I16 && dst == GGML_TYPE_I16) {
        return true;
    }
#ifdef GGML_SYCL_HAS_BF16
    if (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_BF16 && dst == GGML_TYPE_BF16) {
        return true;
    }
    if (src0 == GGML_TYPE_BF16 && src1 == GGML_TYPE_F32 && dst == GGML_TYPE_BF16) {
        return true;
    }
#endif
    return false;
}

// Shape-only test for the f16->f32 cast chain that ends in an ADD, without the flag gate.
bool ggml_sycl_cast_add_shape(const ggml_cgraph * cgraph, int i, int * span);

// True if node i starts an f16->f32 cast that the following ADD can read directly.
// span, if given, gets the number of nodes in the chain (2 without a reshape, 3 with one).
bool ggml_sycl_can_fuse_cast_add(const ggml_cgraph * cgraph, int i, int * span);

// Fuses an f16->f32 cast into the ADD that consumes it, reading the f16 source directly.
// Returns the number of extra nodes consumed, or 0 if the pattern does not match.
int ggml_sycl_fuse_cast_add(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

// True if node i is a CONT of a permuted tensor that only the following ADD reads.
// span gets the number of nodes in the chain, cast_span the part an f16 mask cast takes.
bool ggml_sycl_can_fuse_cont_add(const ggml_cgraph * cgraph, int i, int * span, int * cast_span);

// Fuses that CONT into the ADD, which walks the permuted view instead of a copy.
// Returns the number of extra nodes consumed, or 0 if the pattern does not match.
int ggml_sycl_fuse_cont_add(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i);

#endif //GGML_SYCL_BINBCAST_HPP

