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

void ggml_sycl_op_add_n_fused(ggml_backend_sycl_context & ctx, ggml_tensor * const * nodes, int n_nodes,
                              ggml_tensor * scale = nullptr);

void ggml_sycl_op_scale_sigmoid_scale_fused(ggml_backend_sycl_context & ctx, const ggml_tensor * input,
                                             ggml_tensor * dst, float scale_in, float scale_out);

// Fused REPEAT + MUL + ADD of the hyper-connection combine step:
//   combine = residual + repeat(block_out) * weight
// where repeat() fans [n_embd, 1, nt] out over hc streams and weight is [1, hc, nt].
// The REPEAT node itself is elided: the kernel reads block_out directly and never
// materializes the repeated buffer. The pending state lets the dispatch loop fire
// the fused kernel at the MUL node, once the weight chain (scale/sigmoid/scale)
// between the REPEAT and the MUL has produced `weight`.
struct ggml_sycl_repeat_mul_add_fused {
    bool                  active     = false; // a REPEAT was matched and is pending
    const ggml_tensor *   block_out  = nullptr; // REPEAT source, contiguous [n_embd, 1, nt]
    const ggml_tensor *   weight     = nullptr; // MUL broadcast operand, contiguous [1, hc, nt]
    const ggml_tensor *   addend     = nullptr; // ADD residual, same shape as dst
    const ggml_tensor *   dst        = nullptr; // ADD result tensor
    const ggml_tensor *   mul        = nullptr; // the MUL node the launch binds to
    const ggml_tensor *   weight_scale_in  = nullptr;
    const ggml_tensor *   weight_sigmoid   = nullptr;
    const ggml_tensor *   weight_scale_out = nullptr;
    float                 scale_in         = 1.0f;
    float                 scale_out        = 1.0f;
    unsigned short        skip       = 0;       // nodes after MUL to elide (the ADD)
};

void ggml_sycl_op_repeat_mul_add_fused(ggml_backend_sycl_context & ctx,
                                       const ggml_sycl_repeat_mul_add_fused & p);

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

#endif //GGML_SYCL_BINBCAST_HPP

