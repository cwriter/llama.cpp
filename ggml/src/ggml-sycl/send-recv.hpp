#ifndef GGML_SYCL_SEND_RECV_HPP
#define GGML_SYCL_SEND_RECV_HPP

#include "common.hpp"

// GGML_OP_RECV, the consumer half of a ggml_backend_sched boundary pair (GGML_SCHED_SEND_RECV).
// The expansion has to give the bits the producer's GET_ROWS would have computed, so only the types
// whose dequantization is exact in f32 here are taken; the scheduler keeps a plain copy for the rest.
bool ggml_sycl_recv_supported(const ggml_tensor * op);
void ggml_sycl_op_recv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// GGML_OP_SEND, the producer half: GGML_SEND_CODEC_BF16 and MASK_BITS (ROWS packages are made on the CPU)
bool ggml_sycl_send_supported(const ggml_tensor * op);
void ggml_sycl_op_send(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// a RECV followed by a REPEAT of it (the hc residual's copies of the token embedding) is one kernel that expands
// the package straight into the REPEAT's dst: the RECV is absorbed and never written. Only when the scheduler marked
// the REPEAT as the RECV's only reader (op_params[1])
bool ggml_sycl_recv_repeat_fusable(const ggml_cgraph * cgraph, int i);
void ggml_sycl_op_recv_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * recv, ggml_tensor * repeat);

#endif // GGML_SYCL_SEND_RECV_HPP
