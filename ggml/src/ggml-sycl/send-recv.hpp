#ifndef GGML_SYCL_SEND_RECV_HPP
#define GGML_SYCL_SEND_RECV_HPP

#include "common.hpp"

// GGML_OP_RECV, the consumer half of a ggml_backend_sched boundary pair (GGML_SCHED_SEND_RECV).
// The expansion has to give the bits the producer's GET_ROWS would have computed, so only the types
// whose dequantization is exact in f32 here are taken; the scheduler keeps a plain copy for the rest.
bool ggml_sycl_recv_supported(const ggml_tensor * op);
void ggml_sycl_op_recv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_SEND_RECV_HPP
