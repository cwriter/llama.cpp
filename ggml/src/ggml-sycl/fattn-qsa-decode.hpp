#ifndef GGML_SYCL_FATTN_QSA_DECODE_HPP
#define GGML_SYCL_FATTN_QSA_DECODE_HPP

#include "common.hpp"

// Decode and short batches of a QSA layer: every query row attends to the cells its list names,
// read straight from the cache (q8_0 or f16), in two launches. mask is the causal mask (f16 or
// packed bits), idx the selection list. Returns false if this node is not one it takes; the
// caller then runs the dense path.
bool ggml_sycl_qsa_decode_fa(ggml_backend_sycl_context & ctx, ggml_tensor * fa, const ggml_tensor * mask,
                             const ggml_tensor * idx);

#endif // GGML_SYCL_FATTN_QSA_DECODE_HPP
