#ifndef GGML_SYCL_FWHT_HPP
#define GGML_SYCL_FWHT_HPP

#include "common.hpp"

// Fast Walsh-Hadamard transform, the fast path for a MUL_MAT whose src0 ggml has
// tagged GGML_HINT_SRC0_IS_HADAMARD. src0 is not read at all. Returns false if the
// shape is not one this can serve, in which case the caller must fall through to the
// ordinary mat-mul dispatch.
bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// The Hadamard MUL_MAT at node_idx and the SET_ROWS into a q8_0 cache that is its only reader, as one launch
// (GGML_SYCL_FUSE_FWHT_SET_ROWS). Returns the number of nodes it covers, or 0.
int ggml_sycl_fwht_set_rows(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);
// true when the SET_ROWS at node_idx was already written by ggml_sycl_fwht_set_rows at its MUL_MAT
bool ggml_sycl_fwht_set_rows_done(const ggml_cgraph * cgraph, int node_idx);

#endif  // GGML_SYCL_FWHT_HPP
