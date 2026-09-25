#ifndef GGML_SYCL_FUSION_HPP
#define GGML_SYCL_FUSION_HPP

#include <initializer_list>

#include "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops,
                        std::initializer_list<enum ggml_unary_op> unary_ops);

// UNARY at `node_idx` feeding a MUL that broadcasts it over the row: the unary side has one
// value per row (ne0 == 1) and the MUL is wider. ggml_can_fuse() cannot express this - it
// requires every node in the run to have the same shape - so this is its own matcher.
// Purely structural, so ggml_backend_sycl_fusion_absorbs() can ask it at allocation time.
bool ggml_sycl_can_fuse_unary_mul_bcast(const ggml_cgraph * cgraph, int node_idx);

// RMS_NORM at `node_idx` feeding a SCALE that exists only to finish an L2 norm.
bool ggml_sycl_can_fuse_rms_norm_scale(const ggml_cgraph * cgraph, int node_idx);

#endif  // GGML_SYCL_FUSION_HPP
