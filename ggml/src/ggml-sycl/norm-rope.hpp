#ifndef GGML_SYCL_NORM_ROPE_HPP
#define GGML_SYCL_NORM_ROPE_HPP

#include "common.hpp"

// RMS_NORM -> MUL(weight) -> ROPE at node_idx as one kernel, one work-group per row: the q/k norms
// of attention layers. Returns the number of nodes run (3), or 0 if the chain is not one it takes.
int ggml_sycl_rms_norm_rope(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// The span the fusion at node_idx covers (0 if none), the norm's input it reads and the last node it writes:
// with GGML_SYCL_FUSE_NORM_ROPE_ROT it continues through the Hadamard rotation that ggml tags on a MUL_MAT
// and the q8_0 cache write of the rotated rows. graph_optimize keeps x live until the last node.
int ggml_sycl_rms_norm_rope_span(const ggml_cgraph * cgraph, int node_idx, const ggml_tensor ** x, const ggml_tensor ** last);

#endif // GGML_SYCL_NORM_ROPE_HPP
