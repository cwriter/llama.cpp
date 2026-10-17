#ifndef GGML_SYCL_NORM_ROPE_HPP
#define GGML_SYCL_NORM_ROPE_HPP

#include "common.hpp"

// RMS_NORM -> MUL(weight) -> ROPE at node_idx as one kernel, one work-group per row: the q/k norms
// of attention layers. Returns the number of nodes run (3), or 0 if the chain is not one it takes.
int ggml_sycl_rms_norm_rope(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_NORM_ROPE_HPP
