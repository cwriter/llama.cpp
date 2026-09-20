#ifndef GGML_SYCL_QSA_MASK_HPP
#define GGML_SYCL_QSA_MASK_HPP

#include "common.hpp"

int ggml_sycl_qsa_mask_absorbs(const ggml_cgraph * cgraph, int node_idx);
int ggml_sycl_fuse_qsa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

// Runs a flash attention node whose dense QSA mask was never built, from the causal mask plus
// the selection bitmap the chain left behind. Returns false if this node is not one of those.
bool ggml_sycl_qsa_fa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_QSA_MASK_HPP
