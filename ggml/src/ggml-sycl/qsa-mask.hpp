#ifndef GGML_SYCL_QSA_MASK_HPP
#define GGML_SYCL_QSA_MASK_HPP

#include "common.hpp"

// 1 when node_idx belongs to a QSA mask chain whose flash attention reads the selection list
// instead (fusion_absorbs); the compute loop skips such nodes
int ggml_sycl_qsa_mask_absorbs(const ggml_cgraph * cgraph, int node_idx);

// Runs a flash attention node whose dense QSA mask was never built, from the causal mask plus
// the selection list. Returns false if this node is not one of those.
bool ggml_sycl_qsa_fa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_QSA_MASK_HPP
