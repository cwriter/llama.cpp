#ifndef GGML_SYCL_QSA_LIVE_HPP
#define GGML_SYCL_QSA_LIVE_HPP

#include "common.hpp"

// 1 when node_idx is an elementwise step of a QSA selection list chain (which slots are live, and
// where each dead slot is parked) that one kernel computes at the chain's final cast to I32
int ggml_sycl_qsa_live_absorbs(const ggml_cgraph * cgraph, int node_idx);

// Runs that kernel if node_idx is the final cast of such a chain. Returns false otherwise.
bool ggml_sycl_qsa_live(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_QSA_LIVE_HPP
