#ifndef GGML_SYCL_QSA_KPOOL_HPP
#define GGML_SYCL_QSA_KPOOL_HPP

#include "common.hpp"

// 1 when node_idx is a step of a QSA indexer key chain that one kernel runs at the chain's last
// SET_ROWS: the key | zeros row write into the q8_0 cache, or the pooled key (gather, mean, norm,
// rope) written into its pooled slot
int ggml_sycl_qsa_kpool_absorbs(const ggml_cgraph * cgraph, int node_idx);

// Runs that kernel if node_idx is the SET_ROWS of such a chain. Returns false otherwise.
bool ggml_sycl_qsa_kpool(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_QSA_KPOOL_HPP
