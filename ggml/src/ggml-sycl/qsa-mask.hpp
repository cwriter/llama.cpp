#ifndef GGML_SYCL_QSA_MASK_HPP
#define GGML_SYCL_QSA_MASK_HPP

#include "common.hpp"

int ggml_sycl_qsa_mask_absorbs(const ggml_cgraph * cgraph, int node_idx);
int ggml_sycl_fuse_qsa_mask(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_QSA_MASK_HPP
