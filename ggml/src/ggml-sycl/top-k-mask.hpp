#ifndef GGML_SYCL_TOP_K_MASK_HPP
#define GGML_SYCL_TOP_K_MASK_HPP

#include "common.hpp"

// GGML_OP_TOP_K_MASK (see ggml.h): every row starts at -inf, then the mask cells of the live picks
// and the extra cells are copied in. The mask may be the packed causal mask (kq-mask-bits.hpp).
bool ggml_sycl_top_k_mask_supported(const ggml_tensor * op);
void ggml_sycl_op_top_k_mask(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// The scores of the picks of `top` in pick order, [n_top, n_rows], when the tiled lightning indexer
// ran it (ggml-sycl.cpp) and so never built the score tensor; null otherwise. A TOP_K_MASK on that
// top-k reads its liveness from here.
const float * ggml_sycl_indexer_picked(const ggml_backend_sycl_context & ctx, const ggml_tensor * top);

#endif // GGML_SYCL_TOP_K_MASK_HPP
