#ifndef GGML_SYCL_LIGHTNING_INDEXER_HPP
#define GGML_SYCL_LIGHTNING_INDEXER_HPP

#include "common.hpp"

void ggml_sycl_op_lightning_indexer(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const float * value = nullptr);

// K = VIEW/RESHAPE(GET_ROWS(q8_0 rows of 128 values, I32 index)) from its first row: the GET_ROWS, or null.
// The indexer then reads the q8_0 rows through the index itself (bit-identical to the GET_ROWS dequant), so the
// gathered f32 tensor need not exist; ggml_sycl_li_gather_absorbs() decides when it is left unallocated.
const ggml_tensor * ggml_sycl_li_k_gather(const ggml_tensor * k);

#endif // GGML_SYCL_LIGHTNING_INDEXER_HPP
