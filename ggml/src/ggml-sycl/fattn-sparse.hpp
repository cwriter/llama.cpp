#ifndef GGML_SYCL_FATTN_SPARSE_HPP
#define GGML_SYCL_FATTN_SPARSE_HPP

#include "common.hpp"

// op_params[4] of a node whose K/V were gathered: the visible cells come first and every slot after
// them is masked, so a kernel may scan the mask for where the visible cells end
#define GGML_SYCL_FATTN_GATHERED (-1)

// Gather the K/V rows selected by a sparse mask and re-dispatch the dense
// kernels onto them. Returns false if the caller should use the dense path.
bool ggml_sycl_flash_attn_ext_sparse(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// True when ggml_sycl_flash_attn_ext_sparse() takes this node.
bool ggml_sycl_fattn_sparse_applies(const ggml_tensor * dst);

// Bytes after dst->data the gathered node uses (its output and kernel scratch), 0 if the path does not apply.
size_t ggml_sycl_fattn_sparse_alloc_size(const ggml_tensor * dst);

// The most the gathered node of dst uses for any batch the path takes over at most n_kv cells, 0 if none.
size_t ggml_sycl_fattn_sparse_max_alloc_size(const ggml_tensor * dst, int64_t n_kv);

#endif // GGML_SYCL_FATTN_SPARSE_HPP
