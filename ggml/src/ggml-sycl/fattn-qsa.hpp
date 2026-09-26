#ifndef GGML_SYCL_FATTN_QSA_HPP
#define GGML_SYCL_FATTN_QSA_HPP

#include "common.hpp"

// Flash attention for a QSA layer without the dense [n_kv, n_tps] mask the graph builds: the
// node takes the top-k list and the causal mask instead, and attends to the listed cells that
// the mask does not hide. A row with no visible cell gives 0, as the CPU reference does.
// Prefill runs XMX GEMMs over the union of a query tile's lists (fattn-qsa.cpp); everything
// else writes the mask at its real size into pool scratch and runs the dense kernels.

// Structural test of the flash attention node only: types, shapes and op params. Holds for every
// later execution of the same node, so the fusion_absorbs side and the compute side agree.
bool ggml_sycl_qsa_sparse_fa_supported(const ggml_tensor * fa);

// mask: the causal kq_mask [n_kv, n_tps, 1, n_stream]; idx: I32 [width, n_tps, n_stream]
void ggml_sycl_qsa_sparse_fa(ggml_backend_sycl_context & ctx, ggml_tensor * fa,
                             const ggml_tensor * mask, const ggml_tensor * idx);

#endif // GGML_SYCL_FATTN_QSA_HPP
