#ifndef GGML_SYCL_FUSED_GEMM_HPP
#define GGML_SYCL_FUSED_GEMM_HPP

#include "common.hpp"

// dst[n*ldd + m] = sum_k dequant(src0)[m*K + k] * src1_f16[n*K + k]
// Returns false when the case is not handled (type, device, or shape).
bool ggml_sycl_fused_dequant_gemm_f16(ggml_type src0_type, const void * src0, const sycl::half * src1_f16, float * dst,
                                      int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                                      dpct::queue_ptr stream);

#endif // GGML_SYCL_FUSED_GEMM_HPP
