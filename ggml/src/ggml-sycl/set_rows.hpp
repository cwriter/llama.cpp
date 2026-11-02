#ifndef GGML_SYCL_SET_ROWS_HPP
#define GGML_SYCL_SET_ROWS_HPP

#include "common.hpp"

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_set_rows_filled(ggml_backend_sycl_context & ctx, ggml_tensor * dst, float initial, float value);

#endif // GGML_SYCL_SET_ROWS_HPP
