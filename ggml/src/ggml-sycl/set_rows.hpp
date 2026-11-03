#ifndef GGML_SYCL_SET_ROWS_HPP
#define GGML_SYCL_SET_ROWS_HPP

#include "common.hpp"

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_set_rows_filled(ggml_backend_sycl_context & ctx, ggml_tensor * dst, float initial, float value);
// ADD(VIEW(constant SET_ROWS), addend) in one kernel, see ggml_sycl_fill_set_rows_add
struct ggml_sycl_fill_set_rows_fold;
void ggml_sycl_op_set_rows_filled_add(ggml_backend_sycl_context & ctx, ggml_tensor * add,
                                      const ggml_sycl_fill_set_rows_fold & f);

#endif // GGML_SYCL_SET_ROWS_HPP
