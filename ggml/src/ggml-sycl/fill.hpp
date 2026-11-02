#pragma once

#include "common.hpp"

void ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

template <typename T, bool constant>
struct ggml_sycl_value_reader {
    const char * data;
    T value;

    T operator()(size_t offset) const {
        if constexpr (constant) {
            return value;
        } else {
            return *reinterpret_cast<const T *>(data + offset);
        }
    }
};

bool ggml_sycl_constant_input(const ggml_cgraph * graph, const ggml_tensor * input, float & value,
                              const ggml_tensor * member = nullptr);
bool ggml_sycl_fill_set_rows(const ggml_cgraph * graph, const ggml_tensor * dst, float & initial, float & value);
// Keep the SET_ROWS destination allocated even when its initialization is fused.
int ggml_sycl_fill_absorbs(const ggml_cgraph * graph, int node_idx, bool allocation);
