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
// rows of a constant SET_ROWS the fused kernel covers: one selection bit per row in local memory
#define GGML_SYCL_FILL_SET_ROWS_MAX_ROWS 262144

bool ggml_sycl_fill_set_rows(const ggml_cgraph * graph, const ggml_tensor * dst, float & initial, float & value);

// ADD(VIEW(constant SET_ROWS), addend), the DSA top-k mask chain without TOP_K_MASK: the ADD writes
// (selected ? value : initial) + addend in one kernel, and neither the SET_ROWS nor its fill is built
struct ggml_sycl_fill_set_rows_fold {
    const ggml_tensor * set_rows = nullptr;
    const ggml_tensor * addend   = nullptr;
    float               initial  = 0.0f;  // already rounded to the SET_ROWS type
    float               value    = 0.0f;
};
bool ggml_sycl_fill_set_rows_add(const ggml_cgraph * graph, const ggml_tensor * add, ggml_sycl_fill_set_rows_fold & f);
// a constant SET_ROWS whose ADD runs it (fusion_absorbs and the compute loop)
bool ggml_sycl_fill_set_rows_absorbed(const ggml_cgraph * graph, int node_idx);
// 1 for a FILL or REPEAT only a fused consumer reads, and for a constant SET_ROWS its ADD runs.
// With allocation set, the fill a SET_ROWS writes into stays allocated unless that ADD runs it.
int ggml_sycl_fill_absorbs(const ggml_cgraph * graph, int node_idx, bool allocation);
