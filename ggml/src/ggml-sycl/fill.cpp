#include "fill.hpp"
#include "common.hpp"
#include "ggml-impl.h"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
static void fill_kernel(T * dst, const int64_t k, const T value,
                        const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t)item.get_global_id(0);
    if (i >= k) {
        return;
    }
    dst[i] = value;
}

inline void ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    float value;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    void * dst_d = dst->data;

    switch (dst->type) {
        case GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(num_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<float *>(dst_d), k, value, item);
                });
            break;
        case GGML_TYPE_F16:
            {
                sycl::half h_value = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(num_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_value, item);
                    });
            }
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}

void ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}

bool ggml_sycl_constant_input(const ggml_cgraph * graph, const ggml_tensor * input, float & value,
                              const ggml_tensor * member) {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_CONSTPROP_CONSUMER", 1) != 0;
    if (!enabled || !g_ggml_sycl_enable_fusion || !input || (input->type != GGML_TYPE_F32 && input->type != GGML_TYPE_F16)) {
        return false;
    }
    bool found = member == nullptr;
    const ggml_type type = input->type;
    for (int depth = 0; input && depth < 16; ++depth) {
        const size_t pos = ggml_hash_find(&graph->visited_hash_set, input);
        if (input->type != type || (input->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) ||
            pos == GGML_HASHSET_FULL || !ggml_bitset_get(graph->visited_hash_set.used, pos) || graph->use_counts[pos] != 1) {
            return false;
        }
        found |= input == member;
        if (input->op == GGML_OP_FILL && !input->view_src) {
            value = ggml_get_op_params_f32(input, 0);
            if (type == GGML_TYPE_F16) {
                value = static_cast<float>(sycl::half(value));
            }
            return found;
        }
        switch (input->op) {
            case GGML_OP_REPEAT:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                input = input->src[0];
                break;
            default:
                return false;
        }
    }
    return false;
}

bool ggml_sycl_fill_set_rows(const ggml_cgraph * graph, const ggml_tensor * dst, float & initial, float & value) {
    if (dst->op != GGML_OP_SET_ROWS || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) ||
        !ggml_is_contiguous(dst) || dst->ne[1] < 1 || dst->ne[1] > 65536 ||
        dst->ne[0] < 1 || dst->ne[0] > 65536 / dst->ne[1] ||
        (dst->src[1]->type != GGML_TYPE_I32 && dst->src[1]->type != GGML_TYPE_I64)) {
        return false;
    }
    return ggml_sycl_constant_input(graph, dst->src[0], value) &&
           ggml_sycl_constant_input(graph, dst->src[2], initial);
}

int ggml_sycl_fill_absorbs(const ggml_cgraph * graph, int node_idx, bool allocation) {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_CONSTPROP_CONSUMER", 1) != 0;
    if (!enabled || !g_ggml_sycl_enable_fusion) {
        return 0;
    }
    const ggml_tensor * node = graph->nodes[node_idx];
    if (node->op != GGML_OP_FILL && node->op != GGML_OP_REPEAT) {
        return 0;
    }
    for (int j = node_idx + 1; j < graph->n_nodes && j <= node_idx + 128; ++j) {
        const ggml_tensor * consumer = graph->nodes[j];
        float initial, value;
        if (consumer->op == GGML_OP_LIGHTNING_INDEXER && consumer->src[2]->type == GGML_TYPE_F32 &&
            ggml_sycl_constant_input(graph, consumer->src[2], value, node)) {
            return 1;
        }
        if (ggml_sycl_fill_set_rows(graph, consumer, initial, value) &&
            (ggml_sycl_constant_input(graph, consumer->src[0], value, node) ||
             ggml_sycl_constant_input(graph, consumer->src[2], initial, node))) {
            return allocation && node == consumer->view_src ? 0 : 1;
        }
    }
    return 0;
}
