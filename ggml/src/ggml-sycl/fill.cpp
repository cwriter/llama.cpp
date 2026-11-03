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

// a node nothing else reads: one use in the graph and no input/output flag
static bool fill_private(const ggml_cgraph * graph, const ggml_tensor * t) {
    if (!t || (t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT))) {
        return false;
    }
    const size_t pos = ggml_hash_find(&graph->visited_hash_set, t);
    return pos != GGML_HASHSET_FULL && ggml_bitset_get(graph->visited_hash_set.used, pos) && graph->use_counts[pos] == 1;
}

static bool constprop_enabled() {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_CONSTPROP_CONSUMER", 1) != 0;
    return enabled && g_ggml_sycl_enable_fusion;
}

bool ggml_sycl_constant_input(const ggml_cgraph * graph, const ggml_tensor * input, float & value,
                              const ggml_tensor * member) {
    if (!constprop_enabled() || !input || (input->type != GGML_TYPE_F32 && input->type != GGML_TYPE_F16)) {
        return false;
    }
    bool found = member == nullptr;
    const ggml_type type = input->type;
    for (int depth = 0; input && depth < 16; ++depth) {
        if (input->type != type || !fill_private(graph, input)) {
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
    // the kernel keeps one selection bit per row in local memory, 32 KiB at the limit
    if (dst->op != GGML_OP_SET_ROWS || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) ||
        !ggml_is_contiguous(dst) || dst->ne[0] < 1 || dst->ne[1] < 1 || dst->ne[1] > GGML_SYCL_FILL_SET_ROWS_MAX_ROWS ||
        (dst->src[1]->type != GGML_TYPE_I32 && dst->src[1]->type != GGML_TYPE_I64)) {
        return false;
    }
    // the kernel writes dst and skips the fill under it, so dst has to be the whole fill: a view of
    // the output may reach any part of it
    if (!dst->view_src || dst->view_offs != 0 || ggml_nbytes(dst) != ggml_nbytes(dst->view_src)) {
        return false;
    }
    return ggml_sycl_constant_input(graph, dst->src[0], value) &&
           ggml_sycl_constant_input(graph, dst->src[2], initial);
}

// The constant SET_ROWS this ADD operand reads in whole through views only, or null
static const ggml_tensor * fill_set_rows_read_by(const ggml_cgraph * graph, const ggml_tensor * t) {
    if (!t || !t->view_src || t->view_offs != 0 || !ggml_is_contiguous(t)) {
        return nullptr;
    }
    const ggml_tensor * v = t;
    for (int depth = 0; v && depth < 8 && (v->op == GGML_OP_VIEW || v->op == GGML_OP_RESHAPE); ++depth) {
        if (!fill_private(graph, v)) {
            return nullptr;
        }
        v = v->src[0];
    }
    if (!v || v->op != GGML_OP_SET_ROWS || !fill_private(graph, v) || ggml_nelements(v) != ggml_nelements(t)) {
        return nullptr;
    }
    return v;
}

bool ggml_sycl_fill_set_rows_add(const ggml_cgraph * graph, const ggml_tensor * add, ggml_sycl_fill_set_rows_fold & f) {
    if (!add || add->op != GGML_OP_ADD || add->view_src || !ggml_is_contiguous(add) ||
        (add->type != GGML_TYPE_F16 && add->type != GGML_TYPE_F32)) {
        return false;
    }
    for (int j = 0; j < 2; ++j) {
        const ggml_tensor * t     = add->src[j];
        const ggml_tensor * other = add->src[1 - j];
        const ggml_tensor * sr    = fill_set_rows_read_by(graph, t);
        if (!sr || !other || other == t || (other->type != GGML_TYPE_F16 && other->type != GGML_TYPE_F32) ||
            !ggml_is_contiguous(other) || !ggml_are_same_shape(t, add) || !ggml_are_same_shape(other, add)) {
            continue;
        }
        float initial, value;
        if (!ggml_sycl_fill_set_rows(graph, sr, initial, value)) {
            continue;
        }
        // the dense graph stores the scatter in the SET_ROWS type before the ADD reads it
        if (sr->type == GGML_TYPE_F16) {
            initial = static_cast<float>(sycl::half(initial));
            value   = static_cast<float>(sycl::half(value));
        }
        f.set_rows = sr;
        f.addend   = other;
        f.initial  = initial;
        f.value    = value;
        return true;
    }
    return false;
}

// the ADD within the window that takes the output of the SET_ROWS at sr_idx
static bool fill_set_rows_folded(const ggml_cgraph * graph, int sr_idx) {
    const ggml_tensor * sr = graph->nodes[sr_idx];
    for (int j = sr_idx + 1; j < graph->n_nodes && j <= sr_idx + 16; ++j) {
        ggml_sycl_fill_set_rows_fold f;
        if (graph->nodes[j]->op == GGML_OP_ADD && ggml_sycl_fill_set_rows_add(graph, graph->nodes[j], f) &&
            f.set_rows == sr) {
            return true;
        }
    }
    return false;
}

bool ggml_sycl_fill_set_rows_absorbed(const ggml_cgraph * graph, int node_idx) {
    const ggml_tensor * node = graph->nodes[node_idx];
    float initial, value;
    return node->op == GGML_OP_SET_ROWS && ggml_sycl_fill_set_rows(graph, node, initial, value) &&
           fill_set_rows_folded(graph, node_idx);
}

int ggml_sycl_fill_absorbs(const ggml_cgraph * graph, int node_idx, bool allocation) {
    if (!constprop_enabled()) {
        return 0;
    }
    const ggml_tensor * node = graph->nodes[node_idx];
    if (ggml_sycl_fill_set_rows_absorbed(graph, node_idx)) {
        return 1;
    }
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
            // the SET_ROWS output lives in the fill under it, unless an ADD takes it straight away
            return allocation && node == consumer->view_src && !fill_set_rows_folded(graph, j) ? 0 : 1;
        }
    }
    return 0;
}
