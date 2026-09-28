#include "graph-reorder.hpp"

#include <unordered_set>
#include <vector>

#include "ggml-impl.h"

static bool ggml_sycl_reorder_is_view(const ggml_tensor * t) {
    return ggml_op_is_empty(t->op);
}

// the tensor whose memory t lives in
static const ggml_tensor * ggml_sycl_reorder_base(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

// Ops a node may be moved in front of. The dependency test below sees every read and write
// these make; anything else (flash attention with its QSA side channel, custom ops, optimizer
// steps) ends the scan instead. Same idea as h_safe in ggml_metal_graph_optimize_reorder().
static bool ggml_sycl_reorder_can_cross(const ggml_tensor * t) {
    if (ggml_sycl_reorder_is_view(t)) {
        return true;
    }
    switch (t->op) {
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_FILL:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ROPE:
        case GGML_OP_CONCAT:
        case GGML_OP_REPEAT:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_SSM_CONV:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_POST:
            return true;
        default:
            return false;
    }
}

int ggml_sycl_graph_hoist_siblings(ggml_cgraph * cgraph, const ggml_sycl_hoist_rules & rules) {
    const int n = cgraph->n_nodes;

    std::vector<ggml_tensor *> order;
    order.reserve(n);
    std::vector<uint8_t> moved(n, 0);
    int n_moved = 0;

    // what the nodes a candidate would pass over produce and read
    std::unordered_set<const ggml_tensor *> pend_out;
    std::unordered_set<const ggml_tensor *> pend_in;

    for (int i = 0; i < n; ++i) {
        if (moved[i]) {
            continue;
        }
        ggml_tensor * lead = cgraph->nodes[i];
        order.push_back(lead);

        if ((rules.pinned && rules.pinned[i]) || !rules.leader(lead, rules.user_data)) {
            continue;
        }

        pend_out.clear();
        pend_in.clear();
        int group   = 1;
        int crossed = 0;  // nodes a candidate found now would jump over

        for (int j = i + 1; j < n && j <= i + rules.window && group < rules.max_group; ++j) {
            if (moved[j]) {
                continue;  // already placed behind an earlier leader
            }
            ggml_tensor * cand = cgraph->nodes[j];

            if (!(rules.pinned && rules.pinned[j]) && rules.joins(lead, cand, rules.user_data)) {
                // read after write: an input produced, or written through a view, in between
                bool free = true;
                for (int s = 0; s < GGML_MAX_SRC && free; ++s) {
                    const ggml_tensor * src = cand->src[s];
                    if (src) {
                        free = !pend_out.count(src) && !pend_out.count(ggml_sycl_reorder_base(src));
                    }
                }
                // write after read or write: its own storage used in between
                const ggml_tensor * dst = ggml_sycl_reorder_base(cand);
                free = free && !pend_in.count(dst) && !pend_out.count(dst);

                if (free) {
                    order.push_back(cand);
                    moved[j] = 1;
                    n_moved += crossed > 0;
                    group++;
                    continue;
                }
            }

            if (!ggml_sycl_reorder_can_cross(cand)) {
                break;
            }

            crossed++;
            pend_out.insert(cand);
            // a view writes nothing; any other node that is a view writes into its source
            if (!ggml_sycl_reorder_is_view(cand) && cand->view_src) {
                pend_out.insert(cand->view_src);
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (const ggml_tensor * src = cand->src[s]) {
                    pend_in.insert(src);
                    pend_in.insert(ggml_sycl_reorder_base(src));
                }
            }
        }
    }

    GGML_ASSERT((int) order.size() == n);
    if (n_moved > 0) {
        for (int i = 0; i < n; ++i) {
            cgraph->nodes[i] = order[i];
        }
    }
    return n_moved;
}
