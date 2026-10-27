#include "fusion.hpp"
#include "binbcast.hpp"

#include <algorithm>
#include <cstring>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
static bool ggml_sycl_should_fuse_mul_mat_glu(const ggml_tensor * gate, const ggml_tensor * up,
                                              const ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    const ggml_glu_op glu_op = ggml_get_glu_op(glu);
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    if (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        return false;
    }

    const ggml_tensor * wu  = up->src[0];
    const ggml_tensor * wg  = gate->src[0];
    const ggml_tensor * act = up->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    if (!ggml_are_same_shape(wu, wg)) {
        return false;
    }
    if (act != gate->src[1]) {
        return false;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    const bool reorder_pair = wu->type == wg->type && (wu->type == GGML_TYPE_Q4_K ||
                              (wu->type == GGML_TYPE_Q8_0 && glu_op == GGML_GLU_OP_SWIGLU));
    const bool plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
#ifdef GGML_SYCL_MMVQ_HAS_XMX
    // same-type pairs of the types of the XMX kernels, which also take a few more columns
    // (the XMX kernels walk whole QK_K super-blocks, so they need rows of a multiple of QK_K)
    const bool xmx_pair = wu->type == wg->type && ggml_sycl_xmx_supports_type(wu->type) && wu->ne[0] % QK_K == 0;
#else
    const bool xmx_pair = false;
#endif // GGML_SYCL_MMVQ_HAS_XMX
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    const bool esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#else
    const bool esimd_pair   = false;
#endif
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    const int64_t row_align = wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    if ((!reorder_pair && !plain_pair && !xmx_pair && !esimd_pair) || wu->ne[0] % row_align != 0) {
        return false;
    }

    // one 2D weight matrix in and a plain column stride out: no broadcast or padding
    if (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        return false;
    }
    if (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return false;
    }
    if (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        return false;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    if (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        return false;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    const int max_cols = xmx_pair ? std::max(MMVQ_MAX_BATCH_SIZE, g_ggml_sycl_xmx_glu_max_cols) : MMVQ_MAX_BATCH_SIZE;
    if (act->ne[1] > max_cols) {
        return false;
    }
    // the q8_0 reorder GLU kernel always serves one and two columns; 3..8 (MTP verification) are
    // the same kernel template at more columns, behind their own bit
    if (wu->type == GGML_TYPE_Q8_0 && act->ne[1] > 2 && !(g_ggml_sycl_fuse_types & GGML_SYCL_FUSE_GLU_NCOLS)) {
        return false;
    }

    return true;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
static bool ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    return unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops,
                        std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    if (!g_ggml_sycl_enable_fusion) {
        return false;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            return false;
        }

        const ggml_tensor * glu  = cgraph->nodes[node_idx + 2];
        const ggml_tensor * gate = glu->src[0];
        const ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        const bool ok = (gate == cgraph->nodes[node_idx] && up == cgraph->nodes[node_idx + 1]) ||
                        (gate == cgraph->nodes[node_idx + 1] && up == cgraph->nodes[node_idx]);
        if (!ok) {
            return false;
        }

        return ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    // The hyper-connection gate: SCALE -> SIGMOID -> SCALE feeding DSV4_HC_POST's src[2].
    // ggml_can_fuse() cannot express it - the link is src[2], and hc_post has a different shape
    // than the chain - so use the subgraph form with hc_post as the only materialised output.
    if (ops.size() == 4 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_SCALE && ops.begin()[3] == GGML_OP_DSV4_HC_POST) {
        if (!(g_ggml_sycl_fuse_types & GGML_SYCL_FUSE_ELEMENTWISE)) {
            return false;
        }
        if (unary_ops.size() != 1 || unary_ops.begin()[0] != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 })) {
            return false;
        }

        const ggml_tensor * scale0  = cgraph->nodes[node_idx];
        const ggml_tensor * sigmoid = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scale1  = cgraph->nodes[node_idx + 2];
        const ggml_tensor * post    = cgraph->nodes[node_idx + 3];

        if (ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        // the chain has to be exactly that chain, and has to be what hc_post gates with
        if (sigmoid->src[0] != scale0 || scale1->src[0] != sigmoid || post->src[2] != scale1) {
            return false;
        }

        const ggml_tensor * gate_src = scale0->src[0];
        if (gate_src->type != GGML_TYPE_F32 || scale0->type != GGML_TYPE_F32 ||
            sigmoid->type != GGML_TYPE_F32 || scale1->type != GGML_TYPE_F32) {
            return false;
        }
        // hc_post indexes the gate as gate[idst*nb0 + it*nb1]; folding the chain makes it read
        // gate_src with gate_src's own strides, so the shape must be identical
        if (!ggml_are_same_shape(gate_src, scale1)) {
            return false;
        }
        if (!ggml_is_contiguous(gate_src)) {
            return false;
        }

        return true;
    }

    // A SCALE that exists only to feed the next unary.
    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1) {
        if (!(g_ggml_sycl_fuse_types & GGML_SYCL_FUSE_ELEMENTWISE)) {
            return false;
        }
        if (!ggml_can_fuse(cgraph, node_idx, ops)) {
            return false;
        }

        const ggml_tensor * scale = cgraph->nodes[node_idx];
        const ggml_tensor * unary = cgraph->nodes[node_idx + 1];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0]) {
            return false;
        }
        // the ops ggml_sycl_op_scale_unary_fused() has a kernel for
        if (unary_op != GGML_UNARY_OP_SILU && unary_op != GGML_UNARY_OP_SIGMOID &&
            unary_op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
        if (unary->src[0] != scale) {
            return false;
        }

        // SCALE is F32-only in this backend, and the fused kernel indexes every operand flat
        const ggml_tensor * x = scale->src[0];
        if (x->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 || unary->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_are_same_shape(x, unary) || !ggml_is_contiguous(x) || !ggml_is_contiguous(unary)) {
            return false;
        }

        return true;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        if (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            return false;
        }

        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[node_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        const ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        if (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_size(mul_w->type)) {
            return false;
        }

        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add != nullptr) {
            if (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                return false;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            const ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            if (!ggml_are_same_shape(add_w, add)) {
                return false;
            }

            if (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                return false;
            }
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        const ggml_tensor * add0 = cgraph->nodes[node_idx];
        const ggml_tensor * add1 = cgraph->nodes[node_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // The running sum is normally src0 of the next ADD, which keeps the fused float fold
        // matching two sequential add() launches. With add0 as src1 the unfused result is
        // src2 + (src0 + src1) and the fused one is (src0 + src1) + src2: the inner sum is
        // identical and a single IEEE754 addition commutes exactly, so the two agree bit for
        // bit. Associativity is what would not hold, and neither form re-associates.
        // GGML_SYCL_FLOAT_COMMUTATIVE=0 restores the stricter src0-only rule.
        if (add1->src[0] != add0 && !(g_ggml_sycl_float_commutative && add1->src[1] == add0)) {
            return false;
        }

        const ggml_tensor * c = add1->src[1];
        if (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            return false;
        }

        return true;
    }

    // MUL feeding an ADD: the multiply-accumulate that survives every other fusion.
    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL && ops.begin()[1] == GGML_OP_ADD) {
        if (!(g_ggml_sycl_fuse_types & GGML_SYCL_FUSE_MUL_ADD)) {
            return false;
        }
        if (!ggml_can_fuse(cgraph, node_idx, ops)) {
            return false;
        }

        const ggml_tensor * mul = cgraph->nodes[node_idx];
        const ggml_tensor * add = cgraph->nodes[node_idx + 1];

        // ggml_can_fuse() accepts the link through either source, and the fused kernel folds
        // (a*b) then + c, so the multiply has to be the operand being added, not the addend.
        const ggml_tensor * other = (add->src[0] == mul) ? add->src[1] : add->src[0];
        if (other == mul) {
            return false;
        }
        // adding a value to itself would read mul twice; the fold only produces it once
        if (add->src[0] != mul && add->src[1] != mul) {
            return false;
        }

        // the fused kernel is instantiated for f32 only, which is every pair this model emits
        if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
            other->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32) {
            return false;
        }
        // the broadcast indexing folds src1 and src2 against dst, so dst must be the wide shape
        if (!ggml_are_same_shape(mul, add)) {
            return false;
        }
        if (!ggml_sycl_add_kernel_supports(mul->type, other->type, add->type)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0]) {
            return false;
        }

        if (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        if (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            return false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        if (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx + 1];

        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }
        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        if (!ggml_is_contiguous(silu)) {
            return false;
        }

        return true;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        const ggml_tensor * add   = cgraph->nodes[node_idx];
        const ggml_tensor * unary = cgraph->nodes[node_idx + 1];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 2];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0]) {
            return false;
        }

        if (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            return false;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        const ggml_tensor * a     = add->src[0];
        const ggml_tensor * bias  = add->src[1];
        const ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        if (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            return false;
        }

        // the activation and the destination are indexed flat
        if (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            return false;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        if (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            return false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        if (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx + 2];

        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }
        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        if (!ggml_is_contiguous(silu)) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        if (!ggml_is_contiguous_rows(rms_norm) || !ggml_is_contiguous_rows(scale)) {
            return false;
        }
        return true;
    }
    return false;
}

// The shared-expert gate is one sigmoided scalar per token multiplied over the whole expert
// output, so the MUL is wider than the unary and ggml_can_fuse() declines the pair on shape.
// Match it here instead: the unary keeps one value per row and the fused kernel reads it with
// a zero column stride.
bool ggml_sycl_can_fuse_unary_mul_bcast(const ggml_cgraph * cgraph, int node_idx) {
    if (!g_ggml_sycl_enable_fusion || !(g_ggml_sycl_fuse_types & GGML_SYCL_FUSE_UNARY_MUL_B)) {
        return false;
    }
    if (node_idx + 1 >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * unary = cgraph->nodes[node_idx];
    const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];

    if (unary->op != GGML_OP_UNARY || mul->op != GGML_OP_MUL) {
        return false;
    }
    // the compute loop skips an empty node without fusing, so never report one as absorbed
    if (ggml_is_empty(unary) || ggml_is_empty(mul)) {
        return false;
    }

    // the ops ggml_sycl_op_unary_mul_fused() has a kernel for
    const ggml_unary_op unary_op = ggml_get_unary_op(unary);
    if (unary_op != GGML_UNARY_OP_SILU && unary_op != GGML_UNARY_OP_SIGMOID &&
        unary_op != GGML_UNARY_OP_SOFTPLUS) {
        return false;
    }

    // the subgraph form drops the same-shape rule that blocks the pair, and still requires the
    // unary to be used only by the MUL, so skipping it cannot strand another reader
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_UNARY, GGML_OP_MUL }, { node_idx + 1 })) {
        return false;
    }
    if (unary->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }

    if (mul->src[0] != unary && mul->src[1] != unary) {
        return false;
    }
    const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    if (other == unary) {
        return false;
    }

    if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        return false;
    }
    if (other->type != unary->type || mul->type != unary->type) {
        return false;
    }

    // one value per row on the unary side, and the MUL as wide as the other operand
    const ggml_tensor * x = unary->src[0];
    if (x->type != unary->type || !ggml_are_same_shape(x, unary)) {
        return false;
    }
    if (unary->ne[0] != 1 || mul->ne[0] == 1) {
        return false;
    }
    if (!ggml_are_same_shape(other, mul)) {
        return false;
    }
    // the kernel folds rows and columns into one flat index, so keep it to 2D
    if (unary->ne[2] != 1 || unary->ne[3] != 1 || mul->ne[2] != 1 || mul->ne[3] != 1) {
        return false;
    }
    if (unary->ne[1] != mul->ne[1]) {
        return false;
    }

    // one row stride per source comes from nb[1]; dst is written flat
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous_1(other) || !ggml_is_contiguous(mul)) {
        return false;
    }

    // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
    if (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
        return false;
    }

    return true;
}

// build_gdn_l2_norm() is ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrt(n)), so every GDN q and k
// norm costs two launches. The scale is a plain affine map over the norm output, so it folds
// into the store loop without changing the order of the two roundings.
bool ggml_sycl_can_fuse_rms_norm_scale(const ggml_cgraph * cgraph, int node_idx) {
    if (!g_ggml_sycl_enable_fusion || !(g_ggml_sycl_fuse_types & GGML_SYCL_FUSE_NORM_SCALE)) {
        return false;
    }
    if (node_idx + 1 >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * norm  = cgraph->nodes[node_idx];
    const ggml_tensor * scale = cgraph->nodes[node_idx + 1];

    if (norm->op != GGML_OP_RMS_NORM || scale->op != GGML_OP_SCALE) {
        return false;
    }
    // the compute loop skips an empty node without fusing, so never report one as absorbed
    if (ggml_is_empty(norm) || ggml_is_empty(scale)) {
        return false;
    }
    // same shape, adjacent, and the norm used only by the scale
    if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_RMS_NORM, GGML_OP_SCALE })) {
        return false;
    }
    if (norm->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    if (scale->src[0] != norm) {
        return false;
    }

    const ggml_tensor * x = norm->src[0];
    if (x->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
        return false;
    }
    // the fused kernel walks x with x's own strides and writes the scale output directly
    if (!ggml_are_same_shape(x, scale)) {
        return false;
    }
    if (!ggml_is_contiguous_rows(x) || !ggml_is_contiguous_rows(scale)) {
        return false;
    }

    return true;
}

int ggml_sycl_match_gate_view(const ggml_cgraph * cgraph, int node_idx) {
    static const int enabled = ggml_sycl_get_env("GGML_SYCL_FUSE_GATE_VIEW", 1);
    if (!enabled || !g_ggml_sycl_enable_fusion) {
        return -1;
    }
    const ggml_tensor * node = cgraph->nodes[node_idx];
    const ggml_tensor * cont = node->op == GGML_OP_UNARY ? node->src[0] : node;
    if (!cont || cont->op != GGML_OP_CONT || cont->type != GGML_TYPE_F32 ||
        (cont->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) || !ggml_is_contiguous(cont)) {
        return -1;
    }
    const ggml_tensor * x = cont->src[0];
    if (!x || x->op != GGML_OP_VIEW || x->type != GGML_TYPE_F32 || x->ne[0] <= 1 ||
        !ggml_is_contiguous_1(x) || ggml_is_contiguous(x) ||
        x->nb[1] < (size_t) x->ne[0] * sizeof(float) || x->nb[1] % sizeof(float) != 0 ||
        ggml_nelements(x) != ggml_nelements(cont) || ggml_nelements(x) >= INT32_MAX) {
        return -1;
    }
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, cont);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos) ||
        cgraph->use_counts[pos] != 1) {
        return -1;
    }
    int unary_idx = node_idx;
    if (node->op == GGML_OP_CONT) {
        for (++unary_idx; unary_idx < cgraph->n_nodes; ++unary_idx) {
            const ggml_tensor * n = cgraph->nodes[unary_idx];
            if (n->op == GGML_OP_UNARY && n->src[0] == cont) {
                break;
            }
        }
    }
    if (unary_idx + 1 >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * unary = cgraph->nodes[unary_idx];
    const ggml_tensor * mul   = cgraph->nodes[unary_idx + 1];
    if (unary->op != GGML_OP_UNARY || unary->src[0] != cont || mul->op != GGML_OP_MUL ||
        unary->view_src || mul->view_src ||
        !ggml_sycl_can_fuse(cgraph, unary_idx, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(unary) })) {
        return -1;
    }
    const ggml_tensor * other = mul->src[0] == unary ? mul->src[1] : mul->src[0];
    if (other == unary || !ggml_is_contiguous(other) || !ggml_are_same_shape(cont, unary) ||
        !ggml_are_same_shape(unary, mul)) {
        return -1;
    }
    return unary_idx;
}

bool ggml_sycl_written_between(const ggml_cgraph * cgraph, const ggml_tensor * t, int from, int to) {
    const ggml_tensor * tb = t->view_src ? t->view_src : t;
    for (int n = std::max(from + 1, 0); n < to && n < cgraph->n_nodes; ++n) {
        const ggml_tensor * c = cgraph->nodes[n];
        if (ggml_is_empty(c) || c->op == GGML_OP_RESHAPE || c->op == GGML_OP_VIEW || c->op == GGML_OP_PERMUTE ||
            c->op == GGML_OP_TRANSPOSE || c->op == GGML_OP_NONE) {
            continue;
        }
        if (c == tb || c->view_src == tb) {
            return true;
        }
    }
    return false;
}

// RMS_NORM + SCALE at norm_idx whose scale output is read only by gdn as q (src 0) or k (src 1)
static bool ggml_sycl_is_gdn_l2_norm(const ggml_cgraph * cgraph, int norm_idx, const ggml_tensor * gdn, int src) {
    if (norm_idx < 0 || !ggml_sycl_can_fuse_rms_norm_scale(cgraph, norm_idx)) {
        return false;
    }
    const ggml_tensor * norm  = cgraph->nodes[norm_idx];
    const ggml_tensor * scale = cgraph->nodes[norm_idx + 1];
    float bias = 0.0f;
    memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));
    return gdn->src[src] == scale && bias == 0.0f && !(scale->flags & GGML_TENSOR_FLAG_OUTPUT) &&
           ggml_node_get_use_count(cgraph, norm_idx + 1) == 1 && norm->src[0]->ne[0] == gdn->src[2]->ne[0];
}

bool ggml_sycl_match_gdn_qk_norm(const ggml_cgraph * cgraph, int gdn_idx, ggml_sycl_gdn_qk_norm * out) {
    static const int enabled = ggml_sycl_get_env("GGML_SYCL_GDN_QK_NORM", 1);
    if (!enabled || !g_ggml_sycl_enable_fusion) {
        return false;
    }
    const ggml_tensor * gdn = cgraph->nodes[gdn_idx];
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32) {
        return false;
    }
    int norm_idx[2] = { -1, -1 };
    for (int j = gdn_idx - 1; j >= 1 && j >= gdn_idx - 64; --j) {
        for (int s = 0; s < 2; ++s) {
            if (cgraph->nodes[j] == gdn->src[s]) {
                norm_idx[s] = j - 1;
            }
        }
    }
    if (!ggml_sycl_is_gdn_l2_norm(cgraph, norm_idx[0], gdn, 0) || !ggml_sycl_is_gdn_l2_norm(cgraph, norm_idx[1], gdn, 1)) {
        return false;
    }
    // the net reads raw q/k at its own node, so nothing may write them after their norms ran
    for (int s = 0; s < 2; ++s) {
        if (ggml_sycl_written_between(cgraph, cgraph->nodes[norm_idx[s]]->src[0], norm_idx[s], gdn_idx)) {
            return false;
        }
    }
    const ggml_tensor * nq = cgraph->nodes[norm_idx[0]];
    const ggml_tensor * nk = cgraph->nodes[norm_idx[1]];
    const ggml_tensor * sq = cgraph->nodes[norm_idx[0] + 1];
    const ggml_tensor * sk = cgraph->nodes[norm_idx[1] + 1];
    // one eps and one scale for both rows, and the kernel walks q and k with the same strides
    if (memcmp(nq->op_params, nk->op_params, sizeof(float)) != 0 || memcmp(sq->op_params, sk->op_params, sizeof(float)) != 0 ||
        !ggml_are_same_stride(nq->src[0], nk->src[0])) {
        return false;
    }
    float scale = 0.0f;
    memcpy(&scale, sq->op_params, sizeof(float));
    if (scale == 0.0f) {
        return false; // the kernel takes a zero scale to mean no norm
    }
    if (out) {
        out->gdn = gdn;
        out->q   = nq->src[0];
        out->k   = nk->src[0];
        memcpy(&out->eps, nq->op_params, sizeof(float));
        memcpy(&out->scale, sq->op_params, sizeof(float));
    }
    return true;
}

bool ggml_sycl_gdn_qk_norm_now(const ggml_tensor * gdn) {
    // a long batch walks its tokens one by one, and every column would redo the norms of each token
    return gdn->src[2]->ne[2] <= 8;
}

int ggml_sycl_gdn_qk_norm_reader(const ggml_cgraph * cgraph, int node_idx) {
    if (cgraph->nodes[node_idx]->op != GGML_OP_RMS_NORM || node_idx + 1 >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * scale = cgraph->nodes[node_idx + 1];
    for (int j = node_idx + 2; j < cgraph->n_nodes && j <= node_idx + 64; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_GATED_DELTA_NET && (n->src[0] == scale || n->src[1] == scale)) {
            return ggml_sycl_match_gdn_qk_norm(cgraph, j, nullptr) ? j : -1;
        }
    }
    return -1;
}


bool ggml_sycl_match_rms_norm_gate(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_norm_gate_match & m) {
    static const int enabled = ggml_sycl_get_env("GGML_SYCL_FUSE_NORM_GATE", 1);
    if (!enabled || !g_ggml_sycl_enable_fusion) {
        return false;
    }
    const ggml_tensor * rms = cgraph->nodes[node_idx];
    if (rms->op != GGML_OP_RMS_NORM || rms->type != GGML_TYPE_F32 || rms->src[0]->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(rms->src[0]) || (rms->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, node_idx) != 1) {
        return false;
    }
    auto is_gate = [&](const ggml_tensor * u) {
        return u->op == GGML_OP_UNARY && (ggml_get_unary_op(u) == GGML_UNARY_OP_SIGMOID || ggml_get_unary_op(u) == GGML_UNARY_OP_SILU) &&
               u->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(u->src[0]) && ggml_are_same_shape(u->src[0], rms);
    };
    ggml_sycl_norm_gate_match r;
    r.rms = rms;
    for (int j = node_idx + 1; j < cgraph->n_nodes && j <= node_idx + 8 && r.out == nullptr; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
            t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE) {
            continue;
        }
        const bool once = !(t->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(cgraph, j) == 1;
        if (!r.mul_w) {
            if (t->op != GGML_OP_MUL || t->src[0] != rms || t->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(t->src[1]) ||
                ggml_nelements(t->src[1]) != rms->ne[0] || !once) {
                return false;
            }
            r.mul_w = t;
        } else if (!r.gate && is_gate(t) && once) {
            r.gate = t;
        } else if (t->op == GGML_OP_MUL && ggml_is_contiguous(t) && (t->src[0] == r.mul_w || t->src[1] == r.mul_w)) {
            // the gate may also have run earlier: then it stays as it is and the kernel reads its input
            const ggml_tensor * u = t->src[0] == r.mul_w ? t->src[1] : t->src[0];
            if ((r.gate && u != r.gate) || (!r.gate && !is_gate(u))) {
                return false;
            }
            r.gate     = u;
            r.out      = t;
            r.last_idx = j;
        } else {
            return false;
        }
    }
    if (!r.out) {
        return false;
    }
    // the kernel reads the gate input when it runs: a graph input, or a mat-vec (which no fusion skips) before it.
    // An in-place gate that ran earlier has already overwritten that input with its result.
    const ggml_tensor * g  = r.gate->src[0];
    const ggml_tensor * gb = g->view_src ? g->view_src : g;
    if (r.gate->view_src) {
        return false;
    }
    int g_idx = -1;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        if (cgraph->nodes[j] == gb) {
            if (j > node_idx || gb->op != GGML_OP_MUL_MAT) {
                return false;
            }
            g_idx = j;
        }
    }
    if (ggml_sycl_written_between(cgraph, g, g_idx, node_idx)) {
        return false;
    }
    m = r;
    return true;
}
