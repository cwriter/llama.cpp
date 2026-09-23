#include "fusion.hpp"
#include "binbcast.hpp"

#include <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_mul_mat_glu_mmvq_fused().
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

    // one set of block offsets and one quantized activation must serve both weights
    if (wu->type != wg->type || !ggml_are_same_shape(wu, wg) || !ggml_are_same_stride(wu, wg)) {
        return false;
    }
    if (act != gate->src[1]) {
        return false;
    }

    // q4_K and q8_0 have a fused reorder GEMV, and each walks whole blocks
    if (wu->type != GGML_TYPE_Q4_K && wu->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (wu->type == GGML_TYPE_Q8_0 && glu_op != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    if ((wu->type == GGML_TYPE_Q4_K && wu->ne[0] % QK_K != 0) ||
        (wu->type == GGML_TYPE_Q8_0 && wu->ne[0] % QK8_0 != 0)) {
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
    if (act->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }
    // the q8_0 reorder GLU kernel is instantiated for one and two columns
    if (wu->type == GGML_TYPE_Q8_0 && act->ne[1] > 2) {
        return false;
    }

    return true;
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

        // the ops ggml_sycl_op_unary_mul_fused() has a kernel for
        if (unary_op != GGML_UNARY_OP_SILU && unary_op != GGML_UNARY_OP_SIGMOID &&
            unary_op != GGML_UNARY_OP_SOFTPLUS) {
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

    return false;
}
