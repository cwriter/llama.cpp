#ifndef GGML_SYCL_FUSION_HPP
#define GGML_SYCL_FUSION_HPP

#include <initializer_list>

#include "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops,
                        std::initializer_list<enum ggml_unary_op> unary_ops);

// UNARY at `node_idx` feeding a MUL that broadcasts it over the row: the unary side has one
// value per row (ne0 == 1) and the MUL is wider. ggml_can_fuse() cannot express this - it
// requires every node in the run to have the same shape - so this is its own matcher.
// Purely structural, so ggml_backend_sycl_fusion_absorbs() can ask it at allocation time.
bool ggml_sycl_can_fuse_unary_mul_bcast(const ggml_cgraph * cgraph, int node_idx);

// RMS_NORM at `node_idx` feeding a SCALE that exists only to finish an L2 norm.
bool ggml_sycl_can_fuse_rms_norm_scale(const ggml_cgraph * cgraph, int node_idx);

// GATED_DELTA_NET at `gdn_idx` whose q and k are both L2 norms (RMS_NORM + SCALE) read by nothing else
bool ggml_sycl_match_gdn_qk_norm(const ggml_cgraph * cgraph, int gdn_idx, ggml_sycl_gdn_qk_norm * out);

// a matched gated delta net norms its q/k only for a short batch; the norm nodes keep their dst for the rest
bool ggml_sycl_gdn_qk_norm_now(const ggml_tensor * gdn);

// RMS_NORM at `node_idx` that a matched gated delta net norms itself; returns that net's index, or -1
int ggml_sycl_gdn_qk_norm_reader(const ggml_cgraph * cgraph, int node_idx);

// RMS_NORM at `node_idx` * a row weight * sigmoid or silu of a gate (build_norm_gated), in one kernel
struct ggml_sycl_norm_gate_match {
    const ggml_tensor * rms   = nullptr;
    const ggml_tensor * mul_w = nullptr;
    const ggml_tensor * gate  = nullptr; // the UNARY
    const ggml_tensor * out   = nullptr; // the final MUL
    int                 last_idx = -1;
};
bool ggml_sycl_match_rms_norm_gate(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_norm_gate_match & m);

// CONT of a strided view feeding UNARY + MUL; returns the unary index, or -1.
int ggml_sycl_match_gate_view(const ggml_cgraph * cgraph, int node_idx);

#endif  // GGML_SYCL_FUSION_HPP
