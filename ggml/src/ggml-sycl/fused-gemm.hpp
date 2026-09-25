#ifndef GGML_SYCL_FUSED_GEMM_HPP
#define GGML_SYCL_FUSED_GEMM_HPP

#include "common.hpp"


// Shape and type gates, shared by the kernels below and by the graph compatibility check, so
// the two cannot drift. Device capability is separate: it needs a queue to ask.
static constexpr int      GGML_SYCL_FG_BN      = 32;              // B columns of one output tile
static constexpr int      GGML_SYCL_FG_MAX_N   = 2 * GGML_SYCL_FG_BN;
static constexpr int64_t  GGML_SYCL_SK_MAX_MNK = 256 * 256 * 256;

// weight formats the fused A stage decodes; K must cover whole stored blocks
constexpr bool ggml_sycl_fused_dequant_gemm_f16_type_ok(ggml_type src0_type, int64_t K) {
    // iq4_nl and q8_0 store 32 values per block, which is exactly one A-stage k step; every
    // other format here is a 256-value superblock the A stage walks in steps of 32, so K must
    // cover whole superblocks.
    if (src0_type == GGML_TYPE_IQ4_NL) {
        return K % QK4_NL == 0;
    }
    if (src0_type == GGML_TYPE_Q8_0) {
        return K % QK8_0 == 0;
    }
    const bool superblock =
           src0_type == GGML_TYPE_IQ3_S ||
           src0_type == GGML_TYPE_IQ4_XS ||
           src0_type == GGML_TYPE_IQ3_XXS ||
           src0_type == GGML_TYPE_IQ2_XXS ||
           src0_type == GGML_TYPE_IQ2_XS ||
           src0_type == GGML_TYPE_IQ2_S ||
           src0_type == GGML_TYPE_IQ1_S ||
           src0_type == GGML_TYPE_IQ1_M ||
           src0_type == GGML_TYPE_Q4_K ||
           src0_type == GGML_TYPE_Q5_K ||
           src0_type == GGML_TYPE_Q6_K;
    return superblock && QK_K == 256 && K % QK_K == 0;
}

// Runtime type gate, kept out of the constexpr predicate above so it stays pure.
inline bool ggml_sycl_xmx_gather_type_enabled(ggml_type src0_type) {
    switch (src0_type) {
        case GGML_TYPE_IQ4_NL:  return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_NL  ) != 0;
        case GGML_TYPE_IQ3_S:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_S   ) != 0;
        case GGML_TYPE_IQ4_XS:  return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_XS  ) != 0;
        case GGML_TYPE_IQ3_XXS: return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_XXS ) != 0;
        case GGML_TYPE_IQ2_XXS: return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XXS ) != 0;
        case GGML_TYPE_IQ2_XS:  return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XS  ) != 0;
        case GGML_TYPE_IQ2_S:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_S   ) != 0;
        case GGML_TYPE_IQ1_S:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_S   ) != 0;
        case GGML_TYPE_IQ1_M:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_M   ) != 0;
        case GGML_TYPE_Q8_0:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q8_0    ) != 0;
        // q4_K only under GGML_SYCL_FAST_AND_SLOPPY: see the flag's comment in common.hpp.
        case GGML_TYPE_Q4_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q4_K    ) != 0 &&
                                       g_ggml_sycl_fast_and_sloppy != 0;
        case GGML_TYPE_Q5_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q5_K    ) != 0;
        case GGML_TYPE_Q6_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q6_K    ) != 0;
        default:          return false;
    }
}

// weight formats whose A stage also reads the reorder (SoA) layout. A reordered weight of any
// other type must not reach the fused kernels: the decode would be silently wrong.
//
// This list is load-bearing for coverage, not just for correctness. ggml_sycl_mul_mat_id()
// reorders q4_K, q5_K and q6_K unconditionally (ggml_sycl_mul_mat_id_reorders_type()), so for
// those three the weights are ALWAYS in the SoA layout by the time the grouped GEMM is asked:
// without an entry here the gather bit alone would buy nothing, because every launch would
// decline on `reordered && !reorder_ok`. q8_0 is reordered only when GGML_SYCL_REORDER_Q8_0 is
// set (off by default), so it must be listed too or it would lose coverage the moment it is.
constexpr bool ggml_sycl_fused_dequant_gemm_f16_reorder_ok(ggml_type src0_type) {
    return src0_type == GGML_TYPE_IQ3_S || src0_type == GGML_TYPE_IQ4_NL ||
           src0_type == GGML_TYPE_Q8_0 || src0_type == GGML_TYPE_Q4_K ||
           src0_type == GGML_TYPE_Q5_K || src0_type == GGML_TYPE_Q6_K;
}

constexpr bool ggml_sycl_fused_dequant_gemm_f16_shape_ok(ggml_type src0_type, int64_t M, int64_t N, int64_t K,
                                                         int64_t ldd) {
    return ggml_sycl_fused_dequant_gemm_f16_type_ok(src0_type, K) && M > 0 && N > 0 && K > 0 &&
           N <= GGML_SYCL_FG_MAX_N &&
           M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX && ldd <= INT32_MAX;
}

constexpr bool ggml_sycl_small_gemm_f32_shape_ok(int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldd) {
    return M > 0 && N > 0 && K > 0 && M * N * K <= GGML_SYCL_SK_MAX_MNK &&
           M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX && lda <= INT32_MAX && ldd <= INT32_MAX;
}

// grouped variant, routing-independent half: type, weight geometry and row count
constexpr bool ggml_sycl_grouped_dequant_gemm_f16_shape_ok(ggml_type src0_type, int64_t M, int64_t K,
                                                           int64_t total_rows) {
    return ggml_sycl_fused_dequant_gemm_f16_shape_ok(src0_type, M, 1, K, M) && total_rows > 0 &&
           total_rows <= INT32_MAX;
}

// The per-expert fused kernel is only worth it while each expert slice is narrow, so wider
// average slices are left to the per-expert library GEMM loop. This is the one clause that reads
// the routing, so it is a pure performance heuristic: a caller that cannot count the active
// experts (the schedule lives on the device) passes the bound min(n_as, total_rows) instead.
constexpr bool ggml_sycl_grouped_dequant_gemm_f16_width_ok(int64_t total_rows, int64_t n_active) {
    return total_rows <= n_active * GGML_SYCL_FG_MAX_N;
}

// Upper bound on the schedule size, for a caller that must size the tile table before the
// device has built it. An expert slice of r rows costs ceil(r / FG_BN) tiles, and
// sum ceil(r_e / FG_BN) <= total_rows / FG_BN + n_active; every tile holds at least one row,
// so total_rows caps it as well.
// Packed-B column stride when the tile table is decoupled from the packed-B layout: tiles cover
// disjoint row ranges, so a tile can read its FG_BN columns starting at its own first row instead
// of at t * FG_BN. The buffer then holds one column per routed row rather than one per tile slot,
// plus FG_BN of slack so the last tile's read stays in bounds.
constexpr int64_t ggml_sycl_grouped_gemm_tight_npad(int64_t total_rows) {
    return ((total_rows + GGML_SYCL_FG_BN - 1) / GGML_SYCL_FG_BN + 1) * GGML_SYCL_FG_BN;
}

constexpr int64_t ggml_sycl_grouped_gemm_max_tiles(int64_t total_rows, int64_t n_as) {
    const int64_t n_active_max = n_as < total_rows ? n_as : total_rows;
    const int64_t bound        = total_rows / GGML_SYCL_FG_BN + n_active_max;
    return bound < total_rows ? bound : total_rows;
}

// True if the device can run the kernel at all; cached, so it is cheap to ask per node.
bool ggml_sycl_fused_dequant_gemm_f16_device_ok(dpct::queue_ptr stream);
bool ggml_sycl_small_gemm_f32_device_ok(dpct::queue_ptr stream);

// dst[n*ldd + m] = sum_k dequant(src0)[m*K + k] * src1_f16[n*K + k]
// Returns false when the case is not handled (type, device, or shape).
// `reordered` says src0 is in the reorder (SoA) layout; src0 must then be the base of the whole
// reordered region, because the SoA offsets are relative to it.
bool ggml_sycl_fused_dequant_gemm_f16(ggml_type src0_type, const void * src0, const sycl::half * src1_f16, float * dst,
                                      int64_t M, int64_t N, int64_t K, int64_t ldd, bool reordered,
                                      ggml_sycl_pool & pool, dpct::queue_ptr stream);

// One launch for every expert of a MUL_MAT_ID: rows of src1/dst are grouped by expert, expert e
// owns rows [expert_row_offsets[e], expert_row_offsets[e+1]) and reads its weights at
// src0_base + e*expert_stride. tiles is host scratch that must stay alive until the queue drains.
// dst[n*M + m] = sum_k dequant(src0_e)[m*K + k] * src1[n*K + k]
// Returns false when the case is not handled (type, device, or shape).
// `reordered` says every expert slice is in the reorder (SoA) layout, reordered per slice.
bool ggml_sycl_grouped_dequant_gemm_f16(ggml_type src0_type, const void * src0_base, size_t expert_stride,
                                        const float * src1, float * dst, const int64_t * expert_row_offsets,
                                        int64_t n_as, int64_t M, int64_t K, int64_t total_rows, bool reordered,
                                        std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                                        dpct::queue_ptr stream);

// Largest expert count the device schedule kernel can hold in shared local memory.
static constexpr int64_t GGML_SYCL_MMID_SCHED_MAX_EXPERTS = 4096;

// Build the grouped-GEMM schedule on the device, in one dispatch, so the host never reads the
// routing back: a counting sort of the routed rows by expert id, the row mapping the gather and
// scatter kernels read, and the tile table. `tiles` must hold
// ggml_sycl_grouped_gemm_max_tiles(total_rows, n_as) entries; the tail past the tiles actually
// used is filled with empty tiles that the GEMM skips.
// `row_mapping` is zeroed first: a route whose expert id is out of range is dropped, and a zero
// entry names row (0, 0), which is a wrong value but never an out-of-range write.
// Returns false when the shape is out of range for the kernel.
bool ggml_sycl_build_mmid_schedule(const int32_t * ids_dev, size_t ids_token_stride, int64_t n_as, int64_t n_ids,
                                   int64_t n_tokens, int64_t n_tiles_max, mmid_row_mapping * row_mapping,
                                   ggml_sycl_gg_tile * tiles, uint32_t * expert_offsets, dpct::queue_ptr stream);

// Host-side half of the gate for the device-scheduled arm: everything the caller can decide
// without reading the routing. n_active is not knowable here, so the width heuristic is asked
// with its host bound min(n_as, total_rows).
bool ggml_sycl_grouped_dequant_gemm_f16_dev_ok(ggml_type src0_type, int64_t M, int64_t K, int64_t total_rows,
                                               int64_t n_as, bool reordered, dpct::queue_ptr stream);

// Grouped GEMM fed by a device-built tile table. The schedule is not on the host, so the launch is
// bounded by n_tiles_max and the empty tiles exit early.
// Npad is derived from n_tiles_max ONCE here and is the packed-B column stride for both the pack
// and the GEMM; the two must never be given different values.
bool ggml_sycl_grouped_dequant_gemm_f16_dev(ggml_type src0_type, const void * src0_base, size_t expert_stride,
                                            const float * src1, float * dst, const ggml_sycl_gg_tile * tiles_dev,
                                            int64_t n_tiles_max, int64_t M, int64_t K, int64_t total_rows,
                                            bool reordered, ggml_sycl_pool & pool, dpct::queue_ptr stream);

// dst[n*ldd + m] = sum_k a[m*lda + k] * b[n*K + k], split-K for small M*N with long K
// Returns false when the case is not handled (shape too large or device too small).
bool ggml_sycl_small_gemm_f32(const float * a, const float * b, float * dst, int64_t M, int64_t N, int64_t K,
                              int64_t lda, int64_t ldd, dpct::queue_ptr stream);

#endif // GGML_SYCL_FUSED_GEMM_HPP
