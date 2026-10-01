#include <atomic>

#include "fused-gemm.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

namespace mx = sycl::ext::oneapi::experimental::matrix;

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
static constexpr int FG_BK     = QK4_NL;
static constexpr int FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
template <typename T> struct fg_elem;

template <> struct fg_elem<sycl::half> {
    using store = sycl::half;
    using pair  = sycl::half2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::fp16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    static constexpr ggml_type       src   = GGML_TYPE_F16;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return (store) x; }
    static pair make(float x, float y) { return pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
static inline uint32_t fg_round_bits(float x, int drop) {
    const uint32_t u = sycl::bit_cast<uint32_t>(x);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        return (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    return u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

struct alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

template <> struct fg_elem<sycl::ext::oneapi::bfloat16> {
    using store = sycl::ext::oneapi::bfloat16;
    using pair  = fg_bf16x2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::bf16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    static constexpr ggml_type       src   = GGML_TYPE_BF16;
    static constexpr int             mant  = 7;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return store(x); }
    static pair make(float x, float y) { return { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
template <> struct fg_elem<mx::precision::tf32> {
    using store = float;
    using pair  = sycl::float2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::tf32;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    static constexpr ggml_type       src   = GGML_TYPE_COUNT;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 1;
    static store cvt(float x) { return sycl::bit_cast<float>(fg_round_bits(x, 13) & ~0x1fffu); }
    static pair make(float x, float y) { return pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
template <typename TA, typename TB, int TM_, int TN_, int TK_, int SG_> struct fg_combo {
    using ta  = TA;
    using tb  = TB;
    using EA  = fg_elem<TA>;
    using EB  = fg_elem<TB>;
    using tsa = typename EA::store;
    using tsb = typename EB::store;
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
    static constexpr int TK = TK_;
    static constexpr int SG = SG_;
    static constexpr int VNNI    = 4 / sizeof(tsb);  // K rows of B packed in one 32-bit word
    static constexpr int SG_ROWS = TM > 16 ? TM : 16;
    static constexpr int BN      = TN > 32 ? TN : 32;
    static constexpr int NT      = BN / TN;
    static constexpr int WG_SIZE = FG_KSPLIT * SG;
    static constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::row_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    static constexpr bool efficient = TN <= 32;
    static_assert(SG_ROWS % SG == 0 && SG_ROWS % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

using fg_half = sycl::half;
using fg_bf16 = sycl::ext::oneapi::bfloat16;
using fg_tf32 = mx::precision::tf32;

// The combination the GGML_SYCL_MMID_SCHED tile and A-stage variants were tuned on (B60). Only it
// builds them; every other combination runs the plain tile, which keeps the kernel count bounded.
using fg_tuned_combo = fg_combo<fg_half, fg_half, 8, 16, 16, 16>;
template <typename S> static constexpr bool fg_is_tuned() {
    return std::is_same_v<S, fg_tuned_combo>;
}

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
template <typename F> static void fg_visit_combo(int idx, F && f);
static constexpr int FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
template <int SG> static constexpr bool fg_listed() {
#if defined(GGML_SYCL_XMX_AOT_SG)
    return SG == GGML_SYCL_XMX_AOT_SG;
#else
    return true;
#endif
}

template <typename S, typename F> static void fg_call_combo(F && f) {
    if constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

template <typename F> static void fg_visit_combo(int idx, F && f) {
    switch (idx) {
        case 0: fg_call_combo<fg_tuned_combo>(f);                              break; // Xe2, Xe3, Xe-HPC
        case 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
        case 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   break; // Xe2, Xe3, Xe-HPC
        case 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
        case 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
        default: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
#if defined(__SYCL_DEVICE_ONLY__)
#    if __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
        __SYCL_TARGET_INTEL_GPU_ARL_H__
#        define FG_AOT_SG 8
#    elif __SYCL_TARGET_INTEL_GPU_PVC__ || __SYCL_TARGET_INTEL_GPU_PVC_VG__ || __SYCL_TARGET_INTEL_GPU_BMG_G21__ || \
        __SYCL_TARGET_INTEL_GPU_BMG_G31__ || __SYCL_TARGET_INTEL_GPU_LNL_M__ || __SYCL_TARGET_INTEL_GPU_PTL_H__ ||   \
        __SYCL_TARGET_INTEL_GPU_PTL_U__ || __SYCL_TARGET_INTEL_GPU_WCL__ || __SYCL_TARGET_INTEL_GPU_NVL_S__ ||       \
        __SYCL_TARGET_INTEL_GPU_NVL_U__ || __SYCL_TARGET_INTEL_GPU_NVL_P__
#        define FG_AOT_SG 16
#    elif __SYCL_TARGET_INTEL_GPU_TGLLP__ || __SYCL_TARGET_INTEL_GPU_RKL__ || __SYCL_TARGET_INTEL_GPU_ADL_S__ || \
        __SYCL_TARGET_INTEL_GPU_ADL_P__ || __SYCL_TARGET_INTEL_GPU_ADL_N__ || __SYCL_TARGET_INTEL_GPU_DG1__ ||   \
        __SYCL_TARGET_INTEL_GPU_MTL_U__ || __SYCL_TARGET_INTEL_GPU_MTL_H__
#        define FG_AOT_SG 0 // no XMX
#    endif
#endif

template <int SG> static constexpr bool fg_built() {
#if defined(FG_AOT_SG)
    return SG == FG_AOT_SG;
#else
    return true;
#endif
}

static size_t grouped_gemm_packed_capacity(size_t size) {
    size_t capacity = 1;
    while (capacity < size) {
        capacity *= 2;
    }
    return capacity;
}

// the device lists S with an f32 accumulator and output
template <typename S> static bool fg_device_has_combo(const std::vector<mx::combination> & combinations) {
    for (const auto & c : combinations) {
        if (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::matrix_type::fp32 &&
            c.dtype == mx::matrix_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            return true;
        }
    }
    return false;
}

template <typename T> static const char * fg_type_name() {
    return std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

static std::string fg_combo_name(int idx) {
    std::string name;
    fg_visit_combo(idx, [&](auto s) {
        using S = decltype(s);
        name = std::string(fg_type_name<typename S::ta>()) + "x" + fg_type_name<typename S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    return name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
static int fg_device_combos(const sycl::device & dev) {
    static std::mutex                            mtx;
    static std::unordered_map<sycl::device, int> known;
    std::lock_guard<std::mutex>                  lock(mtx);
    const auto                                   it = known.find(dev);
    if (it != known.end()) {
        return it->second;
    }
    int available = 0;
    try {
        const auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        const auto sg_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                const bool sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                if (sg_ok && fg_device_has_combo<S>(combinations)) {
                    available |= 1 << idx;
                }
            });
        }
    } catch (const sycl::exception &) {
        available = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), available, g_ggml_sycl_xmx_gather_shapes);
    known.emplace(dev, available);
    return available;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
template <typename S> static int64_t fg_rank(ggml_type src1_type) {
    const int64_t spills  = !S::efficient;
    const int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    const int64_t convert = S::EB::src != src1_type;
    const int64_t lossy   = S::EB::mant < 10;
    const int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    const int64_t mk      = 1024 - S::TM * S::TK;
    return ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
static bool fg_mode_meets(int mode, int32_t src1_prec) {
    if (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        return true;
    }
    if (src1_prec >= GGML_PREC_BF16) {
        return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    switch (g_ggml_sycl_dynamic_required_precision) {
        case GGML_SYCL_DYNAMIC_PRECISION_TF32: return mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        case GGML_SYCL_DYNAMIC_PRECISION_BF16: return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        default:                               return false;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first). `bn`, if not 0, keeps only combinations of that tile width.
static int fg_pick_combo(dpct::queue_ptr stream, ggml_type src1_type, int32_t src1_prec, int bn = 0) {
    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32) {
        return -1;
    }
    const sycl::device dev     = stream->get_device();
    const int          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    const bool         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    const int          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    int     best      = -1;
    int64_t best_rank = 0;
    for (int i = 0; i < 3 && best < 0; ++i) {
        const int mode = modes[i];
        if (!fg_mode_meets(mode, src1_prec)) {
            continue;
        }
        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
            if (!(allowed & (1 << idx))) {
                continue;
            }
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                if (S::EA::mode != mode || S::EB::mode != mode || (bn != 0 && S::BN != bn)) {
                    return;
                }
                const int64_t rank = fg_rank<S>(src1_type);
                if (best < 0 || rank < best_rank) {
                    best      = idx;
                    best_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    static std::mutex                                           mtx;
    static std::set<std::tuple<size_t, int, int32_t, int, int>> seen;
    std::lock_guard<std::mutex>                                 lock(mtx);
    if (seen.emplace(std::hash<sycl::device>{}(dev), (int) src1_type, src1_prec, bn, best).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d%s -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      bn != 0 ? ", device schedule" : "", best >= 0 ? fg_combo_name(best).c_str() : "none (library GEMM)");
    }
    return best;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
template <typename E, typename T_src>
static void fused_gemm_pack_b(const T_src * y, typename E::store * packed, int N, int Npad, int K,
                              dpct::queue_ptr stream) {
    constexpr int V   = 4 / sizeof(typename E::store);
    const int     kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        const int idx = id[0];
        const int n   = idx / kqs;
        const int kq  = idx - n * kqs;
        typename E::store vals[V] = {};
        if (n < N) {
            const T_src * src = y + (size_t) n * K + V * kq;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                vals[v] = E::cvt((float) src[v]);
            }
        }
        typename E::store * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq4_nl * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq4_nl blk = xrow[kb];
    const float        d   = (float) blk.d;
#pragma unroll
    for (int j = 0; j < QK4_NL / 2; j += 2) {
        const uint8_t q0 = blk.qs[j];
        const uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// q8_0: 32 values per block and FG_BK == 32, so one k step is exactly one block. No superblock
// walk, no nibble unpack and no value lookup: the quants are already signed bytes.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q8_0 * __restrict__ xrow, const int kb, typename E::pair * a) {
    static_assert(QK8_0 == FG_BK, "the q8_0 A stage assumes one block per k step");
    const block_q8_0 * blk = xrow + kb;
    const float        d   = (float) blk->d;
    const int8_t *     qs  = blk->qs;
#pragma unroll
    for (int j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// One 32-value sub-block of an iq3_s superblock: its 8 qs bytes, its qh byte, its 4 sign bytes
// and a d that already carries the sub-block scale. Same decode as dequantize_block_iq3_s: grid
// entries are taken as dwords and the sign bit is a plain shift. Shared by the canonical, the SoA
// and the register A stages, which differ only in where those four fields live.
// `grid` is iq3s_grid, or a work-group's SLM copy of it under GGML_SYCL_MMID_SCHED_GRID_SLM.
template <typename E>
static __dpct_inline__ void fg_decode_iq3_s(const uint8_t * __restrict__ qs, const int qh,
                                            const uint8_t * __restrict__ signs, const float d, typename E::pair * a,
                                            const uint32_t * __restrict__ grid) {
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint32_t grid1 = grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        const uint32_t grid2 = grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        const int      sg    = signs[il];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float g1a = (float) ((grid1 >> (16 * j + 0)) & 0xff);
            const float g1b = (float) ((grid1 >> (16 * j + 8)) & 0xff);
            const float g2a = (float) ((grid2 >> (16 * j + 0)) & 0xff);
            const float g2b = (float) ((grid2 >> (16 * j + 8)) & 0xff);
            const int   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((sg & (1 << (s + 0))) ? -g1a : g1a), d * ((sg & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((sg & (1 << (s + 4))) ? -g2a : g2a), d * ((sg & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq3_s * __restrict__ xrow, const int kb, typename E::pair * a,
                                       const uint32_t * __restrict__ grid = iq3s_grid) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    const block_iq3_s * blk = xrow + kb / (QK_K / 32);
    const int           ib8 = kb % (QK_K / 32);
    const float         d   = (float) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
    fg_decode_iq3_s<E>(blk->qs + 8 * ib8, blk->qh[ib8], blk->signs + 4 * ib8, d, a, grid);
}

// Reorder (SoA) A stage. The reorder is a pure permutation of a slice: the same block fields in
// the same intra-field order, but each field is one stream over the nblocks of the slice. Only
// the addresses change, so the decode below is identical to the canonical overload.
// Every superblock format below steps its 256 values in 32-wide k steps.
static_assert(QK_K == 256, "the superblock A stages assume 8 sub-blocks per superblock");

template <typename E>
static __dpct_inline__ void fg_pack_quarter(const float * __restrict__ t, typename E::pair * a, int il) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq4_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq4_xs * blk = xrow + kb / (QK_K / 32);
    const int ib = kb % (QK_K / 32);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    const float d = (float) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    const uint8_t * q4 = blk->qs + 16 * ib;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4], d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq3_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq3_xxs * blk = xrow + kb / (QK_K / 32);
    const int ib = kb % (QK_K / 32);
    const uint8_t *  q3    = blk->qs + 8 * ib;
    const uint16_t * gas   = (const uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    const uint32_t   aux32 = gas[0] | (gas[1] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq2_xxs * blk = xrow + kb / (QK_K / 32);
    const int ib = kb % (QK_K / 32);
    const uint16_t * q2    = blk->qs + 4 * ib;
    const uint8_t *  aux8  = (const uint8_t *) q2;
    const uint32_t   aux32 = q2[2] | (q2[3] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xxs_grid + aux8[il]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq2_xs * blk = xrow + kb / (QK_K / 32);
    const int ib = kb % (QK_K / 32);
    const uint16_t * q2 = blk->qs + 4 * ib;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xs_grid + (q2[il] & 511));
        const float     d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq2_s * blk = xrow + kb / (QK_K / 32);
    const int ib = kb % (QK_K / 32);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid =
            (const uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        const float   d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq1_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq1_s * blk = xrow + kb / (QK_K / 32);
    const int ib = kb % (QK_K / 32);
    const float delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d     = (float) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq1_m * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq1_m * blk = xrow + kb / (QK_K / 32);
    const int ib = kb % (QK_K / 32);
    const uint16_t * sc = (const uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const int   ib16  = 2 * ib + il / 2;
        const float d     = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        const float delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

// ---------------------------------------------------------------------------------------------
// k-quant A stages (q4_K, q5_K, q6_K). QK_K == 256, so a superblock is 8 k steps of 32 values.
// The decode of one sub-block is factored out of the canonical and the SoA overload, exactly as
// dequantize.hpp shares dequantize_q4_K_common(): the two layouts differ only in where the
// fields live, never in what they mean.
// ---------------------------------------------------------------------------------------------

// the 12 packed bytes hold eight 6-bit scales and eight 6-bit mins; same unpack as
// get_scale_min_k4() in ggml-quants.c / dequantize.hpp, renamed so this TU owns its copy
static __dpct_inline__ void fg_scale_min_k4(const int j, const uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// q4_K sub-block ib32 (0..7) of one superblock. Following dequantize_row_q4_K: the outer loop
// there walks 64 values at a time from q = qs + 32*t with scale indices 2t and 2t+1, so
// sub-block ib32 uses scale/min pair ib32 and the low (ib32 even) or high (ib32 odd) nibbles of
// qs[32*(ib32/2) ...]. Value l of the sub-block comes from byte l, so a[j] packs bytes 2j, 2j+1.
template <typename E>
static __dpct_inline__ void fg_decode_q4_k(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                                           const float dall, const float dmin, const int ib32, typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib32, scales, sc, mb);
    const float     d     = dall * sc;
    const float     m     = dmin * mb;
    const uint8_t * q     = qs + 32 * (ib32 / 2);
    const int       shift = 4 * (ib32 & 1);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float v0 = d * (float) ((q[2 * j + 0] >> shift) & 0xF) - m;
        const float v1 = d * (float) ((q[2 * j + 1] >> shift) & 0xF) - m;
        a[j] = E::make(v0, v1);
    }
}

// q5_K adds one high bit per value. dequantize_row_q5_K keeps qh fixed over the superblock and
// shifts its mask by 2 per 64 values (u1 = 1 << 2t, u2 = 2 << 2t), which is bit ib32 of qh[l]
// for sub-block ib32. ql advances the same way as q4_K.
template <typename E>
static __dpct_inline__ void fg_decode_q5_k(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ qh,
                                           const uint8_t * __restrict__ scales, const float dall, const float dmin,
                                           const int ib32, typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib32, scales, sc, mb);
    const float     d     = dall * sc;
    const float     m     = dmin * mb;
    const uint8_t * ql    = qs + 32 * (ib32 / 2);
    const int       shift = 4 * (ib32 & 1);
    const int       hm    = 1 << ib32;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int l0 = 2 * j;
        const int l1 = 2 * j + 1;
        const float v0 = d * (float) (((ql[l0] >> shift) & 0xF) + ((qh[l0] & hm) ? 16 : 0)) - m;
        const float v1 = d * (float) (((ql[l1] >> shift) & 0xF) + ((qh[l1] & hm) ? 16 : 0)) - m;
        a[j] = E::make(v0, v1);
    }
}

// q6_K has 8-bit signed scales and no min. dequantize_row_q6_K walks 128 values at a time:
// half h = ib32 / 4 selects ql + 64h, qh + 32h and scales + 8h, and quarter r = ib32 % 4 selects
// ql offset 32*(r&1), ql nibble 4*(r>>1), qh shift 2r and scale offset 2r. Within a sub-block the
// scale still changes at value 16 (is = l/16), so it is indexed per half of the 32 values.
template <typename E>
static __dpct_inline__ void fg_decode_q6_k(const uint8_t * __restrict__ ql_sb, const uint8_t * __restrict__ qh_sb,
                                           const int8_t * __restrict__ sc, const float d, const int r,
                                           typename E::pair * a) {
    const uint8_t * ql      = ql_sb + 32 * (r & 1);
    const int       shift_l = 4 * (r >> 1);
    const int       shift_h = 2 * r;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int   l0 = 2 * j;
        const int   l1 = 2 * j + 1;
        // l0 and l1 always fall in the same 16-value scale group, so one lookup serves both
        const float dl = d * (float) sc[j >> 3];
        const int   q0 = (int) (((ql[l0] >> shift_l) & 0xF) | (((qh_sb[l0] >> shift_h) & 3) << 4)) - 32;
        const int   q1 = (int) (((ql[l1] >> shift_l) & 0xF) | (((qh_sb[l1] >> shift_h) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q4_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_q4_K * blk  = xrow + kb / (QK_K / 32);
    const int          ib32 = kb % (QK_K / 32);
    const sycl::half2  dm   = blk->dm;
    fg_decode_q4_k<E>(blk->qs, blk->scales, (float) dm[0], (float) dm[1], ib32, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q5_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_q5_K * blk  = xrow + kb / (QK_K / 32);
    const int          ib32 = kb % (QK_K / 32);
    const sycl::half2  dm   = blk->dm;
    fg_decode_q5_k<E>(blk->qs, blk->qh, blk->scales, (float) dm[0], (float) dm[1], ib32, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q6_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_q6_K * blk  = xrow + kb / (QK_K / 32);
    const int          ib32 = kb % (QK_K / 32);
    const int          h    = ib32 / 4;
    const int          r    = ib32 % 4;
    fg_decode_q6_k<E>(blk->ql + 64 * h, blk->qh + 32 * h, blk->scales + 8 * h + 2 * r, (float) blk->d, r, a);
}

// Copy N bytes of one stored block into a private array. The weight base of a MoE slice is not
// dword aligned for every format, so the copy stays byte wide and the compiler widens what it can,
// exactly as the A stages already read these fields byte by byte.
template <int N> static __dpct_inline__ void fg_load_bytes(uint8_t * dst, const uint8_t * __restrict__ src) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        dst[i] = src[i];
    }
}

// stage() reads one sub-block straight from memory, which is what a k step needs. load() and
// decode() split the same work in two, for MMID_SCHED_REGS_A: load() takes the whole stored block
// into registers once and decode() takes one sub-block out of those registers, so the KPB k steps
// of a block cost one set of global reads instead of KPB. The decode is shared, so the two entry
// points cannot drift apart.
//
// load_lite() and decode_lite() are the same split with only part of the block held, for
// MMID_SCHED_REGS_LITE_A. Every stream of a block is re-read once per k step whatever its width,
// so a stream of a few bytes removes as many re-reads as the widest one does and costs almost no
// registers. The rule for what load_lite() takes is therefore the stream's width: at most 4 dwords
// a block is cached, anything wider keeps streaming and decode_lite() re-reads it exactly as
// stage() does. In practice that is the metadata of every format, plus the one-byte-per-k-step qh
// of iq3_s. `lite` is false where one stored block is one k step: there is no re-read to remove.
template <typename block_q_t> struct fg_reorder_a {
    static constexpr bool supported = false;
    static constexpr bool lite      = false;
};

template <> struct fg_reorder_a<block_iq4_nl> {
    static constexpr bool supported = true;
    static constexpr bool lite      = false;  // one block is one k step

    static_assert(QK4_NL / 2 + sizeof(ggml_half) == sizeof(block_iq4_nl),
                  "the iq4_nl reorder layout must be a byte permutation of the canonical block");

    // One k step is one whole block here (FG_BK == QK4_NL). The streams are [qs][d], each
    // contiguous over the nblocks of the slice. Decode is the canonical overload verbatim.
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * __restrict__ xb, const int ib_row, const int nblocks,
                                      const int kb, typename E::pair * a) {
        const int       ib = ib_row + kb;
        const uint8_t * qs = xb + (size_t) ib * (QK4_NL / 2);
        const float     d  = (float) *(const ggml_half *) (xb + (size_t) nblocks * (QK4_NL / 2) +
                                                           (size_t) ib * sizeof(ggml_half));
#pragma unroll
        for (int j = 0; j < QK4_NL / 2; j += 2) {
            const uint8_t q0 = qs[j];
            const uint8_t q1 = qs[j + 1];
            a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
            a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
        }
    }

    // one k step is one whole block, so there is nothing to hold across k steps: load() and
    // decode() are the two halves of stage() and the k loop runs them back to back
    struct regs {
        uint8_t qs[QK4_NL / 2];
        float   d;
    };

    static __dpct_inline__ regs load(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs r;
        fg_load_bytes<QK4_NL / 2>(r.qs, xb + (size_t) ib * (QK4_NL / 2));
        r.d = (float) *(const ggml_half *) (xb + (size_t) nblocks * (QK4_NL / 2) +
                                            (size_t) ib * sizeof(ggml_half));
        return r;
    }

    template <typename E>
    static __dpct_inline__ void decode(const regs & r, const int, typename E::pair * a) {
#pragma unroll
        for (int j = 0; j < QK4_NL / 2; j += 2) {
            const uint8_t q0 = r.qs[j];
            const uint8_t q1 = r.qs[j + 1];
            a[j / 2]     = E::make(r.d * kvalues_iq4nl[q0 & 0xf], r.d * kvalues_iq4nl[q1 & 0xf]);
            a[j / 2 + 8] = E::make(r.d * kvalues_iq4nl[q0 >> 4], r.d * kvalues_iq4nl[q1 >> 4]);
        }
    }
};

template <> struct fg_reorder_a<block_iq3_s> {
    static constexpr bool supported = true;
    static constexpr bool lite      = true;

    static_assert(QK_K / 4 + QK_K / 32 + QK_K / 8 + sizeof(ggml_half) + IQ3S_N_SCALE == sizeof(block_iq3_s),
                  "the iq3_s reorder layout must be a byte permutation of the canonical block");

    // ib is the block index inside the slice, nblocks the slice block count. The streams are
    // [qs][qh][signs][{d, scales}], each contiguous over the nblocks.
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * __restrict__ xb, const int ib_row, const int nblocks,
                                      const int kb, typename E::pair * a, const uint32_t * __restrict__ grid = iq3s_grid) {
        static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
        const int ib  = ib_row + kb / (QK_K / 32);
        const int ib8 = kb % (QK_K / 32);

        const uint8_t * qs       = xb + (size_t) ib * (QK_K / 4) + 8 * ib8;
        const uint8_t * qh       = xb + (size_t) nblocks * (QK_K / 4) + (size_t) ib * (QK_K / 32);
        const uint8_t * signs    = xb + (size_t) nblocks * (QK_K / 4 + QK_K / 32) + (size_t) ib * (QK_K / 8) + 4 * ib8;
        const uint8_t * metadata = xb + (size_t) nblocks * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                                   (size_t) ib * (sizeof(ggml_half) + IQ3S_N_SCALE);

        const uint8_t * scales = metadata + sizeof(ggml_half);
        const float     d      = (float) *(const ggml_half *) metadata *
                                 (1 + 2 * ((scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        fg_decode_iq3_s<E>(qs, qh[ib8], signs, d, a, grid);
    }

    // decode() and decode_lite() keep reading iq3s_grid where it lives: the SLM copy
    // (GGML_SYCL_MMID_SCHED_GRID_SLM) is not offered next to the register A stages.
    struct regs {
        uint8_t qs[QK_K / 4];
        uint8_t qh[QK_K / 32];
        uint8_t signs[QK_K / 8];
        uint8_t scales[IQ3S_N_SCALE];
        float   d;
    };

    static __dpct_inline__ regs load(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs r;
        fg_load_bytes<QK_K / 4>(r.qs, xb + (size_t) ib * (QK_K / 4));
        fg_load_bytes<QK_K / 32>(r.qh, xb + (size_t) nblocks * (QK_K / 4) + (size_t) ib * (QK_K / 32));
        fg_load_bytes<QK_K / 8>(r.signs,
                                xb + (size_t) nblocks * (QK_K / 4 + QK_K / 32) + (size_t) ib * (QK_K / 8));
        const uint8_t * metadata = xb + (size_t) nblocks * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                                   (size_t) ib * (sizeof(ggml_half) + IQ3S_N_SCALE);
        r.d = (float) *(const ggml_half *) metadata;
        fg_load_bytes<IQ3S_N_SCALE>(r.scales, metadata + sizeof(ggml_half));
        return r;
    }

    template <typename E>
    static __dpct_inline__ void decode(const regs & r, const int ib8, typename E::pair * a) {
        const float d = r.d * (1 + 2 * ((r.scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        fg_decode_iq3_s<E>(r.qs + 8 * ib8, r.qh[ib8], r.signs + 4 * ib8, d, a, iq3s_grid);
    }

    // 14 of the 110 bytes: the metadata, and qh, which is one byte per k step but a whole stream
    // read per k step today. qs (64 B) and signs (32 B) keep streaming.
    struct regs_lite {
        uint8_t qh[QK_K / 32];
        uint8_t scales[IQ3S_N_SCALE];
        float   d;
    };

    static __dpct_inline__ regs_lite load_lite(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs_lite r;
        fg_load_bytes<QK_K / 32>(r.qh, xb + (size_t) nblocks * (QK_K / 4) + (size_t) ib * (QK_K / 32));
        const uint8_t * metadata = xb + (size_t) nblocks * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                                   (size_t) ib * (sizeof(ggml_half) + IQ3S_N_SCALE);
        r.d = (float) *(const ggml_half *) metadata;
        fg_load_bytes<IQ3S_N_SCALE>(r.scales, metadata + sizeof(ggml_half));
        return r;
    }

    // the qs and signs addresses are the ones stage() computes, so the streamed half cannot drift
    template <typename E>
    static __dpct_inline__ void decode_lite(const regs_lite & r, const uint8_t * __restrict__ xb, const int ib,
                                            const int nblocks, const int ib8, typename E::pair * a) {
        const uint8_t * qs    = xb + (size_t) ib * (QK_K / 4) + 8 * ib8;
        const uint8_t * signs = xb + (size_t) nblocks * (QK_K / 4 + QK_K / 32) + (size_t) ib * (QK_K / 8) + 4 * ib8;
        const float     d     = r.d * (1 + 2 * ((r.scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        fg_decode_iq3_s<E>(qs, r.qh[ib8], signs, d, a, iq3s_grid);
    }
};

template <> struct fg_reorder_a<block_q8_0> {
    static constexpr bool supported = true;
    static constexpr bool lite      = false;  // one block is one k step

    static_assert(QK8_0 + sizeof(ggml_half) == sizeof(block_q8_0),
                  "the q8_0 reorder layout must be a byte permutation of the canonical block");

    // Same producer as iq4_nl: reorder_qw_soa2_moe<block_q8_0, QK8_0>() (and reorder_qw_q8_0()
    // for the plain MUL_MAT case) writes [qs][d], each contiguous over the nblocks of the slice.
    // Only the qs stride differs: 32 signed bytes per block instead of 16 packed nibbles.
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * __restrict__ xb, const int ib_row, const int nblocks,
                                      const int kb, typename E::pair * a) {
        const int      ib = ib_row + kb;
        const int8_t * qs = (const int8_t *) (xb + (size_t) ib * QK8_0);
        const float    d  = (float) *(const ggml_half *) (xb + (size_t) nblocks * QK8_0 +
                                                          (size_t) ib * sizeof(ggml_half));
#pragma unroll
        for (int j = 0; j < QK8_0 / 2; ++j) {
            a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
        }
    }

    // one k step is one whole block here too; see the iq4_nl note above
    struct regs {
        int8_t qs[QK8_0];
        float  d;
    };

    static __dpct_inline__ regs load(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs r;
        fg_load_bytes<QK8_0>((uint8_t *) r.qs, xb + (size_t) ib * QK8_0);
        r.d = (float) *(const ggml_half *) (xb + (size_t) nblocks * QK8_0 + (size_t) ib * sizeof(ggml_half));
        return r;
    }

    template <typename E>
    static __dpct_inline__ void decode(const regs & r, const int, typename E::pair * a) {
#pragma unroll
        for (int j = 0; j < QK8_0 / 2; ++j) {
            a[j] = E::make(r.d * r.qs[2 * j], r.d * r.qs[2 * j + 1]);
        }
    }
};

template <> struct fg_reorder_a<block_q4_K> {
    static constexpr bool supported = true;
    static constexpr bool lite      = true;

    static_assert(QK_K / 2 + K_SCALE_SIZE + sizeof(ggml_half2) == sizeof(block_q4_K),
                  "the q4_K reorder layout must be a byte permutation of the canonical block");

    // Streams are [qs][scales][dm], each contiguous over the nblocks of the slice; see
    // reorder_qw_q4_k_moe() / reorder_qw_q4_k() and block_q_t<GGML_TYPE_Q4_K>::get_d_offset().
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * __restrict__ xb, const int ib_row, const int nblocks,
                                      const int kb, typename E::pair * a) {
        const int ib   = ib_row + kb / (QK_K / 32);
        const int ib32 = kb % (QK_K / 32);

        const uint8_t * qs     = xb + (size_t) ib * (QK_K / 2);
        const uint8_t * scales = xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * K_SCALE_SIZE;
        const sycl::half2 dm   = *(const sycl::half2 *) (xb + (size_t) nblocks * (QK_K / 2 + K_SCALE_SIZE) +
                                                         (size_t) ib * sizeof(ggml_half2));
        fg_decode_q4_k<E>(qs, scales, (float) dm[0], (float) dm[1], ib32, a);
    }

    struct regs {
        uint8_t qs[QK_K / 2];
        uint8_t scales[K_SCALE_SIZE];
        float   dall;
        float   dmin;
    };

    static __dpct_inline__ regs load(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs r;
        fg_load_bytes<QK_K / 2>(r.qs, xb + (size_t) ib * (QK_K / 2));
        fg_load_bytes<K_SCALE_SIZE>(r.scales, xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * K_SCALE_SIZE);
        const sycl::half2 dm = *(const sycl::half2 *) (xb + (size_t) nblocks * (QK_K / 2 + K_SCALE_SIZE) +
                                                       (size_t) ib * sizeof(ggml_half2));
        r.dall = (float) dm[0];
        r.dmin = (float) dm[1];
        return r;
    }

    template <typename E>
    static __dpct_inline__ void decode(const regs & r, const int ib32, typename E::pair * a) {
        fg_decode_q4_k<E>(r.qs, r.scales, r.dall, r.dmin, ib32, a);
    }

    // 16 of the 144 bytes: scales and dm. qs (128 B) keeps streaming.
    struct regs_lite {
        uint8_t scales[K_SCALE_SIZE];
        float   dall;
        float   dmin;
    };

    static __dpct_inline__ regs_lite load_lite(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs_lite r;
        fg_load_bytes<K_SCALE_SIZE>(r.scales, xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * K_SCALE_SIZE);
        const sycl::half2 dm = *(const sycl::half2 *) (xb + (size_t) nblocks * (QK_K / 2 + K_SCALE_SIZE) +
                                                       (size_t) ib * sizeof(ggml_half2));
        r.dall = (float) dm[0];
        r.dmin = (float) dm[1];
        return r;
    }

    template <typename E>
    static __dpct_inline__ void decode_lite(const regs_lite & r, const uint8_t * __restrict__ xb, const int ib,
                                            const int, const int ib32, typename E::pair * a) {
        fg_decode_q4_k<E>(xb + (size_t) ib * (QK_K / 2), r.scales, r.dall, r.dmin, ib32, a);
    }
};

template <> struct fg_reorder_a<block_q5_K> {
    static constexpr bool supported = true;
    static constexpr bool lite      = true;

    static_assert(QK_K / 2 + QK_K / 8 + K_SCALE_SIZE + sizeof(ggml_half2) == sizeof(block_q5_K),
                  "the q5_K reorder layout must be a byte permutation of the canonical block");

    // Streams are [qs][qh][scales][dm]; see reorder_qw_q5_k_moe() / reorder_qw_q5_k() and
    // block_q_t<GGML_TYPE_Q5_K>::get_block_offset()/get_d_offset().
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * __restrict__ xb, const int ib_row, const int nblocks,
                                      const int kb, typename E::pair * a) {
        const int ib   = ib_row + kb / (QK_K / 32);
        const int ib32 = kb % (QK_K / 32);

        const uint8_t * qs     = xb + (size_t) ib * (QK_K / 2);
        const uint8_t * qh     = xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * (QK_K / 8);
        const uint8_t * scales = xb + (size_t) nblocks * (QK_K / 2 + QK_K / 8) + (size_t) ib * K_SCALE_SIZE;
        const sycl::half2 dm   = *(const sycl::half2 *) (xb +
                                    (size_t) nblocks * (QK_K / 2 + QK_K / 8 + K_SCALE_SIZE) +
                                    (size_t) ib * sizeof(ggml_half2));
        fg_decode_q5_k<E>(qs, qh, scales, (float) dm[0], (float) dm[1], ib32, a);
    }

    struct regs {
        uint8_t qs[QK_K / 2];
        uint8_t qh[QK_K / 8];
        uint8_t scales[K_SCALE_SIZE];
        float   dall;
        float   dmin;
    };

    static __dpct_inline__ regs load(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs r;
        fg_load_bytes<QK_K / 2>(r.qs, xb + (size_t) ib * (QK_K / 2));
        fg_load_bytes<QK_K / 8>(r.qh, xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * (QK_K / 8));
        fg_load_bytes<K_SCALE_SIZE>(r.scales,
                                    xb + (size_t) nblocks * (QK_K / 2 + QK_K / 8) + (size_t) ib * K_SCALE_SIZE);
        const sycl::half2 dm = *(const sycl::half2 *) (xb +
                                    (size_t) nblocks * (QK_K / 2 + QK_K / 8 + K_SCALE_SIZE) +
                                    (size_t) ib * sizeof(ggml_half2));
        r.dall = (float) dm[0];
        r.dmin = (float) dm[1];
        return r;
    }

    template <typename E>
    static __dpct_inline__ void decode(const regs & r, const int ib32, typename E::pair * a) {
        fg_decode_q5_k<E>(r.qs, r.qh, r.scales, r.dall, r.dmin, ib32, a);
    }

    // 16 of the 176 bytes: scales and dm. qs (128 B) and qh (32 B) keep streaming -- qh is read
    // whole at every k step here, so it is a wide stream, not a cheap one like the iq3_s qh.
    struct regs_lite {
        uint8_t scales[K_SCALE_SIZE];
        float   dall;
        float   dmin;
    };

    static __dpct_inline__ regs_lite load_lite(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs_lite r;
        fg_load_bytes<K_SCALE_SIZE>(r.scales,
                                    xb + (size_t) nblocks * (QK_K / 2 + QK_K / 8) + (size_t) ib * K_SCALE_SIZE);
        const sycl::half2 dm = *(const sycl::half2 *) (xb +
                                    (size_t) nblocks * (QK_K / 2 + QK_K / 8 + K_SCALE_SIZE) +
                                    (size_t) ib * sizeof(ggml_half2));
        r.dall = (float) dm[0];
        r.dmin = (float) dm[1];
        return r;
    }

    template <typename E>
    static __dpct_inline__ void decode_lite(const regs_lite & r, const uint8_t * __restrict__ xb, const int ib,
                                            const int nblocks, const int ib32, typename E::pair * a) {
        const uint8_t * qs = xb + (size_t) ib * (QK_K / 2);
        const uint8_t * qh = xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * (QK_K / 8);
        fg_decode_q5_k<E>(qs, qh, r.scales, r.dall, r.dmin, ib32, a);
    }
};

template <> struct fg_reorder_a<block_q6_K> {
    static constexpr bool supported = true;
    static constexpr bool lite      = true;

    static_assert(QK_K / 2 + QK_K / 4 + QK_K / 16 + sizeof(ggml_half) == sizeof(block_q6_K),
                  "the q6_K reorder layout must be a byte permutation of the canonical block");

    // Streams are [ql][qh][scales][d]; see reorder_qw_q6_k_moe() / reorder_qw_q6_k() and
    // block_q_t<GGML_TYPE_Q6_K>::get_block_offset()/get_d_offset(). The scales are stored
    // through a uint8_t pointer but are signed, so they are read back as int8_t.
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * __restrict__ xb, const int ib_row, const int nblocks,
                                      const int kb, typename E::pair * a) {
        const int ib   = ib_row + kb / (QK_K / 32);
        const int ib32 = kb % (QK_K / 32);
        const int h    = ib32 / 4;
        const int r    = ib32 % 4;

        const uint8_t * ql     = xb + (size_t) ib * (QK_K / 2);
        const uint8_t * qh     = xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * (QK_K / 4);
        const int8_t *  scales = (const int8_t *) (xb + (size_t) nblocks * (QK_K / 2 + QK_K / 4) +
                                                   (size_t) ib * (QK_K / 16));
        const float d = (float) *(const ggml_half *) (xb +
                            (size_t) nblocks * (QK_K / 2 + QK_K / 4 + QK_K / 16) +
                            (size_t) ib * sizeof(ggml_half));
        fg_decode_q6_k<E>(ql + 64 * h, qh + 32 * h, scales + 8 * h + 2 * r, d, r, a);
    }

    struct regs {
        uint8_t ql[QK_K / 2];
        uint8_t qh[QK_K / 4];
        int8_t  scales[QK_K / 16];
        float   d;
    };

    static __dpct_inline__ regs load(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs r;
        fg_load_bytes<QK_K / 2>(r.ql, xb + (size_t) ib * (QK_K / 2));
        fg_load_bytes<QK_K / 4>(r.qh, xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * (QK_K / 4));
        fg_load_bytes<QK_K / 16>((uint8_t *) r.scales,
                                 xb + (size_t) nblocks * (QK_K / 2 + QK_K / 4) + (size_t) ib * (QK_K / 16));
        r.d = (float) *(const ggml_half *) (xb + (size_t) nblocks * (QK_K / 2 + QK_K / 4 + QK_K / 16) +
                                            (size_t) ib * sizeof(ggml_half));
        return r;
    }

    template <typename E>
    static __dpct_inline__ void decode(const regs & r, const int ib32, typename E::pair * a) {
        const int h = ib32 / 4;
        const int q = ib32 % 4;
        fg_decode_q6_k<E>(r.ql + 64 * h, r.qh + 32 * h, r.scales + 8 * h + 2 * q, r.d, q, a);
    }

    // 18 of the 212 bytes: scales and d. ql (128 B) and qh (64 B) keep streaming.
    struct regs_lite {
        int8_t scales[QK_K / 16];
        float  d;
    };

    static __dpct_inline__ regs_lite load_lite(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        regs_lite r;
        fg_load_bytes<QK_K / 16>((uint8_t *) r.scales,
                                 xb + (size_t) nblocks * (QK_K / 2 + QK_K / 4) + (size_t) ib * (QK_K / 16));
        r.d = (float) *(const ggml_half *) (xb + (size_t) nblocks * (QK_K / 2 + QK_K / 4 + QK_K / 16) +
                                            (size_t) ib * sizeof(ggml_half));
        return r;
    }

    template <typename E>
    static __dpct_inline__ void decode_lite(const regs_lite & r, const uint8_t * __restrict__ xb, const int ib,
                                            const int nblocks, const int ib32, typename E::pair * a) {
        const int       h  = ib32 / 4;
        const int       q  = ib32 % 4;
        const uint8_t * ql = xb + (size_t) ib * (QK_K / 2);
        const uint8_t * qh = xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * (QK_K / 4);
        fg_decode_q6_k<E>(ql + 64 * h, qh + 32 * h, r.scales + 8 * h + 2 * q, r.d, q, a);
    }
};

// values per stored block, so a row of K values is K/qk blocks
template <typename block_q_t> struct fg_block_traits;
template <> struct fg_block_traits<block_iq4_nl> { static constexpr int qk = QK4_NL; };
template <> struct fg_block_traits<block_iq3_s>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq4_xs>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq3_xxs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xxs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xs>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_s>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_s>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_m>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q8_0>     { static constexpr int qk = QK8_0; };
template <> struct fg_block_traits<block_q4_K>     { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q5_K>     { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q6_K>     { static constexpr int qk = QK_K; };

// Which half of a stored block the register k loop holds: the whole block (MMID_SCHED_REGS_A) or
// only its cheap streams (MMID_SCHED_REGS_LITE_A). The walk over stored blocks is the same either
// way, so the loop is written once and only these two calls change. decode() of the full variant
// needs no weight pointer; it takes and ignores one so both policies read the same.
template <typename block_q_t, bool LITE> struct fg_regs_policy;

template <typename block_q_t> struct fg_regs_policy<block_q_t, false> {
    using held = typename fg_reorder_a<block_q_t>::regs;

    static __dpct_inline__ held load(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        return fg_reorder_a<block_q_t>::load(xb, ib, nblocks);
    }

    template <typename E>
    static __dpct_inline__ void decode(const held & r, const uint8_t * __restrict__, const int, const int,
                                       const int ib8, typename E::pair * a) {
        fg_reorder_a<block_q_t>::template decode<E>(r, ib8, a);
    }
};

template <typename block_q_t> struct fg_regs_policy<block_q_t, true> {
    using held = typename fg_reorder_a<block_q_t>::regs_lite;

    static __dpct_inline__ held load(const uint8_t * __restrict__ xb, const int ib, const int nblocks) {
        return fg_reorder_a<block_q_t>::load_lite(xb, ib, nblocks);
    }

    template <typename E>
    static __dpct_inline__ void decode(const held & r, const uint8_t * __restrict__ xb, const int ib,
                                       const int nblocks, const int ib8, typename E::pair * a) {
        fg_reorder_a<block_q_t>::template decode_lite<E>(r, xb, ib, nblocks, ib8, a);
    }
};

// ---------------------------------------------------------------------------------------------
// SLM A staging (GGML_SYCL_MMID_SCHED_SLM_A). The A gather is the L3 read this kernel is bound
// on: a lane owns one row, so the lane-to-lane stride of every weight field is
// blocks_per_row * field_stride, which is far wider than a cache line -- each of a format's
// streams touches one line per lane to deliver a handful of bytes. The same line then serves the
// 8 k steps of its superblock, but ~32 KiB of packed B streams past in between and evicts it.
//
// So: copy the RAW QUANTIZED bytes of one stored block of every row of the M tile into SLM,
// cooperatively, and decode out of SLM. The lanes then cooperate on contiguous runs (one line per
// run instead of one line per lane) and the block stays resident for all of its k steps.
//
// The image is laid out so the A stages above decode it UNCHANGED: the SoA image is the same
// stream layout with the slice's block count replaced by SG_ROWS and the block index by the local
// row, and the canonical image is simply SG_ROWS consecutive blocks. Only the base pointer and
// its address space change, so all 13 fg_stage_a() overloads and all 6 fg_reorder_a<>
// specializations stay exactly as they are.
// ---------------------------------------------------------------------------------------------

// Bytes per stored block of each stream of the reorder (SoA) layout, in the order the reorder
// writes them: the prefix sums of this list are exactly the stream bases the fg_reorder_a<>
// specializations above compute, so the two must be kept in step. The primary template describes
// the canonical (AoS) block, which is one stream of the whole block.
template <typename block_q_t> struct fg_soa_layout {
    static constexpr int nstreams  = 1;
    static constexpr int stream[1] = { (int) sizeof(block_q_t) };
};

template <> struct fg_soa_layout<block_iq4_nl> {
    static constexpr int nstreams  = 2;
    static constexpr int stream[2] = { QK4_NL / 2, (int) sizeof(ggml_half) };
};

template <> struct fg_soa_layout<block_iq3_s> {
    static constexpr int nstreams  = 4;
    static constexpr int stream[4] = { QK_K / 4, QK_K / 32, QK_K / 8, (int) sizeof(ggml_half) + IQ3S_N_SCALE };
};

template <> struct fg_soa_layout<block_q8_0> {
    static constexpr int nstreams  = 2;
    static constexpr int stream[2] = { QK8_0, (int) sizeof(ggml_half) };
};

template <> struct fg_soa_layout<block_q4_K> {
    static constexpr int nstreams  = 3;
    static constexpr int stream[3] = { QK_K / 2, K_SCALE_SIZE, (int) sizeof(ggml_half2) };
};

template <> struct fg_soa_layout<block_q5_K> {
    static constexpr int nstreams  = 4;
    static constexpr int stream[4] = { QK_K / 2, QK_K / 8, K_SCALE_SIZE, (int) sizeof(ggml_half2) };
};

template <> struct fg_soa_layout<block_q6_K> {
    static constexpr int nstreams  = 4;
    static constexpr int stream[4] = { QK_K / 2, QK_K / 4, QK_K / 16, (int) sizeof(ggml_half) };
};

// The SoA layout is a byte permutation of the block, so the streams must account for every byte
// of it -- that is also what makes one staged row cost sizeof(block_q_t) in either layout.
template <typename block_q_t> static constexpr int fg_soa_total() {
    int total = 0;
    for (int f = 0; f < fg_soa_layout<block_q_t>::nstreams; ++f) {
        total += fg_soa_layout<block_q_t>::stream[f];
    }
    return total;
}

static_assert(fg_soa_total<block_iq4_nl>() == (int) sizeof(block_iq4_nl), "iq4_nl SoA streams must cover the block");
static_assert(fg_soa_total<block_iq3_s>()  == (int) sizeof(block_iq3_s),  "iq3_s SoA streams must cover the block");
static_assert(fg_soa_total<block_q8_0>()   == (int) sizeof(block_q8_0),   "q8_0 SoA streams must cover the block");
static_assert(fg_soa_total<block_q4_K>()   == (int) sizeof(block_q4_K),   "q4_K SoA streams must cover the block");
static_assert(fg_soa_total<block_q5_K>()   == (int) sizeof(block_q5_K),   "q5_K SoA streams must cover the block");
static_assert(fg_soa_total<block_q6_K>()   == (int) sizeof(block_q6_K),   "q6_K SoA streams must cover the block");

// SLM one sub-group's staged image takes, padded to a dword so the next sub-group's image starts
// dword aligned as well.
template <typename block_q_t, int SG_ROWS> static constexpr int fg_slm_x_region() {
    return (SG_ROWS * (int) sizeof(block_q_t) + 3) & ~3;
}

// One contiguous run, moved cooperatively by the lanes of a sub-group. This is the whole point of
// the staging: the run is contiguous, so the lanes coalesce onto the one or two lines it spans
// instead of taking a line each.
template <typename T>
static __dpct_inline__ void fg_copy_run_as(uint8_t * __restrict__ dst, const uint8_t * __restrict__ src, const int n,
                                           const int lane) {
    T *       d = (T *) dst;
    const T * s = (const T *) src;
    for (int i = lane; i < n / (int) sizeof(T); i += WARP_SIZE) {
        d[i] = s[i];
    }
}

// `al` is how wide an element the run may be moved in: 1, 2 or 4 bytes.
static __dpct_inline__ void fg_copy_run(uint8_t * __restrict__ dst, const uint8_t * __restrict__ src, const int n,
                                        const int al, const int lane) {
    if (al >= 4) {
        fg_copy_run_as<uint32_t>(dst, src, n, lane);
    } else if (al >= 2) {
        fg_copy_run_as<uint16_t>(dst, src, n, lane);
    } else {
        fg_copy_run_as<uint8_t>(dst, src, n, lane);
    }
}

// Stage stored block `ib_blk` of every row of the M tile into this sub-group's SLM image.
// `base_al` is the runtime alignment of the weight base (1, 2 or 4); every stream offset is a
// compile-time multiple of the stream size, so that is the only runtime part of the alignment.
// Rows past M are skipped: the decode below never reads them, exactly as it never decodes them
// today, so their image bytes stay untouched.
template <typename block_q_t, bool reordered, int SG_ROWS>
static __dpct_inline__ void fg_stage_block_slm(const uint8_t * __restrict__ xb, uint8_t * __restrict__ slm,
                                               const int blocks_per_row, const int nblocks, const int m0, const int M,
                                               const int ib_blk, const int lane, const int base_al) {
    if constexpr (reordered) {
        int prefix = 0;
#pragma unroll
        for (int f = 0; f < fg_soa_layout<block_q_t>::nstreams; ++f) {
            const int S = fg_soa_layout<block_q_t>::stream[f];
            // source is xb + nblocks*prefix + ib*S, destination slm + SG_ROWS*prefix + r*S. Both
            // stream bases are multiples of 4 whenever prefix is (SG_ROWS is at least 16), so the
            // alignment of the run is decided by S, prefix and the weight base.
            const int al_ct = ((S | prefix) & 3) == 0 ? 4 : (((S | prefix) & 1) == 0 ? 2 : 1);
            const int al    = al_ct < base_al ? al_ct : base_al;

            const uint8_t * gs = xb + (size_t) nblocks * prefix;
            uint8_t *       ls = slm + SG_ROWS * prefix;
            for (int r = 0; r < SG_ROWS; ++r) {
                if (m0 + r >= M) {
                    break;
                }
                const int ib = (m0 + r) * blocks_per_row + ib_blk;
                fg_copy_run(ls + r * S, gs + (size_t) ib * S, S, al, lane);
            }
            prefix += S;
        }
    } else {
        constexpr int S     = (int) sizeof(block_q_t);
        const int     al_ct = (S & 3) == 0 ? 4 : ((S & 1) == 0 ? 2 : 1);
        const int     al    = al_ct < base_al ? al_ct : base_al;
        for (int r = 0; r < SG_ROWS; ++r) {
            if (m0 + r >= M) {
                break;
            }
            const int ib = (m0 + r) * blocks_per_row + ib_blk;
            fg_copy_run(slm + r * S, xb + (size_t) ib * S, S, al, lane);
        }
    }
}

// Elements one image of the staged A tile takes. The kernel holds one, or two when the A stage is
// software pipelined.
template <int SG_ROWS> static constexpr int fg_a_image() {
    return FG_KSPLIT * SG_ROWS * FG_BK;
}

// Where row `lrow` of a sub-group's A image starts, and where the (mt, kt) TM x TK A tile starts,
// both in elements. Two layouts of the same elements; the image size is the same either way.
//
// BLK_A off: the image is row major with a row stride of FG_BK, and the matrix load of a tile is
// given that stride. Only TK of every FG_BK elements belong to the tile, so the load is not one
// contiguous run and IGC emits one scalar message per row: 8 a tile, 32 a k step (f16 8x16x16).
// BLK_A on: the image holds whole tiles instead, TM * TK elements each, so an f16 8x16 tile is 256
// contiguous bytes and loads in one message. A row is then FG_BK / TK runs of TK elements,
// TM * TK apart, and the lane that owns the row writes it as that many runs.
template <typename S, bool BLK_A> static constexpr int fg_a_row_off(const int lrow) {
    if constexpr (BLK_A) {
        return (lrow / S::TM) * (S::TM * FG_BK) + (lrow % S::TM) * S::TK;
    }
    return lrow * FG_BK;
}

template <typename S, bool BLK_A> static constexpr int fg_a_tile_off(const int mt, const int kt) {
    if constexpr (BLK_A) {
        return (mt * (FG_BK / S::TK) + kt) * (S::TM * S::TK);
    }
    return (mt * S::TM) * FG_BK + kt * S::TK;
}

// Rows of C a work-group holds K-split partials for at one time. The joint_matrix store and the
// K-split reduction walk this many rows per pass, so it sets tile_c; the M tile does not, because a
// wider M tile just runs more passes (PASSES below).
//
// SPLIT_C off: S::SG_ROWS = 16 rows (f16 8x16x16), so tile_c is FG_KSPLIT * 16 * 32 floats = 8 KiB,
// two thirds of the 12 KiB a 16-row work-group asks for.
// SPLIT_C on (GGML_SYCL_MMID_SCHED_SPLIT_C): one accumulator tile's worth of rows, TM = 8, so
// tile_c is 4 KiB and the 16-row work-group asks for 8. The epilogue then runs twice as many of
// exactly the passes it already runs for the wide M tiles, and each pass still sums the four K
// splits in ascending sub-group order, so the result is bit for bit the off path's.
template <typename S, bool SPLIT_C> static constexpr int fg_c_rows() {
    return SPLIT_C ? S::TM : S::SG_ROWS;
}

// floats the C region of tile_c holds. The staged A image and the staged lookup table, when there
// are any, are carved out of the tail past it, so a work-group with both off asks for exactly this.
template <typename S, bool SPLIT_C> static constexpr int fg_tile_c_floats() {
    return FG_KSPLIT * fg_c_rows<S, SPLIT_C>() * S::BN;
}

// ---------------------------------------------------------------------------------------------
// SLM lookup table (GGML_SYCL_MMID_SCHED_GRID_SLM). The iq3_s decode takes 8 iq3s_grid entries per
// k step per lane, at data-dependent indices, so the lanes diverge and IGC emits 8 stateless
// load.ugm.d32.a64 messages and 8 64-bit address builds per step. The table is 2 KiB, so a
// work-group can hold the whole thing in SLM: one cooperative copy at kernel entry, one barrier,
// and the 8 gathers become load.slm.d32.a32 on a copy that stays resident.
//
// The cost is 2 KiB of SLM per work-group, and SLM is what limits residency here: 12 KiB today
// (tile_a 4 KiB + tile_c 8 KiB at 16 rows) gives floor(128/12) = 10 resident work-groups, 14 KiB
// gives 9. So this trades one tenth of the occupancy for the gather.
//
// Only iq3_s is wired up. The other formats with a table are iq3_xxs (iq3xxs_grid, 1 KiB),
// iq2_xxs (iq2xxs_grid, 2 KiB), iq2_xs (iq2xs_grid, 4 KiB), iq2_s (iq2s_grid, 8 KiB) and
// iq1_s/iq1_m (iq1s_grid_gpu, 8 KiB); the first two would take the same mechanism at the same or
// lower SLM cost.
// ---------------------------------------------------------------------------------------------

template <typename block_q_t> struct fg_grid_traits {
    static constexpr int dwords = 0;
};

template <> struct fg_grid_traits<block_iq3_s> {
    static constexpr int dwords = 512;
};

static_assert(fg_grid_traits<block_iq3_s>::dwords == (int) (sizeof(iq3s_grid) / sizeof(iq3s_grid[0])),
              "the staged copy must be the whole iq3s_grid");

template <typename block_q_t, bool GRID_SLM> static constexpr int fg_slm_grid_dwords() {
    return GRID_SLM ? fg_grid_traits<block_q_t>::dwords : 0;
}

// Copy the format's table into the tail of tile_c, past the C region and the A image region, and
// give back the pointer the decode reads. With the bit off nothing is copied and the decode reads
// the table where it reads it today.
template <typename block_q_t, bool GRID_SLM, int TILE_C_FLOATS, int SLM_X_FLOATS>
static __dpct_inline__ const uint32_t * fg_stage_grid_slm(sycl::local_accessor<float, 1> tile_c,
                                                          const sycl::nd_item<2> & item) {
    if constexpr (fg_slm_grid_dwords<block_q_t, GRID_SLM>() > 0) {
        constexpr int N   = fg_grid_traits<block_q_t>::dwords;
        uint32_t *    slm = (uint32_t *) &tile_c[TILE_C_FLOATS + SLM_X_FLOATS];
        for (int i = item.get_local_linear_id(); i < N; i += (int) item.get_local_range(1)) {
            slm[i] = iq3s_grid[i];
        }
        // written by the whole work-group, read by every lane for the rest of the kernel
        sycl::group_barrier(item.get_group());
        return slm;
    } else {
        return iq3s_grid;
    }
}

// The A stage with the table handed in. Only iq3_s reads one, so for every other format this is
// the call it makes today and the pointer is dropped.
template <typename E, typename block_q_t>
static __dpct_inline__ void fg_stage_a_grid(const block_q_t * __restrict__ xrow, const int kb, typename E::pair * a,
                                            const uint32_t * __restrict__ grid) {
    if constexpr (fg_grid_traits<block_q_t>::dwords > 0) {
        fg_stage_a<E>(xrow, kb, a, grid);
    } else {
        (void) grid;
        fg_stage_a<E>(xrow, kb, a);
    }
}

template <typename E, typename block_q_t>
static __dpct_inline__ void fg_reorder_stage_grid(const uint8_t * __restrict__ xb, const int ib_row,
                                                  const int nblocks, const int kb, typename E::pair * a,
                                                  const uint32_t * __restrict__ grid) {
    if constexpr (fg_grid_traits<block_q_t>::dwords > 0) {
        fg_reorder_a<block_q_t>::template stage<E>(xb, ib_row, nblocks, kb, a, grid);
    } else {
        (void) grid;
        fg_reorder_a<block_q_t>::template stage<E>(xb, ib_row, nblocks, kb, a);
    }
}


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
template <typename S, typename block_q_t, bool reordered, int SG_ROWS, bool STAGE_A, bool PIPELINE_A, bool REGS_A,
          bool REGS_LITE, bool BLK_A, bool GRID_SLM, bool SPLIT_C>
static void fused_dequant_gemm_tile(
    const block_q_t * __restrict__ x,
    const typename S::tsb * __restrict__ packed_b,
    float * __restrict__ dst,
    const int M, const int Npad, const int K, const int ldd,
    const int b0, const int n0, const int n1,
    const ggml_sycl_gg_rows dst_rows,
    const bool diag_no_a, const bool diag_no_mad,
    sycl::local_accessor<typename S::tsa, 1> tile_a,
    sycl::local_accessor<float, 1> tile_c,
    const sycl::nd_item<2> & item) {
    static_assert(!reordered || fg_reorder_a<block_q_t>::supported,
                  "no reorder A stage for this weight format; the canonical decode would read garbage");
    static_assert(SG_ROWS % S::SG == 0 && SG_ROWS % S::SG_ROWS == 0,
                  "rows must split over the lanes, and over the S::SG_ROWS the K-split buffer holds");
    static_assert(fg_is_tuned<S>() || (SG_ROWS == S::SG_ROWS && !STAGE_A && !PIPELINE_A && !REGS_A && !REGS_LITE &&
                                       !BLK_A && !GRID_SLM && !SPLIT_C),
                  "the GGML_SYCL_MMID_SCHED variants are built for the tuned combination only");
    static_assert(!STAGE_A || S::SG == WARP_SIZE, "the SLM A staging copies in WARP_SIZE strides");
    static_assert(!(PIPELINE_A && STAGE_A),
                  "the pipeline wants two A images; it is not offered together with the SLM A staging");
    static_assert(!(REGS_A && (STAGE_A || PIPELINE_A)),
                  "the register A stage restructures the k loop itself; it is offered on its own");
    static_assert(!REGS_A || reordered,
                  "the register A stage is built on the reorder (SoA) load/decode split only");
    static_assert(!(REGS_LITE && (STAGE_A || PIPELINE_A || REGS_A)),
                  "the partial register A stage walks stored blocks itself; it is offered on its own");
    static_assert(!REGS_LITE || (reordered && fg_reorder_a<block_q_t>::lite),
                  "the partial register A stage needs the reorder (SoA) load_lite/decode_lite split");
    static_assert(!(BLK_A && (REGS_A || REGS_LITE)),
                  "the register A stages already hold a block; the tiled A image is not offered with them");
    static_assert(!GRID_SLM || fg_grid_traits<block_q_t>::dwords > 0,
                  "this weight format reads no lookup table, so there is nothing to stage for it");
    static_assert(!(GRID_SLM && (REGS_A || REGS_LITE)),
                  "the register A stages decode through their own path, which still reads the global table");
    if constexpr (fg_built<S::SG>()) {
        using EA = typename S::EA;
        using TA = typename S::ta;
        using TB = typename S::tb;
        using pair_t = typename EA::pair;
        // M tiles kept in registers, rows one lane stages, and store passes through tile_c
        constexpr int MT            = SG_ROWS / S::TM;
        constexpr int NT            = S::NT;
        constexpr int RPL           = SG_ROWS / S::SG;
        constexpr int C_ROWS        = fg_c_rows<S, SPLIT_C>();
        constexpr int MT_PASS       = C_ROWS / S::TM;
        constexpr int PASSES        = SG_ROWS / C_ROWS;
        constexpr int TILE_C_FLOATS = fg_tile_c_floats<S, SPLIT_C>();
        static_assert(C_ROWS % S::TM == 0 && SG_ROWS % C_ROWS == 0,
                      "a C pass must be a whole number of accumulator tiles, and the M tile a whole number of C passes");
        static_assert(MT_PASS * PASSES == MT, "every accumulator tile must be stored by exactly one pass");

        const auto sg     = item.get_sub_group();
        const int  sg_id  = sg.get_group_id()[0];
        const int  lane   = sg.get_local_id()[0];
        const int  m0     = item.get_group(1) * SG_ROWS;
        const int  nstep  = K / FG_BK;
        const int  a_base = sg_id * SG_ROWS * FG_BK;
        const int  c_base = sg_id * C_ROWS * S::BN;

        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, S::TM, S::TN> acc[MT][NT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // reorder offsets are relative to the slice, so nblocks counts this slice only
        const int blocks_per_row = K / fg_block_traits<block_q_t>::qk;
        const int nblocks        = M * blocks_per_row;

        // a lane owns rows lane, lane + SG, ... of the tile; tile_a row r is global row m0 + r.
        // With the pipeline on tile_a holds two images of the tile and a[b] points into image b.
        bool              row_ok[RPL];
        const block_q_t * xrow[RPL];
        int               ib_row[RPL];
        pair_t *          a[2][RPL];
#pragma unroll
        for (int rr = 0; rr < RPL; ++rr) {
            const int lrow  = rr * S::SG + lane;
            const int row   = m0 + lrow;
            const int row_a = row < M ? row : 0;
            row_ok[rr]      = row < M;
            xrow[rr]        = x + (size_t) row_a * blocks_per_row;
            ib_row[rr]      = row_a * blocks_per_row;
            a[0][rr]        = (pair_t *) &tile_a[a_base + fg_a_row_off<S, BLK_A>(lrow)];
            if constexpr (PIPELINE_A) {
                a[1][rr] = (pair_t *) &tile_a[fg_a_image<SG_ROWS>() + a_base + fg_a_row_off<S, BLK_A>(lrow)];
            }
        }

        const auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        const int b_stride = Npad * S::VNNI;

        // SLM A staging. The image lives in the tail of tile_c, past the TILE_C_FLOATS floats the
        // epilogue uses: a second local_accessor would be a second SLM allocation, and this way the
        // work-group asks for exactly today's SLM whenever STAGE_A is off. tile_c is float typed, so
        // the image base is dword aligned, and the epilogue never reaches past TILE_C_FLOATS.
        constexpr int KPB          = (STAGE_A || REGS_A || REGS_LITE) ? fg_block_traits<block_q_t>::qk / FG_BK : 1;  // k steps per stored block
        constexpr int SLM_X_REGION = fg_slm_x_region<block_q_t, SG_ROWS>();
        static_assert(!STAGE_A || alignof(block_q_t) <= 4,
                      "the staged image is dword aligned, so a block of this format must not need more");
        uint8_t * slm_x   = nullptr;
        int       base_al = 0;
        if constexpr (STAGE_A) {
            slm_x = (uint8_t *) &tile_c[TILE_C_FLOATS + sg_id * (SLM_X_REGION / 4)];
            // the weight base is the only runtime part of the run alignment: every stream offset is a
            // compile-time multiple of its stream size
            base_al = (((uintptr_t) x) & 3) == 0 ? 4 : ((((uintptr_t) x) & 1) == 0 ? 2 : 1);
        }
        (void) slm_x;
        (void) base_al;

        // The format's lookup table, copied into SLM once per work-group and read by every k step of
        // every lane after it. It sits past the staged A image, which is itself past the C region, so
        // with both bits off the work-group's SLM request is exactly today's.
        constexpr int    SLM_X_FLOATS = STAGE_A ? FG_KSPLIT * (SLM_X_REGION / 4) : 0;
        const uint32_t * grid = fg_stage_grid_slm<block_q_t, GRID_SLM, TILE_C_FLOATS, SLM_X_FLOATS>(tile_c, item);
        (void) grid;

        // The two halves of a k step. Split so the pipelined loop below can put the MADs of one step
        // between the staging of the next step and the barrier that publishes it.
        auto stage_a = [&](pair_t * const * ad, const int kb) {
            if (diag_no_a) { return; }
#pragma unroll
            for (int rr = 0; rr < RPL; ++rr) {
                // With the tiled image a row is not one run of tile_a, so the decode writes a private
                // buffer (FG_BK elements, one row live at a time) and the runs are stored below. The
                // decode itself is untouched, and with BLK_A off `out` is ad[rr] as before.
                pair_t   priv[BLK_A ? FG_BK / 2 : 1];
                pair_t * out = BLK_A ? priv : ad[rr];
                if (row_ok[rr]) {
                    if constexpr (STAGE_A) {
                        // the image is a slice of SG_ROWS blocks, one per row of the M tile, so the
                        // A stage reads it with the local row as its block index and the step inside
                        // the stored block as its k step
                        const int lrow = rr * S::SG + lane;
                        if constexpr (reordered) {
                            fg_reorder_stage_grid<EA, block_q_t>(slm_x, lrow, SG_ROWS, kb % KPB, out, grid);
                        } else {
                            fg_stage_a_grid<EA, block_q_t>((const block_q_t *) slm_x + lrow, kb % KPB, out, grid);
                        }
                    } else if constexpr (reordered) {
                        fg_reorder_stage_grid<EA, block_q_t>((const uint8_t *) x, ib_row[rr], nblocks, kb, out, grid);
                    } else {
                        fg_stage_a_grid<EA, block_q_t>(xrow[rr], kb, out, grid);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < FG_BK / 2; ++j) {
                        out[j] = EA::make(0.0f, 0.0f);
                    }
                }
                if constexpr (BLK_A) {
#pragma unroll
                    for (int kt = 0; kt < FG_BK / S::TK; ++kt) {
#pragma unroll
                        for (int j = 0; j < S::TK / 2; ++j) {
                            ad[rr][kt * (S::TM * S::TK / 2) + j] = priv[kt * (S::TK / 2) + j];
                        }
                    }
                }
            }
        };

        auto mad_a = [&](const int abase, const int kb) {
            if (diag_no_mad) { return; }
#pragma unroll
            for (int kt = 0; kt < FG_BK / S::TK; ++kt) {
                const int kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_matrix<sycl::sub_group, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[NT];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mx::joint_matrix<sycl::sub_group, TA, mx::use::a, S::TM, S::TK, mx::layout::row_major> sub_a;
                    mx::joint_matrix_load(sg, sub_a,
                        tile_a.template get_multi_ptr<sycl::access::decorated::no>() + abase + fg_a_tile_off<S, BLK_A>(mt, kt),
                        BLK_A ? S::TK : FG_BK);
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
        };

        const int kb_begin = (sg_id * nstep) / FG_KSPLIT;
        const int kb_end   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        if constexpr (REGS_A || REGS_LITE) {
            // One stored block serves KPB k steps, so walk stored blocks and hold the block in
            // registers over its sub-blocks: the format's streams are read once per block instead of
            // once per k step, and nothing else about a k step changes. tile_a still holds one k step.
            // REGS_LITE holds only the cheap streams and lets decode() re-read the wide ones from
            // global memory, so the two differ in what load() takes, not in the walk.
            //
            // kb_begin and kb_end come from sg_id alone, so the whole walk is sub-group uniform, but
            // they are not multiples of KPB -- with nstep 80 and FG_KSPLIT 4 they are 20, 40 and 60.
            // The first and the last stored block of a range are therefore partial, which is why the
            // sub-block loop is a full unroll over the block with the k steps outside the range
            // masked off rather than a loop from a runtime first to a runtime last sub-block. The
            // unroll is also what keeps the sub-block index a constant, so the block stays in
            // registers instead of being addressed indirectly.
            using pol = fg_regs_policy<block_q_t, REGS_LITE>;
            typename pol::held blk[RPL];
            const int ib_begin = kb_begin / KPB;
            const int ib_end   = kb_begin < kb_end ? (kb_end + KPB - 1) / KPB : ib_begin;
            for (int ib = ib_begin; ib < ib_end; ++ib) {
#pragma unroll
                for (int rr = 0; rr < RPL; ++rr) {
                    if (row_ok[rr]) {
                        blk[rr] = pol::load((const uint8_t *) x, ib_row[rr] + ib, nblocks);
                    }
                }
#pragma unroll
                for (int ib8 = 0; ib8 < KPB; ++ib8) {
                    const int kb = ib * KPB + ib8;
                    if (kb >= kb_begin && kb < kb_end) {
#pragma unroll
                        for (int rr = 0; rr < RPL; ++rr) {
                            if (row_ok[rr]) {
                                pol::template decode<EA>(blk[rr], (const uint8_t *) x, ib_row[rr] + ib, nblocks, ib8, a[0][rr]);
                            } else {
#pragma unroll
                                for (int j = 0; j < FG_BK / 2; ++j) {
                                    a[0][rr][j] = EA::make(0.0f, 0.0f);
                                }
                            }
                        }
                        sycl::group_barrier(sg);
                        mad_a(a_base, kb);
                        // the next k step overwrites tile_a
                        sycl::group_barrier(sg);
                    }
                }
            }
        } else if constexpr (PIPELINE_A) {
            // The A of step kb + 1 is staged into the other image while the MADs of step kb read the
            // current one, so a staging load is no longer followed by a barrier that the matrix load
            // waiting on it must cross. One barrier per step instead of two: the barrier before the
            // MADs of a step both publishes the image the previous step staged, and holds the staging
            // of the next step off the image the previous step's MADs read. kb_begin and kb_end come
            // from sg_id alone, so every branch here is sub-group uniform.
            // Unrolled by two, which keeps the two image bases a compile-time distance apart.
            if (kb_begin < kb_end) {
                stage_a(a[0], kb_begin);
            }
            int kb = kb_begin;
            for (; kb + 1 < kb_end; kb += 2) {
                sycl::group_barrier(sg);
                stage_a(a[1], kb + 1);
                mad_a(a_base, kb);
                sycl::group_barrier(sg);
                if (kb + 2 < kb_end) {
                    stage_a(a[0], kb + 2);
                }
                mad_a(a_base + fg_a_image<SG_ROWS>(), kb + 1);
            }
            if (kb < kb_end) {
                sycl::group_barrier(sg);
                mad_a(a_base, kb);
            }
        } else {
            for (int kb = kb_begin; kb < kb_end; ++kb) {
                if constexpr (STAGE_A) {
                    // kb is sub-group uniform, so either every lane of the sub-group reaches the
                    // barrier below or none does. The image this overwrites was last read before the
                    // barrier that closes the previous k step, so the copy cannot outrun a decode
                    // still reading it.
                    if (kb == kb_begin || kb % KPB == 0) {
                        fg_stage_block_slm<block_q_t, reordered, SG_ROWS>((const uint8_t *) x, slm_x, blocks_per_row,
                                                                          nblocks, m0, M, kb / KPB, lane, base_al);
                        // written by the whole sub-group, read below by every lane
                        sycl::group_barrier(sg);
                    }
                }
                stage_a(a[0], kb);
                sycl::group_barrier(sg);
                mad_a(a_base, kb);
                // the next step overwrites tile_a
                sycl::group_barrier(sg);
            }
        }

        // tile_c holds C_ROWS rows per sub-group, so a wider M tile -- or, under SPLIT_C, a narrower
        // C region -- stores and reduces in passes. Every pass reduces the FG_KSPLIT partials of its
        // own rows in ascending sub-group (ascending K) order, which is the order regardless of how the
        // rows are split over passes, so PASSES does not change the arithmetic.
#pragma unroll
        for (int p = 0; p < PASSES; ++p) {
            if (p > 0) {
                // the previous pass still reads tile_c
                sycl::group_barrier(item.get_group());
            }
#pragma unroll
            for (int mt = 0; mt < MT_PASS; ++mt) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    mx::joint_matrix_store(sg, acc[p * MT_PASS + mt][nt],
                        tile_c.template get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * S::TM) * S::BN + nt * S::TN,
                        S::BN, mx::layout::row_major);
                }
            }
            sycl::group_barrier(item.get_group());

            // sum the K splits; consecutive lanes write consecutive rows of one dst column
            for (int idx = item.get_local_linear_id(); idx < C_ROWS * S::BN; idx += S::WG_SIZE) {
                const int r = idx % C_ROWS;
                const int c = idx / C_ROWS;
                const int m = m0 + p * C_ROWS + r;
                const int n = n0 + c;
                if (m < M && n < n1) {
                    float sum = 0.0f;
#pragma unroll
                    for (int s = 0; s < FG_KSPLIT; ++s) {
                        sum += tile_c[s * C_ROWS * S::BN + r * S::BN + c];
                    }
                    if (dst_rows.map) {
                        // grouped: routed row n goes straight to its place in the MUL_MAT_ID dst
                        const mmid_row_mapping rm = dst_rows.map[n];
                        ((float *) (dst_rows.base + (rm.i1 % dst_rows.ne1) * dst_rows.nb1 + rm.i2 * dst_rows.nb2))[m] = sum;
                    } else {
                        dst[(size_t) n * ldd + m] = sum;
                    }
                }
            }
        }
    }
}

template <typename S, typename block_q_t, bool reordered>
static void fused_dequant_gemm_launch(const void * src0, const typename S::tsb * packed, float * dst, const int M,
                                      const int N, const int Npad, const int K, const int ldd,
                                      const int64_t groups_n, const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                const int n0 = item.get_group(0) * S::BN;
                // the plain path has at most two column groups, so its B re-read is already small: it keeps
                // the narrow M tile and the tuned variants are instantiated for the grouped path only
                fused_dequant_gemm_tile<S, block_q_t, reordered, S::SG_ROWS, false, false, false, false, false, false,
                                        false>((const block_q_t *) src0, packed, dst, M, Npad, K, ldd, n0, n0, N,
                                               ggml_sycl_gg_rows{}, false, false, tile_a, tile_c, item);
            });
    });
}

// A kernel functor instead of a lambda, so the GRF size can ride along as a kernel property:
// get(properties_tag) is the way to attach one that handler::parallel_for does not deprecate.
// With GRF256 clear the list is empty, which is the same kernel the lambda produced.
// grouped: work-group (t, mt) is tile t of the schedule. Its B columns sit at t * BN in the
// slotted packed-B layout, or at the tile's own first row in the tight one.
template <typename S, typename block_q_t, bool reordered, int SG_ROWS, bool GRF256, bool STAGE_A, bool PIPELINE_A,
          bool REGS_A, bool REGS_LITE, bool BLK_A, bool GRID_SLM, bool SPLIT_C>
struct grouped_dequant_gemm_kernel {
    const char *                             src0_dd;
    size_t                                   expert_stride;
    const ggml_sycl_gg_tile *                tiles;
    const typename S::tsb *                  packed;
    float *                                  dst;
    int                                      M;
    int                                      Npad;
    int                                      K;
    bool                                     tight;
    ggml_sycl_gg_rows                        dst_rows;
    bool                                     diag_no_a;
    bool                                     diag_no_mad;
    sycl::local_accessor<typename S::tsa, 1> tile_a;
    sycl::local_accessor<float, 1>           tile_c;

    [[sycl::reqd_sub_group_size(S::SG)]]
    void operator()(sycl::nd_item<2> item) const {
        const int               t    = item.get_group(0);
        const ggml_sycl_gg_tile tile = tiles[t];
        // a device-built schedule is launched on its upper bound, so the tail tiles are empty.
        // The whole work-group reads the same tile, so this returns before any barrier.
        if (tile.n1 <= tile.n0) {
            return;
        }
        // group-uniform: every work-item of the group takes the same branch
        const int         b0 = tight ? tile.n0 : t * S::BN;
        const block_q_t * x  = (const block_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
        fused_dequant_gemm_tile<S, block_q_t, reordered, SG_ROWS, STAGE_A, PIPELINE_A, REGS_A, REGS_LITE, BLK_A,
                                GRID_SLM, SPLIT_C>(x, packed, dst, M, Npad, K, M, b0, tile.n0, tile.n1, dst_rows,
                                                   diag_no_a, diag_no_mad, tile_a, tile_c, item);
    }

    auto get(syclexp::properties_tag) const {
        if constexpr (GRF256) {
            return syclexp::properties{ sycl::ext::intel::experimental::grf_size<256> };
        } else {
            return syclexp::properties{};
        }
    }
};

// tile_a grows with the M tile; tile_c does not, because the store and the K-split reduction
// walk fg_c_rows<S, SPLIT_C>() rows at a time. For the tuned f16 combination tile_a is
// FG_KSPLIT * SG_ROWS * FG_BK halves and tile_c is fg_tile_c_floats<S, SPLIT_C>() floats, so with
// SPLIT_C off: 16 rows: 4 + 8 KiB. 32 rows: 8 + 8 KiB. 64 rows: 16 + 8 = 24 KiB, still well inside
// the 128 KiB a work-group may ask for. With SPLIT_C on tile_c is 4 KiB throughout: 8, 12 and 20 KiB.
//
// STAGE_A adds one staged block per row per sub-group to the tail of tile_c, which is
// FG_KSPLIT * SG_ROWS * sizeof(block_q_t) bytes: for iq3_s (110 B a superblock) that is 6.9 KiB
// at 16 rows and 13.8 KiB at 32, taking the work-group to 18.9 and 29.8 KiB. With the bit off the
// expression below is the C region exactly, so the off path's SLM request does not move at all.
//
// PIPELINE_A asks for a second image of tile_a instead: 16 KiB at 16 rows and 24 KiB at 32, which
// is what the 32-row and the 64-row tile already ask for today.
//
// REGS_A asks for no shared local memory at all: it keeps the stored block in registers, so the
// work-group's request is today's and the cost is register pressure instead. REGS_LITE is the
// same trade for a few of the block's bytes instead of all of them.
//
// GRID_SLM adds the format's lookup table, once per work-group, to the same tail: 2 KiB for
// iq3_s, which takes the 16-row tile from 12 to 14 KiB and its resident work-groups from
// floor(128/12) = 10 to floor(128/14) = 9. Not free, and not per row or per sub-group.
//
// SPLIT_C goes the other way and is why it matters: with the A-tile load reblocked (BLK_A) the MAD
// phase is no longer message bound, and what limits this kernel is how many work-groups fit on an
// Xe-core, which SLM sets. The work-group is 64 work-items at SIMD16, so it is 4 of an Xe-core's 64
// thread slots and 16 resident work-groups is full occupancy. 12 KiB gives floor(128/12) = 10
// work-groups, 40 threads; halving tile_c to 4 KiB gives 8 KiB total, 16 work-groups and all 64.
template <typename S, typename block_q_t, bool reordered, int SG_ROWS, bool GRF256, bool STAGE_A, bool PIPELINE_A,
          bool REGS_A, bool REGS_LITE, bool BLK_A, bool GRID_SLM, bool SPLIT_C>
static void grouped_dequant_gemm_launch(const char * src0_dd, const size_t expert_stride,
                                        const ggml_sycl_gg_tile * tiles_ptr, const typename S::tsb * packed, float * dst,
                                        const int M, const int Npad, const int K, const bool tight,
                                        const ggml_sycl_gg_rows dst_rows,
                                        const bool diag_no_a, const bool diag_no_mad,
                                        const int64_t n_tiles, const int64_t groups_m, dpct::queue_ptr stream) {
    constexpr int slm_x_floats = STAGE_A ? FG_KSPLIT * (fg_slm_x_region<block_q_t, SG_ROWS>() / 4) : 0;
    constexpr int grid_floats  = fg_slm_grid_dwords<block_q_t, GRID_SLM>();
    constexpr int a_images     = PIPELINE_A ? 2 : 1;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(a_images * fg_a_image<SG_ROWS>(), cgh);
        sycl::local_accessor<float, 1> tile_c(fg_tile_c_floats<S, SPLIT_C>() + slm_x_floats + grid_floats, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            grouped_dequant_gemm_kernel<S, block_q_t, reordered, SG_ROWS, GRF256, STAGE_A, PIPELINE_A, REGS_A,
                                        REGS_LITE, BLK_A, GRID_SLM, SPLIT_C>{
                src0_dd, expert_stride, tiles_ptr, packed, dst, M, Npad, K, tight, dst_rows, diag_no_a, diag_no_mad,
                tile_a, tile_c });
    });
}

// k values one work-item of the wide B pack moves: one 64 B line of f32
static constexpr int FG_PACK_K = 16;

// src1 f32 rows -> packed [K/V][Npad][V]. The column runs fastest so a sub-group writes one
// contiguous run. Row n of B is read through y: in place through the route map when y.map is set,
// else at y.base + r * y.nb1 (a gathered copy).
//
// Two layouts, selected by `tight`:
//  - slotted (the original): tile t owns columns [t*BN, (t+1)*BN) and holds its rows [n0, n1)
//    there, zero past n1. Every tile base is BN-aligned, but a tile with one routed row still costs
//    a full BN-wide slot.
//  - tight: column n is simply row n. Tiles cover disjoint row ranges, so a tile can read its BN
//    columns from its own first row; the columns it reads past n1 belong to the next expert and are
//    real data, which is harmless because the GEMM masks its store by n1. Costs one column per
//    routed row instead of one per tile slot, at the price of an unaligned B base.
//
// Two widths: the narrow pack moves V values per work-item, the wide one FG_PACK_K values of one
// row, read as two aligned 32 B vectors, so the tile and route lookups and the index math run once
// per 64 B instead of once per 8 B. The wide one needs every row 32 B aligned. Same packed values.
// Slotted layout: the columns of an empty tile are not written by the wide pack. The GEMM returns
// before it reads them, and only the device schedule has empty tiles (its launch bound).
template <typename S>
static void grouped_gemm_pack_b(const ggml_sycl_gg_rows & y, typename S::tsb * packed, const ggml_sycl_gg_tile * tiles,
                                int Npad, int K, int total_rows, bool tight, bool wide, dpct::queue_ptr stream) {
    using E                  = typename S::EB;
    using tsb                = typename S::tsb;
    constexpr int            V    = S::VNNI;
    constexpr int            BN   = S::BN;
    const char *             base = y.base;
    const mmid_row_mapping * map  = y.map;
    const int64_t            ne1  = y.ne1;
    const size_t             nb1  = y.nb1;
    const size_t             nb2  = y.nb2;
    // which B row column n holds, and whether it holds one; false also for an empty tile
    auto col_row = [=](const int n, int & row) {
        if (tight) {
            row = n;
            return n < total_rows;
        }
        const ggml_sycl_gg_tile tile = tiles[n / BN];
        row = tile.n0 + n % BN;
        return row < tile.n1;
    };
    auto row_ptr = [=](const int row) {
        if (map) {
            const mmid_row_mapping rm = map[row];
            return (const float *) (base + (rm.i1 % ne1) * nb1 + rm.i2 * nb2);
        }
        return (const float *) (base + (size_t) row * nb1);
    };
    if (wide) {
        const int kgroups = K / FG_PACK_K;
        const int wg_k    = kgroups % 4 == 0 ? 4 : (kgroups % 2 == 0 ? 2 : 1);
        GGML_ASSERT(Npad % BN == 0);
        using vec8 = sycl::vec<float, 8>;
        stream->parallel_for(
            sycl::nd_range<2>(sycl::range<2>(kgroups, Npad), sycl::range<2>(wg_k, BN)), [=](sycl::nd_item<2> it) {
                const int kg = it.get_global_id(0);
                const int n  = it.get_global_id(1);
                if (!tight && tiles[n / BN].n1 <= tiles[n / BN].n0) {
                    return;
                }
                int        row;
                const bool in = col_row(n, row);
                vec8       lo(0.0f);
                vec8       hi(0.0f);
                if (in) {
                    const vec8 * src = (const vec8 *) (row_ptr(row) + kg * FG_PACK_K);
                    lo = src[0];
                    hi = src[1];
                }
                tsb * out = packed + ((size_t) kg * (FG_PACK_K / V) * Npad + n) * V;
#pragma unroll
                for (int j = 0; j < FG_PACK_K / 2; ++j) {
                    const float v = lo[j];
                    out[(size_t) (j / V) * Npad * V + j % V] = E::cvt(v);
                }
#pragma unroll
                for (int j = 0; j < FG_PACK_K / 2; ++j) {
                    const float v = hi[j];
                    const int   k = FG_PACK_K / 2 + j;
                    out[(size_t) (k / V) * Npad * V + k % V] = E::cvt(v);
                }
            });
        return;
    }
    const int kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        const size_t idx = id[0];
        const int    kq  = idx / Npad;
        const int    n   = idx - (size_t) kq * Npad;
        int          row;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        tsb vals[V] = {};
        if (col_row(n, row)) {
            const float * src = row_ptr(row) + V * kq;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        tsb * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// The wide pack reads a row as two 32 B vectors, so every row must start 32 B aligned.
static bool grouped_gemm_pack_b_wide_ok(const char * base, size_t nb1, size_t nb2, int K) {
    if (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_PACKB_NARROW) {
        return false;
    }
    return K % FG_PACK_K == 0 && ((uintptr_t) base | nb1 | nb2) % (8 * sizeof(float)) == 0;
}

template <typename T, bool R> struct fg_tag {
    using type                      = T;
    static constexpr bool reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
template <typename T, typename F> static bool fg_visit_layout(bool reordered, F && f) {
    if (!reordered) {
        f(fg_tag<T, false>{});
        return true;
    }
    if constexpr (fg_reorder_a<T>::supported) {
        f(fg_tag<T, true>{});
        return true;
    }
    return false;
}

// One place that maps a weight type to its block struct, so a new format is added once instead
// of once per launch site.
template <typename F> static bool fg_visit_type(ggml_type type, bool reordered, F && f) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:  return fg_visit_layout<block_iq4_nl>(reordered, f);
        case GGML_TYPE_IQ3_S:   return fg_visit_layout<block_iq3_s>(reordered, f);
        case GGML_TYPE_IQ4_XS:  return fg_visit_layout<block_iq4_xs>(reordered, f);
        case GGML_TYPE_IQ3_XXS: return fg_visit_layout<block_iq3_xxs>(reordered, f);
        case GGML_TYPE_IQ2_XXS: return fg_visit_layout<block_iq2_xxs>(reordered, f);
        case GGML_TYPE_IQ2_XS:  return fg_visit_layout<block_iq2_xs>(reordered, f);
        case GGML_TYPE_IQ2_S:   return fg_visit_layout<block_iq2_s>(reordered, f);
        case GGML_TYPE_IQ1_S:   return fg_visit_layout<block_iq1_s>(reordered, f);
        case GGML_TYPE_IQ1_M:   return fg_visit_layout<block_iq1_m>(reordered, f);
        case GGML_TYPE_Q8_0:    return fg_visit_layout<block_q8_0>(reordered, f);
        case GGML_TYPE_Q4_K:    return fg_visit_layout<block_q4_K>(reordered, f);
        case GGML_TYPE_Q5_K:    return fg_visit_layout<block_q5_K>(reordered, f);
        case GGML_TYPE_Q6_K:    return fg_visit_layout<block_q6_K>(reordered, f);
        default:                return false;
    }
}

// Rows of one work-group's M tile. A work-group streams the whole B tile of its column group,
// so a wider M tile means fewer row-groups and proportionally less packed-B traffic.
// The 32-row tile (bit 64) and the 64-row tile (bit 256) of the tuned combination.
static constexpr int FG_WIDE_ROWS = 2 * fg_tuned_combo::SG_ROWS;
static constexpr int FG_HUGE_ROWS = 2 * FG_WIDE_ROWS;

static int fg_mtile_rows() {
    // 64 wins over 32 when both bits are set: it is the strictly wider tile.
    if (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_MTILE64) {
        return FG_HUGE_ROWS;
    }
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_MTILE32) ? FG_WIDE_ROWS : fg_tuned_combo::SG_ROWS;
}

// 256 GRF for the 32-row M tile only. That tile holds 8 accumulator tiles instead of 4, which is
// where a spill would come from; the narrow one keeps the default and its thread count. The 64-row
// tile does not consult this: 16 accumulator tiles are already ~128 registers before sub_a and
// sub_b, so it is always built with 256 GRF and this flag could only ask it for a certain spill.
static bool fg_grf256() {
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_GRF256) != 0;
}

// Stage the raw quantized A bytes of a stored block through SLM. Not offered on the 64-row tile:
// that tile already asks for 24 KiB, and one staged iq3_s superblock per row per sub-group would
// add 27.5 KiB on top, which is a worse trade than the gather it removes. The plain (non-grouped)
// fused GEMM never takes it either -- it has at most two column groups, so it is not the kernel
// the A gather dominates.
static bool fg_slm_a(const int mrows) {
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_SLM_A) != 0 && mrows != FG_HUGE_ROWS;
}

// Software pipeline the A stage: stage k step n+1 into the second image of tile_a while the MADs
// of step n read the first. It doubles tile_a, so it is not offered on the 64-row tile, which asks
// for 24 KiB already. Never combined with the SLM staging above: that one costs a barrier per
// stored block, which is what this change is removing.
static bool fg_pipeline_a(const int mrows, const bool slm_a, const bool regs_a, const bool regs_lite_a) {
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_PIPELINE_A) != 0 && !slm_a && !regs_a && !regs_lite_a &&
           mrows != FG_HUGE_ROWS;
}

// Hold one stored weight block in registers over all of its k steps, instead of re-reading its
// fields at every k step. This is the same temporal reuse bit 512 buys through SLM, without the
// round trip and without its barrier per block, so the two are never combined. Not offered on the
// 64-row tile: a lane owns four rows there, so four blocks would be live at once. The reorder
// (SoA) layout only -- the canonical A stages have no load/decode split -- which the launcher
// below decides at compile time.
static bool fg_regs_a(const int mrows, const bool slm_a, const bool reordered) {
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_REGS_A) != 0 && reordered && !slm_a &&
           mrows != FG_HUGE_ROWS;
}

// Hold only the cheap, high-reuse streams of a stored block in registers and keep re-reading the
// wide ones per k step. Every stream is re-read once per k step whatever its width, so the few
// bytes of metadata (and the iq3_s qh) remove half of a block's global reads for about 4 registers
// a row, where the whole block costs 28 and does not fit in 128 next to the accumulators. Same
// exclusions as bit 2048: SoA only, never with bit 512, and not on the 64-row tile. Bit 2048 wins
// when both are set, because it is the strictly wider variant of the same idea.
static bool fg_regs_lite_a(const int mrows, const bool slm_a, const bool regs_a, const bool reordered) {
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_REGS_LITE_A) != 0 && reordered && !slm_a && !regs_a &&
           mrows != FG_HUGE_ROWS;
}

// Lay the staged A image out as whole 8x16 matrix tiles instead of row major. The matrix load of
// an A tile reads TK of every FG_BK halves today, which is not one run, so it costs one scalar
// message a row: 32 of the k loop's messages, more than half of them. A tiled image makes each of
// those loads one 256-byte message. Same SLM, same barriers, same decode; only the addresses the
// decode writes and the matrix load reads change. Offered on every tile width, but not with the
// two register A stages: those hold a stored block themselves and already spill at 128 registers,
// and this adds a row of decoded halves on top.
static bool fg_blk_a(const bool regs_a, const bool regs_lite_a) {
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_BLK_A) != 0 && !regs_a && !regs_lite_a;
}

// Hold the format's lookup table in SLM. Only iq3_s reads one in this kernel, and only its 8
// entries per k step per lane are a gather wide enough to pay for 2 KiB of SLM; the launcher below
// drops the request for every other format at compile time. Not offered with the two register A
// stages, whose decode path still reads the global table.
static bool fg_grid_slm(const bool regs_a, const bool regs_lite_a) {
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_GRID_SLM) != 0 && !regs_a && !regs_lite_a;
}

// Reduce the K-split partials one accumulator tile (TM = 8 rows) at a time instead of
// the 16 rows of S::SG_ROWS, which halves tile_c: 8 KiB to 4, and a 16-row work-group from 12 KiB to 8. With
// SLM the binding constraint on residency that is 10 resident work-groups to 16, 40 of an
// Xe-core's 64 thread slots to all 64. The cost is that the epilogue runs twice the passes, so two
// more group barriers per work-group, and that a dst store run is 8 rows (32 B) instead of 16
// (64 B). No exclusions: the pass loop is orthogonal to every A-stage arm, it is the loop the wide
// M tiles already run, and the summation order per output element is unchanged.
static bool fg_split_c() {
    return (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_SPLIT_C) != 0;
}



// The GGML_SYCL_MMID_SCHED tile and A-stage variants of one grouped launch. Only the tuned
// combination reads them; any other combination runs the plain tile.
struct fg_arms {
    int  mrows;        // fg_tuned_combo::SG_ROWS, FG_WIDE_ROWS or FG_HUGE_ROWS
    bool grf256;       // 32-row tile only; the 64-row one is always 256 GRF
    bool slm_a;        // stage the quantized A bytes through SLM; never on the 64-row tile
    bool pipeline_a;   // second tile_a image; never with slm_a, never on the 64-row tile
    bool regs_a;       // hold a stored block in registers; SoA only, never on the 64-row tile
    bool regs_lite_a;  // hold only the block's cheap streams; same exclusions as regs_a
    bool blk_a;        // tiled A image in SLM; never with the two regs arms
    bool grid_slm;     // the format's lookup table in SLM; iq3_s only, never with the regs arms
    bool split_c;      // half-height tile_c, reduced in twice the passes; no exclusions
    bool diag_no_a;    // DIAGNOSTIC: skip the A decode. Wrong output by construction.
    bool diag_no_mad;  // DIAGNOSTIC: skip the XMX MADs. Wrong output by construction.
};

static fg_arms fg_pick_arms(const bool reordered) {
    fg_arms r{};
    r.mrows       = fg_mtile_rows();
    r.grf256      = fg_grf256();
    r.slm_a       = fg_slm_a(r.mrows);
    r.regs_a      = fg_regs_a(r.mrows, r.slm_a, reordered);
    r.regs_lite_a = fg_regs_lite_a(r.mrows, r.slm_a, r.regs_a, reordered);
    r.pipeline_a  = fg_pipeline_a(r.mrows, r.slm_a, r.regs_a, r.regs_lite_a);
    r.blk_a       = fg_blk_a(r.regs_a, r.regs_lite_a);
    r.grid_slm    = fg_grid_slm(r.regs_a, r.regs_lite_a);
    r.split_c     = fg_split_c();
    r.diag_no_a   = (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_DIAG_NO_A) != 0;
    r.diag_no_mad = (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_DIAG_NO_MAD) != 0;
    return r;
}

template <typename S> struct grouped_gemm_launcher {
    const char *              src0_dd;
    size_t                    expert_stride;
    const ggml_sycl_gg_tile * tiles;
    const typename S::tsb *   packed;
    float *                   dst;
    int                       M;
    int                       Npad;
    int                       K;
    bool                      tight;
    ggml_sycl_gg_rows         dst_rows;  // map null: dst is expert-major and contiguous
    fg_arms                   arms;
    int64_t                   n_tiles;
    int64_t                   groups_m;
    dpct::queue_ptr           stream;

    template <typename block_q_t, bool reordered, int SG_ROWS, bool GRF256, bool STAGE_A, bool PIPELINE_A,
              bool REGS_A, bool REGS_LITE, bool BLK_A, bool GRID_SLM, bool SPLIT_C>
    void launch_c() const {
        grouped_dequant_gemm_launch<S, block_q_t, reordered, SG_ROWS, GRF256, STAGE_A, PIPELINE_A, REGS_A,
                                    REGS_LITE, BLK_A, GRID_SLM, SPLIT_C>(
            src0_dd, expert_stride, tiles, packed, dst, M, Npad, K, tight, dst_rows, arms.diag_no_a, arms.diag_no_mad,
            n_tiles, groups_m, stream);
    }

    // The height of the K-split buffer. It sits innermost, below every arm above it, because it is
    // orthogonal to all of them: no arm reads tile_c's C region and no arm changes the pass loop.
    template <typename block_q_t, bool reordered, int SG_ROWS, bool GRF256, bool STAGE_A, bool PIPELINE_A,
              bool REGS_A, bool REGS_LITE, bool BLK_A, bool GRID_SLM>
    void launch() const {
        if (arms.split_c) {
            launch_c<block_q_t, reordered, SG_ROWS, GRF256, STAGE_A, PIPELINE_A, REGS_A, REGS_LITE, BLK_A,
                     GRID_SLM, true>();
        } else {
            launch_c<block_q_t, reordered, SG_ROWS, GRF256, STAGE_A, PIPELINE_A, REGS_A, REGS_LITE, BLK_A,
                     GRID_SLM, false>();
        }
    }

    // Where the decoded A halves sit in tile_a. Orthogonal to the three arms below, so it rides
    // along as a second parameter instead of adding an arm of its own.
    template <typename block_q_t, bool reordered, int SG_ROWS, bool GRF256, bool BLK_A, bool GRID_SLM>
    void launch_a_blk() const {
        if (arms.slm_a) {
            launch<block_q_t, reordered, SG_ROWS, GRF256, true, false, false, false, BLK_A, GRID_SLM>();
        } else if (arms.pipeline_a) {
            launch<block_q_t, reordered, SG_ROWS, GRF256, false, true, false, false, BLK_A, GRID_SLM>();
        } else {
            launch<block_q_t, reordered, SG_ROWS, GRF256, false, false, false, false, BLK_A, GRID_SLM>();
        }
    }

    // Whether the format's lookup table is staged. A format without one never instantiates the
    // second kernel, so only iq3_s pays the extra compile.
    template <typename block_q_t, bool reordered, int SG_ROWS, bool GRF256, bool BLK_A>
    void launch_a_grid() const {
        if constexpr (fg_grid_traits<block_q_t>::dwords > 0) {
            if (arms.grid_slm) {
                launch_a_blk<block_q_t, reordered, SG_ROWS, GRF256, BLK_A, true>();
                return;
            }
        }
        launch_a_blk<block_q_t, reordered, SG_ROWS, GRF256, BLK_A, false>();
    }

    // BLK_A and the staged lookup table are the only choices left on the 64-row tile
    template <typename block_q_t, bool reordered, bool BLK_A> void launch_huge() const {
        if constexpr (fg_grid_traits<block_q_t>::dwords > 0) {
            if (arms.grid_slm) {
                launch<block_q_t, reordered, FG_HUGE_ROWS, true, false, false, false, false, BLK_A, true>();
                return;
            }
        }
        launch<block_q_t, reordered, FG_HUGE_ROWS, true, false, false, false, false, BLK_A, false>();
    }

    // the A-stage bits are mutually exclusive, so one tile width is eight kernels, not thirty-two;
    // iq3_s doubles that, because the staged lookup table is offered next to each of them
    template <typename block_q_t, bool reordered, int SG_ROWS, bool GRF256> void launch_a() const {
        if constexpr (reordered) {
            // A stored block held across its k steps was expected not to fit in 128 registers
            // next to the accumulators, so this used to force 256. Measured: it does not spill at
            // 256 (spill counter 0) but the forced 256 halves threads per vector engine, and that
            // cost more than the variant saved (-71% L3 traffic, -5.5% ALU1 instructions, still
            // 10.5% slower). So the register count is now bit 128's choice like everywhere else,
            // and whether it actually spills at 128 is a measurement, not an assumption.
            if (arms.regs_a) {
                launch<block_q_t, reordered, SG_ROWS, GRF256, false, false, true, false, false, false>();
                return;
            }
            // A format whose stored block is one k step has no re-read to remove, so it keeps the
            // plain kernel and the bit is a no-op for it.
            if constexpr (fg_reorder_a<block_q_t>::lite) {
                if (arms.regs_lite_a) {
                    launch<block_q_t, reordered, SG_ROWS, GRF256, false, false, false, true, false, false>();
                    return;
                }
            }
        }
        if (arms.blk_a) {
            launch_a_grid<block_q_t, reordered, SG_ROWS, GRF256, true>();
        } else {
            launch_a_grid<block_q_t, reordered, SG_ROWS, GRF256, false>();
        }
    }

    template <typename block_q_t, bool reordered> void run() const {
        if constexpr (!fg_is_tuned<S>()) {
            launch_c<block_q_t, reordered, S::SG_ROWS, false, false, false, false, false, false, false, false>();
        } else if (arms.mrows == FG_HUGE_ROWS) {
            // 16 accumulator tiles: 256 GRF is not a tuning choice here, it is what makes the
            // kernel fit, so the A image layout is the only choice left on the 64-row tile.
            if (arms.blk_a) {
                launch_huge<block_q_t, reordered, true>();
            } else {
                launch_huge<block_q_t, reordered, false>();
            }
        } else if (arms.mrows == S::SG_ROWS) {
            // bit 128 reaches the narrow tile only through the two regs arms; without either the
            // narrow kernel has no use for 256 registers and keeps the default build.
            if ((arms.regs_a || arms.regs_lite_a) && arms.grf256) {
                launch_a<block_q_t, reordered, S::SG_ROWS, true>();
            } else {
                launch_a<block_q_t, reordered, S::SG_ROWS, false>();
            }
        } else if (arms.grf256) {
            launch_a<block_q_t, reordered, FG_WIDE_ROWS, true>();
        } else {
            launch_a<block_q_t, reordered, FG_WIDE_ROWS, false>();
        }
    }
};

template <typename S>
static bool fg_fused_run(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
                         float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, bool reordered,
                         ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    const int64_t groups_n = (N + S::BN - 1) / S::BN;
    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    const int     Npad     = (int) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * Npad);
    if (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<typename S::EB>((const sycl::half *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    } else if (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<typename S::EB>((const fg_bf16 *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    } else {
        fused_gemm_pack_b<typename S::EB>((const float *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    }

    const typename S::tsb * packed = packed_b.get();
    return fg_visit_type(src0_type, reordered, [&](auto tag) {
        using T = decltype(tag);
        fused_dequant_gemm_launch<S, typename T::type, T::reordered>(src0, packed, dst, (int) M, (int) N, Npad, (int) K,
                                                                     (int) ldd, groups_n, groups_m, stream);
    });
}

bool ggml_sycl_fused_dequant_gemm(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
                                  int32_t src1_prec, float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  bool reordered, ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM. q8_0 and the
    // k-quants take this path too: Qwen3-30B-A3B Q4_K_M on B60 measured pp16 +5%, pp32 +5.5%, pp64 flat
    if (!ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_visit_type(src0_type, reordered, [](auto) {})) {
        return false;
    }
    if (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        return false;
    }
    if (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        return false;
    }
    const int combo = fg_pick_combo(stream, src1_type, src1_prec);
    if (combo < 0) {
        return false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_fused_run<decltype(s)>(src0_type, src0, src1, src1_type, dst, M, N, K, ldd, reordered, pool,
                                             stream);
    });
    return launched;
}

template <typename S>
static bool fg_grouped_run(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                           const float * src1, float * dst, const int64_t * expert_row_offsets, int64_t n_as, int64_t M,
                           int64_t K, int64_t total_rows, int64_t n_active, std::vector<ggml_sycl_gg_tile> & tiles,
                           ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    for (int64_t e = 0; e < n_as; ++e) {
        const int64_t end = expert_row_offsets[e + 1];
        for (int64_t n0 = expert_row_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    int64_t n_tiles = tiles.size();
    // the device schedule's bound and the tight layout are sized for its tile width
    const bool fg_bn = S::BN == GGML_SYCL_FG_BN;
    if (fg_bn && (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_HOST_TABLE)) {
        // host schedule, but launched on the device arm's bound and its Npad: separates the cost
        // of the wider launch from the cost of building the schedule on the device
        const int64_t n_tiles_max = ggml_sycl_grouped_gemm_max_tiles(total_rows, n_as);
        GGML_ASSERT(n_tiles <= n_tiles_max);
        tiles.resize(n_tiles_max, ggml_sycl_gg_tile{ 0, 0, 0 });
        n_tiles = n_tiles_max;
    }
    static std::atomic<int> gg_trace_left{getenv("GGML_SYCL_GG_TRACE") ? 6 : 0};
    if (gg_trace_left.fetch_sub(1) > 0) {
        fprintf(stderr, "[GG] fired type=%s reordered=%d M=%ld K=%ld rows=%ld n_active=%ld tiles=%ld\n",
                ggml_type_name(src0_type), (int) reordered, (long) M, (long) K, (long) total_rows,
                (long) n_active, (long) n_tiles);
    }
    const fg_arms arms     = fg_pick_arms(reordered);
    const int     mrows    = fg_is_tuned<S>() ? arms.mrows : S::SG_ROWS;
    const int64_t groups_m = (M + mrows - 1) / mrows;
    // ONE source for the packed-B column stride: the pack below and the GEMM must agree exactly
    const bool    tight    = fg_bn && (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_PACKB_TIGHT) != 0;
    const int     Npad     = (int) (tight ? ggml_sycl_grouped_gemm_tight_npad(total_rows) : n_tiles * S::BN);

    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, n_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(tiles_dev.get(), tiles.data(), n_tiles * sizeof(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, grouped_gemm_packed_capacity((size_t) K * Npad));
    const ggml_sycl_gg_rows rows = { (char *) src1, nullptr, 0, (size_t) K * sizeof(float), 0 };
    grouped_gemm_pack_b<S>(rows, packed_b.get(), tiles_dev.get(), Npad, (int) K, (int) total_rows, tight,
                           grouped_gemm_pack_b_wide_ok(rows.base, rows.nb1, 0, (int) K), stream);

    const grouped_gemm_launcher<S> launcher{ (const char *) src0_base, expert_stride, tiles_dev.get(), packed_b.get(),
                                             dst, (int) M, Npad, (int) K, tight, ggml_sycl_gg_rows{}, arms, n_tiles,
                                             groups_m, stream };
    return fg_visit_type(src0_type, reordered, [&](auto tag) {
        using T = decltype(tag);
        launcher.template run<typename T::type, T::reordered>();
    });
}

bool ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, const void * src0_base, size_t expert_stride,
                                    const float * src1, int32_t src1_prec, float * dst,
                                    const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t total_rows, bool reordered, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    int64_t n_active = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
    }
    if (!ggml_sycl_xmx_gather_type_enabled(src0_type)) {
        return false;
    }
    if (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, total_rows)) {
        return false;
    }
    // HOST_TABLE makes the host arm decide exactly as the device arm does, so a whole-op A/B
    // compares the same algorithm. Without it the two gates disagree for a slice-heavy node and
    // the host falls back to the per-expert library GEMM, which is a different computation.
    const int64_t n_active_gate = (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_HOST_TABLE) ?
                                      std::min(n_as, total_rows) : n_active;
    if (!ggml_sycl_grouped_dequant_gemm_width_ok(total_rows, n_active_gate)) {
        return false;
    }
    if (reordered && !ggml_sycl_fused_dequant_gemm_reorder_ok(src0_type)) {
        return false;
    }
    // the SoA offsets are derived from the slice block count, so a slice must be exactly that
    GGML_ASSERT(!reordered ||
                expert_stride == (size_t) M * (K / ggml_blck_size(src0_type)) * ggml_type_size(src0_type));
    const int combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    if (combo < 0) {
        return false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_grouped_run<decltype(s)>(src0_type, reordered, src0_base, expert_stride, src1, dst,
                                               expert_row_offsets, n_as, M, K, total_rows, n_active, tiles, pool,
                                               stream);
    });
    return launched;
}

// One work-group builds the whole MUL_MAT_ID schedule: counting sort of the routed rows by
// expert id, then the row mapping and the tile table. Single work-group so the three phases can
// be separated by group barriers instead of by separate dispatches.
static constexpr int MMID_SCHED_WG = 256;

using mmid_sched_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                           sycl::memory_scope::work_group,
                                           sycl::access::address_space::local_space>;

bool ggml_sycl_build_mmid_schedule(const int32_t * ids_dev, size_t ids_token_stride, int64_t n_as, int64_t n_ids,
                                   int64_t n_tokens, int64_t n_tiles_max, mmid_row_mapping * row_mapping,
                                   ggml_sycl_gg_tile * tiles, uint32_t * expert_offsets, dpct::queue_ptr stream) {
    if (n_as <= 0 || n_as > GGML_SYCL_MMID_SCHED_MAX_EXPERTS || n_ids <= 0 || n_tokens <= 0) {
        return false;
    }
    const int64_t n_routes = n_tokens * n_ids;
    if (n_routes > INT32_MAX || n_tiles_max <= 0) {
        return false;
    }

    stream->submit([&](sycl::handler & cgh) {
        // l_cnt is reused: route counts, then the write cursors, then the tile count per expert
        sycl::local_accessor<uint32_t, 1> l_cnt(n_as + 1, cgh);
        sycl::local_accessor<uint32_t, 1> l_off(n_as + 1, cgh);
        sycl::local_accessor<uint32_t, 1> l_toff(n_as + 1, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(MMID_SCHED_WG, MMID_SCHED_WG),
            [=](sycl::nd_item<1> item) {
                const auto     grp = item.get_group();
                const int64_t  tid = item.get_local_id(0);
                uint32_t * cnt  = l_cnt.get_multi_ptr<sycl::access::decorated::no>().get();
                uint32_t * off  = l_off.get_multi_ptr<sycl::access::decorated::no>().get();
                uint32_t * toff = l_toff.get_multi_ptr<sycl::access::decorated::no>().get();

                for (int64_t e = tid; e < n_as; e += MMID_SCHED_WG) {
                    cnt[e] = 0;
                }
                // a dropped route must leave a row that names (0, 0) rather than stale memory
                for (int64_t r = tid; r < n_routes; r += MMID_SCHED_WG) {
                    row_mapping[r] = { 0, 0 };
                }
                sycl::group_barrier(grp);

                for (int64_t r = tid; r < n_routes; r += MMID_SCHED_WG) {
                    const int64_t   token  = r / n_ids;
                    const int64_t   slot   = r - token * n_ids;
                    const int32_t * id_row = (const int32_t *) ((const char *) ids_dev + token * ids_token_stride);
                    const int32_t   e      = id_row[slot];
                    if (e >= 0 && e < n_as) {
                        mmid_sched_atomic(cnt[e]).fetch_add(1u);
                    }
                }
                sycl::group_barrier(grp);

                sycl::joint_exclusive_scan(grp, cnt, cnt + n_as, off, 0u, sycl::plus<uint32_t>());
                sycl::group_barrier(grp);
                if (tid == 0) {
                    off[n_as] = off[n_as - 1] + cnt[n_as - 1];
                }
                sycl::group_barrier(grp);

                for (int64_t e = tid; e < n_as; e += MMID_SCHED_WG) {
                    cnt[e] = off[e];
                }
                sycl::group_barrier(grp);

                // within one expert the order is atomic arrival order, not the host's stable
                // (token, slot); the slice contents are the same set of rows either way
                for (int64_t r = tid; r < n_routes; r += MMID_SCHED_WG) {
                    const int64_t   token  = r / n_ids;
                    const int64_t   slot   = r - token * n_ids;
                    const int32_t * id_row = (const int32_t *) ((const char *) ids_dev + token * ids_token_stride);
                    const int32_t   e      = id_row[slot];
                    if (e >= 0 && e < n_as) {
                        const uint32_t pos = mmid_sched_atomic(cnt[e]).fetch_add(1u);
                        row_mapping[pos]   = { (int32_t) slot, (int32_t) token };
                    }
                }
                sycl::group_barrier(grp);

                for (int64_t e = tid; e < n_as; e += MMID_SCHED_WG) {
                    cnt[e] = (off[e + 1] - off[e] + GGML_SYCL_FG_BN - 1) / GGML_SYCL_FG_BN;
                }
                sycl::group_barrier(grp);

                sycl::joint_exclusive_scan(grp, cnt, cnt + n_as, toff, 0u, sycl::plus<uint32_t>());
                sycl::group_barrier(grp);
                if (tid == 0) {
                    toff[n_as] = toff[n_as - 1] + cnt[n_as - 1];
                }
                sycl::group_barrier(grp);

                const uint32_t n_tiles = toff[n_as];
                for (int64_t e = tid; e < n_as; e += MMID_SCHED_WG) {
                    const uint32_t base = off[e];
                    const uint32_t end  = off[e + 1];
                    for (uint32_t j = 0; j < cnt[e]; ++j) {
                        const uint32_t n0 = base + j * GGML_SYCL_FG_BN;
                        const uint32_t n1 = n0 + GGML_SYCL_FG_BN < end ? n0 + GGML_SYCL_FG_BN : end;
                        tiles[toff[e] + j] = { (int32_t) e, (int32_t) n0, (int32_t) n1 };
                    }
                }
                // the launch is bounded, so the unused tail must be tiles the GEMM skips
                for (int64_t t = n_tiles + tid; t < n_tiles_max; t += MMID_SCHED_WG) {
                    tiles[t] = { 0, 0, 0 };
                }
                for (int64_t e = tid; e <= n_as; e += MMID_SCHED_WG) {
                    expert_offsets[e] = off[e];
                }
            });
    });
    return true;
}

bool ggml_sycl_grouped_dequant_gemm_dev_ok(ggml_type src0_type, int64_t M, int64_t K, int64_t total_rows,
                                           int64_t n_as, bool reordered, int32_t src1_prec, dpct::queue_ptr stream) {
    if (!ggml_sycl_xmx_gather_type_enabled(src0_type)) {
        return false;
    }
    if (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, total_rows)) {
        return false;
    }
    if (!(g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_UNBOUNDED)) {
        const int64_t n_active_max = std::min(n_as, total_rows);
        if (!ggml_sycl_grouped_dequant_gemm_width_ok(total_rows, n_active_max)) {
            return false;
        }
    }
    if (reordered && !ggml_sycl_fused_dequant_gemm_reorder_ok(src0_type)) {
        return false;
    }
    if (n_as <= 0 || n_as > GGML_SYCL_MMID_SCHED_MAX_EXPERTS) {
        return false;
    }
    // the device builds its tiles GGML_SYCL_FG_BN wide
    return fg_pick_combo(stream, GGML_TYPE_F32, src1_prec, GGML_SYCL_FG_BN) >= 0;
}

template <typename S>
static bool fg_grouped_dev_run(ggml_type src0_type, const void * src0_base, size_t expert_stride,
                               const ggml_sycl_gg_rows & src1, const ggml_sycl_gg_rows & dst,
                               const ggml_sycl_gg_tile * tiles_dev, int64_t n_tiles_max, int64_t M, int64_t K,
                               int64_t total_rows, bool reordered, ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    static_assert(S::BN == GGML_SYCL_FG_BN, "the device schedule builds GGML_SYCL_FG_BN-wide tiles");
    const fg_arms arms     = fg_pick_arms(reordered);
    const int     mrows    = fg_is_tuned<S>() ? arms.mrows : S::SG_ROWS;
    const int64_t groups_m = (M + mrows - 1) / mrows;
    // ONE source for the packed-B column stride: the pack below and the GEMM must agree exactly.
    // The tight layout is what makes this buffer independent of the tile count: it is sized by the
    // routed rows, which the host knows without the schedule.
    const bool    tight    = (g_ggml_sycl_mmid_sched & GGML_SYCL_MMID_SCHED_PACKB_TIGHT) != 0;
    const int     Npad     = (int) (tight ? ggml_sycl_grouped_gemm_tight_npad(total_rows) : n_tiles_max * S::BN);

    // Npad depends only on the routed row count here, so the exact size is as stable as the rounded one
    const size_t packed_b_size = (g_ggml_sycl_mem_save & GGML_SYCL_MEM_SAVE_PACKB_EXACT) ?
                                     (size_t) K * Npad : grouped_gemm_packed_capacity((size_t) K * Npad);
    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, packed_b_size);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev, Npad, (int) K, (int) total_rows, tight,
                           grouped_gemm_pack_b_wide_ok(src1.base, src1.nb1, src1.nb2, (int) K), stream);

    const grouped_gemm_launcher<S> launcher{ (const char *) src0_base, expert_stride, tiles_dev, packed_b.get(),
                                             nullptr, (int) M, Npad, (int) K, tight, dst, arms, n_tiles_max,
                                             groups_m, stream };
    return fg_visit_type(src0_type, reordered, [&](auto tag) {
        using T = decltype(tag);
        launcher.template run<typename T::type, T::reordered>();
    });
}

bool ggml_sycl_grouped_dequant_gemm_dev(ggml_type src0_type, const void * src0_base, size_t expert_stride,
                                        const ggml_sycl_gg_rows & src1, int32_t src1_prec, const ggml_sycl_gg_rows & dst,
                                        const ggml_sycl_gg_tile * tiles_dev,
                                        int64_t n_tiles_max, int64_t M, int64_t K, int64_t total_rows,
                                        bool reordered, ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // the SoA offsets are derived from the slice block count, so a slice must be exactly that
    GGML_ASSERT(!reordered ||
                expert_stride == (size_t) M * (K / ggml_blck_size(src0_type)) * ggml_type_size(src0_type));

    static std::atomic<int> gg_dev_trace_left{ getenv("GGML_SYCL_GG_TRACE") ? 6 : 0 };
    if (gg_dev_trace_left.fetch_sub(1) > 0) {
        fprintf(stderr, "[GG-DEV] fired type=%s reordered=%d M=%ld K=%ld tiles_max=%ld\n",
                ggml_type_name(src0_type), (int) reordered, (long) M, (long) K, (long) n_tiles_max);
    }

    const int combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec, GGML_SYCL_FG_BN);
    if (combo < 0) {
        return false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        using S = decltype(s);
        if constexpr (S::BN == GGML_SYCL_FG_BN) {
            launched = fg_grouped_dev_run<S>(src0_type, src0_base, expert_stride, src1, dst, tiles_dev, n_tiles_max,
                                             M, K, total_rows, reordered, pool, stream);
        }
    });
    return launched;
}

// split-K f32 GEMM for small M*N and long K: a work-item owns one output and one K slice,
// the KSPLIT partials of an output are summed through the sub-group and SLM
static constexpr int SK_WG_SIZE      = 256;
static constexpr int SK_VEC          = 4;
static constexpr int SK_WIDE_TILE_MN = 16;
static constexpr int SK_MID_TILE_MN  = 4;
static constexpr int SK_TINY_TILE_MN = 1;
static constexpr int SK_MIN_GROUPS   = 32;
static constexpr int64_t SK_MAX_MNK  = GGML_SYCL_SK_MAX_MNK;

template <int TILE_MN, int VEC>
static void small_gemm_f32(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ dst,
                           const int M, const int N, const int K, const int lda, const int ldd,
                           sycl::local_accessor<float, 1> part, const sycl::nd_item<1> & item) {
    constexpr int KSPLIT = SK_WG_SIZE / TILE_MN;
    static_assert(KSPLIT % WARP_SIZE == 0, "an output must own whole sub-groups");
    constexpr int NSG = KSPLIT / WARP_SIZE;

    const int  lid = item.get_local_id(0);
    const int  t   = lid / KSPLIT;
    const int  ks  = lid - t * KSPLIT;
    const int  o   = item.get_group(0) * TILE_MN + t;
    const bool ok  = o < M * N;
    const int  m   = ok ? o % M : 0;
    const int  n   = ok ? o / M : 0;

    float acc = 0.0f;
    if (ok) {
        const float * arow = a + (size_t) m * lda;
        const float * brow = b + (size_t) n * K;
        // the k slice index runs fastest, so a sub-group reads a contiguous run of both rows
        for (int k = ks * VEC; k < K; k += KSPLIT * VEC) {
            if constexpr (VEC == 4) {
                const sycl::float4 av = *(const sycl::float4 *) (arow + k);
                const sycl::float4 bv = *(const sycl::float4 *) (brow + k);
                acc += av.x() * bv.x() + av.y() * bv.y() + av.z() * bv.z() + av.w() * bv.w();
            } else {
                acc += arow[k] * brow[k];
            }
        }
    }

    acc = sycl::reduce_over_group(item.get_sub_group(), acc, std::plus<>());
    if (ks % WARP_SIZE == 0) {
        part[t * NSG + ks / WARP_SIZE] = acc;
    }
    sycl::group_barrier(item.get_group());

    if (ok && ks == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < NSG; ++s) {
            sum += part[t * NSG + s];
        }
        dst[(size_t) n * ldd + m] = sum;
    }
}

template <int TILE_MN, int VEC>
static void small_gemm_f32_launch(const float * a, const float * b, float * dst, int M, int N, int K, int lda, int ldd,
                                  dpct::queue_ptr stream) {
    const int64_t groups = ((int64_t) M * N + TILE_MN - 1) / TILE_MN;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> part(SK_WG_SIZE / WARP_SIZE, cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * SK_WG_SIZE), sycl::range<1>(SK_WG_SIZE)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             small_gemm_f32<TILE_MN, VEC>(a, b, dst, M, N, K, lda, ldd, part, item);
                         });
    });
}

template <int TILE_MN>
static void small_gemm_f32_launch(const float * a, const float * b, float * dst, int M, int N, int K, int lda, int ldd,
                                  bool vec, dpct::queue_ptr stream) {
    if (vec) {
        small_gemm_f32_launch<TILE_MN, SK_VEC>(a, b, dst, M, N, K, lda, ldd, stream);
    } else {
        small_gemm_f32_launch<TILE_MN, 1>(a, b, dst, M, N, K, lda, ldd, stream);
    }
}

bool ggml_sycl_small_gemm_f32_device_ok(dpct::queue_ptr stream) {
    static const bool ok =
        stream->get_device().get_info<sycl::info::device::max_work_group_size>() >= (size_t) SK_WG_SIZE;
    return ok;
}

bool ggml_sycl_small_gemm_f32(const float * a, const float * b, float * dst, int64_t M, int64_t N, int64_t K,
                              int64_t lda, int64_t ldd, dpct::queue_ptr stream) {
    if (!ggml_sycl_small_gemm_f32_shape_ok(M, N, K, lda, ldd)) {
        return false;
    }
    if (!ggml_sycl_small_gemm_f32_device_ok(stream)) {
        return false;
    }

    const int64_t mn  = M * N;
    const bool    vec = K % SK_VEC == 0 && lda % SK_VEC == 0 &&
                        ((uintptr_t) a | (uintptr_t) b) % (SK_VEC * sizeof(float)) == 0;

    // spread the outputs over enough work-groups; when M*N is too small for that, split K deeper instead
    if (mn >= SK_MIN_GROUPS * SK_WIDE_TILE_MN) {
        small_gemm_f32_launch<SK_WIDE_TILE_MN>(a, b, dst, (int) M, (int) N, (int) K, (int) lda, (int) ldd, vec, stream);
    } else if (mn >= SK_MIN_GROUPS * SK_MID_TILE_MN) {
        small_gemm_f32_launch<SK_MID_TILE_MN>(a, b, dst, (int) M, (int) N, (int) K, (int) lda, (int) ldd, vec, stream);
    } else {
        small_gemm_f32_launch<SK_TINY_TILE_MN>(a, b, dst, (int) M, (int) N, (int) K, (int) lda, (int) ldd, vec, stream);
    }
    return true;
}
