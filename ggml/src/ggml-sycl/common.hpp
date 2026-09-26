//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_COMMON_HPP
#define GGML_SYCL_COMMON_HPP

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "base.hpp"
#include "dpct/helper.hpp"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-sycl.h"
#include "presets.hpp"
#include "type.hpp"
#include "sycl_hw.hpp"
#include "fattn-buffers.hpp"
#include "memtrace.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;

#if defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#if GGML_SYCL_DNNL
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#define GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "ggml-common.h"
#pragma clang diagnostic pop
#include "ggml-impl.h"

void* ggml_sycl_host_malloc(size_t size);
void ggml_sycl_host_free(void* ptr);


extern int g_ggml_sycl_debug;
extern int g_ggml_sycl_enable_optimize;
extern int g_ggml_sycl_enable_fusion;
extern int g_ggml_sycl_enable_dnn;
// GGML_SYCL_ENABLE_ESIMD. Any nonzero value turns the ESIMD kernels on, exactly as when it was a
// boolean; bit 1 additionally lets the reordered q8_0 / q6_K mat-vec kernels take 2..8 (q8_0: 2..4) activation
// columns (MTP verification, small-batch decode) instead of the quantize + MMVQ path. On by default
// (GGML_SYCL_ENABLE_ESIMD=3); =1 keeps single-column ESIMD only, =0 turns ESIMD off.
enum ggml_sycl_esimd_bit {
    GGML_SYCL_ESIMD_ON    = 1 << 0,
    GGML_SYCL_ESIMD_NCOLS = 1 << 1,
};
// q6_K keeps winning over MMVQ up to 8 columns (1.85x at 8 on the LM head); q8_0 stops at 4: at 8
// columns the per-row-pair f32 activation re-reads outweigh the shared weight load (0.87-0.89x).
static constexpr int GGML_SYCL_ESIMD_MAX_NCOLS      = 8;
static constexpr int GGML_SYCL_ESIMD_MAX_NCOLS_Q8_0 = 4;
extern int g_ggml_sycl_enable_esimd;
extern int g_ggml_sycl_esimd_q8_0;
extern int g_ggml_sycl_prioritize_dmmv;
extern int g_ggml_sycl_moe_reorder;
extern int g_ggml_sycl_moe_xmx;
extern int g_ggml_sycl_fused_gemm;
extern int g_ggml_sycl_grouped_gemm;
extern int g_ggml_sycl_mmvq_wide;
extern int g_ggml_sycl_fuse_cast_add;
extern int g_ggml_sycl_fuse_cont_add;
extern int g_ggml_sycl_fuse_qsa_gather;
extern int g_ggml_sycl_fuse_qsa_topk;
extern int g_ggml_sycl_fuse_qsa_score;
extern int g_ggml_sycl_fuse_qsa_mask;
extern int g_ggml_sycl_fuse_qsa_fa_mask;
extern int g_ggml_sycl_small_gemm;
extern int g_ggml_sycl_mv_fuse;
extern int g_ggml_sycl_topk_moe_radix;
// Which quant types may have their MoE expert weights reordered into the per-expert SoA
// layout. One bit per type so a new type is one bit, not another environment variable.
enum ggml_sycl_reorder_type {
    GGML_SYCL_REORDER_IQ3_S  = 1 << 0,
    GGML_SYCL_REORDER_IQ4_NL = 1 << 1,
    GGML_SYCL_REORDER_Q8_0   = 1 << 2,
};

// Everything except Q8_0: its MoE mat-vec is about 4% of decode and its quants are already
// contiguous within a block, so the reorder measured flat. A new bit is on by default.
static constexpr int GGML_SYCL_REORDER_DEFAULT = ~GGML_SYCL_REORDER_Q8_0;

extern int g_ggml_sycl_reorder_types;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
enum ggml_sycl_xmx_gather_type {
    GGML_SYCL_XMX_GATHER_IQ4_NL   = 1 << 0,
    GGML_SYCL_XMX_GATHER_IQ3_S    = 1 << 1,
    GGML_SYCL_XMX_GATHER_IQ4_XS   = 1 << 2,
    GGML_SYCL_XMX_GATHER_IQ3_XXS  = 1 << 3,
    GGML_SYCL_XMX_GATHER_IQ2_XXS  = 1 << 4,
    GGML_SYCL_XMX_GATHER_IQ2_XS   = 1 << 5,
    GGML_SYCL_XMX_GATHER_IQ2_S    = 1 << 6,
    GGML_SYCL_XMX_GATHER_IQ1_S    = 1 << 7,
    GGML_SYCL_XMX_GATHER_IQ1_M    = 1 << 8,
    GGML_SYCL_XMX_GATHER_Q8_0     = 1 << 9,
    GGML_SYCL_XMX_GATHER_Q4_K     = 1 << 10,
    GGML_SYCL_XMX_GATHER_Q5_K     = 1 << 11,
    GGML_SYCL_XMX_GATHER_Q6_K     = 1 << 12,
};
static constexpr int GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
extern int g_ggml_sycl_xmx_gather_types;

// Opt-in to paths that are faster but give up accuracy in edge cases the library GEMM handles.
// Currently gates q4_K on the XMX gather: the A stage carries the dequantized weight as f16, so a
// weight whose magnitude exceeds the f16 range (65504) becomes inf and the product NaN. Real
// weights are nowhere near that; the synthetic amax=100000 case in test-backend-ops is.
extern int g_ggml_sycl_fast_and_sloppy;

// MUL_MAT_ID tile scheduling. The host path reads the routing back and sorts it on the CPU, which
// costs a full queue drain per node; these bits select a device-built schedule instead and the
// checks that prove the two agree. Off by default: the host path stays the correctness oracle.
enum ggml_sycl_mmid_sched_bit {
    GGML_SYCL_MMID_SCHED_DEVICE    = 1 << 0,  // build the schedule on the device, no drain
    GGML_SYCL_MMID_SCHED_UNBOUNDED = 1 << 1,  // skip the slice-width heuristic the device cannot ask
    GGML_SYCL_MMID_SCHED_VERIFY    = 1 << 2,  // read the device schedule back and check it
    GGML_SYCL_MMID_SCHED_VERIFY_DST = 1 << 3, // run both arms and compare the output bitwise
    GGML_SYCL_MMID_SCHED_HOST_TABLE = 1 << 4, // host schedule, but padded to the device arm's bound
    GGML_SYCL_MMID_SCHED_PACKB_TIGHT = 1 << 5, // pack B at the tile's own row base, not at t * FG_BN
    GGML_SYCL_MMID_SCHED_MTILE32     = 1 << 6, // 32 rows per work-group, not 16: half the packed-B re-reads
    GGML_SYCL_MMID_SCHED_GRF256      = 1 << 7, // 256 GRF on the wide M tile, which holds 8 accumulator tiles
    GGML_SYCL_MMID_SCHED_MTILE64     = 1 << 8, // 64 rows per work-group; always 256 GRF, and wins over bit 64
    GGML_SYCL_MMID_SCHED_SLM_A       = 1 << 9, // stage the raw quantized A bytes of a stored block through SLM
    GGML_SYCL_MMID_SCHED_PIPELINE_A  = 1 << 10, // stage the A of k step n+1 while the MADs of step n run
    GGML_SYCL_MMID_SCHED_REGS_A      = 1 << 11, // hold one stored A block in registers over all of its k steps
    GGML_SYCL_MMID_SCHED_REGS_LITE_A = 1 << 12, // hold only the cheap, high-reuse streams of the block in registers
    GGML_SYCL_MMID_SCHED_BLK_A       = 1 << 15, // lay the staged A tile out as whole 8x16 matrix tiles, not row major
    GGML_SYCL_MMID_SCHED_GRID_SLM    = 1 << 16, // hold the iq3_s lookup table in SLM, one copy per work-group
    GGML_SYCL_MMID_SCHED_SPLIT_C     = 1 << 17, // reduce the K-split partials 8 rows at a time: half the tile_c
    GGML_SYCL_MMID_SCHED_PACKB_NARROW = 1 << 18, // turn OFF the wide B pack: one k pair per work-item, as before
    // DIAGNOSTIC ONLY - these produce WRONG OUTPUT. They exist to time half the kernel: with the
    // execution units idle ~92% and no traffic, spill, instruction-count or barrier explanation
    // left, the question is whether the time is in the A dequant or in the XMX MADs.
    GGML_SYCL_MMID_SCHED_DIAG_NO_MAD = 1 << 13, // decode A, skip the MADs
    GGML_SYCL_MMID_SCHED_DIAG_NO_A   = 1 << 14, // skip the A decode, run the MADs on stale tile_a
};
extern int g_ggml_sycl_mmid_sched;

// USM system allocations. Bit 0 is what the flag meant as a boolean: back a large SYCL buffer with
// the system allocator instead of device memory. Bit 1 lets the device read a weight that llama.cpp
// keeps mapped in host memory, so a gather of that weight runs here and only the row indices cross
// the bus, instead of the host gathering and staging the dense result. Both need a device that
// reports usm_system_allocations.
enum ggml_sycl_usm_system_bit {
    GGML_SYCL_USM_SYSTEM_ALLOC          = 1 << 0, // large SYCL buffers come from the system allocator
    GGML_SYCL_USM_SYSTEM_MAPPED_WEIGHTS = 1 << 1, // gather a host-mapped weight on the device
};
extern int g_ggml_sycl_usm_system;

// Ways to lower the peak device memory of a process. All on by default (7); GGML_SYCL_MEM_SAVE=0
// turns them off. POOL_RELEASE does nothing while SYCL graphs are enabled.
enum ggml_sycl_mem_save_bit {
    GGML_SYCL_MEM_SAVE_REORDER_CHUNK = 1 << 0, // reorder large weights in chunks, with a small temp buffer
    GGML_SYCL_MEM_SAVE_POOL_RELEASE  = 1 << 1, // before the pool grows, it frees cached buffers smaller than the request
    GGML_SYCL_MEM_SAVE_PACKB_EXACT   = 1 << 2, // device-scheduled grouped GEMM: packed B at its size, not the next power of 2
};
static constexpr int GGML_SYCL_MEM_SAVE_DEFAULT =
    GGML_SYCL_MEM_SAVE_REORDER_CHUNK | GGML_SYCL_MEM_SAVE_POOL_RELEASE | GGML_SYCL_MEM_SAVE_PACKB_EXACT;
extern int g_ggml_sycl_mem_save;
// the most a chunked reorder copies into its temp buffer at once; GGML_SYCL_REORDER_CHUNK_KIB, for tests
extern size_t g_ggml_sycl_reorder_chunk_bytes;

extern int g_ggml_sycl_enable_flash_attention;
extern int g_ggml_sycl_dev2dev_memcpy;
// Wait for a cross-split event by enqueuing a barrier instead of blocking the host on it.
extern int g_ggml_sycl_device_event_wait;
// Copy into a SYCL backend by enqueuing, instead of draining both devices on the host.
extern int g_ggml_sycl_async_copy;
extern int g_ggml_sycl_copy_ring_depth;
// Which of the graph-level fusions may fire. One bit per fusion so a new one is a bit rather
// than another environment variable, and so a bisect over them is a single value.
enum ggml_sycl_fuse_type {
    GGML_SYCL_FUSE_ELEMENTWISE = 1 << 0,  // hyper-connection gate chain, scale+unary
    GGML_SYCL_FUSE_MUL_ADD     = 1 << 1,  // multiply-accumulate pair
    GGML_SYCL_FUSE_MOE_REDUCE  = 1 << 2,  // MoE weighted sum: mul + per-expert views + add chain
    GGML_SYCL_FUSE_MOE_GLU_ID  = 1 << 3,  // gate + up MoE mat-vec folded with their GLU
    GGML_SYCL_FUSE_UNARY_MUL_B = 1 << 4,  // unary + mul where the unary side is one value per row
    GGML_SYCL_FUSE_NORM_SCALE  = 1 << 5,  // rms_norm + the scale that turns it into an l2 norm
    GGML_SYCL_FUSE_GLU_NCOLS   = 1 << 6,  // dense q8_0 gate + up + SWIGLU also at 3..8 columns, not only 1..2
    GGML_SYCL_FUSE_FLAT_BATCH  = 1 << 7,  // 2D quantized weight x contiguous 3D/4D activation: one launch, not one per batch
};

// The fusions that are on by default. Every fusion has its own bit, so GGML_SYCL_FUSE_TYPES can
// switch any of them on or off alone.
static constexpr int GGML_SYCL_FUSE_DEFAULT =
    GGML_SYCL_FUSE_ELEMENTWISE | GGML_SYCL_FUSE_MUL_ADD | GGML_SYCL_FUSE_MOE_REDUCE | GGML_SYCL_FUSE_MOE_GLU_ID |
    GGML_SYCL_FUSE_NORM_SCALE | GGML_SYCL_FUSE_GLU_NCOLS | GGML_SYCL_FUSE_FLAT_BATCH;   // = 239; UNARY_MUL_B (bit 4) stays off: measured as noise
static constexpr int GGML_SYCL_ESIMD_DEFAULT = 3;  // GGML_SYCL_ESIMD_ON | GGML_SYCL_ESIMD_NCOLS

extern int g_ggml_sycl_fuse_types;
// Allow a fusion to rely on a+b == b+a, which is exact in IEEE754. Not associativity.
extern int g_ggml_sycl_float_commutative;
// Store the q8_0 KV cache per-row SoA so its quants load 16-byte aligned. See kv-soa.hpp.
extern int g_ggml_sycl_kv_soa;
// Pack the causal mask to one bit per cell in device memory. The tensor keeps its f16 type and
// shape, so ggml and supports_op never see a difference; only the bytes change, and only the
// backend reads them. Cuts what a flash-attention kernel re-reads per cell from 16 bits to 1.
extern int g_ggml_sycl_kq_mask_bits;
// Wide loads that a kernel takes only after a runtime check of its pointers and strides; the
// narrow path stays for everything else. One bit per site, all on by default: clear a bit to get
// the narrow path back. (The grouped GEMM B pack is GGML_SYCL_MMID_SCHED_PACKB_NARROW instead.)
enum ggml_sycl_wide_load_bit {
    GGML_SYCL_WIDE_HC      = 1 << 0,  // dsv4_hc_pre / dsv4_hc_post: float4 rows, no 64-bit index division
    GGML_SYCL_WIDE_GDN     = 1 << 1,  // gated_delta_net: block loads of the k/q rows, next token loaded ahead
    GGML_SYCL_WIDE_CONVERT = 1 << 2,  // f32 -> f16 rows and reordered q8_0 -> f16/f32 dequant: 16/32 B loads and stores
};
static constexpr int GGML_SYCL_WIDE_LOADS_DEFAULT = ~0;
extern int g_ggml_sycl_wide_loads;
// ggml_can_fuse_subgraph() takes at most 31 nodes, and the span is 2*n_expert_used.
static constexpr int GGML_SYCL_MOE_REDUCE_MAX_EXPERTS = 15;
extern int g_ggml_sycl_fa_onednn;
extern int g_ggml_sycl_fa_onednn_max_kv;
extern int g_ggml_sycl_enable_mkl_fa;
extern int g_ggml_sycl_fa_max_mem_mib;
extern int g_ggml_sycl_memtrace;
extern int g_ggml_sycl_memtrace_step;


#define CHECK_TRY_ERROR(expr)                                            \
  [&]() {                                                                \
    try {                                                                \
      expr;                                                              \
      return dpct::success;                                              \
    } catch (std::exception const& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      return dpct::default_error;                                        \
    }                                                                    \
  }()


#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC 610 // todo for hardware optimize.
#define VER_GEN9 700 // todo for hardware optimize.
#define VER_GEN12 1000000 // todo for hardware optimize.
#define VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#define GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#if !defined(GGML_SYCL_FORCE_MMQ)
    #define SYCL_USE_XMX
#endif

// max batch size to use MMQ kernels when tensor cores are available
#define MMQ_MAX_BATCH_SIZE 32

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#define GGML_SYCL_MMV_Y 1
#endif

typedef sycl::queue *queue_ptr;

enum ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

// GGML_SYCL_ASYNC_COPY is a bitset, not a bool. Bit 0 is the original behaviour, so the
// historical value 1 still means exactly what it used to.
enum ggml_sycl_async_copy_bits {
    // Order a cross-device peer copy with a barrier on the destination queue instead of
    // draining both devices on the host. Requires ext_oneapi_can_access_peer.
    GGML_SYCL_ASYNC_COPY_PEER = 1 << 0,
    // L0_ASYNC_COPY: append the Level Zero device-to-device copy to an ASYNCHRONOUS immediate
    // command list and order the destination SYCL queue behind it with a host-visible L0 event,
    // instead of blocking the host inside zeCommandListAppendMemoryCopy. Off by default: it is
    // unproven, and measures as noise in layer-split mode where cross-device traffic is ~10 KB
    // per token. Enable with GGML_SYCL_ASYNC_COPY=3 together with GGML_SYCL_DEV2DEV_MEMCPY=1.
    GGML_SYCL_ASYNC_COPY_L0 = 1 << 1,
    // SRC_RING: stage a cross-device copy through a small ring of buffers on the SOURCE device.
    // The source queue copies into the ring locally and moves on; the destination pulls from the
    // ring. Without it the source queue must wait until the destination has read the tensor,
    // because in pipeline mode the source's next ubatch overwrites that same compute buffer. Clear
    // the bit to trade that overlap for memory: the ring holds DEPTH copies of the largest tensor
    // crossing from each device (~86 MiB per card measured for qwen4exp at ubatch 1024).
    GGML_SYCL_ASYNC_COPY_SRC_RING = 1 << 2,
};
static constexpr int GGML_SYCL_ASYNC_COPY_DEFAULT = GGML_SYCL_ASYNC_COPY_PEER | GGML_SYCL_ASYNC_COPY_SRC_RING;
// Ring slots per source device (GGML_SYCL_COPY_RING_DEPTH, clamped to 1..MAX). Slot k is reused
// DEPTH copies later, and only then does the source queue wait, device-side, for the destination
// to have finished reading it. Each slot costs the largest tensor copied from that device.
static constexpr int GGML_SYCL_COPY_RING_DEPTH_DEFAULT = 2;
static constexpr int GGML_SYCL_COPY_RING_MAX_DEPTH     = 8;

// intel/compute-runtime issue #995: a peer-to-peer copy hangs the GPU engine (10 s timeout,
// "Fault response: Unsuccessful -ENOENT", "exec queue reset detected") when the remote range
// crosses from one zeVirtualMemMap()'d physical allocation into the next inside a single
// zeVirtualMemReserve() range, for copies larger than the 2 MiB page. Local access to the same
// memory is fine. PR #996 fixes it by splitting the copy at block ends, but it is still open, so
// no shipping driver has it. We never split, so decline peer/L0 routes in the exposed case.
static constexpr size_t GGML_SYCL_P2P_BLOCK_SPLIT_BYTES = 2u * 1024u * 1024u;

enum ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void crash() {
  int* ptr = NULL;
  *ptr = 0;
}

[[noreturn]] static void ggml_sycl_error(
    const char* stmt,
    const char* func,
    const char* file,
    const int line,
    const char* msg) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    do {                                                                                                   \
        auto err_ = (err);                                                                                 \
        if (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } while (0)

#if DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#else
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half dfloat; // dequantize float
typedef sycl::half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

static int g_all_sycl_device_count = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

static void* g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

[[noreturn]] static inline void bad_arch(const sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (void)bad_arch; // suppress unused function warning
}

int get_current_device_id();

inline int ggml_sycl_get_device() {
    return get_current_device_id();
}

inline dpct::err0 ggml_sycl_set_device(const int device) try {
  int current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  if (device == current_device_id) {
    return 0;
  }

  return CHECK_TRY_ERROR(dpct::select_device(device));
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
// How a tensor's bytes are actually packed, when that differs from the canonical ggml packing
// for its ggml_type. This is backend-local on purpose: a new ggml_type would be a global enum
// entry that every backend must implement or reject, would need a CPU reference for
// test-backend-ops, and would leak into GGUF and session files. A descriptor on the tensor costs
// nothing outside this backend, and the CPU never sees these buffers.
//
// `reorder` below is the older, narrower expression of the same idea (whole-tensor SoA for
// weights) and is kept as-is because ~50 call sites read it.
enum ggml_sycl_layout_kind : uint8_t {
    GGML_SYCL_LAYOUT_CANONICAL = 0,  // exactly as ggml packs the type
    GGML_SYCL_LAYOUT_SOA_SPAN  = 1,  // quants then scales, repeating every `span` elements
    GGML_SYCL_LAYOUT_SOA_WHOLE = 2,  // quants then scales, once over the whole tensor
    GGML_SYCL_LAYOUT_MASK_BITS = 3,  // a {0,-inf} mask packed to one bit per element
};

struct ggml_sycl_layout {
    ggml_sycl_layout_kind kind = GGML_SYCL_LAYOUT_CANONICAL;
    ggml_type             type = GGML_TYPE_COUNT;  // the block type the permutation applies to
    int32_t               span = 0;                // elements per self-contained unit

    bool is_canonical() const { return kind == GGML_SYCL_LAYOUT_CANONICAL; }
};

// What layout a tensor's bytes are actually in. The weight reorder and the SoA KV cache are the
// same idea at different spans, so they share one descriptor rather than a bool beside it.
struct optimize_feature {
    ggml_sycl_layout layout;

    // the whole-tensor SoA that reorder_qw() produces
    bool is_reordered() const { return layout.kind == GGML_SYCL_LAYOUT_SOA_WHOLE; }

    void set_reordered(ggml_type t) {
        layout.kind = GGML_SYCL_LAYOUT_SOA_WHOLE;
        layout.type = t;
    }
};

// Per-device policy, not a layout: whether this device should use reordered weights at all.
struct device_opt_feature {
    bool reorder = false;
};

struct sycl_device_info {
    int cc;  // compute capability
    int nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    int warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    int max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    bool    vmm;                // virtual memory support
    bool    l0_device_type_valid;
    bool    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    device_opt_feature opt_feature;
    bool    usm_system_support; // support for USM system allocations
#ifdef GGML_SYCL_GRAPH
    bool    graph_support;        // command graphs can be recorded and replayed
    bool    graph_update_support; // a finalized command graph can be updated
#endif
};


struct ggml_sycl_device_info {
    int device_count;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    int max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    bool ext_oneapi_level_zero = true; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

const ggml_sycl_device_info & ggml_sycl_info();

static constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

struct ggml_sycl_pool {
    virtual ~ggml_sycl_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        if (ptr)
            pool->free(ptr, actual_size);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_sycl_pool_alloc() = default;
    ggml_sycl_pool_alloc(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {
  void* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

extern int g_ggml_sycl_use_level_zero_api;
void * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
void ggml_sycl_free_device(void *ptr, sycl::queue &q);

void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

// one work-group of the grouped GEMM: rows [n0, n1) of the expert-major buffers belong to expert

struct ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

namespace sycl_ex = sycl::ext::oneapi::experimental;

#ifdef GGML_SYCL_GRAPH
struct ggml_sycl_graph {
    // src data/ne/nb are kept next to the node copy: the scheduler can hand back the same src
    // pointer with different contents or shape, see https://github.com/ggml-org/llama.cpp/pull/21736
    struct node_properties {
        ggml_tensor node;
        void *      node_src_data_ptrs[GGML_MAX_SRC];
        int64_t     node_src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
        size_t      node_src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];
    };

    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
    std::vector<node_properties> node_props;
    bool     warmup_complete = false;
    uint64_t uid             = 0;
    int64_t  last_used_time  = 0;

    // result of check_graph_compatibility() and graph_needs_reorder(), and the uid they were made for
    bool     compatible     = false;
    bool     needs_reorder  = true;
    uint64_t compatible_uid = 0;
};

static_assert(std::is_trivial<ggml_sycl_graph::node_properties>::value, "node_properties must be trivial");
#endif

struct ggml_backend_sycl_context {
    int device;
    std::string name;
    device_opt_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(int device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(int device, int stream) {
        if (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        return qptrs[device][stream];
    }

    queue_ptr stream() {
        return stream(device, 0);
    }

#if GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        return eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(int device, int _stream) {
        auto q = stream(device, _stream);
        return stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        if (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            return eng;
        }
        else
        {
            return it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        if (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            return stream;
        }
        else
        {
            return it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        return stream_dnnl(device, 0);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_fattn_kv_buffers> fattn_bufs[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;



    static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device);

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, int device);

    static std::unique_ptr<ggml_sycl_fattn_kv_buffers> new_fattn_kv_buffers(queue_ptr qptr, int device);

    ggml_sycl_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        return *pools[device];
    }

    ggml_sycl_pool & pool() {
        return pool(device);
    }

    ggml_sycl_fattn_kv_buffers & fattn_buffers(int device) {
        if (fattn_bufs[device] == nullptr) {
            fattn_bufs[device] = new_fattn_kv_buffers(stream(device, 0), device);
        }
        return *fattn_bufs[device];
    }

    ggml_sycl_fattn_kv_buffers & fattn_buffers() {
        return fattn_buffers(device);
    }

#ifdef GGML_SYCL_GRAPH
    // Map from first node pointer to graph - allows multiple graphs per context when the
    // computation is split across CPU/GPU (e.g. with --n-cpu-moe)
    std::unordered_map<const void *, std::unique_ptr<ggml_sycl_graph>> sycl_graphs;

    int64_t last_graph_eviction_sweep = 0;

    ggml_sycl_graph * sycl_graph(const void * first_node_ptr) {
        const int64_t time_now = ggml_time_us();

        // sweep every 5s, evicting graphs unused for >=10s
        if (time_now - last_graph_eviction_sweep >= 5'000'000) {
            last_graph_eviction_sweep = time_now;
            for (auto it = sycl_graphs.begin(); it != sycl_graphs.end(); ) {
                if (time_now - it->second->last_used_time >= 10'000'000) {
                    it = sycl_graphs.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto it = sycl_graphs.find(first_node_ptr);
        if (it == sycl_graphs.end()) {
            it = sycl_graphs.emplace(first_node_ptr, std::make_unique<ggml_sycl_graph>()).first;
        }
        it->second->last_used_time = time_now;
        return it->second.get();
    }
#endif

    // A QSA selection bitmap is built at the mask chain, where the top-k list is still live,
    // and read by the flash attention node the chain feeds. The reader returns the pool block
    // as soon as it has it; anything left over is returned when the next graph starts, so an
    // abandoned graph cannot leak.
    struct qsa_sel_block {
        const ggml_tensor * key;
        void *              ptr;
        size_t              size;
    };

    std::vector<qsa_sel_block> qsa_sel;
    size_t                     qsa_sel_high_water = 0;

    void qsa_sel_put(const ggml_tensor * key, void * ptr, size_t size) {
        qsa_sel.push_back({ key, ptr, size });
        qsa_sel_high_water = std::max(qsa_sel_high_water, qsa_sel.size());
    }

    // null if the key is unknown. The block stays reserved until qsa_sel_drop(), because the
    // reader allocates from the same pool and would otherwise be handed its own input back.
    void * qsa_sel_find(const ggml_tensor * key) {
        for (const qsa_sel_block & b : qsa_sel) {
            if (b.key == key) {
                return b.ptr;
            }
        }
        return nullptr;
    }

    void qsa_sel_drop(const ggml_tensor * key) {
        for (size_t i = 0; i < qsa_sel.size(); i++) {
            if (qsa_sel[i].key == key) {
                pool().free(qsa_sel[i].ptr, qsa_sel[i].size);
                qsa_sel.erase(qsa_sel.begin() + i);
                return;
            }
        }
    }

    void qsa_sel_reset() {
        for (const qsa_sel_block & b : qsa_sel) {
            pool().free(b.ptr, b.size);
        }
        qsa_sel.clear();
    }

    ggml_sycl_pool & host_pool(int device) {
        if (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        return *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { return host_pool(device); }
};

// common device functions

static __dpct_inline__ float warp_reduce_sum(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    return x;
}

static __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    return a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ int warp_reduce_sum(int x) {
  return sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  return x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ float warp_reduce_sum(float x, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  return x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  return a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  return a;
}

static constexpr int ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  return WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ int warp_reduce_all(int x) {
    if (width == ggml_sycl_get_physical_warp_size()) {
        return sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } else {
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        return x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ int warp_reduce_any(int x) {
    if (width == ggml_sycl_get_physical_warp_size()) {
        return sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } else {
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        return x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  return x;
}

static __dpct_inline__ float warp_reduce_max(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    return x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<int N>
__dpct_inline__ size_t calculate_offset(const std::array<int, N> & strides, const std::array<int, N> & indices) {
    size_t offset = 0;
#pragma unroll
    for (int i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    return offset;
}

// Helper for vec loading aligned data
template <typename Tp, int n>
inline sycl::vec<Tp, n> vec_aligned_load(const Tp* aligned_ptr) {
    return *reinterpret_cast<const sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, int dim>
static __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size);

constexpr size_t ceil_div(const size_t m, const size_t n) {
    return (m + n - 1) / n;
}

bool gpu_has_xmx(sycl::device &dev);

int ggml_sycl_get_env(const char *env_name, int default_val);

template <int N, class T> std::string debug_get_array_str(const std::string & prefix, const T array[N]) {
    if (LIKELY(!g_ggml_sycl_debug)) {
        return "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    for (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    if constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    return ss.str();
}

inline std::string debug_get_tensor_str(const std::string &prefix,
        const ggml_tensor *tensor, const std::string &suffix = "") {
    std::stringstream ss;
    if (LIKELY(!g_ggml_sycl_debug)) { return ss.str(); }
    ss << prefix.c_str() << "=";
    if (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        if (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        if (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } else {
        ss << "nullptr";
    }
    ss << suffix;
    return ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
struct scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(const std::string_view & func, const std::string_view & func_suffix, const ggml_tensor * dst,
                         std::size_t num_src, const std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        if (LIKELY(!g_ggml_sycl_debug)) {
            return;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        if (dst) {
            for (std::size_t i = 0; i < num_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(const std::string_view & func, const ggml_tensor * dst, std::size_t num_src,
                         const std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, num_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  private:
    std::string_view func;
    std::string_view func_suffix;
};

static __dpct_inline__ float get_alibi_slope(const float    max_bias,
                                             const uint32_t h,
                                             const uint32_t n_head_log2,
                                             const float    m0,
                                             const float    m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return dpct::pow(base, exph);
}

static const sycl::uint3 init_fastdiv_values(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
static constexpr int ggml_sycl_get_max_cpy_bytes() {
    return 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <int nbytes, int alignment = 0>
static __dpct_inline__ void ggml_sycl_memcpy_1(void * dst, const void * src) {
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((const sycl::int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((const sycl::int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    return res;
}

static __dpct_inline__ uint32_t fastdiv(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_values.x());
    return (hi + n) >> fastdiv_values.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<float>(x),static_cast<float>(y));
    return res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_value(static_cast<float>(H.x()), static_cast<float>(H.y()));
    return float2_value;
}

static __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z();
    return sycl::uint2(div_val, mod_val);
}

static __dpct_inline__ int ggml_sycl_dp4a(const int a, const int b, int c) {
    return dpct::dp4a(a, b, c);
}

static __dpct_inline__ float ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

sycl::float2 __dpct_inline__ __half22float2(const sycl::half2 &H) {
    sycl::float2 float2_value(static_cast<float>(H.x()), static_cast<float>(H.y()));
    return float2_value;
}

float __dpct_inline__ __half2float(sycl::half H) {
    return static_cast<float>(H);
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const sycl::float2 v, const sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const sycl::half2 v, const sycl::half2 u) {
#ifdef GGML_SYCL_F16
    const sycl::float2 tmp = (v * u).template convert<float, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#else
    const sycl::float2 tmpv = __half22float2(v);
    const sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void ggml_sycl_mad(sycl::half2 & acc, const sycl::half2 v, const sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#else
    const sycl::float2 tmpv = __half22float2(v);
    const sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <int n>
struct ggml_sycl_unroll {
    template <typename Func, typename... Args>
    void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

static __dpct_inline__ sycl::half2 ggml_sycl_hmax2(const sycl::half2 a, const sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<float, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<float, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    return ret;
}

static __dpct_inline__ sycl::half ggml_sycl_hmax(const sycl::half a, const sycl::half b) {
    return sycl::vec<float, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<float, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<float, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static __dpct_inline__ uint32_t __hgt2_mask(const sycl::half2 a, const sycl::half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float(a[0]) > float(b[0]));
    const uint32_t mask_high = 0xFFFF0000 * (float(a[1]) > float(b[1]));
    return mask_low | mask_high;
}

static __dpct_inline__ uint32_t fastmodulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z();
}

static bool fast_fp16_available(const int cc) {
    GGML_UNUSED(cc);
    return true;   //Intel GPUs always support FP16.
}

enum class block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, int warp_size>
struct block_reduce_policy;

template <typename T, typename... Ts>
inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
inline constexpr bool ggml_sycl_dependent_false_v = false;

#define WARP_32_SIZE 32

template <typename T, int warp_size> struct block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    static T reduce(T val) {
        if constexpr (is_any<T, float, sycl::float2, sycl::half2, int>) {
            return warp_reduce_sum<warp_size>(val);
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    static T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return 0.0f;
        } else if constexpr (std::is_same_v<T, sycl::float2>) {
            return sycl::float2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, sycl::half2>) {
            return sycl::half2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, int>) {
            return 0;
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, int warp_size> struct block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    static T reduce(T val) {
        if constexpr (is_any<T, float, sycl::half2>) {
            return warp_reduce_max<warp_size>(val);
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    static T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return -INFINITY;
        } else if constexpr (std::is_same_v<T, sycl::half2>) {
            return sycl::half2(-INFINITY, -INFINITY);
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, int warp_size, typename T>
static T block_reduce(T val, T * shared_vals, int block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    const int block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    const int nthreads = item_ct1.get_local_range(2);
    const int nwarps = nthreads / WARP_SIZE;

    if (block_size > warp_size) {
        assert((block_size <= 1024) && (block_size % warp_size) == 0);
        const int warp_id = item_ct1.get_local_id(2) / warp_size;
        const int lane_id = item_ct1.get_local_id(2) % warp_size;
        if (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        float tmp = 0.f;
        if (lane_id < (static_cast<int>(block_size) / warp_size)) {
            for (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        return block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    return val;
}

static __dpct_inline__ float ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } else {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (float) exp - 7.0f);
    }
    return raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP
