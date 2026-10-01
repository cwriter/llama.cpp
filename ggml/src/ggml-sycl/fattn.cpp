//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//


#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include "fattn-vec.hpp"
#include "fattn.hpp"
#include "kq-mask-bits.hpp"
#include "fattn-onednn.hpp"
#include "fattn-sparse.hpp"

extern int g_ggml_sycl_fattn_prefer_vec;

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

static void ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_SYCL_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#endif // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_MKL      = 300,
};


// The shape envelope the oneMKL kernel is validated for, apart from the batch and the KV length.
// Split out of the dispatcher so the staging budget below can ask the same question before it declines oneDNN.
static bool ggml_sycl_fattn_mkl_envelope(const ggml_tensor * dst) {
    if (g_ggml_sycl_enable_mkl_fa <= 0) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!Q || !K || !V || !mask || sinks || K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }

    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (Q->ne[2] / K->ne[2] < 2 || Q->ne[0] < 64 || Q->ne[0] > 512 || Q->ne[0] % 64 != 0 ||
        Q->ne[0] != V->ne[0] ||
        max_bias != 0.0f || logit_softcap != 0.0f ||
        (Q->ne[3] != K->ne[3] && K->ne[3] != 1)) {
        return false;
    }
    // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
    // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
    // (nb1 == H * ne0*2). Only pathological test strides like nb1=32 or
    // nb1=75 for ne0=40 fall through to TILE.
    for (const ggml_tensor * t : {K, V}) {
        if (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
            return false;
        }
    }
    return true;
}

static bool ggml_sycl_fattn_mkl_supported(const ggml_tensor * dst) {
    return ggml_sycl_fattn_mkl_envelope(dst) && dst->src[0]->ne[1] >= 32 && dst->src[1]->ne[1] >= 1024;
}

// cells of the whole KV cache that K views
static int64_t ggml_sycl_fattn_cache_cells(const ggml_tensor * K) {
    if (K->ne[1] <= 0) {
        return K->ne[1];
    }
    const int64_t       per_cell = ggml_nelements(K) / K->ne[1];
    const ggml_tensor * Kc       = K->view_src ? K->view_src : K;
    return per_cell > 0 ? std::max(K->ne[1], ggml_nelements(Kc) / per_cell) : K->ne[1];
}

// oneDNN SDPA and TILE both stage an F16 copy of the WHOLE KV cache, which the graph allocator
// has to reserve, and which grows with the context and not with the batch. At the
// GGML_SYCL_FA_MAX_MEM_MIB ceiling the node goes to the oneMKL kernel instead: it walks the cache
// in chunks sized to fit the same ceiling, so it reserves nothing.
//
// Size the decision from the whole KV cache and the static kernel envelope, not from this graph's
// n_kv or batch. The allocator plans a node for every batch and KV length it may later run at
// (get_max_alloc_size), so a capped node must stay bounded on every path; see
// ggml_sycl_fattn_capped_tile().
bool ggml_sycl_fattn_stage_capped(const ggml_tensor * dst) {
    if (g_ggml_sycl_fa_max_mem_mib <= 0 || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    // the sparse path's gathered node is small by construction, and the slots past its live cells
    // hold stale data that only TILE and VEC know to skip
    if (ggml_get_op_params_i32(dst, 4) == GGML_SYCL_FATTN_GATHERED) {
        return false;
    }
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!K || !V || ggml_sycl_fattn_onednn_binds_kv(K, V)) {
        return false;
    }
    const ggml_tensor * Kc = K->view_src ? K->view_src : K;
    const ggml_tensor * Vc = V->view_src ? V->view_src : V;
    // count both even when V aliases K: oneDNN stages them apart, only TILE shares the copy
    const size_t stage = (size_t) (ggml_nelements(Kc) + ggml_nelements(Vc)) * 2;
    return stage >= (size_t) g_ggml_sycl_fa_max_mem_mib * 1024 * 1024 && ggml_sycl_fattn_mkl_envelope(dst);
}

static ggml_sycl_fattn_extra ggml_sycl_fattn_layout(const ggml_tensor * dst, int64_t n_kv, bool onednn, bool tile);

static size_t ggml_sycl_fattn_layout_size(const ggml_tensor * dst, int64_t n_kv, bool onednn, bool tile) {
    return (size_t) (ggml_sycl_fattn_layout(dst, n_kv, onednn, tile).end - (uintptr_t) dst->data);
}

// A capped node never stages the whole cache. TILE (which stages K/V) still takes a short context,
// where oneMKL does not apply, and a short batch whose staging is no larger than what the sparse
// path stages for its largest batch. The chunked oneMKL kernel takes every other batch.
static bool ggml_sycl_fattn_capped_tile(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    if (K->ne[1] < 1024) {
        return true;
    }
    if (Q->ne[1] >= 32) {
        return false;
    }
    const size_t budget = ggml_sycl_fattn_sparse_max_alloc_size(dst, ggml_sycl_fattn_cache_cells(K));
    return ggml_sycl_fattn_layout_size(dst, K->ne[1], false, true) <= budget;
}

static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef SYCL_FLASH_ATTN
    GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// SYCL_FLASH_ATTN

    if(!g_ggml_sycl_enable_flash_attention) return BEST_FATTN_KERNEL_NONE;

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // XMX-accelerated path: oneDNN SDPA (native F16 and dequant+non-F16).
    // ONEDNN requires min 32 query tokens — short-circuit decode to avoid
    // calling _supported() on every decode FA call.
    const bool stage_capped = ggml_sycl_fattn_stage_capped(dst);
    if (stage_capped && !ggml_sycl_fattn_capped_tile(dst)) {
        return BEST_FATTN_KERNEL_MKL;
    }

    // oneDNN needs a dense f16 mask, and a compact one could only be expanded into as much
    // scratch as the compaction saved; the MKL kernel below reads the bits directly
    if (Q->ne[1] >= 32 && !stage_capped && !ggml_sycl_kq_mask_is_compact(mask)
        && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        return BEST_FATTN_KERNEL_ONEDNN;
    }

    // MKL path: XMX-accelerated GEMM for prompt processing (all KV cache types).
    // The MKL kernel converts non-F16 K/V to F16 via to_fp16_sycl before GEMM,
    // so quantized, F16, BF16, and F32 caches all benefit from XMX acceleration.
    // Activates automatically when flash-attn is enabled (--flash-attn on or -fa)
    // and n_kv >= 1024. Falls through to TILE/VEC for ALiBi, logit softcap,
    // and mismatched batch dimensions (unsupported by the MKL kernel).
    // Set GGML_SYCL_ENABLE_MKL_FA=0 to force TILE/VEC path for A/B testing.
    // Example: GGML_SYCL_ENABLE_MKL_FA=0 llama-cli -m model.gguf -fa -ngl 99 ...
    // Note: MKL GEMM calls are incompatible with SYCL graph capture replay.
    // MKL is validated for the mainstream GQA envelope: grouped-query
    // (gqa_ratio >= 2), head_dim a multiple of 64 in [64,512] with matching
    // K/V head size, mask, no sinks/ALiBi/softcap. Gemma's global layers use
    // head_dim 512, so the cap must include it. Head sizes not a multiple of
    // 64 (72/80/96), MHA (gqa_ratio == 1), and MLA (DKQ != DV, e.g. 576/512)
    // fall through to TILE/VEC; see follow-up work.
    if (ggml_sycl_fattn_mkl_supported(dst)) {
        return BEST_FATTN_KERNEL_MKL;
    }
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_SYCL_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_SYCL_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_SYCL_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes.
    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    const bool has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    const bool can_use_vector_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    if (!stage_capped && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        return BEST_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                const gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                if (!g_ggml_sycl_fattn_prefer_vec &&
                    (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31)) {
                    return BEST_FATTN_KERNEL_TILE;
                }
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

// Which flash-attention kernels can read the packed mask directly. Everything else is handed a
// dense expansion, so adding a kernel here is what turns the saving on for it.
bool ggml_sycl_fattn_reads_mask_bits(const ggml_tensor * dst) {
    if (!ggml_sycl_kq_mask_is_bits(dst->src[3])) {
        return false;
    }
    switch (ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst)) {
        case BEST_FATTN_KERNEL_TILE:
            return ggml_sycl_fattn_tile_reads_mask_bits(dst);
        case BEST_FATTN_KERNEL_MKL:
            // the chunked oneMKL kernel already takes a one-bit-per-cell selection map, added for
            // the QSA mask fusion; its SEL=2 mode folds the mask into the bit and never reads a
            // dense one, which is exactly what a packed causal mask is. A bit carries no
            // magnitude, so only where no ALiBi slope applies.
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
                return max_bias == 0.0f;
            }
        default:
            return false;
    }
}

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);

    // The packed mask is backend-local and only some kernels read it. Rather than let an
    // untaught one reinterpret bits as f16 - the way the QSA mask fusion once did, silently -
    // expand it here for every reader that has not been taught, and substitute the dense copy.
    // A taught kernel is excluded above this point, so it never pays for the expansion.
    ggml_tensor                      dst_sub;
    ggml_tensor                      mask_sub;
    ggml_sycl_pool_alloc<sycl::half> mask_dense(ctx.pool());
    const bool mask_packed = ggml_sycl_kq_mask_is_bits(dst->src[3]);
    if (mask_packed) {
        ggml_sycl_kq_mask_taught(ctx, dst);  // both branches below read it correctly
    }
    const bool mkl_takes_bits =
        mask_packed && ggml_sycl_fattn_reads_mask_bits(dst) &&
        ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst) == BEST_FATTN_KERNEL_MKL;

    if (mask_packed && ggml_sycl_fattn_reads_mask_bits(dst) && !mkl_takes_bits) {
        // taught reader: same bytes, but the row stride it must walk is the packed one
        const ggml_tensor * m = dst->src[3];
        mask_sub        = *m;
        mask_sub.nb[1]  = ggml_sycl_kq_mask_row_bytes(m->ne[0]);
        mask_sub.nb[2]  = mask_sub.nb[1] * m->ne[1];
        mask_sub.nb[3]  = mask_sub.nb[2] * m->ne[2];
        dst_sub         = *dst;
        dst_sub.src[3]  = &mask_sub;
        dst             = &dst_sub;
    } else if (mask_packed && !mkl_takes_bits) {
        const ggml_tensor * m     = dst->src[3];
        const int64_t       nrows = ggml_nelements(m) / m->ne[0];
        mask_dense.alloc(ggml_nelements(m));
        ggml_sycl_kq_mask_to_f16(m->data, mask_dense.get(), m->ne[0], nrows, ctx.stream());

        mask_sub        = *m;
        mask_sub.data   = mask_dense.get();
        mask_sub.extra  = nullptr;  // so it no longer answers "packed"
        dst_sub         = *dst;
        dst_sub.src[3]  = &mask_sub;
        dst             = &dst_sub;
    }

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    if (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        return;
    }

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    static int nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    if (nkv_debug == 1) {
        const ggml_tensor * K_dbg = dst->src[1];
        const ggml_tensor * V_dbg = dst->src[2];
        static int64_t last_nkv_d256 = 0, last_nkv_d512 = 0;
        static int fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        int Dk = (int)K_dbg->ne[0];
        const char * kname = "TILE";
        best_fattn_kernel k = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
        if (k == BEST_FATTN_KERNEL_MKL)  kname = "MKL";
        if (k == BEST_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        if (k == BEST_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        if (Dk == 256) {
            delta = cur_nkv - last_nkv_d256;
            last_nkv_d256 = cur_nkv;
        } else if (Dk == 512) {
            delta = cur_nkv - last_nkv_d512;
            last_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (long long)cur_nkv, (long long)delta,
                (long long)V_dbg->ne[1]);
    }

    const best_fattn_kernel fk = ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst);
    switch (fk) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        case BEST_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#if GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#endif
            break;
        case BEST_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MKL:
            if (mkl_takes_bits) {
                const ggml_tensor * m = dst->src[3];
                ggml_sycl_flash_attn_ext_mkl(ctx, dst, (const uint32_t *) m->data,
                                             (int64_t) (ggml_sycl_kq_mask_row_bytes(m->ne[0]) / sizeof(uint32_t)),
                                             /* sel_mode = */ 2);
            } else {
                ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            }
            break;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    static int fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    static int fa_diag_count = 0;
    if (fa_diag == 1 && fa_diag_count < 6) {
        const ggml_tensor * K_diag = dst->src[1];
        const ggml_tensor * V_diag = dst->src[2];
        const ggml_tensor * Q_diag = dst->src[0];
        if (K_diag->ne[1] >= 1024) {
            fa_diag_count++;
            float diag_buf[64];
            dpct::queue_ptr q = ctx.stream();
            q->memcpy(diag_buf, dst->data, 64 * sizeof(float));
            q->wait();
            const char * kname = "???";
            best_fattn_kernel kb = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
            if (kb == BEST_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            if (kb == BEST_FATTN_KERNEL_MKL) kname = "MKL";
            if (kb == BEST_FATTN_KERNEL_TILE) kname = "TILE";
            if (kb == BEST_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_count, kname,
                    (int)K_diag->ne[0], (long long)K_diag->ne[1],
                    (long long)Q_diag->ne[1],
                    (long long)Q_diag->ne[2], (long long)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            for (int i = 0; i < 64; i += 8) {
                GGML_LOG_INFO("  [%2d] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        i,
                        *(unsigned *)&diag_buf[i+0], *(unsigned *)&diag_buf[i+1],
                        *(unsigned *)&diag_buf[i+2], *(unsigned *)&diag_buf[i+3],
                        *(unsigned *)&diag_buf[i+4], *(unsigned *)&diag_buf[i+5],
                        *(unsigned *)&diag_buf[i+6], *(unsigned *)&diag_buf[i+7]);
            }
        }
    }

}

bool ggml_sycl_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

// Mirrors the oneDNN-then-MKL order of ggml_sycl_get_best_fattn_kernel() for a prefill batch. The
// device only enters the VEC/TILE choice below MKL, so the answer does not depend on it. A capped
// node also runs a long short batch on oneMKL; that is not reported here.
bool ggml_sycl_fattn_picks_mkl(const ggml_tensor * dst) {
    if (!g_ggml_sycl_enable_flash_attention || dst->op != GGML_OP_FLASH_ATTN_EXT || !dst->src[0]) {
        return false;
    }
    const bool stage_capped = ggml_sycl_fattn_stage_capped(dst);
    if (dst->src[0]->ne[1] >= 32 && !stage_capped && ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        return false;
    }
    return ggml_sycl_fattn_mkl_supported(dst);
}

bool ggml_sycl_fattn_mkl_shape_ok(const ggml_tensor * dst) {
    return g_ggml_sycl_enable_flash_attention && dst->op == GGML_OP_FLASH_ATTN_EXT && dst->src[0] &&
           ggml_sycl_fattn_mkl_supported(dst);
}

bool ggml_sycl_flash_attn_ext_uses_library(int device, const ggml_tensor * dst) {
    const best_fattn_kernel kernel = ggml_sycl_get_best_fattn_kernel(device, dst);
    return kernel == BEST_FATTN_KERNEL_ONEDNN || kernel == BEST_FATTN_KERNEL_MKL;
}

static uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    if (n_halves == 0) {
        return 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_BUFFER_ALIGNMENT);
    const uintptr_t block = extra.end;
    extra.end += n_halves * sizeof(sycl::half);
    return block;
}

// The scratch after dst for oneDNN and/or TILE, with K/V at n_kv cells.
static ggml_sycl_fattn_extra ggml_sycl_fattn_layout(const ggml_tensor * dst, int64_t n_kv, bool onednn, bool tile) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int64_t d = K->ne[0];
    const int64_t H = Q->ne[2];
    const int64_t q = Q->ne[1];

    const size_t n_K = K->ne[1] > 0 ? (size_t) (ggml_nelements(K) / K->ne[1] * n_kv) : 0;
    const size_t n_V = V->ne[1] > 0 ? (size_t) (ggml_nelements(V) / V->ne[1] * n_kv) : 0;

    const bool tile_needs_K = tile && K->type != GGML_TYPE_F16;
    const bool tile_needs_V = tile && V->type != GGML_TYPE_F16;

    const bool V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_out = 0, need_scale = 0;
    if (onednn) {
        need_Q     = (size_t) H * q * d;
        need_out   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        if (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = n_K;
            need_V = n_V;
        }
    }
    if (tile_needs_K) {
        need_K = std::max(need_K, n_K);
    }
    if (tile_needs_V) {
        need_V = std::max(need_V, n_V);
    }

    extra.Q_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_buffer_ptr = (V_is_K_view && !onednn && need_V)
                       ? extra.K_buffer_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.out_buffer_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_out);

    return extra;
}

// The node as the sparse check in ggml_sycl_flash_attn_ext() sees it: a packed mask that the
// kernel does not read is expanded to a dense one first. Fills mask/node when it differs.
static const ggml_tensor * ggml_sycl_fattn_sparse_view(const ggml_tensor * dst, bool read_bits, ggml_tensor & mask, ggml_tensor & node) {
    if (!ggml_sycl_kq_mask_is_bits(dst->src[3]) || read_bits) {
        return dst;
    }
    mask         = *dst->src[3];
    mask.extra   = nullptr;
    node         = *dst;
    node.src[3]  = &mask;
    return &node;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(const ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (dst->op != GGML_OP_FLASH_ATTN_EXT || !dst->src[0] || !dst->src[1] || !dst->src[2]) {
        return extra;
    }

    // calculate the worst-case memory consumption across the kernels this node may take
    if (!ggml_sycl_fattn_stage_capped(dst)) {
        return ggml_sycl_fattn_layout(dst, dst->src[1]->ne[1], ggml_sycl_flash_attn_ext_onednn_supported(dst, false), true);
    }
    // a capped node never takes oneDNN. The sparse path lays out its gathered node from dst->data
    ggml_tensor mask, node;
    const ggml_tensor * sv = ggml_sycl_fattn_sparse_view(dst, ggml_sycl_fattn_reads_mask_bits(dst), mask, node);
    if (ggml_sycl_fattn_sparse_applies(sv)) {
        extra.end = (uintptr_t) dst->data + ggml_sycl_fattn_sparse_alloc_size(sv);
        return extra;
    }
    if (ggml_sycl_fattn_capped_tile(dst)) {
        return ggml_sycl_fattn_layout(dst, dst->src[1]->ne[1], false, true);
    }
    return extra;  // oneMKL takes its scratch from the pool
}

size_t ggml_sycl_flash_attn_ext_get_alloc_size(const ggml_tensor * dst) {
    const ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    return (size_t) (extra.end - (uintptr_t) dst->data);
}

// The KV view may grow up to the whole cache and the batch may shrink, which can move the node to
// another kernel. Bound what get_alloc_size reports over all of them.
size_t ggml_sycl_flash_attn_ext_get_max_alloc_size(const ggml_tensor * dst) {
    size_t size = ggml_sycl_flash_attn_ext_get_alloc_size(dst);
    if (dst->op != GGML_OP_FLASH_ATTN_EXT || !dst->src[0] || !dst->src[1] || !dst->src[2]) {
        return size;
    }
    const int64_t cells = ggml_sycl_fattn_cache_cells(dst->src[1]);
    if (!ggml_sycl_fattn_stage_capped(dst)) {
        // the oneDNN KV ceiling only depends on n_kv, so drop it here: below it oneDNN may run
        ggml_tensor K = *dst->src[1];
        ggml_tensor d = *dst;
        K.ne[1]   = 0;
        d.src[1]  = &K;
        const bool onednn = ggml_sycl_flash_attn_ext_onednn_supported(&d, false);
        return std::max(size, ggml_sycl_fattn_layout_size(dst, cells, onednn, true));
    }
    // TILE on a short context at this batch, TILE within the sparse budget or the sparse path itself, oneMKL
    size = std::max(size, ggml_sycl_fattn_layout_size(dst, std::min<int64_t>(cells, 1024), false, true));
    ggml_tensor mask, node;
    const ggml_tensor * sv = ggml_sycl_fattn_sparse_view(dst, false, mask, node);
    size = std::max(size, ggml_sycl_fattn_sparse_max_alloc_size(sv, cells));
    return size;
}
