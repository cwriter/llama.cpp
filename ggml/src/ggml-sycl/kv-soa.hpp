#ifndef GGML_SYCL_KV_SOA_HPP
#define GGML_SYCL_KV_SOA_HPP

#include "common.hpp"

// A q8_0 KV cache whose bytes are permuted inside fixed spans of GGML_SYCL_KV_SOA_SPAN elements.
//
//   canonical : [d0 qs0[32]][d1 qs1[32]] ...        34 bytes per block, scale interleaved
//   this      : [qs0[32] .. qs7[32]][d0 .. d7]      same 272 bytes, permuted inside the span
//
// Same bytes, same strides, nothing added: only the order inside a span differs. Canonically a
// block's quants sit at a 2-byte aligned offset, so 16 features cost eight 2-byte loads; here
// they sit 16-byte aligned and cost one. That 8x message amplification is why five earlier
// online q8_0 flash-attention loaders lost to plain f16 staging, while a ceiling probe assuming
// this layout measured 1.27-1.78x faster than staging at decode.
//
// THE SPAN IS 256 ELEMENTS, NOT A ROW. llama-kv-cache.cpp hands flash attention a view that
// splits each row into heads (ggml_view_4d(k, n_embd_head_k, n_head_kv, n_kv, ns)), so a
// per-row permutation would make a head slice meaningless - the first attempt asserted on
// exactly that. 256 elements is 8 blocks is 272 bytes = LCM(34,16), the smallest unit that is
// both self-contained and 16-byte aligned throughout, and it divides head_dim here.
//
// A consumer must dispatch on the layout, never assume: permuted bytes read as interleaved
// blocks produce plausible-looking garbage rather than an error.

#define GGML_SYCL_KV_SOA_SPAN 256

// Static dispatch over the packing. Both specializations take the base of the containing span
// and a block index inside it, so a kernel is written once and the canonical case compiles to
// exactly the indexing it uses today.
template <ggml_sycl_layout_kind K> struct ggml_sycl_q8_0_access;

template <> struct ggml_sycl_q8_0_access<GGML_SYCL_LAYOUT_CANONICAL> {
    static constexpr int    span      = QK8_0;
    static constexpr size_t span_size = sizeof(block_q8_0);
    static __dpct_inline__ const int8_t * qs(const char * base, int iblk) {
        return ((const block_q8_0 *) base)[iblk].qs;
    }
    static __dpct_inline__ float d(const char * base, int iblk) {
        return (float) ((const block_q8_0 *) base)[iblk].d;
    }
};

template <> struct ggml_sycl_q8_0_access<GGML_SYCL_LAYOUT_SOA_SPAN> {
    static constexpr int    span      = GGML_SYCL_KV_SOA_SPAN;
    static constexpr int    nblk      = GGML_SYCL_KV_SOA_SPAN / QK8_0;
    static constexpr size_t span_size = (size_t) nblk * sizeof(block_q8_0);
    static __dpct_inline__ const int8_t * qs(const char * base, int iblk) {
        return (const int8_t *) base + iblk * QK8_0;
    }
    static __dpct_inline__ float d(const char * base, int iblk) {
        return (float) ((const sycl::half *) (base + GGML_SYCL_KV_SOA_SPAN))[iblk];
    }
};

// byte offset of the span containing element i0, plus the block index inside that span
template <ggml_sycl_layout_kind K>
static __dpct_inline__ void ggml_sycl_q8_0_locate(int64_t i0, size_t & span_off, int & iblk) {
    using A = ggml_sycl_q8_0_access<K>;
    const int64_t ispan = i0 / A::span;
    span_off            = (size_t) ispan * A::span_size;
    iblk                = (int) ((i0 - ispan * A::span) / QK8_0);
}

// Writable counterparts of the accessors above, for the paths that produce a span rather than
// consume one. The layout is stated once here so a writer and a reader cannot drift apart.
static __dpct_inline__ int8_t * ggml_sycl_q8_0_soa_qs_mut(char * span, int iblk) {
    return (int8_t *) span + iblk * QK8_0;
}

static __dpct_inline__ void ggml_sycl_q8_0_soa_set_d(char * span, int iblk, float d) {
    ((sycl::half *) (span + GGML_SYCL_KV_SOA_SPAN))[iblk] = (sycl::half) d;
}

// True only when the bytes this tensor refers to are permuted. Keyed on the tensor that OWNS
// the bytes, never on a view's shape: a false answer means "these bytes are canonical", so
// deciding it from a view is how the first attempt produced garbage.
bool ggml_sycl_kv_is_soa(const ggml_tensor * t);

// shape/type eligibility of an owning tensor, independent of whether it is marked
bool ggml_sycl_kv_soa_eligible(const ggml_tensor * t);

// marks an eligible tensor; called from buffer init before anything writes to it
void ggml_sycl_kv_soa_mark(ggml_tensor * t);

// dequantize a SoA-span q8_0 tensor (or any view of one) to a densely packed f16 buffer
void ggml_sycl_kv_soa_to_fp16(const void * vx, sycl::half * y, int64_t ne0, int64_t ne1,
                              int64_t ne2, size_t nb1, size_t nb2, queue_ptr stream);

// Canonical and SoA spans occupy the same bytes (8 * sizeof(block_q8_0) == 256 + 8 * sizeof(half)),
// so converting permutes inside each span and never moves one. That is why the host can hand out
// canonical bytes for a session file without a device round trip, and why the range check is a
// plain span-alignment test. dst may alias src.
bool ggml_sycl_kv_soa_range_ok(size_t offset, size_t nbytes);
void ggml_sycl_kv_soa_span_to_canonical(void * dst, const void * src, size_t nbytes);
void ggml_sycl_kv_soa_canonical_to_span(void * dst, const void * src, size_t nbytes);

#endif  // GGML_SYCL_KV_SOA_HPP
