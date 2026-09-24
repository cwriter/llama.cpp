#ifndef GGML_SYCL_KQ_MASK_BITS_HPP
#define GGML_SYCL_KQ_MASK_BITS_HPP

#include "common.hpp"

// The causal mask holds exactly two values, 0 and -inf, so f16 spends 16 bits carrying one bit.
// Packing it costs nothing at the ggml level: the tensor keeps its f16 type and shape, and only
// the bytes inside the backend's own allocation change, so ggml, the scheduler and
// supports_op() are unaffected and no other backend can be handed one.
//
// The saving is read traffic, not footprint. ggml sizes the allocation from type x shape and
// ggml_backend_buft_get_alloc_size() may only ever grow it, so the reservation stays; what
// changes is that a flash-attention kernel re-reads 1 bit per cell instead of 16, and the host
// uploads a sixteenth as many bytes.
//
// EXPECTED NULL: this is not expected to pay off, on arithmetic rather than implementation.
// The mask is 2 bytes per (cell, token) while K and V are 2*head_size bytes per cell, so at
// head_size 256 the mask is about 0.4% of what a flash-attention kernel reads. Cutting 0.4% by
// 16x is invisible, and that ratio does not change with depth, model or device count. A first
// A/B did measure flat, but in a configuration not worth quoting - wrong model, one device,
// graphs off, and likely oversubscribed into host memory - so it is withdrawn; a run in the
// production configuration is still outstanding. The upload shrinks as advertised (257 MiB ->
// 16 MiB per card at a 131k prefill). Correctness is verified; the win is not there.
// Kept behind GGML_SYCL_KQ_MASK_BITS (default 0) because the machinery is correct and may matter
// on a head_size where the ratio is less lopsided. Do the traffic-ratio division before
// extending it.
//
// Row r of the packed form lives at r * ggml_sycl_kq_mask_row_bytes(ne0), bit (c & 31) of word
// (c >> 5) set meaning "attend to cell c". A packed row is always smaller than the f16 row it
// replaces, so it fits inside the original allocation with room to spare.

static inline size_t ggml_sycl_kq_mask_row_bytes(int64_t ne0) {
    return (size_t) ((ne0 + 31) / 32) * sizeof(uint32_t);
}

// Shape/type/name eligibility of a tensor that owns its bytes.
bool ggml_sycl_kq_mask_eligible(const ggml_tensor * t);

// True only when this tensor's bytes are packed. Keyed on the tensor that OWNS the bytes, never
// on a view's shape - deciding it from a view is how the KV layout first produced garbage.
bool ggml_sycl_kq_mask_is_bits(const ggml_tensor * t);

// marks an eligible tensor; called from buffer init before anything writes to it
void ggml_sycl_kq_mask_mark(ggml_tensor * t);

// host-side pack of a dense f16 {0,-inf} mask, and the inverse, for get_tensor
void ggml_sycl_kq_mask_pack  (void * dst, const void * src, int64_t ne0, int64_t nrows);
void ggml_sycl_kq_mask_unpack(void * dst, const void * src, int64_t ne0, int64_t nrows);

// device-side expansion back to a dense f16 mask, for readers that have not been taught the
// packed form; correctness first, so an untaught kernel is slow rather than wrong
void ggml_sycl_kq_mask_to_f16(const void * bits, sycl::half * y, int64_t ne0, int64_t nrows,
                              queue_ptr stream);

#endif  // GGML_SYCL_KQ_MASK_BITS_HPP
