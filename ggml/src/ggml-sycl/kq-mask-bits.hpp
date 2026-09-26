#ifndef GGML_SYCL_KQ_MASK_BITS_HPP
#define GGML_SYCL_KQ_MASK_BITS_HPP

#include "common.hpp"

// The causal mask holds exactly two values, 0 and -inf, so f16 spends 16 bits carrying one bit.
// Packing it costs nothing at the ggml level: the tensor keeps its f16 type and shape, and only
// the bytes inside the backend's own allocation change, so ggml, the scheduler and
// supports_op() are unaffected and no other backend can be handed one.
//
// ggml sizes the allocation from type x shape, so packing alone saves read traffic and upload
// bytes (257 -> 16 MiB per card at a 131k prefill), not footprint; at head_size 256 the mask is
// ~0.4% of what flash attention reads. The footprint saving comes from COMPACT below, which
// reports the packed size to the allocator: measured on qwen4exp at -c 131072, 3x B60,
// -240..-272 MiB peak VRAM per card, and -928..-957 MiB in pipeline mode, where every graph copy
// holds its own mask; prefill and text unchanged.
//
// Row r of the packed form lives at r * ggml_sycl_kq_mask_row_bytes(ne0), bit (c & 31) of word
// (c >> 5) set meaning "attend to cell c". A packed row is always smaller than the f16 row it
// replaces, so it fits inside the original allocation with room to spare.

static inline size_t ggml_sycl_kq_mask_row_bytes(int64_t ne0) {
    return (size_t) ((ne0 + 31) / 32) * sizeof(uint32_t);
}

// GGML_SYCL_KQ_MASK_BITS is a bitset. Any non-zero value packs the mask; the other bits choose
// what the saved bytes are used for:
enum ggml_sycl_kq_mask_mode {
    GGML_SYCL_KQ_MASK_PACK    = 1 << 0,  // pack the upload; the allocation stays f16-sized
    GGML_SYCL_KQ_MASK_TAIL    = 1 << 1,  // lend the unused part of the allocation as pool scratch
    GGML_SYCL_KQ_MASK_COMPACT = 1 << 2,  // allocate only the packed size; takes precedence over TAIL
};
static constexpr int GGML_SYCL_KQ_MASK_DEFAULT = GGML_SYCL_KQ_MASK_PACK | GGML_SYCL_KQ_MASK_COMPACT;

// COMPACT: get_alloc_size reports the packed size, below ggml_nbytes. Nothing can expand such a
// mask in place, so it is only chosen when every reader is taught. graph_optimize runs before
// each allocation and checks the readers; one untaught reader, or ALiBi (whose mask one bit
// cannot hold), turns COMPACT off for the rest of the process and the mask is allocated at its
// f16 size again.
// Only masks a scheduled graph has shown to graph_optimize are compacted, and only while every
// graph passes; a mask whose values turn out to be more than 0/-inf switches it off for good.
bool ggml_sycl_kq_mask_compact_on();
void ggml_sycl_kq_mask_compact_check(const ggml_cgraph * cgraph);
void ggml_sycl_kq_mask_compact_veto(const char * why, const char * name);
// true when t was allocated at the packed size
bool ggml_sycl_kq_mask_is_compact(const ggml_tensor * t);

// Shape/type/name eligibility of a tensor that owns its bytes.
bool ggml_sycl_kq_mask_eligible(const ggml_tensor * t);

// True only when this tensor's bytes are packed. Keyed on the tensor that OWNS the bytes, never
// on a view's shape - deciding it from a view is how the KV layout first produced garbage.
bool ggml_sycl_kq_mask_is_bits(const ggml_tensor * t);

// marks an eligible tensor; called from buffer init before anything writes to it
void ggml_sycl_kq_mask_mark(ggml_tensor * t);

// host-side pack of a dense f16 mask, and the inverse, for get_tensor. The pack returns false
// (dst undefined) when a value is neither +0 nor -inf: one bit cannot hold it, so the caller
// stores that upload dense instead
bool ggml_sycl_kq_mask_pack  (void * dst, const void * src, int64_t ne0, int64_t nrows);
void ggml_sycl_kq_mask_unpack(void * dst, const void * src, int64_t ne0, int64_t nrows);

// device-side expansion back to a dense f16 mask, for readers that have not been taught the
// packed form; correctness first, so an untaught kernel is slow rather than wrong
void ggml_sycl_kq_mask_to_f16(const void * bits, sycl::half * y, int64_t ne0, int64_t nrows,
                              queue_ptr stream);

// Taught readers. The mask holds +0 or -inf, so x + mask is x or -inf, bit for bit what the
// dense op computes. Each returns false when the node is not a packed-mask reader it can run.
bool ggml_sycl_kq_mask_try_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_kq_mask_try_cpy(ggml_backend_sycl_context & ctx, ggml_tensor * node);
// dst = a + mask, for the fused f16 mask cast + ADD; m is the packed mask, same rows as dst
void ggml_sycl_kq_mask_add_bits(ggml_backend_sycl_context & ctx, const ggml_tensor * a, const ggml_tensor * m,
                                ggml_tensor * dst);
// a taught kernel ran `reader`; graph_end checks that every reader of the mask did
void ggml_sycl_kq_mask_taught(ggml_backend_sycl_context & ctx, const ggml_tensor * reader);

// GGML_SYCL_KQ_MASK_BITS=2 also hands the dead part of the mask's allocation out as scratch.
//
// The mask needs row_bytes(ne0) * rows bytes; the rest of its allocation is unused. Three
// things make that rest safe to lend to the SYCL pool for a whole split:
//   - get_alloc_size gives an eligible mask the largest size seen on the device, so its slot is
//     the worst-case one in every allocator plan, not just the reserve plan;
//   - graph_optimize keeps the mask allocated until the last node of every split that reads it,
//     so ggml-alloc places nothing else in the slot during that split;
//   - graph_begin checks both again on the real graph (no node or source overlaps the tail) and
//     lends nothing if either fails, or if the mask is not packed in this graph.
// Scratch is taken from the tail first and from the pool when it does not fit. Every tail block
// must be returned before the split ends, which graph_end checks.
bool   ggml_sycl_kq_mask_tail_on();
size_t ggml_sycl_kq_mask_alloc_size(int device, const ggml_tensor * t, size_t size);
// the eligible mask a split graph reads, or null; for graph_optimize
ggml_tensor * ggml_sycl_kq_mask_read_by(const ggml_cgraph * cgraph);

struct ggml_sycl_kq_mask_graph {
    ggml_backend_sycl_context *      ctx  = nullptr;
    const ggml_tensor *              mask = nullptr;  // the packed mask this graph reads
    std::vector<const ggml_tensor *> readers;         // nodes that read its bytes
    bool                             tail = false;    // the pool override is set
};

// classify the readers (an untaught one gets a dense copy for this graph) and arm the tail
void ggml_sycl_kq_mask_graph_begin(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph,
                                   ggml_sycl_kq_mask_graph & g);
// the node the graph runs next; only the overlap probe needs it
void ggml_sycl_kq_mask_graph_at(ggml_sycl_kq_mask_graph & g, const ggml_tensor * node);
void ggml_sycl_kq_mask_graph_end(ggml_sycl_kq_mask_graph & g);

#endif  // GGML_SYCL_KQ_MASK_BITS_HPP
