#include "kq-mask-bits.hpp"
#include "qsa-mask.hpp"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstring>
#include <atomic>
#include <mutex>
#include <set>
#include <string>

extern int g_ggml_sycl_enable_graph;

static bool kq_mask_compact_applies(const ggml_tensor * t);

static size_t kq_mask_packed_bytes(const ggml_tensor * t) {
    return ggml_sycl_kq_mask_row_bytes(t->ne[0]) * (size_t) (ggml_nelements(t) / t->ne[0]);
}

bool ggml_sycl_kq_mask_eligible(const ggml_tensor * t) {
    if (!g_ggml_sycl_kq_mask_bits || t == nullptr || t->type != GGML_TYPE_F16) {
        return false;
    }
    // only the graph input and the scheduler's copies of it, never a tensor computed from it
    if (t->view_src != nullptr || t->op != GGML_OP_NONE || !ggml_is_contiguous(t)) {
        return false;
    }
    if ((g_ggml_sycl_kq_mask_bits & GGML_SYCL_KQ_MASK_COMPACT) && !ggml_sycl_kq_mask_compact_on()) {
        return false;  // COMPACT was vetoed: the mask stays dense f16
    }
    // the scheduler names its copy "<backend>#<src name>#<n>", so match anywhere in the name
    if (strstr(t->name, "kq_mask") == nullptr) {
        return false;
    }
    // a packed row must fit inside the f16 row it replaces
    return t->ne[0] >= 32 && ggml_sycl_kq_mask_row_bytes(t->ne[0]) <= (size_t) t->ne[0] * sizeof(sycl::half);
}

bool ggml_sycl_kq_mask_is_bits(const ggml_tensor * t) {
    if (!g_ggml_sycl_kq_mask_bits || t == nullptr || t->type != GGML_TYPE_F16) {
        return false;
    }
    const ggml_tensor * owner = t;
    while (owner->view_src) {
        owner = owner->view_src;
    }
    const ggml_tensor_extra_gpu * extra = static_cast<const ggml_tensor_extra_gpu *>(owner->extra);
    return extra && extra->optimized_feature.layout.kind == GGML_SYCL_LAYOUT_MASK_BITS;
}

void ggml_sycl_kq_mask_mark(ggml_tensor * t) {
    if (!ggml_sycl_kq_mask_eligible(t)) {
        return;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(t->extra);
    if (!extra) {
        return;
    }
    extra->optimized_feature.layout.kind = GGML_SYCL_LAYOUT_MASK_BITS;
    extra->optimized_feature.layout.type = GGML_TYPE_F16;
    extra->optimized_feature.layout.span = 32;
    // recorded rather than recomputed, so a later COMPACT veto cannot change a live tensor's size
    extra->optimized_feature.layout.nbytes = kq_mask_compact_applies(t) ? kq_mask_packed_bytes(t) : 0;
}

bool ggml_sycl_kq_mask_pack(void * vdst, const void * vsrc, int64_t ne0, int64_t nrows) {
    const uint16_t * src   = (const uint16_t *) vsrc;
    char *           dst   = (char *) vdst;
    const size_t     rbyte = ggml_sycl_kq_mask_row_bytes(ne0);
    for (int64_t r = 0; r < nrows; ++r) {
        uint32_t *       out = (uint32_t *) (dst + (size_t) r * rbyte);
        const uint16_t * in  = src + (size_t) r * ne0;
        memset(out, 0, rbyte);
        for (int64_t c = 0; c < ne0; ++c) {
            if (in[c] == 0x0000u) {
                out[c >> 5] |= 1u << (c & 31);
            } else if (in[c] != 0xFC00u) {  // f16 -inf
                return false;
            }
        }
    }
    return true;
}

void ggml_sycl_kq_mask_unpack(void * vdst, const void * vsrc, int64_t ne0, int64_t nrows) {
    uint16_t *   dst   = (uint16_t *) vdst;
    const char * src   = (const char *) vsrc;
    const size_t rbyte = ggml_sycl_kq_mask_row_bytes(ne0);
    for (int64_t r = 0; r < nrows; ++r) {
        const uint32_t * in  = (const uint32_t *) (src + (size_t) r * rbyte);
        uint16_t *       out = dst + (size_t) r * ne0;
        for (int64_t c = 0; c < ne0; ++c) {
            out[c] = ((in[c >> 5] >> (c & 31)) & 1u) ? 0x0000u : 0xFC00u;
        }
    }
}

void ggml_sycl_kq_mask_to_f16(const void * vbits, sycl::half * y, int64_t ne0, int64_t nrows,
                              queue_ptr stream) {
    const size_t rbyte  = ggml_sycl_kq_mask_row_bytes(ne0);
    const char * bits   = (const char *) vbits;
    const int64_t total = ne0 * nrows;
    constexpr int WG    = 256;
    const int64_t ngrp  = (total + WG - 1) / WG;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ngrp * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
            const int64_t i = item.get_global_linear_id();
            if (i >= total) {
                return;
            }
            const int64_t r = i / ne0;
            const int64_t c = i - r * ne0;
            const uint32_t * in = (const uint32_t *) (bits + (size_t) r * rbyte);
            y[i] = ((in[c >> 5] >> (c & 31)) & 1u) ? (sycl::half) 0.0f
                                                   : (sycl::half) (-INFINITY);
        });
}

// ---------------------------------------------------------------------------------------------
// taught readers
// ---------------------------------------------------------------------------------------------

static inline bool kq_mask_bit(const char * bits, size_t rbyte, int64_t r, int64_t c) {
    const uint32_t * row = (const uint32_t *) (bits + (size_t) r * rbyte);
    return (row[c >> 5] >> (c & 31)) & 1u;
}

// the tensor `t` covers the whole packed mask m, row for row
static bool kq_mask_rows_match(const ggml_tensor * t, const ggml_tensor * m) {
    return t->ne[0] == m->ne[0] && ggml_nelements(t) == ggml_nelements(m);
}

// a view that is the whole mask in its own row order: same bytes, same rows
static bool kq_mask_whole(const ggml_tensor * v, const ggml_tensor * m) {
    return v->data == m->data && ggml_is_contiguous(v) && kq_mask_rows_match(v, m);
}

static const ggml_tensor * kq_mask_owner(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

template <typename TA, typename TD>
static void kq_mask_add_bits_launch(const char * a, const char * bits, char * d, const int64_t ne[4],
                                    const size_t nba[4], const size_t nbd[4], size_t rbyte, queue_ptr stream) {
    const int64_t ne0 = ne[0], ne1 = ne[1], ne2 = ne[2];
    const int64_t nrows = ne[1] * ne[2] * ne[3];
    const size_t  a1 = nba[1], a2 = nba[2], a3 = nba[3];
    const size_t  d1 = nbd[1], d2 = nbd[2], d3 = nbd[3];
    constexpr int WG = 256;
    const int64_t ngx = (ne0 + WG - 1) / WG;
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>(nrows, ngx * WG), sycl::range<2>(1, WG)),
                         [=](sycl::nd_item<2> it) {
                             const int64_t c = it.get_global_id(1);
                             if (c >= ne0) {
                                 return;
                             }
                             const int64_t r  = it.get_global_id(0);
                             const int64_t i1 = r % ne1;
                             const int64_t i2 = (r / ne1) % ne2;
                             const int64_t i3 = r / (ne1 * ne2);
                             const TA x = *(const TA *) (a + i1 * a1 + i2 * a2 + i3 * a3 + c * sizeof(TA));
                             const float m = kq_mask_bit(bits, rbyte, r, c) ? 0.0f : -INFINITY;
                             *(TD *) (d + i1 * d1 + i2 * d2 + i3 * d3 + c * sizeof(TD)) = (TD) ((float) x + m);
                         });
}

static bool kq_mask_add_types_ok(const ggml_tensor * a, const ggml_tensor * dst) {
    const bool ta = a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_F32;
    const bool td = dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32;
    return ta && td && a->nb[0] == ggml_type_size(a->type) && dst->nb[0] == ggml_type_size(dst->type) &&
           ggml_are_same_shape(a, dst);
}

void ggml_sycl_kq_mask_add_bits(ggml_backend_sycl_context & ctx, const ggml_tensor * a, const ggml_tensor * m,
                                ggml_tensor * dst) {
    const ggml_tensor * owner = kq_mask_owner(m);
    GGML_ASSERT(ggml_sycl_kq_mask_is_bits(owner) && kq_mask_whole(m, owner) && kq_mask_rows_match(dst, owner));
    GGML_ASSERT(kq_mask_add_types_ok(a, dst));
    const size_t rbyte = ggml_sycl_kq_mask_row_bytes(owner->ne[0]);
    const char * ad    = (const char *) a->data;
    const char * bits  = (const char *) owner->data;
    char *       dd    = (char *) dst->data;
    queue_ptr    q     = ctx.stream();
    if (a->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        kq_mask_add_bits_launch<sycl::half, sycl::half>(ad, bits, dd, dst->ne, a->nb, dst->nb, rbyte, q);
    } else if (a->type == GGML_TYPE_F16) {
        kq_mask_add_bits_launch<sycl::half, float>(ad, bits, dd, dst->ne, a->nb, dst->nb, rbyte, q);
    } else if (dst->type == GGML_TYPE_F16) {
        kq_mask_add_bits_launch<float, sycl::half>(ad, bits, dd, dst->ne, a->nb, dst->nb, rbyte, q);
    } else {
        kq_mask_add_bits_launch<float, float>(ad, bits, dd, dst->ne, a->nb, dst->nb, rbyte, q);
    }
}

// which operand of an ADD is the packed mask, or -1. With `mask` null the operand has to be
// packed already; graph_optimize passes the mask it is about to decide on instead.
static int kq_mask_add_operand(const ggml_tensor * dst, const ggml_tensor * mask = nullptr) {
    if (dst->op != GGML_OP_ADD || !dst->src[0] || !dst->src[1]) {
        return -1;
    }
    for (int j = 0; j < 2; ++j) {
        const ggml_tensor * m     = dst->src[j];
        const ggml_tensor * owner = kq_mask_owner(m);
        if (mask ? owner != mask : !ggml_sycl_kq_mask_is_bits(owner)) {
            continue;
        }
        const ggml_tensor * a = dst->src[1 - j];
        if (kq_mask_owner(a) == owner || !kq_mask_whole(m, owner) || !kq_mask_rows_match(dst, owner) ||
            !kq_mask_add_types_ok(a, dst)) {
            return -2;  // reads the mask, but not in a shape this kernel covers
        }
        return j;
    }
    return -1;
}

bool ggml_sycl_kq_mask_try_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const int j = kq_mask_add_operand(dst);
    if (j < 0) {
        return false;
    }
    // x + mask == mask + x in float, so the operand order does not change a bit
    ggml_sycl_kq_mask_add_bits(ctx, dst->src[1 - j], dst->src[j], dst);
    ggml_sycl_kq_mask_taught(ctx, dst);
    return true;
}

template <typename TD>
static void kq_mask_unpack_launch(const char * bits, TD * y, int64_t ne0, int64_t nrows, size_t rbyte, queue_ptr stream) {
    constexpr int WG  = 256;
    const int64_t ngx = (ne0 + WG - 1) / WG;
    stream->parallel_for(sycl::nd_range<2>(sycl::range<2>(nrows, ngx * WG), sycl::range<2>(1, WG)),
                         [=](sycl::nd_item<2> it) {
                             const int64_t c = it.get_global_id(1);
                             if (c >= ne0) {
                                 return;
                             }
                             const int64_t r = it.get_global_id(0);
                             y[r * ne0 + c]  = (TD) (kq_mask_bit(bits, rbyte, r, c) ? 0.0f : -INFINITY);
                         });
}

// the CPY dst when `src` is the whole packed mask and the copy is one this kernel covers
static const ggml_tensor * kq_mask_cpy_dst(const ggml_tensor * src, const ggml_tensor * dst,
                                           const ggml_tensor * mask = nullptr) {
    const ggml_tensor * owner = kq_mask_owner(src);
    if (mask ? owner != mask : !ggml_sycl_kq_mask_is_bits(owner)) {
        return nullptr;
    }
    const bool td = dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32;
    if (!td || !kq_mask_whole(src, owner) || !ggml_is_contiguous(dst) || !kq_mask_rows_match(dst, owner) ||
        kq_mask_owner(dst) == owner) {
        return nullptr;
    }
    return owner;
}

bool ggml_sycl_kq_mask_try_cpy(ggml_backend_sycl_context & ctx, ggml_tensor * node) {
    if (node->op != GGML_OP_CPY || !node->src[0]) {
        return false;
    }
    ggml_tensor *       dst   = node->src[1] ? node->src[1] : node;
    const ggml_tensor * owner = kq_mask_cpy_dst(node->src[0], dst);
    if (!owner) {
        return false;
    }
    const size_t  rbyte = ggml_sycl_kq_mask_row_bytes(owner->ne[0]);
    const int64_t nrows = ggml_nelements(owner) / owner->ne[0];
    if (dst->type == GGML_TYPE_F16) {
        kq_mask_unpack_launch<sycl::half>((const char *) owner->data, (sycl::half *) dst->data, owner->ne[0], nrows,
                                          rbyte, ctx.stream());
    } else {
        kq_mask_unpack_launch<float>((const char *) owner->data, (float *) dst->data, owner->ne[0], nrows, rbyte,
                                     ctx.stream());
    }
    ggml_sycl_kq_mask_taught(ctx, node);
    return true;
}

void ggml_sycl_kq_mask_taught(ggml_backend_sycl_context & ctx, const ggml_tensor * reader) {
    ctx.kq_mask_taught.push_back(reader);
}

// ---------------------------------------------------------------------------------------------
// the tail: scratch lent out of the mask's own allocation
// ---------------------------------------------------------------------------------------------

static int kq_mask_debug() {
    static const int v = ggml_sycl_get_env("GGML_SYCL_KQ_MASK_DEBUG", 0);
    return v;
}

// An LD_PRELOAD overlap probe may define this to see every block the tail hands out.
extern "C" __attribute__((weak)) void ggml_sycl_kq_mask_tail_probe(int device, const ggml_tensor * mask,
                                                                   const ggml_tensor * node, const void * ptr,
                                                                   size_t size);

bool ggml_sycl_kq_mask_tail_on() {
    return (g_ggml_sycl_kq_mask_bits & GGML_SYCL_KQ_MASK_TAIL) &&
           !(g_ggml_sycl_kq_mask_bits & GGML_SYCL_KQ_MASK_COMPACT) && !g_ggml_sycl_enable_graph;
}

static std::atomic<bool> g_kq_mask_compact_veto{ false };

bool ggml_sycl_kq_mask_compact_on() {
    return (g_ggml_sycl_kq_mask_bits & GGML_SYCL_KQ_MASK_COMPACT) && !g_kq_mask_compact_veto.load();
}

void ggml_sycl_kq_mask_compact_veto(const char * why, const char * name) {
    if (!g_kq_mask_compact_veto.exchange(true) && (g_ggml_sycl_kq_mask_bits & GGML_SYCL_KQ_MASK_COMPACT)) {
        GGML_LOG_WARN("%s: KQ mask %s: %s; compact mask allocation is off from here on\n", __func__, name, why);
    }
}

// Masks graph_optimize has approved for COMPACT, by the name the model gave them. The scheduler's
// per-backend and per-copy duplicates ("<backend>#<name>#<copy>") share it, including the copies a
// pipelined graph never shows graph_optimize. Nothing else is ever compacted: a mask built outside
// the scheduler, as in test-backend-ops, stays f16-sized and packs only when its values allow.
static std::mutex            g_kq_mask_approved_mtx;
static std::set<std::string> g_kq_mask_approved;

static std::string kq_mask_base_name(const ggml_tensor * t) {
    const char * n = t->name;
    const char * a = strchr(n, '#');
    const char * b = strrchr(n, '#');
    return a && b > a ? std::string(a + 1, b) : std::string(n);
}

static bool kq_mask_compact_applies(const ggml_tensor * t) {
    if (!ggml_sycl_kq_mask_compact_on() || !ggml_sycl_kq_mask_eligible(t)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_kq_mask_approved_mtx);
    return g_kq_mask_approved.count(kq_mask_base_name(t)) != 0;
}

bool ggml_sycl_kq_mask_is_compact(const ggml_tensor * t) {
    const ggml_tensor * owner = kq_mask_owner(t);
    return owner && ggml_sycl_kq_mask_is_bits(owner) && ggml_compacted_nbytes(owner) < ggml_nbytes(owner);
}

// largest eligible mask per device; every eligible mask is given this slot
static std::mutex g_kq_mask_slot_mtx;
static size_t     g_kq_mask_slot[GGML_SYCL_MAX_DEVICES] = {};

size_t ggml_sycl_kq_mask_alloc_size(int device, const ggml_tensor * t, size_t size) {
    // a marked mask keeps the size it was allocated with
    if (ggml_sycl_kq_mask_is_bits(t) && t->view_src == nullptr) {
        const size_t n = static_cast<const ggml_tensor_extra_gpu *>(t->extra)->optimized_feature.layout.nbytes;
        if (n != 0) {
            return n;
        }
    }
    if (kq_mask_compact_applies(t)) {
        return kq_mask_packed_bytes(t);
    }
    if (!ggml_sycl_kq_mask_tail_on() || device < 0 || device >= GGML_SYCL_MAX_DEVICES ||
        !ggml_sycl_kq_mask_eligible(t)) {
        return size;
    }
    std::lock_guard<std::mutex> lock(g_kq_mask_slot_mtx);
    g_kq_mask_slot[device] = std::max(g_kq_mask_slot[device], size);
    return g_kq_mask_slot[device];
}

ggml_tensor * ggml_sycl_kq_mask_read_by(const ggml_cgraph * cgraph) {
    if (!ggml_sycl_kq_mask_tail_on()) {
        return nullptr;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * s = cgraph->nodes[i]->src[j];
            if (s && ggml_sycl_kq_mask_eligible(kq_mask_owner(s))) {
                return const_cast<ggml_tensor *>(kq_mask_owner(s));
            }
        }
    }
    return nullptr;
}

struct kq_mask_tail_pool : public ggml_sycl_pool {
    static constexpr size_t ALIGN = 256;

    struct block {
        char * p;
        size_t size;
        bool   live;
    };

    ggml_sycl_pool *    inner  = nullptr;
    int                 device = -1;
    const ggml_tensor * mask   = nullptr;
    const ggml_tensor * node   = nullptr;  // the node running now, for the probe
    char *              lo     = nullptr;  // empty range while no graph lends it
    char *              hi     = nullptr;
    std::vector<block>  stack;

    // per armed graph, for the debug report
    size_t n_served = 0, served = 0, n_declined = 0, declined_max = 0, top_hw = 0;

    void * alloc(size_t size, size_t * actual_size) override {
        if (lo && size > 0) {
            char *       top = stack.empty() ? lo : stack.back().p + stack.back().size;
            const size_t a   = GGML_PAD(size, ALIGN);
            if (a <= (size_t) (hi - top)) {
                stack.push_back({ top, a, true });
                *actual_size = a;
                n_served++;
                served += a;
                top_hw = std::max(top_hw, (size_t) (top + a - lo));
                if (ggml_sycl_kq_mask_tail_probe) {
                    ggml_sycl_kq_mask_tail_probe(device, mask, node, top, a);
                }
                return top;
            }
            n_declined++;
            declined_max = std::max(declined_max, size);
        }
        return inner->alloc(size, actual_size);
    }

    void free(void * ptr, size_t size) override {
        for (size_t k = stack.size(); k-- > 0;) {
            if (stack[k].p == ptr && stack[k].live) {
                stack[k].live = false;
                while (!stack.empty() && !stack.back().live) {
                    stack.pop_back();
                }
                return;
            }
        }
        inner->free(ptr, size);
    }
};

// [lo, hi) of t, counting what the allocator reserved past its bytes
static bool kq_mask_overlaps(const ggml_tensor * t, const char * lo, const char * hi) {
    if (!t || !t->data) {
        return false;
    }
    size_t n = ggml_nbytes(t);
    if (!t->view_src && t->buffer) {
        n = std::max(n, ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(t->buffer), t));
    }
    const char * a = (const char *) t->data;
    return a < hi && lo < a + n;
}

static bool kq_mask_is_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// Readers the backend runs from the packed bits. Anything else gets a dense copy.
static bool kq_mask_reader_taught(const ggml_tensor * n, int j, const ggml_tensor * owner) {
    switch (n->op) {
        case GGML_OP_ADD:
            return kq_mask_add_operand(n, owner) == j;
        case GGML_OP_CPY:
            // a cast: the fused QSA top-k, the fused cast+add or the standalone copy reads it
            return j == 0 && kq_mask_cpy_dst(n->src[0], n->src[1] ? n->src[1] : n, owner) == owner;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                // fattn.cpp reads the bits or expands them itself, from the whole mask only. ALiBi
                // scales the mask by a slope, which a bit does not carry
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) n->op_params + 1, sizeof(float));
                return j == 3 && max_bias == 0.0f && kq_mask_whole(n->src[3], owner);
            }
        default:
            return false;
    }
}

// Why `owner` cannot be read from bits in this graph, or null; `who` is the reader at fault.
// `readers` collects the taught readers when given.
static const char * kq_mask_untaught_reader(const ggml_cgraph * cgraph, const ggml_tensor * owner,
                                            std::vector<const ggml_tensor *> * readers, const ggml_tensor ** who) {
    bool has_cast = false;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (kq_mask_is_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!n->src[j] || kq_mask_owner(n->src[j]) != owner) {
                continue;
            }
            if (n->op == GGML_OP_FILL && j == 0) {
                // only its shape is read; the QSA chain fusions that start here read the mask
                if ((g_ggml_sycl_fuse_qsa_mask || g_ggml_sycl_fuse_qsa_fa_mask) &&
                    ggml_sycl_qsa_mask_absorbs(cgraph, i) > 0) {
                    *who = n;
                    return "QSA mask fusion";
                }
                continue;
            }
            if (!kq_mask_reader_taught(n, j, owner)) {
                *who = n;
                return "untaught op";
            }
            has_cast |= n->op == GGML_OP_CPY;
            if (readers) {
                readers->push_back(n);
            }
        }
    }
    // the CONT+ADD fusion reads a cast's f16 source itself
    if (has_cast && g_ggml_sycl_fuse_cont_add) {
        *who = nullptr;
        return "CONT+ADD fusion";
    }
    return nullptr;
}

void ggml_sycl_kq_mask_compact_check(const ggml_cgraph * cgraph) {
    if (!ggml_sycl_kq_mask_compact_on()) {
        return;
    }
    // One bit holds 0 or -inf and nothing else. llama writes other values only for ALiBi, which
    // every attention op then announces with a non-zero max_bias, whichever mask it reads.
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_FLASH_ATTN_EXT && n->op != GGML_OP_SOFT_MAX) {
            continue;
        }
        float max_bias = 0.0f;
        memcpy(&max_bias, (const float *) n->op_params + 1, sizeof(float));
        if (max_bias != 0.0f) {
            ggml_sycl_kq_mask_compact_veto("ALiBi (max_bias != 0) makes the mask more than 0/-inf", n->name);
            return;
        }
    }
    std::vector<const ggml_tensor *> masks;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * o = cgraph->nodes[i]->src[j] ? kq_mask_owner(cgraph->nodes[i]->src[j]) : nullptr;
            if (o && ggml_sycl_kq_mask_eligible(o) && std::find(masks.begin(), masks.end(), o) == masks.end()) {
                masks.push_back(o);
            }
        }
    }
    for (const ggml_tensor * owner : masks) {
        const ggml_tensor * who = nullptr;
        if (const char * why = kq_mask_untaught_reader(cgraph, owner, nullptr, &who)) {
            const std::string msg = std::string(why) + " " + (who ? who->name : "") + " (" +
                                    (who ? ggml_op_desc(who) : "") + ") reads it";
            ggml_sycl_kq_mask_compact_veto(msg.c_str(), owner->name);
            return;
        }
    }
    std::lock_guard<std::mutex> lock(g_kq_mask_approved_mtx);
    for (const ggml_tensor * owner : masks) {
        g_kq_mask_approved.insert(kq_mask_base_name(owner));
    }
}

// Expand the mask to dense f16 in its own allocation, for a graph with an untaught reader.
static void kq_mask_unpack_in_place(ggml_backend_sycl_context & ctx, const ggml_tensor * owner) {
    const int64_t ne0   = owner->ne[0];
    const int64_t nrows = ggml_nelements(owner) / ne0;
    const size_t  pb    = ggml_sycl_kq_mask_row_bytes(ne0) * nrows;
    queue_ptr     q     = ctx.stream();

    ggml_sycl_pool_alloc<char> tmp(ctx.pool(), pb);
    SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(tmp.get(), owner->data, pb)));
    ggml_sycl_kq_mask_to_f16(tmp.get(), (sycl::half *) owner->data, ne0, nrows, q);

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(owner->extra);
    extra->optimized_feature.layout.kind = GGML_SYCL_LAYOUT_CANONICAL;
}

void ggml_sycl_kq_mask_graph_begin(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph,
                                   ggml_sycl_kq_mask_graph & g) {
    g         = {};
    g.ctx     = &ctx;
    ctx.kq_mask_taught.clear();
    if (!g_ggml_sycl_kq_mask_bits) {
        return;
    }

    // the packed masks this graph reads
    std::vector<const ggml_tensor *> masks;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * s = cgraph->nodes[i]->src[j];
            const ggml_tensor * o = s ? kq_mask_owner(s) : nullptr;
            if (o && ggml_sycl_kq_mask_is_bits(o) && std::find(masks.begin(), masks.end(), o) == masks.end()) {
                masks.push_back(o);
            }
        }
    }
    if (masks.empty()) {
        return;
    }

    for (const ggml_tensor * owner : masks) {
        const ggml_tensor *              who = nullptr;
        std::vector<const ggml_tensor *> readers;
        const char *                     why = kq_mask_untaught_reader(cgraph, owner, &readers, &who);
        if (why && ggml_sycl_kq_mask_is_compact(owner)) {
            // graph_optimize runs the same check before allocating, so this cannot happen
            GGML_ABORT("compact KQ mask %s: %s %s (%s) cannot read it, and it has no room to expand", owner->name,
                       why, who ? who->name : "", who ? ggml_op_desc(who) : "");
        }
        if (why) {
            static std::once_flag warned;
            std::call_once(warned, [&] {
                GGML_LOG_WARN("%s: %s %s (%s) reads the packed KQ mask; expanding it for this graph\n", __func__,
                              why, who ? who->name : "", who ? ggml_op_desc(who) : "");
            });
            if (kq_mask_debug()) {
                GGML_LOG_INFO("[KQ-MASK] dev %d: %s %s (%s): mask %s expanded\n", ctx.device, why,
                              who ? who->name : "", who ? ggml_op_desc(who) : "", owner->name);
            }
            kq_mask_unpack_in_place(ctx, owner);
            continue;
        }
        // GGML_SYCL_KQ_MASK_DEBUG=3 lists the readers of every graph, 1 and 2 of the first per device
        static bool listed[GGML_SYCL_MAX_DEVICES] = {};
        if (kq_mask_debug() >= 3 || (kq_mask_debug() && !listed[ctx.device] && owner->ne[1] > 1)) {
            listed[ctx.device] = true;
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * n = cgraph->nodes[i];
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    if (n->src[j] && kq_mask_owner(n->src[j]) == owner) {
                        const bool r = std::find(readers.begin(), readers.end(), n) != readers.end();
                        GGML_LOG_INFO("[KQ-MASK] dev %d mask %s ne=[%lld,%lld]: #%d %s %s src[%d] %s\n", ctx.device,
                                      owner->name, (long long) owner->ne[0], (long long) owner->ne[1], i,
                                      ggml_op_desc(n), n->name, j,
                                      r ? "reader" : kq_mask_is_noop(n) ? "view" : n->op == GGML_OP_FILL ? "shape only" : "not run");
                    }
                }
            }
        }
        g.readers.insert(g.readers.end(), readers.begin(), readers.end());
        if (!g.mask) {
            g.mask = owner;
        } else {
            g.mask = nullptr;  // two packed masks: lend neither
        }
    }

    if (!g.mask || !ggml_sycl_kq_mask_tail_on() || !g.mask->buffer ||
        ggml_backend_buffer_get_usage(g.mask->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return;
    }

    const ggml_tensor * m     = g.mask;
    const int64_t       nrows = ggml_nelements(m) / m->ne[0];
    const size_t        slot  = ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(m->buffer), m);
    char *              base  = (char *) m->data;
    char * lo = base + GGML_PAD(ggml_sycl_kq_mask_row_bytes(m->ne[0]) * nrows, kq_mask_tail_pool::ALIGN);
    char * hi = base + (slot / kq_mask_tail_pool::ALIGN) * kq_mask_tail_pool::ALIGN;
    if (hi <= lo || (size_t) (hi - lo) < (1u << 20)) {
        return;
    }

    // the keep-alive from graph_optimize means nothing else of this split sits in the slot
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        for (int j = -1; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * t = j < 0 ? n : n->src[j];
            if (t && kq_mask_owner(t) != m && kq_mask_overlaps(t, lo, hi)) {
                static std::once_flag warned;
                std::call_once(warned, [&] {
                    GGML_LOG_WARN("%s: %s overlaps the KQ mask tail; not lending it\n", __func__, t->name);
                });
                if (kq_mask_debug()) {
                    GGML_LOG_INFO("[KQ-MASK] dev %d: node #%d %s overlaps the tail, not lent\n", ctx.device, i,
                                  t->name);
                }
                return;
            }
        }
    }

    if (!ctx.kq_mask_tail) {
        auto * tp   = new kq_mask_tail_pool();
        tp->inner   = &ctx.pool(ctx.device);
        tp->device  = ctx.device;
        ctx.kq_mask_tail.reset(tp);
    }
    auto * tp = static_cast<kq_mask_tail_pool *>(ctx.kq_mask_tail.get());
    GGML_ASSERT(tp->stack.empty());
    tp->mask       = m;
    tp->lo         = lo;
    tp->hi         = hi;
    tp->n_served   = tp->served = tp->n_declined = tp->declined_max = tp->top_hw = 0;
    ctx.pool_override = tp;
    g.tail            = true;
}

void ggml_sycl_kq_mask_graph_at(ggml_sycl_kq_mask_graph & g, const ggml_tensor * node) {
    if (g.tail && ggml_sycl_kq_mask_tail_probe) {
        static_cast<kq_mask_tail_pool *>(g.ctx->kq_mask_tail.get())->node = node;
    }
}

void ggml_sycl_kq_mask_graph_end(ggml_sycl_kq_mask_graph & g) {
    if (!g.ctx) {
        return;
    }
    ggml_backend_sycl_context & ctx = *g.ctx;
    if (g.tail) {
        ctx.pool_override = nullptr;
        auto * tp         = static_cast<kq_mask_tail_pool *>(ctx.kq_mask_tail.get());
        if (!tp->stack.empty()) {
            GGML_ABORT("KQ mask tail: %zu block(s) still lent at the end of the split", tp->stack.size());
        }
        if (kq_mask_debug() >= 2 || (kq_mask_debug() && tp->served)) {
            GGML_LOG_INFO("[KQ-MASK] dev %d mask %s ne=[%lld,%lld] tail %.2f MiB: served %zu (%.2f MiB), hw %.2f MiB, "
                          "declined %zu (max %.2f MiB)\n",
                          ctx.device, g.mask->name, (long long) g.mask->ne[0], (long long) g.mask->ne[1],
                          (tp->hi - tp->lo) / 1048576.0, tp->n_served, tp->served / 1048576.0, tp->top_hw / 1048576.0,
                          tp->n_declined, tp->declined_max / 1048576.0);
        }
        tp->lo   = nullptr;
        tp->hi   = nullptr;
        tp->mask = nullptr;
        tp->node = nullptr;
    }
    for (const ggml_tensor * r : g.readers) {
        if (std::find(ctx.kq_mask_taught.begin(), ctx.kq_mask_taught.end(), r) == ctx.kq_mask_taught.end()) {
            GGML_ABORT("packed KQ mask: %s (%s) ran without a taught kernel", r->name, ggml_op_desc(r));
        }
    }
    ctx.kq_mask_taught.clear();
    g = {};
}
