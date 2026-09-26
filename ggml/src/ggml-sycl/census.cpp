#include "census.hpp"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

int g_ggml_sycl_census = 0;

static constexpr double MIB = 1024.0 * 1024.0;

// the head node of the group that runs now, so a pool request can name who made it
static thread_local const ggml_tensor * t_census_node = nullptr;

// ---------------------------------------------------------------------------------------------
// pool
// ---------------------------------------------------------------------------------------------

struct census_pool_site {
    size_t n           = 0;
    size_t req_max     = 0;
    size_t fresh_n     = 0;
    size_t fresh_bytes = 0;
};

struct census_pool_dev {
    size_t in_use    = 0;
    size_t in_use_hw = 0;  // since start
    size_t graph_hw  = 0;  // since the last reported graph began
    size_t reserved  = 0;  // what the pool holds from the driver, cached or not
    std::map<std::string, census_pool_site> sites;
};

static std::mutex      g_census_pool_mtx;
static census_pool_dev g_census_pool[GGML_SYCL_MAX_DEVICES][2];

// "ffn_moe_down-17" and "ffn_moe_down-3" are the same call site
static std::string census_stem(const ggml_tensor * t) {
    std::string s = t->name;
    const size_t dash = s.rfind('-');
    if (dash != std::string::npos && dash + 1 < s.size() &&
        s.find_first_not_of("0123456789", dash + 1) == std::string::npos) {
        s.resize(dash);
    }
    if (s.rfind("node_", 0) == 0 || s.rfind("leaf_", 0) == 0) {
        s = s.substr(0, 4);
    }
    return s;
}

static std::string census_label(const ggml_tensor * t) {
    if (t == nullptr) {
        return "<outside graph>";
    }
    return std::string(ggml_op_desc(t)) + " " + census_stem(t);
}

void ggml_sycl_census_pool_alloc(int device, bool host, size_t req, size_t actual, bool fresh, size_t pool_size) {
    if (!g_ggml_sycl_census || device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_census_pool_mtx);
    census_pool_dev & p = g_census_pool[device][host ? 1 : 0];
    p.in_use += actual;
    p.in_use_hw = std::max(p.in_use_hw, p.in_use);
    p.graph_hw  = std::max(p.graph_hw, p.in_use);
    p.reserved  = pool_size;
    census_pool_site & s = p.sites[census_label(t_census_node)];
    s.n++;
    s.req_max = std::max(s.req_max, req);
    if (fresh) {
        s.fresh_n++;
        s.fresh_bytes += actual;
        fprintf(stderr, "[CENSUS-POOL] dev %d %s grows by %.2f MiB (req %.2f) to %.2f MiB, in use %.2f MiB, by %s\n",
                device, host ? "host" : "device", actual / MIB, req / MIB, pool_size / MIB, p.in_use / MIB,
                census_label(t_census_node).c_str());
    }
}

void ggml_sycl_census_pool_free(int device, bool host, size_t size) {
    if (!g_ggml_sycl_census || device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_census_pool_mtx);
    census_pool_dev & p = g_census_pool[device][host ? 1 : 0];
    p.in_use -= std::min(p.in_use, size);
}

// ---------------------------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------------------------

struct census_node {
    bool     visited  = false;  // the loop reached it as a group head or as a view
    bool     view     = false;
    int      head     = -1;     // group it ran in
    int      absorbed = -1;     // node whose fusion_absorbs answer covers it
    bool     pre_ok   = false;
    uint64_t pre      = 0;
    int      written  = -1;     // -1 no memory or no verdict, 0 unchanged, 1 changed
};

struct census_base {
    const ggml_tensor * t        = nullptr;
    bool                produced = false;  // some node of this split has it as its base
    bool                input    = false;  // referenced here but produced elsewhere
    int                 def      = INT_MAX;
    int                 last     = -1;
    bool                written  = false;
    bool                wr_known = false;
    bool                read     = false;
    bool                shape_only_ref = false;
    bool                aliased        = false;  // a later node of its fused group writes over it
    std::string         readers;
};

struct ggml_sycl_census_graph {
    ggml_backend_sycl_context *              ctx = nullptr;
    const ggml_cgraph *                      g   = nullptr;
    std::vector<census_node>                 nodes;
    std::unordered_map<const ggml_tensor *, census_base> bases;
    int                                      cur_head = -1;
    unsigned long long *                     d_hash   = nullptr;
    int                                      seq      = 0;
};

static constexpr int CENSUS_WINDOW = 48;

static const ggml_tensor * census_base_of(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

static bool census_in_compute(const ggml_tensor * t) {
    return t && t->data && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE;
}

static bool census_is_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
}

// a source whose bytes the op never reads, only its shape
static bool census_shape_only(const ggml_tensor * node, int j) {
    return node->op == GGML_OP_FILL && j == 0;
}

static void census_hash_launch(sycl::queue & q, const void * p, size_t nbytes, unsigned long long * out) {
    const uint32_t * w = (const uint32_t *) p;
    const size_t     n = nbytes / 4;
    if (n == 0) {
        return;
    }
    constexpr int WG     = 256;
    const size_t  groups = std::min<size_t>((n + WG * 16 - 1) / (WG * 16), 4096);
    q.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        unsigned long long acc = 0;
        for (size_t i = it.get_global_id(0); i < n; i += it.get_global_range(0)) {
            unsigned long long x = (unsigned long long) w[i] ^ ((unsigned long long) i * 0x9E3779B97F4A7C15ull);
            x *= 0xff51afd7ed558ccdull;
            x ^= x >> 33;
            acc += x;
        }
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<unsigned long long>());
        if (it.get_local_id(0) == 0) {
            sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                r(*out);
            r.fetch_add(acc);
        }
    });
}

// hash the dst of every listed node that owns memory in the compute buffer
static void census_hash_nodes(ggml_sycl_census_graph * cg, const std::vector<int> & idx, std::vector<uint64_t> & out) {
    sycl::queue & q = *cg->ctx->stream();
    out.assign(idx.size(), 0);
    if (idx.empty()) {
        return;
    }
    q.memset(cg->d_hash, 0, idx.size() * sizeof(unsigned long long));
    for (size_t k = 0; k < idx.size(); ++k) {
        const ggml_tensor * t = cg->g->nodes[idx[k]];
        census_hash_launch(q, t->data, ggml_nbytes(t), cg->d_hash + k);
    }
    std::vector<unsigned long long> h(idx.size());
    q.memcpy(h.data(), cg->d_hash, idx.size() * sizeof(unsigned long long));
    q.wait_and_throw();
    for (size_t k = 0; k < idx.size(); ++k) {
        out[k] = h[k];
    }
}

static bool census_hashable(const ggml_tensor * t) {
    return !census_is_noop(t) && census_in_compute(t) && ggml_nbytes(t) >= 4;
}

ggml_sycl_census_graph * ggml_sycl_census_begin(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph,
                                                const std::vector<int> & absorbed_by) {
    if (!g_ggml_sycl_census) {
        return nullptr;
    }
    static const int min_tokens = ggml_sycl_get_env("GGML_SYCL_CENSUS_MIN_TOKENS", 512);
    static int       reported[GGML_SYCL_MAX_DEVICES] = {};

    // tokens in the ubatch: the widest weight GEMM
    int64_t widest = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op == GGML_OP_MUL_MAT && n->src[0] && n->src[0]->buffer &&
            ggml_backend_buffer_get_usage(n->src[0]->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            widest = std::max(widest, n->ne[1]);
        }
    }
    if (widest < min_tokens || reported[ctx.device] >= g_ggml_sycl_census) {
        return nullptr;
    }
    reported[ctx.device]++;

    auto * cg  = new ggml_sycl_census_graph();
    cg->ctx    = &ctx;
    cg->g      = cgraph;
    cg->seq    = reported[ctx.device];
    cg->nodes.resize(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        cg->nodes[i].absorbed = absorbed_by[i];
    }
    cg->d_hash = sycl::malloc_device<unsigned long long>(CENSUS_WINDOW + 1, *ctx.stream());
    GGML_ASSERT(cg->d_hash);

    {
        std::lock_guard<std::mutex> lock(g_census_pool_mtx);
        g_census_pool[ctx.device][0].graph_hw = g_census_pool[ctx.device][0].in_use;
    }

    ctx.stream()->wait_and_throw();
    return cg;
}

static void census_close_group(ggml_sycl_census_graph * cg, int h, int e) {
    const ggml_cgraph * g = cg->g;

    std::vector<int> idx;
    for (int j = h; j <= e; ++j) {
        if (cg->nodes[j].pre_ok && census_hashable(g->nodes[j])) {
            idx.push_back(j);
        }
    }
    std::vector<uint64_t> post;
    census_hash_nodes(cg, idx, post);
    for (size_t k = 0; k < idx.size(); ++k) {
        cg->nodes[idx[k]].written = post[k] != cg->nodes[idx[k]].pre ? 1 : 0;
    }

    for (int j = h; j <= e; ++j) {
        cg->nodes[j].head = h;
    }

}

void ggml_sycl_census_visit(ggml_sycl_census_graph * cg, int i) {
    if (!cg) {
        return;
    }
    const ggml_cgraph * g = cg->g;
    if (cg->cur_head >= 0) {
        census_close_group(cg, cg->cur_head, i - 1);
        cg->cur_head = -1;
    }
    if (i >= g->n_nodes) {
        t_census_node = nullptr;
        return;
    }
    census_node & cn = cg->nodes[i];
    cn.visited       = true;
    if (census_is_noop(g->nodes[i])) {
        cn.view = true;
        return;
    }
    cg->cur_head  = i;
    t_census_node = g->nodes[i];

    std::vector<int> idx;
    for (int j = i; j < g->n_nodes && j < i + CENSUS_WINDOW; ++j) {
        if (census_hashable(g->nodes[j])) {
            idx.push_back(j);
        }
    }
    std::vector<uint64_t> pre;
    census_hash_nodes(cg, idx, pre);
    for (size_t k = 0; k < idx.size(); ++k) {
        cg->nodes[idx[k]].pre    = pre[k];
        cg->nodes[idx[k]].pre_ok = true;
    }
}

// collect every compute-buffer tensor that the split references, with its lifetime in node order
static void census_collect(ggml_sycl_census_graph * cg) {
    const ggml_cgraph * g = cg->g;
    auto touch = [&](const ggml_tensor * t, int i, bool def) {
        const ggml_tensor * b = census_base_of(t);
        if (!census_in_compute(b)) {
            return;
        }
        census_base & cb = cg->bases[b];
        cb.t = b;
        if (def) {
            cb.produced = true;
            cb.def      = std::min(cb.def, i);
        }
        cb.last = std::max(cb.last, i);
    };
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * n = g->nodes[i];
        if (census_base_of(n) == n) {
            touch(n, i, true);
        } else {
            touch(n, i, false);
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s]) {
                touch(n->src[s], i, false);
            }
        }
    }
    for (auto & kv : cg->bases) {
        if (!kv.second.produced) {
            kv.second.input = true;
            kv.second.def   = -1;
        }
    }
}

static const char * census_verdict(const census_base & b) {
    if (b.input) {
        return b.read ? "INPUT-READ" : "INPUT-UNREAD(dead)";
    }
    if (!b.wr_known) {
        return b.read ? "READ,WRITE?" : "UNREAD,WRITE?";
    }
    if (b.written && b.read) {
        return "USED";
    }
    if (b.aliased) {
        return "FUSED-INPLACE";
    }
    if (b.written) {
        return (b.t->flags & GGML_TENSOR_FLAG_OUTPUT) ? "OUTPUT" : "WRITE-ONLY";
    }
    return b.read ? "READ-UNWRITTEN(!)" : "UNTOUCHED(dead)";
}

void ggml_sycl_census_end(ggml_sycl_census_graph * cg) {
    if (!cg) {
        return;
    }
    const ggml_cgraph * g = cg->g;
    census_collect(cg);
    // the census visit of the node past the end closes the last group
    ggml_sycl_census_visit(cg, g->n_nodes);

    // reads are structural: the sources of each executed group that the group did not produce
    for (int i = 0; i < g->n_nodes;) {
        if (cg->nodes[i].head != i) {
            ++i;
            continue;
        }
        int e = i;
        while (e + 1 < g->n_nodes && cg->nodes[e + 1].head == i) {
            ++e;
        }
        // reads only; no hashing
        for (int j = i; j <= e; ++j) {
            const ggml_tensor * n = g->nodes[j];
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = n->src[s];
                if (!src) {
                    continue;
                }
                const ggml_tensor * b  = census_base_of(src);
                auto                it = cg->bases.find(b);
                if (it == cg->bases.end()) {
                    continue;
                }
                if (census_shape_only(n, s)) {
                    it->second.shape_only_ref = true;
                    continue;
                }
                bool internal = false;
                for (int k = i; k < j && !internal; ++k) {
                    internal = census_base_of(g->nodes[k]) == b && !cg->nodes[k].view;
                }
                if (internal) {
                    continue;
                }
                if (!it->second.read) {
                    it->second.readers = std::string(ggml_op_desc(g->nodes[i])) + " " + g->nodes[i]->name;
                }
                it->second.read = true;
                // overlap probe: a multi-node group must not write over what it still reads
                if (e > i) {
                    const char * rlo = (const char *) b->data;
                    const char * rhi = rlo + ggml_nbytes(b);
                    for (int k = i; k <= e; ++k) {
                        const ggml_tensor * w = g->nodes[k];
                        if (!census_in_compute(w) || cg->nodes[k].written != 1 || census_base_of(w) == b) {
                            continue;
                        }
                        const char * wlo = (const char *) w->data;
                        if (wlo < rhi && rlo < wlo + ggml_nbytes(w)) {
                            fprintf(stderr, "[CENSUS-OVERLAP] dev %d group #%d %s: writes %s over its input %s%s\n",
                                    cg->ctx->device, i, g->nodes[i]->name, w->name, b->name,
                                    wlo == rlo ? " (same base: in place)" : "");
                        }
                    }
                }
            }
        }
        i = e + 1;
    }
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * b  = census_base_of(g->nodes[i]);
        auto                it = cg->bases.find(b);
        if (it == cg->bases.end() || cg->nodes[i].written < 0) {
            continue;
        }
        it->second.wr_known = true;
        it->second.written |= cg->nodes[i].written == 1;
    }

    // a fused-away node that ggml-alloc placed in place under a later node of the same group
    // shares memory with the group's output, so it costs nothing of its own
    for (int i = 0; i < g->n_nodes; ++i) {
        const census_node & cn = cg->nodes[i];
        const ggml_tensor * n  = g->nodes[i];
        if (cn.view || cn.head < 0 || cn.head == i || census_base_of(n) != n) {
            continue;
        }
        auto it = cg->bases.find(n);
        if (it == cg->bases.end()) {
            continue;
        }
        const char * lo = (const char *) n->data;
        const char * hi = lo + ggml_nbytes(n);
        for (int k = i + 1; k < g->n_nodes && cg->nodes[k].head == cn.head; ++k) {
            const ggml_tensor * m = g->nodes[k];
            if (!census_in_compute(m) || cg->nodes[k].written != 1 || census_base_of(m) == n) {
                continue;
            }
            const char * mlo = (const char *) m->data;
            if (mlo < hi && lo < mlo + ggml_nbytes(m)) {
                it->second.aliased = true;
            }
        }
    }

    const int dev = cg->ctx->device;
    int n_exec = 0, n_fused = 0, n_abs = 0, n_view = 0;
    for (int i = 0; i < g->n_nodes; ++i) {
        const census_node & cn = cg->nodes[i];
        if (cn.absorbed >= 0) {
            n_abs++;
        }
        if (cn.view) {
            n_view++;
        } else if (cn.head == i) {
            n_exec++;
        } else if (!census_is_noop(g->nodes[i])) {
            n_fused++;
        }
    }
    fprintf(stderr, "[CENSUS] dev %d graph %d: %d nodes, %d executed heads, %d fused-away, %d absorbed, %d views/noops\n",
            dev, cg->seq, g->n_nodes, n_exec, n_fused, n_abs, n_view);

    // node table: fused-away and absorbed nodes, which are the candidates; first graph only
    for (int i = 0; i < g->n_nodes && cg->seq == 1; ++i) {
        const census_node & cn = cg->nodes[i];
        const ggml_tensor * n  = g->nodes[i];
        const bool fused = !cn.view && cn.head != i && !census_is_noop(n);
        if (!fused && cn.absorbed < 0) {
            continue;
        }
        const bool mem = census_in_compute(n) && census_base_of(n) == n;
        const double off = mem ? ((const char *) n->data - (const char *) ggml_backend_buffer_get_base(n->buffer)) / MIB : -1.0;
        fprintf(stderr, "[CENSUS-NODE] dev %d #%d %-12s %-28s %8.2f MiB @%.2f %s%s head=#%d(%s) written=%d\n", dev, i,
                ggml_op_desc(n), n->name, mem ? ggml_nbytes(n) / MIB : 0.0, off, fused ? "FUSED-AWAY" : "EXECUTED",
                cn.absorbed >= 0 ? "+ABSORBED" : "", cn.head, cn.head >= 0 ? g->nodes[cn.head]->name : "-",
                cn.written);
    }

    // tensor table and totals by verdict
    std::map<std::string, std::pair<int, double>> totals;
    std::vector<const census_base *>             order;
    for (auto & kv : cg->bases) {
        order.push_back(&kv.second);
    }
    std::sort(order.begin(), order.end(), [](const census_base * a, const census_base * b) {
        return (const char *) a->t->data < (const char *) b->t->data;
    });
    for (const census_base * b : order) {
        const char * v = census_verdict(*b);
        totals[v].first++;
        totals[v].second += ggml_nbytes(b->t) / MIB;
        const bool interesting = strcmp(v, "USED") != 0 && strcmp(v, "INPUT-READ") != 0;
        if (cg->seq == 1 && (interesting || ggml_nbytes(b->t) >= 32 * 1024 * 1024)) {
            fprintf(stderr, "[CENSUS-TENSOR] dev %d %-20s %-12s %-28s %8.2f MiB def=#%d last=#%d%s first-reader=%s\n",
                    dev, v, ggml_op_desc(b->t), b->t->name, ggml_nbytes(b->t) / MIB, b->def, b->last,
                    b->shape_only_ref ? " shape-only-ref" : "", b->readers.empty() ? "-" : b->readers.c_str());
        }
    }
    for (auto & kv : totals) {
        fprintf(stderr, "[CENSUS-TOTAL] dev %d %-20s %5d tensors %9.2f MiB\n", dev, kv.first.c_str(), kv.second.first,
                kv.second.second);
    }

    // arena top per node from the placements the allocator chose, and the live set at the highest
    const char * lo = nullptr;
    for (const census_base * b : order) {
        const char * base = (const char *) ggml_backend_buffer_get_base(b->t->buffer);
        lo                = lo ? std::min(lo, base) : base;
    }
    std::vector<double> top(g->n_nodes, 0.0);
    for (const census_base * b : order) {
        const char * base = (const char *) ggml_backend_buffer_get_base(b->t->buffer);
        const double end  = ((const char *) b->t->data - base + ggml_nbytes(b->t)) / MIB;
        for (int k = std::max(0, b->def); k <= b->last && k < g->n_nodes; ++k) {
            top[k] = std::max(top[k], end);
        }
    }
    // the three highest local maxima, one per distinct value
    std::vector<int> peaks;
    for (int k = 0; k < g->n_nodes; ++k) {
        const bool rise = k == 0 || top[k] > top[k - 1];
        const bool fall = k + 1 == g->n_nodes || top[k] >= top[k + 1];
        if (rise && fall && top[k] > 0) {
            peaks.push_back(k);
        }
    }
    std::sort(peaks.begin(), peaks.end(), [&](int a, int b) { return top[a] > top[b]; });
    std::vector<double> seen;
    int                 shown = 0;
    for (int k : peaks) {
        bool dup = false;
        for (double s : seen) {
            dup |= std::abs(s - top[k]) < 0.01;
        }
        if (dup) {
            continue;
        }
        seen.push_back(top[k]);
        fprintf(stderr, "[CENSUS-PEAK] dev %d top %.2f MiB at #%d %s %s; live:\n", dev, top[k], k,
                ggml_op_desc(g->nodes[k]), g->nodes[k]->name);
        for (const census_base * b : order) {
            if (b->def > k || b->last < k) {
                continue;
            }
            const char * base = (const char *) ggml_backend_buffer_get_base(b->t->buffer);
            const double off  = ((const char *) b->t->data - base) / MIB;
            if (ggml_nbytes(b->t) < 1024 * 1024) {
                continue;
            }
            fprintf(stderr, "[CENSUS-PEAK]    %8.2f..%8.2f %-20s %-12s %s\n", off, off + ggml_nbytes(b->t) / MIB,
                    census_verdict(*b), ggml_op_desc(b->t), b->t->name);
        }
        if (++shown == (cg->seq == 1 ? 3 : 1)) {
            break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_census_pool_mtx);
        for (int h = 0; h < 2; ++h) {
            const census_pool_dev & p = g_census_pool[dev][h];
            if (p.sites.empty()) {
                continue;
            }
            fprintf(stderr, "[CENSUS-POOL] dev %d %s: reserved %.2f MiB, in use now %.2f, high-water this graph %.2f, overall %.2f MiB\n",
                    dev, h ? "host" : "device", p.reserved / MIB, p.in_use / MIB, p.graph_hw / MIB, p.in_use_hw / MIB);
            for (const auto & s : p.sites) {
                fprintf(stderr, "[CENSUS-POOL]    %-44s n=%-7zu max req %9.2f MiB, grew pool %zu times by %.2f MiB\n",
                        s.first.c_str(), s.second.n, s.second.req_max / MIB, s.second.fresh_n, s.second.fresh_bytes / MIB);
            }
        }
    }

    sycl::free(cg->d_hash, *cg->ctx->stream());
    t_census_node = nullptr;
    delete cg;
}
