// test-sycl-graph-reorder: the SYCL backend's sibling reorder (ggml/src/ggml-sycl/graph-reorder.cpp)
// is pure graph inspection, so it is checked here on the CPU backend without a SYCL device:
//   - siblings that read the same activation end up adjacent to the first of them
//   - a sibling that reads something produced, or written through a view, in between stays put
//   - a pinned node neither moves nor leads
//   - the reordered graph is still topologically valid and computes bit-identical results

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "ggml-impl.h"
#include "ggml-sycl/graph-reorder.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            n_failed++;                                                          \
        }                                                                        \
    } while (0)

static bool leader_mm(const ggml_tensor * cur, void *) {
    return cur->op == GGML_OP_MUL_MAT;
}

static bool joins_mm(const ggml_tensor * leader, const ggml_tensor * cand, void *) {
    return cand->op == GGML_OP_MUL_MAT && cand->src[1] == leader->src[1] && cand->src[0]->type == leader->src[0]->type;
}

static int index_of(const ggml_cgraph * gf, const char * name) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (strcmp(gf->nodes[i]->name, name) == 0) {
            return i;
        }
    }
    return -1;
}

// every source is a leaf or a node that comes earlier
static bool topologically_valid(const ggml_cgraph * gf) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * node = gf->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (!src) {
                continue;
            }
            for (int j = i + 1; j < gf->n_nodes; ++j) {
                if (gf->nodes[j] == src) {
                    fprintf(stderr, "node %s reads %s, which now comes later\n", node->name, src->name);
                    return false;
                }
            }
        }
    }
    return true;
}

struct model {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;
    std::vector<ggml_tensor *> outputs;
    ggml_tensor * x     = nullptr;
    ggml_tensor * cache = nullptr;
    std::vector<ggml_tensor *> weights;
};

// Shaped after the layer inputs the reorder is for: projections of one normed activation,
// interleaved with the ops that consume the first of them, plus the cases that must stay put.
static model build(int n_embd, int n_tok) {
    model m;
    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    m.ctx = ggml_init(ip);
    ggml_context * ctx = m.ctx;

    auto weight = [&](const char * name, int rows) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rows);
        ggml_set_name(w, name);
        m.weights.push_back(w);
        return w;
    };

    m.x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tok);
    m.cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 16);
    ggml_set_name(m.x, "x");
    ggml_set_name(m.cache, "cache");

    // a view of the cache made before anything else, and read again after the cache is written
    ggml_tensor * cache_v = ggml_view_2d(ctx, m.cache, n_embd, 16, m.cache->nb[1], 0);
    ggml_set_name(cache_v, "cache_v");
    ggml_tensor * early = ggml_scale(ctx, cache_v, 1.0f);
    ggml_set_name(early, "early");

    ggml_tensor * act = ggml_rms_norm(ctx, m.x, 1e-6f);
    ggml_set_name(act, "act");

    // first sibling and a chain on its output
    ggml_tensor * q = ggml_mul_mat(ctx, weight("wq", 32), act);
    ggml_set_name(q, "q");
    ggml_tensor * q_act = ggml_sigmoid(ctx, q);
    ggml_set_name(q_act, "q_act");
    ggml_tensor * q_sc = ggml_scale(ctx, q_act, 0.5f);
    ggml_set_name(q_sc, "q_sc");

    // a second sibling, free to move
    ggml_tensor * k = ggml_mul_mat(ctx, weight("wk", 32), act);
    ggml_set_name(k, "k");

    // writes into the cache through a view, then a sibling uses the cache as its weight: it would
    // join, but must not move above the write
    ggml_tensor * cache_upd = ggml_cpy(ctx, act, ggml_view_2d(ctx, m.cache, n_embd, n_tok, m.cache->nb[1], 0));
    ggml_set_name(cache_upd, "cache_upd");
    ggml_tensor * from_cache = ggml_mul_mat(ctx, m.cache, act);
    ggml_set_name(from_cache, "from_cache");
    // the same, through the view made before the leader: only the view's source shows the write
    ggml_tensor * via_view = ggml_mul_mat(ctx, cache_v, act);
    ggml_set_name(via_view, "via_view");

    // a third sibling: moves up next to q and k
    ggml_tensor * v = ggml_mul_mat(ctx, weight("wv", 32), act);
    ggml_set_name(v, "v");

    // same activation, but its weight is produced after q: must stay behind it
    ggml_tensor * dyn_w = ggml_scale(ctx, weight("wd", 32), 2.0f);
    ggml_set_name(dyn_w, "dyn_w");
    ggml_tensor * dyn_mm = ggml_mul_mat(ctx, dyn_w, act);
    ggml_set_name(dyn_mm, "dyn_mm");

    // a fourth free sibling, past a full group of three: the next leader takes it
    ggml_tensor * g = ggml_mul_mat(ctx, weight("wg", 32), act);
    ggml_set_name(g, "g");

    m.gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : { early, q_sc, k, cache_upd, from_cache, via_view, v, dyn_mm, g }) {
        ggml_build_forward_expand(m.gf, t);
        m.outputs.push_back(t);
    }
    return m;
}

static void fill(model & m, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto rand_fill = [&](ggml_tensor * t) {
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            d[i] = dist(rng);
        }
    };
    rand_fill(m.x);
    rand_fill(m.cache);
    for (ggml_tensor * w : m.weights) {
        rand_fill(w);
    }
}

static std::vector<std::vector<float>> compute(model & m) {
    ggml_graph_compute_with_ctx(m.ctx, m.gf, 1);
    std::vector<std::vector<float>> res;
    for (ggml_tensor * t : m.outputs) {
        const float * d = (const float *) t->data;
        res.emplace_back(d, d + ggml_nelements(t));
    }
    return res;
}

static void test_order_and_results() {
    model ref = build(64, 3);
    model opt = build(64, 3);
    fill(ref, 42);
    fill(opt, 42);

    std::vector<uint8_t> pinned(opt.gf->n_nodes, 0);
    ggml_sycl_hoist_rules rules = {};
    rules.leader    = leader_mm;
    rules.joins     = joins_mm;
    rules.pinned    = pinned.data();
    rules.window    = 128;
    rules.max_group = 3;

    const int n_moved = ggml_sycl_graph_hoist_siblings(opt.gf, rules);

    const int iq = index_of(opt.gf, "q");
    const int ik = index_of(opt.gf, "k");
    const int iv = index_of(opt.gf, "v");
    CHECK(n_moved == 3);
    CHECK(ik == iq + 1);
    CHECK(iv == iq + 2);
    // stayed behind what they depend on
    CHECK(index_of(opt.gf, "from_cache") > index_of(opt.gf, "cache_upd"));
    CHECK(index_of(opt.gf, "via_view") > index_of(opt.gf, "cache_upd"));
    CHECK(index_of(opt.gf, "dyn_mm") == index_of(opt.gf, "dyn_w") + 1);
    // the blocked from_cache leads the next group: via_view was already next to it, g moves up
    CHECK(index_of(opt.gf, "via_view") == index_of(opt.gf, "from_cache") + 1);
    CHECK(index_of(opt.gf, "g") == index_of(opt.gf, "via_view") + 1);
    CHECK(topologically_valid(opt.gf));

    const auto a = compute(ref);
    const auto b = compute(opt);
    CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        CHECK(a[i].size() == b[i].size() && memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)) == 0);
    }

    ggml_free(ref.ctx);
    ggml_free(opt.ctx);
}

static void test_pinned() {
    model m = build(64, 3);

    // pin the second sibling: the third may still join q, the second may not
    std::vector<uint8_t> pinned(m.gf->n_nodes, 0);
    pinned[index_of(m.gf, "k")] = 1;

    ggml_sycl_hoist_rules rules = {};
    rules.leader    = leader_mm;
    rules.joins     = joins_mm;
    rules.pinned    = pinned.data();
    rules.window    = 128;
    rules.max_group = 3;

    ggml_sycl_graph_hoist_siblings(m.gf, rules);
    CHECK(index_of(m.gf, "v") == index_of(m.gf, "q") + 1);
    CHECK(index_of(m.gf, "k") > index_of(m.gf, "q_sc"));
    CHECK(topologically_valid(m.gf));

    // a pinned leader starts nothing
    ggml_free(m.ctx);
    m = build(64, 3);
    std::vector<uint8_t> pin_lead(m.gf->n_nodes, 0);
    pin_lead[index_of(m.gf, "q")] = 1;
    rules.pinned = pin_lead.data();
    const int iq = index_of(m.gf, "q");
    ggml_sycl_graph_hoist_siblings(m.gf, rules);
    CHECK(index_of(m.gf, "q") == iq);
    // k then leads, and v joins it
    CHECK(index_of(m.gf, "v") == index_of(m.gf, "k") + 1);
    CHECK(topologically_valid(m.gf));
    ggml_free(m.ctx);
}

static void test_window_and_barrier() {
    model m = build(64, 3);
    std::vector<ggml_tensor *> before(m.gf->nodes, m.gf->nodes + m.gf->n_nodes);

    ggml_sycl_hoist_rules rules = {};
    rules.leader    = leader_mm;
    rules.joins     = joins_mm;
    rules.pinned    = nullptr;
    rules.window    = 1;  // too short to reach any sibling
    rules.max_group = 3;

    CHECK(ggml_sycl_graph_hoist_siblings(m.gf, rules) == 0);
    for (int i = 0; i < m.gf->n_nodes; ++i) {
        CHECK(m.gf->nodes[i] == before[i]);
    }
    ggml_free(m.ctx);
}

// Random graphs over a few shared activations: mat-muls, elementwise chains, reshapes, in-place
// ops and copies into views of a persistent buffer. Whatever the reorder does, every result has
// to come out bit-identical and the order has to stay valid.
static void test_fuzz() {
    const int n_embd = 32;
    const int n_tok  = 2;
    int total_moved  = 0;

    for (uint32_t seed = 1; seed <= 200; ++seed) {
        std::mt19937 rng(seed);
        auto pick = [&](int n) { return (int) (rng() % (uint32_t) n); };

        model runs[2];
        for (auto & m : runs) {
            std::mt19937 grng(seed * 7919u);
            auto gpick = [&](int n) { return (int) (grng() % (uint32_t) n); };

            ggml_init_params ip = { 32 * 1024 * 1024, nullptr, false };
            m.ctx = ggml_init(ip);
            ggml_context * ctx = m.ctx;

            m.x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tok);
            m.cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 8);

            // values a node can read: [n_embd, n_tok] tensors
            std::vector<ggml_tensor *> pool = { ggml_rms_norm(ctx, m.x, 1e-6f) };
            pool.push_back(ggml_scale(ctx, pool[0], 0.25f));
            std::vector<ggml_tensor *> outs;

            const int n_ops = 12 + gpick(24);
            for (int o = 0; o < n_ops; ++o) {
                ggml_tensor * a = pool[gpick((int) pool.size())];
                ggml_tensor * t = nullptr;
                switch (gpick(7)) {
                    case 0:
                    case 1: {  // a projection, often of the first activations
                        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
                        m.weights.push_back(w);
                        t = ggml_mul_mat(ctx, w, gpick(2) ? pool[gpick(2)] : a);
                        break;
                    }
                    case 2: t = ggml_sigmoid(ctx, a); break;
                    case 3: t = ggml_add(ctx, a, pool[gpick((int) pool.size())]); break;
                    case 4: t = ggml_reshape_2d(ctx, ggml_cont(ctx, a), n_embd, n_tok); break;
                    case 5: {  // write into the persistent buffer, and sometimes read it back as a weight
                        const int row = gpick(8 - n_tok + 1);
                        ggml_tensor * dst = ggml_view_2d(ctx, m.cache, n_embd, n_tok, m.cache->nb[1], row * m.cache->nb[1]);
                        outs.push_back(ggml_cpy(ctx, a, dst));
                        if (gpick(2)) {
                            t = ggml_mul_mat(ctx, ggml_view_2d(ctx, m.cache, n_embd, n_embd / 4, m.cache->nb[1], 0),
                                             pool[gpick(2)]);
                            t = ggml_cont(ctx, ggml_pad(ctx, t, n_embd - t->ne[0], 0, 0, 0));
                        }
                        break;
                    }
                    case 6: t = ggml_add_inplace(ctx, ggml_cont(ctx, a), pool[0]); break;
                }
                if (t) {
                    pool.push_back(t);
                    outs.push_back(t);
                }
            }

            m.gf = ggml_new_graph_custom(ctx, 4096, false);
            for (ggml_tensor * t : outs) {
                ggml_build_forward_expand(m.gf, t);
            }
            // a result can be overwritten by a later in-place op or copy, so compare every node's
            // final contents rather than values captured along the way
            for (int i = 0; i < m.gf->n_nodes; ++i) {
                m.outputs.push_back(m.gf->nodes[i]);
            }
        }
        GGML_UNUSED(pick);

        fill(runs[0], seed);
        fill(runs[1], seed);

        std::vector<uint8_t> pinned(runs[1].gf->n_nodes, 0);
        for (auto & p : pinned) {
            p = rng() % 16 == 0;
        }
        ggml_sycl_hoist_rules rules = {};
        rules.leader    = leader_mm;
        rules.joins     = joins_mm;
        rules.pinned    = pinned.data();
        rules.window    = 1 + pick(64);
        rules.max_group = 2 + pick(3);

        total_moved += ggml_sycl_graph_hoist_siblings(runs[1].gf, rules);
        CHECK(topologically_valid(runs[1].gf));

        const auto a = compute(runs[0]);
        const auto b = compute(runs[1]);
        // outputs were collected in the original order, so they line up
        bool same = a.size() == b.size();
        for (size_t i = 0; same && i < a.size(); ++i) {
            same = a[i].size() == b[i].size() && memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)) == 0;
        }
        const float * ca = (const float *) runs[0].cache->data;
        const float * cb = (const float *) runs[1].cache->data;
        same = same && memcmp(ca, cb, ggml_nbytes(runs[0].cache)) == 0;
        if (!same) {
            fprintf(stderr, "fuzz seed %u: results differ after the reorder\n", seed);
        }
        CHECK(same);

        ggml_free(runs[0].ctx);
        ggml_free(runs[1].ctx);
    }
    // the generator has to give the reorder something to do, or this proves nothing
    CHECK(total_moved > 50);
    printf("fuzz: %d nodes moved over 200 graphs\n", total_moved);
}

int main() {
    ggml_cpu_init();

    test_order_and_results();
    test_pinned();
    test_window_and_barrier();
    test_fuzz();

    if (n_failed) {
        fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    printf("OK\n");
    return 0;
}
