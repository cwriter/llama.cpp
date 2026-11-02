#include "graph-dump.hpp"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_sycl_graph_dump {
    ggml_backend_sycl_context * ctx = nullptr;
    const ggml_cgraph *         g   = nullptr;
    int                         seq    = 0;
    int64_t                     tokens = 0;

    std::vector<int>  jumped_by;  // index of the node whose iteration skipped over this one, or -1
    std::vector<char> visited;    // the loop reached this node at the top of an iteration
    std::vector<std::vector<std::string>> work;  // what that iteration submitted: kernel names, "copy", "fill"

    int cur = -1;
};

// Submissions are seen in-process through the Level Zero loader's tracing layer, which has to be on before the
// SYCL runtime initializes Level Zero: ZE_ENABLE_TRACING_LAYER=1 (set here at load time when the dump is asked for).
// Without it every visited node counts as a launch and the dump says so ("traced": false).
static thread_local std::vector<std::string> * t_graph_dump_work = nullptr;
static bool g_graph_dump_traced = false;

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <cxxabi.h>
#include <level_zero/layers/zel_tracing_api.h>
#include <level_zero/layers/zel_tracing_register_cb.h>

__attribute__((constructor)) static void graph_dump_enable_tracing_layer() {
    const char * dir = getenv("GGML_SYCL_GRAPH_DUMP");
    if (dir && *dir) {
        setenv("ZE_ENABLE_TRACING_LAYER", "1", 0);
    }
}

static void graph_dump_kernel(ze_kernel_handle_t k) {
    if (!t_graph_dump_work) {
        return;
    }
    size_t n = 0;
    std::string name = "?";
    if (zeKernelGetName(k, &n, nullptr) == ZE_RESULT_SUCCESS && n > 1) {
        name.assign(n, '\0');
        zeKernelGetName(k, &n, name.data());
        name.resize(n - 1);
        int    st = 0;
        char * d  = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &st);
        if (st == 0 && d) {
            name = d;
            const std::string pre = "typeinfo name for ";
            if (name.compare(0, pre.size(), pre) == 0) {
                name.erase(0, pre.size());
            }
        }
        free(d);
    }
    t_graph_dump_work->push_back(name);
}

static void graph_dump_cb_kernel(ze_command_list_append_launch_kernel_params_t * p, ze_result_t, void *, void **) {
    graph_dump_kernel(*p->phKernel);
}
static void graph_dump_cb_kernel_args(ze_command_list_append_launch_kernel_with_arguments_params_t * p, ze_result_t, void *, void **) {
    graph_dump_kernel(*p->phKernel);
}
static void graph_dump_cb_kernel_params(ze_command_list_append_launch_kernel_with_parameters_params_t * p, ze_result_t, void *, void **) {
    graph_dump_kernel(*p->phKernel);
}
static void graph_dump_cb_coop(ze_command_list_append_launch_cooperative_kernel_params_t * p, ze_result_t, void *, void **) {
    graph_dump_kernel(*p->phKernel);
}
static void graph_dump_cb_copy(ze_command_list_append_memory_copy_params_t *, ze_result_t, void *, void **) {
    if (t_graph_dump_work) {
        t_graph_dump_work->push_back("copy");
    }
}
static void graph_dump_cb_fill(ze_command_list_append_memory_fill_params_t *, ze_result_t, void *, void **) {
    if (t_graph_dump_work) {
        t_graph_dump_work->push_back("fill");
    }
}

static void graph_dump_tracer_init() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    const char * layer = getenv("ZE_ENABLE_TRACING_LAYER");
    if (!layer || atoi(layer) == 0) {
        GGML_LOG_WARN("%s: ZE_ENABLE_TRACING_LAYER is not set: every visited node counts as a launch\n", __func__);
        return;
    }
    zel_tracer_desc_t   desc = { ZEL_STRUCTURE_TYPE_TRACER_DESC, nullptr, nullptr };
    zel_tracer_handle_t t    = nullptr;
    if (zelTracerCreate(&desc, &t) != ZE_RESULT_SUCCESS) {
        GGML_LOG_WARN("%s: zelTracerCreate failed: every visited node counts as a launch\n", __func__);
        return;
    }
    zelTracerCommandListAppendLaunchKernelRegisterCallback(t, ZEL_REGISTER_PROLOGUE, graph_dump_cb_kernel);
    zelTracerCommandListAppendLaunchKernelWithArgumentsRegisterCallback(t, ZEL_REGISTER_PROLOGUE, graph_dump_cb_kernel_args);
    zelTracerCommandListAppendLaunchKernelWithParametersRegisterCallback(t, ZEL_REGISTER_PROLOGUE, graph_dump_cb_kernel_params);
    zelTracerCommandListAppendLaunchCooperativeKernelRegisterCallback(t, ZEL_REGISTER_PROLOGUE, graph_dump_cb_coop);
    zelTracerCommandListAppendMemoryCopyRegisterCallback(t, ZEL_REGISTER_PROLOGUE, graph_dump_cb_copy);
    zelTracerCommandListAppendMemoryFillRegisterCallback(t, ZEL_REGISTER_PROLOGUE, graph_dump_cb_fill);
    g_graph_dump_traced = zelTracerSetEnabled(t, true) == ZE_RESULT_SUCCESS;
}
#else
static void graph_dump_tracer_init() {
    GGML_LOG_WARN("%s: built without Level Zero: every visited node counts as a launch\n", __func__);
}
#endif

// close the iteration that started at d->cur, now that the loop is at next
static void graph_dump_close(ggml_sycl_graph_dump * d, int next) {
    if (d->cur < 0) {
        return;
    }
    t_graph_dump_work = nullptr;
    for (int j = d->cur + 1; j < next; ++j) {
        d->jumped_by[j] = d->cur;
    }
}

ggml_sycl_graph_dump * ggml_sycl_graph_dump_begin(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph) {
    static const char * dir        = getenv("GGML_SYCL_GRAPH_DUMP");
    static const int    want       = ggml_sycl_get_env("GGML_SYCL_GRAPH_DUMP_TOKENS", -1);
    static const int    max_graphs = ggml_sycl_get_env("GGML_SYCL_GRAPH_DUMP_MAX", 4);
    static int          dumped[GGML_SYCL_MAX_DEVICES] = {};
    if (!dir || !*dir || dumped[ctx.device] >= max_graphs) {
        return nullptr;
    }

    // tokens in the ubatch: the widest weight mat-mul
    int64_t tokens = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if ((n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_MUL_MAT_ID) && n->src[0] && n->src[0]->buffer &&
            ggml_backend_buffer_get_usage(n->src[0]->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            tokens = std::max(tokens, n->op == GGML_OP_MUL_MAT ? n->ne[1] : n->ne[2]);
        }
    }
    if (want >= 0 && tokens != want) {
        return nullptr;
    }

    auto * d   = new ggml_sycl_graph_dump();
    d->ctx     = &ctx;
    d->g       = cgraph;
    d->seq     = dumped[ctx.device]++;
    d->tokens  = tokens;
    d->jumped_by.assign(cgraph->n_nodes, -1);
    d->visited.assign(cgraph->n_nodes, 0);
    d->work.assign(cgraph->n_nodes, {});
    graph_dump_tracer_init();
    return d;
}

void ggml_sycl_graph_dump_visit(ggml_sycl_graph_dump * d, int i) {
    if (!d) {
        return;
    }
    graph_dump_close(d, i);
    d->cur        = i;
    d->visited[i] = 1;
    t_graph_dump_work = &d->work[i];
}

static std::string graph_dump_str(const char * s) {
    std::string o = "\"";
    for (; *s; ++s) {
        if (*s == '"' || *s == '\\') {
            o += '\\';
            o += *s;
        } else if ((unsigned char) *s < 0x20) {
            char b[8];
            snprintf(b, sizeof(b), "\\u%04x", *s);
            o += b;
        } else {
            o += *s;
        }
    }
    return o + "\"";
}

static const char * graph_dump_usage(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    if (!base->buffer) {
        return "none";
    }
    switch (ggml_backend_buffer_get_usage(base->buffer)) {
        case GGML_BACKEND_BUFFER_USAGE_WEIGHTS: return "weights";
        case GGML_BACKEND_BUFFER_USAGE_COMPUTE: return "compute";
        default:                                return "any";
    }
}

// fields shared by nodes and outside tensors
static void graph_dump_tensor(FILE * f, const ggml_tensor * t) {
    fprintf(f, "\"name\":%s,\"op\":%s,\"type\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"nb\":[%zu,%zu,%zu,%zu]",
            graph_dump_str(t->name).c_str(), graph_dump_str(ggml_op_desc(t)).c_str(), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    fprintf(f, ",\"bytes\":%zu,\"data\":\"%p\",\"flags\":%d,\"usage\":\"%s\",\"contiguous\":%s", ggml_nbytes(t), t->data,
            t->flags, graph_dump_usage(t), ggml_is_contiguous(t) ? "true" : "false");
}

void ggml_sycl_graph_dump_end(ggml_sycl_graph_dump * d) {
    if (!d) {
        return;
    }
    const ggml_cgraph * g = d->g;
    graph_dump_close(d, g->n_nodes);

    static const char * dir = getenv("GGML_SYCL_GRAPH_DUMP");
    char path[1024];
    snprintf(path, sizeof(path), "%s/graph-dev%d-%d-t%lld.json", dir, d->ctx->device, d->seq, (long long) d->tokens);
    FILE * f = fopen(path, "w");
    if (!f) {
        GGML_LOG_WARN("%s: cannot write %s\n", __func__, path);
        delete d;
        return;
    }

    // every tensor a node reads that is not itself a node of this graph: weights, inputs, other splits' outputs
    std::unordered_map<const ggml_tensor *, int> node_idx;
    for (int i = 0; i < g->n_nodes; ++i) {
        node_idx[g->nodes[i]] = i;
    }
    std::unordered_map<const ggml_tensor *, int> ext_idx;
    std::vector<const ggml_tensor *>             ext;
    auto ref = [&](const ggml_tensor * t) -> std::string {
        const auto it = node_idx.find(t);
        if (it != node_idx.end()) {
            return "{\"node\":" + std::to_string(it->second) + "}";
        }
        auto e = ext_idx.find(t);
        if (e == ext_idx.end()) {
            e = ext_idx.emplace(t, (int) ext.size()).first;
            ext.push_back(t);
        }
        return "{\"ext\":" + std::to_string(e->second) + "}";
    };

    fprintf(f, "{\"device\":%d,\"seq\":%d,\"tokens\":%lld,\"n_nodes\":%d,\"traced\":%s,\"nodes\":[\n", d->ctx->device,
            d->seq, (long long) d->tokens, g->n_nodes, g_graph_dump_traced ? "true" : "false");
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * t = g->nodes[i];
        fprintf(f, "%s{\"i\":%d,", i ? ",\n" : "", i);
        graph_dump_tensor(f, t);
        const bool submitted = g_graph_dump_traced ? !d->work[i].empty() : (bool) d->visited[i];
        fprintf(f, ",\"visited\":%s,\"submitted\":%s,\"jumped_by\":%d,\"view_src\":%s,\"work\":[",
                d->visited[i] ? "true" : "false", submitted ? "true" : "false", d->jumped_by[i],
                t->view_src ? ref(t->view_src).c_str() : "null");
        for (size_t w = 0; w < d->work[i].size(); ++w) {
            fprintf(f, "%s%s", w ? "," : "", graph_dump_str(d->work[i][w].c_str()).c_str());
        }
        fprintf(f, "],\"src\":[");
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            fprintf(f, "%s%s", s ? "," : "", t->src[s] ? ref(t->src[s]).c_str() : "null");
        }
        fprintf(f, "],\"op_params\":[");
        for (int p = 0; p < 8; ++p) {
            fprintf(f, "%s%d", p ? "," : "", t->op_params[p]);
        }
        fprintf(f, "]}");
    }
    fprintf(f, "\n],\"ext\":[\n");
    // ext grows while it is written: an outside tensor's view_src is outside too
    for (size_t e = 0; e < ext.size(); ++e) {
        const ggml_tensor * t = ext[e];
        fprintf(f, "%s{\"e\":%zu,", e ? ",\n" : "", e);
        graph_dump_tensor(f, t);
        fprintf(f, ",\"view_src\":%s}", t->view_src ? ref(t->view_src).c_str() : "null");
    }
    fprintf(f, "\n]}\n");
    fclose(f);
    GGML_LOG_INFO("%s: wrote %s (%d nodes, %lld tokens)\n", __func__, path, g->n_nodes, (long long) d->tokens);
    delete d;
}
