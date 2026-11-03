#ifndef GGML_SYCL_GRAPH_DUMP_HPP
#define GGML_SYCL_GRAPH_DUMP_HPP

// GGML_SYCL_GRAPH_DUMP=<dir>: debug dump of the graphs the backend executes, off by default.
// For each selected graph it writes <dir>/graph-dev<D>-<seq>-t<tokens>.json: every node in execution order
// (after graph_optimize) with its op, type, shape, sources and data pointer, and how the compute loop
// handled it: the kernels and copies it submitted, skipped over by a fused node (jumped_by), or returned
// without submitting (absorbed elsewhere, a view or a no-op). scripts/sycl-graph-viz.py turns it into a
// dependency graph with the fusions and fusion candidates.
// GGML_SYCL_GRAPH_DUMP_TOKENS (default -1, any) selects graphs by ubatch width: the widest weight mat-mul.
// GGML_SYCL_GRAPH_DUMP_MAX (default 4) caps the graphs dumped per device.
// Submissions are traced through the Level Zero loader (ZE_ENABLE_TRACING_LAYER, set automatically when the
// dump is on). Run with SYCL graphs off (the default): a replayed graph does not walk the nodes.

#include "common.hpp"

struct ggml_sycl_graph_dump;

// returns null when the dump is off or this graph is not selected
ggml_sycl_graph_dump * ggml_sycl_graph_dump_begin(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph);
// call at the top of every compute-loop iteration, before any skip
void ggml_sycl_graph_dump_visit(ggml_sycl_graph_dump * d, int i);
void ggml_sycl_graph_dump_end(ggml_sycl_graph_dump * d);

#endif  // GGML_SYCL_GRAPH_DUMP_HPP
