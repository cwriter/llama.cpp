#ifndef GGML_SYCL_GRAPH_REORDER_HPP
#define GGML_SYCL_GRAPH_REORDER_HPP

#include <cstdint>

#include "ggml.h"

// Dependency-aware node reorder for graph_optimize, after the passes in ggml-vulkan
// (ggml_vk_graph_optimize) and ggml-metal (ggml_metal_graph_optimize_reorder). Those reorder to
// cut barriers or to encode concurrently; one in-order queue gains neither, so this one only
// moves a node when that makes it adjacent to a sibling a fused kernel can take with it - the
// mat-vec batcher (GGML_SYCL_MV_FUSE) needs its group back to back, and the graph builders
// interleave siblings with the ops that consume the first of them.
//
// Pure graph inspection: it never reads tensor data or addresses (nothing is allocated when
// graph_optimize runs), so it lives apart from the SYCL code and can be tested on the host.

struct ggml_sycl_hoist_rules {
    // cur may start a group
    bool (*leader)(const ggml_tensor * cur, void * user_data);
    // cand can join the group that leader started
    bool (*joins)(const ggml_tensor * leader, const ggml_tensor * cand, void * user_data);
    void * user_data;

    // pinned[i] != 0: node i is part of a multi-node fusion the compute loop matches, so it may
    // neither move nor lead, and nothing is inserted next to it (see graph_optimize)
    const uint8_t * pinned;

    int window;     // nodes scanned past a leader
    int max_group;  // the most nodes a fused kernel takes, leader included
};

// Moves every node a leader can take up to sit right after it (or after the siblings already
// moved there), provided nothing it passes over produces what it reads or touches what it
// writes. Nodes keep their relative order otherwise. Returns the number of nodes that changed
// place.
int ggml_sycl_graph_hoist_siblings(ggml_cgraph * cgraph, const ggml_sycl_hoist_rules & rules);

#endif  // GGML_SYCL_GRAPH_REORDER_HPP
