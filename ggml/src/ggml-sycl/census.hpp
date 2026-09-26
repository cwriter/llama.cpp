#ifndef GGML_SYCL_CENSUS_HPP
#define GGML_SYCL_CENSUS_HPP

// GGML_SYCL_CENSUS=<n>: debug census of reserved vs used memory, off (0) by default.
// For the first n prefill graphs per device it reports, for every compute-buffer tensor, whether
// a kernel wrote it (checksum before/after each fused group) and whether an executed group reads
// it (structural), plus the live set at the arena peaks and the SYCL pool traffic per op.
// GGML_SYCL_CENSUS_MIN_TOKENS (default 512) selects the graphs: the widest node ne[1].

#include "common.hpp"

extern int g_ggml_sycl_census;

struct ggml_sycl_census_graph;

// returns null when the census is off or this graph is not reported
ggml_sycl_census_graph * ggml_sycl_census_begin(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph,
                                                const std::vector<int> & absorbed_by);
void ggml_sycl_census_visit(ggml_sycl_census_graph * cg, int i);
void ggml_sycl_census_end(ggml_sycl_census_graph * cg);

// pool hooks; cheap no-ops when the census is off
void ggml_sycl_census_pool_alloc(int device, bool host, size_t req, size_t actual, bool fresh, size_t pool_size);
void ggml_sycl_census_pool_free(int device, bool host, size_t size);

#endif  // GGML_SYCL_CENSUS_HPP
