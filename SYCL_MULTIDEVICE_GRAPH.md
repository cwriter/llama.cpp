# SYCL multi-device graphs: findings and approaches

This note records the investigation into enabling SYCL command graphs when the
`device_count > 1` and per-op exclusions in `check_graph_compatibility`
(`ggml/src/ggml-sycl/ggml-sycl.cpp`) reject a graph. All work is behind
`GGML_SYCL_GRAPH` (build) and `GGML_SYCL_ENABLE_GRAPH` (run).

## 1. The multi-device ban was overly broad

`check_graph_compatibility` used to reject every graph on a multi-GPU machine
because "a `sycl_ex::command_graph` object can only be created for a single
device" (from #13587). That premise is true per graph, but each SYCL backend
instance already owns exactly one device, and the scheduler
(`ggml_backend_sched_split_graph` / `compute_splits` in
`ggml/src/ggml-backend.cpp`) hands each backend a contiguous split whose nodes
all live on that device. Cross-device inputs are copied on the host before /
after `graph_compute`, never inside the recorded graph. The debug asserts in
`ggml_backend_sycl_graph_compute_impl` (node and src buffers all on the
backend's device) encode exactly this invariant.

Change: the blanket `device_count > 1` early-out is replaced with a per-node
check (`ggml_sycl_node_graph_supported`). A graph records iff every compute
node and every SYCL src buffer belongs to this backend's device buffer type.
Split sources are rejected explicitly: split buffers fail the
`ggml_backend_buffer_is_sycl` test, so without the explicit
`ggml_backend_buffer_is_sycl_split` check a split-src0 `MUL_MAT` would have
been classified as supported and recorded its multi-device dispatches.
Split buffers (`--split-mode row`) dispatch across all devices inside a single
op (`ggml_sycl_op_mul_mat` iterates devices only when
`ggml_backend_buffer_is_sycl_split(src0->buffer)`), so they are rejected
automatically and keep the eager path. `--split-mode none/layer` records
per device.

Pre-existing bug this depends on: `ggml_backend_sycl_cpy_tensor_async` copies
cross-device inputs with a plain queue `memcpy`, which is a silent no-op for
peer USM across separate SYCL contexts (see the COMM-D2D-FIX comments in
`ggml_backend_sycl_comm_allreduce_tensor`). It should route through
`dev2dev_memcpy`. Until that is fixed, multi-GPU layer offload is suspect
with or without graphs.

## 2. CONCAT exclusion was stale, removed

The `GGML_OP_CONCAT` exclusion dated from #13587, when the dim==3 path ended
its two memcpys with `.wait()`. Commit d415e65a5 removed those waits; current
`concat.cpp` submits only kernels and async memcpys on the recording queue.
The two back-to-back memcpys target disjoint ranges on an in-order queue, so
capture order is safe. The case (with its `[[fallthrough]]`) is deleted.

## 3. MUL_MAT_ID: fused path allowed, fallback still barriers

Unlike CONCAT, this exclusion is load-bearing. `ggml_sycl_mul_mat_id` has two
paths:

- Fused (`ggml_sycl_mul_mat_id_mmvq_fused`): never reads `ids` on the host,
  passes the `ids->data` device pointer into the kernel, fixed kernel
  sequence, topology static across tokens. Capturable (mirrors CUDA's
  `TAG_MUL_MAT_ID_CUDA_GRAPHS`).
- Fallback: `memcpy` ids device-to-host + `stream->wait()`, host counting
  sort, host loop over experts dispatching varying `ggml_sycl_mul_mat`
  calls. Data-dependent control flow cannot be captured, and `wait()` on a
  recording queue throws.

Change: `MUL_MAT_ID` records iff `g_ggml_sycl_use_async_mem_op` (pool allocs
during recording, same rationale as `MUL_MAT`) and
`ggml_sycl_mul_mat_id_graph_supported` holds. The gate replicates the fused
gates (`ne12==1`, F32, `ne10` match/alignment, contiguous `src1`, ids shape)
plus `ggml_sycl_mul_mat_vec_q_id_supports_type` (`mmvq.cpp`), which mirrors
the plain kernel table. The reorder kernel table only handles a subset
(Q4_K/Q5_K/Q6_K), so plain-table membership covers both reorder states.
Reject reasons are logged per node
(`ggml_sycl_mul_mat_id_graph_reject_reason`) since the message alone cannot
tell async-mem-off apart from a shape/type miss or prefill (`ne[2] > 1`).

Caveat: on the first recording the one-time weight reorder
(`opt_for_reorder_id`) may be captured along with the GEMM; the second call
re-records without it, the update throws on topology mismatch, and the path
re-finalizes. Same exposure as `MUL_MAT` today, but confirm on hardware.

Multi-token batches (`src1->ne[2] > 1`, i.e. prefill) run as a static token
loop (`ggml_sycl_mul_mat_id_token_loop`) when the token count is small
(`GGML_SYCL_MMID_TOKEN_LOOP_MAX_TOKENS`, 8): each token's activations are
staged to scratch and the fused single-token path runs per token. The loop
bound is a host-known shape dim and every iteration submits the same
sequence, so it records into a graph, and the graph check accepts it via the
same predicate. GEMV-per-token streams the selected experts' weights once
per token instead of once per expert, so large prefills keep the grouped
fallback (bandwidth wins there; graphs buy little in compute-bound
prefill anyway).

## 4. Segmented graphs: capture around the barriers

When the whole-graph check fails, `ggml_backend_sycl_graph_compute_segmented`
splits the node list at unsupported ops. Maximal supported runs of at least
`GGML_SYCL_GRAPH_MIN_SEGMENT_NODES` (8, heuristic) compute nodes are each
recorded to their own command graph via `ggml_graph_view`; the rest runs
eagerly in between on the same in-order queue, so eager ops (including host
waits) act as barriers. Adjacent eager pieces merge into one view.
Executable graphs are cached in `exec_graphs` on the context; slot-count
mismatch rebuilds, topology drift heals via update-throws-to-refinalize.
Finalize/update/submit is shared with the fast path
(`ggml_backend_sycl_submit_recorded_graph`). Zero qualifying runs degrades
to plain eager.

Notes:

- Fusion lookahead (`cgraph->nodes[i+1]`) stays in-bounds on views, same as
  scheduler splits. No fusion pattern spans a barrier node today, so
  segmentation cannot break a fusion.
- The fast path (whole graph supported) is byte-for-byte the old behavior.
- Watch for segment-count oscillation between calls (cache rebuild churn);
  `GGML_SYCL_DEBUG` logs the segment count.

## 5. MUL_MAT_ID fusion: output side only

`MUL_MAT_ID` is a gather-GEMM; the indirectness is input-side, so producer
fusion is not useful (scattered reads; the shared `src1` producer would be
duplicated per expert). The router-chain fusion that produces `ids`/`weights`
already exists (`ggml_sycl_fuse_topk_moe`). The unfused consumer side is the
MoE combine `MUL` (expert output x routing weight): a GEMM epilogue in the
`mmvq.cpp` moe kernels is the concrete next target and composes cleanly with
both graphs and segmentation.

## 6. Possible follow-up: split mapping from execution

The fallback splits naturally into mapping (ids D2H + wait, host counting
sort, H2D mapping upload, fixed-topology gather kernel) and execution
(data-dependent per-expert loop + scatter). As scaffolding this isolates the
barrier, but the split alone captures nothing: the mapping half records only
after the sort moves on-device, and the execution half only after a
static-topology rewrite (all-experts-masked loop or grouped GEMM, quantizing
the gathered buffer once instead of per expert). Only worth it if the
fallback shows up in a workload run under graphs (prefill-heavy MoE,
non-fused dtypes); decode is already covered by the fused path. Device-side
launch is not an option at all: SYCL has no dynamic parallelism, kernels
cannot enqueue kernels.

## 8. Library kernels: oneDNN / oneMKL stay off when graphs are on

oneDNN executes on its own interop stream, so recording a oneDNN call fails
with "Graph nodes cannot depend on events from outside the graph". This bit
two op families (oneDNN paths are on by default,
`GGML_SYCL_ENABLE_DNN=1`):

- Dense `MUL_MAT`: `DnnlGemmWrapper::row_gemm`/`gemm` for prefill-sized
  GEMMs (F32 site at/above the 256^3 direct-MKL flops cutoff, the BF16 and
  F16 convert sites, and the F16 batched path). The throwing site has no
  fallback: its catch-all prints and calls `std::exit(1)`.
- `FLASH_ATTN_EXT`: the oneDNN SDPA selection has an internal catch that
  falls back to TILE, but a failed oneDNN attempt may leave partial nodes
  in the recording, so it cannot be relied on. (The codebase already notes
  at `fattn.cpp` that MKL GEMM calls are incompatible with graph capture
  replay.)

Fix: the dispatch sites stay on native kernels whenever graphs are enabled
(`!g_ggml_sycl_enable_graph` added to the oneDNN branches in
`ggml_sycl_op_mul_mat_sycl` / `ggml_sycl_mul_mat_batched_sycl` and to the
ONEDNN/MKL selections in `ggml_sycl_get_best_fattn_kernel`). The graph
predicate needs no library exclusion because it only runs in that mode.
Trade-off is deliberate: in graph mode the big compute-bound ops run the
native kernels instead of oneDNN/XMX, betting dispatch savings beat kernel
speed; measure prefill t/s both ways if it matters. oneMKL BLAS branches
are assumed recordable (plain kernel launches; no crash observed); revisit
if decode graphs fault there.

These paths were unreachable under recording before because graphs default
off and any `MUL_MAT_ID` disabled the whole graph; segmentation and the
fused/token-loop allowances exposed the backlog.

## 9. Record-and-free hazards: prime before recording

Any device allocation that is freed before the graph executes must never be
captured. Two instances, both primed up front by
`ggml_backend_sycl_prime_recorded_resources` (idempotent, runs before any
`begin_recording`):

- Weight reorder (`opt_for_reorder*`): allocates a transient
  `sycl_reorder_temp_buffer` (up to the full weight size), submits
  memcpy+kernel, and frees the temp on return. Recorded nodes only; eager
  nodes reorder outside capture as usual. (Dense reorder is gated to small
  batch and MoE reorder to the fused path, so this fires on
  decode-first-token, not large prefill.)
- Flash-attention F16 KV scratch (`fattn kv_buffer::ensure_half`): grows
  with a queue wait plus a real free of the old buffer. Under capture the
  wait is illegal and the free could unmap memory baked into
  already-submitted segments of the same call, so every FA node pre-grows
  to full K/V size first (same total as lazy growth, just earlier).

Sharing within a subgraph needs no extra work: the device pool already
reuses freed blocks best-fit, and pool blocks are never unmapped, so baked
pointers stay valid while single-queue in-order submission keeps lifetimes
ordered.

## 10. Stale replay across shape changes: fingerprint slots

Recording once and replaying forever is not an option: a recorded graph
bakes in concrete USM pointers (tensor data moves as ggml-alloc reuses
arenas across ubatches/tokens; pool scratch blocks differ call to call;
MoE id buffers change), and topology changes legitimely (prefill vs
decode shapes, fused vs fallback MoE). Every call must therefore either
patch the pointers (`update()`) or rebuild (refinalize) — the same
requirement CUDA graphs have with `cudaGraphExecUpdate`. Pinning every
address forever would cost all allocator reuse (i.e. OOM).

`update()` is only a shortcut; replaying an executable graph whose launch
geometry no longer matches the buffers is a page-fault storm (`xe ...
Fault response: Unsuccessful -ENOENT`, CCS CAT errors, engine reset,
`UR_RESULT_ERROR_DEVICE_LOST`). Each cache slot (`ggml_sycl_exec_graph_slot`,
fast path and segments alike) carries an FNV fingerprint of its node range
(op/type/shapes/strides, deliberately no data pointers, which legitimately
change every call through pool reuse). Mismatch refinalizes instead of
updating; the update-throw path remains as a backstop for collisions.

Observed on hardware: `update()` throws `node_type::memcpy nodes are not
supported for update. Only kernel, host_task, barrier and empty nodes are
supported.` So any segment containing a memcpy (DUP/CPY of strided views
take the memcpy path) refinalizes on every call — correct via the
throw-to-refinalize path, but it caps the update win and partly explains
"no better performance". Pure-kernel segments update normally.

Debug aids: the startup log prints `GGML_SYCL_GRAPH_FEATS` (extend it with
every behavior added here so logs always identify the build), and
`GGML_SYCL_GRAPH_NO_UPDATE=1` forces refinalize everywhere to isolate
`update()` itself as a suspect.

## 12. Synchronous op bisect: `GGML_SYCL_SYNC_OPS=1`

Opt-in, off by default. `ggml_backend_sycl_graph_compute_impl` waits on the
queue before every compute node and prints `[SYCL-SYNC] node i/N name (op)`
(with a `[recording]` tag when under capture, where the wait is skipped
because `wait()` is illegal on a recording queue, plus a final drain at the
end). The first wait to throw names the previous printed node as the
faulter — group-level for fused nodes. Use it to bisect eager-vs-recorded
faults; note it serializes execution, which can mask races all by itself.

Why memcpy nodes cannot update at all: the oneAPI update path only supports
kernel/host_task/barrier/empty node types — memcpy nodes need their
addresses re-specified in a way the update API does not cover, so the
runtime rejects the whole update. Nothing to do on our side; refinalize.

## 11. Pool lifetime: flight-gated reclaim via barrier events

`update()` only supports kernel/host_task/barrier/empty nodes; memcpy
nodes always refinalize. Why not fix update instead: the UR command-buffer
update API only covers kernel launches — there is no memcpy-update entry
point, so SYCL rejects any update touching one. Stable addresses would not
help; it is an API gap, and our throw-to-refinalize path benefits
automatically if Intel ever closes it.

The race debug prints mask: with graphs, submits return instantly and the
host runs ahead of the device, so a block baked into a submitted graph may
still be in flight when its host-side free runs. The legacy pool really
freed blocks past its cap, unmapping memory out from under in-flight graphs
(page-fault storm); the in-order queue cannot save a host-side free. Eager
mode survives because per-op submission latency and host waits keep it
roughly in lockstep, and `GGML_SYCL_DEBUG` prints throttle the host the
same way.

Fix, as suggested: reclaim by signal, not by hope. After every graph
submit, `ext_oneapi_submit_barrier` yields an event ordered after the whole
graph; both pools (device and host — the host pool had the same ring-reuse
shape feeding oneMKL batch metadata) record one flight per submit. Cached
blocks are freely reusable (in-order queue), but a block is only really
freed while no flight is pending — i.e. no submitted graph can reference
it. Eager-only operation keeps the old caps bit-for-bit (256 device / 64
host). No host waits anywhere near recording: the event status is polled,
never waited on. Matches the VMM pool, which never unmaps.

## 13. Verification checklist (needs hardware)

- `test-backend-ops -b SYCL0 -o CONCAT` with graphs on (original #13587
  repro).
- MoE decode with graphs on: fused path records, topology static across
  tokens, update path steady-state.
- Short-prefill / speculative-verify MoE (`ne[2]` in 2..8): token loop
  records; check numerics against the grouped fallback per token slice
  (offsets use `nb[2]`/`nb[1]` strides) and compare against eager fallback.
- Prefill MoE: segmented path, measure re-record + multi-update overhead vs
  eager; check segment-count stability.
- First-token reorder interplay for `MUL_MAT` / fused `MUL_MAT_ID`.
- Multi-GPU layer offload only after the `cpy_tensor_async` fix above.
