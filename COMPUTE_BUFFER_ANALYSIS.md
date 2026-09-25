# Where the 1.9 GB per-card compute buffer goes (qwen4exp, SYCL)

Measured 2026-09-18 on 3x Intel Arc Pro B60 (bmg-g21), unsloth Qwen3.8-Flash-Next
UD-IQ4_XS, `-c 131072 -b 4096 -ub 1024 --flash-attn on -ctk q8_0 -ctv q8_0
--no-kv-unified -np 1 --split-mode layer`.

## The numbers

```
sched_reserve:      SYCL0 compute buffer size =  1893.85 MiB
sched_reserve:      SYCL1 compute buffer size =  1901.10 MiB
sched_reserve:      SYCL2 compute buffer size =  1901.10 MiB
sched_reserve:  SYCL_Host compute buffer size =   415.87 MiB
sched_reserve: graph nodes  = 6912
sched_reserve: graph splits = 4
sched_reserve: reserve took 83.37 ms, sched copies = 1
```

## Two things ruled out immediately

**`GGML_SCHED_MAX_COPIES=4` costs nothing here.** The build defines it, but the log
reports `sched copies = 1`. Pipeline-parallel input copies are only allocated when a
pipeline actually exists. Not a lever.

**The buffer is not a sum of allocations.** `ggml-alloc` already reuses aggressively:
a tensor's buffer is handed to a later tensor as soon as its last consumer has run
(`ggml/src/ggml-alloc.c:794-817`). 1893 MiB is the *peak simultaneously-live set*, not
a total. So "could this be shared" is largely already true; the question is what is
live at the peak.

## Decomposition

Two runs of the same model differing only in context and ubatch:

| run | n_ctx | n_ubatch | device | host |
|---|---|---|---|---|
| A | 4096 | 512 | 161.86 MiB | 21.37 MiB |
| B | 131072 | 1024 | 1893.85 MiB | 415.87 MiB |

Modelling the device buffer as `F` (scales with ubatch only) plus `M` (scales with
n_kv * n_ubatch), and solving `F + M = 161.86`, `2F + 64M = 1893.85`:

- **M ~ 1621 MiB (86%)** scales with `n_kv * n_ubatch`
- F ~ 273 MiB is context-independent (activations, MoE intermediates, hc tensors)

Caveat: two data points, two variables, one assumed functional form. The root cause
below was found independently and predicts ~1536 MiB, which is why the fit is trusted.

## Root cause: the QSA indexer, not the attention masks

At `n_kv=131072`, `n_tps=1024`, one `[n_kv, n_tps]` f32 tensor is **512 MiB**. The
query-sparse-attention indexer builds a chain of them
(`src/models/qwen4exp.cpp:668-684`):

```c
expanded = ggml_get_rows(ctx0, ggml_cont(ggml_permute(score, 1,0,2,3)), inp->cell_blk);
expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));
ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32
                   ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);   // 512 MiB
expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));
```

Roughly three of these live at the peak is ~1536 MiB, against the 1621 MiB the
decomposition predicts.

Attention is **already sparse** here (`top_k` keeps `indexer_top_k + r - 1` cells) but
the **scoring is dense over every KV cell**. That asymmetry is where the memory goes.

### The masks themselves are fine

`blk_bias` is already doing its job. `inp->bias` is allocated `n_blocks` wide instead
of `n_kv` when causal and non-alibi (`qwen4exp.cpp:569-586`), which is the saving that
path exists for. The KQ mask proper is f16 under flash attention
(`src/llama-graph.cpp:39`) and lives in a **host** buffer
(`llama-graph.cpp:452` asserts it), so it is inside the 415.87 MiB `SYCL_Host` line,
not the per-device figure.

Note masks are graph *inputs* (`ggml_set_input`), so they are live for the whole graph
and cannot be overlapped with anything. That is true but is not what dominates here.

### These are prefill-only costs

`n_tps` is 1024 at prefill and 1 at decode, so at decode these tensors are ~512 KB.
The 1.9 GB reserve is the worst-case prefill graph being provisioned on every card.

## Fusion does not reclaim memory

`ggml_sycl_fuse` absorbs nodes at *execution* time, but `ggml_gallocr_reserve_n` has
already allocated a dst buffer for every node in the graph during reserve. Skipping a
node's execution does not unallocate its tensor. **Fusion buys launches and bandwidth;
only a graph change buys VRAM.** Worth doing for speed regardless, since these tensors
are half a gigabyte each.

## Opportunities, cheapest first

### 1. Drop the f32 cast of the KQ mask (512 MiB) - NO NEW OP

`ggml_add` already permits it at the graph level: it only checks
`ggml_can_repeat(b, a)` and takes dst type from `a`, so `ggml_add(f32_scores,
f16_mask)` constructs fine. What is missing is the SYCL dispatch:
`ggml_sycl_op_bin_bcast` (`ggml/src/ggml-sycl/binbcast.cpp:263`) enumerates
`(F32,F32,F32)`, `(F16,F16,F16)`, `(F16,F32,F16)`, I32, I16, BF16 - but not
`(F32,F16,F32)`. The underlying `bin_bcast_sycl<op>` is already templated on all three
types and instantiated for mixed combinations, so this is one more `else if` plus a
`supports_op` entry, then delete the `ggml_cast` at `qwen4exp.cpp:674`.

### 2a. Drop the first `cont(permute(score))` (512 MiB) - kernel change, no new op

`k_get_rows_float` already takes `nb01/nb02/nb03` and indexes with them, so strided
rows work. But its signature has `/*size_t nb00,*/` commented out: it assumes dim 0 is
contiguous, and `ggml_permute(score, 1,0,2,3)` is exactly a dim-0 swap. Needs `nb00`
plumbed through and the element loop generalised. The vector-width selection
(`ggml/src/ggml-sycl/getrows.cpp:247`) would fall back to width 1 for this case, so
this trades 512 MiB for a possibly slower gather. Measure, do not assume.

### 2b. Drop the second `cont(permute(expanded))` - probably needs a new op

`ggml_top_k` reduces along dim 0, so `n_kv` must be dim 0 and contiguous. Removing
this materialisation needs either a strided `top_k` or a fused "expand block scores +
add mask + top-k" op that never materialises `[n_kv, n_tps]`. The fused op is the
clean design (it removes the dense intermediate entirely rather than shuffling it) but
it is a genuine new op.

### 3. `-ub 512` halves whatever remains

Every one of these is linear in `n_tps`, so 1024 -> 512 takes ~1621 MiB to ~810 MiB,
about 800 MiB back per card. The ubatch=1024 choice was tuned *before* the grouped MoE
GEMM existed; prefill is now 480 t/s rather than ~320 and the MoE path's batch-shape
sensitivity has changed, so the old conclusion may not hold. One env change, one
server start.

## Strategic caveat

All of 1, 2a and 2b are changes to `src/models/qwen4exp.cpp`, i.e. llama.cpp core, not
the SYCL backend. `ggml_backend_sched` assigns each op to a backend that supports it,
so a SYCL-only capability is fine on this build but would break CPU-only builds
upstream. Upstreaming any of these needs at least the CPU backend to match.

## Why deleting the cast needs a capability probe

The cast is in `llama_model_qwen4exp::graph::build_qsa_top_k`, so no other architecture
reaches it. But the `ggml_add(f32, f16)` node it leaves is backend-agnostic, and
`ggml_backend_sched` assigns each node to a backend that supports it. A CPU-only build
would have no candidate; a mixed build would migrate the node and drag a 512 MiB tensor
across the bus, which is worse than the cast.

llama.cpp already has the pattern: `resolve_fused_ops` (`src/llama-context.cpp:505`)
reserves a probe graph, asks `ggml_backend_sched_get_tensor_backend` which backend
actually received the node, and disables the feature on a device mismatch. The startup
log line `resolve_fused_ops: fused DeepSeek V4 HC pre enabled` is that mechanism. So the
correct shape is a probed cparams flag that emits the cast only when the probe fails,
not an unconditional delete.

## Can allocations be freed after fusion?

No, not with today's mechanisms, though the memory is genuinely wasted.

There is a pre-allocation rewrite hook, `graph_optimize`
(`ggml/src/ggml-backend-impl.h:155`), invoked at `ggml/src/ggml-backend.cpp:1462` from
`split_graph`, which runs before `ggml_gallocr_reserve_n`. **ggml-sycl does not
implement it** (`ggml/src/ggml-sycl/ggml-sycl.cpp:6944` sets it to NULL); CPU, CUDA,
Metal, CANN, BLAS and Hexagon all do. That gap is worth closing on its own merits
(Metal uses it to reorder for concurrency).

It cannot delete nodes, though. `split->graph` is a `ggml_graph_view` aliasing the
parent node array, and the companion `add_alloc_dep` callback exists to *extend*
tensor lifetimes for out-of-order execution, the opposite direction. Removing nodes
would desync the split boundaries and the `graph_copy` built immediately after.

Freeing fused-away tensors would need `ggml-alloc` to know about fusion at reserve
time: a backend query consulted by `ggml_gallocr_reserve_n`. Feasible in principle,
since `ggml_can_fuse` is a pure function of the graph, so a reserve-time prediction
would match execution. The hazard is that any divergence means writing to an
unallocated tensor. Not worth it for this case: current fusions absorb tens of MB, not
512 MiB, and the cast is not fused at all, so this route needs the fusion written first.

## Flash attention: separate findings (read-only investigation)

Reported against the same config. Listed here because they compete for the same effort.

**SWA is not present.** `n_swa = 0`, `is_swa_any = 0` in the run log; `qwen4exp.cpp`
never touches `hparams.n_swa`. Any plan premised on sliding-window sparsity is void.

**The sparsity that does exist is QSA top-k**: `indexer.top_k = 2048` with
`compress_ratios = [0,0,0,4, ...]` on the `il%4==3` attention layers. At n_kv=131072
that is 1.56% density in runs of 4 cells. It is already materialised into a dense f16
mask and no SYCL FA kernel exploits it. `qwen4exp.cpp:764-767` has the sparse call
written and commented out with `// TODO: enable sparse attention when we are ready`,
hard-wiring the `n_kv_max` hint to 0.

**Rank 1, and it outranks everything above: `nsm` is wrong for Battlemage.**
`ggml/src/ggml-sycl/ggml-sycl.cpp:181` computes `nsm = max_compute_units / 16`, which
is the Xe-HPG (Alchemist) ratio. Xe2 has 8 XVEs per Xe-core, so with 160 reported XVEs
the true Xe-core count is 20, not 10. Combined with
`max_wg_per_cu = max_work_group_size / max_compute_units` (1024/160 = 6, a
dimensionally meaningless formula) this yields `blocks_per_wave = 60`, and the
efficiency search at `fattn-common.hpp:1076-1095` stops at `parallel_blocks = 10`.
Real capacity is ~320 resident work-groups, so 60/320 = 18.75%, which matches the
measured 20.0% XVE occupancy and 66.7% idle. Two constants, affects every FA launch.

**Rank 2, needs measurement: the whole K and V cache is dequantised to f16 on every FA
call.** `fattn-tile.hpp:1088-1090` sets `need_f16_K = need_f16_V = true` and
`launch_fattn` (`fattn-common.hpp:944-1009`) runs `to_fp16` over `ggml_nelements(K)`
unconditionally; `fattn-buffers.cpp` caches the allocation but not the contents. With
`-ctk q8_0 -ctv q8_0` that is order n_kv * 3.1 KB per attention layer per decode token
(~410 MB at n_kv=131072). Between decode steps only one KV cell changes, so it is
memoisable. Check the profile for a `to_fp16`/dequantize kernel adjacent to FA.

**Rank 3: interior all-masked block skip.** Does not exist in SYCL *or* CUDA's tile
kernel, so it is new code rather than a restoration. The trailing-block skip
(`flash_attn_mask_to_KV_max`, `fattn-common.hpp:619`) *is* ported and present, but is
gated off at decode (`fattn-common.hpp:1017` requires `Q->ne[1] >= 1024`) and would buy
nothing anyway, since QSA always selects the most recent blocks. A 64-cell-granularity
in-kernel test is worth ~3x more than a KV_max-style pre-scan: estimated 1.14x at
n_kv=16384 rising to 4.5x at 131072, decode only (prefill routes to MKL FA at
`fattn.cpp:159-175`). Nothing below n_kv ~ 2051, where top-k selects everything.

**Rank 4: `warp_size` is hardcoded to 32.** `fattn-tile.hpp:1077` says
`WARP_32_SIZE; //can't support WARP_16_SIZE` with no explanation, on a device whose
natural sub-group is 16 and whose `GGML_SYCL_WARP_SIZE` is 16. SIMD32 halves per-item
GRF and is the likely source of the 3072 B spill (the only spilling kernel in the
profile) and the 20.8% Send stalls. The file carries the compiler's own register
pressure warning at `fattn-tile.hpp:412`. Possibly a stale guard.

**A compressed per-block mask input is NOT worth it**: it cannot replace the dense mask
(FA needs per-cell values) so it is an additional input, and the in-kernel test it would
accelerate is already ~0.2% of tile cost. The win is in not materialising the mask at
all, which is the `n_kv_max` path above.

## Result: max_wg_per_cu was the FA occupancy bug (+1.6% decode)

`ggml-sycl.cpp` computed `max_wg_per_cu = max_work_group_size / max_compute_units`
(1024/160 = 6 on BMG), dividing a work-group SIZE by a compute-unit COUNT. It is the
seed for `parallel_blocks` and the bound `blocks_per_wave = nsm * max_blocks_per_sm`
in `fattn-common.hpp:1052-1078`, so it under-sized every flash-attention grid.

Swept as `GGML_SYCL_MAX_WG_PER_CU`, single binary, 3 reps per arm:

| value | decode mean | prefill |
|---|---|---|
| 6 (old) | 20.44 | 476-479 |
| 12 | 20.70 | 478-480 |
| **16 (new default)** | **20.77** | 478-479 |
| 24 | 20.65 | 476-479 |
| 32 | 20.71 | 480 |

**+1.6% decode.** The arms do not overlap (worst non-6 sample 20.62, best 6 sample
20.52) against a within-arm spread of ~0.1. Everything from 12 up is equivalent; 16 is
both the derivation (8 XVEs/Xe-core x 8 threads / 4 threads per work-group) and the best
measured.

**Prefill is flat across all arms, as predicted before running**: at prefill
`Q->ne[1] >= 32` routes to MKL FA (`fattn.cpp:159-175`), so the tile kernel this constant
feeds is never reached. That flatness is the check that the mechanism story is right.

Size is consistent with the arithmetic: the FA kernel is ~2% of decode, so even a large
occupancy gain on it caps near 2%.

### nsm is also wrong, and was deliberately NOT changed

`info.devices[i].nsm = prop.get_max_compute_units() / 16` uses the Xe-HPG ratio; Xe2 has
8 XVEs per Xe-core, so nsm reads 10 where the true Xe-core count is 20. Correcting it in
isolation would regress three paths whose empirical tuning absorbed the error:
`getrows.cpp:252` (GET_ROWS_WORK_GROUPS_PER_CU was swept to 4 against nsm=10; doubling
nsm doubles `min_items`, equivalent to 8, which that sweep rejected),
`topk-radix.cpp:232` and `count-equal.cpp:48`. Fixing nsm properly means re-tuning those
together. `max_wg_per_cu` has exactly one caller (FA), which is why it was safe to change
alone.

## Cast+add fusion: was UNREACHABLE, since FIXED (see correction at the end of this section)

`ggml_sycl_fuse_cast_add` in binbcast.cpp matches the real graph shape - a diagnostic
confirmed 8 occurrences of exactly `CPY(f16->f32) -> RESHAPE -> ADD` feeding TOP_K, so
`blk_bias` is true and the cast is real. But it fires ZERO times, because
`ggml_can_fuse_subgraph` can never accept it: `ggml_node_has_n_uses`
(`ggml/src/ggml-impl.h:663-665`) rejects any node with `view_src` set, and
`ggml_reshape_3d` is a view by construction. The helper additionally requires all nodes
in the subgraph to share a shape, which `[n_kv,n_tps,1,n_stream]` and
`[n_kv,n_tps,n_stream]` do not.

Making it fire needs a bespoke use-check rather than the shared helper, including
handling `ggml_cast` setting `result->src[1] = result` (a self-reference that inflates
use counts). Estimated worth ~1% of prefill, so it is parked, not abandoned. The
`(F32,F16,F32)` branch added to `ggml_sycl_op_bin_bcast` is currently unused.

### CORRECTION (2026-09-19): the bespoke matcher was written and this now fires

The fix prescribed above was implemented. `ggml_sycl_cast_add_shape` (binbcast.cpp:328)
is a bespoke matcher that does not go through `ggml_can_fuse_subgraph`; its comment
records why ("the reshape between the cast and the ADD"). `ggml_sycl_can_fuse_cast_add`
calls it instead of the shared helper, and `GGML_SYCL_FUSE_CAST_ADD` defaults to 1, so
this is currently the ONLY fusion in the QSA area enabled by default (the four
GGML_SYCL_FUSE_QSA_* flags all default to 0).

`ggml_sycl_cast_add_shape` is also reused by the QSA top-k fusion (topk-radix.cpp:449)
as part of its longer chain, so it is load-bearing even where the standalone path does
not fire.

The dispatch in `ggml_sycl_fuse` (topk-moe.cpp) is an explicit largest-first cascade:
qsa_mask -> qsa_topk -> qsa_gather -> cont_add -> cast_add -> topk_moe, so the biggest
applicable fusion wins and the smaller ones act as fallbacks. No collision.

STILL UNMEASURED: the "~1% of prefill" estimate above was made while the fusion never
fired and has not been re-measured since. A/B GGML_SYCL_FUSE_CAST_ADD=0 vs 1 to find out.

---

# Update 2026-09-18: measured peak map, allocator fix, and what is left

## The ggml-alloc view-inplace bug (landed, -128 MiB/card)

`ggml_gallocr_allocate_node()` has a branch meant to let an inplace op reuse the buffer
behind a *view* parent, e.g. `relu(reshape(mul_mat))`. It is unreachable. The guard above
it calls `ggml_gallocr_is_own(parent)`, which returns `hn->allocated`, and `allocated` is
only ever set inside `if (... && !ggml_impl_is_view(node))`. A view parent is therefore
never "own". The branch is also broken if reached: it reads `p_hn->addr`, which is never
populated for a view, and its `view_src->data == parent->data` guard is vacuous because
both are NULL at allocation time.

Fix: test ownership on `ggml_impl_is_view(parent) ? parent->view_src : parent`, inherit
`view_src_hn->buffer_id/addr`, and guard on `parent->view_offs == 0` plus equal nbytes.
Behind `GGML_ALLOC_INPLACE_VIEWS`, default 0.

Measured: 1893.85 -> 1765.85 MiB on SYCL0 (same -128 on all three cards). Prefill
465.93 -> 466.77 t/s, decode 23.30 -> 23.38 t/s (both noise), generated text
byte-identical, and no new `test-backend-ops` failures (18 pre-existing: 16 CONV_2D,
1 FLASH_ATTN_EXT, 1 ROLL, plus flaky borderline CPY tolerance misses).

This is upstream and backend-independent. It belongs in its own PR, not the SYCL series.

## The live map at the peak

Use the compile-time `GGML_ALLOCATOR_DEBUG` in ggml-alloc.c: it dumps every live tensor
sorted by offset on each peak raise. This is far more reliable than inferring from
peak-setter names, which led to two wrong conclusions earlier in this document.

SYCL0 peak with the fix on (1765.85 MiB): 23 live tensors, 1593.86 MiB live,
171.99 MiB in holes.

```
   53.35..  181.35   128 MiB  leaf_118            NONE        ne=[32768,1024]
  181.85..  437.85   256 MiB  attn_inp_kq_mask    NONE        ne=[131072,1024]  (f16)
  437.85..  477.85    40 MiB  l_last-2            DSV4_HC_POST
  477.85..  517.85    40 MiB  node_473            RMS_NORM
  517.85..  565.85    48 MiB  Qcur_full-3         MUL_MAT
  589.85..  613.85    24 MiB  node_493            MUL_MAT
  [ hole 613.85..741.85, 128 MiB ]
  741.85.. 1253.85   512 MiB  node_561            GET_ROWS    ne=[1024,131072]
 1253.85.. 1765.85   512 MiB  (cont)              CONT        ne=[131072,1024]
```

The 128 MiB hole is the bottom slice of `node_547`'s slot (the indexer score GEMM,
512 MiB, ne=[32768,4096], living at 613.85..1125.85). `node_561` is placed 128 MiB above
where `node_547` sat, and nothing can use the remainder because everything competing for
that region is 512 MiB. `cont#1` (128 MiB, ne=[32768,1024]) is NOT live at the peak; it
sits at 1125.85 at other moments and is freed once the GET_ROWS consumes it.

## CORRECTION to "The masks themselves are fine" above

That section claims the KQ mask lives in a host buffer and so is inside the 415.87 MiB
`SYCL_Host` line rather than the per-device figure. The live map disproves this:
`SYCL0#attn_inp_kq_mask#0` is 256 MiB in SYCL0's own chunk 0. Each device holds its own
copy. The mask is the single largest graph input in the per-device buffer.

## The gather is a 4x expansion

`qwen4exp.cpp:668-670` computes exactly `expanded[c, t] = score[cell_blk[c], t]`. The two
permutes and two conts exist ONLY because `ggml_get_rows` gathers along ne1 while `score`
needs gathering along ne0. So the graph transposes 128 MiB, gathers it into a 512 MiB
intermediate, then transposes that 512 MiB back.

Sizing what-if (make the gather reserve nothing, read the reserve number, discard the
run since its results are meaningless): 1765.85 -> 1429.85 on SYCL0 and 1773.10 ->
1355.85 on SYCL1/2, i.e. -336 and -417 MiB. Per-card wins differ because the layer split
is 31,33,36.

Because the gather EXPANDS (128 MiB source -> 512 MiB output), fusing it substitutes no
storage and should REDUCE bandwidth: today it writes 512 MiB then reads it back, versus
scattered reads over a 128 MiB source that already exists.

## The floor, and what reaches it

Drop both 512 MiB tensors and the live set at this peak instant is 569.86 MiB, of which
384 MiB is graph inputs (`op=NONE`, nothing to fuse): the 256 MiB mask and the 128 MiB
leaf_118. So the realistic floor is ~570-640 MiB, not lower.

Reaching it needs both:
1. fuse the GET_ROWS so the 512 MiB intermediate is never materialized (in progress)
2. fuse TOP_K to read `score[cell_blk[c],t] + mask[c,t]` on the fly, so the final 512 MiB
   `expanded` is never materialized either

Caveat: the buffer is the max over all 48 layers and every allocation moment, not this one
snapshot. Removing only the gather measured 1429.85 rather than the ~1130 this snapshot
alone would suggest, because a different moment then binds. Treat ~640 MiB as a direction,
not a number to bank.

## CANDIDATE: make the KQ mask bitwise (240 MiB)

`llama-kv-cache.cpp:1569-1570` writes only two values in the non-alibi case:

```c
const T mask_keep = llama_cast<T>(0.0f);
const T mask_drop = llama_cast<T>(-INFINITY);
```

It is an ADDITIVE mask: added to the scores so dropped cells become -inf, which softmax
turns into exactly 0. That is why the consuming kernel needs no branch, and why the tensor
is numeric rather than a bitfield. The second reason is ALiBi, which at line 1691 stores a
continuous `-|p0 - p1|` penalty, so the same tensor must hold arbitrary floats.

qwen4exp does NOT use ALiBi, so every entry really is one of two states. A bitmask would be
`131072 * 1024 / 8` = 16 MiB instead of 256 MiB, saving 240 MiB per card. That is
comparable to the entire GET_ROWS fusion, and after the two chain fusions land it is the
largest single remaining item in the per-device buffer.

Two obstacles. The mask is a shared graph input consumed by flash attention as well, so
every consumer changes, not just this chain. And kernels would trade a free fused add for
an unpack-and-select in the inner loop. Scope is therefore wider than the QSA chain and it
should be costed before being committed to.

## Option 3 follow-up: the mask is a per-device COPY, re-sent every evaluation

Investigated read-only while the TOP_K fusion was building. Two mechanisms matter, and
together they make the "compress at the boundary" framing the right one.

**1. Why every card holds 256 MiB.** The mask is created once as a graph input
(`llama-graph.cpp:977`, f16 when flash_attn is on, f32 otherwise). But in
`ggml_backend_sched_split_graph` (`ggml-backend.cpp:1452`):

```c
if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(...)) {
    // create a copy of the input in the split's backend
    tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
    ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
```

So each device that consumes the mask gets its own full-size duplicate allocated in ITS
compute buffer. That is exactly the `SYCL0#attn_inp_kq_mask#0` entry in the live map. It is
not one tensor seen three times; it is three tensors.

**2. It is re-transferred per evaluation.** `ggml-backend.cpp:1736`, inside
`ggml_backend_sched_compute_splits`:

```c
ggml_backend_tensor_copy(input, input_cpy);
```

Split inputs are copied on every graph evaluation, i.e. once per ubatch at prefill, not
once per model load.

**Consequence.** Compressing the mask to a bitfield is worth more than the 240 MiB of
buffer per card. It also cuts the host-to-device transfer of that tensor by the same
factor, on every ubatch, on every card. For a long prefill that is a repeated cost, and it
may show up as throughput rather than only as capacity.

**Why this makes the "option to allocate compressed" framing correct.** A bitmask is
worthless where a fusion already consumes the mask on the fly, because there the 256 MiB is
never materialized at all. It pays precisely where the mask must exist as a real buffer and
cross a device boundary. So the right shape is to let the INPUT be stored and transferred
compressed and expanded at the consumer, rather than changing the additive f16 contract
that `ggml_flash_attn_ext` and `ggml_soft_max_ext` rely on. That also sidesteps the
"every consumer must change" obstacle logged in the candidate section above.

**NOT YET MEASURED.** The 240 MiB is arithmetic from the tensor shape. The transfer saving
is inferred from the code path above, not timed. Before building anything:
  - confirm with a sizing probe (make the mask allocate 16 MiB, read the reserve) that the
    buffer saving is real, exactly as the -336/-417 gather estimate was obtained;
  - time the actual per-ubatch input copy, since `n_kv` at runtime may be smaller than the
    worst-case 131072 used at reserve, which would shrink the bandwidth argument.

---

# Update: the TOP_K fusion works, is bit-exact, and is worth 0 MiB (reverted)

Built, flag-gated `GGML_SYCL_FUSE_QSA_TOPK`, bit-exact against a CPU reference on 11 shapes
including production and 100% mask-drop, no new test-backend-ops failures, throughput flat
at 15k and 63.7k prefill. And it saves **nothing**: compute buffer identical in both arms
on all three devices, with a byte-identical 313-line peak trace.

Reverted (patch kept at scratchpad/qsa_topk_fusion.patch). The gather fusion and the
allocator fix are unaffected and remain in commit 6ed9a8be4.

**Why it pays nothing.** The premise it was built on was stale. After the gather fusion
landed, the 512 MiB `expanded` CONT stopped being a peak raiser at all: it lands in the hole
the indexer score GEMM leaves when that GEMM is freed. Removing a tensor that sits in a hole
frees no high-water mark. Verified independently, peak ladder with both landed flags on:

```
  699.10 <- ffn_moe_down-16   100 MiB  MUL_MAT_ID
 1125.85 <- node_547          512 MiB  MUL_MAT  ne=[32768,4096]  src0=indexer_k-3
 1253.85 <- (view) (cont)     128 MiB  CONT     ne=[32768,1024]
```

Lesson, consistent with the rest of this document: a tensor's SIZE does not predict its
contribution to the buffer. Only whether it raises the high-water mark does. Always re-read
the peak ladder after each landed change rather than reusing the previous target list.

## Text is NOT a valid acceptance criterion for QSA top-k changes

The gather fusion gave byte-identical text, so that was used as the bar here. It does not
hold for anything touching the top-k kernel: two runs of the SAME arm also diverge. The QSA
top-k input is massively tied (all r=4 cells of a block carry the same score, and the mask is
0 or -inf), and the radix emit assigns pivot-equal columns in `atomic_ref::fetch_add` order,
which is not reproducible. `build_attn_qsa` then sums attention in that order.

A permutation-only change like the gather cannot alter which ties the emit picks first, which
is why it WAS byte-reproducible. Any change to the top-k lane schedule is not. Use a
sorted-values-at-returned-indices bit-exactness check against a host reference instead.

## Next target: the indexer score GEMM epilogue (CORRECTED: ~80 MiB, and only WITH the TOP_K fusion)

`qwen4exp.cpp:646-658`. `ggml_mul_mat(pooled, q)` produces [n_blocks, n_idx_h*n_tps] =
[32768, 4096] = 512 MiB, is relu'd inplace, then collapsed to [n_blocks, n_tps] (128 MiB) by
summing 4 head slices into a CONT. Fusing the relu and the head sum into the GEMM epilogue
would write 128 MiB instead of 512 and kill the 128 MiB head-sum CONT as well. Those are
precisely the two tensors at the top of the ladder above.

### CORRECTION (same day): this is worth ~80 MiB, not ~512, and needs the TOP_K fusion

The estimate above was wrong. It sized the win from the two tensors' bytes, which is exactly
the error this document keeps recording. The measured what-if (force the GEMM AND its relu to
reserve nothing) says:

| | SYCL0 | SYCL1/2 | peak then set by |
|---|---|---|---|
| QSA_TOPK off | 1253.85 (unchanged) | 1261.10 | the 512 MiB `expanded` CONT ne=[131072,1024] |
| QSA_TOPK on  | 1173.85 | 1181.10 | `node_5107` FILL 256 MiB, then `node_5112` FLASH_ATTN_EXT 304 MiB |

So the GEMM epilogue ALONE is worth 0 MiB: remove it and `expanded` immediately becomes the
peak setter in its place. GEMM + TOP_K together are worth about 80 MiB. Treat 80 as an order
of magnitude; the what-if is crude and also breaks the relu's inplace reuse.

This makes the TOP_K fusion a PREREQUISITE for the GEMM work rather than an independent win,
which is why it was restored to the tree (flag-gated default OFF, so inert until enabled).

### The real remaining wall

`node_5107` FILL 256 MiB ne=[131072,1024] src0=attn_inp_kq_mask (the QSA path materialising
its own copy of the KQ mask) plus FLASH_ATTN_EXT 304 MiB. That is option 3 territory, and it
is now the LARGER target: ~560 MiB of wall versus ~80 MiB for both remaining chain fusions.
Option 3 should therefore come BEFORE the GEMM epilogue, reversing the ordering stated above.

---

# Update 2026-09-19: the indexer score GEMM epilogue fusion (-80 MiB, measured)

Built as a SYCL backend fusion, flag-gated `GGML_SYCL_FUSE_QSA_SCORE`, default OFF. New files
`ggml/src/ggml-sycl/qsa-score.{cpp,hpp}`. No model-graph change, so no other backend is affected.

## What it matches and what it absorbs

`qwen4exp.cpp:646-658` emits, per layer:

```
i    MUL_MAT   [n_blocks, n_head*n_tps, n_stream]      <- absorbed
i+1  RESHAPE   (view of the MUL_MAT)
i+2  UNARY/RELU                                        <- absorbed
i+3  VIEW head 0
i+4  CONT                                              <- absorbed
i+5  VIEW head 1
i+6  ADD                                               <- absorbed
i+7  VIEW head 2
i+8  ADD                                               <- absorbed
i+9  VIEW head 3
i+10 ADD  = the only node that keeps a buffer
```

Confirmed with `GGML_SCHED_DEBUG=2`: `5 nodes absorbed by backend fusion`, the five marked above,
and the reshape and the four head views keep their normal bookkeeping - marking a view absorbed
strands its parent, exactly as the top-k fusion found.

## How the GEMM avoids the wide product

The reduction is over the GEMM's ne1 in groups of n_head, so the product is computed in column
tiles: for each tile of tokens, one library GEMM (oneDNN, same primitive the unfused path uses)
writes `[n_blocks, n_head*t_tile]` into a pool buffer, then one kernel does relu plus the head sum
straight into the final ADD's buffer. `GGML_SYCL_QSA_SCORE_TILE_MIB` caps the pool buffer, default
32. At 128k the tile is 64 tokens (16 tiles per layer); at 15k it is 559 (2 tiles); at decode
(n_tps = 1) it is a single tile and the fusion turns 6 kernel launches into 2.

This is an honest trade, not a free win: 80 MiB leaves the compute buffer and up to 32 MiB per card
appears in the SYCL pool instead. Net VRAM is about -48 MiB per card.

## Measured, GGML_ALLOC_INPLACE_VIEWS=1 GGML_SYCL_FUSE_QSA_GATHER=1 GGML_SYCL_FUSE_QSA_TOPK=1

| | SYCL0 | SYCL1 | SYCL2 |
|---|---|---|---|
| off | 1253.85 | 1261.10 | 1261.10 |
| on  | 1173.85 | 1181.10 | 1181.10 |

Exactly the -80 MiB the what-if predicted, and the peak moves exactly where it predicted:

```
off:  696.89 <- ffn_moe_down-0    100 MiB  MUL_MAT_ID
     1125.85 <- node_547          512 MiB  MUL_MAT         ne=[32768,4096]
     1253.85 <- (view) (cont)     128 MiB  CONT            ne=[32768,1024]

on:   696.89 <- ffn_moe_down-0    100 MiB  MUL_MAT_ID
      741.85 <- indexer_score-3   128 MiB  ADD             ne=[32768,1024]
      869.85 <- node_571          256 MiB  FILL            ne=[131072,1024] src0=SYCL0#attn_inp_kq_mask#0
     1173.85 <- node_576          304 MiB  FLASH_ATTN_EXT  ne=[256,24,1024]
```

The wall is now the QSA mask chain, 560 MiB of FILL plus FLASH_ATTN_EXT, which is option 3
territory and the next target.

## Throughput, two runs per arm

| | off | off | on | on |
|---|---|---|---|---|
| prefill 15k  | 468.82 | 470.48 | 467.79 | 467.04 |
| prefill 63.7k| 409.60 | 410.15 | 412.87 | 411.78 |
| decode short | 23.68/23.66 | 23.66/23.72 | 23.55/23.53 | 23.64/23.68 |
| decode 15k   | 19.95 | 19.87 | 19.76 | 19.81 |
| decode 63.7k | 13.05 | 13.07 | 13.14 | 13.08 |

Flat: every delta is under 0.7% and the signs are mixed.

## Correctness

Standalone host-reference check on 9 shapes through `ggml_backend_sched` (so `fusion_absorbs`
really applies), including the production shape and n_stream 2 and 3: both arms land the same
distance from a double-precision CPU reference, max |err| about 1.3e-5 against values up to 31.

NOT bit-exact against the unfused path at every shape. Where the tile count is 1 it is bit-exact;
where the GEMM is split into tiles oneDNN picks a different blocking, and the two arms differ by at
most 1.1e-5, i.e. 3.7e-7 of the peak magnitude. So generated text is not reproducible across the
flag - but it already is not reproducible across two runs of the SAME arm once the top-k fusion is
on. Measured same-arm floor: 2/5 prompts identical off vs off, 3/5 on vs on, similarity down to
0.16 on a prompt; cross-arm sits inside that band at 1-3 of 5 identical.

`test-backend-ops test -b SYCL0`: same failure set in both arms - the 18 known pre-existing
(16 CONV_2D, 1 FLASH_ATTN_EXT, 1 ROLL). The extras that showed up on one side only
(ADD_ADD f16, two CPY, one MUL_MAT_ID q5_0) all pass in isolation, three runs per arm.

---

# Update 2026-09-19 (2): the QSA mask chain fusion - bit-exact, 0 MiB, +12% prefill

Built as a SYCL backend fusion, flag-gated `GGML_SYCL_FUSE_QSA_MASK`, default OFF. New files
`ggml/src/ggml-sycl/qsa-mask.{cpp,hpp}`. No model-graph change.

## What it matches

`qwen4exp.cpp:732-758` emits five CONTIGUOUS nodes per QSA layer:

```
i    FILL      [n_kv, n_tps, 1, n_stream] -inf, src0 = the kq mask copy   <- absorbed
i+1  VIEW      the same tensor as rows of one cell
i+2  SET_ROWS  writes 0.0f at the top-k cells (itself a VIEW of the FILL)
i+3  VIEW      back to the mask shape
i+4  ADD       + kq_mask = the only node of the chain that keeps a buffer
```

The set_rows payload is a sixth node, a FILL of zeros, and it is NOT adjacent: the graph emits
it exactly 53 nodes earlier (node_5054 against node_5107 for layer 35, and the same gap in
every layer). That is plain DFS order, not a reordering pass. `ggml_visit_parents` takes
SET_ROWS src[0], the zeros, first; then src[1], the top-k index view, which drags the whole
indexer tail (score GEMM, relu, gather, TOP_K, CONT) into the graph before it ever reaches
src[2] and the mask FILL. The matcher therefore anchors on the mask FILL and reaches the zeros
through `src[]`, never by node index. A matcher that assumed the seven nodes were contiguous
matched in a standalone harness, where `top_k` is an input leaf, and never fired in the model.

Since `-inf + x == -inf` and `0 + x == x`, the chain is exactly
`out[c, t] = selected(c, t) ? kq_mask[c, t] : -inf`, which one fill plus one scatter writes
straight into the ADD's buffer.

## Worth 0 MiB, because the FILL's buffer already IS the final mask

`ggml_set_rows` returns a VIEW of its `a` argument, so SET_ROWS never had a buffer. And with
the inplace-over-view fix on, ggml-alloc gives the ADD the FILL's buffer. The chain therefore
materialises exactly ONE 256 MiB tensor in either arm, and that tensor is the mask flash
attention reads. `GGML_ALLOC_INPLACE_DEBUG=1`, flag off:

```
[INPL] node_575 (256 MiB) parent (view)(view)(view): n_children=1 n_views=0 is_view=1
[INPL]   view_src node_571: n_children=0 n_views=1 same_data=1
```

By the time the ADD is allocated the three views of the FILL have all been released, so
`n_views` is 1 and `n_children` is 0 and the branch fires. Nothing is allocated or freed
between the FILL and the ADD, so the allocator hands the ADD the same offset, and the ladder
is identical except for the name on one line:

```
off:  869.85 <- node_571  (256.00 MiB) op=FILL ne=[131072,1024] src0=SYCL0#attn_inp_kq_mask#0
on:   869.85 <- node_575  (256.00 MiB) op=ADD  ne=[131072,1024] src0=(view)(view)(view)
both: 1173.85 <- node_576 (304.00 MiB) op=FLASH_ATTN_EXT
```

Measured both arms, three devices, `GGML_ALLOC_INPLACE_VIEWS=1 FUSE_QSA_GATHER=1
FUSE_QSA_TOPK=1 FUSE_QSA_SCORE=1 SPARSE_FA=0`: 1173.85 / 1181.10 / 1181.10 in BOTH arms.
The fusion needs no pool buffer, so net VRAM is 0 as well.

## It is worth exactly 256 MiB when the allocator fix is off

Same binary, same arms, `GGML_ALLOC_INPLACE_VIEWS=0`:

| | SYCL0 | SYCL1 | SYCL2 |
|---|---|---|---|
| off | 1125.85 | 1133.10 | 1133.10 |
| on  |  869.85 |  877.10 |  877.10 |

Exactly -256.00 MiB on all three. The fusion and the allocator's inplace-over-view fix are two
routes to the same 256 MiB and they do not stack. The allocator fix got there first.

## CORRECTION to "The real remaining wall"

That section reads the ladder as "560 MiB of wall, FILL 256 plus FLASH_ATTN_EXT 304". The
FILL's 256 MiB is not a redundant copy a fusion can delete: it is the mask itself, under the
name of whichever node happens to allocate it. The wall is 304 MiB of FA output plus 256 MiB
of irreducible mask, and no rearrangement of the chain touches the second number.

Removing it needs flash attention to stop reading a dense [n_kv, n_tps] mask:
- absorb the whole chain INCLUDING the ADD and hand the FA kernel `kq_mask` plus the `top_k`
  list. `GGML_SYCL_SPARSE_FA` already passes a sparsity hint but still reads the dense mask,
  which is why it does not change the reserve;
- or the bitfield mask candidate, which shrinks the graph input as well.

The matcher in `qsa-mask.cpp` is most of the plumbing for the first option.

## Where it does pay: prefill throughput

The unfused chain moves about 4x the bytes the fused one does: FILL writes the mask, then ADD
reads two masks and writes a third. Four arms under a GPU mutex, order on/off/off/on so that
page-cache warming cannot favour one side:

| | off | off | on | on |
|---|---|---|---|---|
| prefill 15k   | 333.03 | 332.74 | 374.99 | 373.86 |
| prefill 63.7k | 159.57 | 159.55 | 169.51 | 169.21 |
| decode short  | 23.68/23.75 | 23.68/23.65 | 23.70/23.70 | 23.69/23.68 |
| decode 15k    | 19.89 | 19.84 | 19.83 | 19.85 |
| decode 63.7k  | 13.14 | 13.15 | 12.99 | 13.01 |

+12.4% prefill at 15k and +6.2% at 63.7k, with under 0.4% spread inside each arm. Decode is
flat, except 63.7k which is 1.1% down and consistently so - small, but the sign does not vary.

These absolute numbers are much lower than the 468 / 410 baseline recorded above because the
tree carried another in-flight change to the SYCL flash attention at the time. The A/B is
sound: every arm ran on one binary, alone on the GPUs under the lock.

## Correctness

Standalone check through `ggml_backend_sched` on 9 shapes (production 131072 x 16 x 2051,
decode n_tps = 1, n_stream 2 and 3, f16 and f32 masks, duplicate indices, and rows where the
top-k picks cells the causal mask already dropped): zero mismatches against a host reference
AND byte-identical output between the two arms on every shape. The fused path copies the f16
mask word for word rather than adding 0.0f to it, so `-inf` and every finite value survive
untouched.

---

# Consolidated state and corrected roadmap (2026-09-19)

## Where the buffer actually stands

Measured, `GGML_ALLOC_INPLACE_VIEWS=1 FUSE_QSA_{GATHER,TOPK,SCORE}=1 SPARSE_FA=0`:
SYCL0 **1173.85 MiB**, SYCL1/2 1181.10. Down from 1893.85/1901.10 originally (-38%).

Peak ladder ends: `ADD` 256 MiB (the QSA mask) at 613.85..869.85, then `FLASH_ATTN_EXT`
304 MiB at 869.85..1173.85, of which only 24 MiB is output and ~280 MiB is f16 staging
for the q8_0 KV cache.

## Corrected: an earlier estimate of ~638 MiB was WRONG

That figure assumed the mask-chain fusion would remove 256 MiB AND the FA clamp another
280. It double-counted: the mask fusion and the `GGML_ALLOC_INPLACE_VIEWS` allocator fix
are two routes to the SAME 256 MiB and they do not stack (measured: fusion is worth 0 MiB
with the allocator fix on, and exactly -256 MiB with it off).

Realistic floor from the currently planned work is therefore about **870-894 MiB**, not 638.

## What is left, in order of size

| Target | Size | Status |
|---|---|---|
| FA f16 staging for the q8_0 KV cache | ~280 MiB | clamp implemented, measurement pending |
| The dense QSA mask `ADD` | 256 MiB | needs FA to consume `top_k` directly (see below) |
| `attn_inp_kq_mask` graph input | 256 MiB | bitmask candidate, cross-cutting |
| `leaf_118` graph input | 128 MiB | STILL UNIDENTIFIED, cheapest thing to investigate |
| fragmentation | ~42 MiB | allocator packing order, unexplored |

## The only route below ~870

Flash attention currently reads a dense `[n_kv, n_tps]` mask, so the 256 MiB must be
materialised no matter how the chain that builds it is rearranged. `GGML_SYCL_SPARSE_FA`
does NOT change this: it passes a sparsity hint and still reads the dense mask, which is
why it was measured as not moving the reserve at all.

Removing it means absorbing the mask ADD as well and handing the FA kernel `kq_mask` plus
the `top_k` index list directly, so the selected mask is never built. The QSA mask fusion's
matcher is most of the plumbing for that.

## Two traps recorded for whoever continues

1. The QSA chain is NOT contiguous in the graph. The zeros `FILL` is emitted 53 nodes
   before the mask `FILL`, because DFS follows SET_ROWS `src[1]` (the top-k index view) and
   drags the whole indexer tail in first. A matcher assuming adjacency validates perfectly
   in a standalone harness, where `top_k` is an input leaf that pulls in nothing, and then
   silently never fires in the model. Anchor on the mask FILL and reach the rest via `src[]`.
2. A standalone harness can validate SEMANTICS but cannot validate ADJACENCY assumptions.

---

# Why this mattered: master is 40 MiB from the wall (measured 2026-09-19)

During the end-to-end verification, the SYCL runtime's per-device free-memory query at peak
on STOCK MASTER read:

    free = 40 / 1485 / 769 MiB   (SYCL0 / SYCL1 / SYCL2)

**SYCL0 has 40 MiB of headroom on master.** That is not an optimisation target, it is a
stability cliff. `run_llama_flash_next.sh` already records the consequence of crossing it:
with an even 33,33,34 split, SYCL0 goes 135 MiB over and the whole thing "COLLAPSES to
0.15 t/s via Level Zero host eviction". The 31,33,36 split exists solely to keep this card
under its limit, and even so the margin is 40 MiB.

So the compute-buffer work (1893.85 -> 869.85 MiB per card) should be understood as buying
headroom against an eviction cliff, not as freeing memory for its own sake. Concretely it
is what makes a longer context, a less carefully hand-tuned tensor split, or a slightly
larger model viable on this hardware at all.

Master baseline reproduced the historical record closely in the same run: 16384-token warm
prefill 297.82 t/s here against 297.84 recorded earlier, and the full `test-backend-ops`
failure set came back as exactly the 18 known cases (16 CONV_2D, 1 ROLL, 1 FLASH_ATTN_EXT)
with no flaky extras.

## Methodology finding: the same-arm text floor is far looser than assumed

From the same run, master pass 1, two reps of the same prompt in the SAME server instance
with greedy decoding:

- prompt a: not identical, similarity 0.378, common prefix 451 chars of ~1300.
- prompt b: not identical, similarity 0.144, common prefix 106 chars. Rep 1 produced 1368
  chars of correct analysis; **rep 2 stopped after a single 106-char sentence** (early EOS).

Same binary, same arm, same process. The floor therefore admits not only divergent phrasing
but a complete change in output LENGTH. Any cross-arm text comparison on this workload can
only be qualitative (coherent, on-topic, correct), and a length or content difference between
two arms is not by itself evidence of a defect. This retroactively justifies the
"text is not bit-reproducible" caveat used throughout this document.

Root cause is upstream of any change here: the QSA top-k radix emit assigns pivot-equal
columns in `atomic_ref::fetch_add` order, and `build_attn_qsa` then sums attention in that
order. See [[qsa-topk-nondeterministic]] in the session memory.

---

# Update: the QSA FA mask fusion (GGML_SYCL_FUSE_QSA_FA_MASK), 2026-09-19

Flash attention now derives selection from the `top_k` index list in-kernel, so the 256 MiB
dense QSA mask is never materialised. Behind `GGML_SYCL_FUSE_QSA_FA_MASK`, default 0.
Mode 1 = bit is selection only, FA still reads the causal mask. Mode 2 = bit is selection AND
causality, FA never reads the mask.

## Result: -128 MiB, memory only

| | SYCL0 | SYCL1/2 |
|---|---|---|
| baseline | 869.85 | 877.10 |
| mode 1 or 2 | **741.85** | **749.10** |

-128.00 MiB per card, six independent server starts. Net VRAM -126 MiB/device (mode 1).
Throughput: prefill -3.5% at 15k, +1.1% at 64k, decode flat. Failure sets identical across
FA_MASK 0/1/2. Text cross-arm inside the same-arm floor.

## SHIP MODE 1, NOT MODE 2

Reserve is identical for both modes. Net VRAM is not: mode 2's pool is reproducibly ~100 MiB
larger (1338 MiB in three runs, against 1237/1244 for mode 1), costing about 34 MiB/device to
buy +0.5% throughput. **The cause is unexplained and was reported as unexplained.** A
candidate worth testing is the pool caching one distinct bitmap size per n_kv step (63
distinct sizes across a 64k prefill); rounding the bitmap allocation to a fixed granularity is
the cheap experiment, untested.

This only became visible because net VRAM was required alongside reserve. Reserve alone would
have said the two modes were identical.

## CORRECTION: the ~614 MiB prediction was wrong in its PREMISE

The prediction assumed nothing sat between `node_493` (ending 613.85) and the mask ADD at
869.85. The **indexer score ADD, 128 MiB at 613.85..741.85**, does. It was invisible in the
ladder only because the mask ADD was taller. Removing a 256 MiB tensor therefore bought 128,
and `indexer_score-3` is the new peak setter.

Generalised lesson, beyond "size does not predict contribution": a peak ladder shows what
currently RAISES the high-water mark, not what is queued immediately beneath it. Predicting
the result of removing the top entry requires knowing the next-tallest LIVE tensor, which the
ladder does not show. Use the compile-time `GGML_ALLOCATOR_DEBUG` live map for that.

## Sparse prefill: definitively dead, measured

Union ratio of `top_k` over a token window, n_kv=50176, width=2051 (density 0.0409):

| block (tokens) | measured union | independence model |
|---|---|---|
| 1 | 0.041 | 0.041 |
| 32 | 0.265 - 0.394 | 0.737 |
| 160 | 0.474 - 0.752 | 0.999 |
| 1024 | 0.752 - 0.954 | 1.000 |

Adjacent tokens overlap far more than independence predicts, so the earlier model was wrong.
The conclusion survives on a better number: **`chunks_touched = 1.0000` at every block size
from 8 tokens up, and 0.98-1.00 even for a SINGLE token.** One token's 2051 cells are spread
across essentially every 8192-cell chunk, so no tile is empty at any granularity. **Tile
skipping is settled: dead.**

Still open, different design: GATHERING rather than skipping. A union of 0.47-0.75 at a
160-token tile implies ~0.6x compute for ~4x K/V traffic, and dense MKL sits at ~6000
flop/byte on K/V, so 4x of a negligible term stays negligible. Plausibly ~1.5x on FA, and the
union should improve at 131k where density halves. Substantial separate project.

## Pre-existing bug found in passing

`ggml/src/ggml-sycl/set_rows.cpp:271` computes `dst + dst_row*nb1` with NO range check, so an
out-of-range index scribbles into the neighbouring row. Out of contract for `ggml_set_rows`,
so not a defect this work introduces, but it means the unfused path and every fused path
diverge on out-of-range input (the fused paths skip it). Worth reporting upstream.

---

# CRITICAL 2026-09-19: two QSA fusions produce CORRUPT OUTPUT

Found by the end-to-end verification against master, after these flags had already been
made default-ON on my instruction. Both are now default 0.

## The two defects

1. **`GGML_SYCL_FUSE_QSA_MASK`** corrupts once a prefill of about 16k tokens or larger has
   been processed. Short prompts pass. One large prefill then POISONS THE SERVER: the same
   short prose prompt returns coherent text before it and garbage after, until restart.
2. **`GGML_SYCL_FUSE_QSA_FA_MASK=1`** corrupts at ANY size. A ~100-token prose prompt in a
   fresh session with no large prefill returned garbage. Mode 1 is broken outright, not a
   long-context edge case.

Signature: one plausible token then a solid run of `/` or `!`. Real samples:
`'\n\n///////////////////////////////////////////////////////////'`, `'7!!!!!!!!!!!'`,
`'ly//////////'`, `'相当!!!!!!!!!!'`.

## The cumulative ladder that found it

Binary sha256 `2a3823f89132d1ef` throughout, verified unchanged after the run, so every rung
means exactly what its name says. Probe: prose -> 16384 prefill -> prose.

| rung | flags (cumulative) | before | 16k | after | SYCL0 reserve |
|---|---|---|---|---|---|
| F0 | none | OK | OK | OK | 1893.85 |
| F1 | +ALLOC_INPLACE_VIEWS | OK | OK | OK | 1765.85 |
| F2 | +FUSE_QSA_GATHER | OK | OK | OK | 1253.85 |
| F3 | +FUSE_QSA_TOPK | OK | OK | OK | 1253.85 |
| F4 | +FUSE_QSA_SCORE | OK | OK | OK | **869.85** |
| F5 | +FUSE_QSA_MASK | OK | **CORRUPT** | **CORRUPT** | 869.85 |
| F6 | +SPARSE_FA +FUSE_QSA_FA_MASK=1 | **CORRUPT** | **CORRUPT** | **CORRUPT** | 869.85 |

`SPARSE_FA` alone is clean at 16k and 65k, so it is not implicated in F6.

## WHY IT HID FOR HOURS, and the structural lesson

Throughput was UNAFFECTED and looked spectacular: 528.64 t/s prefill at 16k against master's
292.86, and 13.71 t/s decode at 113k against master's 6.52 - while every generation was
`'ly//////////'`. `predicted_n` was correct, timings were clean, no error was logged.

But the deeper reason is structural, and it is the lesson to carry:

**Each fusion agent DID check generated text. Each compared branch-with-its-flag against
branch-without-its-flag, and in both arms the REST of the QSA set was enabled. So both arms
were corrupt, the texts matched each other, and the check reported "no difference, inside the
same-arm noise floor".** Comparing two broken configurations against each other is blind by
construction. The `FUSE_QSA_FA_MASK` verification did exactly this and concluded its change
was clean.

Only a comparison against MASTER exposes this. An A/B within a branch validates a DELTA, never
a BASELINE. Any flag-gated change must be text-checked against master at least once, at a
context length large enough to exercise it.

## The memory win is unaffected

F4 is the maximum safe set and already reaches the full reserve. `FUSE_QSA_MASK` contributes
**exactly 0 MiB** on top of it: pure risk, no benefit at this combination.

Shipped defaults are therefore F4 + SPARSE_FA:

    INPLACE_VIEWS=1  GATHER=1  TOPK=1  SCORE=1  SPARSE_FA=1
    MASK=0  FA_MASK=0

Still -1024.00 MiB per card versus master, with correct output at every probed size.

## Over-reverting has a cost too

On the first warning I reverted TOPK and SCORE to 0 as well. The ladder then showed F3 and F4
clean, so that would have silently given up -384 MiB of a safe win. Reverting beyond the
evidence is not a free "safe" choice; it discards verified value. Revert to the last rung the
evidence actually clears, not further.

## Throughput that survives

`SPARSE_FA` alone, correct at every probed size, single reps against master:
16k prefill 242.3 -> 326.7 (+34.8%), 16k decode 15.39 -> 20.75 (+34.8%),
65k prefill 247.7 -> 353.3 (+42.6%), 65k decode 8.67 -> 15.79 (+82.1%).

Every throughput figure previously recorded for the all-on branch configuration is VOID.

## ROOT CAUSE of the QSA mask corruption (found 2026-09-19)

**One defect, not two.** The fused kernel runs at the chain's FIRST node but writes the LAST
node's buffer, and the allocator frees a source in between.

Chain from `src/models/qwen4exp.cpp:736-762`:

    [i] FILL(kq_mask,-inf)  [i+1] VIEW  [i+2] SET_ROWS(zeros, top_k_3d, view)  [i+3] VIEW  [i+4] ADD

- `ggml_sycl_qsa_mask_absorbs` (qsa-mask.cpp:233-250) reports ONLY the FILL as absorbed; the
  three views deliberately "keep their normal bookkeeping" (comment at 230-232).
- Therefore at the SET_ROWS (node i+2), `ggml_gallocr_release_parent` (ggml-alloc.c:918-926)
  frees `tk` (the top-k index list) and `zeros`.
- At node i+4 the ADD gets a best-fit slot. The 4-byte index list is exactly twice the 2-byte
  f16 ADD, so that slot is regularly **the top-k list itself**.
- But the fusion executes at node i (ggml-sycl.cpp:6540-6545) and writes `dst = ad->data`:
  `k_qsa_mask_drop` fills the whole ADD buffer with -inf, then `k_qsa_mask_keep` reads the
  index list it has just overwritten. Indices become 0xFC00FC00, negative, rejected by the
  `c < 0` guard, so every row keeps no cell -> all -inf -> NaN out of softmax -> argmax lands
  on token 0 (`!`) or a low id (`/`).

Evidence: an LD_PRELOAD probe (`scratchpad/qsa_probe.cpp`) reporting literal overlapping
address ranges in a live server, e.g. `add=[0xffffd55be0800000+520192]
topk((cont))=[0xffffd55be0800000+1040384] OVERLAP`; plus a GPU-free CPU repro
(`scratchpad/alloc_repro.cpp`) reproducing the overlap through `ggml_gallocr` alone.

### The threshold is 512 tokens and it is a RE-LAYOUT event, not a size

448 tokens clean, 512 corrupt, and BOTH have the same padded `n_kv = 512`. The 512-token graph
is simply the first to fail `ggml_gallocr_needs_realloc` and force a new layout; the chain's
ADD moves from 0xffffd55c06dda080 to 0xffffd55be0800000 and lands on the top-k list. Every
description of this as a "long context" bug, including mine, was wrong.

"Poisons the server" needs no persistent-state damage: ggml-alloc keeps a layout until a graph
stops fitting it, so the poisoned layout is reused by every later request until restart.

### FUSE_QSA_FA_MASK mode 1 never executed in llama-server

Its reader requires `ggml_sycl_fattn_picks_mkl`, i.e. `Q->ne[1] >= 32 && K->ne[1] >= 1024`
(fattn.cpp:126-127) AND oneDNN not chosen, and the stage cap needs `n_kv >= 131072`. So it is
unreachable for decode, for prompts under 1024 cells, and for any prefill under full context.
A traced arm logged ZERO `[QSAFA]` lines. Setting the flag only changed the reserve-time plan
so the re-layout happened on the first graph.

**Consequence: every measurement of that fusion is VOID** - the -128 MiB, the -126 MiB net, the
-3.5%/+1.1% throughput, and the mode 1 versus mode 2 net-VRAM comparison all describe a
configuration whose kernel never ran.

### Fix: bounded, two files, CPU-verified, not yet built

1. `ggml_sycl_qsa_mask_absorbs`: report the FILL **and the three views** (span 4; 5 with
   FA_MASK). Then nothing inside the span is freed and the ADD cannot alias a live source.
2. `ggml_gallocr_release_parent`: when the released parent is an absorbed view, also release
   that view's own `src[]` recursively, or absorbing the views strands the top-k list.

Gating on n_kv is NOT a valid mitigation: the overlap depends on the whole graph's layout and
was observed at n_kv=256 once the layout had shifted.

### LATENT RISK: the same pattern exists in three fusions that are now default ON

`GATHER`, `TOPK` and `SCORE` share the structure "kernel runs at the chain's first node and
writes a later node's buffer while non-absorbed views in between release sources"
(topk-radix.cpp:526-528 records the same deliberate choice). They were clean at every probed
layout, but this failure is layout-dependent, so that is not proof. The LD_PRELOAD probe
extends to them without a rebuild and that audit is the highest-priority open item.

## Audit of the three default-ON fusions: clean, but exposed by luck not design

Run with the LD_PRELOAD probe inside a real llama-server, shipped flags explicit
(`INPLACE_VIEWS=1 GATHER=1 TOPK=1 SCORE=1 SPARSE_FA=1 MASK=0 FA_MASK=0`), binary
`2a3823f89132d1ef`:

**121 probed graphs, 12 TOPK and 12 SCORE chains each, every one confirmed firing, ZERO
hazards.** Coverage included roughly 80 allocator re-layouts: the 512-token event that breaks
MASK, every 1024-token ubatch step of the 16k and 65k prefills, and the decode and prose
graphs after each prefill (the persistent post-re-layout states). Corroborated by a CPU twin
whose detector was validated by enabling MASK, which reproduced both of MASK's known victims.

**But the structural exposure is real and was confirmed by code reading, not just absent:**

- SCORE's GEMM inputs `pooled` and `q` are released by the RESHAPE at run+1, and its output is
  placed 10 nodes later.
- TOPK's `score` is released by the PERMUTE view at run+2, and the TOP_K output is placed 5-7
  nodes later.

They survive on BUFFER SIZE RATIOS, not by construction. SCORE's output is about 8x smaller
than the freed `q` region, so best-fit preferred other holes. TOPK's output is about 4x larger
than the freed `score`, so only a merged free block could have held it. MASK's was 2:1, which
made the freed index list the perfect fit, and that is why MASK is the one that broke.

So the correct statement is "clean at every layout this server produces for this model at
68/448/512/16k/65k with -ub 1024", NOT "safe". Untested: other ubatch sizes, other n_ctx,
other tensor splits, `n_stream > 1`, and the post-fix plan (absorbing four more nodes per
layer changes every later placement).

## THE RULE, for anyone writing a fusion in this tree

**A fusion that RUNS at node i and WRITES node j must report every node in [i, j) as absorbed.**

Otherwise ggml-alloc frees the kernel's inputs somewhere inside the span (correctly, for the
schedule it was told about) and may then hand node j's buffer that freed memory. The kernel
then destroys its own input. Both the allocator and the fusion are individually correct; the
contract between them is what breaks.

Absorbing the span also requires `ggml_gallocr_release_parent` to recurse through absorbed
view nodes, or the span's sources are never freed at all (see `scratchpad/ggml-alloc-fix.diff`).

Four fusions in this tree currently run at a chain's first node and write a later node's
buffer. MASK broke; GATHER, TOPK and SCORE have not, at the layouts probed.

**Regression gate:** `bash gpu_run.sh ns_probe2_arm.sh <label> "<flags>" "448 512 16384 65536"`
and require `real hazards: 0`. Note that `test-backend-ops` CANNOT see this bug class at all,
because it builds single-op graphs and these fusions need a multi-node chain plus
`fusion_absorbs`; its exact match with master therefore proves nothing here either way.

---

# END-TO-END vs MASTER, shipped safe set (2026-09-19) - SUPERSEDED, see FINAL below

First trustworthy composed measurement in this project: one binary, flags explicit, library
hash verified per arm, and **generated text inspected at every point** (`TEXT=OK` throughout).

Configuration: `INPLACE_VIEWS=1 GATHER=1 TOPK=1 SCORE=1 SPARSE_FA=1 MASK=0 FA_MASK=0`
against stock master. Binary `2a3823f89132d1ef`.

| tokens | prefill safe (r1/r2) | prefill master | delta | decode safe | decode master | delta |
|---|---|---|---|---|---|---|
| 64 | 78.38 / 92.59 | 43.80 / 46.27 | +84% | 23.66 / 23.61 | 19.19 / 19.28 | +23% |
| 256 | 186.39 / 234.28 | 99.09 / 109.42 | +102% | 23.60 / 23.52 | 19.19 / 19.38 | +23% |
| 16384 | 463.07 / 462.03 | 292.86 / 291.44 | **+58%** | 20.78 / 20.69 | 15.50 / 15.68 | **+33%** |
| 65535 | 365.67 / 366.51 | 242.61 / 238.47 | **+51%** | 16.51 / 16.55 | 9.15 / 9.11 | **+81%** |
| 113752 | 301.27 / 302.70 | 204.17 / 203.86 | **+48%** | 13.76 / 13.76 | 6.52 | **+111%** |

Compute buffer 869.85 / 877.10 / 877.10 MiB, i.e. **-1024.00 MiB per card** versus master's
1893.85 / 1901.10 / 1901.10.

Gains grow with context, which is the expected shape: both the sparse-FA work and the memory
work bite hardest at long context, and master is the arm running with 40 MiB of headroom.

## Caveats on these specific numbers

- **PASS 1 ONLY.** Interleaved means from safe p2 and master p4 are pending; the arms were not
  yet thermally paired.
- **Model load time 39.1 s vs master 102.1 s is NOT yet credible.** The master arm ran first
  on a colder page cache. Part may be real, since master allocates a compute buffer twice the
  size, but do not quote 2.6x until the interleaved passes settle it.
- Net VRAM (memtrace) for the shipped set is still pending; the figures above are reserve.
- The safe set is correct at every probed layout, NOT correct by construction: TOPK and SCORE
  share MASK's structural exposure and survive on buffer size ratios. See the audit section.

## The CPY lead was retracted

The one extra `CPY(bf16->q2_0)` in the first op-suite diff was flake. Isolation, 5 reps per
arm: MASK=1 hit 2/5 reps, MASK=0 hit 4/5 - i.e. MORE often with the fusion OFF. So the op
suite shows ZERO difference between the safe and mask sets, consistent with it being unable to
fire scheduler-level fusions at all. The 16k completion remains the only reproducer for the
MASK defect, minimal pair `scratchpad/dg3_F4_score.txt` vs `dg3_F5_mask.txt`.

---

# FINAL end-to-end result vs master (2026-09-19)

Medians over 4 raw reps per side, interleaved, cold prefills discarded, library hash verified
at the START and END of every arm, all flags explicit, **generated text inspected at every
point**. branch `2a3823f89132d1ef`; master `/root/wt-master` rebuilt with GGML_VULKAN=ON so
the two configurations match.

Configuration: `INPLACE_VIEWS=1 GATHER=1 TOPK=1 SCORE=1 SPARSE_FA=1 MASK=0 FA_MASK=0`.

| tokens | master pp | safe pp | delta | master tg | safe tg | delta |
|---|---|---|---|---|---|---|
| 64 | 45.25 | 84.98 | +87.8% | 19.23 | 23.58 | +22.6% |
| 256 | 104.99 | 209.90 | +99.9% | 19.35 | 23.51 | +21.5% |
| 16384 | 292.15 | 462.54 | **+58.3%** | 15.66 | 20.70 | **+32.3%** |
| 65535 | 241.15 | 365.44 | **+51.5%** | 9.05 | 16.36 | **+80.7%** |
| 113752 | 204.66 | 300.90 | **+47.0%** | 6.51 | 13.74 | **+111.1%** |

**Model load 102.1 s -> 39.1 s (-61.7%).** Stable at 102.1 across all three master arms and
39.1 across both safe arms, same GGUF, adjacent runs. An earlier caveat in this document
attributing this to page-cache warming was WRONG; it is real and nothing in the brief
anticipated it.

## Memory: reserve overstates the win by about a third

| | per card |
|---|---|
| reserve | -1024.00 MiB |
| **net VRAM** | **-665 to -693 MiB** |

Buffers fall by exactly 3072 MiB total (the full -1024/card the reserve claims) but the SYCL
pool GROWS by 993 MiB, so the net peak improvement is 2079 MiB. This is invisible from
`sched_reserve` alone and is why net was required alongside reserve.

Headroom at peak, which is what actually matters: `40 / 1485 / 769` MiB on master becomes
`574 / 2217 / 1497` MiB. SYCL0 moves off a 40 MiB cliff.

## The mask fusion's "+12.4% prefill" was the cost of correctness

Measured directly at 16k: `MASK=1` gives 528.64 / 528.88 prefill against `MASK=0`'s
463.07 / 462.03, about +14%, closely matching the +12.4% recorded earlier in this document.
But `MASK=1` output is garbage and decode is unchanged. **The entire apparent prefill benefit
coincides exactly with the fusion not computing the right answer.** Treat that ledger line as
measuring the cost of correctness, not a saving.

## Composed result versus the sum of the individual A/Bs

- **Memory composes perfectly.** -128 (allocator) -512 (gather) 0 (topk) -384 (score) =
  -1024/card, and the composed measurement is -1024.00/card exactly.
- **Throughput does NOT, and is far better than recorded.** The ledger's headline was +12.4%
  prefill; the composed measurement is +58.3% at 16k WITHOUT the change that was credited with
  it. Causes: the softmax rewrite landed between those entries, and no individual A/B ever
  measured this combination. Fifteen pairwise measurements on fifteen tree states do not
  compose; only an end-to-end run against master answers the question.

## A real hole in the measurement guard, found and fixed

Master p4 passed `builds start=0 end=0` yet its 15-minute loadavg was 4.63 against a 1-minute
of 3.28, and its numbers carry the CPU-starvation signature (65536 prefill 196.54 against
238-245 in every other master arm). **A build that starts AND ends entirely inside an arm is
invisible to start/end sampling.** That arm was excluded from the medians. `gpu_run.sh` now
samples compilers and loadavg every 20 s for the duration of the arm and reports the PEAK, not
the boundaries.

---

# Pool growth fully attributed (2026-09-20): it is `packed_b`, and 226 MiB/card is stranded

The net win was a third smaller than the reserve because `pool_leg` grew ~331 MiB/card. That
growth is now attributed per allocation, per card, per call site, with no source changes, via
an LD_PRELOAD interposer on `zeMemAllocDevice`/`zeMemFree` with backtraces
(`scratchpad/memprobe.cpp`, resolved by `scratchpad/resolve_bt.py`). It reproduced the original
totals exactly: safe 1550 vs 1549 MiB, master 556.1 vs 556.

| call site | master | safe | delta/card |
|---|---|---|---|
| `ggml_sycl_grouped_dequant_gemm_f16` (`packed_b`) | 0 | 973.8 | **+324.6** |
| `ggml_sycl_flash_attn_ext_mkl` | 0 | 20.4 | +6.8 |
| `launch_fattn<flash_attn_tile>` | 0 | 0.6 | +0.2 |
| `ggml_sycl_mul_mat_id` | 315.9 | 315.0 | -0.3 |
| `ggml_sycl_op_mul_mat_sycl` | 240.2 | 240.2 | 0 |
| total | 556.1 | 1550.2 | **+331.3** |

Deltas sum to +331.6 against a measured +331.3. Nothing unattributed.

## Mechanism: a routing-dependent size against a grow-only cache

`fused-gemm.cpp:344-395`: `n_tiles = sum_e ceil(rows_e / FG_BN)` uses the per-expert row count
of THIS ubatch, so the allocation tracks MoE ROUTING, not tensor shape. As a prefill proceeds
routing covers more experts and the request jitters upward about 1.5x.

`ggml_sycl_pool_leg::alloc` (ggml-sycl.cpp:1835-1892) reuses a cached block only when
`b.size >= size`, and `pool_leg` frees nothing until destruction. So every request that sets a
new maximum allocates a fresh `1.05 * size` block and strands its predecessor permanently.
Observed ladder, SYCL0: 69.56 / 73.50 / 82.36 / 88.10 / 94.01 MiB, i.e. **407.5 MiB held for
94.0 MiB of working set**.

Note this is gated by `GGML_SYCL_GROUPED_GEMM`, which DEFAULTS TO 1 and so does not appear in
the flag set anyone bisects over. It is the "grouped MoE GEMM, +33% prefill" work.

Also note the varying dimension is NOT `n_kv`: no pool site scales with it, and 1448 of the
final 1550 MiB is already in place after the 16k prefill. An earlier hypothesis in this
document guessing `n_kv` was right in kind and wrong in the dimension.

## Fix: ~226.5 MiB/card, one line

Only one `packed_b` is live at a time. Round `K * Npad` up to a coarse granularity before the
`pool_alloc`, while still passing the true `Npad` to the kernel as the pack stride. Collapses
5/3/3 blocks to 1 per card. Generic alternative: bucket `look_ahead_size` in `pool_leg::alloc`
rather than `1.05 * size`, which also collects the 1.9 MiB/card FA ladder but changes pool
behaviour for every SYCL user.

Projected: pool peak 1550 -> ~865 MiB, global 68457 -> ~67772, net win over master
693 -> ~921 MiB/card, pool's share of the win 32% -> 11%.

NOT reclaimable: ~98 MiB/card is one live `packed_b`, the honest working set the +33% prefill
buys; and 27.6 MiB/card at `op_mul_mat_sycl` is three concurrently live buffers, identical in
master.

## CORRECTIONS to attributions recorded earlier in this document

- `GGML_SYCL_FUSE_QSA_SCORE` does NOT cost 32 MiB/card of pool. Zero backtraces; its 32 MiB
  tile always found a cached block. The earlier figure was an indirect effect of perturbing
  another site's record-breaking sequence.
- `GGML_SYCL_SPARSE_FA` costs ZERO pool. Zero backtraces; decode-only, and with 64 decode
  tokens per request its allocations never set a pool record.
- The FA staging clamp's ~7 MiB/card IS confirmed: MKL FA totals 6.8 MiB/card.

## Robustness gap

`ggml-cuda`'s `ggml_cuda_pool_leg` has `clear_pool()` and a retry-on-`cudaErrorMemoryAllocation`
path. The SYCL pool has neither, so stranded blocks are a hard peak rather than recoverable
slack under pressure. Porting that pattern is small and precedented.
