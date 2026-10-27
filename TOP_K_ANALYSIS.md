# SYCL: large-k `TOP_K` — analysis and two implementations

> **SUPERSEDED 2026-09-24 — do not reason from this document.**
>
> Its premise no longer holds in this tree. It assumes `supports_op` gates `TOP_K` on
> `k > 0 && k <= 32`, so the model's `k = 2051` indexer top-k falls to the CPU and produces
> "~24 of the model's 28 graph splits at bs=1". Current `ggml-sycl.cpp:8418-8426` reads
> `k > 0 && k <= src0->ne[0]`, and `ggml_sycl_op_top_k` routes `k > SYCL_TOP_K_SCAN_MERGE_MAX_K`
> to `ggml_sycl_top_k_radix` (`topk-radix.cpp:400`). Confirmed empirically: a VTune GPU capture of
> a decode run shows `top_k_radix_f32_sycl<top_k_rows_qsa<half>>` with 8184 instances, exactly
> 12 per token, running on the GPU. Those CPU round-trips are gone.
>
> Kept for the kernel design discussion and the measured scan-merge vs radix comparison, which
> remain valid. The split/round-trip motivation does not.


Branches (both based on `iq_type_fusion`, HEAD `b1a7c2279`):

| branch | approach | diff |
|---|---|---|
| `topk_large_k` | global-memory bitonic argsort + slice | +53 / −5, 1 file |
| **`topk_radix_select`** | **radix select, feature-gated at k > 32** | **+192 / −3, 1 file** |

`topk_radix_select` is the recommended one. See §5 for the head-to-head.

---

## 1. The problem

`ggml/src/ggml-sycl/ggml-sycl.cpp`, `supports_op`:

```c
case GGML_OP_TOP_K: { const int k = op->ne[0]; return ... && k > 0 && k <= 32; }
```

`k > 32` is therefore **refused by the SYCL backend and computed on the CPU**, with the score
tensor shipped host-ward and indices shipped back — a backend round-trip per call.

For `qwen4exp` / Qwen3.8-Flash-Next this fires constantly. `src/models/qwen4exp.cpp:665`:

```c
const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);
ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));
```

with `qwen4exp.attention.indexer.top_k = 2048` in the GGUF. The sparse-attention indexer runs in
**12 attention layers, every token**. Note `width = min(n_kv, ~2048)`, so even at a 64-token
context `k = 64 > 32` — **this fires at every context length, not only long ones.**

Measured effect: ~24 of the model's 28 graph splits at bs=1 are these round-trips.

## 2. Why the limit exists — Shared Local Memory

Not conservatism. `top_k_f32_sycl()` keeps `(split_block + 1) * k` candidate (value,index) pairs
in SLM via two `sycl::local_accessor`s, i.e. `(split_block+1) * k * 8` bytes per work-group, with
`split_block = 128`. The B60 has **131072 B of SLM per Xe-core**:

| k | SLM required | outcome |
|---|---|---|
| 32 | 33.0 KB | fits (~1 resident group) |
| 64 | 66.0 KB | fits |
| **128** | **132.1 KB** | **exceeds the hardware limit — cannot launch** |
| 2048 | 2.11 MB | 16x over |

The existing code comment says as much: *"Revisit if the supported k ever grows."* The footprint
is **O(k)**, so the op hits a wall long before the k this model needs.

## 3. Approach 2 — global-memory bitonic argsort (`topk_large_k`)

Mirrors the pattern `argsort_f32_i32_sycl()` already uses ten lines away: compute the SLM
requirement, compare against `ggml_sycl_info().devices[device].smpbo`, and fall back to a
multi-pass bitonic sort in **global** memory when it does not fit. Then slice the first k of the
descending result.

- SLM footprint becomes k-independent. Correct, and reuses tested code.
- **But**: it is a full sort to select 2048 of 131072, costing **153 kernel launches** per op at
  ncols=131072, each paying submission overhead (`GGML_SYCL_ENABLE_GRAPH=0` on multi-GPU).
- Scratch: `nrows*ncols*4` B, plus `nrows*next_pow2(ncols)*4` B when the row misses SLM —
  ~12 MB transient at ncols=131072, nrows=12.

## 4. Approach 3 — radix select (`topk_radix_select`) — RECOMMENDED

Select the k-th largest by **digit passes over an order-preserving unsigned key**, 8 bits per pass
(256 buckets, 4 passes), then emit. SLM holds **histogram bins, not candidates**, so the footprint
is **independent of k** — which is the entire point.

- Key transform: `u = bit_cast<uint32>(f); if (u == 0x80000000) u = 0; key = (u & sign) ? ~u : (u | sign)`.
  The `-0.0 -> +0.0` fold makes the key's equivalence classes match the CPU comparator, under
  which `-0.0` and `+0.0` tie.
- Ties: the select carries `need` (how many still required) through the passes; after the last
  digit the pivot is the exact 32-bit key and `need` is how many pivot-equal elements to take.
  If a pivot bucket's count equals `need`, it exits early — later digits cannot change the answer.
- Exact-k emission: two SLM atomic counters write the `> pivot` set and then top up from the
  `== pivot` set until exactly k indices exist. Disjoint regions, no duplicates.
- Output order: **not required**, so not paid for. `ggml-cpu/ops.cpp` deliberately swaps
  `dst[0]`/`dst[1]` ("emphasize that the order is not important") and `test-backend-ops` compares
  index multisets. Sorting the result would be wasted work.

**Feature gate, as specified:** a single named constant

```c
static constexpr int SYCL_TOP_K_SCAN_MERGE_MAX_K = 32;
```

`k <= 32` takes the existing scan-merge path **bit-for-bit unchanged** (zero deletions in those
kernels or their launch shapes); `k > 32` takes the radix path. `supports_op` and the assert are
relaxed to `k <= ncols`. Strictly additive.

### Structural comparison

| | approach 2 (bitonic) | **approach 3 (radix)** |
|---|---|---|
| kernel launches @ ncols=131072 | 153 | **1** |
| SLM per group | k-independent | **8 212 B**, k- and ncols-independent |
| pool scratch | ~12 MB transient | **none** |
| complexity | O(n log^2 n) full sort | O(n) x ~5 sweeps |

## 5. Performance

3x Intel Arc Pro B60, `Qwen3.8-Flash-Next UD-IQ4_XS`, `llama-bench` `tg64`,
`-ngl 99 -dev SYCL0/SYCL1/SYCL2 -sm layer -ts 31/33/36 -ncmoe 0 -ot per_layer_token_embd\.weight=CPU`
`-fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 -t 4 --poll 0`, `-r 2`.
Baseline column `iq_type_fusion` = the MoE-fusion branch these are built on.
`master` = `49c0dc82b`. Only `libggml-sycl.so` differs between arms (md5-verified distinct).

> **Read the `radix vs master` column with care -- it does not describe the branch that was
> eventually rebased for upstream.** Two things about it are stale:
>
> 1. **Most of it is not top-k.** The column chains two steps, `master -> iq_type_fusion ->
>    +radix`, and the large one is the middle step: the MoE fusion alone takes d=0 from 14.54
>    to 18.77, +29%. The radix step on top of it is the `radix vs iq` column, +5.5% to +15.3%.
> 2. **`master` here means `49c0dc82b`**, which upstream has since moved 73 commits past.
>
> `topk_radix_select_master`, the rebased branch, is `master + top-k only` -- the IQ-fusion
> commit was dropped as unrelated (it touches only `mmvq.cpp` and `vecdotq.hpp`). It therefore
> does **not** inherit the +36-40% shown here. Its own gain over master is roughly the
> `radix vs iq` column plus the +2.4% of §8.3, and it has **not been measured against master
> directly** -- neither `49c0dc82b` nor current upstream. Every number in §8 is a single
> delta, `ae05c9236` versus `ae05c9236 + the §8.2 changes`; both arms report
> `build: ae05c9236`, and no arm in this document is a master build.

| depth | master | iq_type_fusion | +bitonic | **+radix** | radix vs iq | radix vs master |
|---|---|---|---|---|---|---|
| 0 | 14.54 | 18.77 | 19.86 (+5.8 %) | **19.80** | +5.5 % | **+36.2 %** |
| 2 048 | 13.95 | 17.96 | 18.90 (+5.2 %) | **19.31** | +7.5 % | **+38.4 %** |
| 8 192 | 12.48 | 15.73 | 15.79 (+0.4 %) | **17.52** | **+11.4 %** | **+40.4 %** |
| 32 768 | 9.78 | 11.35 | 12.59 (+10.9 %) | **13.09** | **+15.3 %** | **+33.8 %** |
| 131 072 | — | 5.29 | not measured | **5.86** | **+10.8 %** | — |

At the deepest point, `d = 131072`, radix delivers **5.86 vs 5.29 t/s, +10.8 %**. Note the gain
peaks at d=32768 (+15.3 %) and eases slightly at 131072 — at that depth the model's own attention
cost grows faster than the top-k work being displaced, so the top-k is a smaller share of the step.

**Radix beats bitonic everywhere except d=0, where they tie within noise.** The decisive point is
**d=8192: bitonic +0.4 %, radix +11.4 %.** At ~8k the row still fits SLM, so bitonic takes the
single-kernel SLM argsort, which lands about even with the CPU path it replaces; radix has no such
soft spot. Radix's gain also rises with depth over most of the range (+5.5 % at d=0 to +15.3 % at
d=32768), which is what the mechanism predicts — the displaced CPU work grows with `n_kv`. It is
not monotonic to the end, easing to +10.8 % at d=131072 for the reason given above.

## 6. Correctness

`test-backend-ops -o TOP_K -b SYCL0` on the radix build, run independently of the implementing
agent: **7/7 backends passed, 525/525 cases, zero "not supported" lines** (previously every k > 32
case was refused). Tie-heavy large-k cases pass at every size tested, including
`ne=[131083,1,2,1], k=9999, ties=1`.

`test-backend-ops -o MUL_MAT_ID -b SYCL0`: **880/880**, confirming the MoE work on the base branch
is undisturbed.

### Perplexity (wikitext-2, `-c 512 --chunks 200`)

| | iq_type_fusion | +radix | Change |
|---|---|---|---|
| PPL | **4.2944 ± 0.04366** | **4.2942 ± 0.04368** | **−0.0002 (−0.005 %)** |

The two are identical to within 0.0002 — **200x smaller than the reported standard error**, and
~35x smaller than the run-to-run spread of an unchanged binary measured on this machine earlier
(master gave 4.2920 and 4.2993 on two runs, a spread of 0.0073). This matters more here than for a
pure arithmetic change: radix select can pick a *different* tied element than the CPU reference,
so the concern was set membership, not rounding. It is unaffected in practice.

Neither `llama-bench` nor `llama-perplexity` samples, so there is no temperature to set — both are
deterministic by construction.

## 7. Known limitations / follow-ups

- **Parallelism, not correctness, is the open question.** One work-group per row means 12 rows
  occupy ~12 of 20 Xe-cores, each doing ~5 strided sweeps of the row. A two-stage histogram
  (per-partition histograms reduced across groups) would raise occupancy at the cost of ~5 launches
  and `nrows*nsplit*256*4` B of scratch. Not attempted; measure before adding it.
- **Candidate compaction** between digit passes would cut passes 2-4 to near-zero reads, at the
  cost of scratch and a compaction step. Only worth it if profiling shows the sweeps dominate.
- The gate at 32 means k in 33..127 now takes the radix path even though the old scan-merge could
  in principle have handled it. That range was never the target; the constant makes it trivial to
  retune if it ever matters.
- NaN ordering is deterministic here (+NaN above +inf, -NaN below -inf) but the CPU reference is
  undefined on NaN (its comparator is not a strict weak order), so there is nothing to match.


## 8. Follow-up: the two open questions, measured

Both notes below were tested on the 3x B60 host with `test-backend-ops perf -o TOP_K -b SYCL0`
and `llama-bench` on the same model and flags as §5. Verdict first: **the `nrows == 1` note is
right and understated; the sub-group aggregation note is not worth doing.**

### 8.1 Occupancy: the note is right, and §7 understated it

§7 says "12 rows occupy ~12 of 20 Xe-cores". That is wrong for the case §5 benchmarks.
`qwen4exp.cpp:667` tops-k a tensor of shape `[n_kv, n_tps, n_stream]` with
`n_tps = n_tokens/n_stream` (`qwen4exp.cpp:546`), so at tg with one sequence **`nrows == 1`**.
The 12 layers are 12 separate ops of one row each, not one op of 12 rows. Every top-k in §5
ran on **one work-group, i.e. one of 20 Xe-cores**.

Measured directly, without a profiler: at `k=400`, `ne=[200000,1]` costs 358.01 us and
`ne=[200000,16]` costs 363.42 us. **Sixteen rows for 1.5% more wall-clock** -- the device is
~94% idle at one row. (The old scan-merge path scales normally over the same range, 343 -> 2193
us, because it already partitions a row across groups.)

A second call site has the same shape: `llama-sampler.cpp:1486` reshapes logits to 1-D, so
backend sampling is `nrows=1`, `ncols=n_vocab`, and `k` is whatever the user set.

### 8.2 What was implemented

Three changes, all in `ggml-sycl.cpp`, all measured:

1. **`block_size` now takes the device's `max_work_group_size`** rather than a cap. The old
   `std::min(512, max_block)` clamped a device that allows 1024 down to half its width; the
   clamp was never a floor, so a device reporting 512 was and still is given 512. On this
   device it means 512 -> 1024, worth 20-25% on the op on its own.
2. **A multi-group radix path** for rows too few to cover the device. The per-pass state moves
   to global memory, each digit pass becomes its own launch, and the last group of a row (the
   one whose `fetch_add` returns `G-1`) does that pass's scan, so the launch count is one per
   digit plus one emit rather than two per digit. `G` comes from the device
   (`GROUPS_PER_NSM * nsm`), floor-divided by `nrows` so a row count that already covers the
   device is left whole.
3. **The scan-merge gate lowered from `k <= 32` to `k <= 8`**, which is where the two paths
   actually cross. This one was not in the notes and is the largest single win in the file.

Two traps worth recording, because both cost more than the algorithm did:

- **Device-scope atomic loads are uncached.** Reading `prefix`/`mask`/`need` through
  `atomic_ref<..., memory_scope::device>` from every work-item made passes 2-4 cost 49 us each
  against 12 us for pass 1. Having one lane read them into SLM and the group take them from
  there: **208 -> 44.6 us**. That single change is most of the split's benefit.
- **A `.wait()` on the scratch `memset`** serialises the host against the queue on every call.
  The queue is in-order; the fill needs no wait.

### 8.3 Results

Provenance, so the numbers can be trusted for what they are. Every arm below is
`ae05c9236` with and without the §8.2 changes -- **nothing here is measured against master**;
see the note in §5 for why the `radix vs master` column there does not carry over to the
rebased branch. The perplexity runs in §8.3b were measured on the final code. The two tables below were measured one edit earlier, before
`std::min(1024, max_block)` became plain `max_work_group_size`. On this device that
substitution is an identity -- `max_work_group_size` is 1024, so the clamp never bound -- and
the correctness suite was re-run after it (525/525). On a device that reports a different
limit it is not an identity, and these figures would have to be re-taken there.

Op-level, `test-backend-ops perf`, before = this branch as committed:

| shape | k | before | after | |
|---|---|---|---|---|
| `ne=[131072,1]` (qwen4exp tg) | 2048 | 208.36 us | **41.82 us** | **4.98x** |
| `ne=[200000,1]` | 400 | 358.61 us | **46.10 us** | **7.78x** |
| `ne=[151936,1]` (sampler) | 40 | 281.68 us | **42.33 us** | **6.65x** |
| `ne=[151936,1]` (sampler) | 20 | 565.14 us | **42.33 us** | **13.35x** |
| `ne=[65000,16]` | 32 | 11016.34 us | **92.79 us** | **118.7x** |
| `ne=[1024,8192]` | 32 | 49716.84 us | **1158.91 us** | **42.9x** |

No shape measured got slower. The last two rows are the gate change, not the split.

End to end, `llama-bench tg64`, same flags as §5, `-r 2`, both arms measured back to back in
one session with the identical invocation:

| depth | HEAD | + these changes | |
|---|---|---|---|
| 0 | 19.58 +/- 0.11 | 19.61 +/- 0.08 | +0.2% |
| 8 192 | 17.79 +/- 0.05 | 17.85 +/- 0.06 | +0.3% |
| 32 768 | 13.25 +/- 0.03 | 13.25 +/- 0.06 | 0.0% |
| 131 072 | 5.91 +/- 0.02 | **6.05 +/- 0.01** | **+2.4%** |

The gain is concentrated at the deepest point because that is the only depth where `ncols`
reaches the width at which splitting pays. Below ~64K columns the single-group kernel finishes
inside the cost of the four extra launches, and the policy leaves it alone.

Note the honest shape of this: **the op got 5x faster and the model got 2.4% faster.** Top-k was
~1.5% of the step at d=131072 before this work. The large §5 gains came from removing the CPU
round-trip, not from the kernel's own cost, and that had already been collected.

### 8.3b Perplexity

wikitext-2, same device flags. Both arms measured back to back in one session. The `-c 512`
setting §6 used never reaches the split path (`ncols = n_kv <= 512`, far under the 64K
threshold), so a long-context arm was added; `-c 131072` runs out of device memory under
`llama-perplexity`, which needs logits for every position, so 81920 is the widest that fits.

| | HEAD | + these changes | delta |
|---|---|---|---|
| `-c 512 --chunks 200` | 4.2954 +/- 0.04367 | 4.2980 +/- 0.04373 | +0.0026 |
| `-c 81920 --chunks 3` | 3.8119 +/- 0.02284 | 3.8249 +/- 0.02292 | +0.0130 |

Both differences sit inside the stated standard error, and the c512 one inside the
run-to-run spread of an unchanged binary recorded in §6 (0.0073). A non-zero difference is
expected rather than alarming: the split changes which of several tied columns wins a slot,
since selection now depends on partition boundaries and group arrival order. What it is not
is evidence of correctness, at 3 chunks: the resolution there is only about +/-0.02, and
`llama-perplexity` is not bit-reproducible across runs on this multi-GPU setup anyway.

The real guarantee is `test-backend-ops`, which compares index multisets against the CPU
reference exactly, including tie-heavy cases at widths that trigger the split
(`ne=[131083,1,2,1] k=9999 ties=1`, `ne=[33024,4,1,1] k=2051 ties=1`): 525/525.

### 8.4 Where the gate crossover actually is

Measured across `ncols` from 2 to 200K and `nrows` from 1 to 8192, against the scan-merge path:

- **k <= 2**: scan-merge wins everywhere (radix 0.19-0.56x). Its fixed cost is simply smaller.
- **k = 4**: mixed, 0.42x to 1.60x.
- **k >= 8**: radix wins everywhere, from 1.3x at the crossing to 119x at `k=32` on a wide row.

Hence `SYCL_TOP_K_SCAN_MERGE_MAX_K = 8`. The risky corner -- short rows, very many of them, as
in bailingmoe2 group selection (`ncols=2, nrows=65536`) -- was measured explicitly before
moving the gate; it is exactly where radix loses at low k, and the gate keeps it on scan-merge.

`test-backend-ops` had no perf coverage for `k > 32` at large widths, nor for the short-row
corner; both were added.

### 8.5 The sub-group aggregation idea: not worth doing

The patch sketched below in the original notes has a correctness bug and targets the wrong loop.

- **Divergent loop.** Both patched loops keep `col < ncols` while adding a `valid`/`in_bounds`
  flag that is therefore always true. The sub-group collectives now sit in a loop whose trip
  count diverges across lanes: at `ncols=131083, block=512`, lanes 0-10 run 257 iterations and
  the rest run 256, so the last iteration calls a group collective with 11 of 32 lanes. UB.
- **`sycl::select_from_sub_group` is not a name** -- DPC++ spells it `sycl::select_from_group`.
  `1u << sg_lane` also silently caps the sub-group at 32 lanes with no `reqd_sub_group_size`.
- **The emit loop is the wrong target.** At `ncols=131072, k=2048` under 2% of iterations touch
  an atomic; the patch adds two unconditional scans plus broadcasts to *every* iteration.
  Replacing the emit's atomics with a two-level SLM reservation was tried here and changed
  nothing measurable, which confirms they were never the cost.
- **The histogram loop is the right target but the wrong shape.** Only pass 1 has every lane
  hitting an atomic, and the `while (active_mask)` loop costs two collectives per *distinct
  bucket* in the sub-group -- 20-32 of them on any spread data, against 32 SLM atomics already
  spread over 8 interleaved copies.

If atomics ever do show up in a profile, raising `SYCL_TOP_K_RADIX_HIST_COPIES` is one constant
and carries none of the convergence hazards.

### 8.6 Still open

- The passes re-read the row rather than compacting candidates between them. Compaction would
  take the full-row reads from 5 to 2. Worth trying only if top-k grows back into a visible
  share of the step; at 1.5% before this work it was not the place to spend effort.
- `nsm` is `max_compute_units / 16`, which counts an Xe core as 16 EUs. That is a core's width
  on Xe-HPG but twice it on Xe2, so the field reads 10 on a 20-core B60. `GROUPS_PER_NSM = 2`
  compensates, and lands in the measured-flat region either way, but it is the one constant
  here that a correct core count would remove. Everything else in the dispatch is either a
  device value or a width threshold established by measurement.
- All tuning was done on Xe2 (Battlemage). Nothing here was measured on Xe-HPG (Alchemist);
  the block size and split width now come from the device, but `GROUPS_PER_NSM` and the 64K
  split threshold are unvalidated there.
