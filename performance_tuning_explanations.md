# SYCL performance tuning: what changed, what it bought, what did not work

Session of 2026-09-18, branch `performance_uplift_mega_branch`, commit `31cd5076b`
("Checkpoint +50% prefill, +8% decode").

Hardware: 3x Intel Arc Pro B60 (die **bmg-g21**, Xe2; 160 XVEs = 20 Xe-cores x 8 XVEs,
8 threads per XVE, sub-group size 16), oneAPI 2026.1.
Model: unsloth Qwen3.8-Flash-Next UD-IQ4_XS (qwen4exp), 177B total / 6B active, 512
experts / 10 used, 48 layers, n_embd 2560.
Config: `-c 131072 -b 4096 -ub 1024 --flash-attn on -ctk q8_0 -ctv q8_0 --no-kv-unified
-np 1 --split-mode layer --tensor-split 31,33,36`.

**Totals: prefill 319 -> 478 t/s (+50%), decode 19.6 -> 22.5 t/s (+15%).**

## How everything here was measured

Every number below came from an interleaved single-binary A/B (arms ordered 1,0,1,0 so
drift shows up rather than hiding), three reps per arm, discarding rep 1 as cold, with
**an assertion on the generated text in every sample**. That last point is not
ceremony: a wrong kernel on this branch once reported 764 t/s while emitting
"A A A A A". Throughput alone is not evidence of anything.

Each change is behind an env flag so both arms live in one binary. Never swap `.so`
files between arms - `cp x.so build/bin/... && cmake --build` is a no-op that still
prints "Built target", which silently benchmarked the same build twice earlier in this
project.

## What shipped

### 1. Grouped MoE GEMM, IQ4_NL + IQ3_S - **+33% prefill**

`GGML_SYCL_GROUPED_GEMM=1` (default). `fused-gemm.cpp`, dispatched from
`ggml_sycl_mul_mat_id`.

ggml-sycl's batched MUL_MAT_ID path (`ne12 > 8`) issued **one GEMM per expert**. With
512 experts and 10 used, a 1024-token ubatch gives ~20 routed rows per expert - a
pathological regime: 512 tiny dequantize+GEMM dispatches per op. Replaced with a single
grouped launch using a host-built `{expert, n0, n1}` tile schedule, B packed per tile,
and the dequantize fused into an XMX f16 GEMM.

Per-op, against the per-expert loop:

| shape | grouped | per-expert loop |
|---|---|---|
| iq3_s M=640 K=2560, 512 exp, 1024 tok | 8.9 ms | 24.3 ms |
| iq3_s same, 300 tok | 6.8 ms | 26.2 ms |
| iq4_nl M=2560 K=640, 300 tok | 7.7 ms | 35.0 ms |

End to end 361.1 -> 480.8 t/s. IQ4_NL alone was +8.7%; adding IQ3_S (ffn_gate + ffn_up,
two thirds of the MoE surface) took it to +33.1%. Verified by mechanism as well as
throughput: `dequantize_row_iq3_s_sycl` and its 55,504 launches vanish from the profile
between captures.

**Do not retune `FG_BN`.** Swept: 32 is optimal (64 costs 5.1%, 16 costs 3.2%, and
**128 corrupts output while reporting a spurious 764 t/s**). Reducing the A-decode
*count* does not help, which also refutes the "one work-group per expert reuses A"
rewrite - registers cap that at 2 tiles, which is arithmetically identical to FG_BN=64,
which loses.

### 2. q8_0 ESIMD mat-vec - **+8.5% decode**

`GGML_SYCL_ESIMD_Q8_0=1` (default). `dmmv.cpp`, trait-style, alongside the existing
K-quant ESIMD kernels in `esimd.hpp` (which is untouched).

Motivation was a measured existence proof, not a guess. VTune per-kernel achieved read
bandwidth, against a ~456 GB/s peak:

| kernel | BW read |
|---|---|
| every generic mmvq kernel | 77-86 GB/s (~18% of peak) |
| `dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd` | 301 GB/s (~66%) |

**Why ESIMD is faster - isolated by experiment, not argument.** Three q8_0 kernels read
the *same* reordered bytes: MMVQ (237 us), scalar DMMV with **no q8_1 quantize at all**
(294 us), and ESIMD (155 us). The middle one being *slower* rules out "the generic path
pays for activation quantization / dp4a". The whole difference is how loads are issued:
the generic path does **5 divergent messages per 256 bytes** (a SIMD16 gather of 4-byte
elements at 16-byte stride, touching 256 bytes to deliver 64, plus a 2-byte scale
gather), ESIMD does **2 contiguous `block_load`s**. The new kernel reaches 403 GB/s,
88% of peak.

Results are f32-exact (nmse ~1e-14), which is *better* than the MMVQ path it replaces,
since that quantizes activations to q8_1 first.

**A silent-corruption bug was found in review and fixed.** ESIMD `block_load` assumes
4-byte alignment; the scale base is `d + row*nblk_row + b`, so an **odd** `nblk_row`
leaves odd rows only 2-byte aligned and the hardware returns the halves DW-swapped.
Triggered on any q8_0 weight with `ne00 % 64 == 32` and `ne00 >= 288`. Fixed with
`element_aligned_tag{}` on both half loads, measured zero cost. **`test-backend-ops`
passed 1126/1126 in both arms while this was live**, because it only generates q8_0
`n=1` cases at k in {256, 1024, 4096, 5120} - all even `nblk`.

### 3. Wide-load MMVQ - **+5.3% decode standalone**, kept as the non-ESIMD fallback

`GGML_SYCL_MMVQ_WIDE=1` (default). `reorder_vec_dot_q8_0_wide` in `vecdotq.hpp`,
selected at the launch site in `mmvq.cpp`.

Same insight as ESIMD, in plain SYCL. `get_int_from_int8` assembles every dword from
**two `uint16_t` loads** ("assume at least 2 byte alignment"), so a lane issues 8 narrow
loads to fetch 16 contiguous bytes. For q8_0's reorder layout the block offset is
`ibx*32` and `iqs` is 0 or 4, so each lane's base is **16-byte aligned** and one
`sycl::int4` load is legal. Output is **bit-identical** to the narrow path across all 28
test shapes, including every odd-`nblk` case that broke the ESIMD kernel.

Kernel: 31.3 -> 28.9 us (10240x320), 31.4 -> 28.5 us (320x10240), flat on 640x2560
(submission-floor limited). So wide loads recover ~62% of the ESIMD end-to-end benefit
with no new kernel language and no alignment hazard - much easier to defend upstream.

ESIMD claims q8_0 first, so this only runs when ESIMD is unavailable or disabled.
Layering: ESIMD 22.51 / wide 21.76 / original 20.67 t/s, all switchable in one binary.

### 4. `max_wg_per_cu` occupancy fix - **+1.6% decode**

`ggml-sycl.cpp`, `GGML_SYCL_MAX_WG_PER_CU=16` (default).

The old value was `max_work_group_size / max_compute_units` = 1024/160 = **6**, which
divides a work-group SIZE by a compute-unit COUNT - not a meaningful quantity. It seeds
`parallel_blocks` and bounds `blocks_per_wave = nsm * max_blocks_per_sm` in
`fattn-common.hpp:1052-1078`, so it under-sized every flash-attention grid. 60
work-groups against a real capacity of ~320 predicts 18.75% occupancy; VTune measured
**20.0%**.

Swept 6/12/16/24/32: everything from 12 up is equivalent, 16 is both the derivation
(8 XVEs/Xe-core x 8 threads / 4 threads per work-group) and the best measured. Prefill
is flat across all arms **as predicted before running**, because prefill routes to MKL
FA and never reaches the tile kernel - that flatness is the check that the mechanism
story is right.

**`nsm` is also wrong and was deliberately NOT changed.** `nsm = max_compute_units / 16`
uses the Xe-HPG ratio; Xe2 has 8 XVEs per Xe-core, so nsm reads 10 where the true count
is 20. Correcting it in isolation would regress three paths whose tuning absorbed the
error: `getrows.cpp:252` (GET_ROWS_WORK_GROUPS_PER_CU was swept to 4 *against nsm=10*;
doubling nsm doubles `min_items`, equivalent to 8, which that sweep rejected),
`topk-radix.cpp:232`, `count-equal.cpp:48`. Fixing nsm means re-tuning those together.
`max_wg_per_cu` was safe to change alone because it has exactly one caller.

## What is in the commit but bought nothing

Committed on request as a checkpoint; strip before the PR split.

- **Multi mat-vec fusion** (`mmvq.cpp/.hpp`, `topk-moe.cpp`, parts of `ggml-sycl.cpp`).
  Correct, fires 3120x per run (48 layers x 65 evaluations), removes 96 submissions per
  token, **zero measurable gain**. Now doubly redundant: enabling ESIMD makes it decline
  q8_0 anyway. It also measured that the two cases we most wanted - `attn_qkv`+`attn_gate`
  and `attn_q`/`k`/`v` - **cannot** fuse: they are 20 and 1-4 launches apart, and the
  `v`->`k` gap is not hoistable because k's destination overlaps a tensor the Hadamard
  rotation still reads.
- **Cast+add fusion** (`binbcast.cpp/.hpp`). Structurally **unreachable**:
  `ggml_can_fuse_subgraph` rejects any node with `view_src` (`ggml-impl.h:663-665`) and
  `ggml_reshape_3d` is a view, so the real `CPY -> RESHAPE -> ADD` chain can never match.
  Fires zero times. The `(F32,F16,F32)` branch added to `ggml_sycl_op_bin_bcast` is its
  only consumer and is therefore dead too.
- **Env-gated debug traces**: `[GG]` (`fused-gemm.cpp`, `dmmv.cpp`), `[CASTDIAG]` /
  `[CASTADD]` (`binbcast.cpp`), `[MV]` (`ggml-sycl.cpp`). Silent unless their env var is
  set. They earned their place - they are what proved the grouped GEMM ran on real model
  shapes and what caught the cast+add fusion never firing - but they should not go
  upstream.

## Measured negatives - do not re-attempt without new evidence

- **SYCL graph capture never engages.** Three blockers were cleared in sequence (a stale
  CONCAT guard; a library-GEMM predicate our own kernels made stale; a MUL_MAT_ID guard
  asking the wrong reorder list). After all three, decode splits had nothing rejecting
  them and still produced zero `[SYCL-GRAPH]` messages: warmup needs two consecutive
  calls with byte-identical node properties, and the scheduler hands back different
  `src->data` each token. Forcing more splits eligible cost **24% decode** in
  `ggml_sycl_graph_update_required()` bookkeeping.
- **Submission-count reduction does not help decode.** Two independent results: the MV
  fusion removed 96 submissions/token for zero gain, and graph capture (whose entire
  purpose is collapsing submission overhead) does nothing. Decode is not
  submission-bound, despite GPU-busy being 14.5%.
- **MoE sort cache**: dead at decode by construction. `ggml_sycl_mul_mat_id_mmvq_fused`
  returns before the `ids` download whenever `ne12 <= 8`, so decode never reaches the
  code it optimizes. Measured `decode_fallthrough=0`. Worth ~0.4% at prefill and it
  carried an async-memcpy race.
- **Route-order floor 64 -> 8** cost 23% decode (19.58 -> 15.09) despite
  `test-backend-ops perf` endorsing it. Op microbenchmarks cannot see submission cost.

## The MoE types are blocked, and why

`mul_mat_vec_q_moe<iq3_s>` (8.79 s, 84.9 GB/s) and `<iq4_nl>` (8.47 s, 79.2 GB/s) are
the largest remaining decode kernels - 42% of decode kernel time - and neither the ESIMD
nor the wide-load treatment reaches them.

- **iq3_s**: `vec_dot_iq3_s_q8_1` never calls `get_int_from_int8`. It reads
  `block_iq3_s` struct fields and then `grid1[0]` - a data-dependent gather into the
  512-entry (2 KB) `iq3s_grid` codebook. The load-width mechanism does not apply at all.
  Its dominant stall is **SBID 88.8%** (waiting on outstanding memory returns), i.e.
  gather latency.
- **iq4_nl**: it *does* have the narrow pattern
  (`aux = q4[2*l] | (q4[2*l+1] << 16)`, two adjacent `uint16_t` loads), but
  `sizeof(block_iq4_nl)` is **18 bytes**, so `qs` sits at offset 2 in an 18-byte stride
  and every block is only **2-byte aligned**. The narrow loads are forced by the AoS
  layout, not a missed optimization. A wider load there would be exactly the
  misalignment bug found in the ESIMD kernel.
- **ESIMD is wired only into the dense path** (`ggml_sycl_op_dequantize_mul_mat_vec`).
  The MoE cost is in `launch_mul_mat_vec_q_moe`, reached via MUL_MAT_ID, which takes raw
  unreordered blocks and has no reorder concept.

So the prerequisite is a **reorder layout for 3-D expert tensors** plus a new ESIMD MoE
GEMV - a subsystem that also has to teach the XMX MoE kernel and the dequant-to-f16
paths about the new layout. Note that for this model iq4_nl appears **only** as
`ffn_down_exps`, so the usual stepping stone (reorder + MMVQ on the *dense* path) would
deliver nothing here.

## Things that keep being true on this codebase

1. **A green `test-backend-ops` means very little.** Three times in one session: the
   grouped GEMM reached 2 of 913 MUL_MAT_ID tests; the cast+add fusion fired zero times;
   the q8_0 ESIMD kernel passed 1126/1126 twice while silently corrupting odd rows. The
   suite only covers the shapes it happens to generate. Prove the path executed with a
   trace, and test the shapes your model actually uses.
2. **Kernel improvements return more than their GPU-time share suggests.** GPU busy is
   14.5%, yet a kernel worth ~35% of decode kernel time returned +5.3% wall from an 8%
   kernel gain. Sizing a decode optimization from a whole-run (prefill-contaminated)
   profile understated it by 5x.
3. **Guards go stale.** Several found this session declined work the backend could do,
   citing reasons that were true when written and are not now. Verify a guard's stated
   reason against the code it names.
4. **Estimates of where time goes have been wrong more often than right.** An agent
   estimated the fused GEMM was 60-80% A-decode bound; the FG_BN sweep refuted it. The
   one case where a prediction matched measurement (18.75% predicted FA occupancy vs
   20.0% measured) is the one where the fix worked first try.

## Open, ranked

1. **iq3_s grid in SLM.** 2 KB is far too large for register-indirect selection (a
   thread has 128 GRFs = 8 KB, and each thread would need its own copy), but it fits SLM
   easily and `esimd::slm_gather` is the right primitive. Targets the SBID stall
   directly. Settle it with a throwaway prototype before building layout work under it.
2. **Reorder layout for 3-D expert tensors**, the unlock for everything else on the MoE
   path. Large, no intermediate payoff on this model.
3. **`-np 2` / `-np 4`.** Layer-split at batch 1 leaves each card idle two-thirds of the
   time by construction; aggregate throughput should scale with concurrency without
   touching a kernel.
4. **`nsm` correction plus re-tuning** of getrows / topk-radix / count-equal together.
5. Pre-existing upstream bug worth reporting: `FLASH_ATTN_EXT` fails (ERR ~1.3) at
   `hsk=hsv=256, q8_0 K/V, permute=[0,2,1,3], kv_view=1` with **ragged** `n_kv`. Confirmed
   on pristine master. Unreachable from llama.cpp itself because `get_n_kv()` pads to a
   multiple of 256; needs both ragged n_kv and the permute (either alone passes).

## Addendum: radix select for topk_moe (+0.98% decode)

`GGML_SYCL_TOPK_MOE_RADIX=1` (default). `topk-moe.cpp`, reusing the `top_k_radix_select_f32`
device function that upstream PR #28670 added for the general TOP_K op (moved into
`topk-radix.hpp` as `static inline` so both callers share one copy; the split-row kernels
and the TOP_K entry point stay in the .cpp and are behaviourally unchanged).

The old kernel launched at **global 16, local 16** - a single 16-wide sub-group for a
512-expert top-k, using 16 of the machine's 1280 threads - and ran an iterative argmax:
for each of the 10 selections, every lane scanned its 32 registers then a 4-step shuffle
reduce, so ~320 serial comparisons plus 40 shuffles. Radix runs one work-group of 512 over
four digit passes.

Kernel 16.4 -> 6.4 us (VTune, 2201 instances each). End to end, arms interleaved:
**tg 22.530 -> 22.752 t/s, +0.98%**, prefill unchanged. Clean separation: the worst enabled
rep beats the best disabled rep. That matches the ceiling arithmetic (1.392 ms of 82 ms is
1.7%; ~60-70% of the kernel removed predicts ~1.0-1.2%).

**Selection runs on the raw logits**, which is safe because softmax and sigmoid are both
monotonic, so the top-k of the transformed values equals the top-k of the logits.

**A correction worth recording.** The obvious optimization is to skip the full 512-expert
softmax denominator when `with_norm` is set, on the grounds that it cancels:
`p_i / sum_sel p`. It does NOT cancel, because of the clamp - the weight is
`p_i / max(sum_sel p, clamp_val)`, and whenever `sum_sel < clamp_val` the clamp wins and
the full denominator survives. The implementation computes it unconditionally.

Ordering changed (radix emits arbitrary order, the old kernel descending) and that was
verified safe against all three consumers: `ggml_sycl_fuse_topk_moe` (the ARGSORT node has
no use outside the subgraph), `build_moe_ffn` (the combination is `sum_i experts[i] *
weights[i]`, so slot order is irrelevant provided pairing holds), and
`ggml_sycl_mul_mat_id` (re-sorts rows on expert id anyway). Pairing was checked
elementwise as `dict(id -> weight)` across 540 runs (6 config modes x 5 shapes x 3 row
counts x 3 seeds): ids set-equal everywhere, worst relative weight error 3.5e-07, and the
sigmoid mode bit-exact - which confirms the error is reduction-order rounding, not a
different selection.

**Known behaviour differences, deliberate:** NaN logits are ordered by radix key (positive
NaN sorts above +inf) rather than sanitized to -FLT_MAX as the old kernel does, so a NaN
logit would be selected rather than excluded; and exact ties may pick a different tied
expert (same value, so the same weight). Documented at the kernel.

`n_experts < 256` falls through to the old kernel: the group would be wider than the row
and the iterative argmax only scans <=16 registers per lane there.

**Pre-existing bug found and fixed while here:** `ggml_sycl_topk_moe_fusion` read
`nodes[node_idx]->op` at four points in the optional norm/scale tail with no
`node_idx >= n_nodes` guard, unlike every earlier step in the same detector. A graph whose
last node was the norm tail would read past the end. Ported from the CUDA version, predates
all of this work.
