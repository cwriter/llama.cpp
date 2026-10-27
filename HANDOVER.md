# Handover: llama.cpp SYCL optimisation, performance_uplift_mega_branch

Written 2026-09-20. Self-contained state for whoever picks this up.
Companion documents: `COMPUTE_BUFFER_ANALYSIS.md` (1226 lines, the measurement ledger) and
`BRANCH_CLEANUP.md` (the PR split plan). **Both contain corrected entries; trust the latest
section on any topic, and verify against the tree rather than the doc.**

## 1. What this is

Optimising the SYCL backend for Qwen3.8-Flash-Next (`qwen4exp`, UD-IQ4_XS, 177B total /
6B active) on 3x Intel Arc Pro B60 (bmg-g21), oneAPI 2026.1. Server script
`/root/run_llama_flash_next.sh`, context 131072, ubatch 1024, tensor split 31,33,36.

## 2. VERIFIED RESULT, shipped flag set vs stock master

Medians over 4 interleaved reps per side, cold prefills discarded, library hash verified per
arm, **generated text inspected at every point**.

| tokens | master pp | safe pp | delta | master tg | safe tg | delta |
|---|---|---|---|---|---|---|
| 64 | 45.25 | 84.98 | +87.8% | 19.23 | 23.58 | +22.6% |
| 256 | 104.99 | 209.90 | +99.9% | 19.35 | 23.51 | +21.5% |
| 16384 | 292.15 | 462.54 | +58.3% | 15.66 | 20.70 | +32.3% |
| 65535 | 241.15 | 365.44 | +51.5% | 9.05 | 16.36 | +80.7% |
| 113752 | 204.66 | 300.90 | +47.0% | 6.51 | 13.74 | +111.1% |

- Model load 102.1 s -> 39.1 s (-61.7%), stable across all arms.
- Compute buffer 1893.85 -> 869.85 MiB/card (reserve). **Net VRAM only -665 to -693/card**:
  buffers fall the full -1024 but the SYCL pool grows ~331/card.
- Free VRAM at peak: master `40 / 1485 / 769` MiB -> safe `574 / 2217 / 1497`. **Master runs
  40 MiB from a Level Zero eviction cliff**; the run script documents that 135 MiB of
  overshoot collapses throughput to 0.15 t/s. That is the real value of the memory work.
- `test-backend-ops` full suite: failure set IDENTICAL to master (16 CONV_2D, 1 ROLL,
  1 FLASH_ATTN_EXT), zero new.

Not yet folded into that number: the `fattn-mkl` drain removal, measured separately at
+2.2% / +4.7% / +6.4% prefill at 17k / 64k / 113k, decode flat.

CORRECTED 2026-09-20 by per-allocation backtrace attribution: `GGML_SYCL_FUSE_QSA_SCORE`
does NOT cost 32 MiB/card of pool (zero backtraces; its 32 MiB tile always found a cached
block), and `GGML_SYCL_SPARSE_FA` costs ZERO pool (decode-only, its requests never set a
pool record). Both earlier figures were indirect effects misread as allocations. The FA
staging clamp's 6.8 MiB/card is confirmed.

## 3. Current flag defaults (in source; see section 5 about binaries)

    ON : GGML_ALLOC_INPLACE_VIEWS, GGML_SYCL_FUSE_QSA_GATHER, _TOPK, _SCORE,
         GGML_SYCL_SPARSE_FA, GGML_SYCL_FA_MAX_MEM_MIB=256,
         GGML_SYCL_FUSE_CAST_ADD, GGML_SYCL_TOPK_MOE_RADIX,
         GGML_SYCL_ESIMD_Q8_0, GGML_SYCL_MMVQ_WIDE
    OFF: GGML_SYCL_FUSE_QSA_MASK, GGML_SYCL_FUSE_QSA_FA_MASK

`MASK` corrupted output from a 512-token allocator re-layout onward; its fix is applied and
validated but it is worth **0 MiB**, so it stays off pending a re-measure of what it is
actually worth. `FA_MASK`'s reader is UNREACHABLE in this deployment (needs
`Q->ne[1]>=32 && K->ne[1]>=1024` and oneDNN not chosen, i.e. `n_kv>=131072`); every
measurement ever taken of it is void.

## 4. THE FIVE TRAPS. Read these before measuring anything.

1. **Throughput hides corruption.** A configuration produced 528 t/s prefill (master: 292)
   with `predicted_n` correct, no errors, and every generation was `'ly//////////'`.
2. **An A/B inside the branch is blind.** Each fusion was text-checked against the branch with
   its own flag off - but the rest of the QSA set was on in BOTH arms, so both were corrupt,
   the texts matched, and the check said "no difference". **Check against MASTER.**
3. **`test-backend-ops` cannot see scheduler-level fusions.** Single-op graphs never present
   the multi-node chain plus `fusion_absorbs`. Its match with master proves nothing here.
4. **The first large prefill after load is cold** (21% penalty). Discard it, always.
5. **Peak-raising, not size, determines the buffer.** Two correct, bit-exact fusions were
   worth exactly 0 MiB. And a peak ladder shows what raises the high-water mark, not what is
   queued beneath it: removing a 256 MiB tensor bought 128 because another was hiding behind.

Also: text is NOT reproducible here even against itself. Same binary, same server, greedy:
one rep produced 1368 chars, the next 106. Judge coherence, never identity.

## 5. Tools built for this, all in the scratchpad

`/tmp/claude-0/-root-llama-cpp/d718f6cf-55e5-4d66-9ea3-981545e7ff12/scratchpad/`

- **`gpu_run.sh <arm.sh>`** - MANDATORY for all GPU work. Exclusive flock, refuses a stray
  server, waits for competing compilers, samples load every 20s, warns if a build ran inside
  the arm. Your script must do a whole arm and not background a server past its exit.
- **`qsa_probe.cpp` / `libqsa_probe2.so`** - LD_PRELOAD interposer on
  `ggml_backend_sched_graph_compute_async` reporting any tensor live at a fusion's run node
  whose address range overlaps its write target. **This is the only instrument that can see
  the fusion/allocator bug class.** Arm: `ns_probe2_arm.sh <label> "<flags>" "448 512 16384 65536"`,
  require `real hazards: 0` AND non-zero `active` counts.
- `alloc_repro.cpp` - GPU-free reproduction through `ggml_gallocr` alone.
- `GGML_ALLOC_PEAK_DEBUG=1` names every tensor raising a chunk's peak; compile-time
  `GGML_ALLOCATOR_DEBUG` in ggml-alloc.c dumps the whole live set sorted by offset.
- `GGML_SYCL_MEMTRACE=1` for net VRAM. Resize warnings are `#ifndef NDEBUG` so invisible in
  Release; use memtrace to prove no runtime buffer resize.

## 6. Build directories (do not cross-contaminate)

    /root/llama.cpp/build   shipped set, hash 2a3823f89132d1ef (mtime 15:12 Sep 19).
                            SOURCE HAS SINCE DIVERGED: default flips, the MASK fix, and the
                            drain change are all in source but NOT in this binary.
    /root/build-maskfix     MASK fix, validated
    /root/build-waits       drain removal, validated
    /root/wt-master         stock master baseline, GGML_VULKAN=ON to match
    /root/wt-dsv4-hc        the DSV4_HC PR branch (merged by the user)

## 7. THE RULE that came out of this

**A fusion that RUNS at node i and WRITES node j must report every node in `[i, j)` as
absorbed.** Otherwise ggml-alloc frees the kernel's inputs inside the span (correctly, for the
schedule it was told about) and best-fit may hand node j that freed memory, so the kernel
destroys its own input. Absorbing views additionally requires `ggml_gallocr_release_parent` to
recurse through absorbed views, or the span's sources leak.

`GATHER`, `TOPK` and `SCORE` currently violate this. They are clean at 121 probed graphs and
~80 re-layouts, but survive on buffer size ratios, not by construction. Hardening patch
drafted at `scratchpad/qsa-harden-spans.diff`; its memory cost is UNMEASURED and could be
significant (worst case ~128 MiB/layer transient for TOPK).

## 8. Open work, ranked

1. **Pool growth: DIAGNOSED 2026-09-20, one-line fix worth ~226 MiB/card. NOT YET APPLIED.**
   98% of the 331 MiB/card is `packed_b` in the grouped MoE GEMM
   (`ggml/src/ggml-sycl/fused-gemm.cpp:344-395`, reached from `ggml_sycl_mul_mat_id`,
   gated by `GGML_SYCL_GROUPED_GEMM` which defaults to 1 and is therefore NOT in the flag
   set people bisect over). Full attribution, per card:

       packed_b (grouped MoE GEMM)   +324.6    <- the whole problem
       MKL chunked FA scratch          +6.8
       tile FA                         +0.2
       everything else                  ~0     (identical to master)

   Mechanism: `n_tiles = sum_e ceil(rows_e / FG_BN)` depends on MoE ROUTING, not shape, so
   the request jitters UPWARD about 1.5x as a prefill covers more experts. `pool_leg::alloc`
   reuses a cached block only when `b.size >= size`, and `pool_leg` never frees until
   destruction, so each new maximum strands its predecessor forever. Observed ladder on
   SYCL0: 69.56 / 73.50 / 82.36 / 88.10 / 94.01 MiB, i.e. 407.5 held for 94.0 needed.

   Only one `packed_b` is live at a time, so **~226.5 MiB/card is stranded cache**. Fix:
   round `K * Npad` up to a coarse granularity before the `pool_alloc` while still passing
   the true `Npad` to the kernel as pack stride. One line, in a file this branch owns.
   Generic alternative: bucket `look_ahead_size` in `ggml_sycl_pool_leg::alloc` instead of
   `1.05 * size` - same win plus the FA ladder, but changes behaviour for every SYCL user.

   Expected: pool peak 1550 -> ~865 MiB, net win over master 693 -> ~921 MiB/card, pool's
   share of the win 32% -> 11%.

   Tooling: `scratchpad/memprobe.cpp` LD_PRELOAD interposer on `zeMemAllocDevice`/`zeMemFree`
   with backtraces, resolved by `scratchpad/resolve_bt.py`. Reproduced the original totals
   exactly (safe 1550 vs 1549, master 556.1 vs 556).

   Robustness gap found alongside: `ggml-cuda`'s pool has `clear_pool()` and a
   retry-on-OOM path; the SYCL pool has NEITHER, so stranded blocks are a hard peak rather
   than recoverable slack. Porting that pattern is small and precedented.
2. **Span hardening: MEASURED 2026-09-20, cost is ZERO. ADOPT. NOT YET APPLIED.**
   `scratchpad/qsa-harden-spans.diff`. Reserve identical to the digit on all three devices
   (869.85 / 877.10 / 877.10), net VRAM change below the pool's own ~100 MiB chunk
   granularity, throughput within the 0.4-0.6% within-side spread. The drafter's worst case
   (128 MiB/layer for TOPK) did NOT materialise.

   Why it is free, and this is structural rather than layout luck: **every node the patch
   newly absorbs is a view whose owner is already absorbed, so it never held a buffer.** The
   patch changes only WHEN the span's external sources are released.

   Proven in effect, not silently declining: absorbed flags per graph 120 -> 204, decomposing
   exactly as 12 QSA layers x (TOPK 5->7, SCORE 5->10) = +84. Proven free: the full probe
   stream over 448 and 16384 prefills, 1406 lines including every chain's destination
   ADDRESS, diffs to ZERO lines between builds - the allocator layout is bit-identical.
   Hazards 0, TOPK/SCORE active=1392 each, text OK at all sizes, FLASH_ATTN_EXT failure set
   identical.

   Adopt for the maintenance argument as much as the safety one: the patch makes ABSORPTION
   EQUAL THE EXECUTION SKIP. The runners already return `3`, `c.i_topk - i` and `c.i_out - i`
   as nodes consumed and the compute loop does `i += n; continue`. Before the patch those two
   numbers disagreed on all three fusions; after it they are the same expression. Divergence
   between them is exactly what corrupted MASK.

   Before landing: delete the now-unused `SYCL_QSA_TOPK_MAX_SPAN` and
   `SYCL_QSA_SCORE_MAX_SPAN` constants (two new warnings). Note the GATHER 2->3 change is
   untested by measurement - standalone gather never fires in this model because TOPK claims
   the chain first (`GATHER chains=0`) - so it rests on the invariant argument, not data.

   Builds: `/root/build-harden` (patched) and `/root/build-harden-base`, both from the
   snapshot `/root/src-harden`. `/root/llama.cpp` was NOT modified.
3. **IQ3 reorder** - three patches staged (`scratchpad/0001..0003`), NOT started. The 94
   IQ3_S tensors are the MoE expert weights and we reorder no IQ type at all; largest
   untouched decode lever. Patch 0001 alone does NOTHING here (these are 3-D MUL_MAT_ID
   tensors and the reorder is gated on MUL_MAT), which is what 0003 fixes.
4. **Bitmask `attn_inp_kq_mask`** - 256 -> 16 MiB, NOT started. Cross-cutting.
5. **PR split** - 1 of 11 groups done (`sycl-dsv4-hc`, merged). See BRANCH_CLEANUP.md.

**Do NOT** downcast `inp->bias` to f16: it holds a `1e9f` sentinel that becomes `+inf`, and
`inf + (-inf) = NaN` at the mask add. Needs an encoding change, not a type change.
**Do NOT** pursue sparse prefill by tile SKIPPING: measured `chunks_touched = 1.0000` at every
block size. Gathering remains open (~1.5x on FA) as a separate project.

## 9. Process constraints (AGENTS.md)

No PR descriptions or commit messages beyond a concise line with `Assisted-by:` (never
`Co-authored-by:`). No commits or pushes without explicit per-action approval. `git push` and
PR creation require the user to acknowledge that automated PR submission can cause a
contributor ban. No files under `tests/`. ASCII only, comments 1-2 lines.
