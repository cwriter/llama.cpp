# SYCL performance work: what was tried, what blocks, what is left

Working notes for agents continuing the SYCL performance work on `performance_uplift_mega_branch`. Target: Qwen3.8-Flash-Next (qwen4exp) `unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS` on 3x Intel Arc Pro B60 (bmg-g21), `--split-mode layer --tensor-split 31,33,36`, `-fa on`, q8_0 KV, `-c 131072`, ubatch 1024. Numbers are measured unless marked *estimate*.

## Measuring on this box

- **Check the text, not just the speed.** A corrupt build decodes at full speed. Every end-to-end run asserts a fixed answer at temperature 0 (short arithmetic plus a needle at 13k/54k/104k).
- **The first large prefill after a load is cold** (183 vs 466 t/s at 4k). Discard it in every arm. In llama-bench the first arm of each repetition reads ~360-376 pp2048 against ~703 for the next, whatever the mode.
- **Guard CPU load, not just the GPU.** Prefill has a host-side part (`-t 4 -tb 16`); a concurrent build costs double digits.
- **md5 the library each arm actually loaded** (`/proc/<pid>/maps`). A copied `.so` defeats CMake's rebuild; production scripts use the `/usr/local` install.
- **VRAM truth is fdinfo** (`drm-total-vram0`, `drm-active-gtt`); xpu-smi sees nothing. Large active GTT means paging. `sched_reserve` does not include the SYCL pool or load-time temporaries.
- **A tensor saves memory only if it sets the high-water mark.** Re-read the allocation ladder after every change.
- **An A/B inside one branch can have both arms corrupt and agreeing.** Check against master.
- QSA top-k is nondeterministic, so byte-identical text is not a valid acceptance test for anything that touches it. CPY-to-quant tests are flaky in the baseline too; repeat the baseline before believing a regression.

## Where the time goes

- **Decode is submission-bound**: ~17 ms of kernel work in a ~61 ms token, ~2800 dispatches per token. Only fewer dispatches help.
  - Fusing the 96x hyper-connection motif gave +6.3%.
  - DAG lanes lost 33% (every cross-lane barrier is another submission).
  - SYCL graph capture never engages on this workload and costs ~22% when on.
- **Prefill** is dominated by the grouped MoE GEMM (iq3_s gate/up, iq4_nl down) and, at long context, by attention.
- **mmvq for iq4_nl is gather-bound; that is settled.** The table gathers cost nothing end to end. Keep the upstream byte gathers; a permuted layout loses 7.7%.

## Merged (on by default unless noted)

- **Grouped MoE GEMM:** one launch instead of one GEMM per expert; +33% prefill. Operands are read through the route map with no staging copies.
- **`GGML_SYCL_IQ3_REORDER=1`:** +12.8% decode.
- **Hoisting `->data` out of kernels to the host:** +29-32% decode. Fusions must respect tensor lifetimes: a fusion that runs at node i and writes node j must absorb all of [i, j).
- **Fusions:** the hyper-connection motif, ELEMENTWISE, MUL_ADD, MOE_REDUCE, MOE_GLU_ID, NORM_SCALE, GLU_NCOLS, FLAT_BATCH (`GGML_SYCL_FUSE_DEFAULT`).
- **ESIMD kernels:** multi-column ESIMD mat-vec, q8_0 GLU up to 8 columns, flat batched mat-vec (`GGML_SYCL_ENABLE_ESIMD=3`).
- **The oneMKL flash attention reads the packed mask directly.**
- **qwen4exp:** the QSA `set_rows` zeros are one filled row repeated, instead of a full-size leaf.
- **`GGML_SYCL_MEM_SAVE=7`:** -553/-553/-970 MiB fdinfo peak at 131k, same text. Three bits:
  - REORDER_CHUNK: weight reorder through a ≤32 MiB temp;
  - POOL_RELEASE: free outdated pool buffers before growing; inert with SYCL graphs;
  - PACKB_EXACT: grouped-GEMM packed B at its real size.
- **QSA sparse flash attention, `GGML_SYCL_FUSE_QSA_FA_MASK=3`, default off.** Tiles of 64 query tokens share the union of their top-k lists, gathered once and run through the same oneMKL XMX GEMMs. It uses the bitmap path at n_kv ≤ 8192 and dense scratch for decode.
  - Prefill, 2 reps: 334 -> 478 t/s (+43%) at 120k and 429 -> 494 (+15%) at 60k.
  - Flat at 16k; -3% at 2.5k, within run-to-run spread.
  - Decode unchanged; -73..-78 MiB reserve.
  - Waiting for a third repetition before it becomes the default.
  - It also fixes the oneMKL normalize, where a fully masked row now gives 0 instead of NaN.

## Tried and rejected (do not redo without a new idea)

- **iq4_nl permuted table layout:** -7.7%.
- **Grouped-GEMM tile width:** FG_BN=32 is optimal. 64 costs 5.1% and 16 costs 3.2%; 128 corrupts output while reporting 764 t/s.
- **Per-expert A-reuse rewrites:** reducing the A-decode count does not help.
- **DAG lanes / multi-queue decode:** -33%, and the multi-queue ordering has holes.
- **SYCL graphs for decode:** capture never engages and trying costs 24%.
- **q8_0 KV promoted inside attention ("online"):**
  - oneMKL XMX variant: 11.8x slower.
  - TILE variant: +7..16% slower at decode and 3.29x slower at Q=512.
  - Staging is the cheap part, and 2-byte-aligned q8_0 quants mean 8x load messages.
  - Not dead: with aligned wide loads a probe measures 1.3-1.8x faster than staging.
- **KV SoA layout (`GGML_SYCL_KV_SOA=1`):** +10% prefill from the layout itself, -2% decode even with the online loader.
  - Default off.
  - It has a pre-existing NaN in the dense oneDNN/MKL prefill.
  - Span 256 is forced by 34N ≡ 0 mod 16 and only fits head dim 256.
- **First QSA sparse FA (per-token scalar gather):** 83-89 ms per layer at every depth, slower than dense. Replaced by the union-tile version above.
- **`GGML_SYCL_FUSE_QSA_MASK=1` alone:** a speed win, +36/+19/+10% prefill at 13k/54k/104k on one repetition. Zero memory saving.
- **Removing host waits at decode:** flat, because nothing was waiting. The cost is per dispatch.

## Memory and pipeline parallelism (the current goal)

- **Any `-ot` disables pipeline parallelism** (`llama-context.cpp`, `!model.has_tensor_overrides()`). The production `-ot per_layer_token_embd\.weight=CPU` moves nothing: the table is lazily read and `CPU_Mapped` either way. Its only effect is keeping 131k out of pipeline mode, which pages today.
- **Pipeline mode duplicates every graph input per copy.** `GGML_SCHED_MAX_COPIES` is 4 by default and is a compile-time constant in the core scheduler. The f16 causal KQ mask alone is 256 MiB per card at 131k x 1024, so 1 GiB per card in pipeline mode.
- **`-np` does not multiply `-c`.**
  - `--no-kv-unified` gives each slot `-c/np`.
  - `-kvu --kv-unified-per-slot N` without `-c` sizes the pool to `np*N`.
  - Unified KV means one stream (validated paths) but a mask of up to `[np*n_kv, n_ubatch]`.
  - Separate KV means multi-stream paths (`ne[3] > 1`), none of them validated on this branch.
- *Estimate:* `-np 2` gives ~1.7-1.9x aggregate decode, because decode is submission-bound.

### In worktrees, not merged yet

- **`.worktrees/pipeline_ring` (branch `pipeline-ring`): the `cpy_tensor_async` fix.**
  - The copy ran on the destination queue with nothing ordering the source queue after it. In pipeline mode the source's next ubatch can overwrite the activation mid-read.
  - Fix: `GGML_SYCL_ASYNC_COPY` bit 2 (SRC_RING, default on) stages through a ring on the source device. Depth is `GGML_SYCL_COPY_RING_DEPTH`, default 2. With the bit off, a source-queue barrier is used instead.
  - The production scripts set `ASYNC_COPY=3`, which leaves the ring off.
  - Also removes oneDNN flash attention's multi-GPU host `wait_and_throw()`, which drained every card once per attention node below 131k cells.
  - Tests are written in `scratchpad/pipe/`: KL divergence of pipeline vs non-pipeline for old/barrier/ring, 64k prefill, `-np 2` unified. Not run yet.
- **`.worktrees/compact_mask` (branch `compact-mask`, on top of the ring change): compact KQ mask allocation.**
  - `GGML_SYCL_KQ_MASK_BITS` becomes a bitset: 1 pack the upload, 2 lend the unused tail to the pool, 4 allocate only the packed size (takes precedence over 2).
  - Compact applies only to masks a scheduled graph showed to `graph_optimize`, and only while:
    - every reader is taught (ADD, CPY/cast, the cast+add and QSA top-k fusions, FA);
    - no FLASH_ATTN_EXT/SOFT_MAX uses ALiBi, which is llama's only source of values other than 0/-inf.
  - A non-binary upload switches it off for good.
  - Core changes, both in `ggml-backend.*`:
    - the debug-only `assert(size >= ggml_nbytes)` in `ggml_backend_buft_get_alloc_size` becomes a comment;
    - new `ggml_compacted_nbytes()` = min(ggml_nbytes, get_alloc_size).
  - *Estimate:* -240 MiB per card, -960 MiB per card in pipeline mode.
  - Built; end-to-end tests are written (`scratchpad/cm/e2e.sh`) but not run.
- **Mask tail reuse (mode 2)**, from the mask-tail agent: stash `9f1ec75761b1` and `.worktrees/mask_tail_wip.patch`.
  - Measured: -137/-138/-69 MiB on the plateau, 0 overlap hazards.
  - Throughput not measured cleanly.
  - `TOPK_QSA(64,256,2,1,200)` aborts with a double free after other TOPK_QSA cases in every mode. Unresolved.

## Blockers and known bugs

- **`qsa-mask.cpp` packed-mask guard** checks the ADD twice instead of `ad->src[1]`. With `FUSE_QSA_MASK`/`FA_MASK` plus mask packing, the fusion would read bits as f16. The compact check refuses that combination; modes 1/2 are still exposed.
- **Intermittent `MUL_MAT_ID` q4_1 failure:** order-dependent, 0/40 in isolation. Likely a stale-memory read exposed by the grouped GEMM's pool layout; pool NaN-poisoning would find it. The 3 `MUL_MAT_ID` q8_0 amax=1e5 failures are pre-existing.
- **The oneMKL flash attention returned NaN for fully masked rows before the QSA sparse FA merge.** Anything branched earlier still has it.
- **SoA prefill NaN** in the dense oneDNN/MKL path (`GGML_SYCL_KV_SOA=1`).
- **Pipeline overlap is capped at ~1.5x (*estimate*)** by the 4 x 8 MiB pinned staging ring for host uploads: the 256 MiB mask wraps it 8 times per ubatch. Compact masks remove most of that traffic.
- **Per-split host syncs without events** are in the core scheduler.
- **np=3 with non-contiguous slots** splits a decode into several graphs; fixing that is a core change.
- **The PLE gather claim may never fire:** the table is `CPU_Mapped`, and the SYCL claim matches the buffer name `"CPU"`. Unverified.
- **The reordered grouped GEMM does not deliver** for prefill (iq3_s reorder moved prefill -0.1%). iq4_nl has no reorder path at all.
- **An upstream ggml-alloc view-inplace branch is dead code**: fixing it saves 128 MiB per card. Separate upstream PR, not SYCL.
- **The QSA indexer chain expands 128 -> 512 MiB** (get_rows between two permute/cont pairs). The block-bias cast of the mask to f32 is another 512 MiB at 131k. Both are candidates for a fused gather.

## What could be done next

1. Run the pipeline-ring tests, then build with `-DGGML_SCHED_MAX_COPIES=3` and measure pipeline mode at 131k: fdinfo, prefill, KL divergence.
2. Run the compact-mask end-to-end tests (production and pipeline, modes 0 vs 4), then make mode 4 the default if they hold.
3. `-np 2` unified vs separate KV. Validate the multi-stream paths before relying on separate KV.
4. After the third repetition, default `FUSE_QSA_FA_MASK=3` if 2k stays within noise.
5. Fix the `qsa-mask.cpp` guard; poison the pool to find the q4_1 stale read; resolve the TOPK_QSA double free.
6. Aligned wide-load q8_0 KV promotion (1.3-1.8x over staging in a probe); an iq4_nl reorder path; making the reordered grouped GEMM actually fire for prefill.
7. Kernel speedups are worth shipping even when end to end is flat: fewer operations mean less energy. Measure J/token from the xe hwmon counters, `/sys/class/drm/cardN/device/hwmon/hwmon*/energy{1,2}_input` (µJ).
