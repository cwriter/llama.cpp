# SYCL performance work: what was tried, what worked, what is left

Working notes for agents continuing the SYCL performance work on `performance_uplift_mega_branch`.

- **Target:** Qwen3.8-Flash-Next (qwen4exp), `unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS`, on 3x Intel Arc Pro B60 (bmg-g21).
- **Configuration:** `--split-mode layer --tensor-split 31,33,36`, `-fa on`, q8_0 KV, `-c 131072`, ubatch 1024, `-ot per_layer_token_embd\.weight=CPU`.
- Numbers are measured unless marked *estimate*.

## Where the branch stands against master

llama-bench, same flags, each binary with its own defaults, 2 runs per cell in the order master, branch, branch, master. Master is `0f8a414b7`; the branch is `038314ae1`, which predates mode 3 on by default and the softmax change.

| depth | master pp2048 | branch pp2048 | master tg128 | branch tg128 |
|---|---|---|---|---|
| 0 | 165-171 | 815-820 (4.9x) | 15.8-16.0 | 33.8-34.0 (2.1x) |
| 16k | 225-229 | 683-684 (3.0x) | 15.1 | 28.8 (1.9x) |
| 64k | 174-175 | 409 (2.3x) | 8.5-8.6 | 21.3 (2.5x) |
| 112k | 139-140 | 235-351 (1.7-2.5x) | 6.3 | 16.3-16.8 (2.6x) |

Server results after mode 3 became the default, all defaults, 131k context:
- **Prefill:** 657 t/s at 13k and 529 at 104k (the tree's previous defaults gave 356 at 104k).
- **Peak VRAM:** 22166 / 21570 / 22435 MiB.
- **Correctness:** all answers correct.

## Measuring on this box

- **Check the text, not just the speed.** A corrupt build decodes at full speed. Every end-to-end run asserts a fixed answer at temperature 0: short arithmetic plus a needle at 13k/54k/104k.
  - The model sometimes refuses the planted needle as "prompt injection", in any configuration. Judge those runs by the reasoning content.
  - QSA top-k is nondeterministic, so byte-identical text is not a valid acceptance test.
- **The first large prefill after a load is cold.** Discard it in every arm.
  - In llama-bench, whichever arm runs first after another binary or a load reads ~360-600 pp2048 instead of ~810.
  - Use ABBA ordering or discard the first arm.
- **Long-context prefill is bimodal.** The same binary gives, for example, 235 or 351 t/s at 112k, or 452 or 635 at 16k.
  - It is undiagnosed and hits every configuration.
  - A single sample cannot decide a 10-20% difference. Use three interleaved rounds; they often come out tight (+/-0.1%).
- **Guard CPU load, not just the GPU.** Prefill has a host-side part (`-t 4 -tb 16`), and a concurrent build costs double digits.
- **md5 the library each arm actually loaded** (`/proc/<pid>/maps`).
  - A copied `.so` defeats CMake's rebuild.
  - Production scripts run the `/usr/local` install, not the branch build.
  - Editing sources while a build compiles produces a binary without the edit. It happened: a "fixed" build still carried the old veto rule. Check with `strings` or object mtimes.
- **VRAM truth is fdinfo** (`drm-total-vram0`, `drm-active-gtt`); xpu-smi sees nothing.
  - Large active GTT means paging.
  - `sched_reserve` does not include the SYCL pool or load-time temporaries.
- **A tensor saves memory only if it sets the high-water mark.** Re-read the allocation ladder after every change.
- **An A/B inside one branch can have both arms corrupt and agreeing.** Check against master.
- **test-backend-ops never calls `graph_optimize`.** Anything decided there (compact-mask approval, allocation deps) must be proven on a real server.
- **Never force the oneMKL flash attention with a small `GGML_SYCL_FA_MAX_MEM_MIB`.** The cap also sizes oneMKL's KV chunks: at 32 MiB, one 32k perplexity pass took 9.6 h and held the shared GPU queue overnight. Use `GGML_SYCL_FA_ONEDNN=0`.
- **Shared GPU queue.** Scratchpad scripts:
  - `gpu_run.sh`: all cards.
  - `gpu_any.sh`: one card; always pass `-b SYCL0`.
  - `build_run.sh`: wraps builds.

  One full-model job blocks everyone, so:
  - wrap every queued job in `timeout`;
  - kill any server you start on every exit path, with a trap;
  - check progress yourself, because a hung job never notifies.

  A leaked `llama-server` makes `gpu_run.sh` abort every later job.

## Where the time goes

- **Decode is submission-bound.** About 17 ms of kernel work in a ~61 ms token, ~2800 dispatches per token. Only fewer dispatches help.
  - Fusing the 96x hyper-connection motif gave +6.3%.
  - DAG lanes lost 33%.
  - SYCL graph capture never engages on this workload and costs ~22% when on.
- **Prefill** is dominated by the grouped MoE GEMM (iq3_s gate/up, iq4_nl down) and, at long context, by attention.
- **oneMKL flash attention at long context:** before the one-pass change, softmax was 50-56% of attention time and the GEMMs ~41%.
  - The softmax is bandwidth-bound: f32 scores round-trip through DRAM.
  - The q8_0 KV dequant is only ~1%; that is settled.
- **The iq4_nl mat-vec is table-bound, not load-width-bound; that is settled.**
  - The table gathers cost nothing end to end.
  - The dense q8_0 ESIMD mat-vec (39% of decode) and the q6_K LM head are already wide and near bandwidth.

## Merged (on by default unless noted), in order

| change | switch | measured result |
|---|---|---|
| Grouped MoE GEMM, operands through the route map | — | +33% prefill |
| IQ3_S weight reorder | `GGML_SYCL_IQ3_REORDER=1` | +12.8% decode |
| `->data` hoisted to the host | — | +29-32% decode |
| Fusions (hyper-connection motif, ELEMENTWISE, MUL_ADD, MOE_REDUCE, MOE_GLU_ID, NORM_SCALE, GLU_NCOLS, FLAT_BATCH) | `GGML_SYCL_FUSE_TYPES` | hc motif +6.3% decode |
| ESIMD multi-column mat-vec, q8_0 GLU up to 8 columns, flat batched mat-vec | `GGML_SYCL_ENABLE_ESIMD=3` | kernel-level |
| qwen4exp QSA zeros as one repeated row | — | -32 MiB reserve per card |
| `MEM_SAVE`: chunked weight reorder, pool release before growing, exact grouped-GEMM packed-B size | `GGML_SYCL_MEM_SAVE=7` | fdinfo peak -553/-553/-970 MiB |
| QSA sparse FA (union tiles on XMX); later default on, see below | `GGML_SYCL_FUSE_QSA_FA_MASK` | — |
| Source-side copy ring for cross-device copies; oneDNN multi-GPU host wait removed | `GGML_SYCL_ASYNC_COPY` bit 2, `GGML_SYCL_COPY_RING_DEPTH=2`; clear the bit for a barrier | correctness; ~86 MiB per card for the ring; pipeline prefill 545 -> 547 |
| Wide aligned loads: grouped-GEMM B pack (4.45x), gated delta net (-24%), hc_pre/hc_post, f32->f16 convert | `GGML_SYCL_WIDE_LOADS`; `MMID_SCHED` bit 18 restores the narrow B pack | pp2048 707.7 -> 811.0 (+14.6%); prefill energy ~-10% |
| Compact KQ mask (see below) | `GGML_SYCL_KQ_MASK_BITS=5` | fdinfo peak -240..-272 MiB per card; -928..-957 MiB in pipeline mode |
| oneMKL FA per-KV-head coalesced normalize; SoA staging NaN fix; oneDNN stride fix | `GGML_SYCL_ENABLE_MKL_FA=2` restores the old normalize | FA -21.5% at 2k, -4% at 13k |
| q8_0 MoE weight reorder on by default | `GGML_SYCL_REORDER_TYPES=3` for the old default | tg128 +2.5%, pp2048 +1.0%; the reordered q8_0 mat-vec is 5.1x faster |
| QSA sparse FA mode 3 reads the packed mask; on by default | `GGML_SYCL_FUSE_QSA_FA_MASK=0` restores dense | see below |
| One-pass oneMKL FA softmax, VKQ GEMM accumulating in place, L2-sized query tile; same softmax in the mode 3 union path (chunk 4096) | bit 4 of `GGML_SYCL_ENABLE_MKL_FA` (e.g. `=5`) restores the old kernels | *being merged when this was written*; see below |

**Compact KQ mask.**
- `GGML_SYCL_KQ_MASK_BITS` is a bitset:
  - bit 0 packs the causal mask to 1 bit per cell;
  - bit 1 lends the unused f16 tail to the pool;
  - bit 2 allocates only the packed size (takes precedence over bit 1).
- Compaction is chosen only when all of these hold:
  - the mask was shown to `graph_optimize` by a scheduled graph;
  - every reader is taught (ADD, the f32 cast, the cast+add and QSA fusions, FA);
  - no attention op uses ALiBi.
- A non-binary upload switches it off for good.
- Core changes, both in `ggml-backend.*`:
  - the debug-only `assert(size >= ggml_nbytes)` became a comment;
  - new `ggml_compacted_nbytes()`.

**QSA sparse FA, mode 3.**
- **How it works:** tiles of 64 query tokens share the union of their top-k lists, gathered once and run through the oneMKL XMX GEMMs. At n_kv ≤ 8192 the bitmap goes into the dense oneMKL kernel; decode uses dense scratch, which is bit-identical to mode 0.
- **Speed**, fixed build, 3 interleaved rounds against the defaults of the same build: +12.8% at ~48k, +28.5% at ~92k. Equal at 2.5k and in llama-bench.
- **Memory:** compute buffer 598/605/605 -> 525/527/527 MiB.

**One-pass oneMKL softmax.**
- **Kernel:** -17..-23% at every depth; the mode 3 union path is -12..-18%.
- **Server:** +7.5% prefill at 123k; prefill energy per token -1.7/-3.5/-7.8% at 15k/65k/123k; decode unchanged; memory +5 MiB.

## Evaluated and not adopted (do not redo without a new idea)

- **iq4_nl permuted table layout:** -7.7%. A 256-entry LUT for the iq4 grouped-GEMM A stage: -6%.
- **Grouped-GEMM tile width:** FG_BN=32 is optimal. 64 costs 5.1% and 16 costs 3.2%; 128 corrupts output while reporting 764 t/s.
- **Per-expert A-reuse rewrites:** no gain.
- **DAG lanes / multi-queue decode:** -33%.
- **SYCL graphs for decode:** capture never engages, and trying costs 24%.
- **q8_0 KV promoted inside attention, and fused KV staging kernels:** all slower. KV dequant is only ~1% of attention time, so there is at most ~1% to gain.
- **KV SoA layout (`GGML_SYCL_KV_SOA=1`):** +10% prefill, -2% decode; default off.
  - Its NaN is fixed.
  - Span 256 fits only head dim 256.
- **Per-token scalar-gather sparse FA:** slower than dense at every depth.
- **`GGML_SYCL_FUSE_QSA_MASK=1` alone:** a speed win, but superseded by mode 3. No memory saving.
- **Removing host waits at decode:** flat. The cost is per dispatch.
- **KQ GEMM writing f16 scores:** would halve softmax traffic, but f16 at |s| 20-30 costs ~1.5% in exp(). Rejected as not exact.
- **oneDNN flash attention at long context (`GGML_SYCL_FA_MAX_MEM_MIB=1024`):** kept at default 256.
  - Output matches oneMKL (KL divergence 0.0072 vs noise floor 0.0065, same top-1).
  - About +25% prefill at 120k over the old defaults, for +270-300 MiB per card.
  - Mode 3 is faster at long context. Combined with mode 3 after the routing fix, a single sample gave 570 at 60k and 511 at 120k; not repeated.
- **Mask tail reuse (`KQ_MASK_BITS` bit 1):** merged as an option, off.
  - -69..-138 MiB on the plateau only; superseded by compaction.
  - Its test double free was a test bug (the causal init wrote framework tensors with the wrong shape); fixed.
- **Pipeline mode as it stands:**
  - +1.5% prefill at 48k and flat at 104k, even with memory fixed.
  - The ring and the oneDNN wait removal did not change it.
  - The pipeline-overlap agent is tracing where the cards serialize (see below).
- **`-np 2` unified KV (`-kvu --kv-unified-per-slot 131072`):** aggregate decode 31.0 -> 43.0 t/s (+39%), concurrent needles correct.
  - Card 3 peaked at 24044 of 24430 MiB before the compact mask.
  - Not adopted as a default: it is a deployment choice.

## Open work

- **Pipeline overlap** (agent in `.worktrees/pipeline_overlap`): timelines of pipelined vs non-pipelined prefill were recorded. Suspects:
  - host syncs per split in `ggml_backend_sched_compute_splits` (core);
  - the 4 x 8 MiB pinned staging ring;
  - host waits inside SYCL ops (the MUL_MAT_ID ids download).

  Core scheduler changes are to be proposed, not implemented.
- **MTP** (branch `mtp-on-mega-2`, `.worktrees/mtp_mega2`):
  - The mega branch plus upstream PR #28243 (qwen4exp MTP graph) plus the MTP K/V-store-only decode.
  - The head file is `/root/models/mtp/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.
  - The draft layer needs `-ts 34,34,32 -devd SYCL2` so card 3 does not page.
  - A runtime comparison (no MTP vs MTP, decode and acceptance) is queued.
  - Every number above is without MTP.
- **The bimodal long-context slowdown:** undiagnosed; it may be host or runtime scheduling rather than kernels.
- **`-np 2` with separate KV:** the multi-stream paths (`ne[3] > 1` masks, 4-D K/V in oneMKL FA, QSA kernels) are unvalidated.
- **Small kernel items:**
  - *Estimate:* the q8_0 ESIMD mat-vec scale load as one 16 B-aligned load, ~15% of that kernel (~5% of decode).
  - *Estimate:* iq3_s MoE mat-vec `signs` as one d32 instead of two d16; small.
  - `flash_attn_tile` spills 768 B at decode (0.7%).
- **Memory:** the upstream ggml-alloc view-inplace dead code is worth 128 MiB per card; separate upstream PR.
- **Known bugs:**
  - Intermittent order-dependent `MUL_MAT_ID` q4_1 failure; pool NaN-poisoning would find it.
  - The 3 `MUL_MAT_ID` q8_0 amax=1e5 failures and the f16 `FLASH_ATTN_EXT kv=8192 nb=67` failure are pre-existing.
- **Production** (`/root/run_llama_flash_next.sh`):
  - It still runs the old `/usr/local` binary with `GGML_SYCL_ENABLE_GRAPH=1`. Graphs cost ~22% decode and make `MEM_SAVE`'s pool release inert.
  - Installing the branch build and turning graphs off is the user's decision.
- **Energy:** kernel speedups are worth shipping even when end to end is flat. Measure J/token from `/sys/class/drm/cardN/device/hwmon/hwmon*/energy{1,2}_input` (µJ, sum all cards).
