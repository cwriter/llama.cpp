# Splitting `performance_uplift_mega_branch` into upstream PRs

Updated 2026-10-27. It replaces the earlier list and `mega-master-review-20261010.md`. Their commit hashes predate the 2026-10-10 rebase and no longer exist; every hash below is on the current branch history.

- **Compared:** upstream master `a9191799` and the branch at `bac86332`.
- **Merge base:** `19ff817f`, which the branch merged on 2026-10-27.
- **Size:** about 158 first-parent commits. The diff to master is 175 files and +39.6k / -2.5k lines, of which about 12k are notes, scripts and images that never go upstream.
- **Find a subject:** use `git log --first-parent --no-merges --format='%h %s' upstream/master..origin/performance_uplift_mega_branch`.

This file is for an agent that **creates the extraction branches and benchmarks them**. The user reviews each branch, writes the commit messages and the PR text, and submits.

## Rules (read first)

1. **AGENTS.md applies in full.** In particular:
   - The agent never writes PR descriptions, PR commit messages or reviewer replies.
   - The agent never opens a PR and never pushes to ggml-org.
   - Large changes start as a discussion.
2. **Commits on an extraction branch are working commits.**
   - Subject: `WIP <chunk id>: <one line naming the change>`, plus `Assisted-by: <assistant name>`. Never `Co-authored-by`.
   - The user rewrites the history and messages before submitting.
   - Push to the `cwriter/llama.cpp` fork only when the user approves that push.
3. **One branch per chunk, named `pr/<id>-<short-name>`, from current upstream master.** Stack it only on the chunks listed under "Depends".
   - Bring code over by hunk: `git checkout -p <commit> -- <file>`, or `git diff <a> <b> -- <files> | git apply --3way`, then edit.
   - Never cherry-pick the checkpoint commits whole: `52a78e4d`, `489a0e93`, `6f06b087`, `b39b5026`, `79779e92`, `6b92e6b6`. They mix features.
   - The listed commits are where the code came from. Later commits often changed the same code, so take the code **as it is at the branch head** for the files involved, minus what other chunks own.
4. **Strip before handing over:**
   - **Notes and planning files:** every root-level `*.md` that is not upstream's, everything under `docs/backend/SYCL-*.md`, and `docs/development/gate-view-fusion-results.md`.
   - **Local scripts, UI files and images:** `scripts/snapdragon/**`, `tools/ui/**`, `tbo.sh`, `img*`.
   - **Diagnostics:** `census.{cpp,hpp}` and every `GGML_SYCL_CENSUS` hook; `pipe-trace.{cpp,hpp}` and its scopes; the fused-op entry tracing (`d11a1cd0`); every `*_DEBUG` / `*_TRACE` / `*_DIAG` variable; LD_PRELOAD probes.
   - **Tuning-sweep variables:** keep one on/off switch per feature, as `GGML_SYCL_<FEATURE>`, documented in `docs/backend/SYCL.md`.
   - **Comments:** drop model- and machine-specific wording such as "qwen4exp" or "3x B60 measured".
5. **Defaults:** a feature may be on by default in the PR only if its benchmark shows a gain and no loss on the generic model.
6. **Tests:** AGENTS.md asks for restraint.
   - New `test-backend-ops` cases only where the chunk adds behaviour no case covers. Take them from the branch; most chunks already have some.
   - List new cases in the results file, so the user can ask maintainers.
7. **MUL_MAT_ID with id == -1.** Upstream (#26631) now allows -1 ("skip this slot, the dst row is 0") and tests it (`skip=1` cases). Any chunk that adds or touches a MoE kernel must handle -1: no weight read, zeros written. The branch's ESIMD MoE kernels in `dmmv.cpp` do not do this yet; see `docs/backend/SYCL-rebase-notes.md`.

## Benchmark protocol (every chunk)

Read "Measuring on this box" in `docs/backend/SYCL-agent-notes.md` first. These rules came from real mistakes.

- **Two builds from the same upstream commit:**
  - **A:** the base (upstream master, plus dependency chunks if any).
  - **B:** the chunk branch.
  - Both: Release, `GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, graph support on, the same oneAPI.
  - md5 the `libggml-sycl.so` each run loads.
- **Correctness:**
  - `test-backend-ops -b SYCL0 -o <op>` for each op the chunk names, on A and on B.
  - B must not add failures. Known failures on master are not regressions, but record them.
  - Also run every `test-backend-ops` case the chunk adds.
- **Kernel speed:** `test-backend-ops perf -b SYCL0 -o <op>`, A vs B. Report the cases the chunk targets.
- **End to end:** `llama-bench`, ABBA order (A, B, B, A), the first run after a load discarded, three rounds where the difference is under 20%.
  - Report pp512 and tg128 at depth 0. Add the depths the chunk names.
  - Models (unsloth GGUFs only):
    - **reference** `unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS`, with the production flags in the agent notes;
    - **plus** the generic model named in the chunk, so reviewers can reproduce the numbers.
- **Text check:** a fixed prompt at temperature 0. The answer must be correct in A and B.
- **Results:** `bench-results/<id>.md` in the scratchpad, never in the branch. Per arm, record:
  - the build hash, the `test-backend-ops` pass/fail counts and any new failures;
  - the perf table and the llama-bench table;
  - VRAM peak from fdinfo when the chunk is about memory.

## Already upstream: do not resubmit

| Branch feature | Upstream |
|---|---|
| Grouped MoE XMX GEMM, combination and precision selection | #29245 |
| Pinned model upload ring | #29608 |
| Q8_0 DMMV ESIMD and MMVQ wide load | #29186 |
| SYCL graph record and replay | #28725 (`7377bada`) |
| Persistent IQ3 layouts | #29107 |
| Reordered Q8_0 MoE weights | #29337 |
| MXFP4 MoE reorder, Q5_K reorder MMVQ and GLU | #29809, #29375 |
| Reorder temp memory bounded to 32 MiB | #31062 (makes the chunked reorder of `GGML_SYCL_MEM_SAVE` redundant) |
| `set_tensor_2d` | #30358 |
| Fast Q8_0 dequant | #30775 (overlaps the reordered Q8_0 dequant of `GGML_SYCL_WIDE_LOADS`) |
| Q4_K/Q5_K wide dequant stores | #29696 |
| Multi-column XMX mat-vec, XMX gate/up GLU | #29864 (overlaps chunk G2) |
| GLU/SCALE type guards | #29889 (same as `bc03e884`) |
| QSA mask fused op `TOP_K_MASK`, QSA selection without kq_mask | #30640, #30965, #31243 (changes the graph the QSA fusions match; see H1) |
| Backend sampling graph static across ubatches | #30223 |

## Open upstream: coordinate, do not duplicate

- **#30156, device-side MoE scheduling** (cwriter). Rebased on 2026-10-26, head `c47031db`. Land it before chunk M1.
- **#29247, RFC for memory-eliding fusions and the allocator callback.** Chunk K3 goes through it.
- **#30226, Q4_K wide row pairing.** Keep it out of G1.
- **#30137, reference-counted server checkpoints.**
- **#28243, MTP.** The branch's MTP changes belong there.
- **#28872, backend sampler reserves.**

Before starting a chunk, search open PRs on ggml-org/llama.cpp for the op or file names. Absence from this list is not proof that nobody is working on it.

## Chunks in landing order

The ids are stable. Each entry gives:
- **Sources:** where the code came from.
- **Files:** which files to touch.
- **Extract:** what to keep and what to leave out.
- **Depends:** the chunks it must stack on.
- **Tests:** `test-backend-ops` ops.
- **Bench:** the generic model and what to report.

### S: small and independent (first)

#### S1 `sycl-barriers`: modern barrier and sub-group load API
- **Sources:** the SYCL hunks of `6b92e6b6`, and only those. That commit also adds scripts, UI files and notes; leave them out.
- **Files:**
  - `cumsum.cpp`, `dequantize.hpp`, `dmmv.cpp`, `dsv4-hc.cpp`, `fattn-common.hpp`, `fattn-tile.hpp`, `fattn-vec.hpp`, `fwht.cpp`, `gated_delta_net.cpp`, `gla.cpp`, `mmq.cpp`, `mmvq.cpp`, `norm.cpp`, `softmax.cpp`, `solve_tri.cpp`, `ssm_conv.cpp`, `topk-moe.cpp`, `topk-radix.{cpp,hpp}`, `vecdotq.hpp`, `wkv.cpp`, `common.hpp`, `ggml-sycl.cpp`, all in `ggml/src/ggml-sycl/`.
  - All of them exist on master. Keep only the hunks that apply to master's version of each file; drop hunks in code that only the branch has.
- **Extract:**
  - Two changes:
    - `item_ct1.barrier(...)` → `sycl::group_barrier(item_ct1.get_group())`, dropping the DPCT1065 comments;
    - `sg.load` → `sycl::ext::oneapi::experimental::group_load`.
  - The second uses an **experimental** extension. Check it builds with the oldest oneAPI that upstream CI uses. If it does not, drop that part.
  - One `global_and_local` barrier becomes a plain group barrier. Confirm the default fence scope still covers the global writes it ordered.
- **Tests:** a broad `test-backend-ops -b SYCL0`: `NORM`, `RMS_NORM`, `SOFT_MAX`, `CUMSUM`, `SSM_CONV`, `GATED_DELTA_NET`, `MUL_MAT`, `FLASH_ATTN_EXT`, `TOP_K`.
- **Bench:** none needed; check for no slowdown on `test-backend-ops perf -o SOFT_MAX` and `-o RMS_NORM`.

#### S2 `concat-flat`: index CONCAT over the whole destination
- **Sources:** `468b999e`.
- **Files:** `concat.cpp`.
- **Extract:** upstream templated the per-quant concat impls in #31008, so re-apply the indexing change onto that structure. Shorten the long comment.
- **Tests:** `CONCAT`, all dims, narrow (4-wide) and wide rows.
- **Bench:** `test-backend-ops perf -o CONCAT`; add the delta-net conv input case upstream added (`{3, 8192, 1, 1}`, 512).

#### S3 `getrows-vec`: vectorized floating GET_ROWS and iq4_nl dequant stores
- **Sources:** `6e8628f3` (get_rows), `5e738bbb` (iq4_nl dequantize stores).
- **Files:** `getrows.cpp`, `convert.cpp`, `dequantize.hpp`.
- **Extract:** rebase onto upstream's #30430 fix for get_rows on reordered weights; keep its reorder handling. Two PRs if a reviewer prefers.
- **Tests:** `GET_ROWS` (offsets, odd widths, several sequences), `MUL_MAT` with IQ4_NL.
- **Bench:** `test-backend-ops perf -o GET_ROWS`; prefill of an IQ4_NL model.

#### S4 `fa-tile-grf`: large register file for the flash-attention tile kernel
- **Sources:** `5425c609`.
- **Files:** `fattn-tile.hpp`, `tests/test-backend-ops.cpp` (one line).
- **Tests:** `FLASH_ATTN_EXT`.
- **Bench:** `test-backend-ops perf -o FLASH_ATTN_EXT`; pp512 and tg128 at depth 0 and 16k on a dense model, for example `unsloth/Qwen3-8B-GGUF:Q4_K_M`.

### K: kernels for ops upstream already has

#### K1 `lightning-indexer`: faster LIGHTNING_INDEXER kernels
- **Sources:** `b4a3df06` (kernels), `0853d231` (XMX variant on by default).
- **Files:** `lightning-indexer.cpp`, `ggml-sycl.cpp`, `common.hpp`, `docs/backend/SYCL.md`, `tests/test-backend-ops.cpp`.
- **Extract:**
  - Exclude `21ba08e6` (pool mask packed to bits, needs H2).
  - Keep the fallback for unsupported types, shapes and devices.
- **Tests:** `LIGHTNING_INDEXER`, including partial tiles, masks, batches and multi-stream strides.
- **Bench:**
  - `test-backend-ops perf -o LIGHTNING_INDEXER`;
  - a DSA model (GLM-5 or DeepSeek V3.2 class, if an unsloth GGUF fits) or the reference model at 16k/64k.

#### K2 `indexer-topk-tiled`: tiled indexer + top-k with compact workspace
- **Sources:** `77c7473e`, `56e8eaee`.
- **Files:** `ggml-sycl.cpp` (the `ggml_sycl_match_indexer_topk` / `indexer_topk_tiled` matcher), `docs/backend/SYCL.md`.
- **Extract:**
  - It matches `LIGHTNING_INDEXER -> TOP_K [-> RESHAPE -> GET_ROWS]`. Re-check the pattern against master's DSA and QSA graphs after #30640, #30965 and #31243; they may now use `TOP_K_MASK` instead.
  - If it no longer fires on master graphs, drop it.
- **Depends:** K1.
- **Tests:** `LIGHTNING_INDEXER`, `TOP_K`. A graph-level case is not possible in test-backend-ops (no graph_optimize), so prove that it fires on a model run.
- **Bench:** the reference model at 64k/104k, compute buffer and pp.

#### K3 `norm-rope`: fused q/k RMS norm + RoPE
- **Sources:** `21a02027`.
- **Files:** `norm-rope.{cpp,hpp}`, `rope-pair.hpp`, `ggml-sycl.cpp`, `common.hpp`, `docs/backend/SYCL.md`, `tests/test-backend-ops.cpp`.
- **Extract:**
  - **Not** the `qsa-kpool.cpp` hunk of that commit.
  - Not the Hadamard/cache-write folding of `ce271fe3`, which is QSA-only (H1).
  - The generic pattern is RMS_NORM(+MUL) → ROPE on q and k, which most Qwen3-style models have.
- **Tests:** the norm-rope fusion cases from the commit, plus `ROPE`, `RMS_NORM`.
- **Bench:** `unsloth/Qwen3-8B-GGUF:Q4_K_M` tg128 and pp512 (the fusion is mostly decode); reference model tg128.

### R: recurrent models (gated delta net, ssm_conv)

Generic model for this group: an unsloth Qwen3-Next or Qwen3.5 GGUF (gated delta net plus short conv), plus the reference model.

#### R1 `ssm-conv-window`: fold the recurrent conv window into ssm_conv
- **Sources:** `c5e8565b`, `58b3bd4c` (on by default).
- **Files:** `ssm_conv.{cpp,hpp}`, `ggml-sycl.cpp`, `common.hpp`, `docs/backend/SYCL.md`, `tests/test-backend-ops.cpp`.
- **Extract:**
  - Specialised for d_conv = 4; keep the fallback.
  - The memory elision of the skipped CONCAT needs C3 (RFC #29247). Submit without it.
- **Tests:** `SSM_CONV` (state rollover, several slots and sequences, unrelated consumers).
- **Bench:** tg128 and pp512 on the generic model.

#### R2 `gdn-state-gather`: gated delta net reads its state from the recurrent cache
- **Sources:** `18c89874`, `d219dd17` (default).
- **Files:** `gated_delta_net.cpp`, `ggml-sycl.cpp`, `common.hpp`, `docs/backend/SYCL.md`, `tests/test-backend-ops.cpp`.
- **Tests:** `GATED_DELTA_NET` (shuffled and duplicated cell ids, rollback snapshots, several sequences).
- **Bench:** generic model tg128, pp512.

#### R3 `gdn-uncached-store`: uncached state stores and wide operand loads
- **Sources:** `f7c39a56`, `8c6391f8`, the `gated_delta_net.cpp` hunks of `d1fa09c8`.
- **Extract:** keep separate from R2 so the cache-policy effect is measured alone. Check the store-cache intrinsic on the supported oneAPI and on both Arc generations, or gate it.
- **Depends:** R2 only if the code overlaps; otherwise independent.
- **Tests:** `GATED_DELTA_NET`.
- **Bench:** as R2.

#### R4 `gdn-qk-norm`: gated delta net applies its q/k L2 norms for short batches
- **Sources:**
  - `d25a4f51`;
  - the intervening-write guard for this fusion from `3f7f7548` (the GDN q/k-norm part only).
- **Files:** `gated_delta_net.cpp`, `fusion.{cpp,hpp}`, `ggml-sycl.cpp`, `common.hpp`, `tests/test-backend-ops.cpp`.
- **Depends:** R2.
- **Tests:** `GATED_DELTA_NET`, `L2_NORM`, plus the fusion cases from the commits.
- **Bench:** generic model tg128.

### F: f32 mat-vec (decode)

#### F1 `f32-esimd-mv`: ESIMD f32 mat-vec for one token
- **Sources:** `90f12168`.
- **Files:** `dmmv.{cpp,hpp}`, the dispatch in `ggml-sycl.cpp`, `fused-gemm.cpp` (6 lines; check whether they are needed), `tests/test-backend-ops.cpp`.
- **Extract:** keep the non-ESIMD fallback; document the device check.
- **Tests:** `MUL_MAT` f32 x f32 at n=1, odd row counts.
- **Bench:**
  - `test-backend-ops perf -o MUL_MAT` (f32, n=1);
  - tg128 on an MoE model with f32 router and gate mat-vecs, such as `unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M`.

#### F2 `f32-mv-siblings`: sibling f32 mat-vecs over one activation in one launch
- **Sources:** `2f97a823`.
- **Depends:** F1.
- **Tests:** the commit's test cases, `MUL_MAT`.
- **Bench:** tg128 on the same model.

#### F3 `f32-mv-tail`: small f32 mat-vec fused with its sigmoid / softplus tail
- **Sources:** `35e6ea88`.
- **Extract:** port only the graph dependency handling it needs; not the allocator RFC. Include the intervening-write guard from `3f7f7548` if it covers this fusion.
- **Depends:** F1.
- **Tests:** the commit's cases (intervening views, extra consumers), `UNARY`.
- **Bench:** tg128, reference model, and an MoE with sigmoid routing.

### A: attention

#### A1 `mkl-fa-fixes`: oneMKL / oneDNN flash attention correctness
- **Sources:**
  - **Fully masked rows:** a fully masked row gives 0 instead of NaN in the normalize kernels (`KQ_sum > 0 ? 1/x : 0`).
  - **Strides:** oneDNN V strides come from the view's `nb` (`939db1a9`, the stride hunks only).
  - **KV stream:** `86c340ce` reads the KV stream of each batch.
- **Files:** `fattn-mkl.cpp`, `fattn-onednn.cpp`, `tests/test-backend-ops.cpp`.
- **Extract:**
  - Check each fix still applies after upstream's #30684 (explicit dependencies) and #30286 (head-size mismatch); drop the ones master already fixed.
  - Leave out the SoA staging (`ggml_sycl_kv_soa_to_fp16`); that belongs to the held SoA KV layout.
- **Tests:** `FLASH_ATTN_EXT` default and with `GGML_SYCL_FA_ONEDNN=0`, including `hsk=256 kv=1025 q8_0`.
- **Bench:** none needed (correctness).

#### A2 `mkl-fa-perf`: one-pass softmax, in-place VKQ, query tile, coalesced normalize
- **Sources:**
  - `21ff6381`: the `fattn-mkl.cpp` part only, not `fattn-qsa.cpp` or the notes;
  - `939db1a9`: the normalize part;
  - `f4acc2cd`: the 24 MiB tile.
- **Files:** `fattn-mkl.cpp`, `docs/backend/SYCL.md`.
- **Extract:**
  - Start with dense masks and canonical KV.
  - Drop the packed-mask reading (H2) and the online q8 KV (held with SoA KV).
  - Rebase onto master's #30684 dependencies, which master needs for graph capture. Keep them instead of the branch's in-order assumption.
  - Drop `GGML_SYCL_MKL_FA_DRAIN` and the sweep variables.
- **Depends:** A1.
- **Tests:** `FLASH_ATTN_EXT` (default and `GGML_SYCL_FA_ONEDNN=0`).
- **Bench:**
  - `test-backend-ops perf -o FLASH_ATTN_EXT` (cases from `fcb70088`);
  - pp at 16k/64k on a dense model with long context, and the reference model at 13k/54k/104k.

#### A3 `fa-scratch-bound`: bound flash-attention scratch over every batch and KV length
- **Sources:** `a97f556d`, without its `fattn-sparse.*` hunks if A3 goes before H1.
- **Extract:** may need `get_max_alloc_size` (`e015b2c1`), which touches every backend. Show the under-allocation on a small reproducer first, and discuss the interface in an issue.
- **Hold** unless the reproducer exists.

### C: core ggml (backend independent)

#### C1 `sched-input-upload`: upload a split's host inputs in one transfer
- **Sources:** `07d29a75`.
- **Files:**
  - `ggml-backend.cpp`, `ggml-backend-impl.h`;
  - the one-line interface additions in every backend (`ggml-cann`, `ggml-cpu/*`, `ggml-cuda`, `ggml-et`, `ggml-hexagon`, ...);
  - the SYCL implementation.
- **Extract:**
  - It adds a backend interface slot. Check upstream's current interface for a matching slot first.
  - Every backend initializer must list the slot. Build CPU and every backend the container can build.
- **Tests:** `test-backend-ops` on CPU, and `test-alloc`.
- **Bench:**
  - Pipeline-parallel decode on 2+ SYCL cards: tg128 with `--split-mode layer`.
  - One card for no regression.
- **Note:** open an issue first. It touches every backend.

#### C2 `kv-rot-persistent`: keep the KV rotation matrices in backend buffers
- **Sources:** `706b2e41`.
- **Files:** `src/llama-kv-cache.{cpp,h}`, `src/llama-graph.{cpp,h}`, `src/models/{dflash,minimax-m3,qwen4exp}.cpp`.
- **Extract:**
  - Master moved `no_alloc` out of `hparams` and has `llama_memory_bufs`. The branch's merge already adapted `memory_breakdown()` and the no-alloc path; take that state.
- **Tests:**
  - CPU and SYCL end to end with a model that rotates KV (`attn_rot_hadamard`): context shift, state save and load round trip;
  - no-alloc memory estimate (`llama-cli --fit` or the memory breakdown log).
- **Bench:** tg128 on a rotating model; expect a small decode gain (no per-graph upload).

#### C3 `alloc-absorbs`: allocator fusion plan (absorbed nodes)
- **Sources:**
  - `b60cdb0c` (absorbed in-place view lifetimes);
  - `2e66a97f` (release cascade);
  - `7e66e840` (binary search);
  - the `fusion_absorbs` interface parts of the checkpoints. Find them with `git log -S fusion_absorbs`.
- **Extract:** do not open a new PR. Bring it to the open RFC #29247 as a follow-up comment, or as commits the RFC author can take. The user decides.
- **Unblocks:** the memory parts of R1, M3 and H1.

### M: MoE (after #30156)

#### M1 `moe-mv-routes`: expert-major route ordering, parallel scan, routed-expert grid
- **Sources:**
  - `2812753e`, `3b214357`, `c1a76c1f`, `8e8e15cf`;
  - the MoE mat-vec parts of `baead56b`.
- **Files:** `mmvq.{cpp,hpp}`, `ggml-sycl.cpp`, `common.hpp`.
- **Extract:**
  - Take the branch head's version, which already handles id == -1 in the plain and reordered kernels, and zeroes skipped rows after the route-ordered launch (`ggml_sycl_mmid_zero_skipped`).
  - Rebase onto #30156 if it has landed; the shape gates interact.
- **Depends:** #30156.
- **Tests:** `MUL_MAT_ID`, all types, including `skip=1` and n = 1..8 tokens.
- **Bench:** tg128 and MTP-like batches (n = 2..8) on `unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M` and `:Q8_0`, plus the reference model.

#### M2 `moe-esimd`: ESIMD routed-expert mat-vecs at decode
- **Sources:**
  - `6d8a7401` (IQ4_NL, IQ3_S + SWIGLU);
  - `c5df33aa`, `ea103cc8` (int8 dp4a IQ3_S gate/up);
  - `d1141ca9`, `94923a64`;
  - `afecacd6` (Q8_0);
  - `681b60cc`, and the dmmv part of `27c7f106` (persistent threads).
- **Files:** `dmmv.{cpp,hpp}`, `ggml-sycl.cpp`, `common.hpp`, `tests/test-backend-ops.cpp`.
- **Extract:**
  - **First fix id == -1 in every kernel** (rule 7): `if (i02 < 0)` → write zeros for the route's rows, read no weight.
  - Split it into separate PRs:
    - Q8_0 experts vs the IQ types;
    - the row layout vs persistent threads.
  - It depends on the IQ reorder layouts that master has had since #29107. Use master's layout and drop the branch's own copies where they are the same.
- **Depends:** M1.
- **Tests:** `MUL_MAT_ID` (IQ4_NL, IQ3_S, Q8_0, `skip=1`, repeated routing), `MUL_MAT_ID` fusion cases.
- **Bench:** tg128 on the reference model (IQ4_XS / IQ3_S experts), and an unsloth Q8_0 MoE.

#### M3 `moe-reduce`: MoE weighted sum without materializing the expert outputs
- **Sources:**
  - `0329d87c` (collapse, mul+add, `graph_optimize`);
  - `22f86aa2`, `987268a2`, `b902e68c`, `ebe72ba9`.
- **Files:** `moe-reduce.{cpp,hpp}`, `binbcast.{cpp,hpp}`, `fusion.cpp`, `fused-gemm.{cpp,hpp}`, `ggml-sycl.cpp`, `tests/test-backend-ops.cpp`.
- **Extract:**
  - The hc-ffn-tail and dsv4-hc hunks of `22f86aa2` belong to H3.
  - "No buffer for unwritten nodes" (`987268a2`) needs C3. Without C3, submit the compute part only.
- **Depends:** M1. C3 for the memory part.
- **Tests:** `MUL_MAT_ID`, `ADD`, `MUL`, plus the commit's fusion cases.
- **Bench:** tg128 and pp512, generic MoE and reference model; compute buffer size.

#### M4 `moe-tiled-swiglu`: tiled grouped MoE GEMM with fused SwiGLU packing
- **Sources:** `6a325a51`, `72613d37` (paired gate/up packing, 256 GRF), `c7cd31b1` (reduction buffer reuse).
- **Files:** `fused-gemm.{cpp,hpp}`, `ggml-sycl.cpp`.
- **Extract:** builds on master's grouped XMX GEMM (#29245) and on #30156's device schedule. Check what of the scheduling it assumes.
- **Depends:** #30156.
- **Tests:** `MUL_MAT_ID` (all types the grouped GEMM takes, prefill shapes), the MoE fusion cases.
- **Bench:** pp512 and pp2048, generic MoE and reference model.

### G: dense decode mat-vec (benchmark against master first)

#### G1 `q8-esimd-dense`: dense Q8_0 ESIMD
- **Sources:**
  - `4c2b1c3b` (dense Q8_0 SWIGLU pair at one token);
  - `f8ee647e` (rows and threads by row length);
  - `759a687d` (fused mat-vec GLU for Q8_0, 1-2 columns);
  - `f601e372` (Q8_0 verification tails).
- **Extract:** master already has the Q8_0 DMMV ESIMD (#29186). Submit only what beats master: A/B the kernel against master's Q8_0 ESIMD and XMX paths first.
- **Tests:** `MUL_MAT` q8_0, n = 1..8, odd rows; `MUL_MAT_VEC_FUSION`.
- **Bench:** `unsloth/Qwen3-8B-GGUF:Q8_0` tg128 and n = 2..8 batches.

#### G2 `esimd-ncols`: multi-column ESIMD mat-vec (Q8_0 / Q6_K, 2..8 columns), Q8_0 GLU to 8 columns, flat batched mat-vec
- **Sources:** `88e630d7`, `baead56b` (the dense parts).
- **Extract:**
  - Master's XMX multi-column path (#29864) covers the same column range for Q2_K..Q6_K and Q8_0 on BMG/LNL.
  - **Benchmark first:** on the mega branch, `GGML_SYCL_ENABLE_XMX_ESIMD=0` vs `=1`, and `GGML_SYCL_ENABLE_ESIMD` with and without the ncols bit, for n = 1, 2, 4, 8, 16 on Q4_K_M, Q6_K and Q8_0 models.
  - Submit only the cases where the ESIMD kernel wins: for example Arc A-series, which master's XMX path excludes, or the column counts where XMX loses.
- **Tests:** `MUL_MAT` n = 1..16, `MUL_MAT_VEC_FUSION`.
- **Bench:** as above; record the full matrix.

#### G3 `q8-siblings-gated-norm`: fuse sibling Q8_0 mat-vecs and the gated RMS norm
- **Sources:** `a19017fb`, the matching guard of `3f7f7548`.
- **Depends:** G1 if they share the kernel.
- **Tests:** the commit's cases, `RMS_NORM`, `MUL_MAT`.
- **Bench:** tg128 on the reference model and a Qwen3-Next-class model.

#### G4 `gate-view-unary`: fuse gate view materialization into the unary multiply
- **Sources:** `209a1c0a`, minus `docs/development/gate-view-fusion-results.md`. The results go into the user's PR text.
- **Tests:** the commit's cases, `GLU`, `UNARY`.
- **Bench:** tg128 on a model whose graph has a GLU over a gate view; name the model the commit targets.

### P: multi-device and memory

#### P1 `xdev-copy-order`: order cross-device copies against the source queue
- **Sources:** `f3965159`, `69d09fd0`.
- **Files:** `ggml-sycl.cpp`, `common.hpp`, `fattn-onednn.cpp`, `fused-gemm.cpp`.
- **Extract:** compare with master's #29889 (split buffer and mul_mat memory fixes) first; keep only what master lacks. Correctness by construction, so the PR text argues from the code.
- **Tests:** `test-backend-ops` on two cards if possible; end to end with `--split-mode layer` on 2-3 cards.
- **Bench:** pipeline-mode pp and tg, 2-3 cards; no regression.

#### P2 `l0v2-upload-queue`: separate upload queue for the Level Zero v2 adapter
- **Sources:** `a9d1f089`, plus the growing-input-ring hunks of `f0d1e39f` if they are needed.
- **Extract:** no pipe tracing. The branch's merge of 2026-10-27 kept the branch's growing ring and added a `set_tensor_2d` on it. Master's `set_tensor_2d` uses the slot ring, so pick one design for the PR.
- **Tests:** model load and pipeline-parallel decode on v1 and v2 adapters (`UR_L0_V2` / the adapter env).
- **Bench:** load time and pipeline pp on 2-3 cards, v1 and v2.

#### P3 `mem-save`: lower peak device memory
- **Sources:** `afb8f46e`, minus the chunked reorder (master has #31062).
- **Extract:** what is left is the pool release before growth (off while graphs are on) and the exact grouped-GEMM packed-B size. The packed-B part needs M4 or #30156, whichever owns the device-scheduled GEMM.
- **Tests:** `MUL_MAT_ID`, `MUL_MAT`.
- **Bench:** fdinfo VRAM peak at 64k/131k on the reference model; pp and tg for no regression.

#### P4 `graph-followups`: follow-ups to master's graph record/replay
- **Sources:**
  - **From the 2026-10-27 merge, as kept on the branch:** the per-node blocker log, the device-local check, the sparse-attention exemption (`8095bf06`), and the MoE consumer check cache (`7c282026`, `2cec0200`).
  - **Earlier commits:** a graph per device (`282576af`), MUL_MAT_ID on device in graphs (`0028ba67`).
- **Extract:**
  - Several of these are already in master's #28725 in another form. Diff against master and keep only what master lacks.
  - Coordinate with #28725's author, and keep each follow-up small.
  - The old list recorded that graph replay cost 22-24% decode on the reference workload. Re-measure with `GGML_SYCL_ENABLE_GRAPH=1` before arguing for any of this.
- **Tests:** `test-backend-ops` with `GGML_SYCL_ENABLE_GRAPH=1`.
- **Bench:** tg128 with graphs on and off.

## Hold: rework or discuss first

- **H1, QSA fusions, sparse and QSA decode attention:**
  - Commits:
    - `qsa-kpool`, `qsa-live`, `qsa-mask`, `fattn-qsa*`, `fattn-sparse`;
    - `0cc7f1c3`, `ca0792f7`, `67f847f0`, `ce271fe3`, `24e610d2`, `3cd36996`;
    - `73a5354c`, `9a3ca8f6`, `cb398646`, `f47e242f`.
  - Master rebuilt the QSA graph (#30640 `TOP_K_MASK`, #30965, #31243). First check on the reference model which of these fusions still fire on master's graph.
  - The natural upstream pieces are now:
    - a SYCL `TOP_K_MASK` kernel, a new chunk following master's CPU, CUDA and Metal versions;
    - the QSA decode attention on the matrix engines (`0cc7f1c3`), adapted to master's graph.
  - Decide after that check.
- **H2, packed 1-bit KQ mask and compact allocation:**
  - Commits: `8ed529ac`, `d889dcf6`, `05d1f866`, `21ba08e6`, `kq-mask-bits.*`, `ggml_compacted_nbytes`.
  - It needs a core interface change (compact `get_alloc_size`) plus every taught reader.
  - Discuss the interface in an issue first. Then: core PR, SYCL readers, compaction.
- **H3, hyper-connection fusions:**
  - Commits:
    - `5ee275c6`, `5836cf2b`, `88a3fcc0`, `80a2c4f7`, `288f2b21`, `a7295a60`, `294cea5f`, `529b38db`;
    - `dsv4-hc.cpp`, `hc-ffn-tail.*`;
    - the hc hunks of `d1fa09c8` and `22f86aa2`.
  - Model-specific (DSV4_HC ops). Submit after the generic fusions (K3, G3, M3) have set the pattern.
  - Split three ways:
    - the gate up-projection fusion;
    - HC_POST + next RMS norm;
    - the ESIMD kernels.
- **H4, SoA q8_0 KV cache:** `bd083e91`, `9f793197`, `f1731ffa`, `2ed33b1e`, `kv-soa.*`, the `llama-kv-cache.cpp` assert. About +10% prefill and -2% decode, off by default, head dim 256 only, and it touches llama code. Hold unless the decode cost is fixed.
- **H5, scheduler and server features:** open an issue first.
  - `bf0d3e6e` `--minimize-multidevice-seam`;
  - `408ec5bb` `--decode-stagger`;
  - `9ce5ed4e` k-pool per-sequence scoring;
  - `3e07a83a`, `ca82e033` qwen4exp graph changes. These also need rebasing onto #30965/#31243.
- **H6, MTP and draft scheduling:** `12fc0f86`, `643ad8f1`, `9f4f49f1`, `454535f1`, `44b67315`, `5617bc39`, `d5f51e42`, `ab43a9fe`. Belongs to #28243; offer it to that PR's author.
- **H7, scheduler re-plan and copy-entry changes:** `025f1217`, `501d4c60`, `e015b2c1`. Small core changes. Each needs a reproducer showing the bug or cost on master before it can be argued.
- **H8, backend sampler reserves:** `687c8585`, `fce424f8`. Compare with master's #30223 and open #28872 first.
- **H9, server prompt-cache state:** `cf0aa031`. Coordinate with #30137.

## Never upstream

- Census, pipe trace, fused-op tracing, all probes and debug variables.
- `GGML_SCHED_DUMP` (`972e8eef`) and `GGML_ALLOC_PEAK_DEBUG` naming (`ea63eb75`), unless the user wants a separate small debug PR.
- Every analysis, notes and plan file.
- `docs/backend/SYCL-int8-xmx-plan.md`.
- Scripts, UI files and images from `6b92e6b6`.

## Per-branch checklist

1. `git switch -c pr/<id>-<name> upstream/master`, plus dependency branches. Bring the hunks over and strip per rule 4.
2. Build CPU and SYCL with zero new warnings in the touched files. Build every other backend whose files the chunk touches (C1).
3. Run the tests listed for the chunk on A and B.
4. Run the benchmark protocol. Write `bench-results/<id>.md` in the scratchpad.
5. Hand the branch and the results file to the user. Do not push without approval for that push; never open a PR.
