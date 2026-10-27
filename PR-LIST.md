# Splitting `performance_uplift_mega_branch` into upstream PRs

Current status: see [2026-10-10 master comparison](mega-master-review-20261010.md). The historical list below predates several upstream merges and open PRs; do not use its counts or landing order without that review.

The branch is 61 commits on top of master `0f8a414b7` (the fork point). Several of those commits are
checkpoints that mix unrelated features. This file lists the PRs to extract, in landing order, and
tells an agent exactly what goes into each and what must stay out.

## Rules for whoever extracts these (read first)

1. **AGENTS.md applies in full.**
   - An agent never writes PR descriptions, comments or review replies.
   - An agent never opens a PR and never pushes without explicit per-action approval from the user.
   - Commits made on the user's behalf carry `Assisted-by: <assistant name>`, never `Co-authored-by:`.
   - Large changes must be discussed with maintainers first. The core-ggml PRs (tier 0) start as an issue/discussion, not as code.
2. **Build each PR from current upstream master**, not from the mega branch.
   - Create one branch per PR (`pr/NN-short-name`) from `origin/master`.
   - Bring the change over by hand or by hunk (`git checkout -p <commit> -- <file>`, `git apply --3way` of a filtered diff).
   - Do not cherry-pick the checkpoint commits whole.
3. **A PR's branch must build and pass on its own**, stacked only on the PRs it lists as dependencies.
   - Build the SYCL backend (see `docs/backend/SYCL.md`).
   - Run the `test-backend-ops` suites named in the PR entry (`-b SYCL0`, one card via `ONEAPI_DEVICE_SELECTOR`).
   - Run the end-to-end text check.
4. **Re-measure every PR standalone** against its own base, interleaved (ABBA), first run after a load discarded.
   - The numbers below were measured stacked on the mega branch; they are what to expect, not what to quote.
   - Use the traps list in `docs/backend/SYCL-agent-notes.md`: bimodal long-context prefill, cold first run, fdinfo for VRAM, text checks.
5. **Strip before submitting:**
   - Agent and analysis files: `docs/backend/SYCL-agent-notes.md`, `BRANCH_CLEANUP.md`, `COMPUTE_BUFFER_ANALYSIS.md`, this file, anything in the repo root that is not upstream.
   - Diagnostics: `ggml/src/ggml-sycl/census.{cpp,hpp}` and every `GGML_SYCL_CENSUS` hook.
   - Debug and trace environment variables:
     - `*_DEBUG`, `*_TRACE`, `*_DIAG`;
     - `GGML_SYCL_MKL_FA_DEBUG` / `_DIAG`;
     - `GGML_SYCL_QSA_MASK_TRACE`;
     - `GGML_SYCL_GG_TRACE`;
     - `GGML_SYCL_KQ_MASK_DEBUG`;
     - the `MMID_SCHED` `DIAG_*` bits.
   - LD_PRELOAD probe hooks: the weak `ggml_sycl_kq_mask_tail_probe`.
   - Tuning knobs a reviewer will not want. Keep one on/off switch per feature; drop sweep-only variables such as `GGML_SYCL_MKL_FA_Q_TILE` and `GGML_SYCL_REORDER_CHUNK_KIB` unless the PR argues for them.
   - Model-specific wording in comments ("qwen4exp", "3x B60 measured ...") belongs in the PR text the user writes, not in code comments. Keep comments short and technical.
6. **Tests:** AGENTS.md forbids new test files and asks for restraint.
   - Add `test-backend-ops` cases only where a PR changes behaviour that no existing case covers.
   - List them for the user, who asks the maintainers.
   - Existing suites that must pass for SYCL PRs:
     - `MUL_MAT`, `MUL_MAT_ID`;
     - `FLASH_ATTN_EXT`;
     - plus the op-specific ones below.
   - Known pre-existing failures, also on master, are not regressions:
     - 3x `MUL_MAT_ID` q8_0 `amax=100000`;
     - `FLASH_ATTN_EXT` f16 `kv=8192 nb=67`;
     - flaky `CPY` to quant types.
7. **Test models:** only unsloth models for PR testing. The reference model is `unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS`. Where a PR is generic, also check a small, common unsloth GGUF so reviewers can reproduce it.

## Landing order at a glance

| tier | PRs | why this order |
|---|---|---|
| 0 | C1-C3: core ggml | the SYCL memory PRs depend on C1/C2; discuss first |
| 1 | F1-F2: fixes | small, independent, easy to review |
| 2 | P1-P10: SYCL-only performance | independent of each other except where noted |
| 3 | P11: packed/compact KQ mask | needs C2 |
| 4 | P12-P13: qwen4exp-specific | needs P10 and P11 |
| — | D1-D6: drop or hold | not submitted |

## Tier 0: core ggml (issue/discussion first, then small PRs)

### C1: ggml-alloc: let a backend report nodes it fuses away
- **What:** backend interface `fusion_absorbs(backend, cgraph, node_idx)` in `ggml-backend-impl.h`; `ggml_gallocr_set_fusion_plan()` in `ggml-alloc.h/.c`; the scheduler wiring in `ggml-backend.cpp`, so gallocr never reserves the dst of a node the backend will not write.
- **Also:** the absorbed in-place view lifetime fix, and the `graph_optimize` `add_alloc_dep` plumbing.
  - Check whether upstream master already has `ggml_backend_graph_optimize_params` / `add_alloc_dep`; include only what is missing.
- **Source:** `e58475597` (the core part only), `f6bd42a12`, the relevant hunks of `ggml/src/ggml-backend.cpp`.
- **Headline:** compute buffer about -640 MiB per card on qwen4exp. The SYCL side that implements `fusion_absorbs` ships with the fusion PRs (P7, P13).
- **Risk:** it changes a contract in the allocator. Reviewers will want the "must be exact, must not depend on mutable state" rule spelled out, which is already in the header comment.

### C2: ggml-backend: allow a buffer type to store a tensor compactly
- **What:**
  - `ggml_backend_buft_get_alloc_size` may return less than `ggml_nbytes` for a tensor its buffer type stores compactly. The debug-only `assert(size >= ggml_nbytes)` becomes a comment.
  - `ggml_compacted_nbytes(tensor)` in `ggml-backend.h` returns `min(ggml_nbytes, get_alloc_size)` for non-view tensors with a buffer.
- **Source:** the `ggml/include/ggml-backend.h` and `ggml/src/ggml-backend.cpp` hunks of `c6d900ce1`.
- **Needed by:** P11.

### C3: ggml-alloc: fix the dead view-inplace branch
- **What:** in `ggml_gallocr_allocate_node()`, the branch that lets an in-place op reuse a view parent's buffer is unreachable, and also broken if reached.
  - Ownership is tested on the view instead of `view_src`.
  - It reads `p_hn->addr`, which is never set for a view.
  - The fix: test `ggml_impl_is_view(parent) ? parent->view_src : parent`, inherit `view_src_hn`'s buffer_id and addr, and guard on `view_offs == 0` plus equal nbytes.
- **Source:** not on the branch; the design is in the memory note `ggml-alloc-view-inplace-dead-code`. Needs writing and testing from scratch.
- **Headline:** -128 MiB per card on qwen4exp. Backend-independent.

## Tier 1: fixes

### F1: sycl: flash-attention correctness fixes
- **What:**
  - A fully masked row gives 0 instead of NaN in the oneMKL normalize (`KQ_sum > 0 ? 1/x : 0`, in both normalize kernels).
  - oneDNN and oneMKL staging convert a SoA KV cache with `ggml_sycl_kv_soa_to_fp16`. Only if the SoA layout (D1) is submitted; otherwise drop this part.
  - oneDNN staging takes V strides from the view's `nb` instead of deriving them from `ne`. This fixes `FLASH_ATTN_EXT hsk=256 kv=1025 q8_0`.
- **Source:** `8b77aadd6` (the `fattn-mkl.cpp` NaN hunk), `26b87e95b` (the stride and SoA hunks), the NaN guard in the softmax commit's `mkl_fa_normalize_group`.
- **Tests:** `FLASH_ATTN_EXT` (default, `GGML_SYCL_FA_ONEDNN=0`).

### F2: sycl: order cross-device copies against the source queue
- **What:**
  - `ggml_backend_sycl_cpy_tensor_async` queued the peer copy on the destination queue with nothing ordering the source after it, so pipeline mode can overwrite an activation mid-read.
  - The copy now stages through a ring on the source device (`GGML_SYCL_ASYNC_COPY` bit 2, `GGML_SYCL_COPY_RING_DEPTH`). With the bit off, a source-queue barrier is used instead.
  - The oneDNN FA multi-GPU host `wait_and_throw()` is removed.
  - Consider folding in `507a8c9a9` (cross-split work ordered on the device instead of blocking the host) if it is not already upstream.
- **Source:** `cc1a1e3d3`, `507a8c9a9`.
- **Evidence:** correctness by construction. The race did not show in KL divergence and speed is neutral, so the PR text must argue from the code.

## Tier 2: SYCL-only performance

### P1: sycl: lower peak device memory
- **What:** `GGML_SYCL_MEM_SAVE` bitset:
  - chunked weight reorder through a temp of at most 32 MiB;
  - pool release of outdated buffers before growing (off while SYCL graphs are on);
  - exact grouped-GEMM packed-B size.
- **Source:** `1cb29c8e4`. The packed-B hunk needs P3.
- **Headline:** fdinfo peak -553/-553/-970 MiB at 131k.

### P2: sycl: hoist host-side pointer reads out of kernels
- **What:** tensor `->data` is read on the host and captured, instead of being read inside kernels.
- **Source:** inside checkpoint `e54eed3d4`. Find the hunks by searching kernels that dereference `tensor->data` on the device.
- **Headline:** +29-32% decode. Also read the "fusions must absorb all of [i, j)" rule in the notes; it came from this work.

### P3: sycl: grouped MoE GEMM on XMX with fused IQ dequantize
- **What:**
  - One grouped launch for `MUL_MAT_ID` at prefill instead of one GEMM per expert.
  - A tile schedule, B packed per tile, IQ formats dequantized into the XMX f16 GEMM.
  - Pool allocation bucketed; operands read through the route map with no staging copies.
- **Source:** `2a3c84da4`, `0b78c21cf`, `950f1c4aa`, `3a74e3504` (a merge; take the content), `5e431a8a5`, `8b2efe1f2`, and the grouped-scheduling parts of `573971b63` / `e54eed3d4`.
- **Headline:** +33% prefill.
- **Keep:** FG_BN=32; 128 corrupts output.
- **Before submitting:** understand the intermittent order-dependent `MUL_MAT_ID` q4_1 failure. It is likely a stale read exposed by this path's pool layout; NaN-poison the pool to find it.
- **Tests:** `MUL_MAT_ID` (all types), `MUL_MAT`.

### P4: sycl: MoE weight reorder into per-expert SoA, on by default
- **What:**
  - The reorder described as a layout.
  - One bitmask `GGML_SYCL_REORDER_TYPES` for IQ3_S, IQ4_NL and q8_0, triggered from the grouped GEMM too.
  - Reordered IQ3 MoE paths; all types on by default.
- **Source:** `8c63615d8`, `2c91b781f`, `68cbe8c44`, `c13873471`, `030b607fe`, `77181983a`, `038314ae1`.
- **Headline:** IQ3_S +12.8% decode; q8_0 +2.5% tg and +1.0% pp.
- **Depends on:** P3 for the grouped-GEMM trigger.

### P5: sycl: MoE mat-vec for decode
- **What:** expert-major route ordering, parallel scan and a routed-expert grid, a wider row group with double-buffered tiles, and the documented 8-token cap.
- **Source:** `ce17a4271`, `40eec6668`, `405ce2b2a`, `0f8b6e4ac`. `02083050c` and `f723421ce` cancel out; skip both.
- **Headline:** decode. Re-measure; no standalone number was recorded.

### P6: sycl: fused GLU and multi-column ESIMD mat-vec
- **What:**
  - The q8_0 fused mat-vec GLU for 1-2 columns.
  - MoE gate/up mat-vecs folded into their GLU.
  - Multi-column ESIMD mat-vec, the q8_0 GLU up to 8 columns, the flat batched mat-vec (`GGML_SYCL_ENABLE_ESIMD=3`).
- **Source:** `50b8f5d4b`, `2b0092416` (the GLU part), `1a6c4dc8e`.
- **Headline:** decode kernels. The multi-column paths also serve MTP / speculative batches.

### P7: sycl: fusions for the decode graph, with fusion_absorbs
- **What:**
  - The hyper-connection elementwise chain and combine (`5a6d5a3c3`, `16df74beb`).
  - The MoE weighted-sum collapse and mul+add fusion, `graph_optimize` (`63ec37946`).
  - The fusion selection bitmask (`2b0092416`, the bitmask part).
  - The fusion parts of `573971b63` (ELEMENTWISE, MUL_ADD, MOE_REDUCE, MOE_GLU_ID, NORM_SCALE, GLU_NCOLS, FLAT_BATCH).
  - The SYCL `fusion_absorbs` implementation.
  - Skip `2796fb531`: it removes an earlier fusion.
- **Trim:** keep only the fusion bits with a measured gain; the default set is `GGML_SYCL_FUSE_DEFAULT`.
- **Headline:** hc motif -385 dispatches per token, +6.3% decode; compute buffer via C1.
- **Depends on:** C1 for the absorbs part. Without C1, submit the fusions and leave the absorbs hook out.

### P8: sycl: wide aligned loads
- **What:**
  - The grouped-GEMM B pack with 32 B loads (`MMID_SCHED` bit 18 to disable).
  - `gated_delta_net` block loads and prefetch.
  - `hc_pre` / `hc_post` float4.
  - The contiguous f32->f16 convert and reordered q8_0 dequant (`GGML_SYCL_WIDE_LOADS`).
  - The vectorized get_rows (`bd5de6e33`) and iq4_nl dequantize stores (`8217c1bbd`).
  - The concat indexing (`6a831c08d`) fits here too, or as a tiny separate PR.
- **Source:** `bcd2b66d7`, `bd5de6e33`, `8217c1bbd`, `6a831c08d`.
- **Headline:** pp2048 +14.6%; prefill energy about -10%.
- **Depends on:** P3 for the B pack.
- **Tests:** `MUL_MAT_ID`, `MUL_MAT`, `DSV4_HC_PRE`, `DSV4_HC_POST`, `GATED_DELTA_NET`, `GET_ROWS`, `CONCAT`.

### P9: sycl: stage bulk uploads through a pinned ring
- **What:** `set_tensor` copies through a reused pinned host ring (4 x 8 MiB) instead of a malloc bounce per call. `ff2fbf1fb` and its revert `03cc70f0b` cancel out; skip both.
- **Source:** `f97a2cbfe`.
- **Headline:** load and upload time. Re-measure model load.

### P10: sycl: oneMKL flash attention performance
- **What:**
  - The kernel reads a packed mask selection map (`702bc5472`, only the part independent of P11).
  - Coalesced per-KV-head normalize (`26b87e95b`).
  - One-pass softmax with scores in registers, VKQ GEMM accumulating in place, L2-sized query tile (the softmax commit, bit 4 of `GGML_SYCL_ENABLE_MKL_FA`).
- **Headline:** FA -17 to -23% per call at every depth; +7.5% prefill at 123k; prefill energy up to -7.8%.
- **Depends on:** F1.
- **Tests:** `FLASH_ATTN_EXT` (default, `GGML_SYCL_FA_ONEDNN=0`, old kernels via the bit).

## Tier 3

### P11: sycl: pack the causal KQ mask to one bit per cell, and allocate it compactly
- **What:**
  - Pack on upload; unpack on read-back.
  - Taught readers: ADD, the f32 cast, the cast+add fusion, the QSA top-k fusion, FA tile and oneMKL.
  - Compact allocation, approved per mask name in `graph_optimize` and refused for ALiBi or untaught readers. A non-binary upload vetoes it for good.
  - Card-to-card copies of packed masks; `get_tensor` reads only packed bytes.
- **Source:** `503ee97df`, and `c6d900ce1` minus the core hunks (C2), minus `census.*`, minus the tail mode (`KQ_MASK_BITS` bit 1 and the tail pool) and its probe hook.
- **Test fix:** the `test_topk_qsa` causal-init overflow is fixed in `c6d900ce1`.
- **Headline:** fdinfo peak -240..-272 MiB per card; -930..-960 MiB in pipeline mode, which makes pipeline mode fit at 131k.
- **Depends on:** C2, P10.
- **Tests:** `QSA_MASK_ADD`, `TOPK_QSA` (need maintainer approval), `FLASH_ATTN_EXT`. `test-backend-ops` cannot exercise the `graph_optimize` approval, so prove compaction on a real server (no veto warning, lower reserve).

## Tier 4: qwen4exp-specific

### P12: qwen4exp: build the QSA set_rows zeros as one repeated row
- **Source:** `7018932e1` (model code, 12 lines).
- **Headline:** -32 MiB reserve per card.
- Tiny and independent; could also go first.

### P13: sycl: QSA fusions and sparse flash attention for qwen4exp
- **What:**
  - The QSA indexer/top-k/score/gather fusions and their allocation spans (`76b080e39`, `6abcfac36`, `d8a0361cf`).
  - The mode 3 sparse FA: union tiles, bitmap path, dense scratch for decode.
  - Packed-mask support in the fusions; the union path taking long nodes regardless of oneDNN.
  - Default on (`GGML_SYCL_FUSE_QSA_FA_MASK=3`).
- **Source:** `8b77aadd6` (minus the notes file and the F1 NaN hunk), `d9c36b905`, the `fattn-qsa.cpp` part of the softmax commit, the QSA parts of `573971b63`, `e54eed3d4`, `e58475597`, `28ecb2c22`.
- **Headline:** prefill +12.8% at ~48k, +28.5% at ~92k (3 interleaved rounds); compute buffer -75 MiB per card; equal at short context.
- **Depends on:** C1, P10, P11.
- **Tests:** `QSA_SPARSE_FA`, `QSA_MASK_ADD`, `TOPK_QSA` (need maintainer approval), `FLASH_ATTN_EXT`.
- **Split further if reviewers ask:** P13a (fusions), P13b (sparse FA).

## Drop or hold (do not submit as is)

- **D1: KV SoA layout** (`01896dd6b`, `775854149`, `65825d615`, the `src/llama-kv-cache.cpp` assert).
  - +10% prefill, -2% decode, default off; fits head dim 256 only.
  - Hold unless the decode cost is fixed. It also touches llama code.
- **D2: SYCL graph record/replay** (`140402636`, `36aa5e3f9`, `4376331ed`, `1e020748a`).
  - Capture never engages on this workload, and turning it on costs about 22-24% decode.
  - Check authorship first: two of these commits lack the `sycl:` prefix and may come from another contributor's branch.
- **D3: mask tail reuse** (`KQ_MASK_BITS` bit 1): superseded by compact allocation.
- **D4: diagnostics:** the census, the fused-op entry-point tracing (`c7f7bba47`; a tiny debug PR at most), all probes.
- **D5: MTP:** that is upstream PR #28243's own work. The branch `mtp-on-mega-2` exists only for local measurement. The K/V-store-only decode (`c3f06853b`) can go to #28243's author as a suggestion.
- **D6: agent and analysis files and production scripts.**

## Checklist per extracted PR

1. Branch `pr/NN-name` from `origin/master` (plus listed dependencies). Bring over only the listed hunks, and strip per rule 5.
2. Build the SYCL backend; zero new warnings in the touched files.
3. Run the named `test-backend-ops` suites on one card (`-b SYCL0`), comparing against the base branch's results.
4. End to end, production-like flags, interleaved ABBA against the base with the first run discarded:
   - prefill and decode at 2k/13k/54k/104k;
   - fdinfo peak VRAM;
   - text checks every run.
5. Write the numbers into a scratch file for the user. Do not write the PR description.
6. Hand the branch to the user for review. Push or open a PR only on the user's explicit instruction, after the ban-risk acknowledgement in AGENTS.md.
