# Rebase of performance_uplift_mega_branch onto upstream master

Upstream base: ggml-org/llama.cpp `8345f33` (2026-10-05). Old base: `ec7630a` (2026-10-01).

## Dropped commits

- Three revert pairs that cancel exactly (net diff is empty): `c93ee68`/`6d7c508`, `0af8233`/`efcd206`, `1d61ac5`/`5c6d45a`.

## Removed as obsolete

- The QSA indexer score fusion (`qsa-score.cpp`, `GGML_SYCL_FUSE_QSA_SCORE`): no model builds its chain any more, see item 1 below. Its own commit, so it is easy to revert.

## Conflicts resolved

Upstream overlaps the branch in five places; each was resolved by hand:


- `4a213f8` (Checkpoint +50% prefill, +8% decode) vs upstream `392ded6` (#29186, Q8_0 DMMV ESIMD and MMVQ wide load). Upstream holds the reviewed version of the same kernels. Took upstream's code and comments, including its odd-row fix in the Q8_0 ESIMD kernel (`base1 = has_row1 ? ... : base0`, the branch read one row past the end for an odd row count). Kept the branch-only `GGML_SYCL_ESIMD_Q8_0` switch, which later branch code uses. Dropped the branch's `GGML_SYCL_GG_TRACE` print in the Q8_0 ESIMD launcher (debug only).
- `43e1893` / `69edc1f` (layout descriptor, device policy split) vs upstream `c328acc` (#28985), which added `onednn_optimized_gemm` to the old `optimize_feature`. It is a per-device fact, so it moved to the branch's `device_opt_feature`.
- `20115fe`, `787ed62`, `3d0eeba`: flag lists in `ggml-sycl.cpp` that both sides extended. All flags kept, no duplicates: `GGML_SYCL_MMVQ_WIDE` (upstream), `GGML_SYCL_ESIMD_Q8_0`, `GGML_SYCL_MMID_SCHED`, `GGML_SYCL_WIDE_LOADS`, and the `GGML_SYCL_ESIMD_DEFAULT` default.
- `d32d3c0` (`get_max_alloc_size`) vs upstream `631109b` (#23671, `alloc_buffer_n` / `get_alloc_size_n`): both add slots to the buffer-type interface. All kept, see item 6.
- `db41691` (port of the QSA mask fusion to the k-pool graph): followed the commit and removed the gather, top-k, cast+add and cont+add fusion flags it removes.
- `606cfb6` (new): the graph test that predicts which library a node uses now applies upstream's oneDNN check too.

## Upstream changes that affect the branch

Each item says whether it changes behaviour on the branch, and whether it needs an A/B benchmark against the pre-rebase branch (`8e4fb21`) to know if upstream's version is better or worse. Nothing below was run on a GPU: the rebase was done in a container without oneAPI, so the SYCL backend was not compiled here (see "Not verified").

### 1. Indexer score now uses `ggml_lightning_indexer` (#29825) - BREAKS a branch fusion, BENCHMARK

- Upstream builds the qwen4exp indexer score with the new `GGML_OP_LIGHTNING_INDEXER` op and an f16 pool mask, instead of MUL_MAT -> RELU -> per-head ADD chain.
- The branch's QSA score fusion (`qsa-score.cpp`) matched that old chain. No model builds it any more (qwen4exp always uses the op; glm5-next's non-fused fallback builds a different chain). It is removed in `sycl : drop the QSA score fusion` (`40037fa`), together with `GGML_SYCL_FUSE_QSA_SCORE`. Part B of `SYCL-int8-xmx-plan.md` is marked obsolete.
- The op now runs on the existing SYCL kernel `lightning-indexer.cpp`. Two risks:
  - **CPU fallback.** `supports_op` requires the query head size to be `WARP_SIZE * 8` = 128 and the mask to be F16. If the model's `indexer_head_size` is not 128, the op silently runs on the CPU. Check once with `GGML_SCHED_DEBUG=2` (or the verbose scheduler log) that `LIGHTNING_INDEXER` nodes are on SYCL devices.
  - **Speed.** The SYCL kernel is a plain per-row loop with no XMX. The removed fusion used a oneDNN GEMM plus a tiled reduction. Upstream's version should use less memory (no per-head score is materialized; the agent notes recorded -463 MiB per card for the fusion, and paging on card 0 without it), but it may be slower.
- **Benchmark:** qwen4exp, three B60s, same command as the agent notes (131k context, pipeline mode, MTP): compute buffer MiB per card, pp32k and fill-to-100k, old branch vs rebased. If the rebased branch is slower, the next step is a faster SYCL lightning indexer (an XMX or int8 version), not reviving the old fusion.

### 2. QSA mask construction changed (#29824) - probably still fused, VERIFY

- Upstream builds the -inf rows and the zeros from fresh `[n, 1, 1, 1]` tensors (`FILL -> REPEAT -> RESHAPE`) instead of a scalar cast from the selection list.
- The branch's mask matcher (`qsa-mask.cpp`, `qsa_mask_chain_from_fa`) follows views to `REPEAT(FILL(-inf))` and checks element counts and row strides, all of which still hold for the new shapes. The seed-scalar absorption loop simply finds nothing to absorb now; the two FILL source leaves are tiny.
- **Verify** that the fusion fires: compute buffer per card should match the pre-rebase build (the notes recorded -334 MiB per card from this fusion). If the buffer grows by roughly that amount, the matcher no longer matches and needs adjusting.

### 3. oneDNN reference-kernel check (#28985) - merged, CHECK THE LOG

- Upstream probes oneDNN at startup and skips oneDNN matmul and SDPA when oneDNN would only use reference kernels. The check was moved into the branch's `device_opt_feature`, and the branch's graph test that predicts which library a node uses was updated to ask the same question (`606cfb6`).
- On B60 oneDNN should have optimized kernels, so nothing changes. **Check the startup log** for `oneDNN has no optimized matmul for device`. If it appears, every oneDNN path is now off on that device and prefill numbers will move; benchmark pp512/pp2048 in that case.

### 4. Your Q8_0 ESIMD and wide-load PR landed upstream (#29186) - merged, quick BENCHMARK

- Upstream's version replaced the branch checkpoint's copy of the same kernels (see "Conflicts resolved" above). Upstream also fixes an out-of-bounds read of row `n+1` for an odd row count.
- `GGML_SYCL_ESIMD_Q8_0` (branch only) is kept. `ggml_sycl_supports_reorder_esimd` now has a redundant `GGML_TYPE_Q8_0` case from upstream behind the branch's early return; harmless.
- **Benchmark:** tg128 on a Q8_0-heavy model, old vs rebased. Expect no change.

### 5. Large register file for D=512 flash-attention vec kernels (#29062) - merged, BENCHMARK if relevant

- Only affects head size 512 on the vec kernel (Gemma global layers). Merged without conflict. If a D=512 model is in use, run tg128 old vs rebased.

### 6. Buffer-type interface: `alloc_buffer_n` / `get_alloc_size_n` (#23671) - merged

- Upstream and the branch (`get_max_alloc_size`) both added slots to `ggml_backend_buffer_type_i`. All are kept, in the order `alloc_buffer`, `alloc_buffer_n`, ..., `get_alloc_size`, `get_alloc_size_n`, `get_max_alloc_size`, `is_host`, and every backend's initializer was checked to have all slots. SYCL uses the defaults for the two new upstream slots.
- Weights now go through the default `alloc_buffer_n`, which sizes tensors with `get_alloc_size`. The branch's compact KQ mask and flash-attention scratch report sizes through `get_alloc_size` / `get_max_alloc_size` for graph tensors only, so weight placement should not change. **Verify** the model buffer sizes in the load log against the pre-rebase build.

### 7. k-pool graph shape and reserve fixes (#29958, #29819) - merged, RUN the realloc check

- Upstream makes the qwen4exp/glm5-next graph shape independent of `cache_safe`, removes the `cache_safe` graph API, and fixes reserve sizing. The branch's k-pool commit (`keep the k-pool layout before a sequence edit`) touches a different part of the same file and merged cleanly; they are complementary.
- The branch also has several reserve-related commits for MTP (`reserve again when the h_nextn export changes`, `reserve the MTP draft prompt graph without outputs`, ...). **Run** the model with MTP under `GGML_SCHED_DEBUG_REALLOC=1` (decode, prompt cache restore, `seq_rm` after rejected drafts). An abort means a reserve interaction between the two sets of fixes.

### 8. Probabilistic draft acceptance for MTP (#27694) - merged, behaviour change for benchmarks

- Upstream adds probabilistic sampling for simple draft and MTP. It merged cleanly with the branch's MTP server changes (`run the MTP draft catch-up one prompt batch late`, ...).
- At temperature > 0 the acceptance rate can differ from earlier runs, so MTP tg numbers are not directly comparable with older measurements. **Benchmark** MTP decode at temperature 0 (unchanged behaviour) and at the usual sampling temperature, and record the acceptance rate.

### 9. Recurrent state gather (#29856) - merged, no action

- One `get_rows` now gathers all `n_rs` states. No SYCL fusion matches `GET_ROWS` on states, and the gated delta net fusions read the views, so nothing on the branch depends on the old shape.

### Upstream ideas worth porting (not breakage)

- `CUDA: fuse shared experts into MMVQ` (#29184): the same fusion would fit the SYCL MoE mat-vec on qwen4exp, which has a shared expert.
- `CUDA: use MMVF for thin f16/bf16 mul_mat at small batch size` (#29633).

## Not verified

- **The SYCL backend was not compiled.** The container has no oneAPI compiler. All SYCL conflict resolutions were reviewed by hand; the first build on the B60 machine may still show compile errors. The CPU build of the core library and tests is recorded below.
- No GPU test or benchmark was run. Items 1, 2, 7 and 8 above need a run before the rebased branch replaces the old one.

## Checked here

- CPU build (Release, no native) of `ggml`, `llama`, `llama-server`, `test-alloc`, `test-llama-archs`, `test-backend-ops`: builds. The only warnings are the existing missing `fusion_absorbs` initializers in the CPU and meta backends (also present before the rebase).
- `test-alloc`: all pass, including upstream's new `alloc_buffer_n` / `get_alloc_size_n` tests next to the branch's `get_max_alloc_size`.
- `test-llama-archs`: 133/133 pass, including qwen4exp and glm5-next with the branch's MTP and k-pool changes on top of upstream's.
- Every positional `ggml_backend_buffer_type_i` initializer in the tree has both new upstream slots and the branch's slot.
- `test-recurrent-state-rollback` needs a model file; not run.
