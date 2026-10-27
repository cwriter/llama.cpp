# Merge of upstream master on 2026-10-27

Upstream `19ff817f` merged into `performance_uplift_mega_branch` (`7faa54ca`) as a merge commit, like the 2026-10-20 update: no history rewrite, no force push. The branch's own history (including the re-attached pre-rebase history of `bf1f0092`) is untouched.

Nothing below was compiled or run with oneAPI: the merge was done in a container without the SYCL toolchain. The CPU build and the CPU `test-backend-ops` compile were checked (see the end). The first step on a GPU machine is a SYCL build and `test-backend-ops -o MUL_MAT_ID`, `-o MUL_MAT`, `-o FLASH_ATTN_EXT`, `-o GET_ROWS`.

## How the conflicts were resolved

17 files conflicted (75 hunks). Rule of thumb: where the branch already had its own, evolved version of what upstream did, the branch's code was kept and upstream's fixes were ported into it; where upstream reworked code the branch had not touched, upstream's version was taken.

- **SYCL graph record/replay (#28725, `7377bada`).** The branch carried an earlier copy of this PR and extended it (per-node blocker log, sparse attention exemption, MoE consumer check, reorder gate, device-local check). The branch's graph code is kept. Taken from upstream: `can_use_mul_mat_vec_q()` now takes the device (#31014, the XMX batch check needs it), the configurable `GGML_SYCL_GRAPH_EVICTION_TIMEOUT`, and the one-time "enabling / disabling SYCL graphs" log.
- **Pinned upload staging and `set_tensor_2d` (#30358).** The branch replaced the slot staging with a growing ring (`stg.used`, `stg.cap`), which upstream's `set_tensor_2d` (written against `staging.next/events/submitted`) cannot use. A branch version was written: contiguous destination rows are packed into host batches of at most 64 MiB, each sent with one `set_tensor` call (which copies into the pinned ring before returning). Non-contiguous rows, reordered MoE owners, packed KQ masks and SoA KV fall back to one `set_tensor` per row, as the generic path does.
- **Reorder temp memory bound (#31062).** Upstream bounds every reorder's temporary to 32 MiB (`reorder_qw_planar`, chunked MoE reorders). Q8_0, IQ3_S and IQ3_XXS dense reorders now use upstream's `reorder_qw_planar` (same byte layout as before; the branch versions were cosmetic variants of the old code). The branch keeps its own Q6_K dense reorder and its own MoE reorder structure (one shared temporary for all chunks); both are chunked through `GGML_SYCL_MEM_SAVE_REORDER_CHUNK`, which is on by default, so every reorder stays within a small temporary either way.
- **XMX multi-column mat-vec (#29864, `d81d0d8f`).** Added next to the branch's ESIMD multi-column kernels. Dispatch order in `ggml_sycl_mul_mat`: the branch's ESIMD n-cols (Q8_0, Q6_K, 2..N columns) first, then DMMV, then MMVQ, where upstream's XMX kernel takes reordered Q2_K..Q6_K and Q8_0 batches. Upstream's code reads `optimized_feature.reorder`; on the branch that is `is_reordered()`, translated everywhere.
- **Fused GEMM on reordered weights (#30604 part).** The branch already did this with its own signature (reordered flag after `ldc`, base pointer check); the branch's `fused-gemm.cpp/.hpp` are kept as they were.
- **IQ3 crash fix (#30604, `493253fe`).** Upstream changed `dequantize_block_iq3_{s,xxs}_reorder()` to take the block index and a pre-offset `y`, so that GET_ROWS can read reordered IQ3 (tied embeddings). The branch's vectorized IQ3_S version was moved to that signature.
- **convert.cpp (#30740 merged fp16/fp32 switches, #30775 fast Q8_0 dequant).** Upstream's merged `ggml_get_dequant_sycl<dst_t>` is taken, with the branch's additions re-applied: reordered IQ4_NL and IQ4_XS dequant, the IQ4_NL work-group size, the wide f32->f16 conversion, `is_reordered()`. The branch keeps its own wide Q8_0 reordered dequant; upstream's `dequantize_block_q8_0_reorder_wide(..., sycl::id<1>)` overload is present but unused (see benchmark 3).
- **fattn-mkl (#30684, explicit dependencies instead of host waits).** The branch had already removed those waits its own way (online Q8, one-pass softmax, `GGML_SYCL_MKL_FA_DRAIN`), so the branch's file is kept as is.
- **fattn-sparse (#30812, memset not always ordered).** The branch's mask compaction kernel already replaced that memset.
- **fattn (#30286, K/V head-size mismatch).** Upstream's reshape is applied before the branch's packed-mask handling.
- **llama side.** `memory_breakdown()` uses upstream's `llama_memory_bufs` plus the branch's KV rotation buffers; the rotation code now reads `no_alloc` from `ctxs_bufs` (upstream moved it out of `hparams`, #31141). The branch's SoA `kv_size % 256` assert stays; upstream dropped `n_pad` (#31200) but still rounds `n_ctx` to 256. qwen4exp's MTP layer takes upstream's `hc_mix` struct (#31142) and keeps the branch's KV catch-up early return.
- **tests.** `test_mul_mat_id` keeps the branch's `amax` and `b_off` and gains upstream's `skip` as the last parameter. Upstream dropped `amax` (#30924), so its new and changed call sites were rewritten to the branch's argument order; taken unchanged they would have silently passed `m_v` as `amax`, and `init_mul_mat_id_tensors(ctx, n_mats, amax)` would have turned `amax = 1.0f` into `skip = true`.

## MUL_MAT_ID with id == -1 (#26631)

Upstream lets a MUL_MAT_ID id be -1 ("skip this slot, the dst row is 0"). No model emits it yet; only test-backend-ops (`skip=1` cases). State on the branch:

- Handled: the host-scheduled path (counting sort skips them, a kernel zeroes their dst rows; this is upstream's code, ported into `ggml_sycl_mul_mat_id_host_sched`), the plain and reordered MMVQ MoE kernels (zero their rows), the route-ordered MMVQ path and the device-scheduled grouped GEMM (both drop -1 routes; `ggml_sycl_mmid_zero_skipped()` zeroes the rows after the launch).
- **Not handled:** the ESIMD MoE kernels in `dmmv.cpp` (`ggml_sycl_mul_mat_vec_q_id_reorder_esimd` and the fused gate/up ones) index the weights with the id unchecked, so a -1 reads one expert slice before the tensor. The `skip=1` test cases for IQ4_NL / Q8_0 / IQ3_S at a few tokens can hit them. Fix before any model starts emitting -1: an `if (i02 < 0)` guard that writes zeros, per kernel.

## Benchmarks to run (merged vs `7faa54ca`)

1. **XMX takes Q2_K/Q3_K/Q4_K single-column decode.** Upstream turns DMMV off for one column when `ggml_sycl_xmx_min_cols(type) == 1` and the reordered weight can use the XMX kernel (BMG and LNL only), so reordered Q4_K decode now runs the XMX DPAS kernel instead of the branch's tuned path. Upstream measured it against its own kernels, not the branch's. Compare tg with `GGML_SYCL_ENABLE_XMX_ESIMD=0` and `=1` on a Q4_K model (and Q4_K_M MoE). Also short batches (spec decode / MTP verify, 2..16 columns), where XMX competes with the branch's MMVQ.
2. **XMX fused gate/up.** `ggml_sycl_mul_mat_glu_fused` now first tries upstream's XMX GLU kernel for same-type Q2_K..Q6_K/Q8_0 pairs up to `GGML_SYCL_XMX_GLU_MAX_COLS` (16) columns, ahead of the branch's reorder / ESIMD GLU paths. Same A/B with `GGML_SYCL_ENABLE_XMX_ESIMD=0`.
3. **Q8_0 reordered dequant.** The branch's wide kernel vs upstream's #30775 version (switch `dequantize_row_q8_0_sycl_reorder` to it to test). Prefill of a Q8_0 model.
4. **qwen4exp QSA graph (#30965, #31243, #30640).** Upstream builds the QSA selection without the kq_mask and added `GGML_OP_TOP_K_MASK`. SYCL does not implement that op, so llama keeps the non-fused path, but that path was itself rebuilt upstream. The branch's QSA fusions (`GGML_SYCL_FUSE_QSA_*`) match graph patterns and silently stop firing if the pattern changed. Check that the fusions still fire (`GGML_SYCL_CENSUS` or the fusion debug log) and compare pp/tg and compute buffer size on qwen4exp. A SYCL `TOP_K_MASK` kernel is the likely follow-up if the old fusions no longer apply.
5. **Graph replay.** `GGML_SYCL_ENABLE_GRAPH=1` still records (the graph code is the branch's); check the log for the new one-time "enabling SYCL graphs" line and that the replay count is unchanged.

## Checked here

- CPU build of everything, including `test-backend-ops` with the merged MUL_MAT_ID test changes.
- Every top-level function of `ggml-sycl.cpp`, `mmvq.cpp`, `getrows.cpp`, `fusion.cpp`, `common.hpp`, `fattn.cpp` that matches neither the branch nor upstream (that is, one git combined from both) was read; three such mixes were wrong and fixed (upstream's `needs_sync` check inside the branch's MoE shape gate, which would have sent 2..8-token MoE batches to the host path; its assert in the host path; a duplicated `can_use_mul_mat_q`).


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
