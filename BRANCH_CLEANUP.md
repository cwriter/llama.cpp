# Branch cleanup working document

Working doc for turning `performance_uplift_mega_branch` into reviewable sub-PRs.
Workflow: clean the branch first, then cut one PR per coherent group from the cleaned base.

## RULE: verify against the tree, not against a document

`COMPUTE_BUFFER_ANALYSIS.md` is the running ledger and parts of it are STALE. Two entries
have already been wrong:
- it claimed the KQ mask lives in a host buffer and is outside the per-device figure. The
  allocator live map shows `SYCL0#attn_inp_kq_mask#0` at 256 MiB in SYCL0's own chunk.
- it claimed the cast+add fusion was UNREACHABLE. The bespoke matcher the doc itself
  prescribed was subsequently written; it now fires and is on by default.

So every line below carries a verification status. Do not move an item to "drop" on the
strength of a note. Re-check it against the tree or git, and record the date.

Status key: VERIFIED (checked against tree/git on the date shown) | ASSUMED (not re-checked)

---

## Do not ship

| Item | Status | Evidence |
|---|---|---|
| `9230717f0` + `726ea56da` (host-buffer staging on PVC, and its revert) | VERIFIED 2026-09-19 | `git diff 9230717f0^ 726ea56da --stat` is EMPTY. Cancels exactly. Drop both. |
| `414490a3b` + `67b7d92a3` (MoE route order floor 64 -> 8, and its revert) | VERIFIED 2026-09-19 | `git diff 414490a3b^ 67b7d92a3 --stat` is EMPTY. Cancels exactly. Drop both. |
| iq3s_grid in SLM | VERIFIED 2026-09-19 | Not present in the tree; already reverted. Nothing to do. |
| Sparse FA (`fattn-sparse.*`, the qwen4exp enable) | VERIFIED 2026-09-19 | This is upstream PR 28796, integrated here. Do NOT re-submit. Our only novel part would be feedback on their PR. |
| IQ3 reorder patches 0001 | VERIFIED 2026-09-19 | Extracted from upstream draft PR 29107. Do NOT re-submit 0001. Patches 0002 (reorder-aware `fg_stage_a`) and 0003 (MUL_MAT_ID per-expert plumbing) ARE novel and could contribute on top. |

## Needs a decision, do NOT silently drop

| Item | Status | Problem |
|---|---|---|
| SYCL graph record/replay: `f5d726626`, `7ad592a94`, `125a5b3ee`, `e04573833` | VERIFIED 2026-09-19 | Measured to buy nothing here (decode is submission-bound), BUT there are ~50 live references across `ggml-sycl.cpp` (39), `common.hpp` (9), `fattn.hpp` (1), `fattn.cpp` (1). `uses_library` is read by the CURRENT FA dispatch, including the newly integrated sparse/MKL paths. Ripping these out is surgery, not a rebase-drop. Options: (a) keep as their own PR, honestly labelled "no measured benefit on this configuration, may help single-device or non-library kernels"; (b) surgical removal, costly and conflict-prone. Recommend (a). |
| `ggml_sycl_fuse_cont_add` (`GGML_SYCL_FUSE_CONT_ADD`, default 0) | ASSUMED | Measured 0 MiB on its own. Superseded in practice by the QSA fusions. Decide: drop, or keep as a fallback in the cascade. Worth a quick check that nothing else calls it. |
| Fusion-aware allocation infra | VERIFIED 2026-09-19 | Was worth 0 MiB for its original consumer (cast+add), but is now load-bearing for the QSA fusion series. It also carries a real correctness fix (the "update parents" loop must not run for absorbed nodes, or it frees a buffer the fused op still reads). Ships, but only alongside a consumer that justifies it. |

## CORRECTED: these DO ship

| Item | Was | Now |
|---|---|---|
| cast+add fusion | listed as "unreachable, do not ship" | VERIFIED 2026-09-19: `ggml_sycl_cast_add_shape` (binbcast.cpp:328) is a bespoke matcher that bypasses `ggml_can_fuse_subgraph`; `ggml_sycl_can_fuse_cast_add` calls it; `GGML_SYCL_FUSE_CAST_ADD` defaults to 1. It is the ONLY QSA-area fusion enabled by default. It is ALSO reused by the QSA top-k fusion (topk-radix.cpp:449), so it cannot be removed regardless. Its "~1% prefill" estimate predates the fix and has never been re-measured. |

---

## Cleanup tasks before cutting PRs

### 1. Strip or promote the diagnostics

Counts VERIFIED 2026-09-19 (occurrences in ggml-sycl/*.cpp, *.hpp and ggml-alloc.c):

| Trace | Count | Disposition |
|---|---|---|
| `INPL_DIAG` | 4 | strip (mine, served its purpose) |
| `[GG]` | 2 | strip |
| `[MV]` | 2 | strip |
| `[CASTADD]` | 1 | strip |
| `[QSAGATHER]` | 1 | decide: useful for proving a fusion fired |
| `[FA-SPARSE]` | 1 | came with the upstream PR; leave as upstream has it |
| `[PEAK]` (`GGML_ALLOC_PEAK_DEBUG`) | 1 | PROMOTE, do not strip. It found every memory win in this project. If kept it must be documented and the output format stabilised. |
| `[CASTDIAG]`, `LIVEMAP` | 0 | already gone |

Note the compile-time `GGML_ALLOCATOR_DEBUG` live-map in ggml-alloc.c is UPSTREAM and
untouched; it is the single best diagnostic here and needs no change.

### 2. Unpick the two checkpoint commits

`31cd5076b` ("Checkpoint +50% prefill, +8% decode") and `6ed9a8be4` ("Checkpoint -640
MiB/card") are squashed grab-bags spanning several PR groups. Nothing splits cleanly until
they are broken up. This is the largest single cleanup task.

### 3. Commit the untracked work

VERIFIED 2026-09-19: these are UNTRACKED, i.e. not recoverable by git if lost:
`fattn-sparse.{cpp,hpp}`, `qsa-score.{cpp,hpp}`, `qsa-mask.{cpp,hpp}`.
Two of them carry verified wins (sparse FA +29.5% decode, GEMM epilogue -80 MiB).

### 4. Re-measure anything timed before the GPU mutex

Throughput numbers taken while two agents could both hold the GPU are suspect. Buffer sizes
and peak ladders are NOT affected (computed at graph-allocation time). Specifically re-check
the sparse FA decode figures (+29.5% at 64k, +5.3% at 15k) under `scratchpad/gpu_run.sh`.

---

## Open questions

- Flag defaults: five flags currently default OFF (`GGML_ALLOC_INPLACE_VIEWS`,
  `GGML_SYCL_FUSE_QSA_{GATHER,TOPK,SCORE,MASK}`). The allocator fix and the gather fusion
  are verified and arguably should default ON. `GGML_SYCL_FUSE_CAST_ADD` already defaults ON.
- `GGML_SYCL_SPARSE_FA` is read by `src/models/qwen4exp.cpp`, i.e. model code reading a
  backend-namespaced env var. Upstream reviewers will likely object. Not ours to fix unless
  we are contributing to that PR.
