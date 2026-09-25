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

## Strategy: port CHANGES into fresh branches, do not cherry-pick commits

Decided 2026-09-19. For each PR group, branch from `master` and apply the FINAL STATE of the
relevant code, with a history written for the reviewer. Do NOT try to rebase or cherry-pick
the existing commits.

Why: the branch history contains two exact revert pairs, one PARTIAL cancel
(`730837acd` adds both an ADD-chain fusion and a hyper-connection fusion; `2249821b5`
removes only the HC half), and three squashed grab-bag commits (`31cd5076b`, `6ed9a8be4`,
`e6d58423d`) that each span several PR groups. Reproducing that history buys nothing and
costs a lot. Porting also lets the diagnostics be stripped as part of the port rather than
as a separate cleanup pass.

### Completeness check (do this, it is the whole risk of porting by hand)

Porting by hand can silently drop a hunk. After the PR branches exist, verify:

1. Create a scratch branch that merges every PR branch together.
2. `git diff <scratch> performance_uplift_mega_branch -- ggml/ src/` must contain ONLY the
   intentional drops listed above (the revert pairs, the HC fusion half, the graph commits
   if dropped, the stripped diagnostics).
3. Anything else in that diff is a hunk that was lost in the port. Investigate before sending.

Keep `performance_uplift_mega_branch` intact as the reference until that check passes.

### Stacking order

- #1 (ggml-alloc view-inplace fix) is independent of everything. Send it first and alone.
- #2 (fusion-aware allocation infra) must precede #6 (QSA fusion series), so branch #6 from
  #2 rather than from master, and say so in the PR.
- #3, #4, #5, #8, #9, #10 are mutually independent and can branch from master directly.
- #7 (FA staging clamp) touches `fattn.cpp`, which the sparse-FA integration also touches.
  Check for interaction before branching.

### Timing

Do this once the in-flight work has landed and been verified. As of 2026-09-19 the QSA mask
fusion and the FA staging clamp are still being measured, so the final state is still moving.

## Findings from the 2026-09-19 commit review

- `5870e2314` (gated DSV4_HC_PRE + scale) and `24b388e97` (DSV4_HC_POST without a comb
  matrix) are ONE change: both touch only `dsv4-hc.cpp` plus a line of dispatch, and both
  complete the same op's support. Port as a single PR.
- `730837acd` bundles the decode-graph ADD-chain fusion with a hyper-connection
  REPEAT/MUL/ADD fusion. `2249821b5` removes only the HC half (-221 of +507 in
  `binbcast.cpp`, and -51/+51 in `tests/`, which nets to zero; AGENTS.md forbids touching
  `tests/`, so make sure the port introduces no `tests/` change at all). Net surviving
  content is the ADD-chain fusion only, so `730837acd`'s message no longer describes what
  is in the branch. Port as ADD-chain fusion alone.
- `f30cc3593` is comment-only (documents the MoE 8-token cap). Fold into the MoE mat-vec PR.

## PR branches prepared so far

| branch | worktree | commit | state |
|---|---|---|---|
| `sycl-dsv4-hc` | `/root/wt-dsv4-hc` | `f0e82aa1e` | ported, syntax-checked, NOT built, NOT pushed |

`sycl-dsv4-hc` = group #8, the two DSV4_HC commits squashed. Port was trivial because only
those two commits touch `dsv4-hc.cpp` anywhere on the branch, so the file diff against master
IS the change. Verified byte-identical to the mega-branch, plus two `supports_op` guard
relaxations in `ggml-sycl.cpp`. Still needs a real build and a run: both guards previously
REFUSED these cases, so neither path has executed from a clean master base.

Note for the remaining groups: this file-level shortcut will NOT work for groups 3, 4, 5, 6
and 9, which all touch `ggml-sycl.cpp` and/or `binbcast.cpp` alongside unrelated work. Those
need hunk-level extraction plus the merge-and-diff completeness check above.

## Per-group extractability (checked 2026-09-19)

The porting strategy says to port final STATE, not commits. Whether that is easy depends on
whether later work rewrote the same region. Checked so far:

| group | extractable how | evidence |
|---|---|---|
| #8 DSV4_HC | CLEAN, file-level | only 2 commits touch `dsv4-hc.cpp` in the whole branch, so `git diff master..HEAD` on that file IS the change. DONE, branch `sycl-dsv4-hc`. |
| #10 pinned ring buffer | CLEAN, likely commit-level | single commit `53c7c255c`, single file, 68 lines. 30/30 sampled added lines still present verbatim at HEAD, so nothing later rewrote it. Verify the remaining lines before porting. |
| #3 MoE mat-vec | NEEDS HUNK WORK | `mmvq.cpp` final state mixes this series (1010 lines in `7bf712d2e` alone) with the later wide-load and ESIMD work from checkpoint `31cd5076b`. Cannot split by file. |
| #5 decode kernels | NEEDS HUNK WORK | buried inside squashed checkpoint `31cd5076b`, spans `mmvq.cpp`, `dmmv.cpp`, `vecdotq.hpp`, `ggml-sycl.cpp`. |
| #1 alloc view-inplace fix | NEEDS SEPARATION | shares `ggml-alloc.c` with the fusion-aware allocation infra (#2). The two must be split apart, and #1 should go first and alone since it benefits every backend. |
| #4 grouped MoE GEMM | UNCHECKED | `fused-gemm.cpp` may be entirely its own file, in which case it is clean like #8. Worth the same check. |
| #6 QSA fusions | MOSTLY CLEAN | four new files (`qsa-score`, `qsa-mask`, plus matcher additions), but all share wiring in `ggml-sycl.cpp` and `topk-moe.cpp` with each other and with #7. |

Cheap check used for #10, reusable: extract the commit's added non-comment lines and grep each
against HEAD. All present means nothing later rewrote the region, so the commit is close to the
final state and ports directly.

## RESOLVED 2026-09-19: attribution of the fattn-mkl softmax rewrite

Upstream PR 28918 ("one work-group per query row", `scratchpad/pr28918.diff`) rewrites
`mkl_fa_online_softmax_chunk` in exactly the way our agent also rewrote it, touches only
`ggml/src/ggml-sycl/fattn-mkl.cpp`, and is expected to merge. Our agent's report never
mentioned it, so as of 2026-09-19 we do not know whether our version was derived from it or
written in parallel.

Until that is answered, NONE of the softmax work can be split into a PR: a parallel rewrite
of the same function would conflict with upstream and would read as duplicating their work.

The plausibly-novel delta is the register slice (`float s[ITEMS]`, ITEMS=32) which lets the
max pass and the exp pass share loaded values, collapsing the passes over `KQ_f32` and
reading the mask once. PR 28918 does NOT do that; both of its passes call `score(i)`, which
re-reads `KQ_row[i]`. If that delta is genuinely ours it is worth upstreaming ON TOP of 28918,
not instead of it.

Also unresolved: the reported 12.9x softmax speedup is one undifferentiated number covering
both the work-group restructure and the register slice. It needs splitting before any of it
is claimed.

### RESOLUTION

The agent wrote the work-group rewrite INDEPENDENTLY, before being told PR 28918 existed. It
has since rebased: the tree now carries 28918's kernel verbatim and the register slice is
reverted (preserved at `scratchpad/fa/fattn-mkl.mine.cpp`).

Measured split, same configuration (ceiling 256, chunk 8192, q_tile 792), softmax component:

| | softmax us @n_kv 38912 | end to end |
|---|---|---|
| serial (original) | 373881 | - |
| 28918 alone | 30512 / 30728 -> **12.2x** | - |
| + register slice | 28961 / 28983 -> 12.9x cumulative | **+0.6%** |

**Upstream's restructure is essentially the whole win.** The register slice was dropped: 0.6%
is not worth diverging from a kernel upstream will merge, and another agent has since
templated that same kernel body on `SEL`, which the slice would have to be rewritten around.

The re-derived chunk tuning is ALSO worth nothing on the fixed kernel: chunk 1024 vs 8192 on
the 28918 base measures 458.39 / 363.85 against 460.28 / 363.67, i.e. within noise. The old
sweep had been measuring the serial kernel, not the chunking; that observation is real and
worth keeping, but the retuning is not a contribution.

**What IS ours in `fattn-mkl.cpp`:** the `GGML_SYCL_FA_MAX_MEM_MIB` ceiling, the chunk and
q_tile derivation from it, and the clamp in `fattn.cpp` that routes FA to the chunked kernel.
That is the -304 MiB reserve / -297 MiB net per card. The softmax kernel is upstream's and
must not be presented as ours.

**One-character divergence to fix at PR time:** our copy writes a hyphen where 28918's added
comment has an emdash (AGENTS.md forbids non-ASCII). That single line will conflict on merge.
Restore the emdash when the tree is quiet, since we are carrying upstream's kernel rather
than authoring it. `fattn-mkl.cpp` was held by another agent at the time this was found.

## Measurement hygiene note added 2026-09-19

`scratchpad/gpu_run.sh` now guards CPU load as well as GPU exclusivity. A concurrent `-j16`
build starves the CPU-side prefill path and invalidated at least one measurement arm at
loadavg 5.9 with no GPU contention at all. Any throughput number taken before this fix, by
any agent, is suspect and should be re-run before it appears in a PR. Reserve-side numbers
(`sched_reserve`, peak ladders) are unaffected, since they are computed at graph-allocation
time.

## Open questions

- Flag defaults: five flags currently default OFF (`GGML_ALLOC_INPLACE_VIEWS`,
  `GGML_SYCL_FUSE_QSA_{GATHER,TOPK,SCORE,MASK}`). The allocator fix and the gather fusion
  are verified and arguably should default ON. `GGML_SYCL_FUSE_CAST_ADD` already defaults ON.
- `GGML_SYCL_SPARSE_FA` is read by `src/models/qwen4exp.cpp`, i.e. model code reading a
  backend-namespaced env var. Upstream reviewers will likely object. Not ours to fix unless
  we are contributing to that PR.
