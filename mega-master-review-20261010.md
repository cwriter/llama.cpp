# Mega branch review, 2026-10-10

Private extraction notes, not a PR description. Compared with fetched upstream master `781dbc5ac`.

## Branch state

- Working branch: `performance_uplift_mega_branch`.
- Before rebase: `f3cbce3fb`, fork point `8345f3339`, 123 commits unique to mega and 118 unique to master.
- Backup: `backup/mega-pre-rebase-20261010`.
- Rebased HEAD: `27c7f106d`; master is an ancestor and the branch is 121 commits ahead, 0 behind. Validated integration repairs were folded into the final replayed commit without changing its message.
- Replayed 121 commits onto master. Dropped the old pinned-upload commit and the nextn export reserve fix because upstream already supplies those changes.
- Original tracked edits (Q2_K/Q3_K and MMID width override) were saved and restored; original untracked files and experimental worktrees were retained.
- Full comparison is 107 files, including substantial private analysis documents. This is not a sensible single PR.
- Build and validation logs: `/tmp/mega-rebase-20261010/`. Results are recorded below.

## Already covered upstream or by open PRs

| Feature | Current status | Action |
| --- | --- | --- |
| Grouped MoE XMX GEMM and precision/shape selection | Merged [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) | Extract only residual optimizations, not the original feature |
| Pinned model upload ring | Merged [#29608](https://github.com/ggml-org/llama.cpp/pull/29608) | The remaining change is the growing input ring and separate v2 upload queue |
| Q5_K reorder MMVQ and GLU | Merged [#29375](https://github.com/ggml-org/llama.cpp/pull/29375) | Keep master's implementation and device gate |
| MXFP4 MoE reorder | Merged [#29809](https://github.com/ggml-org/llama.cpp/pull/29809) | Preserve its producer, dispatch, and alignment guards |
| Delta-net alpha ADD+UNARY+MUL fusion | Merged [#29687](https://github.com/ggml-org/llama.cpp/pull/29687) | The F32 matvec-plus-tail proposal is a further fusion |
| IQ3_S multi-column MMVQ | Merged [#29500](https://github.com/ggml-org/llama.cpp/pull/29500) | Do not reintroduce the base feature |
| GLM MLA MKL prefill, FA buffer cleanup | Merged [#29171](https://github.com/ggml-org/llama.cpp/pull/29171), [#27689](https://github.com/ggml-org/llama.cpp/pull/27689) | Preserve the new layout handling |
| Device-side MoE scheduling, graph compatibility, Q2_K/Q3_K | Open [#30156](https://github.com/ggml-org/llama.cpp/pull/30156) | Keep in the existing PR stream |
| Reference-counted server checkpoints | Open [#30137](https://github.com/ggml-org/llama.cpp/pull/30137) | Keep in that PR; preserve master's load-failure path |
| Memory-eliding fusions and allocator callback | Open RFC [#29247](https://github.com/ggml-org/llama.cpp/pull/29247) | Develop through the existing RFC |
| SYCL graph record/replay | Open [#28725](https://github.com/ggml-org/llama.cpp/pull/28725) | Coordinate enhancements, avoid another base graph PR |
| Persistent IQ3 layouts | Open [#29107](https://github.com/ggml-org/llama.cpp/pull/29107) | Coordinate base layouts before extracting dependent consumers |
| Reordered Q8_0 MoE weights | Open [#29337](https://github.com/ggml-org/llama.cpp/pull/29337) | Coordinate the layout; ESIMD is a possible follow-up |
| Q4_K wide row pairing | Open [#30226](https://github.com/ggml-org/llama.cpp/pull/30226) | Do not duplicate it in a generic wide-MMVQ PR |
| GLU/SCALE type-support guards | Open [#29889](https://github.com/ggml-org/llama.cpp/pull/29889) | Same guards as mega's `bc03e8840`; coordinate rather than extract another fix |

Open PR status was checked using the public GitHub API on 2026-10-10. Searches also covered conv, lightning, hyper-connection, softmax, get_rows, concat, state, and rotation buffers. Absence from search is not proof that no overlapping work exists.

## Recommended remaining extractions

Each item needs its own master-based branch and standalone correctness/performance measurements. Existing mega timings include other optimizations and cannot establish an isolated improvement.

| Order | Feature | Rebased source commits / files | Scope and remaining work |
| --- | --- | --- | --- |
| 1 | Flat CONCAT grid for narrow rows | `468b999ef`, `concat.cpp` | Small and backend-local, about 71 added lines. Avoid thousands of mostly idle workgroups for a 4-wide destination. Test all dimensions, narrow and wide rows; shorten the existing explanatory comment |
| 2 | Vectorized floating GET_ROWS | `6e8628f3e`, `getrows.cpp` | Backend-local, alignment and stride gates with scalar fallback. Test offsets, odd widths, and multiple sequences; measure enough shapes to justify the occupancy heuristic |
| 3 | Persistent KV rotation matrices | `706b2e41a`, `llama-kv-cache.*`, `llama-graph.*` | Independent of SYCL-specific layouts. Store matrices once per buffer type instead of uploading per graph. Verify CPU, GPU, no-alloc mode, context shifts, and state round trips |
| 4 | F32 ESIMD matvec | `90f121687`, `dmmv.cpp`, dispatch | Extract the single-token F32 kernel first; keep the activation-tail fusion as a follow-up. Device capability and compiler portability need explicit testing |
| 5 | Small F32 matvec with sigmoid/softplus tail | `35e6ea888`, matcher in `ggml-sycl.cpp`, kernel in `dmmv.cpp` | Builds on the F32 ESIMD kernel. Port the required graph dependency handling, not the full allocator RFC. Test intervening views and extra consumers |
| 6 | Recurrent convolution window fusion | `c5e8565bd`, `58b3bd4c4`, `ssm_conv.*`, fusion matcher | Fuse CONCAT, history copies, convolution and optional SiLU. Specialization is d_conv=4. Confirm state rollover, multiple slots/sequences, and unrelated consumers. Memory elision is a separate dependency on #29247 |
| 7 | GDN reads state directly from recurrent cache | `18c898747`, `d219dd173`, `gated_delta_net.*`, matcher | Remove the gathered-state intermediate from execution. Test shuffled/duplicated cell IDs, rollback snapshots, KDA, and multiple sequences. Distinct from existing GDN output-cache fusion. Related CPU work: [#29121](https://github.com/ggml-org/llama.cpp/pull/29121) |
| 8 | Uncached GDN state stores and wide operand loads | `f7c39a561`, `8c6391f8c`, relevant hunks of `d1fa09c81` | Keep separate from state-gather fusion so cache-policy effects can be measured independently. Check supported oneAPI versions and Arc generations |
| 9 | Faster LIGHTNING_INDEXER kernels | `b4a3df065`, `0853d231f`, `lightning-indexer.cpp` | The upstream operator already exists. Extract its direct/vector and XMX implementations only; retain fallback for unsupported types, shapes and devices. Test partial tiles, masks, batches, and multi-stream strides |
| 10 | oneMKL FA one-pass softmax and in-place accumulation | `21ff63812`, `939db1a90`, `f4acc2cdd` | Start with dense masks and canonical KV. Preserve upstream MLA and batch fixes. Split scratch-budget allocation API work into a separate discussion/PR |
| 11 | Hyper-connection fusions | `88a3fcc05`, `80a2c4f7c`, `a7295a605`, then `288f2b212` | Separate gate up-projection fusion from HC_POST+next RMS and from ESIMD implementation. Port only necessary graph dependencies. Test extra consumers and unsupported shapes |
| 12 | Dense Q8_0 and multi-column Q6_K/Q8_0 ESIMD | `88e630d78`, `4c2b1c3b4`, `f8ee647e5` | Can be split by kernel. Use existing dense reorder layout. Check odd output rows, 1-8 columns, partial stripes and graph replay; avoid the overlapping Q4_K proposal |
| 13 | Routed-expert ESIMD and persistent IQ3_S/IQ4_NL decode | `681b60cc8`, `27c7f106d`, `afecacd6f`, earlier ESIMD/DP4A commits | Valuable follow-ups after coordinating layout PRs #29107/#29337. Separate Q8_0 from IQ types, and row-layout changes from persistence. Verify repeated routing and prefill/decode reuse |
| 14 | Separate upload queue for Level Zero v2 | `a9d1f089a`, plus selected growing-ring hunks from `f0d1e39f5` | Distinct from merged model-loading ring. Requires multi-GPU v1/v2 adapter comparisons and event/lifetime checks. Drop pipe tracing from an extraction |

Commit IDs are navigation aids, not a cherry-pick recipe: checkpoint commits mix features and several later commits assume earlier infrastructure. Use `git log origin/master..HEAD --oneline` to resolve subjects after any further history rewrite.

## Hold or discuss before extracting

- Packed masks, compact allocation and scratch lending: depend on allocation accounting, taught consumers and canonical transfers. First split the generic API discussion from the backend kernels; avoid one large coupled PR.
- Whole-context FA scratch budgets: `get_max_alloc_size` touches the generic buffer interface and every backend initializer. Demonstrate under-allocation on a small reproducer and discuss the interface separately from FA speed improvements.
- SoA q8_0 KV cache: crosses allocation, views, CPY, dequantization and session serialization. Keep opt-in until independent round-trip and fallback testing is complete.
- QSA sparse union-tile and no-readback paths: remove obsolete score-fusion work; upstream now uses LIGHTNING_INDEXER. Revalidate against current k-pool sharing and MTP behavior.
- MTP catch-up overlap and nextn double buffering: one coherent scheduling/lifetime design spanning server, scheduler and model graphs. Upstream #30257 changed multimodal MTP inputs, which must be retained. Do not split these by file.
- Backend sampler reserve signatures: upstream [#30223](https://github.com/ggml-org/llama.cpp/pull/30223) addresses graph stability across ubatches, while this branch skips reserves across sampler attachment. There is also open [#28872](https://github.com/ggml-org/llama.cpp/pull/28872). Reproduce the remaining request-boundary cost first.
- K-pool stale-tail reuse: adapt to master's shared-sequence scatter fix [#29994](https://github.com/ggml-org/llama.cpp/pull/29994) and graph allocation fix #29958. Needs sequence-copy/removal, rollback, and checkpoint tests; do not revive the removed `shared`/`cache_safe` representation.
- Private diagnostics and plans (`census`, pipe tracing, analysis documents, tuning sweeps): exclude from functional extractions.

## Integration validation

Integration fixes preserve master's XMX configuration without duplicate declarations, its removed FA-buffer implementation, MLA V strides and non-aliasing, Q5_K GLU and MXFP4 reorder paths, multimodal MTP input handling, k-pool representative layout, and slot checkpoint serialization. The branch's shared checkpoints, no-init state buffers, IQ reorder controls and MKL-FA bitmask modes remain in use. Two MLA prefill regression cases cover single and multiple batches.

The full SYCL build passed for llama-bench, llama-cli, llama-server, test-backend-ops, test-alloc, test-batch-alloc and test-fusion. CPU checks pass: 32 allocator tests, 375 batch-allocation assertions, and 33 F32 CONCAT/GET_ROWS cases. GPU checks pass: 929 supported operator cases (112 MMID, 19 MXFP4, 40 Q5_K GLU, 707 CONCAT/GET_ROWS/recurrent/HC/indexer, 6 MLA, and 45 attention cases), plus 1,152 changing-routing graph executions across host/device scheduling and F16/BF16/TF32 modes. Worst replay NMSE was 0.000333300858, below the 0.0005 limit. Unsupported operator shapes were skipped by the backend test harness. No correctness failures in these checks. The SYCL build uses oneAPI 2026.1, Release, F16 enabled, graph support enabled, oneDNN disabled. Runtime testing is on one selected Arc Pro B60; it does not establish A770, multi-GPU, other backend, or oneDNN compatibility.


Model smoke runs use full GPU offload, batch 2048, ubatch 128, two repetitions, graph support enabled. These are end-to-end smoke measurements, not isolated PR speedups or a pre/post-rebase performance comparison. Qwen3 MMID=0 pp2048 had large variability, so its mean is not a stable baseline.

| Model | MMID | pp128 t/s | pp2048 t/s | tg32 t/s |
| --- | --- | --- | --- | --- |
| Qwen2-1.5Moe.Q4_K_M | 0 | 1317.88 | 1320.14 | 102.59 |
| Qwen2-1.5Moe.Q4_K_M | 3 | 1254.11 | 1272.76 | 102.92 |
| Qwen3-30B-A3B Q2_K | 0 | 279.54 | 157.56 +/- 67.62 | 54.48 |
| Qwen3-30B-A3B Q2_K | 3 | 506.50 | 530.43 | 54.59 |

Both Qwen2 single-turn CLI runs (MMID=0 and 3, greedy seed 42) produced "The capital of France is Paris." Pinned-upload fallback and one-slot configurations also completed pp128 successfully. The initial smoke harness used a CLI flag now restricted to llama-completion; it was corrected to --single-turn before the successful run.

Checkpoint serialization checks passed using the tiny Qwen3.5 hybrid fixture: two checkpoints saved, byte counts verified, suffix replay processed 36 tokens before and after restoration, and oversized/truncated/corrupt-state variants all fell back to full 160-token prefill without crashing or changing output. The fixture caps its context at 256, so the test uses a 160-token prompt. No draft-model checkpoint compatibility test was run.

Final preservation checks: the index is clean, HEAD matches the isolated integration tree, and applying the saved original patch to HEAD reproduces the working tree exactly (six files, 169 additions, 5 deletions). All original untracked paths still exist. The backup branch and saved stash remain available. The tested binaries identify the pre-amend HEAD `9f95dfceb`; the amend only recorded already-tested source edits.
