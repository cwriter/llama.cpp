# Persistent KV rotation extraction - private review notes

This is a local review record, not a PR description.

Branch: `feat/persistent-kv-rotation`, based on master `781dbc5ac`.
Worktree: `/root/llama.cpp/.worktrees/persistent-kv-rotation`.
Source: mega commit `706b2e41a`. Changes are staged, without a new commit or push.

## Scope

Seven source files, 189 additions and 87 deletions. Store immutable F32 K/V Hadamard matrices in backend buffers owned by the KV cache. Share each matrix pair per buffer type and select the resident copy for each layer. The graph no longer fills those matrices as normal attention inputs. Matrix application and quantization behavior stay the same.

Retain separate storage because cache clear/state restore clears the mutable KV buffers. Include rotation buffers in memory_breakdown and support metadata-only allocations. DSA/DSV4 specialized graph input paths and K-shift input paths remain as on master.

Extraction conflict: exclude Qwen4's mega-only build_layer_kv method. Update only build_attn_qsa, which exists on master. No allocator/fusion changes, upload-ring changes, MTP scheduling, SYCL-specific code, new public API, CLI options, or analysis documents are included in the staged diff.

## Review

Reviewed scope, security, ownership/lifetime, and general conventions using skills/code-review/SKILL.md. Tensors and buffers are owned for the KV cache lifetime; layer references are non-owning pointers to that storage. Existing Hadamard dimensions and buffer-type selection are reused. Existing upstream source had an explicit TODO to keep backend copies and avoid repeated transfers. The earlier upstream comparison and additional searches found no exact persistent-KV-matrix PR, but searches do not guarantee absence of overlapping work.

Before submission, the contributor should review every line, discuss upstream scope if needed, and disclose AI assistance. No PR text, commit message, or reviewer response has been drafted.

## Validation

Logs and external harness sources: `/tmp/kv-rotation-extract/`.

- CPU Release build: test-save-load-state, test-state-restore-fragmented, test-llama-archs and llama-bench passed.
- SYCL Release build: test-save-load-state, test-state-restore-fragmented and llama-bench passed; oneAPI 2026.1, F16 on, oneDNN off.
- Project-generated tiny Llama fixture, F16 and Q8_0 KV: CPU logits are bit-identical to master across 33-token prefill, eight single-token decodes, state save/restore, clear(true), repeated prefill and context shift. 1,664 logits per cache configuration.
- Internal persistence checks: shared tensor pointers across layers, no INPUT flag, unchanged matrix bytes across all these operations.
- Existing focused state tests passed on CPU: state blob round trip, malformed-state fallback, rotation-compatible restore and rotation-mismatch rejection. The malformed-state test disables flash attention itself, so it uses F16 V rather than unsupported quantized V without FA.
- Fragmented-cache Q8_0 restore passed on CPU.
- Metadata-only allocation with LLAMA_LOAD_MODE_NONE passed; rotation tensors have dummy buffers and memory accounting is available.

Known upstream test issue: the complete CPU test-save-load-state suite produces a device-copy mismatch in test 5 and crashes in device scatter-copy test 7. The identical invocation fails on unmodified master. GDB locates the crash in llama_io_read_device destruction, reached from test_seq_cp_scatter. It is outside this extraction. Selected existing host-state tests and the independent harness above avoid that failing path; the full suite is not claimed to pass.

B60 correctness checks passed for F16 and Q8_0 KV: master and feature produced bit-identical logits in all 1,664 compared values per configuration. The same internal pointer/byte persistence assertions passed on GPU. For Q8_0 KV, graph input tensor count drops from 11 on master to 9 on the extraction, while graph node count remains 123. Focused state and fragmented-cache tests also passed on the B60. Master baseline uses master libllama/libllama-common with the same unmodified master-source SYCL ggml libraries used by the extracted branch. This isolates the core change, although master core is built with GCC and feature GPU core with icpx. CPU comparison uses GCC for both.

## Review command

```sh
git -C /root/llama.cpp/.worktrees/persistent-kv-rotation diff --cached origin/master
```

## B60 benchmark smoke results

Qwen2-1.5Moe.Q4_K_M, full offload, batch 512, ubatch 128, flash attention on, 8 CPU threads, 3 repetitions, runtime SYCL graph setting left at its default.

| KV | Core | pp128 tokens/s | tg64 tokens/s |
| --- | --- | --- | --- |
| Q8_0 / Q8_0 | master | 1289.09 +/- 4.12 | 98.53 +/- 0.04 |
| Q8_0 / Q8_0 | persistent rotations | 1281.82 +/- 9.47 | 98.48 +/- 0.66 |
| F16 / F16 | master | 1275.94 +/- 3.69 | 100.11 +/- 0.06 |
| F16 / F16 | persistent rotations | 1270.63 +/- 16.22 | 100.14 +/- 0.01 |

No measurable throughput uplift on this model/device. Quantized-KV prefill differs by -0.56% and decode by -0.05%, within the observed run spread. F16 is a rotation-disabled control. Removing invariant graph inputs is verified structurally, but these timings do not establish an end-to-end speedup. More devices/workloads are needed before making a performance claim.

SYCL graph-enabled runs also passed for both F16 and Q8_0 KV, with bit-identical master/feature logits (1,664 each) through the same decode, clear, restore and context-shift sequence. Runtime GGML_SYCL_ENABLE_GRAPH was explicitly set to 1. Multi-GPU placement, other GPU backends, and the specialized DSA/DSV4 paths were not exercised.

Final scope check: seven staged source files, no unstaged edits in the extraction worktree, no pending cherry-pick, no commit/push/PR creation. Root remains on performance_uplift_mega_branch at 27c7f106d with its six original edited files unchanged.

## Additional quantized-KV measurements

Qwen3-30B-A3B-Q2_K.gguf, Q8_0 K and V, flash attention on, one B60, batch 2048, ubatch 512, GGML_SYCL_ENABLE_GRAPH=1, three repetitions. This is a comparison of the isolated extraction against master, not the mega branch.

| Core | pp128 tokens/s | pp2048 tokens/s | tg128 tokens/s |
| --- | --- | --- | --- |
| master | 230.86 +/- 5.70 | 798.14 +/- 3.69 | 54.71 +/- 0.03 |
| persistent rotations | 231.91 +/- 4.96 | 797.59 +/- 2.88 | 54.92 +/- 0.05 |

Prefill is essentially flat (-0.07% at pp2048); decode changes by +0.38% in this run. A single sequential A/B with three repetitions does not establish a robust gain this small.

Flash-Next uses the cached full UD-IQ4_XS model, Q8_0 K/V, three B60s, split 34/34/32, host-mapped PLE with lazy auto, no MTP, ubatch 512 and graph setting 1. Initial attempts used comma-separated device/split syntax copied from the server launcher; llama-bench treats commas as separate configurations, causing a single-card OOM before inference. Corrected slash-separated syntax is confirmed by all three device assignments and model allocations of 21724/21128/18370 MiB. Those failed attempts are not performance results. Logs are in /tmp/kv-rotation-extract/flash-a3b/.

## Flash-Next repeat with matched compiler and reversed order

Both core libraries were built with GCC, identical Release flags; both use the same SYCL backend. Feature ran first, master second, three repetitions each. Quantized rotation is confirmed in logs: n_rot_k=256 and n_rot_v=64. Master explicitly disables SYCL graph capture for multiple devices; both arms therefore execute eagerly despite GGML_SYCL_ENABLE_GRAPH=1. A3B also logs graph capture disabled for unsupported MUL_MAT_ID. These flags must not be interpreted as evidence of capture/replay for either full-model benchmark.

| Core | pp128 tokens/s | pp2048 tokens/s | tg128 tokens/s |
| --- | --- | --- | --- |
| master (warm, GCC) | 276.03 +/- 6.08 | 456.03 +/- 0.49 | 21.78 +/- 0.06 |
| persistent rotations (GCC) | 275.26 +/- 3.90 | 454.27 +/- 0.94 | 21.89 +/- 0.04 |

Changes: pp128 -0.28%, pp2048 -0.39%, decode +0.51%. This does not show a convincing throughput uplift. The initial sequential run had master pp128/pp2048/tg128=241.17/417.31/21.64 and feature=272.90/454.28/21.88; its apparent prefill gain disappeared when master was rerun warm. Do not claim the initial +8.9% pp2048 as an optimization benefit. The repeat also removes the GCC/icpx core compiler difference from the first pass.

## Full-model quantized-KV correctness follow-up

Both full models were checked with Q8_0 K/V using matching GCC core builds and the same SYCL backend. The harness exercises prefill, single-token decode, save/restore, clear and repeated prefill; A3B also exercises context shift. Flash-Next does not support context shift in this configuration.

- A3B: all 1,975,168 compared logits are bit-identical between master and the extraction.
- Flash-Next: all 2,979,840 logits are finite and the highest-logit token matches at every checked step, but outputs differ: NMSE 0.0010082077, maximum absolute difference 0.84538436.
- A separate unchanged-master repeat is bit-identical to the first master run. The Flash-Next discrepancy therefore cannot be dismissed as variation between those master runs.

The Flash-Next discrepancy is unresolved. The extraction is not validated as regression-free or ready for submission. The performance measurements above remain measurements of the current extraction, not evidence of equivalent numerical behavior. Files: /tmp/kv-rotation-extract/flash-a3b/check-*.bin and corresponding logs.

## Flash-Next discrepancy investigation

The discrepancy is isolated to the existing SYCL MoE router fusion, whose eligibility changes with scratch allocation layout:

- Adding llama_synchronize immediately after context creation leaves feature logits bit-identical to the original feature run. It does not change the discrepancy against master.
- Disabling all SYCL fusion makes all 2,979,840 master/feature logits bit-identical.
- Disabling only ggml_sycl_fuse (the MoE router fusion entry), while retaining other fusions, also makes all 2,979,840 logits bit-identical.
- Layer-boundary callbacks preserve the original master and feature outputs exactly. The first differing layer output in the first 16-token microbatch is l_last-15.
- The first differing router fusion decision is also layer 15: master does not fuse, extraction fuses. Master overlaps the normalized-weight output allocation with the router logits allocation (logits start 128 bytes before weights). The existing fusion memory-range guard rejects that overlap.
- In the ordinary per-node callback trace, rotation matrix hashes match, and all final logits match between branches. Intermediate QSA TOP_K index ordering can differ without changing the final selection mask or attention output in that trace.

This identifies the kernel-selection cause; there is no evidence of a missing rotation upload barrier. SYCL buffer uploads and inference use the same per-device default in-order queue. Fused and unfused router softmax implementations use different exponential operations and reduction orders. Numerical sensitivity through the full model still needs to be distinguished from a quality regression; matching greedy tokens is not a quality assessment.

All diagnostics are private files under /tmp/kv-rotation-extract/trace. No repository source was changed for this investigation. The extraction remains staged and uncommitted.

Focused layer-15 captures confirm the initial difference is floating-point rounding: all 8,192 router input logits and all 160 selected expert IDs are bit-identical. The 160 normalized routing weights differ by at most 2.98023224e-8, NMSE 2.72341840e-15. The layer-15 output then differs by at most 1.49011612e-8, NMSE 2.21733862e-16. Larger final-logit differences develop downstream from this small difference. This is not evidence of a missing synchronization; it is an allocation-dependent choice between existing fused and unfused arithmetic.
