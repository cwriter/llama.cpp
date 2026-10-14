# Gate-view fusion experiment

Branch `feat/sycl-gate-view-fusion`, worktree `/root/llama.cpp/.worktrees/gate-view-fusion`, based on mega branch f601e3725. Enabled by default and retained for integration into the mega branch at the user's request.

## Implementation

`GGML_SYCL_FUSE_GATE_VIEW=1` removes an exclusive F32 CONT whose sole consumer is a unary operation feeding an adjacent fused MUL. It reuses the existing strided unary/multiply kernel and reads the original VIEW. The source rows must be contiguous, with a regular row pitch over the outer dimensions. The other MUL operand and output must be contiguous. Shared or graph-output CONT tensors and in-place unary/multiply outputs retain their old paths.

The copy can occur before unrelated attention work. The backend skips it there, and executes the existing arithmetic fusion when its inputs are ready. Its allocation hook marks the copy absorbed and explicitly keeps the original view's base alive through the multiply. The shared Q projection's layout is unchanged. There are no downcasts or arithmetic reassociations.

This targets the 12 attention gate materializations in Qwen4exp. The sigmoid/multiply pair was already fused; this experiment extends that existing fusion. The 36 recurrent convolution copy chains are already fused and were not changed. Pooling and PLE are separate remaining candidates, not implemented here.

## Validation

Seven whole-graph CPU-reference cases passed with the feature off and on. Cases cover the model shape, token counts 1/2/17/257, odd width 31, a shared producer, a CONT with another consumer, and a CONT marked as a graph output.

A separate scheduler harness ran 12 shapes/guard combinations eight times each, updating deterministic inputs between runs. All 96 runs per arm passed an independent reference. Serialized outputs were byte-identical between off and on. Eligible CONT buffers were null after allocation with the feature on; shared/output copies remained allocated. These checks exercise the actual scheduler allocation plan, not only directly allocated tensors.

In that harness the total arena size was unchanged. Removing an intermediate does not guarantee lower peak memory: the larger shared producer must stay alive longer. Production compute-buffer reservations are recorded separately by the end-to-end run.

The initial scheduler harness omitted the required CPU fallback backend and aborted at scheduler creation with the feature off. That setup error was corrected before the final run; its log is retained as scheduler-harness-initial-error.log.

Build: an isolated object overlay compiled ggml-sycl.cpp, fusion.cpp, element_wise.cpp, and test-backend-ops.cpp, remapping include paths to this worktree, then linked the unchanged base objects from /tmp/q8-verification-tails-build. This is not a fresh full-tree build. Manifest: /tmp/gate-view-fusion-build/manifest.json. Build logs contain only the existing unused gate-parameter warning. The first build attempt exposed stale include paths from the base compile commands; those paths were corrected before the successful build.

## Isolated graph performance

One B60, Level Zero v2, SYCL graphs enabled. Two samples per arm, order off/on/on/off. Includes graph and host overhead. Median microseconds per whole test graph:

| Shape and guard | Existing | Gate-view fusion |
|---|---:|---:|
| width=256,heads=24,tokens=1,mode=0 | 11.54 | 9.05 |
| width=256,heads=24,tokens=2,mode=0 | 12.75 | 9.97 |
| width=256,heads=24,tokens=17,mode=0 | 37.04 | 26.86 |
| width=256,heads=24,tokens=257,mode=0 | 493.26 | 379.62 |
| width=31,heads=3,tokens=2,mode=0 | 12.05 | 9.37 |
| width=31,heads=3,tokens=2,mode=1 | 15.27 | 15.43 |
| width=31,heads=3,tokens=2,mode=2 | 11.98 | 12.02 |

The guarded cases retain the original code; their small timing differences are noise. This microbenchmark includes a shared producer and Q-side consumer, not only the copy and arithmetic pair.

## Artifacts

Scripts, full test logs and deterministic output dumps are alongside this report. The implementation and measurements are retained for integration into the mega branch at the user's request. No push was made.

## End-to-end decode

Production model and flags, three B60 GPUs, layer split 34/34/32, Q8 KV, context capacity 131072, no MTP and backend sampling. Four independently started server sessions in baseline/feature/feature/baseline order. Each had an excluded 16k/64-token warmup, then three 2k requests and one 32k request, each generating 256 tokens. Same cached prompts, seeds and final library; only GGML_SYCL_FUSE_GATE_VIEW differed. Process maps verified the overlay library.

| Context | Existing tokens/s | Feature tokens/s | Change |
|---|---:|---:|---:|
| 2048 | 43.70 | 43.56 | -0.33% |
| 32768 | 25.71 | 25.50 | -0.84% |

Medians combine six requests per arm at 2k and two per arm at 32k. Per-session results:

| Session | Arm | 2k median | 32k |
|---|---|---:|---:|
| 0 | baseline | 43.78 | 25.93 |
| 1 | feature | 43.52 | 25.50 |
| 2 | feature | 43.60 | 25.50 |
| 3 | baseline | 43.69 | 25.49 |

There is no useful end-to-end speedup demonstrated here. Following these measurements, the user chose to retain the fusion and enable it by default; GGML_SYCL_FUSE_GATE_VIEW=0 restores the copy. The isolated graph gain must not be presented as a decode gain. No cause for the small end-to-end differences was isolated.

All four sessions reported the same compute-buffer reservations: SYCL0 917.33 MiB, SYCL1 1067.30 MiB, SYCL2 1067.31 MiB and SYCL_Host 1398.52 MiB. The feature removes an intermediate allocation, but total reserved compute memory did not fall.

Generated-text hashes differ even between the two baseline sessions with matching seeds, and also between the two feature sessions. This production sampling comparison therefore does not establish bytewise model-output equivalence. The deterministic scheduler/kernel checks did establish bytewise equality for their inputs. No perplexity evaluation was performed.

The four experiment servers exited. Full results and launch provenance are in /tmp/gate-view-fusion-e2e.

## Production dispatch verification

An additional debug-only run used the same production model/settings and feature library, a 2k prompt and four generated tokens. Startup warmup was disabled for this diagnostic; it is not a performance measurement. Runtime logs contain all 12 attention gate fused-consumer names and zero dispatched CONT operations named gate_reshaped. Thus the actual copy operations are eliminated, rather than merely visited and later skipped in an ambiguous census. Source views, Q-side copies and unrelated copies retain their existing behavior.

Artifacts: /tmp/gate-view-fusion-dispatch/0-audit, dispatch-summary.json and dispatch.py alongside this report. The first attempt was stopped by the GPU wrapper's active-server guard; no foreign process was touched. A read-only check subsequently found no running servers, and the retry succeeded. The diagnostic server exited normally.

The remaining pooling gap was assessed but not implemented in this trial. Its existing generic multi-add helper uses a balanced addition tree. Extending it directly would reassociate the original left-to-right pool sum; a separate ordered reduction is needed to preserve the current arithmetic.

## Default setting

The default was changed from 0 to 1 after validation, at the user's request. The previously tested explicit value 1 exercises the same path as the new unset-variable default. No kernel, matcher eligibility, allocation lifetime or arithmetic changed in this follow-up. The measured compute-buffer reservation remains unchanged; a general peak-memory reduction is not established.
