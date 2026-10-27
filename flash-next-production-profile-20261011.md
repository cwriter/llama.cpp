# Flash-Next, three B60 GPUs: production profiling, 2026-10-11

Both Linux perf and Intel VTune captures completed. The largest demonstrated prefill loss is server checkpoint handling. Long-context attention also blocks graph replay, increasing submission work. Grouped expert compute dominates prefill GPU time; dense Q8 projections are the largest decode kernel family. MTP substantially improves decode throughput in this workload.

## Configuration and method

- Branch: `performance_uplift_mega_branch`, HEAD `80a9ee4bf`. The six inherited modified source files were retained. No source changes were made during profiling. Saved source diff SHA256: `f62772547ed5b3f13ba0d82d512929601e599875071152e37f49162eb73ca744`.
- Rebuilt `/tmp/pr30156-mega-build` with oneAPI 2026.1, Release, SYCL, F16 and oneDNN enabled, native O3 and host profiling line information. Device compilation retained release settings. Vulkan was disabled in this build.
- Three Intel Arc Pro B60 GPUs, layer split `34,34,32`, full GPU offload, no CPU experts. Model: unsloth Qwen3.8-Flash-Next UD-IQ4_XS, cached revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`. 48 layers, 512 experts, 10 active, expert FFN width 640. CPU mmproj retained from production launcher.
- Production settings from `/root/run_llama_flash_next.sh`: context 131072, batch 4096, ubatch 1024, Q8_0 KV, flash attention, one slot, threads 4/16, pipeline enabled, prompt cache RAM 8192 MiB, context checkpoints 16. MTP Q8 sidecar on GPU2, maximum two draft tokens, Q8_0 draft KV.
- Graphs enabled, Level Zero V2, MMID schedule 229377, USM system 2, VMM 0, dynamic F16, QSA FA no-readback, decode fusion 31, async copy 7 and device-to-device memcpy enabled. Hostmap was disabled. The older VTune launcher was not used because its settings differ from production.
- Deterministic constructed technical prompts, exact token lengths, 256 generated tokens, production sampling (temperature 1, top-k 20, top-p .95), seed 42, backend sampling enabled. Three unprofiled repetitions per timing point. Cold model load, compilation and a 16k warm-up were excluded. These are representative inputs, not a replay of live production traffic.

## Unprofiled throughput

Median tokens/s. Decode includes MTP unless explicitly disabled.

| Prompt tokens | Production prefill | Production decode |
| ---: | ---: | ---: |
| 128 | 225.27 | 56.84 |
| 2048 | 662.85 | 59.78 |
| 16384 | 1484.23 | 56.40 |
| 32768 | 1520.78 | 52.32 |

| Control | 2k prefill | 2k decode | 32k prefill | 32k decode |
| --- | ---: | ---: | ---: | ---: |
| Production | 662.85 | 59.78 | 1520.78 | 52.32 |
| Graphs disabled | 663.46 | 56.38 | 1506.07 | 47.45 |
| Pipeline ring depth 4 | 661.94 | 59.17 | 1516.10 | 54.60 |
| Context checkpoints disabled | 1006.17 | 56.77 | 1603.37 | 51.34 |
| MTP disabled | 704.83 | 43.65 | 1553.73 | 26.14 |

MTP provides approximately 1.37x decode throughput at 2k and 2.00x at 32k in these tests. Turning it off modestly improves prefill. MTP acceptance varies even with the same seed, and some runs have long stalls. Small decode differences between graph/ring/checkpoint controls are not reliable causal speedups. Raw samples and acceptance rates are in `timing-summary.json`.

## Gaps, in priority order

### 1. Checkpoint handling costs about one second per prompt

Disabling context checkpoints improves 2k prefill by 51.8% (about 3.09 s to 2.04 s), and 32k prefill by 5.4% (about 21.55 s to 20.44 s).

`tools/server/server-context.cpp:4057` deliberately splits processing at offsets `4 + n_ubatch` and `4` before the prompt end. At 2k this produces batches of 1020, 1024 and 4 tokens. `create_checkpoint()` at line 2547 first calls `spec_catch_up(false)` and then saves target and draft state. Logs show two checkpoints per tested prompt, with roughly 112.6 MiB of state per checkpoint. This combines pipeline drains, draft catch-up, host state serialization and a small final decode. The control measures their combined cost; it does not isolate the transfers alone.

A useful improvement would preserve resume semantics while taking checkpoints at natural processing boundaries and retaining state on the device until host serialization is needed. Disabling checkpoints globally changes recurrent prompt-prefix reuse and is only a diagnostic control here.

### 2. Attention's staging cap disables graph replay

The timeline confirms target graph replay at 2k: median host graph-compute calls are approximately 57 us on GPU0 and 45 us on GPU1. At 32k, target attention selects the library path on all three GPUs; graph capture is rejected and median host calls increase to about 9.5 ms and 8.6 ms. The MTP draft attention uses the library path even at 2k.

`ggml/src/ggml-sycl/fattn.cpp:177` checks the F16 staging size against the memory cap using the configured cache allocation. `ggml_sycl_fattn_capped_tile()` at line 207 allows a bounded native TILE path only when its layout fits the reserved budget. Otherwise line 245 selects chunked oneMKL, including short decode queries. The library path prevents capture because of external event dependencies. Thus an attention selection can force the rest of the GPU graph into individual submissions.

The next candidate is a bounded native attention path that reads quantized KV without staging the full cache and remains capturable. Its speedup has not been measured. Raising staging limits is unattractive with the current VRAM headroom. Host-call durations include waits and nested operations; they are not entirely recoverable overhead.

### 3. Grouped expert compute is the main prefill GPU cost

The clean VTune 32k prefill window attributes 46.6% of summed kernel time to grouped expert compute, or 48.4% including its packing and scheduling. The IQ3_S grouped kernel alone contributes about 30%; IQ4_NL grouped compute about 13%. Grouped input packing is only about 1.4%.

Other prefill costs include library GEMMs (15.1%, a mix of dense and attention work), recurrent GDN/SSM (7.0%), direct attention kernels (5.8%, excluding library GEMMs), standalone dequantization (4.5%) and explicit memory commands (4.6%). These are summed device durations, not additive wall-time fractions across three GPUs.

The relevant implementation is in `ggml/src/ggml-sycl/fused-gemm.cpp`, including the grouped kernel near line 2099 and device scheduler near line 2898. At ubatch 1024, uniform routing would average 20 tokens per expert; actual skew matters. Adaptive handling of heavily used experts and dequantization reuse remain plausible experiments, not demonstrated wins. Optimizing scheduler packing alone would address a much smaller fraction.

### 4. Decode needs attention beyond the expert router

The production graph-enabled VTune decode capture at short context attributes 29.8% of summed device time to dense Q8 projections, 13.7% to IQ3_S expert gate/up, 9.3% to IQ4_NL expert down and 10.6% to Q6_K projections/output. Fused HC projections contribute 5.2% and recurrent GDN/SSM 4.5%. Dense projection and vocabulary-output work therefore deserve profiling alongside expert kernels.

The Q6_K single-column kernel has roughly twice the call count of its verification-width counterpart, consistent with two draft steps per verification. This is a call-count inference, not tensor-level attribution. MTP remains a net win despite its serial draft work.

### 5. Intermittent host stalls deserve a targeted follow-up

A traced 2k decode has three serial stalls of approximately 638, 649 and 635 ms in `MUL_MAT:indexer_k_raw` on the three GPUs, accounting for about 1.92 s. These are BF16 weights `[2560,128]` with a single-column result. That shape routes through native BF16 DMMV (`ggml/src/ggml-sycl/dmmv.cpp:3675`), not oneDNN.

Cold specialization/JIT is a leading hypothesis, but this capture does not prove it. A CPU stack capture covering the actual stall, followed by explicit warm-up of all MTP verification widths and short tails, would test it. Separate startup perf samples do show heavy IGC compiler activity; that is not proof of the decode-stall cause.

## What the data does not support

- No active GTT was observed in the production samples at 0.5 s intervals. Peak resident VRAM was approximately 24016, 23555 and 23924 MiB on GPU0/1/2, against 24480 MiB per card. This gives little headroom but no evidence of sustained paging. Very brief activity could evade sampling.
- Increasing the pipeline ring to four gives no prefill improvement and costs roughly 76 MiB more VRAM on each of GPU0/1. It is not an established fix.
- Host time in traced layer copies is small relative to the request: approximately 53.5 ms in the 5.09 s long-context decode interval and 300 ms in 22.67 s of prefill. Nested synchronization remains relevant, but raw layer-copy volume is not the demonstrated dominant loss.
- Linux perf shows about 41% of self CPU cycles in two Level Zero driver locations. Call stacks include queue finish, event waits, input staging, prompt state serialization and draft-state preparation. These are CPU-cycle shares, not wall-time savings. Long-context decode memcpy samples also include driver command-buffer movement, so they cannot all be attributed to model/KV transfers.

## Profiling limits and artifacts

VTune 2026.4 substantially changes timing. All throughput above comes from unprofiled runs. Some device counters report both 100% active and 100% stalled; those counters were not used to infer occupancy or bandwidth limits. The optional host memory-bandwidth driver is unsupported in this VM and was disabled. Generated text was coherent, but this is not a numerical accuracy or regression test. No model implementation was changed.

Artifacts are under `/tmp/flash-profile-20261011`:

- `timing-summary.json`, `memory-summary.json`, individual response JSONs and all server logs.
- `perf.data`, `perf-flat.txt`, `perf-stacks.txt`, `perf-analysis.json` (199 Hz cycles, DWARF stacks).
- `vtune-prod-decode`: clean graph-enabled production decode capture.
- `vtune-eager-prefill-clean`: clean graph-disabled capture for kernel attribution. Prefill report time filter `104.5:161.1`, steady decode `162:171` seconds. The earlier incomplete `vtune-eager-prefill` result was discarded.
- `vtune-prod-decode-kernels.tsv`, `vtune-clean-prefill-kernels.tsv`, `vtune-clean-decode-kernels.tsv`, and corresponding summary JSONs.
- `pipeline-trace.txt`, `trace-analysis.json`, `server-trace.log` for graph blockers and host spans. Spans nest; do not sum their categories as independent wall time.
- `run.sh`, `client.py`, `server-wrapper.py`, `monitor.py`, saved production launcher, source patch, commit, binary hashes, build log and CMake cache.

Example timing reproduction (requires GPU access and the cached model):

```sh
QUIET_WAIT_S=0 PROFILE_MODE=control PROFILE_LABEL=repeat PROFILE_GRAPH=1 /root/gpu-locks/gpu_run.sh /tmp/flash-profile-20261011/run.sh
# Add PROFILE_NO_MTP=1 for the MTP-off control.
# Add PROFILE_CHECKPOINTS=0 for the checkpoint control.
```

After sourcing `/opt/intel/oneapi/setvars.sh`, open the clean VTune result with `vtune-gui /tmp/flash-profile-20261011/vtune-prod-decode`. Kernel tables can be regenerated with `vtune -report hotspots -group-by computing-task -r RESULT -format csv -csv-delimiter tab`.

The production launcher and inherited source changes were left unchanged. No commits or pushes were made.
