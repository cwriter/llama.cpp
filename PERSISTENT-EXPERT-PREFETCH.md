# Persistent expert prefetch experiment

2026-10-31. CLOSED: unsuccessful in the tested B60 persistent-kernel microbenchmarks. Keep disabled; no further work queued. The generic implementation remains uncommitted on `feat/sycl-expert-prefetch`, based on `performance_uplift_mega_branch` at `79a6fac43`, in `.worktrees/sycl-expert-prefetch`. Production source, libraries, launchers and defaults were not changed by this experiment. All three GPUs were available for validation; the earlier compile-only investigation incorrectly reported that no GPU was exposed. Full-model host-resident PLE performance was not measured.

## Runtime decision

Archive with the unsuccessful performance experiments. All active IQ3_S strategies regressed latency by about 2.7-3.9x; active IQ4_NL next-task strategies regressed latency by about 12-68%. IQ4_NL next-block was inactive because that kernel consumes the whole row in one stripe. Correctness passed: 1,928 CPU-reference cases plus 192 persistent bitwise comparisons. Detailed runtime results follow the historical compile-only findings below.

## Initial compile-only findings (superseded by runtime measurements)

Exact persistent-expert lookahead is implementable without a new allocation, barrier or reported register spill. The best first runtime candidate is IQ3_S next-block-group prefetch. It has the lowest compiled instruction overhead and avoids looking up another expert. The next candidates are first-block prefetch for the next assigned IQ3_S task and full-qs prefetch for the next IQ4_NL row pair. None is recommended for default enablement without timing and numerical checks.

The important launch detail is that the next task is usually in another selected expert, not the next contiguous row. At the default B60 geometry, 1280 threads traverse 6400 IQ3_S rows: each thread advances by two experts across five tasks. IQ4_NL R=2 traverses 12800 row pairs: each thread advances by one expert across ten tasks. The router has already produced those IDs, so these are exact fetches with no token or expert prediction.

## Variants built

The probe extracts the current arithmetic helpers and persistent kernel bodies from `dmmv.cpp`. Prefetch modes are compile-time template parameters, with separate generated kernels. It does not incorporate the separate archived fast-division experiment or change weight layouts, arithmetic, routing, work-group sizes or allocation policy.

| Mode | Change |
| --- | --- |
| 0 | Copied current kernel, no prefetch |
| 1 | First 64 qs bytes of each row in the next assigned task |
| 2 | Mode 1 plus metadata-plane heads: IQ3 qh/signs/scale, IQ4 scale |
| 3 | Full qs payload of each row in the next assigned task, without other planes |
| 4 | IQ3 only: first 64 qs bytes of the next five-block group while computing the current group |

Both gate and up are prefetched in IQ3. IQ4 prefetches the next row pair, with an explicit odd-row guard. Next-task fetches guard the final task and negative expert IDs. Metadata-word addresses are rounded down to four-byte alignment within the relevant metadata plane. Hints target L2 with L1 uncached, so they do not depend on a later task running on the same L1 cache.

## AOT and native-code checks

oneAPI 2026.2 AOT compilation for `bmg-g21` passed for all nine kernels. Compiler options match the relevant archived production AOT settings: O3, F16, 16-wide SYCL subgroup setting and automatic large-GRF mode. Native disassembly was produced with `ocloc disasm`.

| Kernel | Mode | Static instructions | Change from copied baseline | Static prefetch sites |
| --- | ---: | ---: | ---: | ---: |
| IQ3_S gate/up | 0 | 1092 | baseline | 0 |
| IQ3_S gate/up | 1 | 1138 | +4.21% | 2 |
| IQ3_S gate/up | 2 | 1164 | +6.59% | 8 |
| IQ3_S gate/up | 3 | 1211 | +10.90% | 4 |
| IQ3_S gate/up | 4 | 1117 | +2.29% | 2 |
| IQ4_NL down | 0 | 913 | baseline | 0 |
| IQ4_NL down | 1 | 962 | +5.37% | 2 |
| IQ4_NL down | 2 | 982 | +7.56% | 4 |
| IQ4_NL down | 3 | 966 | +5.81% | 4 |

Every kernel retains a 128-GRF allocation with no reported per-thread scratch buffers. IQ3 retains 7168 bytes of SLM and one setup barrier; IQ4 retains zero SLM and zero barriers. This establishes unchanged allocation classes for these compiled variants, not identical live-register counts or measured occupancy. Counts include helper code and are static instruction sites, not dynamic executed-instruction counts. Runtime loops can issue the same prefetch site multiple times.

Hints lower to `send.ugm` loads with a null destination, zero response registers and `.uc.ca` policy. In the actual extracted expert kernels, they appear before the current work they are intended to overlap. IQ3 mode 4 issues its hints before the current group's SLM loads and dequantization/dot work, with the hinted payload needed by the following group. Next-task hints appear before current-task computation, with demand reads of the hinted data in a subsequent persistent iteration. Native placement therefore supplies a real opportunity for lead time; its sufficiency in cycles is unmeasured.

## Costs and reasons this could lose

Mode 4 fetches only one of the five future qs blocks for each weight. It minimizes hint/address overhead but may hide too little latency. Mode 1 similarly prefetches only the row head, while the later demand loads cover much more data.

IQ3 next-task modes add another expert/row mapping and expert-ID load, including generic integer division. Mode 3 adds payload traversal and the largest IQ3 code increase. Combining division removal with prefetch would obscure attribution, so this probe deliberately preserves the original mapping.

The default IQ3 task reads 2200 stored weight bytes across gate/up. Its head-only next-task hints request 128 bytes per future task; full-qs hints request 1280 bytes. An IQ4 row-pair task reads 720 stored bytes; head-only hints request 128 bytes and full-qs hints 640 bytes. These are requested payload bytes, not DRAM transaction counts. Cache-line granularity and reuse change actual traffic.

For 1280 active threads, one-task lookahead has about 160 KiB of requested head payload, versus about 1.56 MiB of full IQ3 qs payload or 800 KiB of full IQ4 qs payload. Actual cache residency and in-flight reads differ. These hints add no explicit allocation, but do consume cache space, memory messages and bandwidth.

Archived `/tmp/moe-div/report.md` measured original AOT kernels at 31.65/41.75 us for IQ3 fixed/rotating routing and 26.16/28.84 us for IQ4. Dividing stored weight bytes by those times gives effective rates around 445/337 and 352/320 GB/s, respectively. Those are not measured DRAM bandwidth: timings include host dispatch and cache hits can serve some bytes. They are historical reference points, not prefetch results or a new baseline. Any benefit may depend strongly on routing reuse.

## Validation completed and pending

- All nine AOT kernels compile; all native images were disassembled and checked for hints and resource metadata.
- The independent host address model passed 96 shape/launch scenarios, checking 2,095,168 next-task hint ranges and 30,720 block-hint ranges. It covers normal/odd row counts, 1/7/10/17 selected experts, four launch strides, final tasks and negative IDs. This validates the modeled address rules against plane intervals; it does not execute or validate GPU address calculations.
- The GPU benchmark executable compiles and links. It prepares 288 comparisons against installed original kernels across normal/odd row counts, all modes, one/two tokens, skipped IDs and whole-buffer guards. These comparisons have not run.
- The benchmark prepares ten ABBA timing rounds per candidate for fixed and rotating expert sets, with 100 warmup launches per mode and 300 calls per timed sample. It compares each candidate against the copied baseline in the same DSO. These timings have not run.
- GPU numerical checks, observed memory transactions, latency and full-model three-slot/long-context measurements remain required. No tokens/s gain is claimed.

## Artifacts and reproduction

Directory: `/tmp/expert-prefetch`.

- `make-probe.py`, `kernels.cpp`, `manifest.json`: extraction, variants and source hashes.
- `build.sh`, `build.log`, `libexpert-prefetch.so`: isolated AOT library and build output.
- `analyze.py`, `isa-stats.json`, `images/`: native resource summaries and disassembly.
- `check-addresses.py`, `address-check.json`: independent address model results.
- `make-benchmark.py`, `benchmark.cpp`, `build-benchmark.sh`, `benchmark-build.log`, `benchmark`: prepared numerical checks and timing harness, adapted from the existing `/tmp/moe-div/probe.cpp`.
- `env.sh`, `run-benchmark.sh`: local dependency environment and runner. The model tensor manifest remains `/tmp/moe-div/tensors.json` and points to the cached real model weights.

Rebuild the probe and benchmark with the existing shared-machine build lock:

```sh
python3 /tmp/expert-prefetch/make-probe.py
bash /root/gpu-locks/build_run.sh bash /tmp/expert-prefetch/build.sh
python3 /tmp/expert-prefetch/analyze.py
python3 /tmp/expert-prefetch/check-addresses.py
python3 /tmp/expert-prefetch/make-benchmark.py
bash /root/gpu-locks/build_run.sh bash /tmp/expert-prefetch/build-benchmark.sh
```

On a GPU-enabled session, the prepared benchmark can use the existing exclusive GPU lock. That wrapper aborts if a foreign server is active; it does not stop services:

```sh
QUIET_WAIT_S=0 bash /root/gpu-locks/gpu_run.sh /tmp/expert-prefetch/run-benchmark.sh
```

No production integration is justified yet. The next decision is whether IQ3 mode 4 and IQ4 mode 3 win in rotating-routing GPU timings while preserving numerical results. Compare the lighter next-task modes as controls. Only successful candidates should proceed to full-model measurements.


## Runtime validation and timings

Branch: `feat/sycl-expert-prefetch`, based on `performance_uplift_mega_branch` at `79a6fac43`. Implementation is uncommitted in `/root/llama.cpp/.worktrees/sycl-expert-prefetch`.

### Build and range checks

oneAPI 2026.2 Release JIT build, F16, XMX subgroup 16 for B60: ggml-sycl, test-backend-ops and test-sycl-prefetch built successfully. Host CTest passed: all 15 formats, canonical and reordered stream layouts, bounds, tails, alignment and mask mapping. Full bmg-g21 AOT link was cancelled because compiling all unrelated attention specializations was slow; do not interpret this as an AOT build result.

### Persistent-expert comparisons

B60 physical card 0, Level Zero V2, default persistent IQ3_S fused gate/up and IQ4_NL down readers, device weights. Compared modes 1/2/3 to disabled mode 0 using the same backend. All 192 bitwise comparisons passed with output guard regions: 96 normal sizes and 96 odd-row sizes, selected expert counts 1/7/10/17, one/two tokens, positive and skipped expert IDs. These are comparisons to the existing kernel, not an independent CPU numerical reference.

### Persistent-expert timings

Kernel microbenchmarks, not full-model decoding or host-offload bandwidth measurements. Normal shapes: IQ3_S M640 K2560 fused gate/up, IQ4_NL M2560 K640 down, 10 experts, one token. Fixed routing reuses selected experts; rotating routing traverses 64 ID sets across 512 experts. Warmup excluded; ten ABBA rounds with 300 queued calls per sample. Median paired percent latency reduction below: positive is faster, negative is slower. Build/timing and per-card GPU locks were used. Other-card XMX correctness jobs were compiling concurrently, so shared host activity was not fully idle.

| Kernel | Routing | Mode | Baseline us | Prefetch us | Median paired latency reduction | Range across rounds |
|---|---|---:|---:|---:|---:|---:|
| IQ3_S gate/up | fixed | 1 | 32.52 | 125.15 | -284.85% | -285.32% to -284.33% |
| IQ3_S gate/up | fixed | 2 | 32.52 | 108.13 | -232.43% | -232.72% to -232.19% |
| IQ3_S gate/up | fixed | 3 | 32.53 | 122.84 | -277.74% | -277.92% to -277.01% |
| IQ3_S gate/up | rotating | 1 | 43.26 | 133.31 | -208.30% | -211.39% to -205.70% |
| IQ3_S gate/up | rotating | 2 | 43.19 | 116.29 | -169.26% | -170.80% to -165.99% |
| IQ3_S gate/up | rotating | 3 | 43.02 | 129.27 | -200.50% | -201.26% to -197.30% |
| IQ4_NL down | fixed | 1 | 27.21 | 27.21 | +0.02% | -0.09% to +0.14% |
| IQ4_NL down | fixed | 2 | 27.20 | 31.71 | -16.52% | -16.74% to -16.42% |
| IQ4_NL down | fixed | 3 | 27.21 | 45.68 | -67.82% | -67.96% to -67.69% |
| IQ4_NL down | rotating | 1 | 32.11 | 31.99 | +0.49% | -2.25% to +2.51% |
| IQ4_NL down | rotating | 2 | 31.83 | 35.41 | -11.62% | -12.03% to -9.08% |
| IQ4_NL down | rotating | 3 | 32.17 | 47.74 | -49.16% | -52.03% to -46.85% |

Activation detail checked against the kernel mapping: B60 has 160 compute units, hence 1,280 persistent threads per token. IQ3_S timings have 6,400 row tasks (five per thread) and ten blocks per row consumed in groups of five: next-block has a following group, and next-task has four following iterations per thread. IQ4_NL timings have 12,800 row-pair tasks (ten per thread), so next-task modes have following iterations. IQ4_NL consumes all 20 blocks in one stripe, so mode 1 issues no next-block hints; its near-neutral table entries are a no-op control, not a measurement of active prefetch.

There is no useful improvement among the active prefetch cases in this matrix. Keep the feature disabled by default. These results do not establish behavior for other formats, kernel variants or host-resident PLE weights.

### Broad GPU checks

784/784 decode CPU-reference checks passed: all 15 formats x one/eight tokens x dense/routed/skipped-ID cases, plus existing matching routed fixtures; 98 cases per run x four modes (baseline 0 and 1/2/3) x reorder off/on. XMX/ESIMD dispatch was disabled to exercise generic fallback readers. The prefetch mask was 32767 even for baseline mode 0. Driver persistent disk caching was disabled for the passing matrix.

Earlier attempts with driver disk caching enabled crashed in mode 0; the full 98-case baseline also passed under gdb without caching. This does not establish the root cause. A 32-token fallback attempt also crashed with mode 0 and is not part of the passing matrix.

1,144/1,144 default-dispatch GPU CPU-reference checks passed: 143 cases per run x four modes x reorder off/on, all 15 formats, one/eight/32 tokens, dense and routed including skipped IDs. Combined with forced fallback: 1,928/1,928 CPU-reference cases, plus 192/192 persistent bitwise comparisons. All three B60 GPUs were used for validation.

First-use XMX compilation took approximately fifteen minutes; later processes reused the compiled driver images and completed quickly. The optional per-kernel relink was cancelled after the original matrix progressed; its output was not used. The intentionally cancelled first mode-3 attempt reported an IGC termination request; the restarted run passed both layouts. No backend code changes were made during these GPU checks.

The coverage exercises default-selected XMX shapes and decode paths; it does not force every XMX shape/staging combination, every ESIMD override, graph replay, or host-resident PLE allocation. Persistent timings use the original completed JIT build; no per-kernel-relinked or AOT library was used.
