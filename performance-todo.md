# Performance TODO

## Quantized KV cache without full-size staging

Status: code-path investigation, Q8_0 XMX prototype, correctness testing, and initial performance testing complete. The prototype remains disabled because the standalone VKQ decomposition is much slower than staged oneMKL.

The current SYCL attention paths promote a quantized KV cache to a full-size type before it reaches oneMKL or oneDNN. The oneMKL path dequantizes K and V chunks into dense f16 buffers in `fattn-mkl.cpp`; the oneDNN path similarly prepares non-f16 KV data as f16 before SDPA. Some library primitives may require f16 or f32 inputs, but other implementations could promote values only as they are consumed, using SLM or registers. This could remove the global-memory write and reread of the expanded KV data.

Code investigation:

1. Trace every quantized-KV dispatch through `fattn.cpp`, `fattn-mkl.cpp`, and `fattn-onednn.cpp`, including contiguous and non-contiguous layouts.
2. Record which oneMKL and oneDNN primitives require dense f16/f32 operands and which paths could use a custom fused load/dequant stage.
3. Prototype promotion inside the consuming kernel, with only a tile in SLM or registers. Keep the existing library path as the fallback where native quantized input is not possible.
4. Check that K and V quantization scales, GQA layouts, padded sequence views, and Gemma interleaving remain correct.
5. Estimate eliminated global-memory traffic and scratch size from tensor shapes before running anything on a GPU.

When the GPUs are available, validate output correctness first, then measure kernel time, prompt throughput, decode throughput, peak scratch allocation, and device-memory traffic against the existing f16-staging paths.

### Code investigation findings

The existing paths divide into three cases:

- oneDNN is preferred for supported prefill shapes. Quantized K and V are fully expanded into separate graph-reserved f16 buffers on every attention call. The oneDNN SDPA graph itself declares f16 query, key, value, probabilities, and output tensors.
- The oneMKL fallback already bounds memory by dequantizing one KV chunk at a time, but each chunk is still written as dense f16 to global memory and then read by two GEMMs. Its K/V scratch is `2 * chunk_size * (DKQ + DV)` bytes.
- The TILE kernel requests full f16 K and V staging. The VEC decode kernel is the existing proof that online promotion works: it computes quantized K dot products directly and dequantizes V fragments into registers. It is selected only for a subset of short-query shapes; BMG quantized decode currently prefers TILE and therefore returns to full-cache staging.

For each K or V scalar, f16 staging adds at least one 2-byte global write and one 2-byte global read after the compressed source is read. Removing staging therefore eliminates about four bytes of global traffic per logical scalar, or about `4 * (nelements(K) + nelements(V))` bytes per attention call, before accounting for cache effects. It also removes the expanded K/V scratch reservation from the graph allocator.

oneMKL is not a direct solution. Its int8 GEMM accepts dense int8 matrices with a single affine offset and produces int32 accumulators. GGML Q8_0 has an f16 scale for every 32 values, and the scale must be applied to each partial dot product before the block results are combined. Q4/Q5 formats add packed fields and, for the `_1` formats, a per-block minimum. A normal oneMKL GEMM cannot express these layouts or blockwise scale application.

The installed oneDNN 3.11 headers expose s4, u4, s8, grouped `DynamicDequantize`, and separate runtime scale and zero-point tensors. The installed SDPA implementation also contains quantized-key/value machinery. It is not currently a usable integration route: the direct SDPA primitive is not part of the installed public C or C++ headers, while the public graph implementation reports that its fused SDPA path does not support quantized SDPA or dynamic dequantization. GGML blocks also interleave metadata with quant bytes, and Q4_0 stores the two halves of a 32-value block in opposite nibbles rather than the adjacent-value packing expected by a plain u4 tensor. Do not depend on the internal oneDNN SDPA symbol.

### Proposed first prototype

Start with Q8_0 and preserve the existing chunked oneMKL attention orchestration, mask handling, query tiling, and online-softmax state. Replace only the two library GEMMs and their K/V staging:

1. Add an XMX KQ kernel that reads Q8_0 K blocks, expands one matrix tile into SLM, and immediately consumes it with `joint_matrix` against the existing packed f16 Q tile. Write the same f32 KQ tile expected by the current softmax kernel.
2. Add an XMX VKQ kernel that reads Q8_0 V blocks, expands one tile into SLM, and immediately consumes it with the existing f16 probability tile. Write the same f32 `VKQ_chunk` output expected by the current accumulator.
3. Remove `K_chunk_f16` and `V_chunk_f16` only when both custom kernels accept the tensor layout. Retain the current dequantize plus oneMKL fallback for unsupported types, devices, layouts, or shapes.
4. Reuse `mkl_fa_kv_desc` for dense, non-contiguous, padded-sequence, and Gemma-interleaved addressing. Do not introduce a second layout interpretation.

A representative 16 x 32 output tile with K depth 32 needs about 1 KiB of f16 query data and 2 KiB of locally promoted K or V data, plus accumulators. The allocation is per work-group and independent of context length. Exact tile dimensions must follow device SLM limits and the supported `joint_matrix` combinations.

This two-kernel design is deliberately smaller than a new quantized FlashAttention implementation. It removes the global f16 K/V buffers while keeping the current softmax and accumulation code. A later iteration could fuse KQ, online softmax, and VKQ to remove the score intermediates, but that is a separate and much larger change.

### Type order

1. Q8_0 first: values are plain signed bytes plus one f16 scale per 32 values.
2. Q4_0 next: nibble unpacking and the implicit `-8` offset are straightforward in a custom loader.
3. Q4_1, Q5_0, and Q5_1 later: their minimum or high-bit metadata makes both loading and validation more complex.

### GPU phase

GPU testing was explicitly authorized. Keep the local-promotion changes uncommitted until the isolated KQ and VKQ checks pass, then measure the complete path against the baseline below.

### TILE and oneMKL implementation investigation

GPU work was authorized after the initial code-only investigation. Baseline measurements used an Intel Arc Pro B60, Q8_0 K and V, head size 256, 2 KV heads, GQA ratio 16, and a 20000-token KV length.

| Path | Query tokens | Time | Throughput |
|---|---:|---:|---:|
| TILE | 1 | 2209.92 us | 296.55 GFLOPS |
| TILE | 512 | 141618.62 us | 2.37 TFLOPS |
| oneMKL | 512 | 22581.73 us | 14.86 TFLOPS |

With `GGML_SYCL_MKL_FA_DRAIN=1`, warm oneMKL calls reported about 0.35 ms in Q8_0 conversion, 5.2 ms in KQ GEMM, 10.0 ms in softmax, and 5.0 ms in VKQ GEMM. The explicit conversion is therefore only about 1.6% of warm oneMKL latency for this shape. Removing it is still useful for memory and traffic, but a replacement kernel must remain close to oneMKL GEMM efficiency to avoid a net regression.

For this test, each complete Q8_0 K or V tensor is about 10.38 MiB and its f16 expansion is about 19.53 MiB. TILE reserves and writes about 39.06 MiB for the two expanded tensors. The oneMKL path bounds the live K/V expansion to 8 MiB with an 8192-token chunk, but it still rewrites the full cache across the chunk loop on every attention call.

#### TILE prototype

TILE already has the right execution structure. `flash_attn_tile` loads a small K or V region into `KV_tmp` in SLM immediately before consuming it. The full-cache conversion happens outside the kernel in `launch_fattn` only because the tile loader accepts f16.

The minimal Q8_0 implementation is:

1. Add Q8_0 overloads of the K and V tile loaders. Each work-item reads a `block_q8_0`, converts its assigned signed bytes with the block f16 scale, and writes f16 directly into the existing `KV_tmp` layout.
2. Specialize only the Q8_0/Q8_0 TILE variant first. Mixed K/V types continue through the current staging path. This avoids multiplying all TILE template combinations.
3. Pass the original byte strides into the Q8_0 variant. Compute row addresses with `nb11`, `nb12`, `nb13` and the corresponding V strides, then index the 32-value Q8_0 block within the row. This naturally preserves dense caches, padded sequence views, batch strides, and Gemma head interleaving.
4. Skip the K/V conversion in `launch_fattn` only for that specialized variant. Leave the existing conversion as the fallback.
5. Update `ggml_sycl_fattn_get_extra` only after the specialized kernel is correct. The reservation can be omitted when the graph will use online Q8_0 TILE, but it must remain when a possible oneDNN path still needs f16 staging.

This is a local change to the existing fused attention kernel. The softmax, mask, sinks, GQA mapping, parallel-block reduction, and output code remain unchanged. It should be the first prototype.

#### oneMKL prototype

oneMKL GEMM cannot consume GGML Q8_0 blocks. Keeping oneMKL calls while removing their dense f16 operands is not possible. The surrounding chunking, masking, online softmax, and accumulation can remain, but both GEMMs need Q8_0-aware XMX replacements.

KQ can reuse much of the existing fused dequant-GEMM design:

- Treat K tokens as quantized matrix rows and Q rows as the dense operand.
- Add a canonical `block_q8_0` SLM stage to the existing joint-matrix machinery.
- Pack Q directly into the VNNI layout required by the XMX B operand.
- Remove the narrow-output gate for this dedicated attention entry point, since prefill commonly has hundreds or thousands of query rows.
- Write `KQ_f32` in the existing query-major layout so the current online-softmax kernel remains unchanged.

VKQ needs a separate joint-matrix kernel. It computes `S[q, token] * V[token, d]`; the Q8_0 scale belongs to each 32-value feature block of each token. A work-group should promote only its V token-by-feature tile into VNNI f16 SLM, consume it immediately with an f16 probability tile, and accumulate f32 output. The result must retain the existing `VKQ_chunk[q, d]` layout.

Both kernels should accept a compact device-safe version of `mkl_fa_kv_desc`. The descriptor is important: the measured view case takes the non-contiguous descriptor path, and the same addressing also distinguishes padded sequence views from true Gemma interleaving.

The oneMKL prototype has a larger risk than TILE. The current library GEMMs are fast, and local promotion can cause the compressed K/V data to be reread for multiple query tiles. Compressed reads are smaller than repeated f16 reads, but XMX occupancy, SLM use, and L2 reuse decide whether this is actually faster. Keep the existing oneMKL route as the default until the custom KQ and VKQ kernels both pass correctness and show no material regression.

#### Prototype status

The experimental implementation in `fattn-mkl.cpp` now has:

- direct VNNI f16 packing of scaled Q;
- an XMX KQ kernel that promotes Q8_0 K blocks into SLM and writes the existing query-major f32 score layout;
- an XMX VKQ kernel that promotes Q8_0 V tiles into SLM and writes the existing f32 chunk output;
- descriptor-based row addressing for contiguous caches, non-contiguous views, padded sequence views, and interleaved heads;
- conditional omission of the dense K/V chunk buffers when the complete online path is selected.

The source builds successfully. KQ and VKQ pass independently and as a combined path. The filtered Q8_0/Q8_0 FlashAttention suite passed all 363 cases, including contiguous and permuted KV views, GQA, masks, long context, and fallback shapes.

The investigation also found a pre-existing non-contiguous descriptor bug. Quantized permuted views were classified as Gemma-style interleaving and had their valid tensor strides reconstructed from the logical KV-head count. A padded physical head dimension therefore produced the wrong sequence stride. Using `nb[1] / type_size` and `nb[2] / type_size` directly fixes the staged oneMKL control and the online kernels; the targeted permuted D=256 test passes after this change.

Performance does not justify enabling the standalone online path. For D=256, 2 KV heads, GQA16, KV=20000, and Q=512 on Arc Pro B60:

| Implementation | Total time | Throughput | KQ | Softmax | VKQ |
|---|---:|---:|---:|---:|---:|
| Staged oneMKL baseline | 22.45 ms | 14.94 TFLOPS | about 5.2 ms | about 10.0 ms | about 5.0 ms |
| Initial online XMX | 2130.7 ms | 157.48 GFLOPS | 32.74 ms | 10.38 ms | 2084.72 ms |
| Online XMX, one Q8 block load per V token | 367.8 ms | 912.41 GFLOPS | 32.67 ms | 10.27 ms | 321.56 ms |
| Online XMX, 128-query V reuse | 265.7 ms | 1.26 TFLOPS | 32.56 ms | 10.34 ms | 220.62 ms |

Avoiding repeated full-block copies in the V loader gives the largest improvement, and reusing each promoted V tile across 128 query rows helps further. The best online version is still about 11.8 times slower than the complete staged baseline. KQ is about 6.3 times slower than the oneMKL KQ stage, while VKQ is about 44 times slower than oneMKL VKQ. The chunked oneMKL path only holds 8 MiB of dense K/V staging for this shape, so the scratch and cache-traffic reduction does not compensate for the regression.

Keep the prototype disabled. The next viable direction is not more tiling of the standalone VKQ GEMM. Promote Q8_0 inside a fused attention kernel that consumes the probability tile immediately, as proposed for TILE, so V promotion shares the existing softmax tile and avoids materializing or repeatedly scanning the full probability matrix.

#### TILE prototype result

Implemented and measured 2026-09-22. `flash_attn_tile_load_tile_q8_0` (half2 and float
destinations) promotes a q8_0 K or V tile directly into the existing `KV_tmp` local-memory
layout. `flash_attn_tile_iter_KQ`, `flash_attn_tile_iter` and `flash_attn_tile` carry a
`kv_q8_0` template flag, and `launch_fattn_tile_case` selects the online variant and passes
`need_f16_K = need_f16_V = false`, so `launch_fattn` stages nothing. Row addressing uses the
original `nb11`/`nb21` byte strides, so dense caches, permuted views, padded sequence views and
interleaved heads all keep working. Runtime gate: `GGML_SYCL_FATTN_TILE_Q8_0`, default 0.

Correctness: the full q8_0/q8_0 FlashAttention set passes, 363/363, with the flag off and on.
That set covers hsk 64/128/256 and hsk=576/hsv=512, permuted views, `v_is_view_of_k`, GQA,
masks, sinks and `n_kv_max`. `test-backend-ops` compares against the CPU reference, so each arm
is an absolute check and not a branch-against-branch comparison.

Performance on Arc Pro B60, D=256, 2 KV heads, GQA16, KV=20000, interleaved arms, 2 reps:

| Query tokens | Staged f16 | Online q8_0 | Delta |
|---:|---:|---:|---:|
| 1 | 2196 us | 4271 us | 1.94x slower |
| 512 | 146.4 ms | 481.2 ms | 3.29x slower |

Profiled with VTune `gpu-hotspots` on the same shapes, so the split below is measured and not
inferred. Per attention call, staged:

| kv | attention kernel | 2x dequantize_block_nc | staging share | occupancy | SBID stall |
|---:|---:|---:|---:|---:|---:|
| 4096 | 49 us | 50 us | 50.5% | 75.3% | 19.8% |
| 20000 | 1922 us | 270 us | 12.3% | 33.5% | 59.1% |

So **dequantization staging is the cheap part, not the expensive part**. The whole prize for
removing it is 12.3% of the op at kv=20000 and 50.5% at kv=4096, and it is only collectable if
the online loader is free. It is not: the online kernel is 4267 us at kv=20000 against a 1922 us
staged attention kernel, and about 117 us at kv=4096 against 49 us. Online promotion costs about
2.2x to 2.4x the attention kernel in order to save a conversion worth at most half of it. That
holds at both ends of the range, so it is not an artifact of one shape.

VTune also confirms the mechanism rather than leaving it as arithmetic. The kv=20000 kernel is
memory-latency bound in both arms, and going online makes it worse: SBID stall 59.1% -> 70.8%,
Send stall 0.0% -> 2.5%, XVE occupancy 33.5% -> 28.6%, XVE array active 26.8% -> 15.3%. SBID is
a scoreboard wait on an outstanding memory message, so the kernel is paying per message. A
q8_0 block is 34 bytes, so `qs` is only 2-byte aligned and 16 features need eight 2-byte loads
plus a scale load, against ONE aligned 16-byte load for f16: 8x message amplification on the
K/V read path. Total traffic moves the other way, about 10 MiB online against 49 MiB staged, and
online still loses, which is what "latency bound, not bandwidth bound" means here.

**The first ceiling probe was wrong, and so was the conclusion drawn from it.** It replaced the
quant gather with one aligned 16-byte load but KEPT the per-lane `block->d` scale load, so it
still cost 2 global messages per 16 features. It measured only 2.9% faster at Q=1 / kv=20000
and that was read as "the direction is dead". It is not.

Removing the scale load as well (taking the scale from a register, values wrong, timing only)
gives the strict ceiling: one aligned 16-byte load per 16 features, dequant ALU kept, no other
memory traffic. D=256, GQA16, Q=1, 2 reps:

| kv | staged f16 | strict ceiling | speedup |
|---:|---:|---:|---:|
| 512 | 30.68 us | 23.68 us | 1.30x |
| 1024 | 39.22 us | 26.30 us | 1.49x |
| 2048 | 58.41 us | 34.34 us | 1.70x |
| 4096 | 106.34 us | 59.58 us | 1.78x |
| 20000 | 2205 us | 1743 us | 1.27x |

So the online path has **22% to 44% of headroom against staging**, not 3%. Nearly all of the
naive loader's cost was message count, and the scale gather was carrying most of what the first
probe left behind. The prefill regression is still expected to stand, because a work-group
re-reads the cache and online promotion repeats per Q tile while staging amortizes; the
oneMKL prototype saw the same thing. Decode is the target.

#### How to reach the ceiling: conflated wide loads

The ceiling is reachable in principle because a q8_0 row can be read with fully aligned wide
loads. LCM(34, 16) = 272 bytes = 8 blocks = 17 aligned 16-byte loads, and for D=256 that is
exactly one row (`nb11` = 272). D=512 is also aligned at 544 bytes. D=64/96/128/576 are not,
so a first implementation should specialize D=256 and leave the rest on the staged path.

The important property is that a wide row load brings in the block scales together with the
quants, so the scale stops being a separate global message - which is exactly what the strict
ceiling assumes. Two ways to redistribute them:

- **Subgroup shuffle, no SLM and no barrier.** With 17 consecutive lanes each holding one
  16-byte chunk of a row, block `b`'s scale sits in chunk `2b` at in-chunk byte offset `2b`,
  for b = 0..7. That is a regular pattern, so `select_from_group` can broadcast it. Cheapest
  option and it costs no local memory, which matters because occupancy is already the binding
  constraint (33.5% at kv=20000).
- **SLM staging.** Cooperatively load the needed byte window per row, barrier, then unpack.
  Costs about 5 KB for a 64x64 K tile on top of the existing 9 KB `KV_tmp`, which may cost
  occupancy. Only worth it if the shuffle mapping turns out to be awkward.

A middle option needs no shuffle at all: give each lane one whole block and read it as three
aligned 16-byte loads from an aligned-down 48-byte window. That is 3 messages per 32 features
against the naive 9 per 32, a 3x reduction, with no SLM and no barrier. It needs a dynamic
byte extract, but `34b mod 16` is always even so the shift can be done in 2-byte units.

Intel 2D block loads / XMX operand fetch are a further option: the raw bytes of a K tile ARE a
regular 2D matrix (pitch 272, 64 rows), even though the quants inside are not, so the hardware
path can fetch the tile efficiently. There is precedent for ESIMD in this tree
(`GGML_SYCL_ESIMD_Q8_0`). This is the heaviest option and should only follow a shuffle-based
version that already demonstrates the win.

Unknowns a real implementation must settle, none of which the ceiling probe pays for: the
scale broadcast (a shuffle is cheap, an SLM round trip is not), register pressure in a kernel
that already warns about exceeding 128 bytes of local variables, and whether any added SLM
costs more occupancy than the 22-44% gap can absorb.

#### Four implementations of the wide load, none of which reaches the ceiling

Measured D=256, GQA16, Q=1, interleaved arms, 2 reps, us per attention call. All four pass
363/363, so this is purely a performance ranking.

| kv | staged f16 | naive gather | SLM staging | reg window | reg window, static pick | strict ceiling |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 30.6 | 32.8 | **26.3** | 34.7 | 90.2 | 23.7 |
| 1024 | 39.2 | 43.6 | **33.5** | 51.0 | 127.8 | 26.3 |
| 2048 | 58.5 | 67.7 | 66.6 | 89.1 | 222.6 | 34.3 |
| 4096 | 106.4 | 124.3 | 117.8 | 167.8 | 450.7 | 59.6 |
| 20000 | 2206 | 4271 | 3956 | 5468 | 10286 | 1743 |

Each variant trades one bottleneck for another, confirmed by VTune rather than argued:

1. **Naive per-lane gather.** 9 global messages per 16 features. Message bound.
2. **Cooperative wide load into local memory, then unpack there.** Global messages drop to
   about 1.25 per 16 features and it is the only variant that ever beats staging, by 14-17%
   at kv <= 1024. But the 6 KB staging buffer cut occupancy from 75.2% to 45.7% and the extra
   barrier per loader call raised barrier stall from 5.0% to 19.5%. It loses above kv=2048,
   and the loss grows with kv because the barrier is paid once per loader call and the number
   of calls scales with kv.
3. **Per-lane 48-byte aligned window in registers**, no local memory and no barrier. Occupancy
   recovers to 74.1%, but the dynamic offset into the window spilled it: 2560 bytes of scratch
   per thread, SBID 51.2%, barrier 44.4%.
4. **Same, with the pick offset made compile time** by unrolling the 16 possible (offset, half)
   cases. This removes the dynamic index but the branch chain is far worse than the spill.

5. **Byte-owned**: a work-item takes one aligned 16-byte chunk and writes whichever features
   those bytes encode, with every byte picked at a compile time offset and the block scales read
   straight from global. 169.8 us at kv=4096. It spilled **8256 bytes**, more than any other
   variant, at 75.4% occupancy and only 38.1% XVE array active.

The pattern across all five is the real finding: **the tile kernel is already register starved.**
It runs at 128 GRF and the source already carries compiler notes that `flash_attn_tile` exceeds
128 bytes of declared local variables. Anything that keeps more than about 16 bytes of extra
live per-work-item state spills, and spilling costs far more than the messages it saves. That is
why the naive gather, which keeps almost no state, is the second best online variant despite
issuing 9x the messages, and why the local-memory variant is the best: it puts the state
somewhere other than registers. The strict ceiling probe does not spill only because it loads 16
bytes and immediately consumes them, keeping no window at all.

#### Large GRF (256) removes the spills and changes nothing

Tested with the per-kernel property `sycl::ext::intel::experimental::grf_size<256>`, applied
only to the q8_0 kernel instantiation so the f16 path keeps 128 GRF and is unaffected. Verified
in VTune that it took effect. D=256, kv=4096, Q=1:

| variant | GRF | spill | occupancy | SBID | XVE active | kernel |
|---|---:|---:|---:|---:|---:|---:|
| staged f16 | 128 | 0 B | 75.1% | 19.7% | 67.8% | 49 us |
| register window | 128 | 2560 B | 74.1% | 51.2% | - | 168 us |
| register window | 256 | **0 B** | 75.6% | 45.8% | 23.2% | 172 us |
| byte-owned | 128 | 8256 B | 75.4% | 41.9% | 38.1% | 170 us |
| byte-owned | 256 | **0 B** | 75.9% | **18.5%** | 33.5% | 179 us |

Two results worth keeping, one of them against the usual advice:

1. **Large GRF does what it promises and costs nothing in occupancy here.** Both spills went to
   exactly zero and occupancy stayed at about 75%, it did NOT halve. The common rule of thumb
   that 256 GRF halves threads per XVE did not show up in these measurements on bmg-g21. It is
   a usable tool on this part, and the per-kernel property means it can be applied to one
   instantiation without touching the rest.
2. **It still makes nothing faster, because in every variant the spill was a symptom.** With the
   spill gone, each variant simply exposes its real bottleneck.

With the register file no longer in the way, the three failures separate cleanly, and none of
them is about registers:

- **Naive gather is message bound.** 9 memory messages per 16 features.
- **Register window is byte bound.** Its 48-byte window per 16 features means overlapping reads:
  each 34-byte block is covered by two 48-byte windows, about 2.8x read amplification, so it
  moves 3 bytes per feature where f16 moves 2 and an ideal q8_0 reader moves 1.06. At 256 GRF
  its SBID stall is still 45.8%.
- **Byte-owned is instruction bound.** At 256 GRF its SBID fell to 18.5%, matching the staged
  kernel, so its memory behaviour is finally correct: chunks do not overlap and it reads the
  cache exactly once. But XVE array active is only 33.5% against 67.8% for staged. Per 16 bytes
  its inner loop does 16 divisions by 34, 16 conditionals and about 15 two-byte local stores,
  where the f16 loader does two wide loads and two wide stores per 16 features.

So the loader can be made message-clean, or byte-clean, or instruction-clean, but not all three
at once, and every one of those costs traces to the same cause below.

#### The blocker is the layout, and this fork already fixes it elsewhere

Every failure above traces to one thing: a `block_q8_0` interleaves a 2-byte scale with its 32
quants every 34 bytes, so the quants are never 16-byte aligned and reaching them needs either
narrow loads, a register window, or a local-memory bounce. This repository already solves that
problem for weights. `dequantize_mul_mat_vec_q8_0_reorder_esimd` consumes a reordered,
structure-of-arrays q8_0:

    const int8_t     * qs = (const int8_t *) vx;                    // all quants, contiguous
    const sycl::half * d  = (const sycl::half *)(qs + nb * QK8_0);  // all scales, after them

with the comment "8 Q8_0 blocks = 256 elements = one 256 B block_load". In that layout the
quants for features [f, f+16) live at `qs + f`, 16-byte aligned whenever f is, and the scale is
a separate flat array. That is exactly the access pattern the strict ceiling probe assumed, and
it is already running in production here for matrix-vector.

So the right next step is not a sixth loader. It is to ask whether the KV cache can carry the
reordered q8_0 layout, the way the weights already do. If it can, the loader becomes trivial,
one aligned wide load for quants plus one hoisted scale load, and the 1.3x to 1.78x ceiling
becomes reachable rather than theoretical. The cost is in the cache write path and in every
other consumer of the KV cache, a much larger blast radius than the FA kernel, and it must be
scoped before anything is built.

On ESIMD: its register-indirect addressing, a `simd_view` with a runtime offset mapping to
indirect GRF addressing, is precisely the mechanism variant 3 needed and would have removed its
spill. It is not reachable surgically. `[[intel::sycl_explicit_simd]]` is a whole-kernel mode in
every existing use in this tree, so using it inside `flash_attn_tile` means rewriting the entire
kernel, softmax and online rescaling and mask and sinks and GQA and stream-k, in explicit SIMD.
`invoke_simd` exists in this oneAPI but marshals arguments across a `__regcall` boundary at
fixed sub-group width; passing a 48-byte-per-lane window, 1.5 KB per sub-group, on every loader
call would cost more than the extraction saves. A full ESIMD FA kernel for q8_0 decode is a
coherent project, but it is a rewrite, not a patch, and it would be far more attractive on top
of a reordered cache where the loads are already aligned and indirect addressing is not needed.

Note also that the naive-gather row above is the honest comparison point for the earlier
"+7% to +16%" claim, and that at kv=20000 every online variant collapses, because that shape
loses the GQA optimization - see the caveat section.

#### Does the same trick help oneMKL? No, and it cannot.

VTune `gpu-hotspots` on the oneMKL path, D=256, GQA16, KV=20000, Q=512, 18.64 ms/run,
18.00 TFLOPS. Share of total GPU time:

| kernel | share |
|---|---:|
| `mkl_fa_online_softmax_chunk` (two variants) | 50.3% |
| `gemm_f16f16f32_nocopy_*` (four variants) | 38.5% |
| `mkl_fa_normalize_head` | 5.9% |
| `dequantize_block_nc_sycl` q8_0 -> f16 (two variants) | **1.5%** |
| `mkl_fa_pack_q_fp16` | 0.9% |

Two conclusions.

First, oneMKL cannot consume q8_0 at all. The kernels it dispatches are literally
`gemm_f16f16f32_nocopy_*`: dense f16 in, f32 out. There is no internal equivalent of a
conflated wide load, because the library never sees a quantized block. The conversion that
feeds it is ours, and it is 1.5% of GPU time. Making that conversion infinitely fast buys 1.5%.

Second, and this is the general rule: **the wide-load win scales inversely with how many times
the cache is re-read.** In prefill the dense f16 is consumed many times, once per Q tile by the
GEMM, so a one-time conversion amortizes to nothing. In decode the cache is read once, so the
conversion is 12% to 50% of the op and never amortizes. That is why the same idea is worth
22-44% in TILE decode and under 2% in oneMKL prefill, and it is the same reason online
promotion loses in prefill: it repeats per tile what staging pays once.

Unrelated to the KV format, the profile says where oneMKL prefill headroom actually is: the
online softmax is 50.3% of it, more than all four GEMM variants combined. XMX pipe active is
only 20-24% in every kernel including the GEMMs, and SBID stall is 45-50% throughout, so this
path is memory-stall bound rather than XMX-throughput bound. Softmax is the target here, not
the KV format.

#### Caveat on the kv=20000 shape

The VTune kernel names show that the two kv values do not run the same TILE variant. At
kv=4096 the kernel is `<DKQ=256, ncols1=1, ncols2=8>`, at kv=20000 it is
`<DKQ=256, ncols1=2, ncols2=1>`. The reason is `use_gqa_opt`, which needs
`K->ne[1] % FATTN_KQ_STRIDE == 0`. 4096 is a multiple of 256, 20000 is not (remainder 32), so
the kv=20000 case loses the GQA optimization and each of the 16 Q heads streams the whole KV
cache separately. That is why it runs at 298 GFLOPS with 33.5% occupancy while kv=4096 runs at
1.25 TFLOPS with 75.3% occupancy.

`llama_kv_cache` pads n_kv with `n_pad_cur = max(n_pad, 256)`, so the real server always
presents a multiple of 256 and keeps the GQA optimization. **kv=20000 is therefore not a
representative decode shape**, and the representative TILE decode regression for online q8_0 is
the kv=512..4096 figure, about +7% to +16%, not the +94% seen at kv=20000. The conclusion does
not change, but the headline number should be the smaller one. Anything else measured on
kv=20000, here or in the oneMKL section above, carries the same caveat.

#### Doing the attention math at q8_0 level instead

The obvious alternative to promoting to f16 is to keep the math quantized. That design already
exists in this backend: the VEC kernel quantizes Q to q8_1 into local memory
(`quantize_q8_1_to_shared`), runs KQ as an integer dot product (`vec_dot_fattn_vec_KQ_q8_0`),
dequantizes V per element, and **stages nothing at all**. It was measured against TILE by
forcing it with `GGML_SYCL_FATTN_PREFER_VEC=1`, D=256, GQA16, Q=1, 2 reps:

| kv | TILE staged | TILE online q8_0 | VEC int8 KQ |
|---:|---:|---:|---:|
| 512 | 30.6 us | 32.8 us | 105.8 us |
| 1024 | 39.2 us | 43.6 us | 210.7 us |
| 2048 | 58.5 us | 67.7 us | 398.6 us |
| 4096 | 107.0 us | 124.3 us | 698.5 us |

VEC is 3.5x to 6.5x slower than TILE with staging, and its throughput is flat at about
160-190 GFLOPS regardless of kv, which is the signature of a purely latency-bound kernel. It
hits the same 2-byte alignment wall: `vec_dot_fattn_vec_KQ_q8_0` reads K through
`ggml_sycl_memcpy_1<sizeof(int), 2>`, two messages per four quants. Keeping the math at int8
does not avoid the cost, because the cost is reading q8_0, not converting it. This is also why
the dispatcher already prefers TILE over VEC for quantized KV decode on Xe2.

Conclusion: **the direction is live for TILE decode, and the current prototype is the wrong
implementation of it, not evidence against it.** The naive per-lane gather loses because it
issues 9 global messages per 16 features. With aligned wide loads the same work has 22-44% of
headroom against staging at D=256 decode shapes, and it additionally frees the
`GGML_SYCL_MEM_FATTN_KV` staging reservation.

What IS ruled out: doing the math at int8 level (VEC, below) and the standalone oneMKL XMX
route, both of which lose for reasons unrelated to load width. And prefill, which should stay
on the staged path.

What is worth keeping independently of this: `flash_attn_tile_iter` guarded its
`nbatch_K_last` tail call with a plain `if` on a `constexpr` value. The tail kernel was
therefore instantiated with `nbatch_K_last == 0` for every head size where `nbatch_K` divides
`DKQ`, which is every configured head size, and the q8_0 loaders exposed it as a division by
zero in that dead code. The guard is now `if constexpr`, which removes the dead instantiation. Behavior
is unchanged.

#### Recommended sequence

1. DONE for correctness. Q8_0/Q8_0 online loading in TILE is implemented and passes all 363 cases, including permuted views, GQA, masks, sinks and `n_kv_max`. The naive per-lane loader is slower (+7% to +16% at representative decode shapes), but see "How to reach the ceiling" above: that is a load-width problem, not a dead end.
2. NEXT, and the real work. Replace the per-lane gather with conflated aligned wide loads plus a subgroup-shuffle scale broadcast, specialized for D=256 first. Target is the measured 22-44% ceiling. Then measure peak `GGML_SYCL_MEM_FATTN_KV`, which this also frees.
2b. Measure on kv values that are multiples of 256 only - see the caveat section.
3. Add the Q8_0 XMX KQ kernel inside the oneMKL orchestration and compare its isolated result and timing with oneMKL KQ while retaining staged V.
4. Add the Q8_0 XMX VKQ kernel, then remove the chunk K/V buffers only when both kernels are selected.
5. Re-run end-to-end model correctness, prompt processing, decode, scratch allocation, and device-memory traffic.

## IQ3_S reorder: enabled by default 2026-09-22

`GGML_SYCL_IQ3_REORDER` was added with the reorder itself (`sycl: add reordered IQ3 MoE paths`,
687 lines over 9 files) and defaulted to 0. No run script ever set it, so every end-to-end
number and every profile taken on this branch ran the NON-reordered path. Default is now 1.

End-to-end, llama-bench, deployment-matched config (3x B60, -ts 31/33/36, -ub 1024, q8_0 KV,
`-ot per_layer_token_embd=CPU`), depth 0, t/s:

| arm | pp512 | pp4096 | tg128 |
|---|---:|---:|---:|
| master 0f8a414b7 | 265.71 | 392.73 | 20.33 |
| branch, reorder off | 435.79 | 509.30 | 24.20 |
| branch, reorder on | 433.96 | 508.71 | **27.29** |

Reorder's own contribution: **+12.8% decode (tg128), prefill flat** (-0.1% at pp4096, both
reps). `llama-cli` independently gives 23.7 -> 26.6 t/s generation, +12.2%, a different harness
agreeing to within half a point. Text is coherent and WORD-IDENTICAL between reorder off and on.

Two methodology notes that matter for anyone repeating this:

- **Rep 1 is cold and must be discarded.** Prefill was 33-70% LOWER in rep 1 than rep 2 on all
  three arms (master 259 -> 393, branch 303 -> 509 at pp4096). The model is mmap'd with
  `--lazy-mode auto`, so the first pass faults 87 GiB off disk. Decode barely moves (2-4%)
  because it re-reads the same weights. The rep-1 "+7% pp512 from the reorder" was pure cold
  noise; rep 2 says -0.4%.
- **All four binaries were stale before this run.** wt-master's llama-server predated the
  46-commit fast-forward, the branch's predated the rebase, and both llama-bench/llama-cli were
  two days older than the libraries they linked. Rebuild and check mtimes before every arm.

Do NOT compare these to the HANDOVER table: those were server runs at 16k-113k context depth,
this is llama-bench at depth 0, and the branch's large decode wins are depth-dependent.

### At 64K depth

Same harness with `-d 65536 -p 512 -n 128 -r 2`, two passes, agreeing to within 0.5% (the page
cache was already warm, so there is no cold rep here). Mean of both passes, t/s:

| arm | pp512 @ d65536 | tg128 @ d65536 |
|---|---:|---:|
| master 0f8a414b7 | 154.81 | 8.97 |
| branch, reorder off | 227.87 | 16.36 |
| branch, reorder on | 229.68 | **17.78** |

Branch+reorder vs master: **+48% prefill, +98% decode** at 64K, against +63%/+34% at depth 0.
The decode advantage roughly triples with depth, the prefill advantage shrinks.

Reorder's own contribution at 64K: **+8.7% decode, +0.8% prefill**, against +12.8%/-0.1% at
depth 0. That compression is expected and was predicted before the run: the reorder speeds up
MoE weight reads, which are depth-independent, while flash attention grows with depth, so the
reorder's share of total work falls as depth rises.

**This cross-validates the HANDOVER server numbers almost exactly.** That table reports
master 9.05 and branch 16.36 t/s at kv=65535; this run gives master 8.97 and branch-off 16.36
from a completely different harness (llama-bench vs llama-server). Two independent measurement
paths landing on the same digits is the strongest evidence either number is right.

### SOLVED: the reordered grouped GEMM never fired (fix uncommitted)

**Cause.** `opt_for_reorder_id()` has exactly ONE call site, inside
`ggml_sycl_mul_mat_id_mmvq_fused()` - the mat-vec path - and it sits *after* that function's
bail checks. The grouped GEMM at `ggml_sycl_mul_mat_id` only READS
`extra->optimized_feature.reorder`; it never triggers it. A large batch bails out of
mmvq_fused before reaching the trigger, so prefill always saw unreordered weights.

Proven by dispatch, not timing: a prefill-only profile showed
`grouped_dequant_gemm_launch<block_iq3_s, (bool)0>` x129 *in the same process where*
`reorder_qw_iq3_s_moe` had run and `reorder_vec_dot_q_sycl<(ggml_type)21>` was dispatched. The
weights were reordered, the mat-vec used it, the GEMM did not.

**Fix:** call `opt_for_reorder_id()` in the grouped path too, gated on
`ggml_sycl_fused_dequant_gemm_f16_reorder_ok(src0->type)` - without that gate a reordered
Q4_K/Q5_K/Q6_K tensor makes `ggml_sycl_grouped_dequant_gemm_f16()` decline the launch entirely
and fall back to the per-expert loop.

After the fix the dispatch is `<block_iq3_s, (bool)1>` x122, and pp4096 goes
**511.6 -> 533.4 t/s, +4.3%**, reproducible over three steady passes (533.28 / 533.56 / 533.35,
sigma < 1.6). Each process pays a one-time ~1 s `reorder_qw_iq3_s_moe`; negligible for a server,
visible in llama-bench as a huge stddev on the first invocation of a session.

**+4.3%, not the third of prefill the profile share suggested.** The grouped GEMM is ~34% of
prefill, so +4.3% overall means the kernel itself got only ~13% faster, against ~3x for the
mat-vec. That is mechanically sensible: the GEMM already amortizes each weight read across 32
columns of a tile, so coalescing matters much less; the mat-vec re-reads weights per output
element with no reuse, so layout dominates there.

**Measurement warning learned here:** `llama-bench` NORMALISES test order to pp-before-tg
regardless of the order of `-p`/`-n` on the command line. `-n 128 -p 4096` still runs pp first.
Any experiment that depends on decode running before prefill cannot be built with llama-bench.
Also: this model cannot fit in page cache (87 GiB + 51 GiB host tensors vs 80 GB RAM), so a
cold invocation reads 300-390 t/s against 510+ warm. Cold/warm swamps every effect here; only
compare arms that are back-to-back and both warm, and prefer a dispatch check over a timing
check when the question is "does this code path run".

### Superseded: the reordered grouped GEMM delivers nothing

The profile has `grouped_dequant_gemm<block_iq3_s, false>` at ~34% of prefill. Enabling the
reorder moved prefill by -0.1%. So the reordered MAT-VEC path fires and is worth +12.8% decode,
but the reordered GROUPED GEMM path either is not selected or is not faster. Check the gates in
`ggml_sycl_grouped_dequant_gemm_f16`: the shape check, and the hard
`GGML_ASSERT(!reordered || expert_stride == M * (K / blck) * type_size)`. A third of prefill is
still on the table.

### Large pages: tested 2026-09-22, NO effect. Do not retry via this route.

Hypothesis: `ggml_sycl_malloc_device()` calls
`zeMemAllocDevice(ctx, &desc, size, /* alignment */ 64, dev, &ptr)`, so a multi-GiB weight
buffer could be backed by 4 KiB pages, which would explain the 10.9M / 9.0M TLB misses on the
MoE mat-vecs (20-100x any other kernel in the profile). `MEM_SIZE_2M` exists in this codebase
but is used only in the host-pinned path, never here.

Tested behind `GGML_SYCL_LARGE_PAGES`, requesting 2 MiB alignment for allocations >= 2 MiB.
Verified live, not assumed: the startup banner showed `GGML_SYCL_LARGE_PAGES: 1` and
`GGML_SYCL_USE_LEVEL_ZERO_API: 1`, so `ggml_sycl_use_level_zero_device_alloc()` returns true
and the branch really ran.

Result, VTune, same build, decode:

| kernel | TLB LP=0 | TLB LP=1 | time LP=0 | time LP=1 |
|---|---:|---:|---:|---:|
| iq4_nl MoE mat-vec | 3,041,740 | 2,886,025 | 0.506s | 0.504s |
| iq3_s reordered mat-vec | 1,040,562 | 1,142,411 | 0.215s | 0.214s |
| q8_0 esimd | 1,005,602 | 939,842 | 0.178s | 0.177s |

TLB misses do not move (+-10%, run-to-run noise) and times are identical to the millisecond.
Throughput over 3 interleaved warm passes: pp4096 533.5 vs 533.6, tg128 27.06 vs 27.08.

**Conclusion: the alignment argument to `zeMemAllocDevice` is not a page-size lever on this
driver.** Either it already backs large allocations with large pages, or page granularity is
decided elsewhere entirely. The remaining TLB pressure looks inherent to the MoE gather - a
20 GiB weight buffer touched at scattered expert offsets exceeds TLB reach at any page size
this driver offers. Change reverted; the flag is gone.

If anyone wants to reopen the TLB angle, it needs a different mechanism (driver/kernel page
policy, or changing the ACCESS pattern), not an allocation-alignment hint.

### IQ4_NL reorder: COMPLETE 2026-09-22, +3.7% prefill / +5.8% decode. Flag defaults 0.

Final: pp4096 534.5 -> 554.1 (+3.7%), tg128 27.47 -> 29.07 (+5.8%), three interleaved warm
passes agreeing. Op tests 3/3 MUL_MAT_ID and 14/14 MUL_MAT both ways. Text word-identical to
the non-reordered run in all three path configurations, including a forced fallback.

IQ4_NL's block is byte-identical to Q4_0 (`ggml_half d; uint8_t qs[16]`), so the SoA split,
offsets and `block_q_t` traits transfer verbatim; only the dequant differs (`kvalues_iq4nl`
lookup instead of `q - 8`). What cost the time was not the reorder, it was CONSUMER COVERAGE.
Six sites were needed:

1. `block_q_t<IQ4_NL>` (quants.hpp) - copy of Q4_0's
2. `reorder_vec_dot_q_sycl<IQ4_NL>` (vecdotq.hpp) - Q4_0's shape with the LUT, no -8
3. `reorder_qw_iq4_nl_moe` (ggml-sycl.cpp) - q4_0's SoA split applied per expert slice
4. `fg_reorder_a<block_iq4_nl>` (fused-gemm.cpp) + IQ4_NL in `..._reorder_ok`
5. `dequantize_block_iq4_nl_reorder` / `dequantize_row_iq4_nl_sycl_reorder` + both selectors
6. `reorder_mul_mat_vec_iq4_nl_q8_1_sycl` (mmvq.cpp) + the plain mat-vec dispatch branch

and BOTH `fused-gemm.cpp` switches had `<block_iq4_nl, false>` hardcoded - the grouped one AND
the non-grouped one. Missing either produced `//////////`.

**How the misses were found, and the lesson.** Op tests passed at every broken stage (3/3, 14/14)
while the model emitted slashes, so they are worthless as a gate for a layout change. What
worked was: force the alternative path with `GGML_SYCL_GROUPED_GEMM=0`, check TEXT, and when it
was still wrong, VTune the run and grep the dispatched kernel names for
`block_iq4_nl, (bool)0` - the one consumer still reading SoA as AoS. Guessing at consumers cost
three rounds; the dispatch check named the culprit immediately each time.

A layout is a contract with every reader of the tensor. Enumerate the readers first.

### Superseded notes from the failed first attempt

IQ4_NL has byte-identical block layout to Q4_0 (`ggml_half d; uint8_t qs[16]`, 18 bytes), so the
SoA split, the offsets and `block_q_t` traits transfer verbatim; only the dequant differs
(lookup through `kvalues_iq4nl` instead of `q - 8`). Implemented behind
`GGML_SYCL_IQ4_NL_REORDER` (default 0): `block_q_t<IQ4_NL>`, `reorder_vec_dot_q_sycl<IQ4_NL>`,
`reorder_qw_iq4_nl_moe`, plus wiring in `init_tensor`, `reorder_qw`,
`ggml_sycl_mul_mat_id_reorders_type` and the mmvq id reorder dispatch.

It DISPATCHES correctly (`reorder_qw_iq4_nl_moe` runs, and
`launch_mul_mat_vec_q_moe_reorder<...>` goes from 1 to 2 instantiations) and it passes
`test-backend-ops` (MUL_MAT_ID 3/3, MUL_MAT 14/14). **It also produces `//////////` from the
real model.**

Cause: only the MoE mat-vec consumer was taught the layout.
`ggml_sycl_fused_dequant_gemm_f16_reorder_ok()` allows only IQ3_S, so a reordered IQ4_NL tensor
makes `ggml_sycl_grouped_dequant_gemm_f16()` DECLINE, control falls through to the per-expert
loop, and that reads SoA bytes as AoS. Prefill corrupts, generation degenerates.

**The measured "+7.2% decode" is therefore void** - it was timed against wrong arithmetic.

To finish it properly, every consumer of these weights needs the layout, which is what made the
IQ3_S change 687 lines over 9 files:
1. `fg_reorder_a<block_iq4_nl>` in fused-gemm.cpp, and add IQ4_NL to
   `ggml_sycl_fused_dequant_gemm_f16_reorder_ok()` so the grouped GEMM consumes it instead of
   declining.
2. Reordered dequant in `convert.cpp` / `dequantize.hpp` so the per-expert fallback, and
   anything else that dequantizes an expert slice, reads SoA.
3. Only then re-measure, and validate with generated TEXT, not op tests - MUL_MAT_ID has just
   3 iq4_nl cases and all of them passed while the model emitted slashes.

**Lesson, again:** op-level tests plus a dispatch check are NOT sufficient for a layout change.
A layout is a contract with every reader of the tensor, and the test suite only exercises some
of them. Text first, next time.

### What else could be reordered on this model: almost nothing left

Tensor type histogram of `Qwen3.8-Flash-Next UD-IQ4_XS`, parsed from the GGUF headers:

| type | tensors | Gelem | reorder |
|---|---:|---:|---|
| IQ4_NL | 44 | 87.27 | done 2026-09-22 |
| IQ3_S | 94 | 78.85 | done, default on |
| Q8_0 | 502 | 8.41 | MUL_MAT yes (ESIMD), MUL_MAT_ID no |
| IQ4_XS | 2 | 1.68 | none |
| Q6_K | 1 | 0.64 | already in the MoE reorder list |
| F32 | 557 | 0.08 | n/a |
| BF16 | 24 | 0.02 | n/a |

**IQ4_NL and IQ3_S together are 166.1 of 176.9 Gelem, about 94% of the model's parameters and
essentially all of its quantized weight bytes.** Both are now reordered and on by default, so
the reorder work on this model is effectively finished.

The only remaining candidate with any share is **Q8_0 through MUL_MAT_ID**.
`ggml_sycl_mul_mat_id_reorders_type()` lists Q4_K, Q5_K, Q6_K, IQ3_S and IQ4_NL but NOT Q8_0,
so the MoE Q8_0 mat-vec (`launch_mul_mat_vec_q_moe<..., block_q8_0, ...>`, 0.624 s of ~15.2 s
decode, 4.1%) runs unreordered. Note the non-MoE Q8_0 path IS already reordered and takes the
ESIMD route (`q8_0_esimd_launch`, 4.18 s), so only the MUL_MAT_ID side is missing. Wiring cost
is low because `reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>` already exists; it needs a
`reorder_qw_q8_0_moe` producer plus the id dispatch entries. Expected value, applying the
Amdahl discipline that has held all day: 4.1% of decode times a partial speedup, so roughly
+2%. Worth doing only if it is cheap.

IQ4_XS is 2 tensors and 0.116 s of decode (0.8%). Not worth it.

## Grouped XMX GEMM: restage the A gather through SLM

Status: measured and attributed, not implemented. This supersedes further M-tile widening.

The grouped dequant-GEMM (`fused-gemm.cpp`) is pinned at ~100% GPU L3 Busy with ~1870 GB/s of L3
read, against only ~95 GB/s of DRAM read. That ratio is cache-line inflation, and the A (weight)
gather is where it comes from, not the packed-B re-read.

Measured, `vtune -collect gpu-hotspots`, `iq3_s` reordered kernel, real model, pp2048:

| M tile | avg      | spill | occupancy | L3 Busy | L3 read   | bytes  |
|--------|----------|-------|-----------|---------|-----------|--------|
| 16 row | 4.439 ms | 0     | 100%      | 100.0%  | 1868.6 GB/s | 8.29 GB |
| 32 row | 4.263 ms | 0     | 100%      | 100.0%  | 1725.6 GB/s | 7.36 GB |

Widening the M tile from 16 to 32 halves the packed-B re-read by construction, and moved total
bytes only -11.3% (end-to-end prefill +1.0%). So packed-B is 23-47% of L3 read and the rest is
invariant in the tile height.

Why A is invariant: rows are partitioned across row-groups, so a 64-row work-group performs 4x the
staging events of a 16-row one and there are 4x fewer work-groups. The A traffic is identical.
Widening the tile therefore cannot touch ~3/4 of the L3 read, while halving resident threads
(SLM 12 -> 16 -> 24 KiB gives 40 -> 32 -> 20 threads per Xe-core).

The mechanism, read from `fg_reorder_a<block_iq3_s>::stage`: each (row, k-step) issues FOUR
separate global streams - `qs` 8 B, `qh` 1 B, `signs` 4 B, `metadata` 3 B = 16 useful bytes. Each is
indexed `ib = row * blocks_per_row + kb/8`, so at K=2560 (blocks_per_row = 10) the lane-to-lane
strides are 640 / 80 / 320 / 60 bytes. Every one exceeds a cache line, so each of the four gathers
touches 16 distinct 64 B lines to deliver 1-8 bytes per lane. Upper bound with no reuse:
30.6M events x 4 x 64 B = 7.84 GB, and the measured tile-invariant residual is 4.4-6.4 GB, i.e.
56-82% of that bound. ~13-15x inflation over the 0.49 GB of useful A bytes.

Two complementary fixes. They attack different factors and can be combined:

1. Temporal - stage a whole superblock per lane. `ib8 = kb % 8`, so the same 64 B `qs` line, 8 B
   `qh`, 32 B `signs` and 6 B metadata serve 8 consecutive k-steps, but ~32 KiB of packed B is
   streamed in between and evicts them. Unroll the `kb` loop by `QK_K/32 = 8` and decode all 8
   sub-blocks from registers. Cuts A line fetches ~8x and A *messages* ~8x. Cost: `tile_a` must
   hold 8 k-steps - `FG_KSPLIT * SG_ROWS * 8 * FG_BK` halves = 32 KiB at 16 rows. Affordable at 16
   rows, not at 64, so this is mutually exclusive with M-tile widening: both spend the same SLM.

2. Spatial - cooperative staging. Today lane i reads row i, so 16 lanes stride across rows. Instead
   have the 16 lanes cooperatively read one row's contiguous bytes into SLM, then let each lane
   dequantize its own row out of SLM. Turns a 16-line gather into ~2 lines. All 13 `fg_stage_a`
   overloads and 6 `fg_reorder_a` specializations keep working unchanged - they take a pointer, so
   only its address space changes.

Do NOT fix this by transposing the reorder to make the row the innermost index. The same reordered
weights are read by the decode mat-vec (`launch_mul_mat_vec_q_moe_reorder*`, the largest GPU kernel
in decode at 0.547 s), which launches one 16-lane work-group per output row with the lanes
splitting K - that pattern already coalesces under today's `[row][block][byte]` layout. A
row-innermost layout would fix the GEMM and scatter decode, and there is no room for two copies of
an 87 GiB model.

Free experiment before building anything: toggle `GGML_SYCL_REORDER_TYPES` so `iq3_s` uses the
canonical A stage instead of the SoA one and re-run the same VTune. Two A layouts, identical B
behaviour. If L3 read barely moves, A is not the traffic and this whole entry is wrong.

## Grouped XMX GEMM: the kernel is latency-bound, not bandwidth-bound

Status: measured 2026-09-24. This supersedes the traffic-reduction framing in the entry above.

Three M-tile widths, `iq3_s` reordered kernel, `vtune -collect gpu-hotspots`, pp2048:

| tile | avg ms | L3 GB/s | GB moved | ALU1 active | XMX active |
|------|--------|---------|----------|-------------|------------|
| 16   | 4.439  | 1868.6  | 8.29     | 6.3%        | 1.8%       |
| 32   | 4.263  | 1725.6  | 7.36     | 6.3%        | 1.9%       |
| 64   | 4.689  | 1066.1  | 5.00     | 5.2%        | 1.5%       |

**The 64-row tile cut L3 traffic 40% and ran 5.6% slower** (end-to-end pp2048 617.9 vs 651.7).
So L3 bandwidth is not the limiter. The execution units are idle ~92% of the time in every arm.
The kernel is waiting on scattered loads that nothing hides, and widening the tile removes the
work-group concurrency that was doing the hiding (`Send` stalls fall 13.1% -> 0.5% at 64 rows).

**Counter warning.** For this kernel VTune reports `XVE Array: Active = 100%`, `Stalled = 100%`,
`Idle = 0%` simultaneously, and `SBID = DistAcc = 100%`. Those are pegged and meaningless - do not
reason from them, as was done earlier in this branch. `Average Time`, `Spill`, the L3 bandwidth
figures and the ALU/XMX active percentages look sane; the array/stall breakdown does not.
`GPU L3: Busy` reads 100.0% in all three arms including the one moving 40% less - also useless.

The lever is therefore **latency hiding, not traffic**:

1. **Double-buffer the A stage.** `fused_dequant_gemm_tile` issues the four A loads for k-step n,
   barriers, then does the MADs for step n - so the loads are never overlapped with arithmetic.
   Stage k-step n+1 into a second `tile_a` buffer while the MADs for n run. Costs one extra
   `tile_a` (4 KiB at 16 rows, cheap, unlike the 8-k-step variant) and one more barrier, and turns
   a serial load->compute chain into a pipeline. This is the change the measurements actually
   point at.
2. **More sub-groups per work-group** (`FG_KSPLIT > 4`) increases threads per work-group and
   therefore in-flight loads, at the cost of a wider K-split reduction. Cheap to try - it is one
   constant - but it changes the reduction shape, so measure correctness as well as speed.
3. Narrower tiles (M-tile 8) go the other way on the same axis: more work-groups, more
   concurrency, more traffic. The 16 -> 32 -> 64 curve is non-monotonic with a maximum at 32,
   so 8 is worth one data point to find out which side of the peak we are on.

**Refinement, measured the same day.** Traffic reduction is not uniformly useless - it pays when it
is free and loses when it is bought with concurrency. Same kernel, same tile, same B behaviour,
only the A layout differs (`GGML_SYCL_REORDER_TYPES` with the IQ3_S bit clear vs set):

| A stage   | avg ms | L3 GB/s | GB moved | ALU1 | XMX |
|-----------|--------|---------|----------|------|-----|
| canonical | 5.292  | 2223.6  | 11.77    | 4.8% | 1.3%|
| SoA       | 4.439  | 1868.6  | 8.29     | 6.3% | 1.8%|

-30% traffic and -16% time, at no cost in concurrency or SLM. So the A gather really is a large
share of L3 read - at least the 3.48 GB that one layout change removes - and reducing it is
worthwhile *provided the reduction is not paid for in occupancy*.

That puts the SLM-A staging entry above (bit 512) genuinely in the middle: it cuts A traffic
further and keeps the work-group count, but costs SLM (12288 -> 19328 B per work-group at 16 rows,
10 -> 6 resident work-groups). Neither outcome would be surprising. Measure it.

## Grouped XMX GEMM: A-gather traffic thesis FALSIFIED — stop optimising this kernel

Status: closed 2026-09-24 after five measured attempts. Read this before proposing anything here.

`iq3_s` reordered, 16-row tile, pp2048, `vtune -collect gpu-hotspots`. All arms verified correct
(`test-backend-ops` 933/935; the 2-3 failures are a pre-existing f16 range overflow on q8_0
at amax=1e5, identical by shape name in every arm including the baseline - see "The 2-3
MUL_MAT_ID failures are an f16 range overflow" below. They are not flaky CPY, and the *count*
is not a usable signal because one shape wanders between devices.)

| arm                        | avg ms | spill | L3 GB/s | GB moved | ALU1 | XMX  | pp2048 |
|----------------------------|--------|-------|---------|----------|------|------|--------|
| baseline (device gather)   | 4.439  | 0     | 1868.6  | 8.29     | 6.3% | 1.8% | 642.8  |
| M-tile 32 (bit 64)         | 4.263  | 0     | 1725.6  | 7.36     | 6.3% | 1.9% | 652.0  |
| M-tile 64 (bit 256)        | 4.689  | 0     | 1066.1  | 5.00     | 5.2% | 1.5% | 617.9  |
| SLM A staging (bit 512)    | 9.525  | 0     | 278.9   | 2.66     | 1.7% | 0.3% | 444.8  |
| pipelined A (bit 1024)     | -      | -     | -       | -        | -    | -    | 642.8  |
| register-held A (bit 2048) | 4.906  | 0     | 482.4   | 2.37     | 6.5% | 1.1% | 628.3  |

**Cutting L3 traffic 40%, 68% and 71% by three independent mechanisms made the kernel slower every
time.** The register-held variant is the clean experiment: it captured the temporal re-fetch better
than SLM staging (2.37 GB vs 2.66), spilled nothing, left ALU utilisation unchanged at 6.5%, added
no SLM and no barriers - and still lost 10.5%. L3 traffic is not what this kernel is waiting on.

The earlier canonical-vs-SoA comparison (-30% traffic, -16% time) was used to argue the opposite.
That attribution was wrong: those two paths differ in address arithmetic and instruction count as
well as traffic, so the win was never isolated to traffic. Do not cite it as evidence again.

Pipelining the A stage (double-buffered `tile_a`, barrier moved off the dependency) measured
**exactly flat**. Its author predicted this in advance: an Intel XVE is in-order with register
scoreboarding, so removing the barrier makes the overlap *legal* but does not make IGC schedule the
decode chain below the MADs.

What is actually true of this kernel: execution units are idle ~92% of the time (ALU1 6.3%, XMX
1.8%), spill is 0, and no traffic reduction helps. The VTune array/stall columns are pegged
(`Active = Stalled = 100%`, `Idle = 0%`, `SBID = DistAcc = 100%`) and cannot be used to find the
real stall. Anyone picking this up needs an instrument that works - IGC's `-print-reg-usage`,
a shader-assembly dump, or a microbenchmark - not more guesses from these counters.

The only change that has ever helped is the 32-row M tile, +1.4%. Everything else in the design
space we explored loses. **Banked result for the whole kernel line: device gather + M-tile 32 =
577.6 -> 652.0 pp2048, +12.8%, both bitwise-verified.** Bits 256, 512, 1024, 2048 are default-off
regressions kept only as documented negative results; strip them before upstreaming.

## Grouped XMX GEMM: REOPENED and solved — the A load was 32 SIMD1 SLM messages

Status: the "stop optimising this kernel" verdict above is **superseded** as of 2026-09-24. The
section stays because its measurements are correct and its reasoning is the useful part of the
record; only its conclusion was premature.

Everything that section says is true: five mechanisms cut L3 traffic 40-71% and every one lost.
What it missed is that **the A load was never going through L3 at all.** `tile_a` lives in SLM, and
no L3 counter in VTune can see SLM traffic. So all five arms were reducing a quantity that was not
the constraint, and the invariant none of them touched was the one that mattered.

Offline IGC dump of the inner loop, all five arms: **32 `load.slm.d32x8t` messages per 8 `dpas`**.
Ratio exactly 4.0 in every arm. The cause is in the `joint_matrix_load` at fused-gemm.cpp:1179:

    mx::joint_matrix_load(sg, sub_a, tile_a...+ abase + (mt*FG_TM)*FG_BK + kt*FG_TK, FG_BK);

The stride argument is `FG_BK` = 32 halves = 64 B, while the tile being read is `FG_TK` = 16 halves
= 32 B wide. Each row of the 8x16 A fragment is therefore a discontiguous 32-byte run, and IGC has
no choice but to emit one SIMD1 message per row per k-step. Nothing about traffic volume; the bytes
were already in SLM. It is 32 messages to fetch what two block loads would fetch.

Fix (`GGML_SYCL_MMID_SCHED_BLK_A`, bit 32768): store A into `tile_a` already blocked by
XMX fragment, so each fragment is 256 contiguous halves and loads with stride `FG_TK`. Pure
addressing change - same bytes, same SLM size, no new barriers, no occupancy cost. Two helpers
(`fg_a_row_off`, `fg_a_tile_off`) template on the flag so the canonical path compiles unchanged.

| arm (sched)                  | pp2048 | kernel ms | L3 GB/s |
|------------------------------|--------|-----------|---------|
| production baseline          | 577.6  | -         | -       |
| device gather + M-tile 32    | 647.1  | 4.376     | 1839    |
| GRID_SLM alone (65536)       | 647.2  | -         | -       |
| **BLK_A (32769)**            | 661.5  | -         | -       |
| **BLK_A + GRID_SLM (98305)** | 662.6  | 4.177     | 1963    |

+2.4% over the previous best, **+14.7% over production**. Kernel time -4.5%. Note L3 read went
*up* 7% while the kernel got faster: final confirmation that L3 was never the mechanism here.

GRID_SLM (bit 65536, staging the dequant grid table into SLM) is worth **exactly nothing** alone
(647.2 vs 647.1) and ~0.2% on top of BLK_A. Its assembly shows why: it moves 8 messages from global
to SLM without reducing the count. Keep it only if `iq3xxs_grid` (1 KiB, half the SLM) reuses the
mechanism; otherwise strip it.

**Open regression: `sched=98369` (BLK_A + GRID_SLM + M-tile 32) is -15%**, although BLK_A alone and
M-tile 32 alone are each gains. At 32 rows `tile_a` is 16 KiB, and the blocked layout changes which
banks a sub-group's rows land in; a bank-conflict or an occupancy cliff are both plausible and
neither is checked. Do not ship the combination until this is understood.

### Method note for whoever comes next

The thing that broke this open was not a better idea, it was a working instrument. VTune's
occupancy and stall columns are pegged for these kernels and its L3 counters are blind to SLM.
The offline route needs no GPU and no lock:

    clang-offload-extract --stem=img build/bin/libggml-sycl.so.0.24.0
    IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir> ocloc compile -file img.NNN \
        -spirv_input -device bmg-g21 -options "-ze-intel-greater-than-4GB-buffer-required"

Then count messages per `dpas` in the inner loop. Seven arms had been measured before anyone did
this; the ratio was 4.0 in all seven and would have pointed here immediately.

## Queue after BLK_A

A kernel survey ranked the rest of decode by the same question - messages per unit of useful work -
rather than by time alone. Ordered by expected return:

1. **`get_int_from_table_16` byte gathers (IN PROGRESS).** `vecdotq.hpp:472`, `1878`, `1907` call
   the 4-argument overload, which does 8 scalar `values[q8[n]]` loads per call, 16 per `vec_dot` at
   `vdr_mmvq == 2` and 32 at the iq4_xs site. In assembly: 16 `load.ugm.d8u32.a64 (16|M0)` per 4
   `dp4a`, one byte per lane, zero model bytes, plus 23 of the kernel's 28 SIMD16 adds spent
   building their 64-bit addresses. A permute-based overload already exists at `vecdotq.hpp:129`
   (4 dword loads + `dpct::byte_level_permute`) and is **already in production** for `kvalues_mxfp4`
   at call sites 1138/1160/1161 - an earlier report calling it dead code was wrong. `kvalues_iq4nl`
   is the same `int8_t[16]` shape, so the swap is mechanical. 4.21 s of a 59.24 s decode capture is
   in the iq4_nl kernel alone; ~1.0-1.7 s looks recoverable, +0.55 s from iq4_xs.
   Unchecked: whether `byte_level_permute` lowers to ALU on Xe2, and signedness equivalence.
2. **56-byte scratch round trip in the unordered MoE mat-vec.** `ggml_sycl_mmvq_moe_multi`,
   `mmvq.hpp:36-40`. 2.57 s, ~1.6-2.0 s recoverable.
3. **`q8_0_esimd_launch` narrow tail**, `dmmv.cpp:2079-2098`, ~1.05 s, plus a 16-message SIMD1 SLM
   reduction epilogue.
4. **Confirm BLK_A transfers to the other quant types.** It already covers iq4_nl / q8_0 / q4_K /
   q5_K / q6_K with zero new code, but only iq3_s is measured.
5. Resolve the 98369 regression above.

### BLK_A transfers to every type this model instantiates (item 4, closed)

No new GPU run needed - the VTune CSVs from the BLK_A run already contain every
`grouped_dequant_gemm_kernel` instance, and the census script that produced the table above had
simply filtered them out. Paired by instance count, which is identical across arms:

| type    | reorder | base us | BLK_A us | delta | GRID_SLM in this arm |
|---------|---------|---------|----------|-------|----------------------|
| iq4_nl  | yes     | 4.563   | 4.225    | -7.4% | no                   |
| iq3_s   | yes     | 4.376   | 4.157    | -5.0% | yes                  |
| iq4_xs  | no      | 5.817   | 5.628    | -3.2% | no                   |

iq4_nl is the clean isolation: BLK_A alone, no GRID_SLM, -7.4%, and it is the *largest* of the
three. iq4_xs gains with the reorder off, so the win does not depend on the reordered layout.
q4_K / q5_K / q6_K are compiled but this model (UD-IQ4_XS) never instantiates them at runtime, so
they stay unmeasured - not unsupported.

### The 98369 regression is not spill, and probably not BLK_A (item 5)

Offline census of all 635 IGC kernel dumps, filtered to the production flag combination (bits 3-7
clear): **not one instantiation spills a single byte**, at either M tile, with or without BLK_A.
BLK_A *lowers* instruction count at both tiles, iq3_s reordered:

| M tile | BLK_A | GRID | instCount |
|--------|-------|------|-----------|
| 16     | 0     | 0    | 1001      |
| 16     | 1     | 1    | 946       |
| 32     | 0     | 0    | 1810      |
| 32     | 1     | 1    | 1650      |

So the -15% is not code quality. Every spilling instantiation in the whole dump has bit 4 or bit 5
set (STAGE_A / PIPELINE_A) - the already-rejected arms - which is a useful independent confirmation
that those two were the wrong shape, and that nothing on the shipping path is register-starved.

That leaves occupancy. Per-work-group SLM, against the 128 KiB per Xe-core:

| sched | M tile | tile_a + tile_c | grid | total  | resident WGs |
|-------|--------|-----------------|------|--------|--------------|
| 32769 | 16     | 4 + 8 KiB       | -    | 12 KiB | 10           |
| 98305 | 16     | 4 + 8 KiB       | +2   | 14 KiB | 9            |
| 32833 | 32     | 8 + 8 KiB       | -    | 16 KiB | 8            |
| 98369 | 32     | 8 + 8 KiB       | +2   | 18 KiB | 7            |

98369 is the only arm that falls to 7, a 22% occupancy loss against the 9 of the current best -
the right order of magnitude for the 15% it loses. Under that reading the culprit is GRID_SLM's
2 KiB crossing a cliff, and **BLK_A + M-tile 32 without the grid (sched 32833) should not
regress.** That arm had never been run. Running it is the test, and it also decides GRID_SLM's
fate: 0.2% of throughput is not worth a resident work-group.

### Two corrections to the kernel survey

**Item 3 (`q8_0_esimd_launch` narrow tail) rests on a premise that is already false.** The survey
cites the tail loop at `dmmv.cpp:2088-2093` and "a 16-message SIMD1 SLM reduction epilogue". But
the dispatcher immediately below, `dmmv.cpp:2145-2153`, already scales the work-group to the row:

    if      (nblk_row >= 64) q8_0_esimd_launch<8>(...);
    else if (nblk_row >= 32) q8_0_esimd_launch<4>(...);
    else if (nblk_row >= 16) q8_0_esimd_launch<2>(...);
    else                     q8_0_esimd_launch<1>(...);

and carries the comment "too many of them over a short row makes the barrier and the lane-0
epilogue the whole cost". At the ncols=320 the survey quotes, nblk_row = 10, so **WG = 1**: the
lane-0 epilogue is two loads, not 32, and the barrier is a single thread. Someone already fixed
this. Demoted to the end of the queue; it needs a decode capture that actually names the
instantiated `WG` before anyone touches it.

**A prefill capture cannot be read without separating model load from steady state.** The largest
"computing task" in the BLK_A capture is `zeCommandListAppendMemoryCopy`: 7.92 s over 6834
instances moving **103.7 GB host-to-device**, more than every kernel in the run combined. That is
not a bottleneck, it is the 87 GiB model being copied into VRAM at 13.1 GB/s during load, plus a
one-time 16.5 GB coming back. The one-time weight reorders are in there too
(`reorder_qw_iq3_s_moe`, 94 instances, 1.01 s - second-largest kernel in the capture and it runs
at load). The actual steady-state prefill ladder is iq3_s GEMM 1.53 s, iq4_nl GEMM 0.73 s,
gated_delta_net 0.55 s, pack_b 0.43 s, oneMKL gemm 0.34 s. Note also that
`Computing Task:Total Time` in these CSVs is in **seconds**, while the per-kernel tables above
quote `Average Time` as if in ms - the numbers are consistent, the label is not.

## Unordered MoE mat-vec: 896 B/thread of private memory removed (queue item 2, code landed)

A by-value struct argument that is indexed with a runtime index cannot be register-allocated. IGC
copies it to private memory in the prologue and loads the wanted element back:

    // prologue, the whole 56-byte ggml_sycl_mmvq_moe_multi, offsets 0 / 0x20 / 0x30
    store.ugm.d32x8t.a64 (1|M0)  [r10:1]        r9:1    // address space: private
    store.ugm.d32x4t.a64 (1|M0)  [r12:1+0x20]  r11:1    // address space: private
    store.ugm.d32x2t.a64 (1|M0)  [r12:1+0x30]  r13:1    // address space: private
    ...
    add  r10.0:q  r126.0:q  r8.0:q                      // private base + mat_idx*8
    load.ugm.d64x4t.a64 (1|M0)  r90:1  [r10:1]          // vx_base[mat_idx], dst_base[mat_idx]

`zeinfo` reported `private_size: 896` for all 22 instantiations of `mul_mat_vec_q_moe`; the
sibling `..._ordered_impl` kernels, which take only scalars, have none. **Not spill** - no
`//.spill size` line anywhere, `numGRF=128` unchanged. Worse than the four messages, the prologue
was serialized: no weight or activation address could be formed until that private load returned.

Fix: pass the three `vx_base` / `dst_base` pairs as six scalar kernel arguments and select with two
comparisons. Unflagged - pure addressing, `n_mats` is already bounded to 3 by the `GGML_ASSERT` at
`ggml-sycl.cpp:5580`, so 0/1/2 map exactly as `[mat_idx]` did.

| | before | after |
|---|---|---|
| `address space: private` ops | 66 (3 x 22) | 0 |
| private runtime-offset load | 1/kernel | 0 |
| uniform `(W) *.ugm` messages | 8/kernel | 4/kernel |
| `private_size` | 896 | absent |
| instCount, 22 kernels | 39482 | 38743 (-1.9%) |

**A bounded `#pragma unroll` select over the struct does NOT work, and this is the useful part.**
That shape is already used at `mmvq.cpp:4332`, and it made things worse: instCount rose 10-16 per
variant and the private load *doubled*, because IGC folds select-of-loaded-values back into
select-of-index plus one variable GEP and then branches, serving only the `mat_idx == 0` arm from
registers. Constant indices are not enough while the data is still a byval struct argument - the
argument has to stop being a struct.

### Follow-up: `mul_mat_vec_q_reorder_multi` has the same defect, 32 kernels

`mmvq.cpp:4315-4339` takes `ggml_sycl_mmvq_multi` by value and reads `mats.vx[mat]`,
`mats.nrows[mat]`, `mats.row_begin[mat]`, `mats.dst[mat][...]` and `mats.stride_col_dst[mat]` with
a runtime `mat`. 32 instantiations still carry 204 private-tagged ops. Five arrays means the
scalar-argument fix is ~15 parameters (120 B, well inside the kernel argument limit) - ugly but
mechanical, and per the paragraph above there is no cheaper shape that actually works. Larger prize
than the one just fixed. Not yet done.

Unrelated leftover: 6 `..._ordered_impl` instantiations carry 152 private-tagged ops between them
although they take only scalars, so the cause there is a local array inside some `vec_dot` path.
Uninvestigated.

## `get_int_from_table_16`: gathers gone, but Xe has no byte-permute instruction (queue item 1)

Code landed in `vecdotq.hpp` (sites 470, 1873, 1900 now use the `sycl::int2` permute overload).
**Bit-exactness is proven, not assumed** - three independent legs:

- `dpct::byte_level_permute` (`dpct/helper.hpp:2976-2987`) masks every selector nibble with `& 0x7`,
  so bit 3 is ignored: there is no CUDA sign-replicate mode to worry about, and only the low 16 bits
  of the selector are read, which is why `q4 >> 16` sign-extending a negative `int` is harmless.
  Each output byte depends on exactly one input nibble in *both* implementations and both move the
  table byte verbatim, so the `int8_t` / `uint8_t` difference is a reinterpretation that `dp4a`
  reads as signed either way - signedness never enters.
- Host replica, 80,557,056 cases against both `kvalues_iq4nl` (signed -127..113) and
  `kvalues_mxfp4`: 0 mismatches. Exhaustive rather than sampled, because of the one-nibble
  dependency above.
- Upstream CUDA already made this exact substitution for these exact two functions:
  `ggml-cuda/vecdotq.cuh:1344` and `:1366` pass `kvalues_iq4nl` to the permute form, and its MUSA
  fallback at `:82-93` is literally our byte-gather version.

**But the trade is not free, and probably not even favourable.** `__byte_perm` is one `prmt` on
NVIDIA and one `v_perm_b32` on AMD. **Xe has no byte-permute instruction**, so
`dpct::byte_level_permute` compiles to its literal C++: `(uint64)b << 32 | a` then a variable 64-bit
`shr` per output byte. Inner loop of the 4.21 s `iq4_nl` kernel (`entry_0289`, loop `_0_017`):

| | before | after |
|---|---|---|
| `load.ugm.d8u32.a64` in loop | 16 | **0** |
| total memory messages in loop | 22 | **8** |
| 64-bit `shr` / `shl` in loop | 0 / 0 | **48 / 37** |
| total instructions in loop | 140 | **340** |
| distinct SWSB tokens used | **16, all of $0-$15** | 8 |
| `dp4a` in loop | 4 | 4 |

`iq4_xs` (`entry_0271`): loop messages 41 -> 15, instructions 272 -> 649.

Two things the census turned up that the plan did not predict:

- The 4-dword table load is not hoisted, it is **eliminated**. The preheader holds
  `0x7159453526190D01` and `0xF6EADDCFBFAD9881` as 64-bit immediates - the two halves of
  `kvalues_iq4nl`. Zero table traffic, zero table address arithmetic.
- **The before loop burns all 16 SWSB scoreboard tokens on the gathers**, leaving none to keep the
  weight and activation loads in flight. After, 8. That is a structural argument the instruction
  count does not show, and it is the reason this might still win despite tripling the ALU.
- Regression to watch: the two `q8_1` quant loads de-coalesced, `d32x2 x2` -> four separate `d32`.
  Real-data messages actually went 6 -> 8; the 22 -> 8 headline is (6+16) -> (8+0).

Being A/B'd now with the mmvq scalar-argument fix held constant in both arms, so the delta is
attributable to this change alone.

### If the ALU cost wins, the Xe-shaped answer is a nibble-pair table

`T2[b] = T[b & 0xf] | (T[b >> 4] << 8)`, 256 entries of `uint16` = **512 bytes**. The index is then
a raw byte, so it is 4 `d16u32` gathers per 32-bit word instead of 8 `d8u32`, packed with
static-stride byte `mov`s and **no 64-bit shifts at all**: roughly 22 -> 14 messages at flat ALU.
Note this does **not** need a new table in `ggml-common.h` and therefore is not a ggml core change:
it is only ever read by SYCL `vec_dot`, so it belongs in `vecdotq.hpp` next to the function that
uses it, and can be built at namespace scope from the existing `kvalues_iq4nl`.

**Independently: the shipping mxfp4 and nvfp4 paths pay this same 64-bit-shift cost today**
(`vecdotq.hpp:1138`, `1160`, `1161`), and nobody has ever measured them against the byte-gather
form. If the A/B says gathers win, those three sites are a free regression waiting to be found.

## The 98369 answer: after BLK_A, SLM is the binding constraint and the M tile is dead weight

My occupancy prediction above was **recorded before the run and then falsified**: I predicted
sched 32833 (BLK_A + M-tile 32, no grid table) would land between 647 and 663. It measured
**615.28**, -7.4% against 98305 and -4.4% against a plain 643.85 baseline. So the regression is not
GRID_SLM's 2 KiB crossing a cliff. Keeping the wrong prediction here on purpose: the arm was worth
running precisely because it could fail, and it did.

What the assembly says. BLK_A is **strictly better on every static metric at both tiles** - it is
not the case that the blocked layout misbehaves at 32 rows:

| iq3_s reordered   | M16 base | M16 BLK_A | M32 base | M32 BLK_A |
|-------------------|----------|-----------|----------|-----------|
| load.slm (kernel) | 36       | **8**     | 72       | **16**    |
| store.slm         | 40       | 36        | 82       | 72        |
| load.ugm          | 20       | 20        | 35       | 35        |
| barriers          | 4        | 4         | 12       | 12        |
| instCount         | 1001     | 924       | 1810     | 1641      |
| inner loop        | 32 load.slm / 8 dpas | **4 / 8** | 64 / 16 | **8 / 16** |

The blocked loads even coalesce into single `d32x64t.a32` messages - one 256-byte message per whole
XMX fragment, which is the ideal. Nothing static is worse. So the cost has to be dynamic, and it
tracks the resident work-group count:

| sched | SLM    | resident WGs | pp2048 | vs 98305 | WG ratio |
|-------|--------|--------------|--------|----------|----------|
| 32769 | 12 KiB | 10           | 661.5  | -0.5%    | 1.11     |
| 98305 | 14 KiB | 9            | 664.7  | -        | 1.00     |
| 32833 | 16 KiB | 8            | 615.3  | -7.4%    | 0.89     |
| 98369 | 18 KiB | 7            | 551.1  | -17.1%   | 0.78     |

**The explanation: a wider M tile exists to amortize the A staging over more MAD work, and BLK_A
just removed 8/9ths of the MAD phase's message cost.** There is no longer anything for the wider
tile to amortize, so all it does is spend 4 KiB of SLM and 20% of the occupancy. Same argument for
GRID_SLM's 2 KiB. That is why each is a gain alone and a loss on top of BLK_A - they are not
interacting badly with the blocked layout, they are paying for a service nobody needs any more.
It also fits the earlier microbenchmark (decode 58%, MADs 36%, barely overlapping): shrink the MAD
phase 8x and the staging phase is simply the kernel.

Note the curve saturates upward: 10 resident work-groups (661.5) is not better than 9 (664.7). So
9-10 is enough to hide what is left, and the loss below 8 is steep.

### Consequences - this reorders the whole queue

1. **Ship BLK_A at M-tile 16. Do not ship M-tile 32 with it.** Bit 64's +1.4% was measured before
   BLK_A existed and does not survive it. Its banked "+12.8%, device gather + M-tile 32" result
   above is now obsolete as a recommendation, though still correct as history.
2. **GRID_SLM (bit 65536) is on probation.** 664.7 vs 661.5 is inside run-to-run noise for 2 KiB of
   SLM. Passes 2 and 3 decide it; if it does not clear noise, strip the bit. The `iq3xxs_grid`
   reuse idea inherits this and should not be built on a 0.5% result.
3. **New top item: SLM is now the binding constraint, so attack `tile_c`.** Of the 12 KiB at M-tile
   16, `tile_a` is only 4; `tile_c` is **8 KiB** (`FG_KSPLIT * FG_SG_ROWS * FG_BN` floats =
   4*16*32*4). Every KiB removed from it is now worth more than any further work on the A path. Two
   shapes worth trying: reduce `FG_KSPLIT`'s footprint by reducing the K-split partial sums through
   sub-group shuffles instead of through SLM, or hold the partials as halves where the accumulation
   order allows. Getting to 8 KiB total would give 16 resident work-groups.
4. RESOLVED: the 932/935 vs 933/935 discrepancy on 98369 was a false alarm. The failing shapes are
   named and identical across arms - a pre-existing f16 overflow, see below. The occupancy table
   stands and BLK_A introduces no failures at either tile size.

## Upstreaming: what the MMID_SCHED bitmask should look like after today

The branch carries 17 bits. Most are experiments that have now been decided, and two produce
deliberately wrong output. A reviewer seeing all 17 will reject the patch on sight, so the strip
list matters as much as the win. Verdicts, all from measurements in this file:

| bit | name | verdict | evidence |
|-----|------|---------|----------|
| 0  | DEVICE        | **keep, shipping** | it is the production baseline (sched=1) |
| 1  | UNBOUNDED     | keep, undecided | never isolated; decide or strip |
| 2  | VERIFY        | strip or move behind a debug macro | reads the schedule back and checks it |
| 3  | VERIFY_DST    | strip or move behind a debug macro | runs both arms, compares bitwise |
| 4  | HOST_TABLE    | strip | an A/B scaffold for bit 0, not a feature |
| 5  | PACKB_TIGHT   | keep, undecided | never isolated |
| 6  | MTILE32       | **strip** | +1.4% alone, **-7.4% on top of BLK_A**; obsolete |
| 7  | GRF256        | **strip** | only ever needed by the wide tiles; at M16 the shipping path is numGRF=128 with zero spill |
| 8  | MTILE64       | **strip** | 617.9 vs 642.8 |
| 9  | SLM_A         | **strip** | 444.8 - catastrophic |
| 10 | PIPELINE_A    | **strip** | exactly flat, and its author predicted that in advance |
| 11 | REGS_A        | **strip** | 628.3 vs 642.8 |
| 12 | REGS_LITE_A   | **strip** | never cleared baseline |
| 13 | DIAG_NO_MAD   | **strip, mandatory** | produces wrong output by design |
| 14 | DIAG_NO_A     | **strip, mandatory** | produces wrong output by design |
| 15 | BLK_A         | **keep — and make it unconditional, not a flag** | +2.2%, -5.0/-7.4/-3.2% kernel time across three types, strictly fewer messages and instructions, no spill, no SLM cost |
| 16 | GRID_SLM      | probation | 0.5% for 2 KiB; the interleaved 6-rep run decides it |

BLK_A should not ship as a flag at all. It is a pure addressing change with no cost term to trade
off - same bytes, same SLM, same barriers, fewer messages, fewer instructions, bit-identical output.
A flag would only invite someone to turn it off. The `fg_a_row_off` / `fg_a_tile_off` helpers stay
templated only for as long as the negative results above need to stay reproducible.

That reduces the upstream diff to: the two offset helpers, the blocked store, the matching
`joint_matrix_load` stride, and the comment explaining why the stride was the bug. Bits 6-14 come
out with it, which also deletes `fg_slm_x_region`, the `STAGE_A` / `PIPELINE_A` / `REGS_A` /
`REGS_LITE` template parameters and the `diag_no_a` / `diag_no_mad` kernel parameters - a net
*reduction* in the file, which is the easiest kind of patch to get reviewed.

Do not strip anything until the negative results in this file are considered permanently recorded;
the table above is the record, and the bits are only its scaffolding.

### Confirmed, three passes each (the table above was pass 1 only)

| sched | what                        | pass 1 | pass 2 | pass 3 | mean   | vs base |
|-------|-----------------------------|--------|--------|--------|--------|---------|
| 1     | production baseline arm     | 643.85 | 647.54 | 648.01 | 646.47 | -       |
| 65    | device gather + M-tile 32   | 650.68 | 653.09 | 651.81 | 651.86 | +0.8%   |
| 98305 | **BLK_A + GRID_SLM**        | 664.74 | 661.50 | 662.40 | **662.88** | **+2.5%** |
| 32833 | BLK_A + M-tile 32           | 615.28 | 604.64 | 615.13 | 611.68 | -5.4%   |
| 98369 | BLK_A + M-tile 32 + GRID    | 551.14 | 549.94 | 551.16 | 550.75 | -14.8%  |

Baseline spread is 4.2 points and the best arm's is 3.2, so both regressions are far outside noise.
Against the 577.6 production number this is **+14.8%**. M-tile 32 alone is a consistent small gain
in every pass (+0.8%), which is what makes its collapse on top of BLK_A a genuine interaction and
not a bad arm.

## The 2-3 MUL_MAT_ID failures are an f16 range overflow, pre-existing, and not the flaky CPY

Every arm in today's runs reports 932-933 of 935, and I twice described those as "the pre-existing
flaky q8_0 CPY baseline". **That was wrong** - the scripts run `test-backend-ops -o MUL_MAT_ID`, so
no CPY test is ever executed. Naming the failures settled it:

    [MUL_MAT_ID] NaN at index 0 (SYCL0=-nan CPU=-209406.843750)
      MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=8,n_used=2,b=0,m=512,n=32,k=256,amax=100000.000000): FAIL

`amax=100000` is the whole story. The build is configured `-DGGML_SYCL_F16=ON`
(`examples/sycl/mybuild.sh`), and `ggml-sycl.cpp:3219` opens an `#ifdef GGML_SYCL_F16` block that
converts **both** operands to f16 - src1 at `:3255`, src0 at `:3282`. The largest finite f16 is
65504. A tensor whose values reach 1e5 becomes inf on conversion, and the accumulation turns inf
into NaN. The CPU reference stays in f32 and returns the finite 1e5-1e6 values shown above.

Consequences:

- **It is pre-existing and independent of everything measured today.** It reproduces at sched=1, the
  baseline arm, on all three devices.
- **The 932 vs 933 discrepancy I flagged on sched 98369 is per-device, not per-arm.** In a single
  baseline run SYCL0 and SYCL1 report 933 and SYCL2 reports 932: `n=32` and `n=64` fail everywhere,
  `n=16` fails only on SYCL2. The arm scripts take `grep "tests passed" | head -1`, i.e. whichever
  backend prints first, so the number they print is not even from a fixed device. The occupancy
  table stands unchanged, and BLK_A introduces no failures.
- Any future arm script should report all three backends, or filter to one, rather than `head -1`.
- Not worth fixing for this branch: it needs f32 accumulation or a range guard on the f16 staging,
  it only bites above 65504, and real model weights are nowhere near that. But it should not be
  described as flaky, because it is not - it is deterministic per shape, and only *which* shapes
  route to the f16 path varies.

**Amendment after three reps.** "Deterministic per shape" is only two thirds right. Across reps the
count moves *between devices*:

| rep | SYCL0 | SYCL1 | SYCL2 | extra failing shape |
|-----|-------|-------|-------|---------------------|
| 1   | 933   | 933   | 932   | n=16 on SYCL2       |
| 2   | 933   | 933   | 932   | n=16 on SYCL2       |
| 3   | **932** | 933 | 933   | n=16 on **SYCL0**   |

`n=32` and `n=64` fail on every device in every rep - those are the plain f16 overflow. `n=16` fails
once per run on *a* device, and which device changes. So the total for identical code is 932 or 933
depending on nothing the caller controls.

That is the real lesson for the arm scripts: **a pass-count comparison between arms is not a valid
correctness signal here, in either direction.** 98369's 932 was a false alarm, and an arm that
genuinely broke something could equally hide inside the same one-test wobble. Compare the failing
shape *names*, not the count.

## SPLIT_C (bit 17): tile_c halved, work-group to 8 KiB, full thread occupancy

Code ready, patch at `scratchpad/split_c.patch`, awaiting GPU. Measure `32769` vs **`163841`**.

The occupancy ladder above says the kernel is now SLM-bound, and `tile_c` was 8 of the 12 KiB. The
implementation is not any of the three shapes I proposed: it reduces the K-split partials **one
accumulator tile (`FG_TM` = 8 rows) at a time instead of `FG_SG_ROWS` = 16**, which halves `tile_c`
by reusing the `PASSES` loop the epilogue *already runs* for the wide M tiles. `PASSES` goes 1->2 at
the 16-row tile, `MT_PASS` 2->1, and `MT_PASS * PASSES == MT` is unchanged, so every accumulator
tile is still stored exactly once. The whole change is `FG_SG_ROWS -> C_ROWS = fg_c_rows<SPLIT_C>()`
in five places plus one template bool.

| M tile | SPLIT_C | tile_a | tile_c | WG SLM | resident WGs (cap 16) | resident threads (of 64) |
|--------|---------|--------|--------|--------|----------------------|--------------------------|
| 16     | off     | 4096   | 8192   | 12288  | 10                   | 40 (62%)                 |
| 16     | **on**  | 4096   | **4096** | **8192** | **16**             | **64 (100%)**            |
| 32     | on      | 8192   | 4096   | 12288  | 10                   | 40                       |
| 64     | on      | 16384  | 4096   | 20480  | 6                    | 24                       |

Byte totals read out of the source by a compile-time probe that asserts them and lets the compiler
name them in a diagnostic, not by hand arithmetic.

**Why it should be trusted more than the usual "should be faster":**

- **The hot k loop is byte-identical.** Same instruction extents, **573 instructions in both arms**,
  identical opcode histogram, 8 SLM messages, 0 barriers, all 8 `dpas` inside. Every one of the
  +366 static instructions is in the epilogue, and they are *unrolling*: static `load.slm` 8->36 and
  `store.ugm` 1->8, but dynamic counts are identical (32 and 8) because the off arm loops 8 times
  with a back-edge and the on arm is fully unrolled into 4 bodies x 2 passes.
- **Bit-identical output by construction.** Summation stays `((p0+p1)+p2)+p3` ascending over
  `FG_KSPLIT`; only the stride between partials changes, 512 -> 256 floats. Verified by simulating
  both epilogue shapes over `SG_ROWS in {16,32,64}`, `M in {16,17,31,64,65,100}`, ragged `n1-n0` and
  every `m0`, comparing both the values *and the literal sequence of hex addends*: 677,966 dst
  slots, 0 mismatches. So a bitwise dst comparison is a valid acceptance bar here - unlike QSA top-k.
- **The default-off path is provably untouched:** all 586 `SPLIT_C=false` kernels are
  metric-identical to the pre-patch dump on instCount, spill, numGRF, load.slm, store.slm, sync.bar,
  dpas and store.ugm.
- No kernel goes from zero spill to nonzero. Of the 23 arms that already spill, 8 spill *less*.

Cost: **two extra group barriers** (`sync.bar` 1 -> 3, i.e. `2*PASSES-1`). Not a new barrier
pattern - MTILE32 and MTILE64 already run `PASSES > 1` with exactly these barriers today, and each
`sync.bar` was confirmed to sit after a `join` reconvergence point, so none is in divergent control
flow. Whether two barriers cost less than 24 extra resident threads earn is the measurement.

Also worth measuring: `229377` = GRID_SLM + SPLIT_C (14 -> 10 KiB, 9 -> 12 work-groups). Bit 17 has
no exclusions and stacks with everything.

**Known cost to the build, not to the kernel:** gating this as a template bool on every arm takes the
TU from 623 to 1209 device kernels and the AOT compile from ~12 to 20 minutes. If that becomes
annoying, restrict the `true` instantiation to the arms actually benchmarked.

### Candidate 2 (`FG_KSPLIT` 4 -> 2) is worse than a wash — do not try it

`FG_WG_SIZE = FG_KSPLIT * WARP_SIZE` (`fused-gemm.cpp:24`), so halving KSPLIT makes a work-group 32
work-items = **2** hardware threads. At 6 KiB the work-group cap binds first: `min(16, 21) = 16`
work-groups x 2 threads = **32 resident threads, below today's 40**, and each work-group also does
twice the K work. The SLM saving buys literally nothing. My own arithmetic on this was wrong in the
direction of optimism; this is the corrected version.

## MEASUREMENT TRAP: this session's own scratch files halved the benchmark

`/tmp` on this box is **tmpfs**, i.e. RAM: `findmnt -T /tmp` reports `tmpfs /tmp tmpfs 62.8G`. The
scratchpad had accumulated **20 GB of IGC shader dumps**, which is 20 of the machine's 80 GB, on a
host that streams an 87 GiB model through page cache with `--load-mode mmap --lazy-mode auto`.

The symptom: two arms that differ only in a shared-local-memory flag both measured **340** pp2048
where the same binary had measured 661 and 664 an hour earlier. Identical numbers for both arms is
the tell - a code-related regression does not hit two different arms to three significant figures.

Deleting 11 GB of superseded dumps freed **zero disk** (`df /` unchanged at 16 G) and 11 GB of RAM.
That is the diagnostic: if a delete in `/tmp` does not move `df`, `/tmp` is tmpfs and you were
spending memory, not disk.

Two false trails on the way, both worth knowing:

- `dmesg` showed `ata8.00: exception ... HSM violation` and `ata8: hard resetting link`, 336 such
  lines. Tempting, and wrong: `ata8` is `sde`, an ~4 GB device, while the model is on
  `/dev/mapper/arcprofast-vm--10015--disk--0`. Unrelated hardware. (This is the second time an `sde`
  false positive has cost time here.)
- I had also killed a `test-backend-ops` mid-run and suspected a wedged GPU context. Checked: exactly
  one process held `/dev/dri` fds, the live `llama-bench`. Not a wedge.

**Confirmed by recovery, which is the part that makes this a diagnosis rather than a guess.** The
degraded run measured 340.5, 340.8, 341.0, 342.3 across both arms - four readings within 1.8 of each
other for two configurations that differ by a shared-local-memory flag. Then **pass 3 came back at
661.31**, against the 661.5 this arm had measured earlier. So the sequence is: dumps grow to 20 GB ->
pp2048 halves -> 11 GB freed -> one more pass still degraded while the page cache refills -> full
recovery. Freeing the space is necessary but not sufficient; the cache has to repopulate.

I halted the pipeline one data point before that recovery landed, on the reasoning that four
identical readings meant nothing measurable could come out of the run. That was defensible on the
evidence at the time but it was one reading too early, and the recovery would have told me the cause
for free.

**Consequences for today's numbers.** The BLK_A occupancy ladder is unaffected - it was measured
before the two large dumps (3.4 GB and 1.6 GB) were written, which is why its arms were stable at
645-665 across three passes. The whole first GRID_SLM run is discarded, kept as
`gridtest.degraded.out`.

**Fixed, not just noted.** `scratchpad/guard.sh` now refuses to measure when the tmpfs scratchpad
exceeds 12 GiB or `MemAvailable` is under 40 GiB, and every arm script calls it first. `gridtest.sh`
also gained the `dd` page-cache warm over both shards plus a throwaway `llama-bench` that `occ.sh`
had and I had omitted. Check tmpfs usage before any timing run, not just `df`.

## reorder_multi: 204 private ops -> 0 across 32 kernels (queue item, code ready)

Patch at `/root/igcdumps-ae6fc5c/mmvq-scalar-args.patch`, `mmvq.cpp` only, applies clean. Same defect
and same fix shape as the MoE sibling above - 17 scalar kernel arguments selected by ternary chains
instead of a 96-byte by-value struct indexed with a runtime `mat`.

| metric (32 instantiations)     | before | after |
|--------------------------------|--------|-------|
| `address space: private` ops   | 204    | **0** |
| `private_size` (zeinfo)        | 1536, all 32 | **entry absent, all 32** |
| uniform `(W) *.ugm` per kernel | 20.38  | **6.50** |
| all `*.ugm` per kernel         | 48.12  | 34.25 |
| summed `.instCount`            | 14575  | **13089 (-10.2%)** |
| spill / numGRF                 | 0 / 128 | 0 / 128 |

**444 fewer messages, every one of them a uniform `(W)` message.** By the ranking rule that has held
all along on this workload - count messages eliminated, not instructions saved - this is the right
kind of reduction, and it is larger than the sibling's. The argument payload also got *smaller*
(92 bytes of scalars vs the 96-byte struct), which collapsed the baseline's two argument-payload
loads into one 128-byte message.

Three things worth carrying forward:

- **The `address space: private` tag undercounts.** Three private *loads* off the same base carry no
  such tag and are only findable by tracing the base register:
  `add r4.4:q  r1.6:q  48:w` then `load.ugm.d32x4t.a64 r15:1 [r14:1]` is `nrows[mat]` +
  `row_begin[mat]`. Grep the tag to find the problem, trace the base to size it.
- **The epilogue was worse than the prologue.** In the widest variant IGC reloads
  `stride_col_dst[mat]` **from the same address** once per output column and never CSEs it - seven
  redundant uniform messages in the store loop. That is why the per-kernel private-op count rose
  monotonically with `ncols_dst` (2,4,5,6,7,8,9,10).
- **I cited the wrong guard in the brief.** I said `ggml-sycl.cpp:5580`; that assert belongs to the
  *sibling* MoE path. This struct is guarded by an early `return false` at **`mmvq.cpp:4500`**
  (`n_mats < 2 || n_mats > GGML_SYCL_MMVQ_MULTI_MAX`), which is stronger than an assert because it is
  a fallback rather than a trap. Verified rather than assumed, which is the right outcome.

Still unmeasured on hardware: the -10.2% is static instruction count. Queued as stage 4, running the
same four arms as the SPLIT_C stage so that the only difference between the two result sets is
`mmvq.cpp`.

### Operational note: worktree isolation is landing on the wrong base

Both worktree agents got a worktree created off **`master` (0f8a414b7)**, not off
`performance_uplift_mega_branch` (702bc5472), so the functions they were sent to fix did not exist
there and `git diff` in the worktree is meaningless. Both worked around it by copying the working
tree in and handing back a patch file, and both had a `git checkout -B` / `git reset --hard`
refused by the permission classifier. The patches are fine and verified against the real base, but
**take the patch, never the worktree diff**, and expect to apply it yourself.

### Correction: the 661 -> 643 reading was cache warming, not CPU contention

I attributed a 643.50 pp2048 reading to CPU contention from a concurrent subagent compile (loadavg
7.8 on 16 cores) and added a hard load gate to the guard. **Both the attribution and the gate were
wrong**, and the evidence arrived from the run I had already killed:

    pass=1 sched=32769  : 643.50
    pass=1 sched=98305  : 661.10

Those two arms differ only by GRID_SLM's 2 KiB, and a clean measurement had them within 0.2 of each
other (661.5 vs 662.6). A 17.6-point gap *between consecutive arms of one run* is drift, not a
property of either arm: the first arm was still filling page cache and the second was warm. Two
independent reasons the CPU story does not hold up:

- `occ.sh` produced its stable 645-665 arms across three passes **while subagents were running AOT
  compiles**, so loadavg ~5 is demonstrably compatible with good numbers here.
- This box does not sit near 2 when idle. Nothing was consuming CPU (top consumer was this session
  at 4.8%, no compilers, no D-state I/O) and loadavg still hovered 4.8-5.6.

The hard gate was therefore protecting against an unproven 2.7% while risking something much worse:
every stage calls the guard, so a threshold the box never reaches would have aborted the entire
pipeline. Load is now a **warning only**; memory stays a hard fail because it is proven.

**The actual fix** is the warm-up, which was reading two of the model's three shards. All arm scripts
now `dd` all three (`00001`, `00002`, `00003`) plus a throwaway `llama-bench`. Even so, one throwaway
was not sufficient in the run above, so: **arms are interleaved across three passes and pass 1 is
discarded if it disagrees with passes 2-3.** Within-stage interleaving is what makes a comparison
survive drift; a single sequential sweep does not.

Second-order lesson, recorded because it cost real time: I spent several turns running
`for p in /proc/[0-9]*` scans and `ps` sweeps *to check whether the box was quiet enough to measure*,
which is itself load. Do not poll the machine you are trying to observe.

## MEASURED: SPLIT_C is worth +1.2%, and the new best is 671.5 pp2048 (+16.3%)

Three interleaved passes, pass 1 discarded per the warming rule, `mmvq` scalar-args and the `vecdotq`
permute present in all four arms so they cancel.

| sched  | config              | SLM    | resident threads | p1 / p2 / p3        | mean(2,3) | tg128 | vs BLK_A |
|--------|---------------------|--------|------------------|---------------------|-----------|-------|----------|
| 32769  | BLK_A               | 12 KiB | 40               | 489.0 661.3 661.1   | 661.20    | 30.81 | -        |
| 163841 | BLK_A+SPLIT_C       | 8 KiB  | 64               | 659.1 669.5 668.6   | 669.09    | 30.65 | **+1.19%** |
| 98305  | BLK_A+grid          | 14 KiB | 36               | 660.3 662.2 662.6   | 662.41    | 30.76 | +0.18%   |
| 229377 | BLK_A+grid+SPLIT_C  | 10 KiB | 48               | 669.4 670.9 672.0   | **671.47** | 30.71 | **+1.55%** |

**SPLIT_C is the gain: +1.19% alone, +1.37% stacked on the grid table.** GRID_SLM remains
+0.18-0.36%, i.e. inside noise, and the two are roughly additive rather than synergistic.

`229377` is the new best: **671.47 vs the 577.6 production baseline, +16.3%.**

The occupancy model predicted this and got the direction right: 40 -> 64 resident threads buys
+1.2%. One wrinkle to be honest about - 48 threads (671.47) edges out 64 (669.09), so the top of the
curve is slightly non-monotonic. That gap is 0.36% and those two arms also differ in where the
dequant table lives, so **there is no evidence for a real optimum at 48**; it would need an arm that
varies occupancy alone. Do not build on the "peak at 12 work-groups" reading.

Decode is flat at 30.65-30.81 across all four arms, as it should be: this GEMM's SLM occupancy has
nothing to do with the mat-vec kernels decode runs. Prefill-only change.

### Correction recorded on purpose

Mid-run I reported from pass-1 events that SPLIT_C alone was **-0.3%** and that GRID_SLM "earns its
keep only once SPLIT_C frees the SLM to pay for it", calling it a mirror image of the M-tile-32
story. Both claims came from `163841`'s pass 1 (659.14), a warming artifact; passes 2-3 are 669.5 and
668.6. The lesson is not new - it is the same warming rule already written above - but it is worth
recording that I quoted a conclusion off streaming pass-1 numbers *after* having written the rule
that says not to.

## MEASURED: reorder_multi is exactly neutral - a clean negative result

Identical four arms as the SPLIT_C stage, only `mmvq.cpp` differs, three interleaved passes, pass 1
discarded.

| config              | pp before | pp after | d pp   | tg before | tg after | d tg   |
|---------------------|-----------|----------|--------|-----------|----------|--------|
| BLK_A               | 661.20    | 661.42   | +0.03% | 30.81     | 30.73    | -0.24% |
| BLK_A+SPLIT_C       | 669.09    | 666.87   | -0.33% | 30.65     | 30.59    | -0.18% |
| BLK_A+grid          | 662.41    | 663.10   | +0.10% | 30.76     | 30.81    | +0.18% |
| BLK_A+grid+SPLIT_C  | 671.47    | 672.01   | +0.08% | 30.71     | 30.78    | +0.23% |
| **mean**            |           |          | **-0.03%** |       |          | **-0.00%** |

**This is the inverse of the lesson the rest of this file teaches.** BLK_A and SPLIT_C were found by
counting messages and the clock agreed. Here the static metrics improved more than either of them -
204 private ops to 0, 444 uniform messages removed, instCount -10.2%, argument payload 96 -> 92
bytes - and the wall clock did not move by more than noise.

Most likely reason is **coverage, not mechanism**: the 32 instantiations are q8_0 / q4_0 / q4_K /
q6_K, and this model is UD-IQ4_XS. Only the q8_0 shared expert can reach them, so the kernel is a
small share of decode however fast it gets. A secondary reason is that everything removed was
*uniform* per-work-group prologue work, which only matters if the kernel is launch-bound rather than
data-bound.

**Verdict: keep, do not claim.** Correctness-clean (MUL_MAT 1126/1126 on all three devices), strictly
fewer instructions, smaller argument payload, no behavioural change - so there is no reason to revert
it. But it is worth 0% on this workload and must not be cited as a win. If it is ever worth
re-measuring, do it on a q4_K or q6_K model where those instantiations are actually hot.

**Rule this adds to the file:** "fewer messages" predicts a win only for kernels that are *on the hot
path for the model under test*. Check which instantiations the model actually launches -
`GGML_SYCL_GG_TRACE` or a VTune computing-task list will say - before investing in a kernel's
assembly. The BLK_A work had this for free because iq3_s and iq4_nl dominate this model; this one did
not, and nobody checked.

## MEASURED AND REVERTED: the permute costs 7.4% of decode. Xe has no byte-permute instruction.

Four interleaved passes per arm, pass 1 discarded, at sched 163841. Arm P = `sycl::int2`
permute overload at `vecdotq.hpp` sites 472 / 1878 / 1907. Arm G = the original byte-gather overload.
Nothing else differs.

| arm                | pp2048 (p2-4)     | mean   | tg128 (p2-4)        | mean       |
|--------------------|-------------------|--------|---------------------|------------|
| P - permute        | 667.7 669.6 668.9 | 668.73 | 30.75 30.77 30.78   | 30.767     |
| G - byte gathers   | 667.4 667.9 668.5 | 667.92 | 33.29 33.18 33.23   | **33.233** |

**Byte gathers are +8.0% on decode. The permute costs 7.4%.** Prefill is unchanged (-0.12%), as it
must be - prefill runs the XMX GEMM, not `vec_dot`. Both arms pass MUL_MAT 1126/1126, so this is
purely a speed result; the change was always bit-exact.

**Reverted in the tree.** Sites 472, 1878 and 1907 are back on the byte-gather overload.

### Why, and why the static analysis pointed the wrong way

The implementing agent stated the risk plainly and I underweighted it. `__byte_perm` is a single
`prmt` on NVIDIA and `__builtin_amdgcn_perm` a single `v_perm_b32` on AMD, but **Xe has no
byte-permute instruction**, so `dpct::byte_level_permute` compiles to its literal C++: build
`(uint64)b << 32 | a`, then one variable 64-bit `shr` per output byte. The inner loop went from 140
to 340 instructions, of which 85 are 64-bit shifts. Sixteen single-byte gathers that all hit the same
16 L1-resident bytes are simply cheaper than that.

The message count said 22 -> 8 and instCount said +25-95%; the clock sided with instCount. The
SWSB argument for the change - that the old loop consumed all 16 scoreboard tokens on the gathers,
leaving none to keep weight and activation loads in flight - was a real observation and still nowhere
near enough to pay for ~200 extra ALU instructions per iteration.

**This is NOT a counterexample to the gather-bound rule - it is a mechanism result, and the
distinction is the valuable part.** An earlier measurement (2026-09-15) removed *these same 16 byte
gathers* from iq4_nl and got **1.49x** at **+21% instructions**. Tonight's attempt removed the same
gathers and got **0.93x** at **+140% instructions**. The difference is entirely what replaces them:

| mechanism                     | extra instructions | what they are          | result |
|-------------------------------|--------------------|------------------------|--------|
| register select tree          | +21%               | 32-bit `sel` + `shr`   | 1.49x  |
| `dpct::byte_level_permute`    | +140%              | **85 variable 64-bit shifts** | 0.93x  |

So: eliminating a small-table gather is worth real ALU, but only cheap 32-bit ALU. The refined rule:
count messages first, but price the replacement - a 64-bit shift per output byte is not in the budget,
three 32-bit selects are. If a change *adds* instructions to a hot inner loop, check their width and
rate before believing the message count.

### Consequence for tonight's other numbers

The permute was present in the SPLIT_C and reorder_multi stages, so **every decode figure in those
tables (~30.7-30.8) was depressed by ~7.4%. The branch's real decode is ~33.2.** Those stages'
*comparisons* stand - the regression sat in both arms of each - but their absolute tg128 values are
wrong and should not be quoted.

### Open lead worth taking: the shipping mxfp4 / nvfp4 sites pay this same cost

`vecdotq.hpp:1138`, `1160`, `1161` use the permute overload with `kvalues_mxfp4` and always have -
that was the pre-existing production path, not something added here. By this measurement they are
likely paying the same ~7% on any mxfp4 or nvfp4 model. **Nobody has ever measured them against a
byte-gather implementation.** This model is UD-IQ4_XS so mxfp4 never fires and it costs nothing here,
but on an mxfp4 model it is a free 7% waiting to be collected. Note the direction is the reverse of
this task: there the 4-argument byte-gather overload would need to be *added* for mxfp4, which is
easy since it is still present at `vecdotq.hpp:114` and now has three callers again.

### TOP OF THE QUEUE: the winning mechanism already exists, on another branch

`lookup_int8_16` and `get_int_from_table_16_reg` were written in commit **d29c700aa** and live on
`decode_multigpu_copy` and `sycl_dag_graph`. **`git merge-base --is-ancestor d29c700aa HEAD` says
they never landed on `performance_uplift_mega_branch`**, and `grep` confirms they are absent from the
working tree - which is why sites 472/1878/1907 still do byte gathers here:

    static __dpct_inline__ uint32_t lookup_int8_16(const uint32_t * table32, uint32_t n) {
        const uint32_t sel = n >> 2;
        const uint32_t a = (sel & 1) ? table32[1] : table32[0];
        const uint32_t b = (sel & 1) ? table32[3] : table32[2];
        const uint32_t w = (sel & 2) ? b : a;
        return (w >> ((n & 3) * 8)) & 0xFF;
    }

Three selects and one 32-bit shift per nibble. Port it to this branch and use it at 472 / 1878 / 1907.
Measured 1.49x on the iq4_nl kernel and 1.43x on iq4_xs when it was written; that kernel is 4.21 s of
a 59 s decode capture, so the headline is plausibly +2-3% decode - larger than anything else queued.

Also still open: the 512-byte nibble-pair table (`T2[b] = T[b&0xf] | (T[b>>4]<<8)`, 4 `d16u32` gathers
per word instead of 8 `d8u32`, no 64-bit shifts). Lower priority than the select tree, which removes
the messages entirely rather than halving them, and which already has a measurement behind it.

## CLOSED: the 16 byte gathers are fine. Neither replacement beats them. No change shipped.

Same stage, same binary lineage, four passes per arm, pass 1 discarded, sched 163841.

| arm                        | tg128 p1..p4              | mean(2-4)  | pp2048 |
|----------------------------|---------------------------|------------|--------|
| R - register select tree   | 33.01 33.34 33.29 32.92   | 33.183     | 667.87 |
| G - byte gathers (upstream)| 33.57 33.28 33.53 33.18   | **33.330** | 667.55 |
| P - byte_level_permute     | (earlier stage)           | 30.767     | 668.73 |

- select tree vs gathers: **-0.44% decode**, inside arm G's own 33.18-33.53 spread. Neutral.
- permute vs gathers: **-7.7% decode**.
- prefill unchanged in both, as it must be: `vec_dot` is not on the prefill path.

**Both mechanisms for removing those 16 gathers fail to beat keeping them.** `vecdotq.hpp` is reverted
to the committed state - `git diff` on it is empty. The survey's #1 ranked item produced no change.

### What this actually means, and the correction it forces

The premise was that 16 `load.ugm.d8u32.a64` per 4 `dp4a` - one byte per lane, zero model bytes, plus
23 of 28 SIMD16 adds spent building their addresses - had to be costing something. It does not. Those
16 messages all hit **the same 16 bytes**, which stay in L1, and the kernel evidently has the issue
slots to absorb them. Removing them with cheap 32-bit ALU buys nothing; removing them with 64-bit
shifts costs 7.7%.

**The 1.49x in the 2026-09-15 note does not contradict this - it measures a different thing.** That
was `test-backend-ops perf` on the kernel in isolation; this is end-to-end decode where the kernel is
~7% of the time and the rest of the pipeline can hide it. A 1.49x isolated-kernel win is an upper
bound of ~+3% end-to-end, and it materialised as 0. Keep both numbers, and never quote an isolated
kernel speedup as an end-to-end expectation.

**So the nibble-pair table idea is also dead**, not merely deprioritised: it aims at the same 16
messages that have now been shown to cost nothing. Do not build it.

**And the mxfp4/nvfp4 lead survives, with its size revised.** `vecdotq.hpp:1170`, `1192`, `1193` still
use the permute overload on `kvalues_mxfp4`. Tonight shows the permute is 7.7% worse than gathers on
*this* kernel shape, so those sites are plausibly leaving a few percent on the table for mxfp4 models -
but the honest expectation is now "the permute is bad", not "removing the gathers is good": the fix
there is to *add* the byte-gather overload (still present at `vecdotq.hpp:147`), not to chase message
counts. Untested, and this model never fires it.

## VERIFIED against the real production default, and wired into the run scripts

Every baseline measured tonight was `sched=1`. Production leaves `GGML_SYCL_MMID_SCHED` unset and
`ggml-sycl.cpp:426` defaults it to **0**, so production was getting none of this work - not even the
device-side schedule. Measured on the same binary, three interleaved passes, pass 1 discarded:

| sched   | pp2048 p1/p2/p3          | mean   | tg128 p1/p2/p3      | mean  |
|---------|--------------------------|--------|---------------------|-------|
| 0       | 568.8 571.6 575.6        | 573.58 | 32.96 32.96 33.48   | 33.22 |
| 229377  | 669.7 669.8 670.2        | 669.98 | 33.54 33.22 33.40   | 33.31 |

**Prefill 573.58 -> 669.98 = +16.8%. Decode +0.3%, i.e. unchanged**, which is correct: this bitmask
only touches the prefill GEMM. (The 577.6 figure quoted throughout this file came from earlier notes;
the measured value on tonight's binary is 573.58, close enough that no conclusion changes.)

`229377` = bit 0 (device schedule) + bit 15 (BLK_A) + bit 16 (GRID_SLM) + bit 17 (SPLIT_C). Now
exported by both `/root/run_llama_flash_next.sh` and `/root/vtune_llama_flash_next.sh`, with the bit
meanings and the do-not-add-MTILE warning in a comment beside it.

**CAVEAT: the enabling code is uncommitted** - `common.hpp`, `fused-gemm.cpp`, `fused-gemm.hpp`,
`ggml-sycl.cpp`, `mmvq.cpp`. If the tree is reset, bits 15-17 cease to exist and the env var becomes
**silently inert** rather than an error. Commit before relying on the scripts.

### Remaining queue, needing a decision rather than a measurement

1. Commit BLK_A + SPLIT_C as a pair (user's call; not done).
2. Strip bits 6-14 for upstream - five measured regressions plus two deliberately-wrong-output
   diagnostic bits. Makes the patch a net code reduction. Deliberately not done: it deletes the
   scaffolding for the negative results recorded in this file, and that is a judgement call about
   whether this file is now a sufficient record.
3. mxfp4/nvfp4 permute sites - cannot be tested on this model (UD-IQ4_XS never fires them).
4. BLK_A on q4_K / q5_K / q6_K - **blocked, and not for want of a q4_K model.** The cache holds
   `unsloth/Qwen3.8-27B-GGUF` in Q4_K_M and Q4_K_S, but a clean parse of all 51 GGUF KV pairs shows
   `general.architecture = qwen35`, `block_count 65`, `feed_forward_length 17408` and **no key
   containing "expert"** - it is dense. BLK_A lives in the grouped `MUL_MAT_ID` path, which a dense
   model never invokes, so neither cached q4_K model can exercise it at all. Confirming this needs a
   **q4_K MoE** GGUF, which is not present. Note also that root is at 96% with ~16 GiB free, so
   fetching one is not currently possible without freeing space first. Secondary requirement when it
   happens: q4_K on the XMX gather path also needs `GGML_SYCL_FAST_AND_SLOPPY=1`
   (`fused-gemm.hpp:54`), so the arm must set both.
   (Parse caveat worth repeating: my first attempt at this read only 4 elements of each GGUF array
   without skipping the rest, desynchronised the stream and threw - and it had *already printed*
   plausible-looking architecture fields before dying. A partial GGUF parse will happily tell you
   there are no expert keys when it simply never reached them.)

## DECODE IS HOST-BOUND ON ONE THREAD, IN THE LEVEL ZERO RUNTIME. ~3.5x headroom.

Measured 2026-09-25 at sched 229377, tg128 = 33.5 t/s. This supersedes the earlier "decode is
submission-bound, host 100% asleep" note: the host is **not** asleep in llama-bench, and the earlier
"0 in R" sampling was `llama-server` with `--poll 0`. The two differ.

**One host thread is pegged; every other thread is idle.** Per-thread CPU deltas over 10 s of decode:
one TID accumulated **997 of 1000 jiffies (99.7% of a core)**, `wchan=0` (genuinely running), and no
other thread accumulated any. 15 of 16 cores idle. That is a serial bottleneck at 6% of machine CPU,
which is why `-t 1/2/4/8/16` was flat (18.62/18.83/18.85/18.79/18.36) - **that flatness was never
evidence against host-boundness, and I misread it as such for a long time.** Extra worker threads
cannot help when the critical path is one orchestration thread.

**That thread is mostly inside Intel's runtime, not llama.cpp.** `perf -e cpu-clock`, self time by DSO:

| DSO                              | self%  |
|----------------------------------|--------|
| `libze_intel_gpu.so` (L0 driver) | 36.3   |
| `libc.so.6`                      | 15.4   |
| `libur_adapter_level_zero.so`    | 9.9    |
| `libsycl.so`                     | 8.8    |
| **`libggml-sycl.so`**            | **7.9** |
| `libigc.so`                      | 6.3    |
| kernel                           | 6.3    |

Flat symbol profile, nothing above 3.66%; the named symbol is
`createEventAndAssociateQueue(ur_queue_handle_t_*, ...)` - per-dispatch UR event creation. No
profiler was attached for this run, so the event cost is real and not VTune forcing event profiling.

**Per-token budget**, from differencing a 64-token and a 256-token profiled run so the model load
cancels (the load is 107 GB of H2D and dominates the raw totals - it fooled me once for prefill and
once again here before I differenced):

| per token                          | value        |
|------------------------------------|--------------|
| kernel dispatches                  | **2609**     |
| kernel time, summed over 3 devices | 25.72 ms     |
| kernel time per device             | **8.57 ms**  |
| host<->device copies               | 95 (~73 KB each) |
| copy time                          | 0.70 ms      |
| H2D / D2H bytes                    | 3.04 / 3.88 MB |
| wall                               | 29.85 ms     |

So: **29% GPU, 2% copies, 69% host dispatch overhead.** 29.85 ms / 2609 dispatches = **11.4 us of host
work per dispatch**, and 2609 x 11.4 us accounts for the whole token. Copies are NOT the bottleneck
despite there being 95 of them. Neither is bandwidth (<8% of VRAM) nor ALU (~8%) nor XMX (~2%).

**Headroom: the GPUs' current 8.57 ms/token would support ~117 t/s.** Today is 33.5. So ~3.5x is
available from dispatch cost alone, with no kernel made faster.

### Why this retroactively explains a lot

- Tonight's two mat-vec optimisations measured **0%** (reorder_multi) and **-0.4%** (select tree)
  despite large static wins. Of course they did: kernels are 29% of the token and those kernels are a
  slice of that.
- BLK_A and SPLIT_C moved prefill +16.8% and decode +0.3%. Prefill has enough work per dispatch to be
  kernel-bound; decode does not.
- SYCL graphs bought nothing and the DAG-lane experiment lost 33%: both were aimed at the GPU side.
- The +6.3% from fusing the 96x hyper-connection motif is the shape that *does* work, because it
  removes dispatches.

### The two levers, in order

1. **Runtime knobs, no code change.** A sweep is running: the **V2 Level Zero adapter**
   (`libur_adapter_level_zero_v2.so` is present in oneAPI 2026.1 and the run scripts explicitly pin
   `UR_LOADER_USE_LEVEL_ZERO_V2=0` - reason unknown, ask before trusting a win),
   `UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=1` (counter-based events avoid the event-pool path that
   `createEventAndAssociateQueue` sits on), `UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=1`,
   `UR_L0_IMMEDIATE_COMMANDLISTS_BATCH_EVENT_COMPLETIONS=1`, `UR_L0_BATCH_SIZE`.
   Note `GGML_SYCL_DEVICE_EVENT_WAIT=1` is already the good path (device-side barrier via
   `ext_oneapi_submit_barrier` instead of a host round trip, `ggml-sycl.cpp:7880`) - do not turn it off.
2. **Fewer dispatches.** 2609/token is ~870 per device, ~18 kernels per layer per device. Every fused
   pair removes ~11.4 us of host time. This is where kernel-level work should go: not making a kernel
   faster, but making two kernels into one.

## SYCL graph replay is INERT on this config, and the run-script comment blames the wrong thing

Measured 2026-09-25. Decode is 69% host-side per-dispatch overhead, so graph replay - record once,
submit once - is the largest theoretical lever available. It does nothing:

| arm                                   | pp2048  | tg128 |
|---------------------------------------|---------|-------|
| A: oneMKL FA + graphs ON (today)      | 670.12  | 33.52 |
| B: no library FA + graphs ON          | 609.05  | 33.76 |
| C: no library FA + graphs OFF         | 613.89  | 33.64 |
| graphs ON vs OFF, library FA on       | 33.47 / 33.48 vs 33.49 / 33.47 |

**B vs C is -0.8% prefill and +0.4% decode: graph state changes nothing**, with or without the
libraries. `GGML_SYCL_ENABLE_GRAPH=1` has been in the production script doing nothing.

`GGML_SYCL_DEBUG=1` produces **zero** `[SYCL-GRAPH]` lines, and no "can not use graphs on device"
line either, which places the bail in `check_graph_compatibility` - whose reasons log through
`GGML_LOG_DEBUG`, invisible to llama-bench.

### Five hypotheses falsified by measurement, so nobody repeats them

1. **"Multiple devices do not support graphs"** - the comment beside the flag in
   `run_llama_flash_next.sh`. **Stale.** `ggml-sycl.cpp:7652` says the opposite: "the scheduler hands
   each backend its own cgraph, so those graphs are still recordable one by one." Only split buffers
   are unrecordable.
2. **Device aspects missing.** All of `ext_oneapi_limited_graph`, `ext_oneapi_graph` and
   `ext_oneapi_async_memory_alloc` are **YES** on all three B60s, under **both** the V1 and V2 Level
   Zero adapters. Tested with a standalone SYCL probe, not inferred.
3. **Per-device SYCL contexts forcing host waits.** **Wrong, and I nearly acted on it.**
   `create_queue_impl` builds queues from the device, which looks per-device, but a probe shows
   queues on devices 0/1/2 **share one context** (the 6 enumerated GPUs are two platforms of 3). So
   `sycl_queue_wait_for_queue` succeeds, the `queues_wait_and_throw()` double-drain at
   `ggml-sycl.cpp:1100-1108` never fires, and `ggml-sycl.cpp:7880` already takes the device-side
   `ext_oneapi_submit_barrier` path. **Cross-device sync is already device-side.** I was one step from
   proposing a context-sharing refactor worth nothing.
4. **oneMKL/oneDNN blocking recording.** Plausible on the code -
   `ggml_sycl_flash_attn_ext_uses_library` returning true rejects the whole cgraph, and this branch
   uses the oneMKL FA kernel on every layer. But disabling all three
   (`GGML_SYCL_ENABLE_MKL_FA=0 GGML_SYCL_FA_ONEDNN=0 GGML_SYCL_ENABLE_DNN=0`) still gave **zero**
   SYCL-GRAPH lines and no throughput change.
5. **`MUL_MAT_ID` shape gate.** `ggml_sycl_mul_mat_id_mmvq_shape_ok` rejects `ne12 > 8`, which is the
   prefill guard; at decode ne12 = 1 and it passes. Also `mul_mat_id_runs_on_device` does **not**
   consult `g_ggml_sycl_mmid_sched`, so the sched bits do not affect graph eligibility.

Remaining candidates, both logging only via `GGML_LOG_DEBUG`: `node_is_device_local` failing on some
node, or `graph_needs_reorder` never clearing. Use `llama-cli -lv 4`, not llama-bench.

**Latent inconsistency worth a look while in here:** graph compatibility decides "MUL_MAT_ID runs on
device" from shapes alone, but the actual dispatch at `ggml-sycl.cpp:6131-6132` *also* requires
`GGML_SYCL_MMID_SCHED_DEVICE`. With the bit clear and shapes fine, compatibility would green-light
recording for a path that then calls `wait()` on a recording queue - which the comment at :7701 says
is illegal. The old production default was sched=0, i.e. bit clear.

### Settled side results

- **Keep oneMKL/oneDNN flash attention.** A vs C is **+9.2% prefill** and neutral decode. Since graphs
  are inert either way, disabling the libraries to unlock recording buys nothing and costs 9%.
- **The per-dispatch cost is not tunable from outside.** Six L0 knobs all landed on the same ~33.5
  plateau: device-scope events off 33.48, barrier-by-signal 33.55, batch event completions 33.61,
  counter-based events 33.66, batch size 16 33.69, counter+barrier 33.80. Only **L0 adapter V2** moved,
  34.23 (+2.2%), which is a different implementation rather than a tuning parameter. Unconfirmed, and
  ask why `UR_LOADER_USE_LEVEL_ZERO_V2=0` is pinned before adopting it.
- **Prefill is CPU-contention sensitive in a way decode is not.** In one contaminated pass, pp2048 read
  405 against its true 670 (-40%) while tg128 read 32.12 against 33.52 (-4%). Cause: five stale
  monitor pipelines of my own at loadavg 6.3. The asymmetry is the host-side PLE gather, which runs
  across a whole ubatch with `-tb 16` and is prefill-only. **Downgrading the load gate to a warning was
  right for decode and wrong for prefill.** Stop background watches before measuring prefill.

## ROOT CAUSE FOUND: 12 oneMKL GEMM calls per token block graph REPLAY for the whole graph

The distinction that matters is **replay**, not recording. Recording is a one-time cost; replay is what
would collapse ~2609 per-token submissions into a handful. Decode is 69% host-side dispatch overhead,
so replay is the largest lever available.

`llama-cli -st -v` (llama-bench cannot surface `GGML_LOG_DEBUG`) prints the reason directly:

    check_graph_compatibility: disabling SYCL graphs due to node type MUL_MAT using oneMKL or oneDNN

**It is `MUL_MAT`, not `FLASH_ATTN_EXT`.** That is why the earlier experiment disabling
`GGML_SYCL_ENABLE_MKL_FA=0 GGML_SYCL_FA_ONEDNN=0 GGML_SYCL_ENABLE_DNN=0` changed nothing:
`mul_mat_uses_library_gemm()` (`ggml-sycl.cpp:7602`) still returns true, and oneMKL is not gated by
the oneDNN switch. It returns true either when src0 is F16 and `src1->ne[2]*src1->ne[3] > 1`
(batched f16 GEMM), or as the fallback when none of `can_use_dequantize_mul_mat_vec`,
`can_use_mul_mat_vec_q`, `can_use_mul_mat_q` will take the node.

**The cost of the blocking nodes, per token** (from the n=64 vs n=256 profile difference, so the model
load cancels):

| task                                | per token | ms/token |
|-------------------------------------|-----------|----------|
| `gemm_kernel` (oneMKL/oneDNN)        | **12.0**  | **0.038** |
| `convert_unary_nc` (its to_fp16 staging) | 24.0  | 0.052    |
| `small_gemm_f32_launch<1,4>` (native)| 96.0      | -        |
| `small_gemm_f32_launch<16,4>` (native)| 48.0     | -        |
| flash attention                      | 24.0      | 0.249    |

**Twelve dispatches costing 0.038 ms - 0.13% of a 29.85 ms token - make every cgraph ineligible for
replay.** Four per device per token, so not per-layer: a small fixed set of nodes.

### Why this is the best decode lead available

Replacing those 12 nodes with a native SYCL kernel makes replay legal. Even a native replacement
**3x slower on those nodes** costs ~0.08 ms/token, against a reclaim measured in milliseconds. The
asymmetry is roughly two orders of magnitude.

Next steps, in order:
1. **Identify the 12 nodes.** The debug line prints only the op name. Either extend it to print
   `node->name` and the deciding branch of `mul_mat_uses_library_gemm`, or dump the graph with
   `GGML_SCHED_DEBUG=2` (flaky here - it has returned empty three times before) and look for MUL_MAT
   with F16 src0 and `ne2*ne3 > 1`. 4 per device per token suggests a head or a linear-attention
   matmul, not a per-layer projection.
2. **Check whether they even need a library.** `convert_unary_nc` runs 24x/token purely to stage f16
   for these GEMMs; a native kernel would not need the conversion either, so the real saving is larger
   than the 0.038 ms.
3. **Then re-measure replay.** Expect `graphs reused` to matter too: `cgraph->uid` is regenerated
   unconditionally for every split (`ggml-backend.cpp:1640`), so replay also needs llama.cpp's graph
   reuse to prevent a re-split. Two gates in series, and only the first is now understood.

### Corrections this forced on my own earlier claims

- I said zero `[SYCL-GRAPH]` debug lines placed the bail in `check_graph_compatibility`. **The
  inference was unsound** even though the conclusion happened to be right: tracing the print sites,
  the "properties changed every iteration" path prints nothing at all, so silence was equally
  consistent with the replay gate. The debug flag was verified working (187,139 lines emitted).
- The upstream SYCL graph-replay PR **is** in this tree; the mechanism is present and correct. Nothing
  here is a missing feature - it is one compatibility predicate rejecting on 12 cheap nodes.

## Graph replay: ENGAGES, and costs 22% of decode. Mechanism unresolved.

Measured 2026-09-25, after two compat fixes (the hc_inject library-GEMM predicate, and skipping
unallocated nodes) cleared the blockers that had made `GGML_SYCL_ENABLE_GRAPH=1` silently inert.

| arm      | tg128 / tg256                  |
|----------|--------------------------------|
| GRAPH=1  | 25.37 / 26.03 / 26.00, 26.19, 26.04 |
| GRAPH=0  | 33.60 / 33.21 / 33.10, 32.76   |

**-22% decode, reproduced across two independent scripts and five readings.** Prefill unaffected
(~650-672 either way). Both production scripts now set `GGML_SYCL_ENABLE_GRAPH=0` with this recorded.

**Replay genuinely fires** - this is not a "records but never replays" case:
`graphs reused = 11` of 12 tokens (llama.cpp counter), `[SYCL-GRAPH] graph id N reused` x30,
`warmup complete` x3 (once per device), decode cgraphs report **zero** blockers.

### Two wrong mechanisms I published before measuring - do not repeat either

1. **"cgraph->uid is regenerated per split so properties_changed is always true."** False. The
   assignment at `ggml-backend.cpp:1640` is unconditional *within*
   `ggml_backend_sched_split_graph`, but that function only runs when the scheduler re-splits, and
   llama.cpp's graph reuse (`llama-context.cpp:1405`) takes the `if` branch and skips the `else` that
   contains `ggml_backend_sched_reset` and re-splitting. So uid is stable and the fast path hits. I
   inferred a per-token behaviour from one unconditional line without checking its caller.
2. **"replay trades kernel launches for extra memcopies."** False. Within one script, memcopy
   instances are 68,655 (graph) vs 75,031 (no graph) - slightly FEWER with graphs. I had compared
   this run's graph arm against a different run's non-graph arm.

### Instrument limitations discovered the hard way

- **VTune gpu-hotspots is blind inside a replayed graph.** It reported 100 dispatches/token and
  1.52 s kernel time with graphs vs 2635/token and 4.76 s without - the same kernels cannot take a
  third of the time, so it is under-counting nodes hidden inside graph submissions. Its wall clock
  also *favours* the graph arm (86.9 s vs 126.1 s) because there is 26x less to instrument, which is
  the opposite of the true result. Do not profile a graph arm with gpu-hotspots.
- **Host thread sampling failed three times**: once because `$!` was a subshell rather than
  llama-bench, once because the sampler found no threads, and once because `-n 256` finishes in ~10 s
  so a `sleep 40` window lands after decode has ended. If retried, use `-n 4096` or sample from t=5s.

### Open question, and why it is parked rather than pursued

The arithmetic hint: 26.19 t/s is 38.2 ms/token, and 2609 kernels at the measured 14.1 us mean is
~37 ms if executed with **no overlap**. That is suspiciously close, so the working hypothesis is that
replay serialises what the in-order queue pipelines. **Unverified.**

Parked because the action does not depend on the answer: graphs stay off, and the thing that helps
either path is **fewer dispatches**, since fusion reduces host submissions *and* graph nodes at the
same time. If fusion ever cuts node count far enough, retest replay - it is the only mechanism that
has been shown capable of removing host dispatch cost wholesale, so it is worth revisiting, just not
worth blocking on.

## MEASURED: two new fusions, -120 dispatches/token, worth at most +0.8% - and that calibrates the whole strategy

`GGML_SYCL_FUSE_TYPES`, three interleaved passes, pass 1 kept because the arms interleave.

| arm                        | tg128 p1/p2/p3    | mean  | vs base | pp2048 |
|----------------------------|-------------------|-------|---------|--------|
| 15 baseline (bits 0-3)     | 33.59 33.57 33.20 | 33.45 | -       | 667.08 |
| 31 +unary_mul_b (-48/tok)  | 33.59 32.99 33.73 | 33.44 | -0.05%  | 669.76 |
| 47 +norm_scale (-72/tok)   | 33.79 33.58 33.79 | **33.72** | **+0.80%** | 669.98 |
| 63 both (-120/tok)         | 33.89 33.61 33.40 | 33.63 | +0.54%  | 669.87 |

- **bit 5 `FUSE_NORM_SCALE`** (rms_norm + the scale of `build_gdn_l2_norm`, q/k conv, 72/token): the
  only arm with a case. Tightest series of the four. **Keep, default on.**
- **bit 4 `FUSE_UNARY_MUL_B`** (shared-expert gate, ne0 broadcast, 48/token): -0.05%, and the noisiest
  series in the set (0.74 spread, one reading at +-0.95). It also costs +13-14 instructions on the
  strided `unary_mul` variant. **Leave off.**
- "Both" (33.63) coming in *below* norm_scale alone (33.72) is incoherent for two independent wins,
  which is itself evidence that most of what is visible here is noise around a small real effect.
- Prefill flat at ~670 across all four arms.

### The calibration this forces, and it matters more than the 0.8%

At the measured ~11.4 us of host cost per dispatch, removing 120 of 2609 dispatches/token "should"
be 1.37 ms of a 29.85 ms token = **4.6%**. Measured: **at most 0.8%**, under a fifth of that. The
implementing agent predicted +2.2% by applying this branch's own earlier elasticity (-385 dispatches
bought +6.3%, ratio ~0.48) and explicitly warned the ratio degrades for cheap kernels. Even that was
2.75x optimistic.

So the linear model "token time = dispatches x 11.4 us" is **wrong as a predictor of savings**, even
though it fits the total. Two candidate explanations, untested:
  1. the removed dispatches are cheaper than average (both are tiny elementwise kernels), or
  2. host dispatch cost is not purely additive - some of it overlaps with GPU execution, so removing
     a dispatch removes less than its nominal share.

**Settle this before investing in more fusion work.** The remaining ~2400 dispatches/token look like
a 27 ms prize under the linear model and perhaps a 5 ms one under the observed elasticity. That is
the difference between "fusion is the whole answer" and "fusion is a rounding error", and every
remaining decode idea in this file is priced off it.

A cheap way to settle it: take an already-measured fusion with a known dispatch delta (the census
memory's -385 for +6.3%, and this -120 for +0.8%) and add a third point at a different scale, then
see whether savings scale with count or with the *cost* of the specific kernels removed.

## SETTLED: the f16 staging conversions are not removable. oneDNN rejects mixed a/b datatypes.

The 30.3 GB of `convert_unary_nc_sycl<float, half>` in the prefill capture (1196 dispatches, 2.9% of
VRAM traffic) is essentially unrecoverable, and the reason is a library ABI limit on both backends.

**oneMKL**: `dpct::gemm` (dpct/helper.hpp:2283) switches on a type-combination id in which every
accepted tuple has `a_type == b_type`; `usm_decls.hpp:21-49` declares exactly ten
`ONEMKL_DECLARE_GEMM(Ta,Tb,Tc,Ts)` and **Ta == Tb in all ten**, so `gemm_impl<half,float,float,float>`
would not compile.

**oneDNN**: measured directly with a primitive-descriptor probe on this GPU
(`scratchpad/dnnl_probe.cpp`, `ONEDNN_VERBOSE=dispatch`, pd creation only, no execute):

    f16 src x f16 wei -> f32   ACCEPTED   impl=jit:gemm:any          <- what ggml does today
    f16 src x f32 wei -> f32   REJECTED at every level:
        jit:xe_hp:gemm  unsupported datatype combination   jit_xe_hp_systolic.cpp:106
        jit:gemm        tensors a and b have inconsistent datatypes  jit.hpp:180
        ocl:ref         unsupported datatype combination   ref.hpp:185
        ocl:with_po     failed to create nested primitive

So the `:4126` conversion is **not deletable**, which was the highest-value remaining lead in that
analysis. Disposition of all eight `to_fp16` sites (the brief claimed four; the real count is eight,
found with `grep -rn "ggml_get_to_fp16_sycl\|ggml_get_to_fp16_nc_sycl\|to_fp16_sycl(\|to_fp16_nc_sycl("`):

| site | consumer | verdict |
|------|----------|---------|
| `ggml-sycl.cpp:3259` -> fused-GEMM | native | **FUSED**, patch at `/root/sycl-fuse-f32-into-fused-gemm-pack-b.patch`; -40% src1 traffic and one dispatch instead of two, but gated `N <= 64` so it **cannot fire at pp2048** - decode only |
| `:3259` -> oneMKL/oneDNN | library | not fusable (ABI), and re-read factor likely > 3 so fusing would lose anyway |
| `:3286` | dequantize path | contributes **zero** to the 30.3 GB - the enclosing `if` at :3249 means src0 is quantized, so `ggml_get_to_fp16_sycl` returns a `dequantize_*`, never `convert_unary` |
| `:4126` | oneDNN | not fusable, **not deletable** (proven above) |
| `:4132` | - | **not compiled** with `GGML_SYCL_DNNL=1` (it is the `#else` of :4106) |
| `dmmv.cpp:2244` | native dmmv | **fusable, not done** - best remaining decode target; precedent already exists in `dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd`, which takes `const float * y` |
| `fattn-common.hpp:961/969/1002/1011` | native FA | re-read factor high (K/V per Q tile) so pre-conversion is probably right; only fires with an f32 KV cache |

**The fusable/unfusable threshold, worth keeping:** for E elements and consumer re-read factor N,
pre-convert costs `6E + 2EN` bytes and convert-on-load costs `4EN`, so **fusing wins iff N < 3**.
A GEMM that re-reads its input per tile is usually above that; a pack kernel or a mat-vec that reads
once is below it. Use this before proposing any further on-load conversion.

**Unclaimed and much larger:** `grouped_gemm_pack_b` at **104.6 GB, 10.1% of prefill VRAM traffic** -
the single biggest item in the capture, same "materialise a repacked copy" pattern, and it already
reads f32 so the type question does not arise. It is now a near-duplicate of `fused_gemm_pack_b`
(they differ only in tile indirection), so collapsing them is a prerequisite worth doing anyway.

## CALIBRATION SETTLED: a removed dispatch is worth ~3.3 us, not 11.4 us

Measured by counting kernel instances directly (VTune, n=256, arms differing only in
`GGML_SYCL_FUSE_TYPES`, so the model load contributes an identical constant and the difference is
the per-token delta outright).

| arm            | total dispatches | per token | removed/token | predicted | ratio |
|----------------|------------------|-----------|---------------|-----------|-------|
| 15 baseline    | 671,293          | 2622      | -             | -         | -     |
| 47 (bit 5)     | 652,789          | 2550      | **-72.3**     | -72       | **100%** |
| 63 (bits 4+5)  | 652,532          | 2549      | -73.3         | -120      | 61%   |

Supporting counts: `scale_f32` 18,907 -> **403** at bit 5 (the whole reduction; the 403 are instances
the pattern does not match), `rms_norm` unchanged at 65,792 (it still runs, now doing the scale too),
`unary_mul` unmoved with bit 4 off - a clean internal control.

### Two conclusions, both firmer than what they replace

**1. `FUSE_NORM_SCALE` fires exactly as designed.** -72.3/token against -72 predicted. So the weak
+0.80% is NOT a matcher gap - the dispatches genuinely went away and were worth that little.

**2. `FUSE_UNARY_MUL_B` removes ~nothing: -1.0/token, not the predicted -48.** `unary_mul` instances
*rise* 21,845 -> 34,438 (+49/token) while the total barely moves, so the fusion **substitutes** one
kernel for another rather than eliminating one - the `mul` it absorbs was already being handled by an
existing fusion. That explains its -0.05% by mechanism, not by noise, and is the real reason to leave
bit 4 off. Its author's static analysis was exactly right for bit 5 and wrong for bit 4; the
difference is that a static graph scan cannot see that a node is already absorbed elsewhere.

### The number to price future work with

    72.3 dispatches/token removed  ->  +0.80% of a 29.85 ms token  ->  ~3.3 us per dispatch

Not the 11.4 us average (29.85 ms / 2609) I had been quoting. **The average conflates submission
overhead with GPU work that fusion does not remove** - the arithmetic still happens, just in one
kernel instead of two. Corrected addressable pool:

    ~2550 remaining dispatches x 3.3 us  ~=  8.4 ms  ~=  28% of a decode token

Still the largest single lever identified, but about a third of what the linear model claimed, and
that is the ceiling for *all* remaining fusion work combined, not for any one change. Price
accordingly: a fusion removing N dispatches/token is worth roughly `N * 3.3 us / 29850 us`, i.e.
**N * 0.011%**. Anything under ~50 dispatches/token is below this box's measurement noise.

**Method note that made this work:** counting instances at a fixed token count across arms is far
more reliable than differencing two token counts, and it is the only way to tell "the matcher did not
fire" from "the matcher fired and it did not help". Those have the same throughput signature and
completely different follow-ups.

## L0 async d2d copy + P2P block guard (2026-09-25) - CLOSED, null result

`dev2dev_memcpy` mode 1 now appends to an ASYNCHRONOUS L0 immediate list and orders the destination
queue with a host-visible L0 event -> `sycl::make_event` -> `ext_oneapi_submit_barrier`; the caller
drops its destination drain. Gated by `GGML_SYCL_ASYNC_COPY` bit 1 (bitset; default 1 = bit 0 only,
so OFF by default). The source drain stays: each device queue is built from a device, so every queue
has its own SYCL/L0 context and events cannot cross it. One multi-device context would remove it.

P2P block guard: with `GGML_SYCL_ENABLE_VMM=1`, cross-device copies > 2 MiB decline peer/L0 routes and
host-stage, one-time warning. Reason: intel/compute-runtime #995 (un-split P2P copy across a
zeVirtualMemMap block end hangs the GPU; fix #996 open, not in NEO 26.31.39395).

Measured, real server config (run_llama_flash_next.sh flags, build/bin verified by md5 per arm):
    BASE    DEV2DEV=0 ASYNC=1   tg 24.725 (warm arm)
    L0SYNC  DEV2DEV=1 ASYNC=1   tg 24.769   pp 578.03
    L0ASYNC DEV2DEV=1 ASYNC=3   tg 24.758   pp 578.37     async vs sync: -0.05% tg, +0.06% pp
llama-bench modes 0/1/2 (2 = FORCED host staging) also flat: tg 33.60/33.55/33.55.
Correctness: MUL_MAT 1126/1126 both; CPY flaky either way (moving cases; on-arm >= off-arm);
generation coherent. Guard never fires in layer-split, even VMM=1: cross-device traffic here does
not go through dev2dev_memcpy at all. Keep the code for 2-GPU / split-mode row; nothing to gain here.

Server decodes ~26% slower than llama-bench on the same build (24.8 vs 33.5 tg) - unexplained, open.

## 2026-09-25 afternoon: memory, pipeline mode, MTP (all on 3x B60, production flags)
- `run_llama_flash_next.sh` has `GGML_SYCL_ENABLE_GRAPH=1` (vtune script has 0): server tg 24.8
  vs 31.5 with 0, consistent with the known -22% graph-replay cost. User's call; untouched.
- `-ot per_layer_token_embd=CPU` moves nothing; it disables pipeline parallelism. Pipeline mode:
  tg +1-2.5%, pp same; at -c 131072 it overflows one card and pages (pp 26). Fits at 32k/64k.
- Peak composition (131k, ub 1024): #1 QSA FILL 877, #2 MoE weighted MUL 799, #3 MoE down 699.
  ~416 MiB/card live at both peaks is inputs/constants: causal mask copy 256, QSA bias 128,
  QSA SET_ROWS zeros payload 4x8. Existing fusion-dead intermediates are NOT at the peak.
- Agents out: dead-buffer removal (.worktrees/dead_buffers), gather-based sparse FA from top_k
  (.worktrees/qsa_sparse_fa), MTP kernel work #1 multi-col ESIMD q8_0/Q6_K, #3 GLU ncols 3-8,
  #4 catch-up dead work (.worktrees/mtp_opt).
- MTP (PR #28243, sidecar MTP/mtp-...-shared-Q8_0.gguf): runs; fits with -ts 34,34,32 -devd SYCL2
  (peak 23.1/22.6/23.2 GiB, no active GTT); acceptance greedy 0.56-0.68, prod 0.41-0.73;
  greedy tg +21% (indicative, box load 16-20); pp -14..17% (draft catch-up). Needs quiet rerun.
- Async L0 d2d copy: null on this config (dev2dev_memcpy never called; cpy_tensor_async carries
  all 3300 cross-device copies). Closed.
