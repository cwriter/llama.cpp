# SYCL decode graph optimization analysis

## Implementation status

The first implementation covers four launch and traffic reductions:

- The hyperconnection `SCALE -> SIGMOID -> SCALE` sequence writes its final weight in one kernel at the original first-SCALE position. The existing repeat/mul/add fusion consumes that weight normally. This removes two launches and avoids extending the input tensor lifetime beyond the point known to the graph allocator.
- Adjacent IQ3_S expert gate/up `MUL_MAT_ID` operations followed by SWIGLU share one Q8_1 activation and one route-major kernel. The kernel reads each route ID once, accumulates gate and up together, and writes the activated product directly.
- Decode Q8_0 shared-expert gate/up `MUL_MAT` operations followed by SWIGLU share one reordered Q8_1 activation and one reordered-layout kernel. This removes one quantization, two intermediate output writes, their later reads, and two launches.
- F32 ADD reductions combine up to eight consecutive ADD nodes in one kernel. The kernel keeps the original left-to-right arithmetic order and supports an independent stride tuple for every input.

Every matcher checks exact types, shapes, layouts, graph adjacency, and intermediate use counts. Unsupported batches, GLU variants, weight formats, split buffers, and layouts retain the existing paths.

The experimental paths are disabled by default while they are validated on the target GPU. `GGML_SYCL_DECODE_FUSION` is a bit mask: 1 enables the existing hyperconnection repeat/mul/add fusion, 2 also folds its weight chain, 4 enables IQ3_S MoE gate/up fusion, 8 enables Q8_0 shared-expert fusion, and 16 enables multi-ADD fusion. Weight folding requires bits 1 and 2, so use value 3 for that test. The default value is 0.

## Scope

This document records the decode investigation on branch `decode_graph_opt`, based on checkpoint `3dba2eac5` and the supplied VTune profile. It separates command-graph work from kernel throughput work and ranks changes by expected decode impact.

The profile appears to cover several decode evaluations. Instance counts differ by layer type and operation, so aggregate `Total Time` values are useful for ranking but cannot be divided into a single token time without the trace interval and token count.

## Graph state

The decode path is eligible for whole-graph or segmented SYCL capture when all tensors in a scheduler split belong to one SYCL device and the selected flash-attention and matrix paths avoid external libraries or blocking host waits.

The direct quantized `MUL_MAT_ID` kernel shown in the profile is graph-compatible. Decode normally has one token and a small number of selected experts, so `R = tokens * experts_used` is below the automatic route-reordering threshold of 64. The direct route-major kernel is therefore intentional. With one token, selected experts are normally distinct and there is little expert-weight reuse to recover through sorting.

Command graphs remove repeated host submission overhead. They do not combine GPU commands or remove the device-side scheduling cost of many small kernels. Fusion remains relevant during graph replay, particularly when a model submits thousands of commands per token.

Several VTune rows have zero active instructions or high idle percentages while still receiving tens of microseconds. Those durations include scheduling, dependency, frequency, or profiler attribution effects. They must not be interpreted as pure execution time. A fusion A/B at constant frequency is the reliable measurement.

## Profile ranking

The largest supplied aggregate rows are:

| Operation family | Instances | Total | Average | Observation |
| --- | ---: | ---: | ---: | --- |
| Hyperconnection repeat/multiply/add fusion | 64 | 8.734 ms | 136.5 us | Only 12 workgroups for a typical `[3072, 4]` output; 0% active in this capture indicates attribution or dependency time dominates |
| IQ3_S direct MoE MMVQ | 60 | 4.985 ms | 83.1 us | Decode has too few routes for expert-major reuse; gate/up pairing can remove duplicate activation quantization and one consumer launch |
| IQ4_NL direct MoE MMVQ | 28 | 4.759 ms | 170.0 us | Down projection; little route reuse at one token, so optimize its dot kernel only after reducing command count |
| Reordered-layout Q8_0 MMVQ variants | 252 | about 9.3 ms | 18.6-102.6 us | These names describe GEMV on reordered weights, not a runtime weight-reorder operation |
| Flash attention tile | 8 | 1.446 ms | 180.7 us | Low active time and 59% idle; depends on context length and frequency, so it needs a context-length sweep before kernel changes |
| Q8_0 direct MoE MMVQ | 4 | 1.033 ms | 258.2 us | Too few samples to separate shape and frequency effects |

## RMS_NORM and MUL fusion

`rms_norm_mul_f32_sycl` is already a real fused kernel. It reads the source once for the sum of squares, performs a subgroup or workgroup reduction, then reads the source again and writes the normalized and multiplied result directly. It does not materialize a standalone RMS_NORM output or launch a separate MUL kernel.

For `ncols < 1024`, one SIMD16 subgroup handles each logical row. Loads are coalesced: each loop iteration has consecutive lanes reading consecutive columns. There is no SLM barrier in this path. For larger rows, a full workgroup and one SLM reduction barrier are used.

The large prompt-profile duration previously attributed to this kernel had 0% active and 100% stalled time. That is not evidence of expensive RMS arithmetic. It is consistent with command dependency or replay time being assigned to a small kernel.

The main qwen4exp hyperconnection RMS output, `hc_norm`, has multiple later uses: it feeds the low-rank down projection and is multiplied by the generated gate. Fusing RMS directly into only one consumer would either duplicate normalization or leave the existing output necessary for the other consumer. This makes RMS-to-GEMV fusion a poor first change.

Possible later RMS work:

- cache a short row in SLM to avoid the second source read;
- specialize common contiguous sizes;
- fuse Q/K RMS normalization with RoPE when the graph proves single use;
- inspect large-row workgroup sizing on the target GPU.

These changes target a kernel that is not prominent in the supplied decode profile. They are deferred until an isolated RMS A/B shows measurable token impact.

## Priority 1: fold hyperconnection weight generation into combine

The model builds each combine as:

```text
scale(inject, 1 / hc)
sigmoid
scale(2)
reshape
repeat(block_out)
multiply by weight
add residual
```

The current `ggml_sycl_op_repeat_mul_add_fused` eliminates materialization of REPEAT and combines the final MUL and ADD. It still waits for three separate elementwise commands that produce the small `[hc, tokens]` weight tensor.

The combine kernel already maps `i1` to the hyperconnection stream and loads one scalar weight for all embedding columns in that stream. It can instead load `inject[i1, token]` and compute:

```text
w = 2 * sigmoid(inject / hc)
dst = residual + block_out * w
```

Expected effect per matched combine:

- remove three GPU commands;
- remove two temporary writes and reads for a very small tensor;
- retain the existing elimination of REPEAT and the intermediate multiplied tensor;
- leave the large combine kernel's memory traffic unchanged except for reading `inject` instead of the final weight.

The matcher must verify the exact SCALE -> SIGMOID -> SCALE chain, constants, single-use edges, contiguous F32 input, and the existing fan-out shape. It must suppress only those three nodes and the REPEAT/MUL/ADD nodes already covered. Other weight-producing expressions must keep the current fusion.

This is the first implementation target because the profile shows 64 combine calls and the graph structure makes three command removals available for each call. The arithmetic added to the combine kernel is one sigmoid per stream and token, amortized across thousands of embedding elements.

## Priority 2: fuse MoE gate, up, and SWIGLU

Each MoE feed-forward block contains adjacent gate and up `MUL_MAT_ID` nodes with the same F32 activation and route IDs, followed by SWIGLU. Today each matrix operation independently quantizes the same activation to Q8_1 and launches a separate quantized consumer.

A fused path can:

- quantize the activation once;
- load gate and up expert weights in one consumer launch;
- compute both dot products;
- apply the SWIGLU epilogue before writing the final intermediate;
- eliminate both gate/up F32 temporary tensors.

For one matched triple this removes one Q8_1 quantization command, one `MUL_MAT_ID` consumer command, and one SWIGLU command. It also removes the round trip through both projection outputs.

The decode route count is small, so this should extend the direct route-major MMVQ kernel rather than force expert-major route ordering. The two weight tensors must have the same quantization type, shape, layout, expert stride, activation tensor, and route IDs. The initial implementation should support the model's IQ3_S gate/up type and fall back unchanged for every mismatch.

This is a larger change than priority 1 because it threads a second weight tensor and epilogue through quantization-type dispatch. It should remain a separate checkpoint and benchmark.

## Priority 3: shared-expert gate/up fusion

The dense shared expert has two Q8_0 matrix-vector products over the same activation followed by SWIGLU. The current dense fusion accepts only Q4_K reordered weights. Adding a generic standard-layout quantized path would remove one activation quantization, one GEMV launch, and the SWIGLU launch for each shared-expert block.

This can reuse the dual-output and GLU machinery introduced for priority 2 without route IDs. Implement it after the MoE version so both paths share the same type-specific dot-product structure.

## Priority 4: wider reductions and hyperconnection mean

The model contains linear ADD chains for expert reduction and hyperconnection stream reduction. Pairwise ADD fusion reduces some commands but still launches several kernels for longer chains.

Two possible implementations remain:

1. Extend F32 ADD fusion to consume up to eight operands in one kernel. This is general and low risk.
2. Fuse the qwen4exp four-stream gated mean directly. This removes more commands but is model-specific and has more complex view/stride validation.

The general F32 multi-ADD path should be measured first. The specialized mean is justified only if the remaining chain is still visible after graph replay and wider ADD fusion.

## Flash attention

The tile kernel has 59% idle time and only 22% XVE occupancy in the sample. Decode attention changes shape with context length, so a single aggregate cannot identify its limiting regime. Collect at least short, medium, and long context points with GPU frequency recorded.

Useful measurements are kernel duration, number of workgroups, EU occupancy, L3 bandwidth, and whether the graph uses tile, vector, oneDNN, or oneMKL attention. Kernel work should begin only after the same implementation remains dominant across the relevant context range.

## Measurement protocol

For every fusion:

1. Keep model, context, batch, device split, power mode, and graph settings fixed.
2. Record GPU frequency and power-limit behavior.
3. Run enough generated tokens to exclude graph warmup and recording.
4. Compare tokens per second and GPU busy time before comparing individual VTune rows.
5. Verify that removed kernels disappear from the command timeline.
6. Run with `GGML_SYCL_ENABLE_FUSION=0` as a broad control, then add a narrow environment switch if attribution between new fusions is needed.

Frequency reduction can hide a kernel improvement. Report both wall performance and average GPU frequency for each A/B.

## Recommended implementation order

```text
1. hyperconnection weight-chain + combine fusion
2. direct MoE gate/up + SWIGLU fusion
3. dense shared-expert gate/up + SWIGLU fusion
4. F32 multi-ADD fusion
5. context-dependent flash-attention work
6. RMS-specific tuning only if isolated measurements justify it
```

Each step should build with `examples/sycl/mybuild.sh`, receive its own correctness comparison, and be measured independently before starting the next step.
