# SYCL: int8 XMX implementation plan

Status: Parts A-D record completed experiments on Intel Arc Pro B60 (bmg-g21) with oneAPI 2026.1, except for D.2, whose device-program build fails. None of those replacement paths demonstrated a repeatable win over its existing competitor. E includes later experiments and unmeasured proposals; F and G are open leads. The grid-cache improvement is already in the source, while the query-tile gains are environment settings. Historical measurements below were not rerun during this document review.

Review date: 2026-10-03. The implementation pointers refer to the current working tree, including its local SYCL changes. Historical line numbers refer to `performance_uplift_mega_branch` at `fbb9699`; use symbol names to locate code. Part H is the current implementation handoff and takes precedence over historical instructions.

Goal: use the int8 XMX path (s8 x s8 -> s32, one 8x16x32 `joint_matrix_mad` per sub-group) where
it beats f16 XMX, and stay in the integer domain where the math allows it.

## How to use this document

- Parts A-D are an experiment archive. Do not copy their sketches into the tree again; the current signatures, dispatch and layout support have changed.
- Start with Part H for remaining work. It defines prerequisites, layout contracts, implementation order, and stop conditions. Implement one experiment at a time and let the contributor review the design before code changes.
- Part E lists ideas that were checked and rejected, so nobody re-does that work.
- Naming: use `XMX_INT` / `xmx_int` for integer-XMX paths. MoE layout work should follow the existing `reorder_qw` and `reorder_vec_dot_q_sycl` names.
- Rules for every part:
  - Every new path is behind an env flag that defaults to **0**, until it has been measured.
  - Never raise a test tolerance to make a test pass. A tolerance miss is a bug.
  - Keep the old path untouched. The new path is an early `if (...) { ...; return; }` in front of it.
  - Follow `AGENTS.md`: ASCII only, short comments.

## 0. Results

Baseline is the existing path in this working branch, not master: ordered int8 XMX for A, f32 score GEMM for B, fused f16 GEMM for narrow dense, and oneDNN/oneMKL for wide dense or attention. Numbers below are reported historical `test-backend-ops perf -b SYCL0` results; profiling and model measurements are labeled separately.

| Part | Flag (default 0) | Correctness | Performance | Verdict |
|---|---|---|---|---|
| A int8 MoE XMX | `GGML_SYCL_MOE_XMX_INT` | 85/85 `MUL_MAT_ID` (q8_0, iq4_nl) pass | dead flat, see below | correct, **no effect** |
| B int8 QSA score | `GGML_SYCL_QSA_SCORE_INT` | pass both modes, odd sizes and 2 streams | 1K blocks equal, 8K **-2.5%**, 32K +0.7% | **rejected** |
| C int8 dense N<=64 | `GGML_SYCL_DENSE_XMX_INT` | pass | never beat the f16 fused GEMM | **rejected** |
| D.1 eight formats + SoA | same as C | 39/39 pass | per format, no win | correct, **rejected** |
| D.2 register epilogue | `GGML_SYCL_XMX_INT_EPI` | **cannot run**, see D.2.4 | n/a | **blocked** |
| D.3.2 wide kernel | `GGML_SYCL_DENSE_XMX_INT_MAX_N` | pass, 65-col tail and 128 cols | 3.02 -> 6.90 TFLOPS, still 6.4x short | **rejected** |
| D.4.1 int8 KQ | `GGML_SYCL_FA_KQ_INT` | full q8_0 K/V matrix passes | 18.83 -> 5.30 TFLOPS, **3.57x slower** | **rejected** |
| D.4.2 skip masked tiles | `GGML_SYCL_FA_SKIP_DEBUG` | n/a (instrumentation only) | 0/44 and 0/66 skippable pairs | **no go** |
| D.4.3 skip negligible chunks | same | n/a (instrumentation only) | 0/32768 and 0/49152 | **no go** |

Representative dense numbers, `MUL_MAT(type_a=q8_0,m=4096,n=512,k=14336)`, 60.13 GFLOP/run:

    library path                     1352.18 us/run   44.47 TFLOPS
    D.3 step 1, one sub-group        ~19911 us/run     3.02 TFLOPS   (time derived from reported throughput)
    D.3 step 2, wide 4 sub-group      8718.36 us/run    6.90 TFLOPS   (6.4x slower)

Representative attention numbers, `FLASH_ATTN_EXT` q8_0 K/V, `D=256, n_kv=2048, n_q=1024,
n_qh=24, n_kvh=2`, 51.54 GFLOP/run:

    library path (dequant K + oneMKL)   2737.61 us/run   18.83 TFLOPS
    D.4.1 int8 KQ                       9721.71 us/run    5.30 TFLOPS   (3.57x slower)

Section 0.1 describes the integer KQ instruction mix. Sections 0.3 and E.4-E.6 show why it is not a universal explanation for dense or MoE results.

## 0.1 Per-block scaling cost in the measured integer KQ kernel

`joint_matrix_mad` on int8 does one 8x16x32 MAC group. Both operands carry a scale that changes
**every 32 values**: `block_q8_1` activations change `ds[0]` every `QK8_1`, and every weight format
in D.1.1 changes its scale at least every 32 values. So the int32 tile must be scaled and
accumulated into f32 once per 32-value step, and that epilogue is the only thing standing between
four DPAS instructions and a working result.

Counted per k step per work-group in `mul_mat_xmx_int` (`mmvq.cpp:3324`):

    4 x joint_matrix_mad            4 DPAS, 16384 MACs
    C round trip                    32 int32 stores + 32 int32 loads per lane, stride 64 B
    scales_a                        32 float loads per lane, consecutive addresses
    A and B staging                 24 global loads + 24 SLM stores per lane
    barriers                        3

IGC's assembly for the D.4.1 kernel confirms the ratio: 4 static `dpas.8x8` against 42 static
SLM loads and 46 static SLM stores, 128 GRFs, no scratch, no spill loads or stores. There is no measured spill problem. The instruction mix suggests testing fewer SLM instructions per 32-value step; it does not exclude register allocation affecting occupancy. Static counts are not dynamic stall attribution.

Consequences, all of which the measurements agree with:

- Widening the work-group to share the staged A tile (D.3.2) gives 2.28x, because it divides the
  staging and barrier cost. It does not touch the epilogue, so it cannot close the gap.
- Growing the query tile from 256 to 4096 rows (D.4.1) gives 1.46x and then plateaus. The
  operation already launches about 98,000 one-subgroup workgroups, so there is no launch
  starvation to fix.
- Larger `TK` is not available as a lever. An `N`-way deeper DPAS would still be wrong, because
  the weights' scales change inside it.

### 0.2 Process lessons, cost three days

- **`IGC_TimeReports=1` makes the JIT unusable as a measurement.** The identical binary and the
  identical 85-case `MUL_MAT_ID` run takes **8 s** without it and does not finish in **300 s**
  with it. This single flag produced the "several minutes inside the device compiler with no
  diagnostic, possibly a compiler hang" conclusion that ended the previous session. There is no
  compiler problem in Part A.
- **`SYCL_CACHE_PERSISTENT=1` segfaults on this driver** (1.17.39395+13), reproducibly, about two
  seconds in, on any operation. Not related to the integer code. Do not use it.
- **A segfault and a hang are different from a slow run.** Check the same command without the
  diagnostic flags before concluding anything about the compiler.
- The D.2.4 crash below is the one compiler failure that survived this check.

### 0.3 Can the SLM round trip be removed? (mixed results and design)

This is the question the measurements point at, so it is worth answering explicitly rather than
leaving D.2 as the only answer. Four options, and the first is the one that does not work.

**A. Do the epilogue less often with unchanged staged integers.** The epilogue runs once
per 32-value step because both operand scales change there. The activation side is ours to choose
(`block_q8_K` has one scale per 256 values, so the B side *could* be widened). The weight side is
not: every format in the D.1.1 "fits" column has a scale that changes at least every 32 values,
and the "no" column changes every 16. A `TK=256` int8 `joint_matrix` would sum eight different
weight scales into one int32 and be silently wrong. So 32 is a floor set by the file format, not
by the kernel for this representation. This is why D.2.1's Q8_K-only idea was dropped. Folding integer sub-scales into a wider representation changes that premise: E.4 measured such a construction for IQ4_XS and rejected it on performance. Do not turn this limitation of the current representation into a mathematical impossibility for every decomposition.

**B. Do the epilogue without SLM. Blocked, and possibly not a win.** Only D.2's
`joint_matrix_apply` is the proposal in D.2, and its device-program build fails (D.2.4). The suspected dynamic-index lowering to scratch is unverified. If a future toolchain can run it, dump the assembly and compare scratch and SLM instruction counts before drawing a performance conclusion.

**C. Make the round trip cheap. IMPLEMENTED AND MEASURED. It does not work.**

The current epilogue reads 32 `int32` per lane at **stride 64 B**, one 4-byte SLM load each
(`mmvq.cpp:3408` and `:3582` before the change). Transposing the C tile should fix that on paper:
`joint_matrix_store(sub_c, ptr, XMX_INT_TM, layout::col_major)` makes each lane's column
contiguous, so 32 strided loads become 8 vector loads, `joint_matrix_store` gets the same
treatment, `scales_a[rl]` becomes a uniform-address broadcast, and `sb`/`sumb` are already in the
lane's own register.

Measured, same binary pair, `test-backend-ops perf -b SYCL0`, q8_0:

    narrow kernel, m=2560 n=12 k=5120   328.23 -> 325.17 us/run   1.01x
    narrow kernel, m=2560 n=16 k=5120   327.68 -> 324.85 us/run   1.01x
    wide kernel,   m=4096 n=512 k=14336 8720.41 -> 13003.05 us/run  0.67x
    MoE narrow,    m=2560 n=1 k=640      39.45 -> 39.55 us/run     1.00x

231/231 `MUL_MAT` cases across q8_0, iq4_nl, iq4_xs, iq3_s, q4_K and q5_K pass with the transposed
tile, so the layout logic is right. It is the hardware's reaction to it that is not.

Two things this tells us, and the second one matters more than the first:

1. A `col_major` store of an 8x16 **accumulator** may not be a free permutation. A lane's DPAS
   registers hold a scattered set of (row, col) pairs; writing them in column-major order may need
   cross-lane movement, and IGC will insert shuffles for it. That would explain the wide kernel
   regressing while correctness is unaffected. Assembly comparison is needed to establish that cause.
2. **The narrow result does not establish that the epilogue dominates.** The intended reduction in load instructions bought 1%, inside noise; the actual instruction reduction was not established here. The "22 SLM instructions per DPAS" figure in section 0.1 came
   from the D.4.1 flash-attention kernel, where the A stage is a pure byte copy. In the dense
   kernel the A stage runs `traits::stage`, and for iq3_s, iq3_xxs and iq2_xxs that is a per-element
   grid lookup plus a sign mask, which is far more expensive than the q8_0 case it was measured on.
   So section 0.1's cost model does not generalise from FA to dense, and it should not be used to
   predict gains there.

The transposed variant is not kept. Both kernels are back to `row_major`.

**D. Accept it and use the library.** This is the current default state of the tree: f16 XMX for
dense `N <= 64`, oneDNN or oneMKL f16 for large prefill, oneMKL for flash attention. Given
section 0, that is the correct answer today, and the honest summary of this document is that int8
XMX does not beat the libraries on bmg-g21 with oneAPI 2026.1.

### Adding an env flag (used by every part)

For a flag `GGML_SYCL_<NAME>` with variable `g_ggml_sycl_<name>`, mirror an existing flag such as
`g_ggml_sycl_moe_xmx` in these four places:

1. Definition: `int g_ggml_sycl_<name> = 0;` next to `ggml-sycl.cpp:126`.
2. Declaration: `extern int g_ggml_sycl_<name>;` next to `common.hpp:85`.
3. Env read: `g_ggml_sycl_<name> = ggml_sycl_get_env("GGML_SYCL_<NAME>", 0);` next to `ggml-sycl.cpp:435`.
4. Startup print: `GGML_LOG_INFO("  GGML_SYCL_<NAME>: %d\n", g_ggml_sycl_<name>);` next to `ggml-sycl.cpp:613`.

New flags in this plan:

| Flag | Part | Meaning |
|---|---|---|
| `GGML_SYCL_MOE_XMX_INT` | A | new int8 MoE kernel instead of the old one |
| `GGML_SYCL_QSA_SCORE_INT` | B | int8 QSA indexer score |
| `GGML_SYCL_DENSE_XMX_INT` | C, D.1, D.3 | int8 dense matmul |
| `GGML_SYCL_DENSE_XMX_INT_MAX_N` | D.3 | column limit for the above, default 64 |
| `GGML_SYCL_XMX_INT_EPI` | D.2 | 0: scale epilogue through SLM, 1: in registers |
| `GGML_SYCL_FA_KQ_INT` | D.4.1 | int8 Q.K^T in the oneMKL flash attention path |

`GGML_SYCL_FA_SKIP_DEBUG` (D.4.2, D.4.3) is read locally in `fattn-mkl.cpp` and `kq-mask-bits.cpp`
rather than through the four-place pattern, because it is instrumentation and not a tuning knob.
`GGML_SYCL_XMX_INT_EPI` needs a compile-time macro as well, see D.2.4.

### 0.4 Every optimisation tried, in one place

Baseline for all of these is the same shape unless stated: `MUL_MAT(type_a=q8_0,type_b=f32,
m=4096,n=512,k=14336)`, 60.13 GFLOP/run, `test-backend-ops perf -b SYCL0`. The library path
(oneDNN/oneMKL f16) is **44.50 TFLOPS**. The existing int8 XMX path is **6.90 TFLOPS**. Every
number below is a median over at least two interleaved reps, and every one of them is a
*measurement*, not an estimate.

| # | attempt | result | why it failed (or did not) |
|---|---|---|---|
| 1 | transpose the C tile | wide 6.90 -> 4.62; narrow +1% | epilogue was ~1% of the cost; col_major accumulator store needs cross-lane moves |
| 2 | merge-last, 2 DPAS per 32 values | 6.44 -> 0.413 | 2x DPAS to buy a merge that #1 priced at ~1% |
| 3 | merge-last at MT=2 (spill fix) | 0.413 -> 0.936 | still 6.9x short; decode dominates, not the merge |
| 4 | staging MLP, prefetch bytes | +0.9% (noise) | byte loads were not the limiter, the table lookups were |
| 5 | **grid in registers** | **6.44 -> 7.42 (+15%)** | **kept** - works, exact; select chain eats half the headroom |
| 6 | load DPAS A operand from global | reordered 6.86 -> 6.91 | staging was not the cost; the 8x weight re-read is |
| 7 | q8_0 SoA `[qs][d]` | row-major 6.86 -> 6.91 | stride only matters if you skip the copy, and #6 showed that is worth nothing |
| 8 | query-tile retune (env only) | 21.55 -> 22.15 (+2.6%) | measured attention gain, separate from dense grid caching |
| 9 | query tile + memory ceiling | -> 23.70 (+10%) | costs 56.8 -> 256.0 MiB scratch; user's call |
| 10 | register epilogue (`joint_matrix_apply`) | device-program build fails | see D.2.4; lowering mechanism unverified |
| 11 | GRF 256 per thread | not testable | no AOT step, so no device-compiler flags reach the driver JIT |

The one-line reason for 1-7: **XMX sits at 3.6% active while the library is at 38.8%.** We are not
compute bound and we are not bandwidth bound at DRAM - we pull 4.6x *less* from DRAM than the
library but push 3.2x *more* through L3 (1355 vs 420 GB/s), because a 32x64 column tile re-reads
the weight tensor `N/64 = 8` times. These counters motivate studying reuse, but do not rule out inner-loop improvements: #5 measured one for IQ4_XS. #8 is a separate attention launch-geometry experiment.

#### The code for each attempt

**1. Transpose the C tile** (`mmvq.cpp`, reverted). Read one lane's column contiguously instead of
at stride 64 B:

```cpp
mx::joint_matrix_store(sg, sub_c, tile_c.get_multi_ptr<no>() + mt * XMX_INT_TM * XMX_INT_TN,
                       XMX_INT_TM, mx::layout::col_major);
...
const int32_t * tc = (const int32_t *) &tile_c[0] + lane * XMX_INT_TM;
for (int mt = 0; mt < XMX_INT_MT; ++mt) {
    const int32_t * c = tc + mt * XMX_INT_TM * XMX_INT_TN;
    for (int sub = 0; sub < XMX_INT_TM; ++sub) {
        float v = scales_a[mt * XMX_INT_TM + sub] * sb * (float) c[sub];
        sums[mt * XMX_INT_TM + sub] += v;
    }
}
```

narrow n=12: 328.23 -> 325.17 us. narrow n=16: 327.68 -> 324.85 us. **wide n=512: 8720.41 ->
13003.05 us, a 1.49x regression.** 231/231 `MUL_MAT` pass across six formats, so the layout logic
is right. Two failures: a `col_major` store of an 8x16 *accumulator* is probably not a free
permutation, because a lane's DPAS registers hold a scattered set of (row, col) pairs and writing them column-major may need cross-lane movement. Neither the shuffle cost nor the intended fourfold reduction in epilogue loads was established by assembly comparison. The measured narrow gain is inside noise; it does not isolate the epilogue's original share of time.

**2 and 3. Merge-last** (`mmvq.cpp`, behind `GGML_SYCL_DENSE_XMX_SUPER`, still present). Weight
split into two exact int8 slices so a 256-value superblock accumulates in int32 and needs one f32
scale instead of eight:

```cpp
// v = sc_j * (q - 4*m) for Q4_K, or (sc6-32)*kvalues for IQ4_XS
static __dpct_inline__ void xmx_super_split(int v, int8_t * hi, int8_t * lo_s, int l) {
    const int ls = ((v + 128) & 255) - 128;   // lo_s signed
    lo_s[l] = (int8_t) ls;
    hi[l]   = (int8_t) ((v - ls) >> 8);       // 256*hi + lo_s == v exactly
}
// per superblock: 16 DPAS, 8 B stagings, then one merge
sums[row] += sy * scales_a[row] * (float) (256 * chi[i] + clo[i]);
```

Only **IQ4_XS** qualifies. Q4_K and Q5_K do not: `dequantize_row_q4_K` applies `dmin * m` with an
*independent* f16 scale, so `d*sc*q - dmin*m` is not an integer times a block scale. Writing it
anyway gave 11 failures at NMSE 0.9957. IQ4_XS has no min term, so `d*(sc6-32)*kvalues` folds
exactly, and the CPU reference already uses `q8_K` activations for it, so the coarser activation
scale is not a precision change.

| config | us/run | TFLOPS | spill | occ% | SBID% | XMX% |
|---|---|---|---|---|---|---|
| MT=4 | 145491 | 0.413 | **128 B** | 29.8 | 54.3 | - |
| MT=2 | 64231 | **0.936** | 0 | 58.9 | 69.5 | **1.0** |
| MT=2 + #4 | 63688 | 0.944 | 0 | - | - | - |
| existing int8 | 9330 | 6.44 | 0 | 78.4 | - | - |
| library | 2545 | 23.63 | - | 91.8 | 39.0 | 38.8 |

The cost was register pressure, not the 2x DPAS: at MT=4 it spills 128 B and occupancy collapses to
29.8%, and MT=2 clears the spill for 2.27x. Even so it is 6.9x behind the existing int8 path,
because it makes *staging* heavier (a multiply and a split per element, on two tiles) in order to
make the epilogue lighter - and XMX is at 1.0% active, waiting on the 32 dependent `kvalues_iq4nl`
lookups per row per sub-block. Merge-last is also structurally impossible for any format whose
scale granularity is fixed at 32 by the file format, which is every format except IQ4_XS.

**4. Staging MLP** (reverted). Issue all 16 weight-byte loads before the table lookups, so the
loads are in flight together instead of each waiting on its own `kvalues_iq4nl` lookup:

```cpp
uint8_t q4[16];
#pragma unroll
for (int j = 0; j < 16; ++j) { q4[j] = b.qs[16 * ib + j]; }   // all in flight
#pragma unroll
for (int j = 0; j < 16; ++j) { /* lookups + split */ }
```

MT=4: 0.413 -> 0.356 TFLOPS (worse). MT=2: 0.936 -> 0.944 (noise). Existing int8: 6.44 -> 6.44.
The byte loads were not the limiter; the dependent lookups were, and prefetching the indices does
not prefetch the table.

**5. Grid in registers** (`iq4nl_grid` in `mmvq.cpp`, **kept**). The IQ4_NL grid is 16 bytes held
in four registers per thread, so decoding is a shift instead of a load. `kvalues_iq4nl` stays the
single source of truth; no value is duplicated:

```cpp
struct iq4nl_grid {
    uint32_t w[4];
    __dpct_inline__ void load() {                 // once per thread, 16 independent byte loads
        for (int i = 0; i < 4; ++i)
            w[i] = (uint32_t)(uint8_t)kvalues_iq4nl[4*i+0]
                 | (uint32_t)(uint8_t)kvalues_iq4nl[4*i+1] << 8
                 | (uint32_t)(uint8_t)kvalues_iq4nl[4*i+2] << 16
                 | (uint32_t)(uint8_t)kvalues_iq4nl[4*i+3] << 24;
    }
    __dpct_inline__ int8_t at(int n) const {
        const uint32_t v = n < 4 ? w[0] : (n < 8 ? w[1] : (n < 12 ? w[2] : w[3]));
        return (int8_t)(uint8_t)(v >> (8 * (n & 3)));
    }
};
```

**+15% on the existing int8 dense path, exact, 32/32 iq4_xs and iq4_nl `MUL_MAT` pass.** iq4_xs
6.44 -> 7.42, iq4_nl -> 7.02, q8_0 unchanged at 6.90 (control - it has no grid). A throwaway
arithmetic stand-in for the lookup gave the ceiling: 8.74 TFLOPS, **+36%**, so the select chain
costs about half the available headroom. The table is hand-tuned (gaps 23,21,18,16,14,13,12,11,12,12,
13,15,16,20,24) with no closed form, so extraction rather than arithmetic is the only exact route.
This is the only change in this document that makes an existing path faster rather than merely
different, and it carries no precision cost of any kind.

Do not extend the four-register cache literally to the other grids. In `ggml-common.h`, `iq3s_grid` is 512 x uint32_t (2 KiB), `iq3xxs_grid` is 256 x uint32_t (1 KiB), and `iq2xxs_grid` is 256 x uint64_t (2 KiB). They are not 16-entry tables. A separate experiment could share a grid through SLM, following `fg_grid_traits` in `fused-gemm.cpp`, but its load cost, barriers and occupancy must be measured independently.

**6. DPAS operand read straight from global** (behind `GGML_SYCL_XMX_DIRECT_A`, still present).
`joint_matrix_load` forbids only `private_space`, and `address_space_cast` turns the USM tensor
into the `multi_ptr` it wants. For the reordered q8_0 layout `[qs][d]`, element (m, k) of an 8x32
row_major A tile is at `(row0+m)*nblk_row*32 + kb*32 + k` - exactly what a tile of stride
`nblk_row*32` expects:

```cpp
using a_ptr_t = decltype(sycl::address_space_cast<
    sycl::access::address_space::global_space, sycl::access::decorated::no>((int8_t *) nullptr));

// inner loop: no tile_a at all, only the scales pass through SLM
if (direct_a) {
    scales_a[lid] = traits::soa_scale((const uint8_t *) x, nrows * nblk_row, row, nblk_row, kb);
} else if (lid < XMX_INT_ROWS) { /* stage into tile_a */ }
...
if (direct_a) {
    if constexpr (soa_a) {
        auto pa = traits::soa_a_base((const uint8_t *) x);
        mx::joint_matrix_load(sg, sub_a,
            pa + traits::soa_a_offset(row_base + mt * XMX_INT_TM, nblk_row, kb),
            traits::soa_a_stride(nblk_row));
    }
} else {
    mx::joint_matrix_load(sg, sub_a, tile_a.get_multi_ptr<no>() + mt * XMX_INT_TM * XMX_INT_TK,
                          XMX_INT_TK);
}
```

89/89 q8_0 `MUL_MAT` pass. **6.90 -> 6.91 TFLOPS, and every counter is unchanged including L3:

| q8_0, reordered | us/run | TFLOPS | dram_rd | l3rd | occ% | SBID% | XMX% |
|---|---|---|---|---|---|---|---|
| staged through SLM | 8759 | 6.86 | 50 | 1355 | 56.7 | 66.2 | 3.6 |
| direct from global | 8696 | 6.91 | 51 | **1356** | 56.4 | 65.8 | 3.6 |

The staging was never the cost. A direct load reads *the same bytes from the same place*; it just
puts them in DPAS registers instead of SLM first. The 1355 GB/s of L3 traffic is the 8x re-read and
is untouched either way.

**7. q8_0 SoA.** `xmx_int_traits<block_q8_0>::has_soa = true` plus a `stage_soa` reading `[qs][d]`,
and `ggml_sycl_xmx_int_has_soa` now includes q8_0. This closes a real capability gap: **reordered
q8_0 - the layout that is on by default (`GGML_SYCL_REORDER_DEFAULT = ~0`) and already recorded as
5.1x faster on MoE mat-vec - could never take the int8 XMX dense path at all**, because the call
site declines `reordered && !has_soa`. 89/89 pass. **row-major 8766.83 -> reordered 8703.47 us, 6.86 -> 6.91 TFLOPS (+0.7%, noise)**:
 the clean
`nblocks*32` stride only matters if you skip the copy, and #6 showed skipping it is free but
pointless.

**8 and 9. Query-tile retune** - `GGML_SYCL_FA_MAX_MEM_MIB` / `GGML_SYCL_MKL_FA_Q_TILE`, no code.

| FA_MAX_MEM_MIB | Q_TILE | effective q_tile | us/run | TFLOPS | buf_mb |
|---|---|---|---|---|---|
| 256 (default) | unset | 768 | 15566 | 21.55 | 56.8 |
| 256 | 1024+ | 918 | 15161 | 22.15 | ~57 |
| 1024 | 6144 | 4930 | 14156 | 23.70 | 256.0 |

**+2.6% for free** - the default is clamped to 768 by the L2 rule,
`MKL_FA_Q_TILE_L2_BYTES / (4 * chunk_ld)`, and setting `Q_TILE` above 1024 skips that clamp so the
memory budget decides. **+10%, not free** - 1024 MiB and 6144 rows reach 4930 rows per tile and
`buf_mb` goes 56.8 -> 256.0 MiB. Per-rep GPU time 15.01 -> 13.90 ms, softmax occupancy
78-79% -> 93-95%. But the KQ GEMM gets 31% *slower* (3.25 -> 4.25 ms per rep, its N is now 4930)
while the VKQ GEMM gets 41% faster (3.64 -> 2.15 ms). Given HANDOVER.md records that card 0 once ran
from a 40 MiB free-VRAM cliff, spending headroom that was deliberately bought back is the user's
call, not a default to change.

## Background: where XMX is used today

| Path | File | Precision |
|---|---|---|
| Fused dequant GEMM, dense, N <= 64 | `fused-gemm.cpp`, called at `ggml-sycl.cpp:3620` | f16 x f16 -> f32 |
| Grouped MoE GEMM | `fused-gemm.cpp`, called at `ggml-sycl.cpp:6332`, `:6550` | f16 |
| Dense N > 64 (normal prefill) | `ggml-sycl.cpp:3622-3660` | f16 (weights expanded to f16 in memory) |
| Ordered MoE mat-vec | `mmvq.cpp:2969` `mul_mat_vec_moe_ordered_xmx` | **s8 x s8 -> s32** |
| QSA indexer score GEMM | `qsa-score.cpp:265-279` | f32 |
| Flash attention, oneMKL | `fattn-mkl.cpp` | f16. Q8_0 KV is dequantized per chunk. The online Q8_0 XMX KQ kernel (`mkl_fa_q8_kq_tile`, `:489`) exists but is off (`online_q8 = false`, `:821`) |
| Flash attention, oneDNN SDPA | `fattn-onednn.cpp` | f16 |

Facts the parts below rely on:
- `WARP_SIZE` is `GGML_SYCL_WARP_SIZE`, 16 for Intel targets (`CMakeLists.txt:172`). The int8
  `joint_matrix` shapes below need a sub-group size of 16.
- `block_q8_1` is `{ half2 ds; int8_t qs[32]; }`, 36 bytes, so `qs` is 4-byte aligned
  (`ggml-common.h:258-268`). `ds[0]` is the scale `d`, `ds[1]` is `d * sum(qs)`.
- `block_q8_0` is `{ half d; int8_t qs[32]; }`, 34 bytes, so `qs` is only 2-byte aligned.
- `block_iq4_nl` is `{ half d; uint8_t qs[16]; }`. `qs[j]` holds element `j` in the low nibble
  and element `j+16` in the high nibble. Values come from the int8 table `kvalues_iq4nl`.
- VNNI-packed int8 B tile for `joint_matrix<..., use::b, 32, 16, layout::ext_intel_packed>`:
  element `(k, n)` sits at byte `(k / 4) * (16 * 4) + n * 4 + (k % 4)`, row stride `16 * 4`.
  So 4 consecutive k values of one column form one `int32` at int index `(k / 4) * 16 + n`.
- Per-element access with coordinates exists as
  `sycl::ext::intel::experimental::matrix::joint_matrix_apply(sg, jm, [=](T & x, size_t row, size_t col) {...})`
  (intel/llvm `sycl/include/sycl/ext/oneapi/matrix/matrix-intel.hpp`). The two-matrix form
  `sycl::ext::oneapi::experimental::matrix::joint_matrix_apply(sg, jm0, jm1, [=](T0 & a, T1 & b) {...})`
  has no coordinates (`matrix-unified.hpp`). D.2 uses both.

---

## Part A: faster int8 MoE kernel (`mmvq.cpp`)

**Implemented. Correct: 85/85 `MUL_MAT_ID` cases pass for q8_0 and iq4_nl with
`GGML_SYCL_MOE_XMX_INT=1`, in 8 s. Performance is a null result - see A.6. See section 0.2 for why
the earlier "JIT hang" report about this part was an artefact.**


### A.1 What is slow now

Kernel `mul_mat_vec_moe_ordered_xmx` (`mmvq.cpp:2969-3105`). One work-group is one sub-group.
Per 32-value block it:
1. Stages an 8x32 A tile one byte per element through `traits::quant` (`:3026-3031`).
2. Stages the 32x16 B tile one byte per element. **Every element** recomputes
   `route -> token -> slot -> pointer`, with two integer divisions (`:3037-3051`).
3. Runs **one** 8x16x32 MAD (`:3070-3074`).
4. Stores the int32 tile to SLM and applies the scales with a scalar loop (`:3075-3085`).

Each work-group covers only 8 weight rows, so the B tile is staged again for every 8 rows.

Step 4 is not the main problem: the work-group is one sub-group, so `group_barrier(sg)` is cheap,
and the loop is 8 FMAs per lane. Steps 1, 2 and the 8-row tile are the problem. D.2 removes
step 4 as well.

### A.2 Changes

1. Process `MT = 4` M tiles (32 weight rows) per work-group, so one staged B tile feeds 4 MADs.
2. Load `sub_b` once per block and reuse it for all `MT` A tiles.
3. Stage B one column per lane: lane `n` owns route `n` of the tile (tile_n == WARP_SIZE == 16).
   It computes its row pointers **once per route tile** and copies its 32 quants as 8 `int32`.
4. Stage A one row per lane through a new `traits::stage`, which also returns the scale (and
   the min, for formats that have one; D.1 needs this).
5. Keep the epilogue: lane `n` owns column `n`, sums stay in registers.
6. Put the routing behind a small policy struct. Part C reuses the same kernel for dense matmuls.

### A.3 New code

Order in `mmvq.cpp`: traits, then the two policy structs, then `mul_mat_xmx_int`, then
`launch_mul_mat_xmx_int`. All of it goes above `launch_mul_mat_vec_moe_ordered_xmx_impl`
(`mmvq.cpp:3108`), which calls it.

**Traits.** Add a new trait template next to `moe_xmx_traits` (`mmvq.cpp:2946-2966`). Leave
`moe_xmx_traits` as it is: the old kernel still uses it.

Contract of `stage`:
- `xrow` points at the first stored block of one weight row.
- `kb` is the 32-value k step (`0 .. ncols/32 - 1`). For 32-value formats it is the block index.
  For 256-value superblocks it is `8 * superblock + sub-block`.
- It writes the 32 int8 values of that k step to `dst`, and sets `d` (and `m` when `has_min`),
  so that the real weight value is `d * dst[k] - m`.

```cpp
template <typename block_q_t> struct xmx_int_traits;

template <> struct xmx_int_traits<block_q8_0> {
    using block_t = block_q8_0;
    static constexpr int  qk      = QK8_0;
    static constexpr bool has_min = false;
    // qs is 2-byte aligned in block_q8_0, so copy as 16 x uint16
    static __dpct_inline__ void stage(const block_t * __restrict__ xrow, int kb, int8_t * dst, float & d, float & m) {
        const block_t &  b   = xrow[kb];
        const uint16_t * src = (const uint16_t *) b.qs;
        uint16_t *       d16 = (uint16_t *) dst;
#pragma unroll
        for (int j = 0; j < QK8_0 / 2; ++j) {
            d16[j] = src[j];
        }
        d = (float) b.d;
        m = 0.0f;
    }
};

template <> struct xmx_int_traits<block_iq4_nl> {
    using block_t = block_iq4_nl;
    static constexpr int  qk      = QK4_NL;
    static constexpr bool has_min = false;
    static __dpct_inline__ void stage(const block_t * __restrict__ xrow, int kb, int8_t * dst, float & d, float & m) {
        const block_t & b = xrow[kb];
#pragma unroll
        for (int j = 0; j < QK4_NL / 2; ++j) {
            const uint8_t q = b.qs[j];
            dst[j]              = kvalues_iq4nl[q & 0xf];
            dst[j + QK4_NL / 2] = kvalues_iq4nl[q >> 4];
        }
        d = (float) b.d;
        m = 0.0f;
    }
};
```

**Routing policies.** `range()` gives the route indices one work-group owns.

```cpp
// MoE: work-group slot = (compacted) expert, routes come from the sorted route list
template <int experts_used> struct xmx_int_rows_moe {
    const uint32_t * expert_offsets;
    const uint32_t * sorted_routes;
    const uint32_t * active_experts;
    const uint32_t * n_active;
    int              n_experts_used;
    size_t           expert_weight_stride;
    size_t           src1_row_stride, src1_token_stride;
    size_t           dst_row_stride, dst_token_stride;

    // false: nothing to do for this slot
    __dpct_inline__ bool range(int slot, int & expert, uint32_t & begin, uint32_t & end) const {
        if (active_experts != nullptr && (uint32_t) slot >= *n_active) {
            return false;
        }
        expert = active_experts != nullptr ? (int) active_experts[slot] : slot;
        begin  = expert_offsets[expert];
        end    = expert_offsets[expert + 1];
        return begin != end;
    }
    __dpct_inline__ size_t weight_offset(int expert) const { return (size_t) expert * expert_weight_stride; }
    __dpct_inline__ void rows(uint32_t r, const void * vy, float * dst_base,
                              const block_q8_1 *& y, float *& d) const {
        const uint32_t route = sorted_routes[r];
        const int      eu    = experts_used > 0 ? experts_used : n_experts_used;
        const uint32_t token = route / eu;
        const uint32_t slot  = route - token * eu;
        y = (const block_q8_1 *) ((const char *) vy + (size_t) token * src1_token_stride + (size_t) slot * src1_row_stride);
        d = (float *) ((char *) dst_base + (size_t) token * dst_token_stride + (size_t) slot * dst_row_stride);
    }
};

// dense: work-group slot = one tile of 16 columns of src1 / dst
struct xmx_int_rows_dense {
    int    n_cols;      // N
    size_t y_stride;    // bytes between q8_1 rows of src1
    int    ldd;         // floats between dst columns

    __dpct_inline__ bool range(int slot, int & expert, uint32_t & begin, uint32_t & end) const {
        expert = 0;
        begin  = (uint32_t) slot * 16;
        end    = sycl::min((uint32_t) n_cols, begin + 16);
        return begin < end;
    }
    __dpct_inline__ size_t weight_offset(int) const { return 0; }
    __dpct_inline__ void rows(uint32_t r, const void * vy, float * dst_base,
                              const block_q8_1 *& y, float *& d) const {
        y = (const block_q8_1 *) ((const char *) vy + (size_t) r * y_stride);
        d = dst_base + (size_t) r * ldd;
    }
};
```

**Kernel.** A new kernel next to the old one. Do not change the old kernel.

```cpp
static constexpr int XMX_INT_TM   = 8;
static constexpr int XMX_INT_TN   = 16;
static constexpr int XMX_INT_TK   = 32;
static constexpr int XMX_INT_MT   = 4;                       // M tiles per work-group
static constexpr int XMX_INT_ROWS = XMX_INT_MT * XMX_INT_TM; // 32 weight rows per work-group
static_assert(XMX_INT_TN == 16, "one B column per lane needs 16 lanes");

template <typename traits, typename rows_t>
[[sycl::reqd_sub_group_size(16)]]
static void mul_mat_xmx_int(
    const void * __restrict__ vx_base, const void * __restrict__ vy, float * __restrict__ dst_base,
    const rows_t rmap, const int ncols, const int nrows,
    sycl::local_accessor<int8_t, 1> tile_a,     // XMX_INT_ROWS * XMX_INT_TK
    sycl::local_accessor<int32_t, 1> tile_b,    // XMX_INT_TK / 4 * XMX_INT_TN  (VNNI packed)
    sycl::local_accessor<float, 1> scales_a,    // XMX_INT_ROWS
    sycl::local_accessor<float, 1> mins_a,      // XMX_INT_ROWS (only read when traits::has_min)
    sycl::local_accessor<int32_t, 1> tile_c,    // XMX_INT_ROWS * XMX_INT_TN
    const sycl::nd_item<2> & item) {
    namespace mx = sycl::ext::oneapi::experimental::matrix;
    using block_t = typename traits::block_t;

    int      expert;
    uint32_t begin, end;
    if (!rmap.range(item.get_group(0), expert, begin, end)) {
        return;
    }
    const auto sg       = item.get_sub_group();
    const int  lane     = item.get_local_linear_id();
    const int  row_base = item.get_group(1) * XMX_INT_ROWS;
    const int  nsteps   = ncols / XMX_INT_TK;     // 32-value k steps
    const int  nblk_row = ncols / traits::qk;     // stored blocks per weight row
    const block_t * x = (const block_t *) ((const char *) vx_base + rmap.weight_offset(expert));

    for (uint32_t route_base = begin; route_base < end; route_base += XMX_INT_TN) {
        // lane n owns column n of this route tile
        const uint32_t     r     = route_base + lane;
        const bool         valid = r < end;
        const block_q8_1 * y     = nullptr;
        float *            d     = nullptr;
        if (valid) {
            rmap.rows(r, vy, dst_base, y, d);
        }
        float sums[XMX_INT_ROWS] = {};

        for (int kb = 0; kb < nsteps; ++kb) {
            // B: 32 quants of column `lane` as 8 int32, VNNI index (k/4)*16 + n
            float sb = 0.0f;
            float sumb = 0.0f;  // d_b * sum(q_b), for the min term
            if (valid) {
                const int32_t * q = (const int32_t *) y[kb].qs;
#pragma unroll
                for (int g = 0; g < XMX_INT_TK / 4; ++g) {
                    tile_b[g * XMX_INT_TN + lane] = q[g];
                }
                sb   = (float) y[kb].ds[0];
                sumb = (float) y[kb].ds[1];
            } else {
#pragma unroll
                for (int g = 0; g < XMX_INT_TK / 4; ++g) {
                    tile_b[g * XMX_INT_TN + lane] = 0;
                }
            }
            // A: lane stages rows lane and lane + 16, row-major 32 int8 per row
            for (int rl = lane; rl < XMX_INT_ROWS; rl += 16) {
                const int row = row_base + rl;
                int8_t *  dst = &tile_a[rl * XMX_INT_TK];
                float     da  = 0.0f;
                float     ma  = 0.0f;
                if (row < nrows) {
                    traits::stage(x + (size_t) row * nblk_row, kb, dst, da, ma);
                } else {
#pragma unroll
                    for (int k = 0; k < XMX_INT_TK; ++k) {
                        dst[k] = 0;
                    }
                }
                scales_a[rl] = da;
                if constexpr (traits::has_min) {
                    mins_a[rl] = ma;
                }
            }
            sycl::group_barrier(sg);

            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, XMX_INT_TK, XMX_INT_TN, mx::layout::ext_intel_packed> sub_b;
            mx::joint_matrix_load(sg, sub_b,
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    (int8_t *) &tile_b[0]),
                XMX_INT_TN * 4);
#pragma unroll
            for (int mt = 0; mt < XMX_INT_MT; ++mt) {
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, XMX_INT_TM, XMX_INT_TK, mx::layout::row_major> sub_a;
                mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, XMX_INT_TM, XMX_INT_TN> sub_c;
                mx::joint_matrix_fill(sg, sub_c, 0);
                mx::joint_matrix_load(sg, sub_a,
                    tile_a.get_multi_ptr<sycl::access::decorated::no>() + mt * XMX_INT_TM * XMX_INT_TK, XMX_INT_TK);
                mx::joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
                mx::joint_matrix_store(sg, sub_c,
                    tile_c.get_multi_ptr<sycl::access::decorated::no>() + mt * XMX_INT_TM * XMX_INT_TN,
                    XMX_INT_TN, mx::layout::row_major);
            }
            sycl::group_barrier(sg);

            // element (row rl, column lane) is at rl * 16 + lane
#pragma unroll
            for (int rl = 0; rl < XMX_INT_ROWS; ++rl) {
                float v = scales_a[rl] * sb * (float) tile_c[rl * XMX_INT_TN + lane];
                if constexpr (traits::has_min) {
                    v -= mins_a[rl] * sumb;
                }
                sums[rl] += v;
            }
            sycl::group_barrier(sg);  // tile_a/b/c are rewritten by the next k step
        }

        if (valid) {
#pragma unroll
            for (int rl = 0; rl < XMX_INT_ROWS; ++rl) {
                if (row_base + rl < nrows) {
                    d[row_base + rl] = sums[rl];
                }
            }
        }
    }
}
```

Notes for the implementer:
- If `sycl::address_space_cast` does not compile for the `tile_b` load, declare `tile_b` as
  `local_accessor<int8_t, 1>` of size `XMX_INT_TK * XMX_INT_TN`, load it with `get_multi_ptr` as the
  old kernel does, and write the int32 values through `((int32_t *) &tile_b[0])[g * 16 + lane]`.
- `sums[32]` is 32 floats per lane. That is fine for 128 GRF.
- The min term: the real weight is `d_a * q_a - m_a` and the real activation is `d_b * q_b`, so
  `sum_k w_k * x_k = d_a * d_b * I - m_a * (d_b * sum_k q_b,k)`, and `d_b * sum_k q_b,k` is `ds[1]`.

**Launcher.**

```cpp
template <typename traits, typename rows_t>
static void launch_mul_mat_xmx_int(const void * vx, const void * vy, float * dst, const rows_t & rmap,
                                   int n_slots, int ncols, int nrows, dpct::queue_ptr stream) {
    const int n_row_groups = (nrows + XMX_INT_ROWS - 1) / XMX_INT_ROWS;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int8_t, 1>  tile_a(XMX_INT_ROWS * XMX_INT_TK, cgh);
        sycl::local_accessor<int32_t, 1> tile_b(XMX_INT_TK / 4 * XMX_INT_TN, cgh);
        sycl::local_accessor<float, 1>   scales_a(XMX_INT_ROWS, cgh);
        sycl::local_accessor<float, 1>   mins_a(XMX_INT_ROWS, cgh);
        sycl::local_accessor<int32_t, 1> tile_c(XMX_INT_ROWS * XMX_INT_TN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_slots, n_row_groups * 16), sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                mul_mat_xmx_int<traits>(vx, vy, dst, rmap, ncols, nrows,
                                        tile_a, tile_b, scales_a, mins_a, tile_c, item);
            });
    });
}
```

**MoE call.** Inside the existing `launch_mul_mat_vec_moe_ordered_xmx_impl<traits, experts_used>`,
at the top:

```cpp
if (g_ggml_sycl_moe_xmx_int) {
    using int_traits = xmx_int_traits<typename traits::block_t>;
    const int n_slots = route_order.active_experts != nullptr ? route_order.n_active_max : route_order.n_experts;
    const xmx_int_rows_moe<experts_used> rmap = {
        route_order.expert_offsets, route_order.sorted_routes, route_order.active_experts, route_order.n_active,
        n_experts_used, expert_weight_stride, src1_row_stride, src1_token_stride, dst_row_stride, dst_token_stride,
    };
    launch_mul_mat_xmx_int<int_traits>(vx_base, vy, dst_base, rmap, n_slots, ncols, nrows, stream);
    return;
}
```

Flag: `GGML_SYCL_MOE_XMX_INT` (see "Adding an env flag").

Also add a sub-group size check at the top of `ggml_sycl_moe_q8_xmx_supported` (`mmvq.cpp:2925`):

```cpp
if (WARP_SIZE != 16) {
    return false;
}
```

### A.4 Test

```sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake --build build -j --target test-backend-ops llama-bench llama-perplexity
GGML_SYCL_MOE_XMX_INT=1 ./build/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID
```

`MUL_MAT_ID` cases with q8_0 and iq4_nl must pass. The XMX path only runs when there are at
least 8 routes per expert on average (`mmvq.cpp:3562`). To confirm the new kernel ran, add a
temporary `fprintf(stderr, ...)` in the launcher, and remove it before the commit.

### A.5 Benchmark

```sh
for v in 0 1; do GGML_SYCL_MOE_XMX_INT=$v ./build/bin/llama-bench -m <q8_0 MoE model> -p 512 -n 0 -ub 512; done
for v in 0 1; do GGML_SYCL_MOE_XMX_INT=$v ./build/bin/llama-bench -m <iq4_nl MoE model> -p 512 -n 0 -ub 512; done
```

Turn the flag on by default only if prompt t/s does not drop on either model. After that, the
old kernel and `moe_xmx_traits` can be deleted in a separate commit.

### A.6 Measured

Two interleaved reps per side, `test-backend-ops perf -b SYCL0 -o MUL_MAT_ID`, filter
`type_a=<t>,type_b=f32,n_mats=512` (the `n_mats=512` cases are the ones that reach the ordered XMX
kernel, they give far more than 8 routes per expert).

    type_a=q8_0, m=2560 n=1    k=640     decode-shaped mat-vec
      flag 0: 39.45, 39.70 us/run
      flag 1: 39.72, 39.45 us/run
    type_a=q8_0, m=2560 n=1024 k=640     wide
      flag 0: 6337.38, 6295.36 us/run
      flag 1: 6297.85, 6271.88 us/run
    type_a=iq4_nl, m=2560 n=1   k=640
      flag 0: 49.24, 48.84 us/run
      flag 1: 48.87, 48.95 us/run
    type_a=iq4_nl, m=2560 n=1024 k=640
      flag 0: 6933.78, 6956.59 us/run
      flag 1: 6994.59, 6961.02 us/run

**Every pair is inside the run-to-run spread, in both directions.** Replacing the existing ordered int8 XMX kernel with the new wider int8 kernel changes nothing measurable in these cases.
That is a null result, not a win: the ordered MoE kernel is bounded by weight streaming and
routing, so the inner loop is not on the critical path and there is nothing for int8 to win back.
The flag stays 0.

Note for anyone repeating this: `test-backend-ops -p` uses `std::regex_search` on the parameter string, so
`type_a=q8_0,n_mats=512` silently matches nothing because `,type_b=f32,` sits between them. Use `type_a=q8_0.*n_mats=512`; an
an unmatched filter reports success. Always confirm the case count in the output.

---

## Part B: QSA indexer score in int8 (`qsa-score.cpp`)

**Implemented, correct in both modes, and REJECTED on performance.** 1024 blocks equal, 8192
blocks 2.5% slower, 32768 blocks 0.7% faster. Flag stays 0.


### B.1 Why this is the best int8 case

`ggml_sycl_fuse_qsa_score` (`qsa-score.cpp:211-300`) computes

    score[b, t] = sum_h relu( dot(pooled[:, b], q[:, h, t]) )

with an f32 GEMM (`:265-279`), then sums the heads (`k_qsa_score_reduce`, `:187-202`). The
graph is built in `src/models/qwen4exp.cpp:944-958`. The score only feeds a bias add and a top-k.

Quantize each row with one symmetric scale over the whole row (k = indexer head size):
- `pooled[:, b]` -> int8 row `A_b`, scale `sa[b] = amax / 127`
- `q[:, h, t]` -> int8 row `B_ht`, scale `sb[h, t] = amax / 127`

Then `dot = sa[b] * sb[h,t] * I[b,h,t]` with `I` an exact int32 dot. The scales are positive, so
`relu(s * I) = s * relu(I)`, and:

    score[b, t] = sa[b] * sum_h sb[h, t] * max(I[b,h,t], 0)

There is one scale per whole row, so the GEMM is a plain s8 x s8 -> s32 GEMM with no per-block
rescale. All scale math moves into the existing reduce kernel. The int32 sum cannot overflow:
`k * 127 * 127` is about 2M for k = 128.

Expected error: per-row int8 of a 128-wide row gives a relative error of about 1e-2 per dot, so
NMSE around 1e-4 on the score. The bias add after the score is additive, so scores must keep
their real scale. They do.

### B.2 Changes

1. `gemm.hpp`: teach `DnnlGemmWrapper::to_dt` about `int8_t` and `int32_t`:

```cpp
template<typename T>
static constexpr dt to_dt() {
    if constexpr (std::is_same_v<T, float>) return dt::f32;
    else if constexpr (std::is_same_v<T, sycl::half>) return dt::f16;
#ifdef GGML_SYCL_HAS_BF16
    else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return dt::bf16;
#endif
    else if constexpr (std::is_same_v<T, int8_t>) return dt::s8;
    else if constexpr (std::is_same_v<T, int32_t>) return dt::s32;
    else static_assert(0);
}
```

   `gemm()` sets `fpmath_mode::f16` when `GGML_SYCL_F16` is set. That mode only affects floating
   point math, so the int8 GEMM ignores it.

2. `qsa-score.cpp`: add a row quantizer. One sub-group per row.

```cpp
// q[r, :] = round(x[r, :] / s[r]), s[r] = amax(|x[r, :]|) / 127
static void k_qsa_quantize_rows_s8(const float * x, int8_t * q, float * s, int64_t k,
                                   const sycl::nd_item<1> & it) {
    const int64_t row  = it.get_group(0);
    const int     lane = it.get_local_id(0);
    const float * xr   = x + row * k;
    int8_t *      qr   = q + row * k;

    float amax = 0.0f;
    for (int64_t i = lane; i < k; i += WARP_SIZE) {
        amax = sycl::fmax(amax, sycl::fabs(xr[i]));
    }
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());

    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    for (int64_t i = lane; i < k; i += WARP_SIZE) {
        qr[i] = (int8_t) sycl::round(xr[i] * id);
    }
    if (lane == 0) {
        s[row] = d;
    }
}

static void qsa_quantize_rows_s8(const float * x, int8_t * q, float * s, int64_t nrows, int64_t k,
                                 dpct::queue_ptr stream) {
    stream->parallel_for(sycl::nd_range<1>(nrows * WARP_SIZE, WARP_SIZE),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             k_qsa_quantize_rows_s8(x, q, s, k, it);
                         });
}
```

3. Add the int8 reduce. The tile layout is the same as for the f32 tile: inside stream `s`,
   column `h + n_heads * t_local`, element `(b, col)` at `b + n_blocks * col`.

```cpp
// dst[b, t0 + t, s] = sa[s, b] * sum over h of sb[s, t0 + t, h] * max(tile[b, h, t, s], 0)
static void k_qsa_score_reduce_int(const int32_t * tile, const float * sa, const float * sb, float * dst,
                                   int64_t n_blocks, int64_t n_heads, int64_t nt, int64_t n_tps, int64_t t0,
                                   const sycl::nd_item<2> & item) {
    const int64_t b = item.get_global_id(1);
    if (b >= n_blocks) {
        return;
    }
    const int64_t   r   = item.get_global_id(0);
    const int64_t   t   = r % nt;
    const int64_t   s   = r / nt;
    const int32_t * col = tile + n_blocks * n_heads * r + b;
    const float *   sbr = sb + (s * n_tps + t0 + t) * n_heads;

    float acc = 0.0f;
    for (int64_t h = 0; h < n_heads; ++h) {
        acc += sbr[h] * (float) sycl::max(col[n_blocks * h], 0);
    }
    dst[n_blocks * ((t0 + t) + n_tps * s) + b] = sa[s * n_blocks + b] * acc;
}
```

   Index check for `sb`: src1 has `ne1 = n_heads * n_tps` rows per stream, row index
   `s * n_heads * n_tps + t * n_heads + h`, which is `(s * n_tps + t) * n_heads + h`.

4. In `ggml_sycl_fuse_qsa_score`, after the trace block and before the `for (t0 ...)` loop, add
   the int8 branch:

```cpp
#if GGML_SYCL_DNNL
    if (g_ggml_sycl_qsa_score_int && g_ggml_sycl_enable_dnn) {
        const int64_t rows0 = n_blocks * n_stream;            // pooled rows
        const int64_t rows1 = n_heads * n_tps * n_stream;     // q rows
        ggml_sycl_pool_alloc<int8_t>  qa(ctx.pool(), (size_t) (rows0 * k));
        ggml_sycl_pool_alloc<int8_t>  qb(ctx.pool(), (size_t) (rows1 * k));
        ggml_sycl_pool_alloc<float>   sa(ctx.pool(), (size_t) rows0);
        ggml_sycl_pool_alloc<float>   sb(ctx.pool(), (size_t) rows1);
        ggml_sycl_pool_alloc<int32_t> tile_i(ctx.pool(), (size_t) (n_blocks * n_heads * t_tile * n_stream));

        qsa_quantize_rows_s8((const float *) src0_dd, qa.get(), sa.get(), rows0, k, stream);
        qsa_quantize_rows_s8((const float *) src1_dd, qb.get(), sb.get(), rows1, k, stream);

        for (int64_t t0 = 0; t0 < n_tps; t0 += t_tile) {
            const int64_t nt = std::min(t_tile, n_tps - t0);
            const int64_t n  = n_heads * nt;
            for (int64_t s = 0; s < n_stream; ++s) {
                const int8_t * a = qa.get() + s * n_blocks * k;
                const int8_t * b = qb.get() + (s * n_heads * n_tps + t0 * n_heads) * k;
                int32_t *      d = tile_i.get() + s * n_blocks * n;
                DnnlGemmWrapper::row_gemm(ctx, (int) n_blocks, (int) n, (int) k,
                                          a, DnnlGemmWrapper::to_dt<int8_t>(),
                                          b, DnnlGemmWrapper::to_dt<int8_t>(),
                                          d, DnnlGemmWrapper::to_dt<int32_t>(), stream);
            }
            constexpr int block = 256;
            const sycl::range<2> local(1, block);
            const sycl::range<2> global(nt * n_stream, ((n_blocks + block - 1) / block) * block);
            const int32_t * tile_dd = tile_i.get();
            const float *   sa_dd   = sa.get();
            const float *   sb_dd   = sb.get();
            stream->parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                k_qsa_score_reduce_int(tile_dd, sa_dd, sb_dd, dst_dd, n_blocks, n_heads, nt, n_tps, t0, item);
            });
        }
        return c.i_out - i;
    }
#endif
```

   Move the existing `ggml_sycl_pool_alloc<float> tile(...)` line **below** this block, so the
   int8 path does not also reserve the f32 tile.

   `src0` and `src1` are contiguous f32 (the shape check guarantees it, `qsa-score.cpp:76-81`),
   so row `r` starts at `r * k`.

5. Flag `GGML_SYCL_QSA_SCORE_INT` (see "Adding an env flag").

### B.3 Test: new `test_qsa_score` in `tests/test-backend-ops.cpp`

There is no test for this chain yet. Add one next to `test_qsa_mask_add` (`:7012`). It builds
the same nodes as `qwen4exp.cpp:944-958`, in the same order, so the fusion matches.

```cpp
// qwen4exp QSA indexer score: MUL_MAT -> RESHAPE -> RELU -> n_head x (VIEW, CONT or ADD)
struct test_qsa_score : public test_case {
    const int64_t k, n_blocks, n_heads, n_tps, n_stream;

    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "QSA_SCORE"; }
    std::string vars() override { return VARS_TO_STR5(k, n_blocks, n_heads, n_tps, n_stream); }

    test_qsa_score(int64_t k = 128, int64_t n_blocks = 1024, int64_t n_heads = 16, int64_t n_tps = 8,
                   int64_t n_stream = 1)
        : k(k), n_blocks(n_blocks), n_heads(n_heads), n_tps(n_tps), n_stream(n_stream) {}

    bool   run_whole_graph() override { return true; }
    double max_nmse_err() override { return 1e-3; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * pooled = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_blocks, n_stream);
        ggml_set_name(pooled, "pooled");
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_heads * n_tps, n_stream);
        ggml_set_name(q, "q");

        ggml_tensor * score = ggml_mul_mat(ctx, pooled, q);
        score = ggml_reshape_4d(ctx, score, n_blocks, n_heads, n_tps, n_stream);
        score = ggml_relu(ctx, score);

        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_heads; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx, score, n_blocks, n_tps, n_stream,
                                               score->nb[2], score->nb[3], h * score->nb[1]);
            summed = summed ? ggml_add(ctx, summed, slice) : ggml_cont(ctx, slice);
        }
        ggml_set_name(summed, "out");
        return summed;
    }
};
```

`VARS_TO_STR5` is defined at `tests/test-backend-ops.cpp:455`.

Register it next to the other QSA cases (`:11170`):

```cpp
test_cases.emplace_back(new test_qsa_score(128, 1024, 16, 8, 1));
test_cases.emplace_back(new test_qsa_score(128, 333,  8,  5, 2));   // odd sizes, two streams
test_cases.emplace_back(new test_qsa_score(64,  4096, 32, 2, 1));
```

Run:

```sh
GGML_SYCL_QSA_SCORE_INT=0 ./build/bin/test-backend-ops -b SYCL0 -o QSA_SCORE
GGML_SYCL_QSA_SCORE_INT=1 ./build/bin/test-backend-ops -b SYCL0 -o QSA_SCORE
```

Both must pass. If the int8 run fails on NMSE, look for an indexing bug (the `sb` index is the
usual suspect). Do not raise the tolerance.

### B.4 Model-level check and benchmark

Top-k selection is what matters. With a qwen4exp model and a long prompt (32k or more):

```sh
for v in 0 1; do GGML_SYCL_QSA_SCORE_INT=$v ./build/bin/llama-perplexity -m <qwen4exp model> -f <long text> -c 32768 --chunks 4; done
for v in 0 1; do GGML_SYCL_QSA_SCORE_INT=$v ./build/bin/llama-bench -m <qwen4exp model> -p 32768 -n 0; done
```

Perplexity must stay within noise (difference under 0.5%). Turn the flag on by default only then.

---

## Part C: dense int8 matmul for Q8_0 and IQ4_NL, 9 to 64 columns

**Implemented and REJECTED.** Correct, but it never beat the f16 fused GEMM it competes with, so
the flag stays 0 and `GGML_SYCL_DENSE_XMX_INT_MAX_N` stays at its 64 default.


Do Part A first. This part reuses `mul_mat_xmx_int` with `xmx_int_rows_dense`.

### C.1 Export a wrapper from `mmvq.cpp`

```cpp
// mmvq.hpp
bool ggml_sycl_mul_mat_xmx_int_dense(ggml_type src0_type, const void * vx, const void * vy_q8_1, float * dst,
                                     int M, int N, int K, int ldd, dpct::queue_ptr stream);
```

```cpp
// mmvq.cpp, after launch_mul_mat_xmx_int
bool ggml_sycl_mul_mat_xmx_int_dense(ggml_type src0_type, const void * vx, const void * vy_q8_1, float * dst,
                                     int M, int N, int K, int ldd, dpct::queue_ptr stream) {
    if (K % QK8_1 != 0 || N < 1 || !ggml_sycl_moe_q8_xmx_supported(stream)) {
        return false;
    }
    const xmx_int_rows_dense rmap = { N, (size_t) (K / QK8_1) * sizeof(block_q8_1), ldd };
    const int n_slots = (N + XMX_INT_TN - 1) / XMX_INT_TN;
    switch (src0_type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_xmx_int<xmx_int_traits<block_q8_0>>(vx, vy_q8_1, dst, rmap, n_slots, K, M, stream);
            return true;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_xmx_int<xmx_int_traits<block_iq4_nl>>(vx, vy_q8_1, dst, rmap, n_slots, K, M, stream);
            return true;
        default:
            return false;
    }
}
```

`ggml_sycl_moe_q8_xmx_supported` must stay above this function, or be forward-declared.

### C.2 Call it from `ggml_sycl_op_mul_mat_sycl` (`ggml-sycl.cpp:3539`)

Put it at the top of the `if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 ...)`
block (`ggml-sycl.cpp:3596`), **before** `src1` is converted to f16, so that conversion is skipped:

```cpp
{
    const auto * src0_extra_int = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool   reordered_int  = src0_extra_int && src0_extra_int->optimized_feature.is_reordered();
    if (g_ggml_sycl_dense_xmx_int && !reordered_int && src1->type == GGML_TYPE_F32 &&
        ggml_sycl_dense_xmx_int_type_ok(src0->type) &&
        src1_ncols <= g_ggml_sycl_dense_xmx_int_max_n && ne10 % QK8_1 == 0) {
        ggml_sycl_pool_alloc<block_q8_1> src1_q8(ctx.pool(), (size_t) src1_ncols * (ne10 / QK8_1));
        quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_q8.get(), (int) ne10, (int) src1_ncols, (int) ne10, stream);
        if (ggml_sycl_mul_mat_xmx_int_dense(src0->type, src0_dd_i, src1_q8.get(), dst_dd_i,
                                            (int) row_diff, (int) src1_ncols, (int) ne10, ldc, stream)) {
            return;
        }
    }
}
```

With two small helpers above `ggml_sycl_op_mul_mat_sycl`:

```cpp
// types the int8 dense path decodes; D.1 extends this list
static bool ggml_sycl_dense_xmx_int_type_ok(ggml_type t) {
    return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_IQ4_NL;
}
// column limit; D.3 raises it after measuring
static const int g_ggml_sycl_dense_xmx_int_max_n = 64;
```

- The surrounding block already requires `row_diff == src0->ne[1]` and a contiguous `src0`.
- That block is only entered when `use_fp16` is true, which needs a `-DGGML_SYCL_F16=ON` build
  (`ggml-sycl.cpp:3566-3570`). Build with it for this part.
- `ldc` is the dst column stride, already computed at `ggml-sycl.cpp:3564`.
- `src1_ddf_i` holds `src1_ncols` contiguous f32 rows of `ne10` values here.
- Batches of 1 to 8 columns normally go to MMVQ before this function is reached, so in practice
  this path sees 9 to 64 columns.
- Flag `GGML_SYCL_DENSE_XMX_INT` (see "Adding an env flag").

### C.3 Test and benchmark

```sh
GGML_SYCL_DENSE_XMX_INT=1 ./build/bin/test-backend-ops -b SYCL0 -o MUL_MAT
```

All q8_0 and iq4_nl cases must pass at the existing tolerance. MMVQ already uses q8_1
activations and passes it, so this path should too.

Benchmark with a micro-batch where the fused f16 GEMM runs today (N <= 64):

```sh
for v in 0 1; do GGML_SYCL_DENSE_XMX_INT=$v ./build/bin/llama-bench -m <q8_0 dense model> -p 512 -n 0 -ub 32,64; done
for v in 0 1; do GGML_SYCL_DENSE_XMX_INT=$v ./build/bin/llama-bench -m <iq4_nl dense model> -p 512 -n 0 -ub 32,64; done
```

The competitor is the fused f16 GEMM (`GGML_SYCL_FUSED_GEMM=1`, the default). If int8 is not
faster, leave the flag off and record the numbers in this document.

---

## Part D: archived design pass

The original experiment order was D.1, D.2, D.3, then D.4. Use H for new work.

### D.1 More weight formats in the int8 kernel

**Implemented, 39/39 tests pass across all eight formats including reordered weights, and
REJECTED on performance** like Part C. The SoA work is what made K-quants reachable at all; it
did not make them fast.


#### D.1.1 Which formats fit

A format fits when one 32-value k step decodes to int8 values that share **one** scale (and at
most one min). One `joint_matrix_mad` is 32 deep, so a scale that changes inside a 32-value step
would need two MADs with half the A tile zeroed, which halves the int8 rate and loses the gain
over f16. Checked against the f16 decoders in `fused-gemm.cpp:80-405`:

| Format | Int8 value per element | Scale per 32-step | Fits |
|---|---|---|---|
| Q8_0 | `qs` | `d` | yes (Part A) |
| IQ4_NL | `kvalues_iq4nl[nibble]` | `d` | yes (Part A) |
| IQ4_XS | `kvalues_iq4nl[nibble]` | `d * (sc6 - 32)` | yes |
| IQ3_S | grid byte (1..15) with sign | `d * (1 + 2 * sc4)` | yes |
| IQ3_XXS | grid byte (4..62) with sign | `d * (0.5 + sc4) * 0.5` | yes |
| IQ2_XXS | grid byte (8, 25, 43) with sign | `d * (0.5 + sc4) * 0.25` | yes |
| Q4_K | `q` in 0..15 | `dall * sc6`, min `dmin * m6` | yes, with min term |
| Q5_K | `q` in 0..31 | `dall * sc6`, min `dmin * m6` | yes, with min term |
| IQ2_XS | grid byte with sign | changes every 16 values (`scales[ib] >> 4*(il/2)`) | no |
| IQ2_S | grid byte with sign | changes every 16 values | no |
| Q6_K | -32..31 | changes every 16 values (`sc[j >> 3]`) | no |
| IQ1_S, IQ1_M | `q + delta`, delta = +-0.125 | not an integer | no |

Formats marked "no" stay on the f16 fused GEMM. Do not try to make them fit.

#### D.1.2 New traits (canonical layout)

Add to `mmvq.cpp` next to the Part A traits. The decode follows the f16 decoders in
`fused-gemm.cpp` line by line, but writes the integer value and returns the scale instead of
multiplying. `iq3s_grid`, `iq3xxs_grid`, `iq2xxs_grid`, `ksigns_iq2xs` and `kmask_iq2xs` are the
same device tables `fused-gemm.cpp` already uses.

```cpp
template <> struct xmx_int_traits<block_iq4_xs> {
    using block_t = block_iq4_xs;
    static constexpr int  qk      = QK_K;
    static constexpr bool has_min = false;
    static __dpct_inline__ void stage(const block_t * __restrict__ xrow, int kb, int8_t * dst, float & d, float & m) {
        const block_t & b  = xrow[kb / 8];
        const int       ib = kb % 8;
        d = (float) b.d * ((((b.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((b.scales_h >> (2 * ib)) & 3) << 4)) - 32);
        m = 0.0f;
        const uint8_t * q4 = b.qs + 16 * ib;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            dst[j]      = kvalues_iq4nl[q4[j] & 0xf];
            dst[j + 16] = kvalues_iq4nl[q4[j] >> 4];
        }
    }
};

template <> struct xmx_int_traits<block_iq3_s> {
    using block_t = block_iq3_s;
    static constexpr int  qk      = QK_K;
    static constexpr bool has_min = false;
    static __dpct_inline__ void stage(const block_t * __restrict__ xrow, int kb, int8_t * dst, float & d, float & m) {
        const block_t & b   = xrow[kb / 8];
        const int       ib8 = kb % 8;
        d = (float) b.d * (1 + 2 * ((b.scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        m = 0.0f;
        const uint8_t * qs    = b.qs + 8 * ib8;
        const int       qh    = b.qh[ib8];
        const uint8_t * signs = b.signs + 4 * ib8;
        // same element order as fg_decode_iq3_s: quarter il covers values 8*il .. 8*il+7
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
            const uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
            const int      sg    = signs[il];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int g1 = (grid1 >> (8 * j)) & 0xff;
                const int g2 = (grid2 >> (8 * j)) & 0xff;
                dst[8 * il + j]     = (int8_t) ((sg & (1 << j))       ? -g1 : g1);
                dst[8 * il + 4 + j] = (int8_t) ((sg & (1 << (j + 4))) ? -g2 : g2);
            }
        }
    }
};

template <> struct xmx_int_traits<block_iq3_xxs> {
    using block_t = block_iq3_xxs;
    static constexpr int  qk      = QK_K;
    static constexpr bool has_min = false;
    static __dpct_inline__ void stage(const block_t * __restrict__ xrow, int kb, int8_t * dst, float & d, float & m) {
        const block_t &  b     = xrow[kb / 8];
        const int        ib    = kb % 8;
        const uint8_t *  q3    = b.qs + 8 * ib;
        const uint16_t * gas   = (const uint16_t *) (b.qs + QK_K / 4) + 2 * ib;
        const uint32_t   aux32 = gas[0] | (gas[1] << 16);
        d = (float) b.d * (0.5f + (aux32 >> 28)) * 0.5f;
        m = 0.0f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
            const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
            const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                dst[8 * il + j]     = (int8_t) (signs & kmask_iq2xs[j + 0] ? -grid1[j] : grid1[j]);
                dst[8 * il + 4 + j] = (int8_t) (signs & kmask_iq2xs[j + 4] ? -grid2[j] : grid2[j]);
            }
        }
    }
};

template <> struct xmx_int_traits<block_iq2_xxs> {
    using block_t = block_iq2_xxs;
    static constexpr int  qk      = QK_K;
    static constexpr bool has_min = false;
    static __dpct_inline__ void stage(const block_t * __restrict__ xrow, int kb, int8_t * dst, float & d, float & m) {
        const block_t &  b     = xrow[kb / 8];
        const int        ib    = kb % 8;
        const uint16_t * q2    = b.qs + 4 * ib;
        const uint8_t *  aux8  = (const uint8_t *) q2;
        const uint32_t   aux32 = q2[2] | (q2[3] << 16);
        d = (float) b.d * (0.5f + (aux32 >> 28)) * 0.25f;
        m = 0.0f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint8_t * grid  = (const uint8_t *) (iq2xxs_grid + aux8[il]);
            const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                dst[8 * il + j] = (int8_t) (signs & kmask_iq2xs[j] ? -grid[j] : grid[j]);
            }
        }
    }
};

// same scale/min unpack as fg_scale_min_k4 in fused-gemm.cpp
static __dpct_inline__ void xmx_int_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

template <> struct xmx_int_traits<block_q4_K> {
    using block_t = block_q4_K;
    static constexpr int  qk      = QK_K;
    static constexpr bool has_min = true;
    static __dpct_inline__ void stage(const block_t * __restrict__ xrow, int kb, int8_t * dst, float & d, float & m) {
        const block_t & b    = xrow[kb / 8];
        const int       ib32 = kb % 8;
        uint8_t sc, mb;
        xmx_int_scale_min_k4(ib32, b.scales, sc, mb);
        d = (float) b.dm[0] * sc;
        m = (float) b.dm[1] * mb;
        const uint8_t * q     = b.qs + 32 * (ib32 / 2);
        const int       shift = 4 * (ib32 & 1);
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            dst[l] = (int8_t) ((q[l] >> shift) & 0xF);
        }
    }
};

template <> struct xmx_int_traits<block_q5_K> {
    using block_t = block_q5_K;
    static constexpr int  qk      = QK_K;
    static constexpr bool has_min = true;
    static __dpct_inline__ void stage(const block_t * __restrict__ xrow, int kb, int8_t * dst, float & d, float & m) {
        const block_t & b    = xrow[kb / 8];
        const int       ib32 = kb % 8;
        uint8_t sc, mb;
        xmx_int_scale_min_k4(ib32, b.scales, sc, mb);
        d = (float) b.dm[0] * sc;
        m = (float) b.dm[1] * mb;
        const uint8_t * ql    = b.qs + 32 * (ib32 / 2);
        const int       shift = 4 * (ib32 & 1);
        const int       hm    = 1 << ib32;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            dst[l] = (int8_t) (((ql[l] >> shift) & 0xF) + ((b.qh[l] & hm) ? 16 : 0));
        }
    }
};
```

Check of the element order. In `fg_decode_*` the half2 `a[j]` holds values `2j` and `2j+1` of the
32-value step. So the f16 A tile row is values `0..31` in order, and the int8 `dst[0..31]` above
uses the same order. The q8_1 B column also holds values `0..31` of the step in order. The order
only has to match between A and B, and it does.

#### D.1.3 Reordered (SoA) weights

Q4_K and Q5_K weights are reordered by default:
- MoE: `ggml_sycl_mul_mat_id()` reorders q4_K, q5_K and q6_K unconditionally (`fused-gemm.hpp`,
  comment above `ggml_sycl_fused_dequant_gemm_f16_reorder_ok`).
- Dense: `ggml_sycl_supports_reorder_mul_mat_sycl()` lists them.

Without a SoA stage, the int8 path would almost never run for K-quants. The SoA layout is a
byte permutation per slice. Streams for q4_K are `[qs][scales][dm]`, and for q5_K
`[qs][qh][scales][dm]`, each contiguous over the `nblocks` of the slice
(`fused-gemm.cpp:614-700`, `fg_reorder_a<block_q4_K>` / `<block_q5_K>`).

Add a second entry point to the K-quant traits:

```cpp
// in xmx_int_traits<block_q4_K>
static constexpr bool has_soa = true;
// xb: base of the reordered slice; ib_row: first superblock of this row; nblocks: superblocks in the slice
static __dpct_inline__ void stage_soa(const uint8_t * __restrict__ xb, int ib_row, int nblocks, int kb,
                                      int8_t * dst, float & d, float & m) {
    const int ib   = ib_row + kb / 8;
    const int ib32 = kb % 8;
    const uint8_t *   qs     = xb + (size_t) ib * (QK_K / 2);
    const uint8_t *   scales = xb + (size_t) nblocks * (QK_K / 2) + (size_t) ib * K_SCALE_SIZE;
    const sycl::half2 dm     = *(const sycl::half2 *) (xb + (size_t) nblocks * (QK_K / 2 + K_SCALE_SIZE) +
                                                       (size_t) ib * sizeof(ggml_half2));
    uint8_t sc, mb;
    xmx_int_scale_min_k4(ib32, scales, sc, mb);
    d = (float) dm[0] * sc;
    m = (float) dm[1] * mb;
    const uint8_t * q     = qs + 32 * (ib32 / 2);
    const int       shift = 4 * (ib32 & 1);
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        dst[l] = (int8_t) ((q[l] >> shift) & 0xF);
    }
}
```

q5_K is the same with the `qh` stream at `nblocks * (QK_K / 2)`, `scales` at
`nblocks * (QK_K / 2 + QK_K / 8)`, and `dm` at `nblocks * (QK_K / 2 + QK_K / 8 + K_SCALE_SIZE)`,
exactly as `fg_reorder_a<block_q5_K>::stage` reads them. Give every other trait
`static constexpr bool has_soa = false;`.

Kernel change: add a template parameter `bool reordered` to `mul_mat_xmx_int` and replace the
A-stage call:

```cpp
if constexpr (reordered) {
    static_assert(traits::has_soa, "no SoA stage for this format");
    // the slice is this expert's weights (MoE) or the whole tensor (dense)
    traits::stage_soa((const uint8_t *) x, row * nblk_row, nrows * nblk_row, kb, dst, da, ma);
} else {
    traits::stage(x + (size_t) row * nblk_row, kb, dst, da, ma);
}
```

`nblocks = nrows * nblk_row` only holds when the reorder covers exactly this weight slice. That is
true for a dense tensor (the reorder covers the whole tensor, and the dense call needs
`src0_dd_i == src0->data`, as `fused-gemm` already checks with `fused_gemm_ok` at
`ggml-sycl.cpp:3612-3613`) and for an MoE expert slice (reordered per slice, `fused-gemm.hpp`).
Assert it in the launcher, and decline (return false) in any other case.

#### D.1.4 Dispatch

- Dense: extend `ggml_sycl_dense_xmx_int_type_ok()` (C.2) and the `switch` in
  `ggml_sycl_mul_mat_xmx_int_dense()` (C.1). Pass `reordered` through, and replace the
  `!reordered_int` gate by `!reordered_int || (ggml_sycl_xmx_int_has_soa(type) && src0_dd_i == (const char *) src0->data)`.
- MoE: the ordered MoE path only reaches the XMX kernel for q8_0 and iq4_nl
  (`mmvq.cpp:3560`, `if constexpr`). K-quant MoE prefill runs on the grouped f16 GEMM
  (`fused-gemm.cpp`). Connecting it is a separate step: add an int8 branch in
  `ggml_sycl_mul_mat_id` next to the grouped GEMM call (`ggml-sycl.cpp:6550`), driven by the
  host route order. Do this only after the dense path shows a gain for these types.

#### D.1.5 Accuracy note

Q4_K would no longer need `GGML_SYCL_FAST_AND_SLOPPY` (`common.hpp:134`): int8 values times f32
scales cannot overflow the way f16-decoded weights can. The `test-backend-ops` case with
`amax = 100000` that blocks the f16 path should pass on the int8 path. Check it explicitly.

#### D.1.6 Test and benchmark

```sh
GGML_SYCL_DENSE_XMX_INT=1 ./build/bin/test-backend-ops -b SYCL0 -o MUL_MAT
for v in 0 1; do GGML_SYCL_DENSE_XMX_INT=$v ./build/bin/llama-bench -m <Q4_K_M model> -p 512 -n 0 -ub 32,64; done
```

Run the benchmark for each newly enabled type (IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS, Q4_K, Q5_K).
Keep a type in `ggml_sycl_dense_xmx_int_type_ok()` only if it is faster than the f16 fused GEMM.

### D.2 Keep the scale epilogue in registers

**Implemented, but its device-program build fails.** See D.2.4, re-tested 2026-10-03, and section 0.3 option B for
why it should not be assumed to pay off once it does.


#### D.2.1 Problem

The Part A kernel does, per 32-value k step and per M tile: `joint_matrix_store` of the int32
tile to SLM, a barrier, 32 SLM reads per lane and 32 FMAs per lane. The MAD itself is one
instruction sequence. The epilogue is probably a large part of the time. The scales differ per
k step and per row, so some per-element work every k step is unavoidable. The SLM round-trip
and the barrier are avoidable.

The Q8_K idea from the earlier version of this plan (one activation scale per 256 values, with
the integer sub-scales applied in int32) does **not** remove this work. The int32 tile still has
to be multiplied by a per-row sub-scale every 32-value step, so the per-element operation count
stays the same. It is dropped.

#### D.2.2 Design

Keep one f32 accumulator tile per M tile in registers. After each MAD:
1. Convert the int32 tile to scaled f32 in place, with the coordinate form of
   `joint_matrix_apply`. The f32 bits are stored in the int32 element (bit cast), because the
   element type cannot change.
2. Add it into the f32 accumulator with the two-matrix form of `joint_matrix_apply`.

The scales are read from SLM by coordinate, so B scales and B sums must be in SLM (not per-lane
registers). No barrier is needed after the MAD.

```cpp
namespace mx  = sycl::ext::oneapi::experimental::matrix;
namespace imx = sycl::ext::intel::experimental::matrix;

// once per route tile
mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, XMX_INT_TM, XMX_INT_TN> acc[XMX_INT_MT];
#pragma unroll
for (int mt = 0; mt < XMX_INT_MT; ++mt) {
    mx::joint_matrix_fill(sg, acc[mt], 0.0f);
}

// per k step, after staging (lane writes scales_b[lane] = sb and sums_b[lane] = sumb next to tile_b)
// and after the one barrier that follows staging:
#pragma unroll
for (int mt = 0; mt < XMX_INT_MT; ++mt) {
    mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, XMX_INT_TM, XMX_INT_TN> c;
    mx::joint_matrix_fill(sg, c, 0);
    mx::joint_matrix_load(sg, sub_a, /* as in Part A */);
    mx::joint_matrix_mad(sg, c, sub_a, sub_b, c);

    imx::joint_matrix_apply(sg, c, [=](int32_t & x, size_t row, size_t col) {
        float v = (float) x * scales_a[mt * XMX_INT_TM + row] * scales_b[col];
        if constexpr (traits::has_min) {
            v -= mins_a[mt * XMX_INT_TM + row] * sums_b[col];
        }
        x = sycl::bit_cast<int32_t>(v);
    });
    mx::joint_matrix_apply(sg, c, acc[mt], [=](int32_t & x, float & f) {
        f += sycl::bit_cast<float>(x);
    });
}
sycl::group_barrier(sg);  // still needed: the next k step rewrites tile_a / tile_b / scales

// once per route tile, after the k loop: store acc[mt] to tile_c as float (row-major, stride 16),
// barrier, then lane n writes column n to dst exactly as in Part A
```

What it removes per k step: 4 tile stores to SLM, 1 barrier, 32 SLM loads per lane. What it adds:
one register pass per tile for the apply. Make `tile_c` a `float` accessor for the final store.

Header: the coordinate apply lives in `sycl/ext/oneapi/matrix/matrix-intel.hpp`. Check that
`<sycl/ext/oneapi/matrix/matrix.hpp>` (already included by `fused-gemm.cpp`) pulls it in. If not,
include it directly.

Selection: template parameter `bool epi_regs` on the kernel, chosen from `GGML_SYCL_XMX_INT_EPI`
in the launcher. Keep both until measured.

#### D.2.3 Test and benchmark

Run the A.4 and C.3 tests with `GGML_SYCL_XMX_INT_EPI=0` and `=1`. Both must pass. Benchmark
A.5 and C.3 with both values. Keep the faster one as the only variant.

#### D.2.4 Current compiler status

On 2026-10-02, the register epilogue compiled and linked with oneAPI 2026.1, but the Level Zero
runtime compiler terminated the process with status 1 while building the device program on Intel
Arc Pro B60. The last `SYCL_UR_TRACE=2` event was entry to `urProgramBuildExp`; there was no return
event, SYCL exception, compiler diagnostic, or build log. The failure reproduced with only the
Q8_0 register-epilogue variant enabled and disappeared when that variant was removed.

Keep the implementation behind the default-off `GGML_SYCL_XMX_INT_EPI` flag so it can be retried
with a future compiler. The device runtime builds the register kernel even when the environment
flag is 0, so its instantiation is also gated by `GGML_SYCL_XMX_INT_EPI_EXPERIMENTAL`, default 0.
Build with that macro set to 1 and use `SYCL_UR_TRACE=2` when retesting the runtime build.

Retested 2026-10-03 with the macro at 1 and **no** IGC or `SYCL_UR_TRACE` instrumentation, to
separate this from the measurement artefact in section 0.2. **The crash is real and it is not
caused by the runtime flag.** With `GGML_SYCL_XMX_INT_EPI_EXPERIMENTAL=1` in the binary:

    GGML_SYCL_XMX_INT_EPI=1   segfault after the 3rd MUL_MAT case, 3 s
    GGML_SYCL_XMX_INT_EPI=0   segfault after the 1st MUL_MAT case, 2 s
    macro back at 0           all cases pass

So merely instantiating the register kernel is enough, exactly as first reported, and the
compile-time gate is load-bearing rather than cosmetic. Do not remove it.

Likely mechanism, from the oneAPI 2026.1 headers. The coordinate form of `joint_matrix_apply` is
`matrix-intel.hpp:643`, and the coordinate it hands the lambda comes from
`wi_element::get_coord` (`matrix-intel.hpp:129`), which is a call to
`__spirv_JointMatrixGetElementCoordINTEL` - a dynamic index into a cooperative matrix. Reading and
writing each element is `__spirv_AccessChain` on the same matrix (`matrix-intel.hpp:142`, `:169`).
A dynamic cooperative-matrix index is the one indexing form IGC is least likely to keep in
registers: the usual lowering is to materialise the whole matrix to scratch memory and index that,
which would replace an explicit, bounded SLM round trip with a compiler-chosen one. That is a
reason to be **sceptical** of D.2 even once it compiles, not a reason to be optimistic about it:
before spending more time, dump the assembly and count SLM and scratch instructions against the
current 42 and 46.

The two-matrix form (`matrix-unified.hpp:146`) is the same mechanism with two matrices, so it is
not an escape route either.

### D.3 Dense prefill with more than 64 columns

**Both steps measured. Step 1 lost badly, step 2 recovered 2.28x and still lost by 6.4x. Keep
the 64 column default; the wide kernel is kept only as the vehicle for future epilogue work.**


#### D.3.1 Today

For N > 64 (any normal prefill ubatch) the fused f16 GEMM declines
(`ggml_sycl_fused_dequant_gemm_f16_shape_ok`, `N <= GGML_SYCL_FG_MAX_N`). The weights are then
expanded to f16 in memory (`to_fp16_sycl`) and oneDNN or oneMKL runs an f16 GEMM
(`ggml-sycl.cpp:3622-3660`).

#### D.3.2 Step 1: measure the Part C kernel as it is

The Part C kernel has no column limit of its own: the grid has one slot per 16 columns. The cost
of a large N is that every slot stages the same A rows again, so A is read `N / 16` times (32
times at N = 512). Those reads mostly hit L2, but this may still be the limit.

Step 1 is zero new code. Make the limit a flag value instead of a constant:

```cpp
// replaces the constant from C.2; GGML_SYCL_DENSE_XMX_INT_MAX_N, default 64
int g_ggml_sycl_dense_xmx_int_max_n = 64;
```

Add it with the four-place pattern, then benchmark:

```sh
for n in 64 512; do GGML_SYCL_DENSE_XMX_INT=1 GGML_SYCL_DENSE_XMX_INT_MAX_N=$n \
    ./build/bin/llama-bench -m <q8_0 dense model> -p 512 -n 0 -ub 512; done
GGML_SYCL_DENSE_XMX_INT=0 ./build/bin/llama-bench -m <q8_0 dense model> -p 512 -n 0 -ub 512
```

If `MAX_N=512` beats the f16 library path, raise the default and stop here. If it loses, go to
step 2.

#### D.3.3 Step 2: share the A tile across sub-groups

Change the work-group from one sub-group to `SG_N = 4` sub-groups (64 lanes). The work-group
covers 32 weight rows by 64 columns:
- All 64 lanes stage A cooperatively: lanes 0..31 stage one row each, so A is staged once per
  64 columns instead of once per 16.
- Sub-group `s` stages and owns its own B tile (columns `16s .. 16s + 15`) and its own
  `sums` / accumulators.
- Barriers become `sycl::group_barrier(item.get_group())`.
- SLM per work-group: A 1 KiB, B 4 x 512 B, scales and C as before times 4. About 11 KiB.

Kernel sketch of what changes against Part A:

```cpp
static constexpr int XMX_INT_SG_N = 4;

// launcher: local range (1, 16 * XMX_INT_SG_N); slots = ceil(N / (16 * XMX_INT_SG_N))
// kernel:
const int sg_id = sg.get_group_id()[0];
const int lid   = item.get_local_linear_id();          // 0 .. 63
// dense range for this sub-group: columns slot * 64 + sg_id * 16 + lane
// A stage: lanes 0..31 each stage row lid; lanes 32..63 skip the A stage
// B stage: per sub-group, into tile_b + sg_id * (XMX_INT_TK / 4 * XMX_INT_TN)
sycl::group_barrier(item.get_group());
// MADs: every sub-group loads the shared tile_a and its own tile_b slice
// epilogue: per sub-group, as in Part A or D.2
sycl::group_barrier(item.get_group());
```

Use a separate kernel (`mul_mat_xmx_int_wide`) and select it for N > 64. Keep Part C's kernel
for N <= 64.

#### D.3.4 Alternative not chosen: oneDNN int8 matmul

oneDNN can run s8 x s8 with grouped scales, but it would need the weights as a plain s8 matrix
plus a separate scale tensor. That is a second copy of every weight (the q8_0 blocks are still
needed by MMVQ for decode), so it doubles weight memory. Rejected unless memory is not a concern.

### D.4 Flash attention

D.4.1 is a real candidate. D.4.2 and D.4.3 start with measurement, because the expected gain
is small or unknown.

#### D.4.1 Int8 Q.K^T for a Q8_0 K cache (oneMKL path)

Today, per KV chunk, `fattn-mkl.cpp:1083-1090` dequantizes K to f16 (`mkl_fa_dequant_chunk`)
and oneMKL runs an f16 GEMM (`:1101-1110`). The online Q8_0 XMX KQ kernel exists but is off,
because it was coupled to an online VKQ that is slower than the staged oneMKL one (comment at
`:820`).

Design: a new KQ-only int8 kernel, with V still staged and multiplied by oneMKL as today.
- A operand: K rows of the chunk, read in place through `mkl_fa_q8_row` / `mkl_fa_q8_block`
  (`fattn-mkl.cpp:452-476`). They already handle the canonical and the SoA (`kv-soa.hpp`)
  layout. The int8 values are used as they are, with scale `d` per 32.
- B operand: Q rows quantized to q8_1 blocks (32 values per block along DKQ, `q_scale`
  applied before quantizing). New pack kernel, one work-item per 32-value block:

```cpp
// dst row r = iqg * n_queries + q, DKQ / 32 blocks per row, same row order as mkl_fa_pack_q_fp16
static void mkl_fa_pack_q_q8_1(dpct::queue_ptr stream, block_q8_1 * dst, const float * q_src,
                               int n_queries, int DKQ, int gqa_ratio, int kvh_base_head, float q_scale,
                               int64_t q_row_stride, int64_t q_head_stride) {
    const int     nb      = DKQ / QK8_1;
    const int64_t n_items = (int64_t) gqa_ratio * n_queries * nb;
    stream->parallel_for(sycl::range<1>(n_items), [=](sycl::id<1> id) {
        const int64_t i   = id[0];
        const int64_t row = i / nb;
        const int     ib  = i - row * nb;
        const int     iqg = row / n_queries;
        const int     q   = row - (int64_t) iqg * n_queries;
        const float * src = q_src + (int64_t) q * q_row_stride + (int64_t) (kvh_base_head + iqg) * q_head_stride + ib * QK8_1;
        float v[QK8_1];
        float amax = 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            v[j] = src[j] * q_scale;
            amax = sycl::fmax(amax, sycl::fabs(v[j]));
        }
        const float d  = amax / 127.0f;
        const float id_ = d > 0.0f ? 1.0f / d : 0.0f;
        block_q8_1 & b = dst[i];
        int sum = 0;
        for (int j = 0; j < QK8_1; ++j) {
            const int qv = (int) sycl::round(v[j] * id_);
            b.qs[j] = (int8_t) qv;
            sum += qv;
        }
        b.ds = sycl::half2(sycl::half(d), sycl::half(d * sum));
    });
}
```

- Kernel: copy `mul_mat_xmx_int` (Part A, with the D.2 epilogue if it won) into `fattn-mkl.cpp`
  as `mkl_fa_kq_int`, and replace the A stage with:

```cpp
if (row < M) {
    const char *   xrow = mkl_fa_q8_row(K_desc, ikvh, chunk_start + row);
    const int8_t * bq;
    float          bd;
    mkl_fa_q8_block(K_desc, xrow, kb, bq, bd);
    // bq is 2-byte aligned (canonical) or 16-byte aligned (SoA); copy as 16 x uint16
    const uint16_t * s16 = (const uint16_t *) bq;
    uint16_t *       d16 = (uint16_t *) dst;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        d16[j] = s16[j];
    }
    da = bd;
}
```

  Rows are the `M = this_chunk` keys, columns are the `N = q_rows` query rows of the tile
  starting at `q0`, and the output goes to `KQ_f32[col * ld + row]` (the same layout oneMKL
  writes with `ldc = ld`). Use the dense row policy with
  `y = Q_q8 + (q0 + col) * (DKQ / 32)` and `d = KQ_f32 + col * ld`.
- Orchestrator (`ggml_sycl_flash_attn_ext_mkl`):
  - New `const bool kq_int = g_ggml_sycl_fa_kq_int && K->type == GGML_TYPE_Q8_0 && !V_is_K_view && DKQ % 32 == 0;`
  - When `kq_int`: pack Q once per KV head with `mkl_fa_pack_q_q8_1` instead of
    `mkl_fa_pack_q_fp16`, skip the K half of `mkl_fa_dequant_chunk` (keep the V half), and call
    `mkl_fa_kq_int` instead of the KQ `gemm(...)`.
  - `!V_is_K_view` is needed because a V that is a view of K reads the dequantized K chunk.
  - Everything after the KQ (softmax, VKQ GEMM, normalize) is unchanged.

Accuracy: the CPU reference for a Q8_0 K cache also quantizes Q to 8 bit (the q8_0 `vec_dot_type`),
so `test-backend-ops` should pass at its current tolerance.

Test and benchmark:

```sh
GGML_SYCL_FA_KQ_INT=1 ./build/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
for v in 0 1; do GGML_SYCL_FA_KQ_INT=$v ./build/bin/llama-bench -m <model> -fa 1 -ctk q8_0 -ctv q8_0 -p 4096,16384 -n 0; done
```

For per-stage times, run one prompt with `GGML_SYCL_MKL_FA_DEBUG=1 GGML_SYCL_MKL_FA_DRAIN=1`
(`fattn-mkl.cpp:878-923`; the drain makes each stage's time its own). Compare
`gemm_kq_time_us + dequant_time_us` between the two runs. Do not use drain mode for the t/s
numbers.

Current result on Intel Arc Pro B60 with oneAPI 2026.1: keep this path default-off. For the
Q8_0 prefill case `D=256, n_kv=2048, n_q=1024, n_qh=24, n_kvh=2`, the normal benchmark was
18.83 TFLOPS with the existing path and 5.30 TFLOPS with integer KQ. Instrumented stage times
were about 0.75 ms for the existing KQ GEMM and 7.82 ms for integer KQ. Softmax and VKQ times
did not change materially, and packing plus other overhead was slightly lower on the integer
path, so the regression is inside the integer KQ kernel.

IGC assembly shows that this is not a register-spill problem. The integer kernel uses 128 GRFs,
reports `eu_thread_count: 8`, has no scratch-memory allocation, and emits no spill loads or
stores. For each 32-value K step it issues four `dpas.8x8` instructions, but stores all four
int32 accumulator tiles to SLM, reads the 32 result rows back from SLM, and applies the f32
scales before the next K step. The kernel has 42 static SLM loads and 46 static SLM stores,
compared with four static DPAS instructions. The source subgroup barriers do not become hardware
gateway barriers for this one-subgroup work-group; the assembly contains one SLM fence. The cost
is the SLM traffic and its data dependencies. The XMX instructions are present, but their duty
cycle is low.

Increasing `GGML_SYCL_MKL_FA_Q_TILE` improved the integer result from 3.64 TFLOPS at 256 rows to
5.30 TFLOPS with a requested 4096-row tile (about 3537 effective rows after the memory cap), then
it was flat through the largest effective tile. The full operation already launches about 98,000
one-subgroup work-groups, so more global batching does not fix the kernel. A useful future
experiment is a wider work-group with several subgroups sharing one K tile across several
16-column Q tiles. That reduces duplicate K staging, but it does not remove the dominant
per-block accumulator SLM round-trip. Retest the D.2 register epilogue first when the compiler
issue is fixed.

#### D.4.2 Skip fully masked (query tile, KV chunk) pairs: measure first

**Measured, NO GO.** 0 of 44 skippable pairs at 10k context with 512 queries, and 0 of 66 at 20k.
The instrumentation is retained because it is cheap and debug-only; the skipping is not built.


The oneMKL path runs KQ, softmax and VKQ for every (KV chunk, query tile) pair
(`fattn-mkl.cpp:1076-1093`), including pairs where the causal mask hides every cell.

Expected value is small for long prompts. For a causal ubatch of `U` tokens at the end of `n_kv`
cells, the hidden part is about `U / 2` cells per query out of `n_kv`. That is 50% for the first
ubatch, but only `U / (2 * n_kv)` later (1.6% at U = 512, n_kv = 16k). There is a second
obstacle: the query rows of a tile are ordered GQA head first (`row = iqg * n_queries + q`), so a
tile usually covers every query position, and a chunk is skippable only when *no* query in the
whole ubatch sees it.

Measurement step (no behaviour change):
- The mask is packed on the host (`ggml_sycl_kq_mask_pack`, `kq-mask-bits.cpp:100`). While
  packing, record per row the last visible column, and keep the result in a small host-side
  table keyed by the mask's `data` pointer.
- In the orchestrator, under a debug env var, count chunks where `chunk_start` is past every
  row's last visible column, and print the count per call.

Only if that count is a meaningful share of FA time on the target workloads: skip those chunks
on the host. To also catch the triangle, the Q packing would have to order rows query first
(`row = q * gqa_ratio + iqg`), so that a tile is a contiguous range of query positions. That
changes the pack, the softmax row indexing and the normalize scatter. Treat it as its own
design task.

#### D.4.3 Skip chunks whose scores cannot matter: measure first

**Measured, NO GO.** 0 of 32768 at 10k context and 0 of 49152 at 20k, against a 30% threshold.
Do not build the bound.


An upper bound on every score of a (query tile, chunk) pair, if it is below
`running_max - T` (for example `T = 20`, so `exp(score - max) < 2e-9`), means the chunk adds
nothing and its exp, row sum and VKQ can be skipped. Two cheap bounds:
- Cauchy-Schwarz: `|q . k| <= |q| * max_j |k_j|`, with the max key norm per chunk computed once.
- Element-wise: `q . k <= sum_d |q_d| * max_j |k_j,d|`.

Both are loose: real dot products are much smaller than these bounds, so they may rarely trigger.

Measurement step: in `mkl_fa_softmax_chunk_1p` (`fattn-mkl.cpp:262`), under a debug env var, count
rows where `cmax < old_max - T` with an atomic counter, and print the share. That is the best
case (it uses the real max, not a bound). Build the bound only if the share is high (above about
30%) on long-context workloads. Any skip changes results by a bounded amount, so it must also be
opt-in, like `GGML_SYCL_FAST_AND_SLOPPY`.

Masks themselves are already one bit per cell (`kq-mask-bits.hpp`, `qsa-mask.cpp`). Nothing
more to gain from storing them as integers.

---

## Part E: checked and rejected

Summary of verdicts, so the table in section 0 is not the only place this is written down:

| Path | Flag default | Why it is not in the default build |
|---|---|---|
| A int8 MoE XMX | 0 | correct, but exactly neutral: the MoE kernel is weight-streaming bound |
| B int8 QSA score | 0 | 2.5% slower at 8k blocks, no win at 1k or 32k |
| C int8 dense <=64 cols | 0 | never beat the f16 fused GEMM |
| D.1 eight formats + SoA | 0 | correct for all 8, still no win over the f16 GEMM |
| D.2 register epilogue | 0 + compile gate | device-program build fails; see D.2.4 and 0.3 option B |
| D.3.2 wide kernel | MAX_N=64 | 2.28x better than D.3 step 1, still 6.4x short of the library |
| D.4.1 int8 KQ | 0 | 3.57x slower; KQ stage alone 10.4x slower |
| D.4.2 / D.4.3 skip gates | instrumentation only | 0 skippable pairs and 0 negligible chunks measured |

If this work is ever revived, the only entry point left from Parts A to D is 0.3 option B, and only
once the compiler is fixed - and section 0.3 says to check its assembly before believing it. Option
C was implemented and measured; it is rejected above. E.2 is where the next idea actually is, and
it is not about the GEMMs.

### E.1 The query tile is mistuned, and that is measurable without any code change

Swept end to end, `FLASH_ATTN_EXT(hsk=256,nh=2,nr23=[16,1],kv=20000,nb=512,type_K=q8_0,
type_V=q8_0)`, two interleaved reps, no drain:

    GGML_SYCL_FA_MAX_MEM_MIB  Q_TILE   effective q_tile   us/run      TFLOPS
    256 (default)             unset    768                15566       21.55
    256                       1024+    918                15161       22.15
    512                       3072     -                  15045       22.30
    1024                      6144     4930               14156       23.70

Two separate effects:

- **+2.6% for free.** The default is clamped to 768 rows by the L2 rule,
  `MKL_FA_Q_TILE_L2_BYTES / (4 * chunk_ld)` (24 MiB / 32 KiB). Setting `GGML_SYCL_MKL_FA_Q_TILE`
  above 1024 skips that clamp and lets the memory budget decide, which lands on 918. Same scratch,
  fewer launches, ~2.6%. A one-constant change, no accuracy risk.
- **+10%, but it is not free.** 1024 MiB and 6144 rows reach 4930 rows per tile and 23.70 TFLOPS,
  and the isolated softmax time falls with it. The cost is scratch: `buf_mb` goes from **56.8 to
  256.0 MiB** per call. On this three-GPU deployment that is not a free knob - HANDOVER.md section 2
  records that the shipped memory work exists precisely because card 0 used to run from a 40 MiB
  free-VRAM cliff, and raising this ceiling spends headroom that was deliberately bought back. This
  is the user's trade-off to make, not a default to change.

Note the drain-mode sweep of the same knob ranks things differently (it prefers 256, because drain
makes every extra launch expensive). Trust the no-drain numbers, which is the same lesson as
section 0.2: the instrument was the problem.

- **Cascaded max (exponent first, then mantissa).** Comparing the float bit pattern as an
  integer (sign-flipped for negatives) already gives exact float order. `topk-radix.hpp` does
  this. On Xe, float max and int max are both one instruction, so a two-stage compare saves
  nothing in registers.
- **Exponent-only max as the softmax stabilizer.** `m = 2^(e_max + 1)` can overshoot the true
  max by up to its own size (max 100 gives m 128, and exp(-28) is about 7e-13). Fine in f32, but
  it underflows if P is stored in f16.
- **Integer exp.** Xe has a hardware exp2. A shift plus lookup table is unlikely to be faster.
- **Int8 P.V.** P in [0, 1] quantized to u8 loses accuracy on peaky rows. Published int8
  attention work (SageAttention) keeps P.V in f16 or fp8. Q8_0 V scales also run along
  head_dim, not along the reduction axis, so each 32-wide output slice would need its own
  quantized copy of P.
- **Q8_K activations for K-quants** (one scale per 256). The per-row sub-scale still has to be
  applied to the int32 tile every 32 values, so the epilogue work does not shrink (see D.2.1).
- **Formats with a scale every 16 values** (IQ2_XS, IQ2_S, Q6_K) and **IQ1_S / IQ1_M**
  (non-integer delta). See D.1.1.
- **oneDNN int8 matmul for dense prefill.** Needs a second s8 copy of the weights. See D.3.4.
- **Single-token decode.** Memory-bound mat-vec. MMVQ and ESIMD stay the right path.

### E.2 "Merge last": accumulate in int32 and apply the f32 scale once

This is the idea that motivates the whole document, so it deserves a straight answer rather than
being spread across sections. It applies to the six formats in D.1.1 that are natively
`int8 value * f32 scale` with no min (q8_0, iq4_nl, iq4_xs, iq3_s, iq3_xxs, iq2_xxs), plus q4_K
and q5_K with a min term. The proposal is: keep the weights in their stored integer form, never
dequantize, and instead of scaling and accumulating after every 32-value block, accumulate the
int32 partial products and apply all the f32 scaling once at the end.

**It cannot be done while the weights are in their stored layout, and the reason is the block
structure, not the implementation.** Write the dot product as

    dot[row,col] = sum_j  dA[row,j] * dB[col,j] * I[row,col,j]

over blocks of 32, where `I` is the exact int32 inner product of one block on both sides. Int32
accumulation can span exactly those `j` whose scale pair `dA[j] * dB[j]` is identical, because only
then is a plain int32 sum equal to the true weighted sum. The scale pair is constant over exactly
one 32-value block, because that is how the format stores it: `block_q8_0` is
`{half d; int8_t qs[32]}`, and every row of the D.1.1 table has a scale granularity of 32 or worse.

So the accumulation span is 32 and there is exactly one f32 merge per 32 values per output element.
That is a property of the file format. `dB` is not the obstacle - activations are generated, so
they could carry one scale for the whole row - it is `dA`. Every scheme that keeps the weights where
they are lands on G = 32.

What the native `int * fscale` representation *does* buy is real, and it is why the experiment was
worth running at all:

- for `N > 64` dense, the library path expands the weights to f16 in memory first
  (`to_fp16_sycl`), which is 2 bytes per element plus a whole conversion pass. The int8 path reads
  1 byte per element in place, so it halves weight traffic and removes the pass;
- no dequant kernel runs inside the loop, the "dequant" is a scale apply on 32 bytes;
- Xe runs s8 x s8 -> s32 DPAS at twice the f16 rate.

Those three are why Part C, D.1 and D.3 were plausible. They just do not remove the merge.

**Where merge-last does work: tensors whose layout we choose.** The rule generalises cleanly. Int32
accumulation span `G` equals the scale granularity of the operands. If an operand is generated by
us, we set `G` to whatever we like, and the f32 merge happens once per output element.

The one place in this tree where the hot GEMM has **both** operands under our control is **P.V in
flash attention**, and Part E dismissed int8 P.V for a reason this construction actually repairs.

    out[q,d] = sum_kv P[q,kv] * V[kv,d]

V is stored q8_0, so `V[kv,d] = dV[kv, d/32] * v[kv,d]`. The scale depends on the column `d/32`
*and* on the row `kv`, and the reduction runs over `kv`, so along the reduction axis the scale
changes on every row. That is the same per-32 obstruction, which is what Part E recorded.

Requantize the staged V chunk to **one scale per column `d`** instead of per 32 values of `d`:

    V'[kv,d] = dV'[d] * v'[kv,d]
    out[q,d] = dV'[d] * dP[q] * sum_kv p[q,kv] * v'[kv,d]

`dV'[d]` now factors out of the `kv` reduction, so the accumulation span is the whole chunk and
there is exactly one f32 multiply per output element. Merge last, exactly as intended, with plain
int8 and no decomposition of anything.

Cost, honestly:

- The requantization would replace the V half of `mkl_fa_dequant_chunk`, but finding a per-column maximum requires a reduction over the KV chunk and usually a second pass. Writing fewer bytes does not establish a cheaper stage.
- Quantizing P requires choosing its scale, rounding, and handling online-softmax rescaling. It is not free merely because softmax already writes P. New P and V quantization adds approximation beyond the stored Q8_0 V cache; neither CPU Q8_K use elsewhere nor published attention results establish this path's accuracy. Validate unchanged backend tolerances and model-level output before any performance claim.
- s8 versus u8: P is non-negative, so a per-row scale over `[0,1]` in s8 costs one bit of range and
  needs no correction term, while a `-128` bias with u8 semantics needs `sum_kv v'[kv,d]` per
  column, which is one extra cheap reduction. The s8 route is simpler and one bit is affordable here.
- V' needs the VNNI-packed layout for the B operand; that repack fuses into the requantization.

**But it aims at the smaller of the two GEMMs.** From the instrumented breakdown in D.4.1,
`D=256, n_kv=2048, n_q=1024`:

    dequant 50 + KQ 749 + softmax 1268 + VKQ 573 = 3246 us

Replacing VKQ with a *perfect* int8 kernel saves at most 573 us, 18% of the op, and realistically a
small fraction of that: our hand-written int8 kernels run 6x below oneMKL on dense (6.90 against
44.47 TFLOPS) and 10x below it on the KQ stage. Meanwhile KQ at 749 us is already 4.5x faster in
oneMKL than our int8 attempt was, so KQ should be left alone.

**So the conclusion of this document is not "int8 XMX is a dead end" but "int8 XMX is aimed at the
wrong thing".** The score matrix is what dominates this path, and it is dominated by being moved
three times through memory rather than by any arithmetic. From the isolated per-stage figures:
dequant 339, KQ 4843, softmax 8131, VKQ 5634 for a 19311 us call, i.e. the two GEMMs plus the
softmax on the scores are essentially all of it.

**But be careful with the absolute numbers.** Those figures come from
`GGML_SYCL_MKL_FA_DRAIN=1`, which forces a `stream->wait()` after every stage, so each stage is
timed alone. The stages sum to 19311 us while the same call end to end, without drain, takes
**15545 us**: the real run overlaps about 19 ms worth of stage work. Without drain the printed
per-stage times are not stage times at all, they measure the CPU-side submit, because the queue has
not run yet.

**The proportions, however, are real.** `vtune -collect gpu-hotspots`, grouped by computing task,
`FLASH_ATTN_EXT(kv=20000,nb=512,type_K=q8_0,type_V=q8_0)`, 65 reps, 975.7 ms of GPU time:

    mkl_fa_softmax_chunk_1p                462.1 ms   47.4%   4158 launches
    gemm f16f16f32 nn_32x32 (VKQ)          236.9 ms   24.3%
    gemm f16f16f32 tn_32x64 (KQ)           175.6 ms   18.0%
    gemm f16f16f32 tn_32x48 (KQ)            35.4 ms    3.6%
    dequantize_block_nc                     17.4 ms    1.8%
    pack_q, normalize, init                  1.7 ms    0.2%

So the softmax really is the largest single kernel, and drain's 42% was about right. What drain
got wrong was only the total. Do not use it for shares.

### E.3 What the counters say: the whole path is memory-latency bound

Per-kernel stall breakdown, percentage of stall cycles:

    kernel                      occ%   active%   SBID%   Dist/Acc%   Barrier%   Send%
    softmax (q_tile 768)        78.4     25.3     44.4       15.5        9.1       5.3
    KQ gemm  tn_32x64           77.9     24.3     46.1       15.9        9.5       4.4
    VKQ gemm nn_32x32           80.1     26.6     45.0       17.0       10.1       5.1

**SBID is the largest stall bucket everywhere, by a factor of three.** SBID is the scoreboard
identifier for an outstanding memory request, so this is warps waiting for loads to return. Three
things follow, and they contradict the obvious reading of the code:

- **Not bandwidth.** The softmax runs at 159 GB/s DRAM read and 126 GB/s write against a measured
  438 GB/s DRAM write ceiling and 688 GB/s when L2-resident. It has headroom.
- **Not compute.** The softmax issues about 41e9 instructions in 0.435 s, roughly 94 G
  instructions/s, which is under 2% of this GPU's issue capability. `XMX active` at 21.7% and
  `ALU1 active` at 10.0% are both idle most of the time.
- **Under-occupied.** `XVE Array:Active` is only 25%, so the compute array frequently has nothing
  resident to run.

Occupancy is the one knob that visibly moves it. The same profile at `FA_MAX_MEM_MIB=1024` and
`Q_TILE=6144`, which raises the tile from 768 to 4930 rows:

    softmax occupancy      78-79%  ->  93-95%
    softmax XVE active     25%     ->  26-30%
    per-rep GPU time       15.01 ms ->  13.90 ms

but the stall structure does not change: SBID still 47-55%, and Barrier rises 10 -> 13-15% because
the tile now needs more cross-lane reduction per launch. And the gain is not free of trade-offs at
the kernel level either - the KQ GEMM gets **31% slower** (3.25 -> 4.25 ms per rep, its N is now
4930) while the VKQ GEMM gets **41% faster** (3.64 -> 2.15 ms).

**A counter-informed next experiment is reducing register lifetime.** The softmax holds
`sycl::float4 v[8]`, 32 floats live across the max reduction and its barrier, and it compiles to
128 GRF with no spill. Its register footprint may affect residency, but the counters do not prove it caps occupancy. Re-reading `KQ_f32` after the maximum reduction removes the score array's long lifetime at the cost of another score and mask read. Neither a 32-GRF allocation nor doubled occupancy follows from the source alone; occupancy already near 78% cannot double as a percentage. Measure compiled GRFs, spills, occupancy and end-to-end time. See H.6.

What this does **not** support: reducing bytes. The f16 score idea removes about a third of the
score traffic, but an SBID-bound kernel is waiting on the *number* of dependent round trips far
more than their size, so the expected win is well under the 29% the traffic model suggested. It
remains worth one measurement, with the tolerance untouched, but it is no longer the obvious first
move.

For completeness, the same construction applied to KQ would need one scale per `kv` row of K, i.e.
per-row requantization of K. A single outlier anywhere in a 256-wide row then destroys that row's
resolution, it buys at most 749 us, and K is read once per query tile so the requantization may not
amortize. Reject.

### E.4 Merge-last, measured and rejected: only IQ4_XS qualifies, and staging is the limit

The formats whose block scale is an exact integer can have it folded into the int8 weight value,
which lets a whole 256-value superblock accumulate in int32 and needs one f32 scale per superblock
instead of one per 32-value sub-block. Implemented behind `GGML_SYCL_DENSE_XMX_SUPER` (default 0) in
`mmvq.cpp` (`xmx_super_traits`, `mul_mat_xmx_super`) with `quantize_row_q8_K_sycl` in `quantize.hpp`.
The activation side needs one scale per 256 values, which is **not** a precision change: the CPU
reference already uses `q8_K` activations for these weight types (`ggml-cpu.c`, `vec_dot_type`).

**Scope is IQ4_XS only.** Q4_K and Q5_K do not qualify: `dequantize_row_q4_K` applies `dmin * m`
with an *independent* f16 scale, so `d*sc*q - dmin*m` is not an integer times a block scale. Writing
it that way gave 11 failures at NMSE 0.9957. IQ4_XS has no min term, so `d*(sc6-32)*kvalues` folds
exactly.

Correctness is fine: 16/16 iq4_xs `MUL_MAT` pass, and the kernel was confirmed to actually run
(temporary instrumentation showed hits at N=9, 10, 64).

Performance, `m=4096, n=512, k=14336, iq4_xs`, 60.13 GFLOP/run:

    library (oneDNN/oneMKL f16)     2545 us   23.63 TFLOPS
    existing int8 XMX               9330 us    6.44 TFLOPS
    merge-last, XMX_INT_MT 4       145491 us    0.413 TFLOPS
    merge-last, XMX_INT_MT 2        64231 us    0.936 TFLOPS
    merge-last, MT 4 + staging MLP 168958 us    0.356 TFLOPS
    merge-last, MT 2 + staging MLP  63688 us    0.944 TFLOPS

Three things came out of it:

1. **The cost was register pressure, not the 2x DPAS.** At MT 4 the kernel spills 128 bytes and
   occupancy falls to 29.8%; dropping `XMX_INT_MT` to 2 clears the spill (0 bytes, 58.9%) and gives
   **2.27x**. It is still 6.9x behind the existing int8 path, and MT 2 costs that path 14%, so the
   two kernels want different tile shapes.
2. **Higher MLP in the staging loop does nothing.** Prefetching all 16 weight codes before the table
   lookups, so the byte loads are in flight together instead of each waiting on its own
   `kvalues_iq4nl` lookup, is worth +0.9% on the merge-last path and nothing on the existing one -
   both inside noise. The byte loads were not the limiter.
3. **What is left is the per-element table lookup.** With no spill, the profile is `SBID 69.5%` and
   `XMX active 1.0%`: the kernel is waiting on the 32 dependent `kvalues_iq4nl` lookups per row per
   sub-block, not on the weight loads and not on the accumulator epilogue. That cost is inherent to
   decoding a 4-bit format and is the same one the f16 dequant path pays, which is a large part of
   why the library reaches 23.63 TFLOPS where this reaches 6.44.

Note that `XMX_INT_MT` is shared between the two kernels, so the 4-vs-2 preference above is
evidence they should be separate kernels with separate tile shapes rather than one templated kernel.

Compute buffers: no saving. `q8_K` is 292 B per 256 values against 288 B for eight `q8_1` blocks,
so the activation buffer is 4 B *larger*.

### E.5 The grid lookup was the real cost: cache it in registers

Section E.4 ended with "the per-element table lookup is the limit". Testing that with a throwaway
arithmetic stand-in for `kvalues_iq4nl` (numerically wrong, `perf` mode does not check) gave the
upper bound on the win: **6.44 -> 8.74 TFLOPS, +36%**, on the existing int8 dense path.

The exact version keeps `kvalues_iq4nl` as the single source of truth and caches it in four
registers per thread (`iq4nl_grid` in `mmvq.cpp`), so decoding a code is a shift and a three-way
select instead of a dependent load. The table is hand-tuned (gaps 23,21,18,16,14,13,12,11,12,12,13,
15,16,20,24) and has no closed form, so extraction rather than arithmetic is the only exact route.

    m=4096, n=512, k=14336, XMX_INT_MT 4

    type     int8 XMX   library    behind
    q8_0        6.90      44.45      6.4x
    iq4_xs      7.42      23.63      3.2x     (was 6.44 before this change, +15%)
    iq4_nl      7.02      14.26      2.0x
    iq3_xxs     6.69      15.54      2.3x

**+15% exact, 32/32 iq4_xs and iq4_nl `MUL_MAT` pass.** The select chain costs about half of what
the arithmetic stand-in gained, so 15% of a possible 36% is the realistic figure. This is the only
change in the whole document that makes an existing path faster rather than merely different, and
it carries no precision cost of any kind.

It does not change any verdict. `iq4_nl` at 7.02 against a 14.26 library is the closest any int8
XMX path has come - 2.0x - and that is still a loss. The other IQ grids require a separate design: they contain 256 or 512 entries, occupying 1-2 KiB, rather than the 16 bytes cached here. The existing `fg_grid_traits` SLM staging is a useful reference; it does not imply a similar gain in this kernel.

The other lesson is about method. Three hypotheses were tested and falsified in a row - transpose
the C tile, merge-last, more MLP in staging - and the counters said the same thing each time: XMX
idle, SBID saturated, the limit is per-element decode. The arithmetic stand-in cost five lines and
would have pointed straight at the answer on the first try, because it separates "how much does
this instruction cost" from "is this instruction the right one".

### E.6 Loading DPAS operands straight from global memory: works, and does not help

The API question came first and it has a clean answer. `joint_matrix_load` on the A/B operands
forbids only `private_space` (`matrix-intel.hpp:760`), and the way to get a global-space `multi_ptr`
from a USM tensor is `sycl::address_space_cast`, not `multi_ptr`'s constructor:

```cpp
auto pA = sycl::address_space_cast<sycl::access::address_space::global_space,
                                    sycl::access::decorated::no>(A);
joint_matrix_load(sg, sub_a, pA + (sg_startx * TM) * K + k * TK, K);
```

That is Intel's own XMX guidance, and it loads both operands from global with no SLM at all.

Spellings, all four tested by compiling them:

    sycl::address_space_cast<global_space, decorated::no>(raw)     works   <- what the tree uses
    syclex::annotated_ptr<int8_t>{raw}                             works   (no property list needed)
    multi_ptr<int8_t, global_space, decorated::no>(raw)            FAILS   no matching constructor
    annotated_ptr<int8_t, properties{address_space::global_space}> FAILS   enum is not a property

`multi_ptr`'s only pointer constructor takes a *decorated* pointer, which is why the bare
`annotated_ptr` works and the direct `multi_ptr` does not. Do not assume a bare `annotated_ptr`
carries `global_space`: it converts to none of `multi_ptr<int8_t, global_space/generic_space,
decorated::no/yes>`, so which space it actually yields was not established. `address_space_cast` is
the safer of the two working spellings because the space is explicit, and it is the one Intel's
optimization guide shows.

Separately, note that DPC++ 2026.0 removed the `sycl_ext_intel_usm_address_spaces` extension and
with it `ext_intel_global_device_space` and `ext_intel_global_host_space`; `global_space` is the
standard replacement. Code written against the old names will not compile on this toolkit, and the
tree already uses the new spelling.

Searching the web found the working spelling in minutes; reading headers had not.

For int8-native formats the layout already matches, and the reordered q8_0 layout
`[qs][d]` (`fg_reorder_a<block_q8_0>`) matches it *better* than canonical: element `(m, k)` of an
8x32 row_major A tile is at `(row0+m)*nblk_row*32 + kb*32 + k`, which is what a tile of stride
`nblk_row*32` expects, versus canonical q8_0's awkward `nblocks*34`.

Implemented behind `GGML_SYCL_XMX_DIRECT_A` (default 0) on the wide dense kernel. 89/89 q8_0
`MUL_MAT` pass. It is not faster, and the counters say why:

| q8_0, m=4096 n=512 k=14336, reordered | us/run | TFLOPS | dram_rd | l3rd | occ% | SBID% | XMX% |
|---|---|---|---|---|---|---|---|
| staged through SLM | 8759 | 6.86 | 50 | 1355 | 56.7 | 66.2 | 3.6 |
| direct from global | 8696 | 6.91 | 51 | 1356 | 56.4 | 65.8 | 3.6 |

Every counter is unchanged, L3 included. **The staging was never the cost.** The kernel re-reads
the weight tensor `N / 64 = 8` times, and a direct load reads exactly the same bytes from the same
place - it just puts them in DPAS registers instead of SLM first. The 1355 GB/s of L3 traffic is
the 8x re-read, and it is untouched either way.

This experiment shows no measurable benefit from removing Q8_0 staging on this shape. It does not prove that every inner-loop change is ineffective: E.5 measured an IQ4 grid-cache improvement. Widening the column tile in D.3.2 also helped; that changed the kernel's sharing and barriers as well as launch geometry.

## Part F: open lead, not yet measured

Everything here is unmeasured. Nothing in this part has been built or benchmarked. It is recorded
because the reasoning is sound and the two missing facts are cheap to get.

### F.1 The lead: fuse scalar decay in the non-KDA chunking branch

This is the graph-level version of the "value x scale gets promoted online" idea. It is not a GEMM
change and it does not touch oneMKL, so it does not collide with the constraint in Part A.

In `src/models/delta-net-base.cpp:128-146`, the ordinary gated delta net builds its decay mask and
applies it after the contraction:

```cpp
// [CS, CS, n_chunks, H_k * n_seqs]
decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
decay_mask = ggml_exp(ctx0, decay_mask);

kb = ggml_mul_mat(ctx0, k, k_b);
kb = ggml_mul(ctx0, kb, decay_mask);

kq = ggml_mul_mat(ctx0, k, q);
kq = ggml_mul(ctx0, kq, decay_mask);
```

`g_cs` is a cumsum of the gate inside a chunk. Before applying `TRI`, the exponential of a difference factorizes:

```
exp(g_cs[j] - g_cs[i]) = exp(g_cs[j]) * exp(-g_cs[i])
```

This does not make the actual masked tensor rank-1. `ggml_tri` writes zero outside the triangle, and the following `ggml_exp` turns those zeros into ones. Also, the repeated `g_cs_j` varies on axis 1, while `g_cs_i` varies on axis 0. With `c` the axis-0 column and `r` the axis-1 row, the exact graph is:

```
decay[c,r] = (c <= r) ? exp(g_cs[r] - g_cs[c]) : 1
kb[c,r] = dot(k[:,c], k_b[:,r]) * decay[c,r]
kq[c,r] = dot(k[:,c], q[:,r]) * decay[c,r]
```

Later `TRI` nodes zero the unused regions of `kb` and `kq`. A fusion that consumes those nodes can write zero there, but a fusion of `MUL_MAT -> MUL` alone must preserve the ones in the decay tensor. Use `exp(g_cs[r] - g_cs[c])` directly for the first experiment. Separate factors can overflow or underflow even when their product is finite, and change rounding. The outer-product identity supports an optimization hypothesis, not a safe implementation by itself.

The point of this is where the multiply happens. Today the mask is a real tensor: `sub` writes it,
`tri` rewrites it, `exp` rewrites it, then two multiplies read it. That is about eight read/write
passes over `CS * CS * n_chunks * CHB` elements every layer. With `CS = 64` that is 16 KB per head
per chunk, 1 MB per head at 4k context. The rank-1 factors are `2 * CS` elements against
`CS * CS`, so they are 32x smaller.

This is the "accumulate into the same register, or reduce" shape the idea was aiming at. The
multiply is already ordered after the contraction in the graph, so no graph reordering is needed -
only a fusion so that it stops making a round trip through global memory.

### F.2 Two prerequisites, both unchecked

1. **Confirm this graph actually runs.** Inspect `build_delta_net`: single-token decode takes the autoregressive or fused path; multi-token calls can take the fused chunk path instead. F.1 only applies when `build_delta_net_chunking` runs with `kda == false`.
2. **Estimate its share of multi-token GPU time.** Profile the target prefill or multi-token workload, including its backend placement and fusion settings. The `cb()` labels identify graph nodes; a GPU timeline measures their time. If the chunking graph is absent or below measurement noise, stop.

Prerequisite 2 is the one that can kill the whole idea on its own. Do it first.

### F.3 It does not work for KDA, and the reason is worth keeping

KDA (Kimi Delta Attention, `llama-model.h:543`) has a **full-rank** gate: `ssm_g` is documented as
"full-rank KDA gate (replaces ssm_g_a/ssm_g_b)" (`llama-model.h:556`). Where an ordinary gated delta
net has one decay scalar per head per token, KDA's gate carries a key-dimension axis.

| | gate shape | CS | decay factorizes out of the `S_k` sum? |
|---|---|---|---|
| ordinary GDN | `[1, n_tokens, H_v, n_seqs]` | 64 | yes on the retained triangle; the full mask is not rank-1 |
| KDA | `[S_k, n_tokens, H_v, n_seqs]` | 16 | no, varies along the reduction axis |

For KDA the decay is `exp(c_j(s) - c_i(s))` with `s` the reduction axis, so it cannot be pulled out
of the sum over `S_k`. Its `decay_mask` genuinely carries an `S_k` axis and the tensor is as large
as it looks. Do not carry this optimization into the `kda` branch at `delta-net-base.cpp:94`.

An earlier draft of this section claimed the `[S_k, CS, CS]` tensor could be removed in the KDA
branch too. It cannot, for the reason above. Recorded so nobody retries it.

### F.4 Read the gate shape before the permutation

`delta-net-base.cpp:31`:

```cpp
const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);
```

At this point `g` has shape `[g_0, H_v, n_tokens, n_seqs]`: axis 1 is the value-head count, not the token count. The function asserts that shape and only later permutes it to `[g_0, n_tokens, H_v, n_seqs]`. The condition tests a full-width gate and equality of the head counts. Do not change this dispatch as part of a SYCL performance experiment. Verify the actual gate shape and the chunking/fused selection first.

### F.5 Recorded dead end: the PLE conv is host-side

`src/models/qwen4exp.cpp:1492-1503` looks like a textbook case for this idea. The `ple_conv1d` weight
keeps its stored type, is explicitly cast to F32, then goes through a `ggml_mul` into an ADD chain:

```cpp
if (wk->type != GGML_TYPE_F32) {
    wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
}
ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
```

For UD-IQ4_XS that can be a real `kvalues_iq4nl` decode plus an F32 round trip per layer. The earlier session reported CPU placement for this convolution. That is workload-specific evidence, not something the graph construction proves. Confirm scheduler placement in the actual run before treating it as a GPU target; do not implement a GPU optimization based on this source pattern alone.

Note also that the shape would not suit DPAS anyway: the kernel is `ple_conv_kernel` long, 3 or 4,
against an int8 minimum K of 32.

### F.6 The HC ADD chain is already fused, for reference

`src/models/qwen4exp.cpp:344-357` is the other place this idea would have applied - a chain of `hc`
`ggml_add` nodes collapsing the hyper-connection streams:

```cpp
mixed = ggml_cont(ctx0, view(gated, ..., 0));
for (int64_t c = 1; c < hc; ++c) {
    mixed = ggml_add(ctx0, mixed, view(gated, ..., c));
}
mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
```

That is `hc - 1` full passes over an F32 tensor. It is already handled upstream as
`ggml_dsv4_hc_pre_gated` / `LLM_FUSED_OP_DSV4_HC_PRE`, behind `cparams.fused_dsv4_hc_pre`. Worth
confirming that flag is on in the shipped server config; if it is off, that is a larger and cheaper
win than anything in Part F.

### F.7 Framing note for whoever picks this up

Two different problems keep getting conflated in this document:

- **Bandwidth.** Places with enough bytes that moving them matters. The GEMM in Part E is the
  measured example, and the KV cache / k-pool is the place D.4.1 already lost.
- **Latency.** Places with no significant traffic but a long serial tail of tiny kernels on the
  critical path. PLE (host-side) was this shape.

F.1 is the first lead in this document that is neither: a modest tensor, a fixed number of elementwise
passes per layer, and a fusion that removes both. That is why it is worth recording even unmeasured.

## Part G: XMX-tiled weight layout (design only, nothing built)

Requested as: explore which weight reordering suits the XMX cores per quant type, no more memory than
the format it is promoted to, reorder and tile only, feature flagged. Only the best candidate is
written up here. **Nothing in Part G has been built or measured.**

### G.0 Two claims from the discussion that were wrong, kept so nobody retries them

**"iq4_xs needs a scale every 16 values, twice as hot as q8_0."** No. `stage()` runs once per 32
value DPAS k-step (`mmvq.cpp:3089`, `ib = kb % 8`, `xrow[kb / 8]`), so a super-block of 256 is **8
sub-blocks of 32**, one 6-bit scale each. The cadence is identical to q8_0: one scale per 32 values.
`QK_K/64` is the width of the `scales_l` array, not a sub-block count.

**"Pre-expanding `qs` into DPAS tiles removes redundant grid lookups without extra bytes."** Expanding 4-bit codes to int8 needs about twice the quant payload and breaks the same-footprint constraint. The current wide kernel stages each weight once per work-group and shares it across four subgroups; redundancy is across column work-groups, not across its M tiles. The tree already says at `mmvq.cpp:3958`: "iq4_xs has no min term, so
d*(sc6-32)*kvalues folds exactly. No SoA stage for iq4_xs."

### G.1 The real defect is alignment, and it is cheaper than it looks

`sizeof(block_iq4_xs)` is **136 bytes** and `qs` is at **offset 8** (`ggml-common.h:456`). Since
136 mod 16 = 8, alignment alternates by super-block:

```
blk 0: qs at   8   mod 16 = 8      <- 8-byte aligned only
blk 1: qs at 144   mod 16 = 0
blk 2: qs at 280   mod 16 = 8
blk 3: qs at 416   mod 16 = 0
```

Assuming a 16-byte aligned tensor base, `const uint8_t * q4 = b.qs + 16 * ib` is only 8-byte aligned on even super-blocks. Whether that changes the number of Intel GPU load instructions depends on compiler lowering; `ld.global.v4.b32` is CUDA PTX terminology and is not evidence about this SYCL kernel.

Splitting each super-block into a nibble plane and a scale plane keeps 128 + 8 = **136 bytes, same
footprint**, and 128 is a multiple of 16, so every super-block's `qs` becomes 16-byte aligned.

| super-block | canonical `qs` offset mod 16 | plane-split offset mod 16 |
|---|---|---|
| 0 | 8 | 0 |
| 1 | 0 | 0 |
| 2 | 8 | 0 |
| 3 | 0 | 0 |

Alignment makes a wider load possible, but does not guarantee one. Compare generated ISA for canonical and SoA staging before claiming fewer loads. Section 0.1's source-level instruction budget is not a measured load count for IQ4_XS.

This is a SoA permutation across a complete slice, not a within-super-block permutation. Its quant payload keeps row order, but its scales are in separate planes. Every reader must know the plane offsets and the slice's block count. E.6 tested direct loading of reordered Q8_0, not IQ4_XS alignment; the latter remains unmeasured.

It is still a byte permutation that no reader can guess, so consumers must be taught it - but that
is the only part of it that is not mechanical.

### G.2 The tiled layout, and why it is not the first thing to build

In the current dense integer kernel, weights are operand A: `[M,K]`. Activations are operand B: `[K,N]` in packed form. A weight layout would tile M and K, not the output-column dimension N. Changing weight order cannot remove the `ceil(N/64)` work-groups that consume each weight tile. Packed IQ4_XS nibbles also still need decoding to int8 before the MAD, so a pure byte permutation cannot make them a directly loadable int8 matrix.

Defer an M/K tiled weight layout until the ordinary SoA experiment is measured. Ragged tiles can be stored without padding, at the cost of indexing and tail handling; extra bytes are not mathematically unavoidable. Its benefit is unknown, and no 16x gain follows from contiguity alone.

#### G.2.1 Follow-up: aligned payload-and-scale tiles

This remains an unmeasured proposal. The ordinary SoA fused-GEMM readers are complete and validated in H.9, so decode and later prefill can share the same weights efficiently. Next compare a tile-local payload/scale permutation against that completed implementation, holding the math, routing, staging policy and persistent byte count fixed.

| format | blocks in a full tile | payload bytes | metadata bytes | tile bytes |
|---|---|---|---|---|
| IQ3_XXS | 8 | 768 | 16 | 784 |
| IQ4_XS | 2 | 256 | 16 | 272 |

Each full tile is a multiple of 16 bytes. With a 16-byte-aligned base, its payload and metadata sections can be accessed in aligned 16-byte chunks without padding. IQ3_XXS's payload includes its sign/sub-scale bytes; only the half scales move into the metadata section. IQ4_XS metadata retains d, scales_h and scales_l bit-for-bit. Metadata fields inside the section need extraction from those chunks; individual fields are not each 16-byte aligned.

Specify the permutation and reader offsets before enabling any producer. Expert boundaries and incomplete tiles need an explicit compact-tail rule: an aligned tile size does not align the next expert when its total byte stride is unaligned. The model in H.1.5 has expert block counts divisible by eight, but the general tests must also cover odd block counts, row tails and partial tiles. Do not assume a 16-byte-aligned expert base or add persistent padding silently.

Measure whether the compiler emits fewer loads and whether nearby metadata improves locality. Check generated ISA separately from clean timing runs; alignment alone does not prove either effect. Repeat decode -> prefill -> decode correctness, all enabled staging readers, transfers, views and graph replay. Accept only a repeated model-level gain without a material later-prefill regression. The tiled layout must have fused-GEMM readers too; changing the producer alone would repeat the consumer gap fixed in H.9.

### G.3 Marking: reuse the existing SoA descriptor

The bool is already gone. `common.hpp:469`:

```cpp
enum ggml_sycl_layout_kind : uint8_t {
    GGML_SYCL_LAYOUT_CANONICAL  = 0,  // exactly as ggml packs the type
    GGML_SYCL_LAYOUT_SOA_SPAN   = 1,  // quants then scales, repeating every `span` elements
    GGML_SYCL_LAYOUT_SOA_WHOLE  = 2,  // quants then scales, once over the whole tensor
    GGML_SYCL_LAYOUT_MASK_BITS  = 3,  // a {0,-inf} mask packed to one bit per element
};

struct ggml_sycl_layout {
    ggml_sycl_layout_kind kind = GGML_SYCL_LAYOUT_CANONICAL;
    ggml_type             type = GGML_TYPE_COUNT;  // the block type the permutation applies to
    int32_t               span = 0;                // elements per self-contained unit
    size_t                nbytes = 0;              // bytes held when below ggml_nbytes, else 0

    bool is_canonical() const { return kind == GGML_SYCL_LAYOUT_CANONICAL; }
};

struct optimize_feature {
    ggml_sycl_layout layout;

    // the whole-tensor SoA that reorder_qw() produces
    bool is_reordered() const { return layout.kind == GGML_SYCL_LAYOUT_SOA_WHOLE; }
    void set_reordered(ggml_type t) { layout.kind = GGML_SYCL_LAYOUT_SOA_WHOLE; layout.type = t; }
};
```

The weight SoA and the SoA KV cache already share one descriptor rather than a bool beside it,
which is the shape this needs. Two notes:

- `is_reordered()` is a kind test. A new kind would return false and could silently reach canonical readers. That is unsafe after bytes have been permuted. The proposed plane split fits `GGML_SYCL_LAYOUT_SOA_WHOLE`; reuse it, with type-specific readers and the existing per-expert interpretation. No new enumerator is needed for this experiment.
- `device_opt_feature::reorder` (`common.hpp:501`) is a **per-device policy** bool, not a layout.
  It should stay a bool. Do not fold it into the enum.

### G.4 The consumers, counted

Historical count: 56 `is_reordered()` call sites. This is a search aid, not a complete consumer inventory: helpers can receive only a pointer or a propagated boolean. Recount and trace actual dispatch for the chosen type. Per file:

| file | sites | what reads the weight |
|---|---|---|
| `convert.cpp` | 20 | `cpy` to/from a reordered src0 |
| `ggml-sycl.cpp` | 15 | dequantize, get_rows, set_rows, mul_mat entry gates, buffer init |
| `mmvq.cpp` | 10 | `mul_mat_vec`, `mul_mat_vec` ncols 2-8, MoE `mul_mat_vec_q` |
| `dmmv.cpp` | 10 | dequantize-then-mul-mat-vec |
| `common.hpp` | 1 | the accessor itself |

`ggml-sycl.cpp:8203` is the gate to copy the shape of - an `if (node->op == ...)` check placed ahead
of the canonical path so the old path stays untouched:

```cpp
if (node->op == GGML_OP_SSM_CONV && ggml_sycl_can_fuse(...)) { ... }
```

### G.5 Minimum to get a measurement, in order

The instruction was to do the minimum needed to see whether there is any difference at all. In
dependency order, each step separately useful.

**First, a correction to what "minimum" means here.** Adding an enum value is inert - nothing reads
it, so nothing breaks. A byte permutation is the opposite. `opt_for_reorder` (`ggml-sycl.cpp:5687`)
repacks and then calls `set_reordered()`, which changes how *every* later reader indexes the tensor.
So producer-only is not a safe intermediate state: if `IQ4_XS` is added to `supports_reorder_mmvq`
without teaching `mul_mat_vec_iq4_xs_q8_1_sycl` the new layout, the tensor is repacked and then read
canonically, which is plausible garbage rather than an error. Minimum testable is producer plus a closed set of supported consumers plus a flag. One consumer is sufficient only for an isolated test that cannot dispatch anywhere else. H.2-H.4 define the production coverage requirement.

1. **Type bit**, reserve an unused bit for `GGML_SYCL_REORDER_IQ4_XS` in the `ggml_sycl_reorder_type` enum
   (`common.hpp:104`). One bit per type, so a new type is one bit and not another env var.
2. **The producer**, `reorder_qw_iq4_xs`, plus a case in the dense `reorder_qw` switch
   (`ggml-sycl.cpp:5664`). The layout contract, from the q8_0 version at `dmmv.cpp:2012`:

   ```
   q8_0 :  [qs: nb*32][d: nb*2]
   iq4_xs: [qs: nb*128][d: nb*2][scales_h: nb*2][scales_l: nb*4]   = 136*nb, unchanged
   ```

   `nb` is `nrows * blocks_per_row`, as in `reorder_qw_q8_0`. Assert `nbytes` unchanged.
   **It cannot reuse `reorder_qw_soa2_moe`** - that template's `static_assert` requires one
   half-scale followed by flat quants, and iq4_xs has three scale fields.
3. **Consumers**: use H.4's coverage list. A dense mat-vec reader alone does not make a persistent MoE weight permutation safe.
4. **Scale-plane order**, decided before the producer is written. There are 22 `get_d_offset` call
   sites. All three stored fields belong to a 256-value superblock: one `d`, one `scales_h`, and four `scales_l` bytes encoding eight 6-bit sub-scales. Getting this wrong makes a
   `get_d_offset` consumer read the wrong 2 bytes and produce plausible garbage.
5. **Correctness before any speed claim**: the full iq4_xs `MUL_MAT` matrix, plus a
   `test-backend-ops` case that compares explicitly against canonical. A permutation bug here
   produces plausible garbage rather than an error - the warning already at `kv-soa.hpp:24`.

On which type: **iq4_xs, not q8_0.** q8_0's alignment is already fixed by `reorder_qw_q8_0`, and its
canonical `qs` at offset 2 in a 34-byte block cycles through residues 2,4,...14,0, so it is never
reliably 16-byte aligned; the SoA plane at `ib * QK8_0` is 32-byte aligned throughout. That is
strictly stronger than what G.1 asks for. But see G.7 before concluding anything from G.1's expected
value. The performance benefit is unknown until the relevant consumers are measured.

### G.6 Expected value of G.1 specifically

Unknown. G.1 might reduce staging instructions, but alignment alone does not prove that, and E.5 already shows an inner-loop change can help. G.2 is deferred, not disproven. Dense integer-XMX results also do not predict the MoE mat-vec result.

An earlier draft of this section argued the whole of Part G was low value because q8_0's reorder
"measured flat". **That was wrong, and the correction is G.7.** Do not read G.1's verdict as a
verdict on reorder.

### G.7 The actual prize: reorder is not flat, and iq4_xs is excluded from where it pays

This is the finding that matters, and it was nearly missed by taking a microbenchmark for the whole
story. From `common.hpp:111`:

```
// Q8_0 was left out after an early A/B measured flat, but
// the reordered q8_0 MoE mat-vec is 5.1x faster and an interleaved llama-bench A/B on qwen4exp
// measured tg128 +2.5% and pp2048 +1.0%.
```

The "flat" in this document's history - 6.86 -> 6.91 - is the **dense int8 XMX** kernel, one
consumer. The same reorder through a different consumer, MoE `mul_mat_vec`, is **5.1x faster**, and
end-to-end on qwen4exp it was **tg128 +2.5%, pp2048 +1.0%**. Reorder pays, and it pays in
`MUL_MAT_ID`.

The types that get it, `ggml_sycl_mul_mat_id_reorders_type` (`ggml-sycl.cpp:5731`):

```cpp
return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K ||
       (type == GGML_TYPE_IQ3_S  && (g_ggml_sycl_reorder_types & GGML_SYCL_REORDER_IQ3_S)) ||
       (type == GGML_TYPE_IQ4_NL && (g_ggml_sycl_reorder_types & GGML_SYCL_REORDER_IQ4_NL)) ||
       (type == GGML_TYPE_Q8_0   && (g_ggml_sycl_reorder_types & GGML_SYCL_REORDER_Q8_0));
```

**IQ4_XS is absent.** So is IQ3_XXS. And there is no `reorder_qw_iq4_xs` producer at all - the
dense producers are `q2_K, q3_K, q4_0, q4_K, q5_K, q6_K, q8_0, iq3_s, iq3_xxs` plus the generic
`reorder_qw_soa2_moe`. Grep for `IQ4_XS` against the reorder machinery returns nothing.

If qwen4exp's expert weights are IQ4_XS, they currently cannot use reordered MoE mat-vec. That is worth testing, but Q8_0's reported 5.1x kernel gain is not an expected gain for a different quant format. Its decoder, table lookups and dispatch costs differ.

**The blocker for a measurement.** The MoE path has no wired producer for these types:
`reorder_qw_iq3_s_moe`, `reorder_qw_q4_k_moe`, `q5_k_moe`, `q6_k_moe`, `q6_k_chunked` and the
generic `soa2_moe` exist, but there is no `reorder_qw_iq4_xs_moe` and no `reorder_qw_iq3_xxs_moe`.
Adding iq4_xs to the list without a MoE producer would set a bit for a producer that returns false,
so the list change is inert until the producer is wired. IQ3_XXS can use the existing `reorder_qw_soa2_moe<block_iq3_xxs, 3*QK_K/8>` template: its block is one half followed by 96 flat `qs` bytes. IQ4_XS cannot use that template because it has additional scale fields.

**Historical model inventory.** The earlier session recorded `/root/models/Qwen3-30B-A3B-UD-IQ3_XXS.gguf` and no qwen4exp GGUF. Recheck paths and tensor metadata before planning model benchmarks.

| | dense producer | MoE list | MoE producer | end-to-end measurable here? |
|---|---|---|---|---|
| iq4_xs | none | absent | none | no - model not local |
| IQ3_XXS | `reorder_qw_iq3_xxs` | absent | none | yes - local model |

IQ3_XXS is the smaller first experiment: it already has a dense producer, reordered `block_q_t` metadata, a reordered vector-dot helper, and conversion support. IQ4_XS lacks those reordered readers. Measure IQ3_XXS first if the local model's expert tensors use it, then consider IQ4_XS separately. A win on one is evidence to investigate the other, not proof of the same gain or implementation cost. Verify local model availability rather than assuming the session's inventory is current.

**Unverified, and it must not be assumed:** whether qwen4exp's expert tensors are iq4_xs at all. The
GGUF is a mixed-precision UD mix and the tree already handles per-tensor type in MoE paths
(`mmvq.cpp:5316`). Check the tensor dtypes before building either variant.

## Part H: current implementation handoff

This is a plan for controlled experiments, not a promise of a speedup. The completed integer-XMX replacements remain default-off. Start with H.1-H.3; do H.4 only after a useful IQ3_XXS result or a confirmed IQ4_XS workload. H.6 is an independent attention experiment. F is deferred until its graph and time share are confirmed.

### H.0 Implementation status (2026-10-04)

H.2 and H.4 were validated as default-off experiments, with the reordered fused-GEMM readers completed in H.9 and default enablement in H.9.3. The earlier handoff's unvalidated IQ3_XXS producer and dispatch were incomplete: metadata allocation still depended on the IQ3_S bit, GLU dispatch was missing, and backend transfer hooks did not invert weight SoA. These gaps are now addressed. Nothing is committed.

| step | state | note |
|---|---|---|
| H.1 baseline | done | model inventory below; runtime environment and measurements in H.8 |
| H.2 IQ3_XXS | implemented and tested | existing dot reader, generic MoE/GLU launchers, reordered fused-GEMM and conversion fallback |
| H.3 layout and consumers | implemented | stateful CPU-reference tests, canonical transfer hooks, unsupported-consumer veto |
| H.4 IQ4_XS | implemented and tested | per-expert four-plane producer, offsets, dot reader, reordered fused-GEMM and F16/F32 conversion |
| H.5 acceptance | B60 workload validated; enabled by default in H.9.3 | H.9 removes the H.8 later-prefill regression and preserves decode gains |
| H.6 softmax | rejected | five interleaved pairs at each of two KV lengths showed no repeatable gain; temporary control removed |
| H.7 deferred work | unchanged | no evidence to reopen D.2 or implement F |

The layout audit confirmed that conversion is scoped to one expert's block count. H.8 lacked reordered fused-GEMM stage readers and used conversion plus library GEMM for later prefill. H.9 adds both readers, including register and shared-memory staging. Shape or policy declines still reach the tested conversion fallback. Reordered IQ4_XS also declines the canonical dense mat-vec reader used by the host-routing fallback. Ordinary and ordered generic MoE routes and their GLU launchers consume the actual layout.

Only contiguous, non-view weights on ordinary SYCL buffers are eligible for the new MoE gates. Split buffers, ne[3] != 1 and expert strides beyond the launcher's int range are excluded. A graph with a view or another unsupported reader vetoes reorder for that weight; if a previous graph already reordered it, the graph preflight restores canonical bytes before recording and clears cached graphs. Backend get/set/copy operations return or accept canonical bytes, including partial ranges and views; these rare transfers synchronize and permute one expert slice at a time on the host.

The MoE bits are 8 for IQ3_XXS and 16 for IQ4_XS. Both are excluded from the default mask. Masks 7, 15 and 31 give the original types, original plus IQ3_XXS, and both experiments respectively. Dense IQ3_XXS continues to use the IQ3_S bit; IQ4_XS dense reorder is not enabled. Metadata allocation for the new MoE bits is scoped to expert tensors, so a dense IQ4_XS weight does not make graph preflight wait for an unsupported reorder. Startup diagnostics enumerate both new bits. Temporary debugger counters are external to the source.

### H.1 Establish a reproducible baseline before writing code

1. Read `AGENTS.md`, `CONTRIBUTING.md` and `skills/code-review/SKILL.md`. The contributor must understand the layout and dispatch choices before implementation. Preserve the existing local changes; this document was reviewed in a dirty tree.
2. Record revision, local diff, compiler, driver, GPU, CMake options and relevant environment settings. Use the existing SYCL build instructions in `docs/backend/SYCL.md`; do not assume the archive's `build` directory describes the current binary.
3. Inventory the actual model with `gguf-py/gguf/scripts/gguf_dump.py` or `GGUFReader`. Report types and bytes specifically for `ffn_*_exps.weight`; a UD filename does not determine individual tensor types. Record model availability separately from type coverage.
4. Profile a warmed baseline for decode, multi-token verification if used, and prefill. Confirm which MoE path runs and how much time the candidate type consumes. Stop if the target type is absent or does not contribute measurable time.
5. Keep unrelated experimental flags fixed. In particular keep `GGML_SYCL_DENSE_XMX_INT`, `GGML_SYCL_DENSE_XMX_SUPER`, `GGML_SYCL_MOE_XMX_INT`, `GGML_SYCL_FA_KQ_INT` and `GGML_SYCL_XMX_INT_EPI` at 0 for a reorder A/B. Keep D.2's compile-time gate at 0.

Related upstream context: [issue #27517](https://github.com/ggml-org/llama.cpp/issues/27517) documents MoE reorder coverage gaps. It is motivation, not validation of IQ3_XXS or IQ4_XS speedups. Recheck existing issues and PRs before preparing a contribution.

#### H.1.5 Model inventory, measured

`/root/models/Qwen3-30B-A3B-UD-IQ3_XXS.gguf`, arch `qwen3moe`, GGUF v3, 579 tensors, 48 layers.
Parsed straight from the GGUF tensor-info table, because `gguf-py` needs numpy and it is not
installed. Step 3's warning is confirmed: the UD filename determines nothing, the expert types are a
mix.

| expert tensors | type | share | in the MoE reorder list today? |
|---|---|---|---|
| 89 | IQ3_XXS | 62% | opt-in bit 8 |
| 42 | IQ3_S | 29% | yes |
| 13 | IQ4_XS | 9% | opt-in bit 16 |

Per layer, over 144 expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`):

| layers | composition |
|---|---|
| 25 | 3 x IQ3_XXS |
| 9 | 2 x IQ3_S + 1 x IQ4_XS |
| 7 | 3 x IQ3_S |
| 4 | 2 x IQ3_XXS + 1 x IQ4_XS |
| 3 | 1 x IQ3_S + 2 x IQ3_XXS |

Expert tensor shapes are `(2048, 768, 128)` for gate/up and `(768, 2048, 128)` for down, so
`ne[2] = 128` experts and `nb[2]` is the per-expert stride the producer scopes its planes to.

This is why IQ3_XXS is the right first target and IQ4_XS is not, **for this model**: 62% versus 9%.
Note this is Qwen3-30B-A3B, not qwen4exp, which is not present locally. Whether qwen4exp's experts
are IQ3_XXS is still unverified - see G.7.

#### H.1.6 H.2.1 contract verification, all three hold

1. `sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*(QK_K/8)` (`ggml-common.h:411`) = 2 + 96 = 98,
   and `reorder_qw_soa2_moe`'s `static_assert(qs_bytes + sizeof(ggml_half) == sizeof(block_t))`
   passes for `qs_bytes = 96`.
2. The payload is flat bytes, and the layout metadata already matches what the producer writes.
   `block_q_t<GGML_TYPE_IQ3_XXS>::get_block_offset` returns `block_index * (3*QK_K/8)` = `b*96`, and
   `get_d_offset` returns `nblocks * (3*QK_K/8) + block_index * 2` = `nb*96 + 2*b`, against a
   producer that writes `qs_ptr[ib*qs_bytes]` with
   `d_ptr = (sycl::half *)(qs_ptr + blocks_per_expert*qs_bytes)`. So `[qs: nb*96][d: nb*2]`, scoped
   per expert. No new metadata was needed.
3. **The reader already existed.** `reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS>`
   (`vecdotq.hpp:1665`) is present and takes its offsets as arguments, so it is layout-agnostic
   given the metadata above. H.2.3 therefore needed only a dispatch case, not a new dot product -
   which is the reuse-over-reinvention result H.2's table was aiming for.

#### H.1.7 Build environment gotcha, worth keeping

`build/CMakeCache.txt` has `GGML_SYCL_DNN=ON` and `DNNL_DIR=/opt/intel/oneapi/dnnl/2026.0/...`, and
`dnnl.hpp` exists at `/opt/intel/oneapi/2026.1/include/dnnl.hpp`, but a bare
`cmake --build build` fails with `'dnnl.hpp' file not found`. The oneAPI environment must be
sourced first:

```sh
source /opt/intel/oneapi/setvars.sh && cmake --build build --target test-backend-ops -j"$(nproc)"
```

A full SYCL rebuild of this target takes well over two minutes, so allow a long timeout. Also do not
set `IGC_TimeReports` or `SYCL_CACHE_PERSISTENT`, per 0.2.

### H.2 First experiment: per-expert IQ3_XXS SoA

Use these existing pieces rather than adding a kernel family:

| Piece | Current source anchor | Required change |
|---|---|---|
| Per-expert producer | `reorder_qw_soa2_moe`, `ggml-sycl.cpp` | Instantiate for `block_iq3_xxs` with `3*QK_K/8` payload bytes |
| Layout offsets | `block_q_t<GGML_TYPE_IQ3_XXS>`, `quants.hpp` | Reuse unchanged, with block count scoped to one expert |
| Dot product | `reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS>`, `vecdotq.hpp` | Reuse unchanged after verifying its activation layout |
| MoE launcher | `launch_mul_mat_vec_q_moe_reorder`, `mmvq.cpp` | Add the IQ3_XXS dispatch case |
| Type gate | `ggml_sycl_mul_mat_vec_q_id_reorder_supports_type`, `mmvq.cpp` | Add IQ3_XXS only when the launcher is implemented |
| Producer gate | `ggml_sycl_mul_mat_id_reorders_type`, `ggml-sycl.cpp` | Add an opt-in IQ3_XXS bit only after reader coverage is complete |

The layout of each expert slice is `[qs: nb*96][d: nb*2]`, where `nb = M*(K/256)`. Preserve all 96 `qs` bytes per block, including the sign/sub-scale metadata at its end. The total is `nb*sizeof(block_iq3_xxs)` (98 bytes per block). Each expert retains its existing `nb[2]` boundary; never build one plane across all experts. The base of block `b` is `slice + 96*b`, and its scale is at `slice + 96*nb + 2*b`.

Implementation order:

1. Verify the existing `reorder_qw_soa2_moe` contract for IQ3_XXS: `sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*QK_K/8`, and the payload is flat bytes. Reuse this template unchanged rather than adding a dedicated producer. It already copies the input chunk before permutation and scopes every plane to one expert.
2. Add IQ3_XXS to the MoE `slice` switch with `reorder_qw_soa2_moe<block_iq3_xxs, 3*QK_K/8>`, and add the type to `reorder_qw`'s MoE supported-type switch. Check divisibility and size arithmetic before indexing. Reuse its chunk allocation; do not call the allocating dense producer per expert. Leave `opt_for_reorder_id` disabled for this type until the complete reader set works. Set layout metadata only after the producer has successfully queued the complete permutation on the in-order queue.
3. Add the case in `ggml_sycl_mul_mat_vec_q_id_reorder`, calling `launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS>>`. This reuses both ordinary and ordered routing. Verify the launcher actually instantiates both variants for this type; do not add IQ3_XXS to the special Q8_0/IQ4_NL XMX path.
4. Audit all later uses of the same permuted weight, including a decode followed by prefill. `convert.cpp` already selects reordered IQ3_XXS dequantizers, but verify their plane count when called on an expert slice. `fg_visit_type` recognizes IQ3_XXS, yet `fg_reorder_a<block_iq3_xxs>` is currently absent, so recognition alone does not enable reordered fused GEMM. Either add a stage reader following `fg_reorder_a<block_iq3_s>`, or verify that all such calls safely use the existing reordered conversion plus library path. Declining a fused path is safe only when the fallback also understands the bytes.
5. Audit `ggml_sycl_mul_mat_vec_q_id_reorder_glu`, graph compatibility via `mul_mat_id_runs_on_device`, grouped GEMM and host-routing fallbacks. Add support where the helper contract fits; otherwise decline the fusion before reading the weights and reach a supported reordered path. Do not enable a canonical specialized IQ4_XS GLU reader for a reordered tensor.
6. Add `GGML_SYCL_REORDER_IQ3_XXS` at an unused bit. Exclude it from `GGML_SYCL_REORDER_DEFAULT`, which is currently `~0`; merely adding a bit would otherwise enable the experiment by default. The dense IQ3_XXS gate currently shares the IQ3_S bit. Preserve that behavior for the first MoE experiment and use the new bit only in the MoE gate, documenting that distinction.

Once a tensor is reordered, changing a flag cannot make its bytes canonical. Use fresh processes for A/B tests. For a test that reuses one weight buffer across operations, pass its actual layout through every dispatch, regardless of the current policy flag.

### H.3 Prove layout correctness and actual dispatch

Extend `tests/test-backend-ops.cpp`, using existing `test_mul_mat_id` and whole-graph tests. A layout experiment needs stateful coverage beyond independent matmul cases:

- Compare canonical and reordered computation against the CPU reference at the existing tolerances. Verify nonzero executed case counts and temporary launcher hits; remove instrumentation after validation. `perf` mode is not a correctness check.
- Use at least two experts with distinguishable contents and scales. Cover K=256, 512 and 768, output-row tails, repeated IDs, unused experts, and one versus multiple selected experts. Exercise token counts 1, 2, 8, 9, 64 and a realistic prefill width so dispatch boundaries are covered.
- Reuse the same weight through decode -> prefill -> decode. Include an ordinary route list and compacted active-expert routing. If GLU fusion can consume the type, check both its enabled path and its fallback.
- Compare unchunked reorder with a chunk smaller than the expert count, including the final partial chunk. Check that byte counts and expert strides are unchanged. Temporary scratch is allowed during reorder; there must be no persistent expanded weight copy.
- Audit backend copies, conversions, views and host readback used by the target graph. The current `ggml_backend_sycl_buffer_get_tensor` handles KV SoA and mask bits but does not explicitly invert weight `SOA_WHOLE`. Do not assume its canonical-bytes comment covers reordered weights. A new readback test must check actual bytes; resolve or narrowly exclude unsupported operations before widening production eligibility.
- Exercise the model's graph setting and supported multi-device placement. Split buffers and views need their own layout/stride analysis; do not broaden reorder eligibility to them just to make a test enter the new path.

Example commands after choosing the build directory and adding the new bit:

```sh
# Set these to the existing default mask with the new bit cleared/set, respectively.
GGML_SYCL_REORDER_TYPES=<baseline-mask> ./build/bin/test-backend-ops test -b SYCL0 -o MUL_MAT_ID -p 'type_a=iq3_xxs'
GGML_SYCL_REORDER_TYPES=<candidate-mask> ./build/bin/test-backend-ops test -b SYCL0 -o MUL_MAT_ID -p 'type_a=iq3_xxs'
GGML_SYCL_REORDER_TYPES=<candidate-mask> ./build/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -p 'type_a=iq3_xxs'
```

Replace placeholders with numeric masks; do not pass them literally. Run the added stateful tests separately using their actual operation descriptions. Keep the IQ3_S dense bit identical on both sides.

### H.4 Follow-up: IQ4_XS SoA, only after choosing its consumers

Freeze this layout per slice before writing a producer:

```text
nb = M*(K/256), b = row*(K/256) + kb/8, u = kb%8
qs       = slice + 128*b + 16*u
d        = half at slice + 128*nb + 2*b
scales_h = uint16 at slice + 130*nb + 2*b
scales_l = slice + 132*nb + 4*b
sc6      = ((scales_l[u/2] >> (4*(u%2))) & 15) | (((scales_h >> (2*u)) & 3) << 4)
weight[j] = float(d)*(sc6-32)*kvalues_iq4nl[code[j]]
slice bytes = 136*nb
```

All metadata fields are unchanged bit-for-bit; this is a byte permutation. With an aligned slice base, the `qs` chunks are 16-byte aligned. Check expert bases as well: `136*nb` is a multiple of 16 only when `nb` is even. Handle odd `nb` without padding or an unjustified alignment promise.

The required reader work is larger than H.2:

- `quants.hpp`: add reordered IQ4_XS offsets following the IQ3 metadata contracts. Use both entries of the offset pairs if needed for the separate scale fields; do not force a multi-field scale into one half offset.
- `vecdotq.hpp`: add `reorder_vec_dot_q_sycl<GGML_TYPE_IQ4_XS>`, preserving `vec_dot_iq4_xs_q8_1`'s nibble order, signed `(sc6-32)`, and Q8_1 scales. The reordered MoE activation is `[qs][ds]`, not an array of canonical `block_q8_1`.
- `mmvq.cpp`: wire the generic reordered MoE launcher and its gate. Add dense single/multi-column readers only if dense reorder is enabled. Keep canonical special GLU paths from reading reordered bytes.
- `convert.cpp` / `dequantize.hpp`: add reordered conversion for every enabled fallback. In `fused-gemm.cpp`, either provide `fg_reorder_a<block_iq4_xs>` or establish a supported conversion fallback. In `mmvq.cpp`, integer XMX can decline reordered IQ4_XS until `stage_soa` is implemented; it must never read it canonically.
- Producer and flag: use the same per-expert chunking as H.2; add an unused IQ4_XS bit, default-off. Keep dense and MoE eligibility separate until their respective readers pass H.3's tests, repeated with IQ4_XS.

Do not modify GGUF, add a quantization type, introduce a tiled-layout enum, or expand nibbles persistently. If the reader coverage cannot be kept small enough for the contributor to review, pause with the exact missing consumers rather than shipping a producer-only experiment.

### H.5 Performance acceptance and stop conditions

After correctness passes, run warmed, interleaved A/B pairs in fresh processes, at least five pairs per important workload. Hold device selection, clocks/power policy, model, routing, graph settings, precision and micro-batch fixed. Report medians and spread, both per-op and end-to-end. Separate one-time reorder latency and peak temporary memory from steady-state time; also record unchanged persistent weight bytes.

For the selected type, benchmark `MUL_MAT_ID` at decode and prefill shapes. Run `llama-bench` on the actual model for tg128 and pp512/pp2048, plus multi-token verification if used. A synthetic kernel win without model coverage is an experimental result, not a model speedup. Keep the new bit default-off unless repeated gains exceed measurement spread and there is no material regression in the target workloads. A null result ends that experiment; record it before considering a different layout.

### H.6 Independent attention experiment: shorter-lived softmax scores

First compare the existing `mkl_fa_softmax_chunk` two-pass kernel with `mkl_fa_softmax_chunk_1p`; much of the proposed reread strategy already exists. Select it experimentally through a default-off control at the existing orchestrator choice, using the normal four-place flag pattern if a new control is needed. Avoid duplicating the softmax implementation just to shorten `v[MKL_FA_SM_NV]`'s lifetime.

Preserve the three selection modes, additive masks, padded tails, all-masked rows, `old_max/new_max`, the prior `VKQ_accum` rescale, and F32 sum versus F16 probability behavior. Verify coverage for multiple KV chunks and GQA heads using `FLASH_ATTN_EXT` tests with the existing tolerance. Then measure the same end-to-end shapes and memory budget as E.1, without drain or diagnostic compiler flags. Inspect GRFs and spills with profiling in separate runs. Keep the variant only if total attention time improves; a register-count decrease alone is not acceptance.

### H.7 Deferred work and evidence limits

D.2 is a device-program build/runtime failure, not a demonstrated C++ compile error. Preserve the compile-time gate and retry only with a changed compiler/driver or a minimal reproducer that isolates the coordinate and mixed-type apply operations. The proposed scratch-lowering mechanism is unverified. Intel's [joint-matrix guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/joint-matrix.html) is an API reference, not proof of code generation or performance on B60.

F requires confirmation that unfused non-KDA chunking is actually hot. Start, if warranted, with a local elementwise fusion computing `exp(g_r-g_c)` directly and preserving the TRI/EXP semantics described in F.1. Do not change model graph selection, invent a new ggml op, or copy the rank-1 factors into a kernel before checking numerical range and consumer lifetimes.

SBID stalls, low XMX activity, and static instruction counts guide experiments but do not uniquely identify their cause. The transpose experiment also did not prove a fourfold instruction reduction or cross-lane shuffles without assembly comparison. Keep causal explanations labeled as hypotheses, and associate each measurement with its own type, shape, baseline and flags. CPU use of Q8_K is a useful accuracy reference, not proof that a different SYCL quantizer produces identical activations.

### H.8 Validation record (2026-10-03)

Environment: revision `d86b7ec7d883d1c935bdde5b5b7e756a69eb1ba4` plus the existing local changes and this follow-up. Release build with icpx 2026.1.1, oneAPI 2026.1, oneDNN, F16 and SYCL graphs compiled in; three Intel Arc Pro B60 cards, about 24 GiB each, driver 1.17.39395+13. GPU runs and native builds require access outside this session's filesystem sandbox; the sandbox alone does not expose `/dev/dri`. No `IGC_TimeReports` or `SYCL_CACHE_PERSISTENT` setting was used.

Correctness at existing tolerances:

| run | device / policy | result |
|---|---|---|
| IQ3_XXS candidate | SYCL0, mask 15, ordinary routing | 54/54 |
| IQ3_XXS independent bit | SYCL1, mask 8, ordered routing, graph enabled, 11 KiB chunks | 54/54 |
| IQ3_XXS baseline | SYCL2, mask 7 | 54/54 |
| IQ4_XS independent bit | SYCL0, mask 16, ordinary routing | 54/54 |
| both candidates | SYCL1, mask 31, ordered routing, graph enabled, 11 KiB chunks | 108/108 |
| both baselines | SYCL2, mask 7 | 108/108 |
| reused weights, replay and GLU | CPU | 30/30 |

These runs cover K=256/512/768, 33-row tails, five distinguishable experts, one/two selected experts, repeated IDs, unused experts and token counts 1/2/8/9/64/256. Reused-weight tests compare decode -> prefill -> decode against CPU, then verify full/partial canonical readback, copy in both directions, partial writes with untouched-byte checks, and a later noncontiguous view/dup graph. The 11 KiB limit exercises a final partial expert chunk. Graph-enabled prefill cases correctly decline recording when host-routing requires a wait.

Final checks after the metadata correction passed 110/110 candidate cases on SYCL1 with mask 31, ordered routing, graph enabled and 11 KiB chunks, 110/110 baseline cases on SYCL2, and 28/28 dense IQ3_XXS/IQ4_XS cases on SYCL0. Two added single-token reused-weight cases run the same graph three times and compare output bytes before/after replay; debug output confirms two graph warmups. The late-view checks then restore canonical bytes and invalidate those captures. CPU passed 30/30 reused-weight/replay/GLU cases. Both native SYCL targets and the CPU target build successfully; `git diff --check` is clean.

The actual model also passed `-dev SYCL0/SYCL1/SYCL2 -ts 1/1/1 -ngl 99 -fa on -ub 512 -t 8 -p 64 -n 16 -r 2` with mask 31 and graphs enabled. Verbose diagnostics after the metadata correction show three completed graph warmups and repeated reuse of graph IDs on all three devices. This is a placement/replay smoke test, not a three-card performance claim. Split buffers remain excluded from the new eligibility.

External GDB breakpoints at `ggml_sycl_mul_mat_vec_q_id_reorder`, conditional on type 18 or 23, recorded seven IQ3_XXS launcher calls with mask 15 and seven IQ4_XS calls with mask 31 for the respective single-token tests. This proves dispatch beyond a printed flag or a passing baseline. No counters remain in the source. Detailed session logs and benchmark scripts are in `/tmp/sycl-*`; they are local artifacts, not committed test fixtures.

Five clean interleaved fresh-process pairs on SYCL0, masks 7/15, Qwen3-30B-A3B UD IQ3_XXS, `-ngl 99 -fa on -ub 512 -t 8 -p 512,2048 -n 128 -r 1`, graphs disabled and all five unrelated integer-XMX flags in H.1 at zero:

| workload | mask 7 median t/s (min..max) | mask 15 median t/s (min..max) |
|---|---|---|
| tg128 | 56.78 (56.64..56.86) | 83.83 (83.75..84.16) |
| fresh pp512 | 1226.57 (1224.35..1230.35) | 1228.18 (1226.72..1229.21) |
| fresh pp2048 | 1170.16 (1168.78..1171.05) | 1170.74 (1168.46..1174.92) |

IQ3_XXS gives +47.6% steady decode throughput in this workload. Fresh prefill does not reorder the expert weights, so its flat result says nothing about prefill after decode. An initial five-pair transition run adding `-pg 512,1 -pg 2048,1` after tg128 found a substantial regression, but overlapped a subsequent build and had wide spread; the clean follow-up below is the acceptance evidence for this transition.

A clean follow-up used five interleaved fresh-process rounds in alternating order, masks 7/15/31, with the same arguments plus `-pg 512,1 -pg 2048,1`. No build or other GPU job overlapped these rounds. Each process first runs fresh prefill, then tg128, then prompt-plus-one-token workloads on the same model buffers. Medians and observed min..max, in t/s:

| workload | baseline mask 7 | IQ3_XXS mask 15 | both mask 31 |
|---|---|---|---|
| fresh pp512 | 1228.89 (1227.60..1230.52) | 1226.52 (1224.49..1228.88) | 1226.38 (1225.76..1228.97) |
| fresh pp2048 | 1170.57 (1167.10..1173.64) | 1170.20 (1167.93..1173.47) | 1171.72 (1169.16..1173.44) |
| tg128 | 56.49 (56.31..57.01) | 84.00 (83.71..84.09) | 85.58 (85.45..85.84) |
| pp512+tg1 after decode | 1181.19 (1178.61..1182.05) | 972.03 (963.23..979.09) | 909.89 (898.09..919.85) |
| pp2048+tg1 after decode | 1154.26 (1150.10..1155.20) | 966.30 (955.87..972.71) | 906.84 (899.63..908.60) |

IQ3_XXS increases decode throughput by 48.7%; IQ4_XS adds 1.9% over that candidate. Later prefill regresses 16.3-17.7% with IQ3_XXS and 21.4-23.0% with both types. These are workload tradeoffs, not general acceptance for either layout. The metadata-allocation correction made after these rounds changes graph eligibility only; graphs were disabled in all timing comparisons.

Per-op measurements used five interleaved rounds in fresh processes on SYCL2, masks 7/15/31, all five unrelated integer-XMX flags zero and graphs disabled. Each process measured four cases, with positive iteration counts: `MUL_MAT_ID`, 128 experts, eight selected, M=768, K=2048, F32 activation, n=1/64. Median us/run and min..max:

| type / tokens | mask 7 | mask 15 | mask 31 |
|---|---|---|---|
| IQ3_XXS, n=1 | 67.89 (67.85..67.91) | 26.00 (25.60..26.20) | 25.94 (25.62..26.41) |
| IQ3_XXS, n=64 | 2305.91 (2129.98..4395.24) | 2305.21 (2221.22..2318.14) | 2311.96 (2092.72..4374.24) |
| IQ4_XS, n=1 | 37.19 (36.83..37.49) | 37.23 (37.20..37.45) | 30.76 (30.62..30.87) |
| IQ4_XS, n=64 | 2372.24 (2172.07..4427.29) | 2374.73 (2362.63..2396.24) | 2369.75 (2128.83..4442.67) |

The decode medians improve about 2.6x for IQ3_XXS and 1.2x for IQ4_XS. Each n=64 case starts with a fresh weight buffer and takes the canonical prefill path even with the candidate bit enabled. Its medians are flat and its wide outliers preclude a prefill improvement claim. The stateful model measurements above cover prefill with weights already permuted. `/tmp/sycl-moe-accept-perf-*.log` contains all 60 completed per-op measurements; the earlier broad exploratory run is excluded.

H.6: the current source already chooses the existing one-pass softmax automatically for these shapes. A temporary default-off control forced the existing two-pass kernel. Correctness passed 16/16 q8_0 K/V, head-size-256 cases. Five interleaved pairs for `hsk=256,hsv=256,nh=2,nr23=[12,1],nb=1024` measured median FLASH_ATTN_EXT times of 1958.34 us (one-pass) versus 1954.59 us (two-pass) at KV=2048, and 8567.85 versus 8573.85 us at KV=13312. The differences are within spread; the control was removed. No register-pressure claim or additional softmax kernel is justified by this result.

The producer keeps exactly 98 bytes per IQ3_XXS block or 136 per IQ4_XS block. There is no persistent expanded weight copy. With the existing chunk policy, temporary device reorder storage is `max(one expert slice, floor(chunk_bytes/expert_bytes)*expert_bytes)`, capped at the whole weight; the default budget is 32 MiB. Canonical transfer uses two host slices, and unsupported-consumer restoration needs one additional host slice. These are derived storage bounds, not a measured device-wide memory census. First-use reorder/JIT latency was not isolated, so the throughput tables describe warmed steady-state behavior only. These limits and the later-prefill regression rule out changing either default bit.

### H.9 Complete reordered fused-GEMM readers (2026-10-04)

The H.8 measurements are the before-readers baseline. Add `fg_reorder_a<block_iq3_xxs>` and `fg_reorder_a<block_iq4_xs>` using the existing per-expert planes, expose their types through `ggml_sycl_fused_dequant_gemm_reorder_ok`, and describe their streams in `fg_soa_layout` for cooperative shared-memory staging. Reuse canonical decoding so grid/sign interpretation and signed IQ4_XS sub-scales remain identical. Support ordinary staging, whole-block registers, metadata-only registers and the existing shared-memory/pipeline variants.

Validate repeated decode -> prefill -> decode against CPU at unchanged tolerances, including odd block counts, K=256/512/768, row tails, compacted routing, 64/256-token transitions, graph replay and canonical readback. Prove that prefill actually launches fused GEMM with reordered weights. Check the canonical paths after sharing the decode helpers, and force conversion fallback separately to retain its coverage. Use all three B60 cards for correctness and model placement checks.

Then rerun at least five interleaved fresh-process rounds with masks 7/15/31 on the same model and flags as H.8, including prefill after decode. Keep both experiments default-off while validating. Record medians and spread, and whether the later-prefill regression is removed. The aligned tiled layout in G.2.1 is the next experiment after this reader completion, not a prerequisite for it.

#### H.9.1 Reader correctness and dispatch

Native SYCL `test-backend-ops` and `llama-bench` targets build successfully. The patch adds two reordered reader specializations, their shared-memory stream descriptions and the fused-GEMM layout gate; canonical and reordered staging share their decoding helpers. The half scale, grid/sign bytes and signed IQ4_XS six-bit scales keep their original interpretation. No producer or persistent weight bytes change.

The updated CPU reused-weight/replay/GLU set passes 32/32. On the B60s, the full candidate set passes 112/112 with mask 31, ordered routing, graph enabled and 11 KiB chunks; the canonical mask-7 set passes 112/112. Dense cases pass 28/28. Disabling all fused-GEMM gather types explicitly exercises the conversion fallback, which passes 16/16 reused-weight cases.

Existing `GGML_SYCL_GG_TRACE` output shows both IQ3_XXS and IQ4_XS reaching host-scheduled grouped GEMM with `reordered=1`, K=256/768, M=33, 128 routed rows and two active experts. With `GGML_SYCL_MMID_SCHED=9` (device schedule plus host/device output comparison), 8/8 transition cases pass and `[GG-DEV]` reports `reordered=1` for both types. There is no added instrumentation in the source.

The actual model passes the three-device command from H.8 plus `-pg 64,16`, so the smoke run includes a later prompt and another decode on the same weight buffers. With mask 31 and graphs enabled, diagnostics record nine completed warmups and repeated graph reuse. This is runtime/placement coverage; numerical correctness is checked by the CPU-reference tests, not by `llama-bench` output.

All logs use the `/tmp/sycl-reordered-gemm-*` prefix. The earlier H.8 measurements remain the archived before-reader results.

Eight staging configurations each pass 8/8 transition cases: MMID_SCHED=0 (ordinary), 512 (shared memory), 1024 (pipeline), 2048 (whole-block registers), 4096 (metadata registers), 32768 (block-shaped A), 33280 (shared memory plus block-shaped A), and 768 (64-row tile plus shared-memory selection). The wide-tile process required several minutes of first-use driver compilation; these runs are correctness checks, not timing data.

Two additional cases use the model's actual M=768, K=2048 geometry through decode -> 64-token prefill -> decode, followed by canonical transfer/view checks. The final full candidate and baseline sets pass 114/114 each, and the CPU reused-weight/replay/GLU set passes 34/34. The model-shaped pair separately passes 2/2 in each of shared-memory, whole-block-register and metadata-register staging. The final forced conversion-fallback set, including the model-shaped pair, passes 18/18. `git diff --check`, LF/final-newline and trailing-whitespace checks pass. The optional editorconfig-checker binary is unavailable locally; its applicable formatting rules were checked directly.

Adding the fused-GEMM layout gate also lets the existing host/device schedulers invoke `opt_for_reorder_id` during prefill. Thus H.8's statement that fresh prefill retains canonical weights applies to the before-reader implementation only. With the completed readers, a warmed fresh-process prefill benchmark can already use reordered weights, and the timed runs exclude first-use reorder/JIT costs.

#### H.9.2 Completed-reader performance

Five clean fresh-process rounds alternate masks 7/15/31 and 31/15/7. Device SYCL0, the H.1.5 model, `-ngl 99 -fa on -ub 512 -t 8 -p 512,2048 -n 128 -pg 512,1 -pg 2048,1 -r 1`, graphs disabled and the same unrelated integer-XMX flags at zero as H.8. No build or other GPU job overlaps these runs. Every process exits successfully and reports all five workloads. Median t/s with observed min..max:

| workload | baseline mask 7 | IQ3_XXS mask 15 | both mask 31 |
|---|---|---|---|
| initial pp512 | 1228.48 (1227.28..1229.68) | 1379.42 (1376.66..1382.60) | 1373.90 (1372.92..1375.95) |
| initial pp2048 | 1172.90 (1171.97..1174.27) | 1309.85 (1307.79..1311.68) | 1304.08 (1301.35..1308.19) |
| tg128 | 56.66 (56.59..56.78) | 83.82 (83.77..84.22) | 85.56 (85.25..85.60) |
| pp512+tg1 after decode | 1179.57 (1175.84..1182.53) | 1340.90 (1334.59..1341.86) | 1334.60 (1333.32..1341.10) |
| pp2048+tg1 after decode | 1154.05 (1150.27..1155.18) | 1294.54 (1293.55..1295.88) | 1291.14 (1285.54..1294.48) |

The completed IQ3_XXS reader gives about +48% decode throughput, +12% initial prefill and +12-14% later prefill relative to the current baseline. With both formats, decode improves about +51%, initial prefill +11-12% and later prefill +12-13%. The H.8 prefill-after-decode regression is removed for this workload. Relative to IQ3_XXS alone, IQ4_XS adds about 2.1% decode throughput and lowers prefill throughput by about 0.3-0.5%; both candidates remain above baseline on every measured workload.

For comparison with the archived pre-reader mask-31 results, later pp512+tg1 was 909.89 t/s and is now 1334.60; later pp2048+tg1 was 906.84 and is now 1291.14. Decode was 85.58 and is now 85.56. These before/after observations come from separate sessions; the interleaved current baseline above is the acceptance comparison. Initial prefill now uses reordered weights as described in H.9.1, so it is no longer the canonical-only measurement from H.8.

The readers are complete and validated on this B60 workload. Both MoE bits remain experimental/default-off; use mask 15 for IQ3_XXS or 31 for both. First-use reorder latency and device-wide peak memory were not isolated, and the timings describe warmed operation. The next layout experiment is G.2.1; it should compare against this completed-reader baseline rather than the conversion fallback in H.8.

#### H.9.3 Default enablement

At the contributor's request, IQ3_XXS and IQ4_XS MoE reorder are enabled by default after the H.9 correctness and performance checks. `GGML_SYCL_REORDER_DEFAULT` is now -1, retaining the all-types policy. Set `GGML_SYCL_REORDER_TYPES=7` to disable both or 15 to enable IQ3_XXS alone. Earlier default-off statements describe the experimental validation period. No integer-XMX defaults change.

The native targets rebuild successfully with the new default. With `GGML_SYCL_REORDER_TYPES` unset, the full B60 candidate suite passes 114/114, and diagnostics report `0xffffffff` with both new bits set. The three-card model smoke passes initial pp64, tg16 and later pp64+tg16 with two repetitions and graphs enabled. Logs are `/tmp/sycl-reorder-default-build.log`, `/tmp/sycl-reorder-default-candidate.log` and `/tmp/sycl-reorder-default-three-gpu.{out,err}`.
