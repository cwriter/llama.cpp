# SYCL: IQ-quant support in the fused MoE `mul_mat_id` path

Branch: `iq_type_fusion` (base `49c0dc82b`)
Diff: **2 files, +94 / −0** — `ggml/src/ggml-sycl/mmvq.cpp`, `ggml/src/ggml-sycl/vecdotq.hpp`
**Purely additive: no pre-existing line is modified.**

---

## 1. The problem

`ggml_sycl_mul_mat_id()` (`ggml-sycl.cpp`) takes a fused fast path at batch size 1:

```c
if (ne12 == 1) {
    if (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
        return;                      // fused MoE GEMV, fully on-device
    }
}
// ---- otherwise: fallback ----
std::vector<char> ids_host(ggml_nbytes(ids));
stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids));
SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));   // FULL GPU PIPELINE STALL
mmid_counting_sort_rows(...);                  // sort on the CPU
stream->memcpy(dev_row_mapping.get(), routed_row_src.data(), ...);
```

The fused path ends in `ggml_sycl_mul_mat_vec_q_id()` (`mmvq.cpp`), a `switch (src0_type)`
dispatch table. Before this change it handled
`Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_0 Q2_K Q3_K Q4_K Q5_K Q6_K MXFP4 NVFP4`
and `default: return false;` — **no IQ type at all**.

So for any IQ-quantised MoE model the fused path declined, and every expert matmul fell
back to *host round-trip + full queue drain + CPU counting sort*, **once per MoE op per
token**. Prefill hides this (the stall is amortised over a whole ubatch); decode pays it in
full on every token.

The kernels were never the problem: all nine `vec_dot_iq*_q8_1` functions already exist in
`vecdotq.hpp` and are already dispatched by the **non-MoE** path
(`ggml_sycl_op_mul_mat_vec_q`, same file). Only the MoE dispatch table was missing them.

## 2. The change

### 2a. Five 3-argument adapters — `mmvq.cpp` +2674..2701 (28 lines)

The MoE kernel template is parameterised by `vec_dot_q_sycl_t`, a **3-argument** function
pointer type. Most `vec_dot_iq*_q8_1` already match it, but five take an extra pointer to a
codebook grid:

```c
vec_dot_iq3_s_q8_1(const void *, const block_q8_1 *, const int &, const uint32_t *iq3s_grid)
```

Each adapter is a one-line forward that binds that global:

```c
static __dpct_inline__ float vec_dot_iq3_s_q8_1_moe(const void * __restrict__ vbq,
                                                    const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}
```

Added: `vec_dot_iq2_xxs_q8_1_moe`, `vec_dot_iq2_xs_q8_1_moe`, `vec_dot_iq3_xxs_q8_1_moe`,
`vec_dot_iq3_s_q8_1_moe`, `vec_dot_iq1_s_q8_1_moe`.

### 2b. Nine dispatch cases — `mmvq.cpp` (45 lines)

Inserted between the existing `case GGML_TYPE_NVFP4:` and `default:`, in the same shape as
every neighbouring case. Template parameters are copied verbatim from the corresponding
non-MoE launcher, so the arithmetic is identical by construction:

| `GGML_TYPE` | needs adapter | `launch_mul_mat_vec_q_moe<qk, qi, block_q_t, vdr, vec_dot>` |
|---|---|---|
| `IQ2_XXS` | yes | `QK_K, QI2_XXS/2, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe` |
| `IQ2_XS`  | yes | `QK_K, QI2_XS/2, block_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe` |
| `IQ2_S`   | no  | `QK_K, QI2_S/2, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1` |
| `IQ3_XXS` | yes | `QK_K, QI3_XXS/2, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe` |
| **`IQ3_S`** | yes | `QK_K, QI3_S/2, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_moe` |
| `IQ1_S`   | yes | `QK_K, QI1_S, block_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe` |
| `IQ1_M`   | no  | `QK_K, QI1_S, block_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1` |
| **`IQ4_NL`** | no | `QK4_NL, QI4_NL, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1` |
| **`IQ4_XS`** | no | `QK_K, QI4_XS/4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1` |

### 2c. Nine `VDR_IQ*_Q8_1_MMVQ` constants — `vecdotq.hpp` (+21 lines)

The `vdr` template argument was initially passed as a bare literal. The SYCL backend already
defines `VDR_*_Q8_1_MMVQ` for all 14 K/legacy/FP4 types, each placed immediately above its own
`vec_dot_*` function, so the nine IQ constants now follow that same convention and placement.

**These deliberately differ from the identically named CUDA constants and must never be
cross-copied.** `qk` and `qi` are format properties from the shared `ggml-common.h` and are
architecture-independent; `vdr` is not — it is how many `iqs` units one `vec_dot` call consumes,
and the two backends take `iqs` in different units:

```c
// SYCL vec_dot_iq3_s_q8_1:   const int ib32 = iqs;              // iqs IS the 32-weight sub-block index
// CUDA vec_dot_iq3_s_q8_1:   bq3->qh[iqs/2]; bq3->scales[iqs/4] // iqs is an int32 WORD index
```

CUDA's `iqs` is 2x SYCL's for the same position (4x for IQ4_XS), which is exactly the ratio
between the two backends' constants (CUDA IQ3_S=2 vs SYCL 1; CUDA IQ4_XS=4 vs SYCL 1). The
invariant across backends is `qi/vdr` — the number of lanes cooperating on one block — which is
`QI3_S/2` in both; SYCL reaches it as `(QI3_S/2)/1`, CUDA as `QI3_S/2`. A three-line comment in
`vecdotq.hpp` records this.

Values, each taken from the corresponding pre-existing SYCL non-MoE launcher (never from CUDA):
`IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/IQ4_XS = 1`, `IQ4_NL = 2`.

**Scope note:** the constants are used only in the nine cases this patch adds. The ten
pre-existing non-MoE call sites still pass bare literals. Converting them was implemented and
then deliberately reverted: it modified 10 lines of working code unrelated to this feature,
turning a patch that *cannot* regress anything into one that could, for purely cosmetic gain.
That conversion belongs in a separate cleanup commit.

Bold = the types that dominate the target model (`Qwen3.8-Flash-Next UD-IQ4_XS`: experts are
`IQ3_S ×94, IQ4_NL ×43, Q8_0 ×5, IQ4_XS ×2`; per layer `ffn_down` is IQ4_NL while
`ffn_gate`/`ffn_up` are IQ3_S, so IQ3_S alone accounts for two thirds of expert matmuls).

## 3. Why this is the least invasive change that still gets the full win

- **Nothing is modified, only added.** `git diff --stat` is `+94 / −0`. No existing line
  changes, so no existing behaviour can regress. This property was consciously protected: the
  constants work (2c) was trimmed back specifically to preserve it.
- **Two files, both additively.** `mul_mat_vec_q_moe`, `launch_mul_mat_vec_q_moe`,
  `vec_dot_q_sycl_t`, `ggml-common.h` and `ggml-sycl.cpp` are all untouched; `vecdotq.hpp`
  gains only nine `#define`s and a comment.
- **No shared template signature was changed.** The obvious alternative — adding a table
  pointer argument to `mul_mat_vec_q_moe` / `launch_mul_mat_vec_q_moe` — would have edited a
  template shared with existing callers and forced every current case to be rewritten, for
  zero benefit: the grids are `GGML_TABLE_BEGIN` global constants in `ggml-common.h`, already
  visible to device code, not runtime buffers that need plumbing.
- **The adapter pattern is the one already in the file.** The non-MoE kernel does exactly
  this, one level lower: `mul_mat_vec_q_iq3_s_q8_1` names `iq3s_grid` directly at the call
  site inside the kernel body. Two `vec_dot` functions (`iq2_s`, `iq1_m`) already bind their
  grids internally. The adapters just apply the same idea at the dispatch layer.
- **Correctness is by construction, not by hope.** Same `vec_dot`, same `qk`/`qi`/`vdr` as
  the non-MoE path. For all nine types `qi/vdr <= WARP_SIZE`, so the MoE kernel's inner loop
  runs once and yields `iqs = vdr * (lane % (qi/vdr))` — identical indexing to the non-MoE
  kernel — and `vdr*WARP_SIZE` is divisible by `qi` in every case, so `blocks_per_warp`
  agrees. The reduction is the same XOR butterfly, so even summation order matches.
- **Scope was capped deliberately.** The four extra IQ types beyond those the target model
  needs cost ~5 lines each and fall out of the identical mechanism; adding them avoids
  leaving an arbitrary hole in the dispatch table. Nothing was contorted to fit them. If a
  strictly minimal patch is wanted, deleting the six non-target cases and four of the five
  adapters is self-contained and breaks nothing.

## 4. Verification

**Correctness** — `test-backend-ops -o MUL_MAT_ID -b SYCL0`: **7/7 backends passed**, 106 IQ
cases `OK`, re-run after every change including the final revert, with every IQ type `OK` (`iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs`) across
all tested shapes. The only `not supported` line is `tq2_0`, unrelated and pre-existing.
Generated text is coherent and correct on a factual reasoning prompt.

**Performance** — 3× Intel Arc Pro B60 (SYCL), `Qwen3.8-Flash-Next UD-IQ4_XS`,
`-ngl 99 -ncmoe 0 -ts 31,33,36 -c 131072 -np 1 -b 4096 -ub 1024 -fa on -ctk/-ctv q8_0`,
n-gram table host-resident via `-ot`. Identical harness both arms, first request discarded
(warm-up artefact), same binaries except the patch:

| | control (`49c0dc82b`) | patched (final) | change |
|---|---|---|---|
| decode, short prompt | 12.76 / 13.81 / 13.32 → **13.30** | 18.08 / 17.94 / 18.01 → **18.01** | **+35.4 %** |
| decode, 18 911-tok prompt | **10.14** | **13.44** | **+32.5 %** |
| prefill, 18 911-tok prompt | 222.6 | 232.6 | +4.5 % (noise) |
| graph splits @ bs=1 | 28 | 28 | unchanged |

Three independent patched builds, all in agreement (short mean / long tg):
18.23 / 13.19, 17.78 / 13.44, 18.01 / 13.44. A separate confirmation session gave
18.26 / 18.19 / 18.09 / 17.89. Control spread was 12.76-13.81, so the gap is far outside noise.

Prefill being flat is the expected signature and confirms the mechanism: only the `ne12 == 1`
fused path changed, and prefill runs the batch>1 path where the stall was already amortised.
Graph splits are unchanged because the remaining 28 are the host-side per-layer-embedding
(`per_layer_token_embd`) gather, which this patch does not touch.


## 4b. Quality and performance validation (llama-bench / llama-perplexity)

Hardware: 3x Intel Arc Pro B60 (SYCL). Model: `Qwen3.8-Flash-Next UD-IQ4_XS`.
Flags both arms: `-ngl 99 -dev SYCL0/SYCL1/SYCL2 -sm layer -ts 31/33/36 -ncmoe 0`
`-ot per_layer_token_embd\.weight=CPU -lzm auto -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 -t 16 --poll 0`.
Only `libggml-sycl.so` differs between arms (md5-verified distinct; the loaded path was checked
in `/proc/<pid>/maps`). `master` = `49c0dc82b`.

**On temperature:** neither tool samples. `llama-bench` measures raw throughput and
`llama-perplexity` computes a deterministic likelihood; there is no temperature parameter for
either, so the runs are already deterministic in that sense and nothing needed setting to 0.

### Perplexity — wikitext-2 test, `-c 512 --chunks 200`

| | master | this PR | Change |
|---|---|---|---|
| PPL | **4.2993 ± 0.04376** | **4.2957 ± 0.04369** | **−0.0036 (−0.08 %)** |

The difference is ~12x smaller than the reported standard error and the intervals overlap almost
entirely. Better still, master was measured twice (cold and warm page cache) and gave
**4.2920** and **4.2993** — a run-to-run spread of **0.0073, twice the master-vs-PR difference**.
The patch's effect on output is below the baseline's own noise floor, which is the expected
signature of identical arithmetic with a different floating-point summation order.

### Throughput — `llama-bench`

| test | master | this PR | Change |
|---|---|---|---|
| pp512 | 206.56 ± 3.54 | 248.77 ± 15.18 | +20.4 % |
| pp2048 | 251.27 ± 5.10 | 353.76 ± 4.44 | +40.8 % |
| pp131072 | 152.92 | 153.12 | **+0.1 %** |
| tg512 | 11.17 ± 0.46 | **17.92 ± 0.08** | **+60.4 %** |
| tg2048 | 11.59 ± 1.34 | **17.24 ± 0.02** | **+48.7 %** |
| tg128 @ d131072 | 4.76 | 5.27 | +10.7 % |

**Reading these honestly:**

- **The decode numbers are the result.** `tg512` +60 % and `tg2048` +49 % are consistent with the
  independent end-to-end server measurement (13.30 -> 18.01 t/s, +35 %, three separate builds) and
  with the mechanism: the fused path engages at `ne12 == 1`.
- **`pp131072` shows +0.1 %, and that is the expected and correct result.** The patch only affects
  the batch-1 fused path, so prefill must be unchanged. This is the longest, steadiest measurement
  in the table and it confirms the mechanism rather than contradicting it.
- **The pp512 / pp2048 deltas are NOT claimed as a win.** They are not explained by the mechanism,
  master's own spread across two runs was 190.65 -> 206.56 and 240.24 -> 251.27, and the PR's
  pp512 carries a +/-15.18 error bar. Treat them as measurement noise on short runs.
- `tg128 @ d131072` is generation speed *at* 131072 context, not generation *of* 131072 tokens
  (the latter would take ~2 h per arm).

**Methodology note worth repeating:** the first master arm ran on a cold page cache and produced
*inflated* deltas (pp131072 appeared +13 %, later shown to be 0 %). With a 26.82 GiB host-resident
lazily-read PLE table, cache state dominates short benchmarks. Both arms in the tables above were
measured warm, and the discarded cold run is kept for reference.

## 5. Known limitations / notes for review

- The six IQ types the target model does not use (`IQ2_*`, `IQ1_*`, `IQ3_XXS`) pass
  `test-backend-ops` but have not been exercised by a real model on this hardware.
- `ggml_sycl_mul_mat_id_mmvq_fused()` checks `ne10 % QK8_1 == 0` but not `% QK_K`, whereas
  the non-MoE launchers carry an explicit `GGML_ASSERT(ncols % QK_K == 0)`. This is
  pre-existing and equally true of the existing `Q*_K` cases — a QK_K-block src0 cannot exist
  with `ne00 % 256 != 0`, and `ne10 == ne00` is checked — but it is an asymmetry worth noting.
- The `_moe` suffix on the adapters is a placement hint (they are used only by the MoE
  dispatch), not a semantic claim.
- Not benchmarked: whether the reorder/SoA path (`opt_for_reorder_id`, currently
  `Q4_K/Q5_K/Q6_K` only) would give a further gain if extended to IQ types. That is a
  separate, larger change and was deliberately left out of scope.
