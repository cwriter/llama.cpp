# SYCL: int8 XMX plan

Status: plan only. Nothing here is implemented, built or benchmarked. Line numbers refer to
`performance_uplift_mega_branch` at `fbb9699`.

Goal: use the int8 XMX path (s8 x s8 -> s32, 8x16x32 per sub-group op) where it beats the
current f16 XMX path, and keep data in the integer domain where the math allows it.

## 1. Where XMX is used today

| Path | File | Precision | Notes |
|---|---|---|---|
| Fused dequant GEMM, dense, N <= 64 | `fused-gemm.cpp`, called from `ggml-sycl.cpp:3620` | f16 x f16 -> f32, 8x16x16 | weights decoded to f16 in registers, activations converted f32 -> f16 first |
| Grouped MoE GEMM | `fused-gemm.cpp`, called from `ggml-sycl.cpp:6332`, `:6550` | f16 | same A stage, host or device schedule |
| Dense N > 64 (normal prefill) | `ggml-sycl.cpp:3622-3660` | f16 | `to_fp16_sycl` writes the whole weight slice, then oneDNN / oneMKL |
| Ordered MoE mat-vec | `mmvq.cpp:2969` `mul_mat_vec_moe_ordered_xmx` | **s8 x s8 -> s32** | Q8_0 and IQ4_NL only, gated on `average_routes >= 8` (`mmvq.cpp:3562`) |
| QSA indexer score GEMM | `qsa-score.cpp:265-279` | f32 x f32 (oneDNN f16 fpmath when `GGML_SYCL_F16`) | result only feeds a top-k ranking |
| Flash attention, oneDNN SDPA / oneMKL | `fattn-onednn.cpp`, `fattn-mkl.cpp` | f16 | quantized KV is converted to f16 first |

## 2. Int8 GEMM work items

### 2.1 Make the existing int8 MoE kernel efficient (`mmvq.cpp:2969-3085`)

Problems seen in the code:
- One 8x16x32 `joint_matrix_mad` per 32-value block, then `joint_matrix_store` to SLM, two
  sub-group barriers and a scalar FMA loop to apply `scale_a * scale_b`. XMX is idle most of the time.
- A and B tiles are staged one byte at a time through `traits::quant`.
- One accumulator tile per sub-group.

Plan:
1. Keep an f32 accumulator in registers. After each block, convert the s32 tile and FMA it into
   the f32 tile with `joint_matrix_apply` (row/col coordinates give the two scales). Drop the
   SLM store and both barriers.
2. Compute MT x NT tiles per sub-group, as `fused-gemm.cpp` does (`FG_SG_ROWS`, 2 N tiles).
3. Reuse the wide-load staging from `fused-gemm.cpp` (`GGML_SYCL_WIDE_LOADS`).
4. Re-tune the `average_routes >= 8` gate after 1-3.

Check: `test-backend-ops -o MUL_MAT_ID -b SYCL0`, then `llama-bench -p 512` on a Q8_0 and an
IQ4_NL MoE model, `GGML_SYCL_MOE_XMX=0` vs `1`.

### 2.2 Int8 variant of `fused-gemm.cpp`

Replace the f16 A/B tiles with s8 tiles and 8x16x32 MADs for formats whose decoded values are
integers:

| Format | Int8 A value | Extra work |
|---|---|---|
| Q8_0 | `qs` as is | none |
| IQ4_NL, IQ4_XS | `kvalues_iq4nl[nibble]` (int8 table) | IQ4_XS: 6-bit sub-scale per 32 |
| IQ2_*, IQ3_* | grid byte x sign | per-32 or per-16 sub-scale |
| Q4_K, Q5_K | unsigned 0..15 / 0..31 | min term: `dmin * m * sum(a)`, needs activation block sums (q8_1 `ds[1]`) |
| Q6_K | -32..31 | sub-blocks of 16, one MAD is 32 deep: needs two masked MADs or a split |
| IQ1_S, IQ1_M | not integer (delta +-0.125) | skip |

B (activations): quantize with `quantize_q8_1` instead of converting to f16. This matches the
accuracy of the decode path, which already uses q8_1 activations.

Side effect: Q4_K no longer needs `GGML_SYCL_FAST_AND_SLOPPY` (`common.hpp:134`). The f16 overflow
that flag guards against cannot happen with int8 values and f32 scales.

Expected gain: int8 MAD covers K=32 per op vs K=16 for f16; the A tile halves in SLM and GRF.
Risk: the per-block rescale epilogue (see 2.3) may eat the gain for 32-value-block formats.
Start with Q8_0 and IQ4_NL and compare against the f16 kernel at the same tile shape.

### 2.3 K-quants: rescale once per 256, not once per 32

K-quant sub-scales are small integers. If activations use one scale per 256 values (Q8_K style,
as the CPU backend does for K-quants) instead of one per 32:

    acc_s32  = sum_j sc_j * (q_j . a_j)          // integer, 8 sub-blocks of 32
    acc_s32 -= m_j * bsum_j terms                 // integer, Q8_K bsums
    acc_f32 += d_w * d_a * acc_s32                // one float epilogue per superblock

This cuts the float epilogue 8x for Q4_K / Q5_K and makes them the strongest int8 candidates.
Needs a Q8_K-style activation quantizer on SYCL (per-256 scale plus per-16 sums).

### 2.4 Dense prefill, N > 64

Still dequantizes the full weight slice to f16 in memory each call. Options:
- (a) Extend the int8 fused GEMM (2.2 / 2.3) past `GGML_SYCL_FG_MAX_N`, with bigger N tiles.
- (b) oneDNN int8 matmul with grouped scales (group 32). Confirm the installed oneDNN supports
  grouped scales on both src and weights on Intel GPU before investing.

Likely the largest remaining prefill cost on dense models.

## 3. Staying in the integer domain beyond GEMM

### 3.1 QSA indexer: int8 end to end (best candidate)

`qsa-score.cpp` computes `score[b,t] = sum_h relu(dot(pooled[:,b], q[:,h,t]))` with an f32 GEMM,
and the result only feeds a top-k. Only the ranking matters.

- Quantize `q[:, :, t]` with one positive scale `s_t` per token (shared over all heads), and
  `pooled` with one scale `s_p` for the whole tensor (or per head-dim group shared over b).
- Then `dot = s_t * s_p * I[b,h,t]` with `I` from int8 XMX. Since `s > 0`,
  `relu(s * I) = s * relu(I)`, so `score[b,t] = s_t * s_p * sum_h relu(I[b,h,t])`.
- The head sum stays in int32. For a fixed t the factor `s_t * s_p` is constant over b, so the
  top-k over b can run on the int32 sums directly: `topk-radix` already ranks by an
  order-preserving unsigned key, and for non-negative int32 the key is the value itself.
- No float math at all between the GEMM and the top-k.
- If `pooled` needs a per-block scale for accuracy, one float multiply per (b,t) after the
  int32 head sum is still cheap.

Accuracy check: compare the selected top-k sets against the f32 path (set overlap), not the
raw scores. Same idea applies to `lightning-indexer.cpp` if its per-head weights are
non-negative; a negative head weight breaks `relu(w*x) = w*relu(x)`, so it then needs a
per-head float multiply before the head sum.

### 3.2 Attention with int8 Q.K^T: integer max and tile skipping

With a Q8_0 K cache (`kv-soa.hpp` layout), Q.K^T can run in int8 XMX if Q is quantized in
32-value blocks along head_dim. Each score is then `sum_blk sQ_blk * sK_blk * I_blk`: float,
because the scales differ per block. To keep the max in int32:

- Quantize Q per (token, head) with one scale and K per (token, head) with one scale (a
  different K layout than Q8_0, or a second scale stream). Then `score_j = sQ * sK_j * I_j`.
- `sK_j` still varies per key, so the exact row max needs one float multiply per element. No
  saving over the current float max.

Where integers do help:
- **Tile skipping.** Per KV tile, an upper bound `sQ * max_j(sK_j) * max_j(I_j)` is one int32
  max reduction plus one float multiply per tile. If the bound is below
  `running_max - T` (for example `T = 20` in log2 units, contribution `< 2^-20`), skip the
  exp, the row sum and the P.V MAD for the whole tile. This changes results by a bounded
  amount, so put it behind an opt-in flag like `GGML_SYCL_FAST_AND_SLOPPY`.
- **Masks.** Already one bit per cell (`kq-mask-bits.hpp`, `qsa-mask.cpp`). Extra step: if a
  tile's mask words are all zero, skip the tile before the Q.K^T MAD; if all ones, skip
  reading the mask. Check whether `fattn-tile.hpp` and `fattn-mkl.cpp` already do this
  before starting.

### 3.3 Not worth it

- **Cascaded max (exponent first, then mantissa).** For IEEE floats, comparing the bit pattern
  as an integer (sign-flipped for negatives) already gives the exact float order; that is what
  `topk-radix.hpp` does. On Xe, float max and int max are both one instruction, so a two-stage
  compare saves nothing in registers. It only pays when it cuts memory traffic (radix select)
  or when it gives a cheap bound (3.2).
- **Exponent-only max as the softmax stabilizer.** `m = 2^(e_max+1)` can overshoot the true
  max by up to its own size (max 100 -> m 128, exp(-28) ~ 7e-13). Fine in f32, but underflows
  if P is stored in f16. Not safe as a default.
- **Integer exp.** Xe has a hardware exp2; a shift + LUT integer exp is unlikely to be faster.
- **Int8 P.V.** P in [0,1] quantized to u8 loses accuracy on peaky rows; published int8
  attention work (SageAttention) keeps P.V in f16 or fp8. Q8_0 V scales also run along
  head_dim, not along the reduction axis, so each 32-wide output slice needs its own quantized
  copy of P. Do 2.x and 3.1 first.
- **Single-token decode.** Memory-bound mat-vec; MMVQ / ESIMD stay the right path.

## 4. Suggested order

1. 2.1 (small, contained; proves the in-register rescale with `joint_matrix_apply`).
2. 3.1 (contained in one fusion; ranking-only, easy to validate by top-k overlap).
3. 2.2 for Q8_0 and IQ4_NL.
4. 2.3 with a Q8_K-style activation quantizer, then 2.4.
5. 3.2 tile skipping, opt-in.

For each step: `test-backend-ops -b SYCL0` for the touched ops, `llama-perplexity` on a short
text vs the f16 path, and `llama-bench -p 512 -n 128` with the new path on and off by env var.
