# SYCL: int8 XMX implementation plan

Status: plan only. None of this code has been compiled or run. Line numbers refer to
`performance_uplift_mega_branch` at `fbb9699`. Search for the quoted code if lines moved.

Goal: use the int8 XMX path (s8 x s8 -> s32, one 8x16x32 `joint_matrix_mad` per sub-group) where
it beats f16 XMX, and stay in the integer domain where the math allows it.

## How to use this document

- Parts A, B and C are written to be applied as given. Do them in order A, B, C. Each part is
  one commit and has its own test and benchmark steps.
- Part D is design only. It needs a design pass before anyone writes code.
- Part E lists ideas that were checked and rejected, so nobody re-does that work.
- Rules for every part:
  - Every new path is behind an env flag that defaults to **0**, until it has been measured.
  - Never raise a test tolerance to make a test pass. A tolerance miss is a bug.
  - Keep the old path untouched. The new path is an early `if (...) { ...; return; }` in front of it.
  - Follow `AGENTS.md`: ASCII only, short comments.

## Background: where XMX is used today

| Path | File | Precision |
|---|---|---|
| Fused dequant GEMM, dense, N <= 64 | `fused-gemm.cpp`, called at `ggml-sycl.cpp:3620` | f16 x f16 -> f32 |
| Grouped MoE GEMM | `fused-gemm.cpp`, called at `ggml-sycl.cpp:6332`, `:6550` | f16 |
| Dense N > 64 (normal prefill) | `ggml-sycl.cpp:3622-3660` | f16 (weights expanded to f16 in memory) |
| Ordered MoE mat-vec | `mmvq.cpp:2969` `mul_mat_vec_moe_ordered_xmx` | **s8 x s8 -> s32** |
| QSA indexer score GEMM | `qsa-score.cpp:265-279` | f32 |
| Flash attention (oneDNN SDPA, oneMKL) | `fattn-onednn.cpp`, `fattn-mkl.cpp` | f16 |

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

---

## Part A: speed up the int8 MoE kernel (`mmvq.cpp`)

### A.1 What is slow now

Kernel `mul_mat_vec_moe_ordered_xmx` (`mmvq.cpp:2969-3105`). One work-group is one sub-group.
Per 32-value block it:
1. Stages an 8x32 A tile one byte per element through `traits::quant` (`:3026-3031`).
2. Stages the 32x16 B tile one byte per element. **Every element** recomputes
   `route -> token -> slot -> pointer`, with two integer divisions (`:3037-3051`).
3. Runs **one** 8x16x32 MAD (`:3070-3074`).
4. Stores the int32 tile to SLM and applies the scales with a scalar loop (`:3075-3085`).

Each work-group covers only 8 weight rows, so the B tile is staged again for every 8 rows.

Step 4 is not the problem: the work-group is one sub-group, so `group_barrier(sg)` is cheap, and
the loop is 8 FMAs per lane. Steps 1, 2 and the 8-row tile are the problem.

### A.2 Changes

1. Process `MT = 4` M tiles (32 weight rows) per work-group, so one staged B tile feeds 4 MADs.
2. Load `sub_b` once per block and reuse it for all `MT` A tiles.
3. Stage B one column per lane: lane `n` owns route `n` of the tile (tile_n == WARP_SIZE == 16).
   It computes its row pointers **once per route tile** and copies its 32 quants as 8 `int32`.
4. Stage A one row per lane, with wide loads, through a new `traits::stage_row`.
5. Keep the epilogue: lane `n` owns column `n`, sums stay in registers.
6. Put the routing behind a small policy struct. Part C reuses the same kernel for dense matmuls.

### A.3 New code

Order in `mmvq.cpp`: traits, then the two policy structs, then `mul_mat_xmx_i8`, then
`launch_mul_mat_xmx_i8`. All of it goes above `launch_mul_mat_vec_moe_ordered_xmx_impl`
(`mmvq.cpp:3108`), which calls it.

Add these traits next to `moe_xmx_traits` (`mmvq.cpp:2946-2966`). They write 32 int8 values of
one block to `dst` (SLM).

```cpp
template <> struct moe_xmx_traits<block_q8_0> {
    using block_t = block_q8_0;
    static constexpr int qk = QK8_0;
    // qs is 2-byte aligned in block_q8_0, so copy as 16 x uint16
    static __dpct_inline__ void stage_row(const block_t & b, int8_t * dst) {
        const uint16_t * src = (const uint16_t *) b.qs;
        uint16_t *       d16 = (uint16_t *) dst;
#pragma unroll
        for (int j = 0; j < QK8_0 / 2; ++j) {
            d16[j] = src[j];
        }
    }
    static __dpct_inline__ float scale(const block_t & b) { return (float) b.d; }
};

template <> struct moe_xmx_traits<block_iq4_nl> {
    using block_t = block_iq4_nl;
    static constexpr int qk = QK4_NL;
    static __dpct_inline__ void stage_row(const block_t & b, int8_t * dst) {
#pragma unroll
        for (int j = 0; j < QK4_NL / 2; ++j) {
            const uint8_t q = b.qs[j];
            dst[j]              = kvalues_iq4nl[q & 0xf];
            dst[j + QK4_NL / 2] = kvalues_iq4nl[q >> 4];
        }
    }
    static __dpct_inline__ float scale(const block_t & b) { return (float) b.d; }
};
```

Replace the old `quant()` members with `stage_row()`. Nothing else uses `quant()`; check with
`grep -n "traits::quant" mmvq.cpp`.

Add the two routing policies. `begin/end` is the range of route indices one work-group owns.

```cpp
// MoE: work-group slot = (compacted) expert, routes come from the sorted route list
template <int experts_used> struct xmx_rows_moe {
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
struct xmx_rows_dense {
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

New kernel. It replaces the body of `mul_mat_vec_moe_ordered_xmx`. Keep the old kernel under a
new name (`..._xmx_v1`) until A.5 shows the new one is correct and faster.

```cpp
static constexpr int XMX_I8_TM = 8;
static constexpr int XMX_I8_TN = 16;
static constexpr int XMX_I8_TK = 32;
static constexpr int XMX_I8_MT = 4;                       // M tiles per work-group
static constexpr int XMX_I8_ROWS = XMX_I8_MT * XMX_I8_TM; // 32 weight rows per work-group
static_assert(XMX_I8_TN == 16, "one B column per lane needs 16 lanes");

template <typename traits, typename rows_t>
[[sycl::reqd_sub_group_size(16)]]
static void mul_mat_xmx_i8(
    const void * __restrict__ vx_base, const void * __restrict__ vy, float * __restrict__ dst_base,
    const rows_t rmap, const int ncols, const int nrows,
    sycl::local_accessor<int8_t, 1> tile_a,     // XMX_I8_ROWS * XMX_I8_TK
    sycl::local_accessor<int32_t, 1> tile_b,    // XMX_I8_TK / 4 * XMX_I8_TN  (VNNI packed)
    sycl::local_accessor<float, 1> scales_a,    // XMX_I8_ROWS
    sycl::local_accessor<int32_t, 1> tile_c,    // XMX_I8_ROWS * XMX_I8_TN
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
    const int  row_base = item.get_group(1) * XMX_I8_ROWS;
    const int  nblocks  = ncols / traits::qk;
    const block_t * x = (const block_t *) ((const char *) vx_base + rmap.weight_offset(expert));

    for (uint32_t route_base = begin; route_base < end; route_base += XMX_I8_TN) {
        // lane n owns column n of this route tile
        const uint32_t     r     = route_base + lane;
        const bool         valid = r < end;
        const block_q8_1 * y     = nullptr;
        float *            d     = nullptr;
        if (valid) {
            rmap.rows(r, vy, dst_base, y, d);
        }
        float sums[XMX_I8_ROWS] = {};

        for (int blk = 0; blk < nblocks; ++blk) {
            // B: 32 quants of column `lane` as 8 int32, VNNI index (k/4)*16 + n
            float sb = 0.0f;
            if (valid) {
                const int32_t * q = (const int32_t *) y[blk].qs;
#pragma unroll
                for (int g = 0; g < XMX_I8_TK / 4; ++g) {
                    tile_b[g * XMX_I8_TN + lane] = q[g];
                }
                sb = (float) y[blk].ds[0];
            } else {
#pragma unroll
                for (int g = 0; g < XMX_I8_TK / 4; ++g) {
                    tile_b[g * XMX_I8_TN + lane] = 0;
                }
            }
            // A: lane stages rows lane and lane + 16, row-major 32 int8 per row
            for (int rl = lane; rl < XMX_I8_ROWS; rl += 16) {
                const int row = row_base + rl;
                int8_t *  dst = &tile_a[rl * XMX_I8_TK];
                if (row < nrows) {
                    const block_t & b = x[(size_t) row * nblocks + blk];
                    traits::stage_row(b, dst);
                    scales_a[rl] = traits::scale(b);
                } else {
#pragma unroll
                    for (int k = 0; k < XMX_I8_TK; ++k) {
                        dst[k] = 0;
                    }
                    scales_a[rl] = 0.0f;
                }
            }
            sycl::group_barrier(sg);

            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, XMX_I8_TK, XMX_I8_TN, mx::layout::ext_intel_packed> sub_b;
            mx::joint_matrix_load(sg, sub_b,
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    (int8_t *) &tile_b[0]),
                XMX_I8_TN * 4);
#pragma unroll
            for (int mt = 0; mt < XMX_I8_MT; ++mt) {
                mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, XMX_I8_TM, XMX_I8_TK, mx::layout::row_major> sub_a;
                mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, XMX_I8_TM, XMX_I8_TN> sub_c;
                mx::joint_matrix_fill(sg, sub_c, 0);
                mx::joint_matrix_load(sg, sub_a,
                    tile_a.get_multi_ptr<sycl::access::decorated::no>() + mt * XMX_I8_TM * XMX_I8_TK, XMX_I8_TK);
                mx::joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
                mx::joint_matrix_store(sg, sub_c,
                    tile_c.get_multi_ptr<sycl::access::decorated::no>() + mt * XMX_I8_TM * XMX_I8_TN,
                    XMX_I8_TN, mx::layout::row_major);
            }
            sycl::group_barrier(sg);

            // element (row rl, column lane) is at rl * 16 + lane
#pragma unroll
            for (int rl = 0; rl < XMX_I8_ROWS; ++rl) {
                sums[rl] += scales_a[rl] * sb * (float) tile_c[rl * XMX_I8_TN + lane];
            }
            sycl::group_barrier(sg);  // tile_a/b/c are rewritten by the next block
        }

        if (valid) {
#pragma unroll
            for (int rl = 0; rl < XMX_I8_ROWS; ++rl) {
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
  `local_accessor<int8_t, 1>` of size `XMX_I8_TK * XMX_I8_TN`, and write the int32 values through
  `((int32_t *) &tile_b[0])[g * 16 + lane]`. The old kernel loads `tile_b` as `int8_t` with
  `get_multi_ptr`, and that is known to compile.
- `sums[32]` is 32 floats per lane. That is fine for 128 GRF.

New launcher. It replaces `launch_mul_mat_vec_moe_ordered_xmx_impl` (`mmvq.cpp:3108-3141`).
Keep the `switch (n_experts_used)` wrapper and pass `xmx_rows_moe<experts_used>`.

```cpp
template <typename traits, typename rows_t>
static void launch_mul_mat_xmx_i8(const void * vx, const void * vy, float * dst, const rows_t & rmap,
                                  int n_slots, int ncols, int nrows, dpct::queue_ptr stream) {
    const int n_row_groups = (nrows + XMX_I8_ROWS - 1) / XMX_I8_ROWS;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int8_t, 1>  tile_a(XMX_I8_ROWS * XMX_I8_TK, cgh);
        sycl::local_accessor<int32_t, 1> tile_b(XMX_I8_TK / 4 * XMX_I8_TN, cgh);
        sycl::local_accessor<float, 1>   scales_a(XMX_I8_ROWS, cgh);
        sycl::local_accessor<int32_t, 1> tile_c(XMX_I8_ROWS * XMX_I8_TN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_slots, n_row_groups * 16), sycl::range<2>(1, 16)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                mul_mat_xmx_i8<traits>(vx, vy, dst, rmap, ncols, nrows, tile_a, tile_b, scales_a, tile_c, item);
            });
    });
}
```

MoE call (inside the existing `launch_mul_mat_vec_moe_ordered_xmx_impl<traits, experts_used>`):

```cpp
const int n_slots = route_order.active_experts != nullptr ? route_order.n_active_max : route_order.n_experts;
const xmx_rows_moe<experts_used> rmap = {
    route_order.expert_offsets, route_order.sorted_routes, route_order.active_experts, route_order.n_active,
    n_experts_used, expert_weight_stride, src1_row_stride, src1_token_stride, dst_row_stride, dst_token_stride,
};
launch_mul_mat_xmx_i8<traits>(vx_base, vy, dst_base, rmap, n_slots, ncols, nrows, stream);
```

Flag: add `int g_ggml_sycl_moe_xmx_v2 = 0;` next to `g_ggml_sycl_moe_xmx`
(`ggml-sycl.cpp:126`), `extern int g_ggml_sycl_moe_xmx_v2;` next to `common.hpp:85`, the env
read `ggml_sycl_get_env("GGML_SYCL_MOE_XMX_V2", 0)` next to `ggml-sycl.cpp:435`, and a
`GGML_LOG_INFO` line next to the other flag prints. In the impl launcher, call the new kernel
when the flag is 1 and the old one otherwise.

Also add a sub-group size check to `ggml_sycl_moe_q8_xmx_supported` (`mmvq.cpp:2925`):

```cpp
if (WARP_SIZE != 16) {
    return false;
}
```

### A.4 Test

```sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake --build build -j --target test-backend-ops llama-bench llama-perplexity
GGML_SYCL_MOE_XMX_V2=1 ./build/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID
```

`MUL_MAT_ID` cases with q8_0 and iq4_nl must pass. If they do not run the XMX path (it needs at
least 8 routes per expert on average, `mmvq.cpp:3562`), add a temporary
`fprintf(stderr, ...)` in the launcher to confirm. Remove it before the commit.

### A.5 Benchmark

```sh
for v in 0 1; do GGML_SYCL_MOE_XMX_V2=$v ./build/bin/llama-bench -m <q8_0 MoE model> -p 512 -n 0 -ub 512; done
for v in 0 1; do GGML_SYCL_MOE_XMX_V2=$v ./build/bin/llama-bench -m <iq4_nl MoE model> -p 512 -n 0 -ub 512; done
```

Turn the flag on by default only if prompt t/s does not drop on either model. Then delete the
`_v1` kernel.

---

## Part B: QSA indexer score in int8 (`qsa-score.cpp`)

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

There is one scale per whole row, so the GEMM is a plain s8 x s8 -> s32 GEMM: no per-block
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
static void k_qsa_score_reduce_i8(const int32_t * tile, const float * sa, const float * sb, float * dst,
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
    if (g_ggml_sycl_qsa_score_i8 && g_ggml_sycl_enable_dnn) {
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
                k_qsa_score_reduce_i8(tile_dd, sa_dd, sb_dd, dst_dd, n_blocks, n_heads, nt, n_tps, t0, item);
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

5. Flag `g_ggml_sycl_qsa_score_i8`, env `GGML_SYCL_QSA_SCORE_I8`, default 0. Add it in the same
   four places as `g_ggml_sycl_fuse_qsa_score`: definition `ggml-sycl.cpp:138`, extern
   `common.hpp:93`, env read `ggml-sycl.cpp:444`, print `ggml-sycl.cpp:613`.

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
GGML_SYCL_QSA_SCORE_I8=0 ./build/bin/test-backend-ops -b SYCL0 -o QSA_SCORE
GGML_SYCL_QSA_SCORE_I8=1 ./build/bin/test-backend-ops -b SYCL0 -o QSA_SCORE
```

Both must pass. If the int8 run fails on NMSE, look for an indexing bug (the `sb` index is the
usual suspect). Do not raise the tolerance.

### B.4 Model-level check and benchmark

Top-k selection is what matters. With a qwen4exp model and a long prompt (32k or more):

```sh
for v in 0 1; do GGML_SYCL_QSA_SCORE_I8=$v ./build/bin/llama-perplexity -m <qwen4exp model> -f <long text> -c 32768 --chunks 4; done
for v in 0 1; do GGML_SYCL_QSA_SCORE_I8=$v ./build/bin/llama-bench -m <qwen4exp model> -p 32768 -n 0; done
```

Perplexity must stay within noise (difference under 0.5%). Turn the flag on by default only then.

---

## Part C: dense int8 matmul for Q8_0 and IQ4_NL, 9 to 64 columns

Do Part A first. This part reuses `mul_mat_xmx_i8` with `xmx_rows_dense`.

### C.1 Export a wrapper from `mmvq.cpp`

```cpp
// mmvq.hpp
bool ggml_sycl_mul_mat_xmx_i8_dense(ggml_type src0_type, const void * vx, const void * vy_q8_1, float * dst,
                                    int M, int N, int K, int ldd, dpct::queue_ptr stream);
```

```cpp
// mmvq.cpp, after launch_mul_mat_xmx_i8
bool ggml_sycl_mul_mat_xmx_i8_dense(ggml_type src0_type, const void * vx, const void * vy_q8_1, float * dst,
                                    int M, int N, int K, int ldd, dpct::queue_ptr stream) {
    if (K % QK8_1 != 0 || N < 1 || !ggml_sycl_moe_q8_xmx_supported(stream)) {
        return false;
    }
    const xmx_rows_dense rmap = { N, (size_t) (K / QK8_1) * sizeof(block_q8_1), ldd };
    const int n_slots = (N + XMX_I8_TN - 1) / XMX_I8_TN;
    switch (src0_type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_xmx_i8<moe_xmx_traits<block_q8_0>>(vx, vy_q8_1, dst, rmap, n_slots, K, M, stream);
            return true;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_xmx_i8<moe_xmx_traits<block_iq4_nl>>(vx, vy_q8_1, dst, rmap, n_slots, K, M, stream);
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
    const auto * src0_extra_i8 = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool   reordered_i8  = src0_extra_i8 && src0_extra_i8->optimized_feature.is_reordered();
    if (g_ggml_sycl_dense_xmx_i8 && !reordered_i8 && src1->type == GGML_TYPE_F32 &&
        (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_IQ4_NL) &&
        src1_ncols <= 64 && ne10 % QK8_1 == 0) {
        ggml_sycl_pool_alloc<block_q8_1> src1_q8(ctx.pool(), (size_t) src1_ncols * (ne10 / QK8_1));
        quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_q8.get(), (int) ne10, (int) src1_ncols, (int) ne10, stream);
        if (ggml_sycl_mul_mat_xmx_i8_dense(src0->type, src0_dd_i, src1_q8.get(), dst_dd_i,
                                           (int) row_diff, (int) src1_ncols, (int) ne10, ldc, stream)) {
            return;
        }
    }
}
```

- The surrounding block already requires `row_diff == src0->ne[1]` and a contiguous `src0`.
- That block is only entered when `use_fp16` is true, which needs a `-DGGML_SYCL_F16=ON` build
  (`ggml-sycl.cpp:3566-3570`). Build with it for this part.
- `ldc` is the dst column stride, already computed at `ggml-sycl.cpp:3564`.
- `src1_ddf_i` holds `src1_ncols` contiguous f32 rows of `ne10` values here.
- Batches of 1 to 8 columns normally go to MMVQ before this function is reached, so in practice
  this path sees 9 to 64 columns.
- Flag `g_ggml_sycl_dense_xmx_i8`, env `GGML_SYCL_DENSE_XMX_I8`, default 0, in the same four
  places as in A.3.

### C.3 Test and benchmark

```sh
GGML_SYCL_DENSE_XMX_I8=1 ./build/bin/test-backend-ops -b SYCL0 -o MUL_MAT
```

All q8_0 and iq4_nl cases must pass at the existing tolerance. MMVQ already uses q8_1
activations and passes it, so this path should too.

Benchmark with a micro-batch where the fused f16 GEMM runs today (N <= 64):

```sh
for v in 0 1; do GGML_SYCL_DENSE_XMX_I8=$v ./build/bin/llama-bench -m <q8_0 dense model> -p 512 -n 0 -ub 32,64; done
for v in 0 1; do GGML_SYCL_DENSE_XMX_I8=$v ./build/bin/llama-bench -m <iq4_nl dense model> -p 512 -n 0 -ub 32,64; done
```

The competitor is the fused f16 GEMM (`GGML_SYCL_FUSED_GEMM=1`, the default). If int8 is not
faster, leave the flag off and record the numbers in this document.

---

## Part D: design only (not ready for a simple agent)

### D.1 Int8 variant of `fused-gemm.cpp` for more formats

Formats whose decoded values are integers:

| Format | Int8 A value | Extra work |
|---|---|---|
| Q8_0, IQ4_NL | done in Part C | |
| IQ4_XS | `kvalues_iq4nl[nibble]` | 6-bit sub-scale per 32 |
| IQ2_*, IQ3_* | grid byte x sign | per-32 or per-16 sub-scale |
| Q4_K, Q5_K | unsigned 0..15 / 0..31 | min term `dmin * m * sum(a)` from q8_1 `ds[1]` |
| Q6_K | -32..31 | sub-blocks of 16, but one MAD is 32 deep |
| IQ1_S, IQ1_M | not integer (delta +-0.125) | skip |

Q4_K would no longer need `GGML_SYCL_FAST_AND_SLOPPY` (`common.hpp:134`): int8 values with f32
scales cannot overflow the way f16-decoded weights can.

### D.2 K-quants: one float rescale per 256 values

K-quant sub-scales are small integers. With activations quantized like Q8_K (one scale per 256
values, plus integer sums per 16), the CPU backend's approach applies:

    acc_s32  = sum_j sc_j * (q_j . a_j)           // 8 sub-blocks of 32, all integer
    acc_s32 -= sum_j m_j * bsum_j                 // integer min term
    acc_f32 += d_w * d_a * acc_s32                // one float epilogue per superblock

That cuts the float epilogue 8x for Q4_K and Q5_K. It needs a Q8_K activation quantizer on SYCL,
and an epilogue that multiplies the int32 tile by a per-row integer before summing, either in
SLM (as in Part A) or with `joint_matrix_apply`.

### D.3 Dense prefill with more than 64 columns

Still expands the weights to f16 in memory, then runs a oneDNN or oneMKL f16 GEMM. Options:
- Extend the int8 kernel to more columns per work-group (needs a 2-D work-group with more than
  one sub-group, and SLM sharing of the A tile).
- oneDNN int8 matmul with grouped scales (group 32). First confirm that the installed oneDNN
  supports grouped scales on both inputs on Intel GPU.

### D.4 Attention: integer bound for tile skipping

With a Q8_0 K cache (`kv-soa.hpp`) and Q quantized per (token, head), Q.K^T can run as int8.
The exact row max still needs one float multiply per score, because the K scale differs per key.
An integer max does give a cheap per-tile upper bound:

    bound = sQ * max_j(sK_j) * max_j(I_j)

If `bound < running_max - T` (for example `T = 20` in log2 units), every score in the tile adds
less than `2^-T` after the exp. The kernel can then skip the exp, the row sum and the P.V MAD
for the whole tile. This changes results by a bounded amount, so it must be opt-in, like
`GGML_SYCL_FAST_AND_SLOPPY`.

Masks are already one bit per cell (`kq-mask-bits.hpp`, `qsa-mask.cpp`). Before starting, check
whether `fattn-tile.hpp` and `fattn-mkl.cpp` already skip tiles whose mask words are all zero.

---

## Part E: checked and rejected

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
- **Single-token decode.** Memory-bound mat-vec. MMVQ and ESIMD stay the right path.
