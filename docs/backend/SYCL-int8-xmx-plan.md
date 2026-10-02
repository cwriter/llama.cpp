# SYCL: int8 XMX implementation plan

Status: plan only. None of this code has been compiled or run. Line numbers refer to
`performance_uplift_mega_branch` at `fbb9699`. Search for the quoted code if lines moved.

Goal: use the int8 XMX path (s8 x s8 -> s32, one 8x16x32 `joint_matrix_mad` per sub-group) where
it beats f16 XMX, and stay in the integer domain where the math allows it.

## How to use this document

- Parts A, B and C are written to be applied as given, in order A, B, C. One commit each, each
  with its own test and benchmark steps.
- Part D has been through a design pass. Each item says what to build, gives the key code, and
  says what to measure before and after. Items are ordered by expected value. D.1 and D.2 build
  on Part A. D.3 builds on Part C.
- Part E lists ideas that were checked and rejected, so nobody re-does that work.
- Naming: every new path, flag and kernel uses `XMX_INT` / `xmx_int`.
- Rules for every part:
  - Every new path is behind an env flag that defaults to **0**, until it has been measured.
  - Never raise a test tolerance to make a test pass. A tolerance miss is a bug.
  - Keep the old path untouched. The new path is an early `if (...) { ...; return; }` in front of it.
  - Follow `AGENTS.md`: ASCII only, short comments.

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
| `GGML_SYCL_XMX_INT_EPI` | D.2 | 0: scale epilogue through SLM, 1: in registers |
| `GGML_SYCL_FA_KQ_INT` | D.4 | int8 Q.K^T in the oneMKL flash attention path |

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

## Part D: design pass

Order by expected value: D.1, D.2, D.3, then D.4. D.4 starts with measurement only.

### D.1 More weight formats in the int8 kernel

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

### D.3 Dense prefill with more than 64 columns

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

#### D.4.2 Skip fully masked (query tile, KV chunk) pairs: measure first

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
