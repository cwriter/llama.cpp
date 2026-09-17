#include "mmvq.hpp"

#include "ggml.h"
#include "common.hpp"
#include "element_wise.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

// Minimum weight-row count at which the Q4_K multi-column MMVQ kernel handles two output rows per
// subgroup (rows_per_sg == 2) instead of one, when ncols_dst == 2.
//
// Pairing rows lets a subgroup load each activation block once and apply it to two rows, at the cost
// of halving the number of subgroups in the launch. With only two destination columns there is too
// little work per row to hide that loss of parallelism, so pairing only pays off once there are
// enough rows to keep the device occupied. This is a measured performance crossover, not a
// correctness or hardware limit - both variants compute the same result for any nrows.
//
// Derived on Intel Arc Pro B70 with `test-backend-ops perf -o MUL_MAT` (Q4_K, ncols_dst == 2),
// sweeping nrows over 5120..6912 at ncols 17408 and 19968: one row per subgroup was up to 9% faster
// below the crossover, two rows per subgroup 8-15% faster above it, and the crossover fell inside
// (6144, 6272] for both ncols with no measurable ncols dependence. A later 32-row granularity sweep
// narrowed it to (6144, 6176], so 6272 is a conservative gate rather than the exact crossover.
// ncols_dst >= 3 amortizes the activation loads over more columns and is faster with two rows at
// every row count, so it does not consult this threshold.
static constexpr int Q4_K_MMVQ_ROW_PAIR_MIN_NROWS = 6272;

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                  const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t* q8_1_quant_ptr = (const int8_t*)vy + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = (const sycl::half2*)((const char*)vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
template <typename reorder_vec_dot_q_sycl, int ncols_dst, bool has_fusion = false, int rows_per_sg = 1>
static void mul_mat_vec_q_reorder_ncols(const void * __restrict__ vx, const void * __restrict__ vgate,
                                        const void * __restrict__ vy, float * __restrict__ dst, const int ncols,
                                        const int nrows, const int stride_col_y_bytes, const int stride_col_dst,
                                        const ggml_glu_op glu_op, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row0         = (workgroup_id * sg_range + sg_id) * rows_per_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    if (row0 >= nrows) {
        return;
    }

    static_assert(rows_per_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum[ncols_dst][rows_per_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] float partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? rows_per_sg : 1] = {};
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int  iby       = i * block_type::block_to_q8_1_ratio();

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            if constexpr (rows_per_sg > 1) {
                typename reorder_vec_dot_q_sycl::weights wx[rows_per_sg];
                [[maybe_unused]] typename reorder_vec_dot_q_sycl::weights wg[rows_per_sg];
#pragma unroll
                for (int r = 0; r < rows_per_sg; ++r) {
                    const int row = sycl::min(row0 + r, nrows - 1);
                    const int ibx = row * blocks_per_row + i;
                    const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                    const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                    if constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                    const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                    const sycl::half2 * q8_1_ds_ptr =
                        (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));
                    const auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#pragma unroll
                    for (int r = 0; r < rows_per_sg; ++r) {
                        partial_sum[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        if constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } else if constexpr (reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value) {
                const int ibx = row0 * blocks_per_row + i;
                const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                const auto wx = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                if constexpr (has_fusion) {
                    const auto wg = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);

#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                        const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                        const sycl::half2 * q8_1_ds_ptr =
                            (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        const auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                        const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                        const sycl::half2 * q8_1_ds_ptr =
                            (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } else {
                const int ibx = row0 * blocks_per_row + i;
                const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                    const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                    const sycl::half2 * q8_1_ds_ptr =
                        (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));

                    partial_sum[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    if constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rows_per_sg; ++r) {
            float sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum[j][r], std::plus<>());

            if constexpr (has_fusion) {
                const float gate = sycl::reduce_over_group(nd_item.get_sub_group(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                sum *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            if (sg.leader() && row0 + r < nrows) {
                dst[j * stride_col_dst + row0 + r] = sum;
            }
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                          const int ncols, const int nrows, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, int ncols_dst>
static void mul_mat_vec_q_ncols(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols,
        const int nrows,
        const int stride_col_y,
        const int stride_col_dst,
        const sycl::nd_item<3> & item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1)
                  + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    // partial sums: one per output column
    float tmp[ncols_dst] = {0.0f};

    const block_q_t  * x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr);
         i < blocks_per_row;
         i += blocks_per_warp) {

        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmp[j] += dpct::permute_sub_group_by_xor(
                item_ct1.get_sub_group(), tmp[j], mask);
        }
    }

    if (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            dst[j * stride_col_dst + row] = tmp[j];
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template<int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<2> &item_ct1) {
    // 2D Row calculation: Group offset along dim 0 + local row offset
    const int row = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                    item_ct1.get_local_id(0);

    if (row >= nrows) {
        return;
    }

    // Local thread index within the warp/sub-group along dim 1 (0...31)
    const int lane_id = item_ct1.get_local_id(1);

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = lane_id / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i; // x block index

        const int iby = i * (qk / QK8_1); // y block index that aligns with ibx

        const int iqs = vdr * (lane_id % (qi / vdr)); // x block quant index when casting quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums across the sub-group/warp
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    // Only the first thread in the warp writes back the row result
    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_m_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}
template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_nl_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<2> &item_ct1) {
    // 2D Row calculation: Group offset along dim 0 + local row offset
    const int row = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                    item_ct1.get_local_id(0);

    if (row >= nrows) {
        return;
    }

    // Local thread index within the warp/sub-group along dim 1 (0...31)
    const int lane_id = item_ct1.get_local_id(1);

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = lane_id / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i; // x block index

        const int iby = i * (qk / QK8_1); // y block index that aligns with ibx

        const int iqs = vdr * (lane_id % (qi / vdr)); // x block quant index when casting quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums across the sub-group/warp
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    // Only the first thread in the warp writes back the row result
    if (lane_id == 0) {
        dst[row] = tmp;
    }
}


template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q4_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_0, block_q4_0,
                                    VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q4_1_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK4_0, QI4_1, block_q4_1,
                                    VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q4_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q4_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q4_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q4_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q4_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q4_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q4_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q4_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_MXFP4, QI_MXFP4, block_mxfp4,
                                    VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_nvfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_nvfp4_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_NVFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_NVFP4, QI_NVFP4, block_nvfp4,
                                    VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_nvfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_nvfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_nvfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_nvfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_nvfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_nvfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_nvfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_nvfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q5_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK5_0, QI5_0, block_q5_0,
                                    VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q5_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q5_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q5_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q5_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q5_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q5_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q5_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q5_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q5_1_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK5_1, QI5_1, block_q5_1,
                                    VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q5_1_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q5_1_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q5_1_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q5_1_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q5_1_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q5_1_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q5_1_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q5_1_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q8_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q8_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK8_0, QI8_0, block_q8_0,
                                    VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q1_0_q8_1_sycl(const void * vx, const void * vy,
                                       float * dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q<QK1_0, QI1_0, block_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q1_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK1_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK1_0, QI1_0, block_q1_0,
                                    VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q1_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q1_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q1_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q1_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q1_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q1_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q1_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q1_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q2_0_q8_1_sycl(const void * vx, const void * vy,
                                       float * dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q<QK2_0, QI2_0, block_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                    vx, vy, dst, ncols, nrows, item_ct1);
            });
    });
}

template <int ncols_dst>
static void mul_mat_vec_q2_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK2_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK2_0, QI2_0, block_q2_0,
                                    VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q2_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q2_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q2_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q2_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q2_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q2_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q2_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q2_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q2_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_K, QI2_K, block_q2_K,
                                    VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q2_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q2_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q2_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q2_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q2_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q2_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q2_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q2_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q2_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q3_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

template <int ncols_dst>
static void mul_mat_vec_q3_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_ncols<QK_K, QI3_K, block_q3_K,
                                    VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, ncols_dst>(
                    vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
            });
    });
}

static void mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q3_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q3_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q3_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q3_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q3_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q3_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q3_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q3_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
}


static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q4_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_K, block_q4_K,
                                        VDR_Q4_K_Q8_1_MMVQ,
                                        vec_dot_q4_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q4_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q4_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q4_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q4_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q4_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q4_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q4_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q4_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <int ncols_dst, int rows_per_sg>
static void reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups * rows_per_sg);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>, ncols_dst,
                                                        /*has_fusion=*/ false, rows_per_sg>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    constexpr int rows_per_sg = ncols_dst >= 3 && ncols_dst <= 4 ? 2 : 1;
    reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<ncols_dst, rows_per_sg>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2:
            if (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            } else {
                reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_impl<2, 1>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            }
            break;
        case 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q5_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI5_K, block_q5_K,
                                        VDR_Q5_K_Q8_1_MMVQ,
                                        vec_dot_q5_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q5_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q5_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q5_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q5_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q5_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q5_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q5_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q5_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q5_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);


    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_q6_K_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI6_K, block_q6_K,
                                        VDR_Q6_K_Q8_1_MMVQ,
                                        vec_dot_q6_K_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_q6_K_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_q6_K_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_q6_K_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_q6_K_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_q6_K_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_q6_K_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_q6_K_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_q6_K_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
}


static void mul_mat_vec_iq2_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_s_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Ensure we process multiple rows per Work-Group (e.g. 8 rows = 256 threads)
    // to give the Intel Xe thread scheduler enough active sub-groups to hide SBID stalls.
    constexpr int rows_per_wg = (GGML_SYCL_MMV_Y < 8) ? 8 : GGML_SYCL_MMV_Y;
    const int block_num_y = (nrows + rows_per_wg - 1) / rows_per_wg;

    const sycl::range<2> global_range(block_num_y * rows_per_wg, WARP_SIZE);
    const sycl::range<2> local_range(rows_per_wg, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1>(
                        vx, vy, dst, ncols, nrows, item_ct1);
                });
    });
}

static void mul_mat_vec_iq1_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_m_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_nl_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);

    // Ensure we process at least 8 rows per work-group (256 threads total)
    // to give Intel Xe thread schedulers sufficient occupancy to hide latency.
    constexpr int rows_per_wg = (GGML_SYCL_MMV_Y < 8) ? 8 : GGML_SYCL_MMV_Y;
    const int block_num_y = (nrows + rows_per_wg - 1) / rows_per_wg;

    const sycl::range<2> global_range(block_num_y * rows_per_wg, WARP_SIZE);
    const sycl::range<2> local_range(rows_per_wg, WARP_SIZE);

    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(
                        vx, vy, dst, ncols, nrows, item_ct1);
                });
    });
}

static void mul_mat_vec_iq4_xs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

template <int ncols_dst>
static void mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, block_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                const dpct::queue_ptr & stream) {
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_1:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_1:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q8_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q1_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q2_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q2_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q3_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q6_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_XXS:
                mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_S:
                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_MXFP4:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_NVFP4:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            default:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
static __dpct_inline__ float vec_dot_iq2_xxs_q8_1_moe(const void * __restrict__ vbq,
                                                      const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

static __dpct_inline__ float vec_dot_iq2_xs_q8_1_moe(const void * __restrict__ vbq,
                                                     const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid, ksigns64);
}

static __dpct_inline__ float vec_dot_iq3_xxs_q8_1_moe(const void * __restrict__ vbq,
                                                      const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid, ksigns64);
}

static __dpct_inline__ float vec_dot_iq3_s_q8_1_moe(const void * __restrict__ vbq,
                                                    const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

static __dpct_inline__ float vec_dot_iq1_s_q8_1_moe(const void * __restrict__ vbq,
                                                    const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
}

void ggml_sycl_build_moe_route_order(
    const int32_t * ids_dev,
    size_t          ids_token_stride,
    int             n_experts,
    int             n_experts_used,
    int             n_tokens,
    uint32_t *      expert_counts,
    uint32_t *      expert_offsets,
    uint32_t *      expert_cursors,
    uint32_t *      sorted_routes,
    dpct::queue_ptr stream) {
    const uint32_t n_routes = (uint32_t) n_tokens * n_experts_used;
    constexpr uint32_t block_size = 256;
    const uint32_t global_size = ((n_routes + block_size - 1) / block_size) * block_size;

    stream->memset(expert_counts, 0, (size_t) n_experts * sizeof(uint32_t));
    stream->parallel_for(
        sycl::nd_range<1>(global_size, block_size),
        [=](sycl::nd_item<1> item) {
            const uint32_t route = item.get_global_linear_id();
            if (route >= n_routes) {
                return;
            }
            const uint32_t token = route / n_experts_used;
            const uint32_t slot  = route - token * n_experts_used;
            const int32_t * ids_token = (const int32_t *) ((const char *) ids_dev + token * ids_token_stride);
            const int32_t expert = ids_token[slot];
            if (expert >= 0 && expert < n_experts) {
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(expert_counts[expert]).fetch_add(1);
            }
        });

    stream->single_task([=]() {
        uint32_t offset = 0;
        for (int expert = 0; expert < n_experts; ++expert) {
            expert_offsets[expert] = offset;
            expert_cursors[expert] = offset;
            offset += expert_counts[expert];
        }
        expert_offsets[n_experts] = offset;
    });

    stream->parallel_for(
        sycl::nd_range<1>(global_size, block_size),
        [=](sycl::nd_item<1> item) {
            const uint32_t route = item.get_global_linear_id();
            if (route >= n_routes) {
                return;
            }
            const uint32_t token = route / n_experts_used;
            const uint32_t slot  = route - token * n_experts_used;
            const int32_t * ids_token = (const int32_t *) ((const char *) ids_dev + token * ids_token_stride);
            const int32_t expert = ids_token[slot];
            if (expert >= 0 && expert < n_experts) {
                const uint32_t position =
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(expert_cursors[expert]).fetch_add(1);
                sorted_routes[position] = route;
            }
        });
}

static bool ggml_sycl_moe_q8_xmx_supported(dpct::queue_ptr stream) {
    namespace mx = sycl::ext::oneapi::experimental::matrix;
    try {
        const auto combinations = stream->get_device().get_info<
            sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        for (const auto & combination : combinations) {
            if (combination.atype == mx::matrix_type::sint8 &&
                combination.btype == mx::matrix_type::sint8 &&
                combination.ctype == mx::matrix_type::sint32 &&
                combination.dtype == mx::matrix_type::sint32 &&
                (combination.max_msize >= 8 || combination.msize == 8) &&
                (combination.max_nsize >= 16 || combination.nsize == 16) &&
                (combination.max_ksize >= 32 || combination.ksize == 32)) {
                return true;
            }
        }
    } catch (const sycl::exception &) {
    }
    return false;
}

template <int experts_used>
[[sycl::reqd_sub_group_size(WARP_SIZE)]]
static void mul_mat_vec_q8_0_moe_ordered_xmx(
    const block_q8_0 * __restrict__ vx_base,
    const void * __restrict__ vy_base,
    float * __restrict__ dst_base,
    const uint32_t * __restrict__ expert_offsets,
    const uint32_t * __restrict__ sorted_routes,
    const int ncols,
    const int nrows,
    const int n_experts_used,
    const size_t expert_weight_stride,
    const size_t dst_row_stride,
    const size_t src1_row_stride,
    const size_t dst_token_stride,
    const size_t src1_token_stride,
    sycl::local_accessor<int8_t, 1> tile_a,
    sycl::local_accessor<int8_t, 1> tile_b,
    sycl::local_accessor<float, 1> scales_a,
    sycl::local_accessor<float, 1> scales_b,
    sycl::local_accessor<int32_t, 1> tile_c,
    const sycl::nd_item<2> & item) {
    namespace mx = sycl::ext::oneapi::experimental::matrix;
    constexpr int tile_m = 8;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;

    const int expert  = item.get_group(0);
    const int group_y = item.get_group(1);
    const int row_base = group_y * tile_m;
    const int lane = item.get_local_linear_id();
    const uint32_t begin = expert_offsets[expert];
    const uint32_t end   = expert_offsets[expert + 1];
    if (begin == end) {
        return;
    }

    const auto sg = item.get_sub_group();
    const block_q8_0 * x = (const block_q8_0 *) ((const char *) vx_base + (size_t) expert * expert_weight_stride);
    const int blocks_per_row = ncols / QK8_0;
    float sums[tile_m] = {};

    for (uint32_t route_base = begin; route_base < end; route_base += tile_n) {
        const int route_count = sycl::min((uint32_t) tile_n, end - route_base);
#pragma unroll
        for (int i = 0; i < tile_m; ++i) {
            sums[i] = 0.0f;
        }

        for (int block = 0; block < blocks_per_row; ++block) {
            for (int index = lane; index < tile_m * tile_k; index += WARP_SIZE) {
                const int row_local = index / tile_k;
                const int k = index - row_local * tile_k;
                const int row = row_base + row_local;
                tile_a[index] = row < nrows ? x[row * blocks_per_row + block].qs[k] : 0;
            }
            for (int row_local = lane; row_local < tile_m; row_local += WARP_SIZE) {
                const int row = row_base + row_local;
                scales_a[row_local] = row < nrows ? (float) x[row * blocks_per_row + block].d : 0.0f;
            }

            for (int index = lane; index < tile_k * tile_n; index += WARP_SIZE) {
                const int k_group = index / (tile_n * 4);
                const int packed_offset = index - k_group * tile_n * 4;
                const int route_local = packed_offset / 4;
                const int k = k_group * 4 + packed_offset % 4;
                int8_t value = 0;
                if (route_local < route_count) {
                    const uint32_t route = sorted_routes[route_base + route_local];
                    const uint32_t token = route / (experts_used > 0 ? experts_used : n_experts_used);
                    const uint32_t slot  = route - token * (experts_used > 0 ? experts_used : n_experts_used);
                    const block_q8_1 * y = (const block_q8_1 *) ((const char *) vy_base +
                        (size_t) token * src1_token_stride + (size_t) slot * src1_row_stride);
                    value = y[block].qs[k];
                }
                tile_b[index] = value;
            }
            for (int route_local = lane; route_local < tile_n; route_local += WARP_SIZE) {
                float scale = 0.0f;
                if (route_local < route_count) {
                    const uint32_t route = sorted_routes[route_base + route_local];
                    const uint32_t token = route / (experts_used > 0 ? experts_used : n_experts_used);
                    const uint32_t slot  = route - token * (experts_used > 0 ? experts_used : n_experts_used);
                    const block_q8_1 * y = (const block_q8_1 *) ((const char *) vy_base +
                        (size_t) token * src1_token_stride + (size_t) slot * src1_row_stride);
                    scale = (float) y[block].ds[0];
                }
                scales_b[route_local] = scale;
            }
            sycl::group_barrier(sg);

            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, tile_m, tile_k, mx::layout::row_major> sub_a;
            mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, tile_k, tile_n, mx::layout::ext_intel_packed> sub_b;
            mx::joint_matrix<sycl::sub_group, int32_t, mx::use::accumulator, tile_m, tile_n, mx::layout::dynamic> sub_c;
            mx::joint_matrix_fill(sg, sub_c, 0);
            mx::joint_matrix_load(sg, sub_a, tile_a.get_multi_ptr<sycl::access::decorated::no>(), tile_k);
            // VNNI-packed B is (tile_k/4) rows of tile_n*4 int8, so the row stride is tile_n*4
            mx::joint_matrix_load(sg, sub_b, tile_b.get_multi_ptr<sycl::access::decorated::no>(), tile_n * 4);
            mx::joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
            mx::joint_matrix_store(sg, sub_c, tile_c.get_multi_ptr<sycl::access::decorated::no>(), tile_n, mx::layout::row_major);
            sycl::group_barrier(sg);

#pragma unroll
            for (int output = lane; output < tile_m * tile_n; output += WARP_SIZE) {
                const int row_local = output / tile_n;
                const int route_local = output - row_local * tile_n;
                sums[output / WARP_SIZE] +=
                    (scales_a[row_local] * scales_b[route_local]) * tile_c[output];
            }
            if (block + 1 < blocks_per_row || route_base + route_count < end) {
                sycl::group_barrier(sg);
            }
        }

#pragma unroll
        for (int output = lane; output < tile_m * tile_n; output += WARP_SIZE) {
            const int row_local = output / tile_n;
            const int route_local = output - row_local * tile_n;
            const int row = row_base + row_local;
            if (row < nrows && route_local < route_count) {
                const uint32_t route = sorted_routes[route_base + route_local];
                const uint32_t token = route / (experts_used > 0 ? experts_used : n_experts_used);
                const uint32_t slot  = route - token * (experts_used > 0 ? experts_used : n_experts_used);
                float * dst = (float *) ((char *) dst_base +
                    (size_t) token * dst_token_stride + (size_t) slot * dst_row_stride);
                dst[row] = sums[output / WARP_SIZE];
            }
        }
    }
}

template <int experts_used>
static void launch_mul_mat_vec_q8_0_moe_ordered_xmx_impl(
    const void * vx_base, const void * vy, float * dst_base,
    const ggml_sycl_moe_route_order & route_order,
    const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const size_t dst_token_stride,
    const size_t src1_token_stride, dpct::queue_ptr stream) {
    constexpr int tile_m = 8;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;
    const int block_num_y = (nrows + tile_m - 1) / tile_m;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int8_t, 1> tile_a(tile_m * tile_k, cgh);
        sycl::local_accessor<int8_t, 1> tile_b(tile_k * tile_n, cgh);
        sycl::local_accessor<float, 1> scales_a(tile_m, cgh);
        sycl::local_accessor<float, 1> scales_b(tile_n, cgh);
        sycl::local_accessor<int32_t, 1> tile_c(tile_m * tile_n, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(route_order.n_experts, block_num_y * WARP_SIZE),
                              sycl::range<2>(1, WARP_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q8_0_moe_ordered_xmx<experts_used>(
                    (const block_q8_0 *) vx_base, vy, dst_base,
                    route_order.expert_offsets, route_order.sorted_routes,
                    ncols, nrows, n_experts_used, expert_weight_stride,
                    dst_row_stride, src1_row_stride, dst_token_stride, src1_token_stride,
                    tile_a, tile_b, scales_a, scales_b, tile_c, item);
            });
    });
}

static void launch_mul_mat_vec_q8_0_moe_ordered_xmx(
    const void * vx_base, const void * vy, float * dst_base,
    const ggml_sycl_moe_route_order & route_order,
    const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const size_t dst_token_stride,
    const size_t src1_token_stride, dpct::queue_ptr stream) {
    switch (n_experts_used) {
        case 2:
            launch_mul_mat_vec_q8_0_moe_ordered_xmx_impl<2>(vx_base, vy, dst_base, route_order, ncols, nrows,
                n_experts_used, expert_weight_stride, dst_row_stride, src1_row_stride, dst_token_stride,
                src1_token_stride, stream);
            break;
        case 4:
            launch_mul_mat_vec_q8_0_moe_ordered_xmx_impl<4>(vx_base, vy, dst_base, route_order, ncols, nrows,
                n_experts_used, expert_weight_stride, dst_row_stride, src1_row_stride, dst_token_stride,
                src1_token_stride, stream);
            break;
        case 8:
            launch_mul_mat_vec_q8_0_moe_ordered_xmx_impl<8>(vx_base, vy, dst_base, route_order, ncols, nrows,
                n_experts_used, expert_weight_stride, dst_row_stride, src1_row_stride, dst_token_stride,
                src1_token_stride, stream);
            break;
        default:
            launch_mul_mat_vec_q8_0_moe_ordered_xmx_impl<0>(vx_base, vy, dst_base, route_order, ncols, nrows,
                n_experts_used, expert_weight_stride, dst_row_stride, src1_row_stride, dst_token_stride,
                src1_token_stride, stream);
            break;
    }
}

// One sub-group pass covers blocks_per_warp blocks. A tile smaller than that leaves the
// upper lanes with nothing to do: for qi/vdr == 2 the stride is 8 while a 4-block tile
// bounds the loop at 4, idling half the sub-group.
template <int qi, int vdr>
static constexpr int moe_ordered_blocks_tile() {
    constexpr int per_warp = (vdr * WARP_SIZE + qi - 1) / qi;
    return per_warp > 4 ? per_warp : 4;
}

// aligned == the caller guarantees blocks_per_row % blocks_tile == 0 and
// nrows % rows_per_wg == 0, which turns block_count into a compile-time power of two
// (so both index divisions become shifts) and removes the row bounds checks.
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl, bool aligned>
[[sycl::reqd_sub_group_size(WARP_SIZE)]]
static void mul_mat_vec_q_moe_ordered_slm(
    const void * __restrict__ vx_base,
    const void * __restrict__ vy_base,
    float * __restrict__ dst_base,
    const uint32_t * __restrict__ expert_offsets,
    const uint32_t * __restrict__ sorted_routes,
    const int ncols,
    const int nrows,
    const sycl::uint3 rows_fastdiv,      // wg_idx -> (expert, group_y) without an integer divide
    const sycl::uint3 experts_fastdiv,   // route -> (token, slot) without an integer divide
    const size_t expert_weight_stride,
    const size_t dst_row_stride,
    const size_t src1_row_stride,
    const size_t dst_token_stride,
    const size_t src1_token_stride,
    block_q_t * __restrict__ tile_x,
    block_q8_1 * __restrict__ tile_y,
    const sycl::nd_item<1> & item) {
    constexpr int rows_per_wg  = 8;
    constexpr int routes_tile  = 8;
    constexpr int blocks_tile  = moe_ordered_blocks_tile<qi, vdr>();
    constexpr int y_per_x      = qk / QK8_1;

    const int wg_idx     = item.get_group(0);
    const sycl::uint2 eg = fast_div_modulo((uint32_t) wg_idx, rows_fastdiv);
    const int expert     = (int) eg.x();
    const int group_y    = (int) eg.y();
    const int sg_id      = item.get_local_id(0) / WARP_SIZE;
    const int lane_id    = item.get_local_id(0) % WARP_SIZE;
    const int row        = group_y * (item.get_local_range(0) / WARP_SIZE) + sg_id;
    const uint32_t begin = expert_offsets[expert];
    const uint32_t end   = expert_offsets[expert + 1];

    if (begin == end) {
        return;
    }

    const auto sg = item.get_sub_group();
    const block_q_t * x = (const block_q_t *) ((const char *) vx_base + (size_t) expert * expert_weight_stride);
    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;
    const int local_id = item.get_local_linear_id();
    const int local_size = item.get_local_range(0);

    for (uint32_t route_base = begin; route_base < end; route_base += routes_tile) {
        const int route_count = sycl::min((uint32_t) routes_tile, end - route_base);
        float sums[routes_tile] = {};

        for (int block_base = 0; block_base < blocks_per_row; block_base += blocks_tile) {
            const int block_count = aligned ? blocks_tile
                                            : sycl::min(blocks_tile, blocks_per_row - block_base);
            // Walk the whole tile rather than block_count slots: a runtime divisor here
            // compiles to math.inv plus two cr0 rounding-mode writes (which serialize),
            // while blocks_tile is a compile-time power of two and folds to a shift.
            constexpr int x_slots = rows_per_wg * blocks_tile;
            for (int index = local_id; index < x_slots; index += local_size) {
                const int tile_row = index / blocks_tile;
                const int tile_block = index - tile_row * blocks_tile;
                const int global_row = group_y * rows_per_wg + tile_row;
                if (tile_block < block_count && (aligned || global_row < nrows)) {
                    tile_x[tile_row * blocks_tile + tile_block] =
                        x[global_row * blocks_per_row + block_base + tile_block];
                }
            }

            constexpr int y_per_route = blocks_tile * y_per_x;
            constexpr int y_slots      = routes_tile * y_per_route;
            for (int index = local_id; index < y_slots; index += local_size) {
                const int route_local = index / y_per_route;
                const int y_local = index - route_local * y_per_route;
                const int block_local = y_local / y_per_x;
                const int y_in_block = y_local - block_local * y_per_x;
                if (route_local >= route_count || block_local >= block_count) {
                    continue;
                }
                const uint32_t route = sorted_routes[route_base + route_local];
                const sycl::uint2 ts = fast_div_modulo(route, experts_fastdiv);
                const uint32_t token = ts.x();
                const uint32_t slot  = ts.y();
                const block_q8_1 * y = (const block_q8_1 *) ((const char *) vy_base +
                    (size_t) token * src1_token_stride + (size_t) slot * src1_row_stride);
                tile_y[(route_local * blocks_tile + block_local) * y_per_x + y_in_block] =
                    y[(block_base + block_local) * y_per_x + y_in_block];
            }

            item.barrier(sycl::access::fence_space::local_space);

            if (aligned || row < nrows) {
                // Block outer, route inner. bx and the (expensive) weight dequant inside
                // vec_dot depend only on bx and iqs, both invariant across the route loop,
                // so the dequant can be hoisted once per block instead of being redone for
                // each of the routes_tile routes sharing that block.
#pragma unroll 4
                for (int block_local = lane_id / (qi / vdr);
                     block_local < block_count; block_local += blocks_per_warp) {
                    const block_q_t * bx = &tile_x[sg_id * blocks_tile + block_local];
#pragma unroll
                    for (int route_local = 0; route_local < routes_tile; ++route_local) {
                        if (route_local >= route_count) {
                            break;
                        }
                        const block_q8_1 * by = &tile_y[(route_local * blocks_tile + block_local) * y_per_x];
                        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
                            const int iqs = elem + vdr * (lane_id % (qi / vdr));
                            sums[route_local] += vec_dot_q_sycl(bx, by, iqs);
                        }
                    }
                }
            }
            if (block_base + block_count < blocks_per_row || route_base + route_count < end) {
                item.barrier(sycl::access::fence_space::local_space);
            }
        }

#pragma unroll
        for (int route_local = 0; route_local < routes_tile; ++route_local) {
            if (route_local >= route_count) {
                break;
            }
            for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
                sums[route_local] += dpct::permute_sub_group_by_xor(sg, sums[route_local], mask);
            }
            if (lane_id == 0 && (aligned || row < nrows)) {
                const uint32_t route = sorted_routes[route_base + route_local];
                const sycl::uint2 ts = fast_div_modulo(route, experts_fastdiv);
                const uint32_t token = ts.x();
                const uint32_t slot  = ts.y();
                float * dst = (float *) ((char *) dst_base +
                    (size_t) token * dst_token_stride + (size_t) slot * dst_row_stride);
                dst[row] = sums[route_local];
            }
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl, bool aligned>
static void launch_mul_mat_vec_q_moe_ordered_impl(
    const void * vx_base, const void * vy, float * dst_base,
    const ggml_sycl_moe_route_order & route_order,
    const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const size_t dst_token_stride,
    const size_t src1_token_stride, dpct::queue_ptr stream) {
    constexpr int rows_per_wg = 8;
    const int block_num_y = (nrows + rows_per_wg - 1) / rows_per_wg;
    const int total_wgs   = route_order.n_experts * block_num_y;
    const sycl::range<1> local_range(rows_per_wg * WARP_SIZE);
    const sycl::range<1> global_range(total_wgs * local_range[0]);

    constexpr int routes_tile = 8;
    constexpr int blocks_tile = moe_ordered_blocks_tile<qi, vdr>();
    constexpr int y_per_x = qk / QK8_1;
    const sycl::uint3 experts_fastdiv = init_fastdiv_values((uint32_t) n_experts_used);
    const sycl::uint3 rows_fastdiv    = init_fastdiv_values((uint32_t) block_num_y);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<block_q_t, 1> tile_x(rows_per_wg * blocks_tile, cgh);
        sycl::local_accessor<block_q8_1, 1> tile_y(routes_tile * blocks_tile * y_per_x, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_ordered_slm<qk, qi, block_q_t, vdr, vec_dot_q_sycl, aligned>(
                    vx_base, vy, dst_base, route_order.expert_offsets, route_order.sorted_routes,
                    ncols, nrows, rows_fastdiv, experts_fastdiv, expert_weight_stride,
                    dst_row_stride, src1_row_stride, dst_token_stride, src1_token_stride,
                    get_pointer(tile_x), get_pointer(tile_y), item);
            });
    });
}

// Take the specialized kernel when the shape guarantees full block tiles and whole
// row groups; otherwise fall back to the generic one, which handles the partial tails.
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_ordered(
    const void * vx_base, const void * vy, float * dst_base,
    const ggml_sycl_moe_route_order & route_order,
    const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const size_t dst_token_stride,
    const size_t src1_token_stride, dpct::queue_ptr stream) {
    constexpr int rows_per_wg = 8;
    constexpr int blocks_tile = moe_ordered_blocks_tile<qi, vdr>();
    const int blocks_per_row  = ncols / qk;
    if (blocks_per_row % blocks_tile == 0 && nrows % rows_per_wg == 0) {
        launch_mul_mat_vec_q_moe_ordered_impl<qk, qi, block_q_t, vdr, vec_dot_q_sycl, true>(
            vx_base, vy, dst_base, route_order, ncols, nrows, n_experts_used,
            expert_weight_stride, dst_row_stride, src1_row_stride, dst_token_stride,
            src1_token_stride, stream);
    } else {
        launch_mul_mat_vec_q_moe_ordered_impl<qk, qi, block_q_t, vdr, vec_dot_q_sycl, false>(
            vx_base, vy, dst_base, route_order, ncols, nrows, n_experts_used,
            expert_weight_stride, dst_row_stride, src1_row_stride, dst_token_stride,
            src1_token_stride, stream);
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
[[sycl::reqd_sub_group_size(WARP_SIZE)]]
static void mul_mat_vec_q_moe(
    const void * __restrict__ vx_base,
    const void * __restrict__ vy_base,
    float * __restrict__ dst_base,
    const int32_t * __restrict__ ids_dev,
    const int ncols,
    const int nrows,
    const int block_num_y,
    const int n_experts_used,
    const size_t expert_weight_stride,
    const size_t dst_row_stride,
    const size_t src1_row_stride,
    const size_t ids_token_stride,
    const size_t dst_token_stride,
    const size_t src1_token_stride,
    const sycl::nd_item<1> & item_ct1) {

    auto sg = item_ct1.get_sub_group();

    const int wg_idx       = item_ct1.get_group(0);
    const int route_idx    = wg_idx / block_num_y;
    const int expert_idx   = route_idx % n_experts_used;
    const int token_idx    = route_idx / n_experts_used;
    const int group_y      = wg_idx % block_num_y;

    // Determine sub-group ID and lane ID within this work-group
    const int sg_id        = item_ct1.get_local_id(0) / WARP_SIZE;
    const int lane_id      = item_ct1.get_local_id(0) % WARP_SIZE;

    // Each sub-group calculates exactly 1 matrix row
    const int row          = (group_y * (item_ct1.get_local_range(0) / WARP_SIZE)) + sg_id;

    if (row >= nrows) {
        return;
    }

    // expert_idx is uniform per work-group, load once per sub-group
    int32_t tmp_id = 0;
    if (lane_id == 0) {
        const int32_t * ids_token = (const int32_t *) ((const char *) ids_dev + (size_t) token_idx * ids_token_stride);
        tmp_id = ids_token[expert_idx];
    }
    const int32_t i02 = sycl::group_broadcast(sg, tmp_id, 0);

    const char * vx  = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
    const char * vy  = (const char *) vy_base + (size_t) token_idx * src1_token_stride + (size_t) expert_idx * src1_row_stride;
    float *      dst = (float *) ((char *) dst_base + (size_t) token_idx * dst_token_stride + (size_t) expert_idx * dst_row_stride);

    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int row_offset = row * blocks_per_row;

    #pragma unroll 4
    for (int i = lane_id / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (lane_id % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // Full 32-lane reduction across the single sub-group
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(sg, tmp, mask);
    }

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    float * dst_base, const int ncols, const int nrows, const int n_experts_used, const int n_tokens,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const size_t ids_token_stride,
    const size_t dst_token_stride, const size_t src1_token_stride,
    const ggml_sycl_moe_route_order * route_order,
    dpct::queue_ptr stream) {

    if (route_order != nullptr) {
        if constexpr (std::is_same_v<block_q_t, block_q8_0>) {
            const int average_routes = n_tokens * n_experts_used / route_order->n_experts;
            if (g_ggml_sycl_moe_xmx && average_routes >= 8 && ggml_sycl_moe_q8_xmx_supported(stream)) {
                launch_mul_mat_vec_q8_0_moe_ordered_xmx(
                    vx_base, vy, dst_base, *route_order, ncols, nrows, n_experts_used,
                    expert_weight_stride, dst_row_stride, src1_row_stride,
                    dst_token_stride, src1_token_stride, stream);
                return;
            }
        }
        launch_mul_mat_vec_q_moe_ordered<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
            vx_base, vy, dst_base, *route_order, ncols, nrows, n_experts_used,
            expert_weight_stride, dst_row_stride, src1_row_stride,
            dst_token_stride, src1_token_stride, stream);
        return;
    }

    constexpr int rows_per_wg = 8; // Process 8 rows per work-group for high occupancy
    const int block_num_y     = (nrows + rows_per_wg - 1) / rows_per_wg;
    const int total_wgs       = n_tokens * n_experts_used * block_num_y;

    const sycl::range<1> global_range(total_wgs * rows_per_wg * WARP_SIZE);
    const sycl::range<1> local_range(rows_per_wg * WARP_SIZE);

    stream->parallel_for(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            mul_mat_vec_q_moe<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                vx_base, vy, dst_base, ids_dev,
                ncols, nrows, block_num_y, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride,
                ids_token_stride, dst_token_stride, src1_token_stride,
                item_ct1);
        });
}

bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    int                n_tokens,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    size_t             ids_token_stride,
    size_t             dst_token_stride,
    size_t             src1_token_stride,
    const ggml_sycl_moe_route_order * route_order,
    dpct::queue_ptr    stream) {
    switch (src0_type) {
        case GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, block_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, block_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        default:
            return false;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_reorder(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    float * __restrict__ dst_base, const int32_t * __restrict__ ids_dev,
    const int ncols, const int nrows,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    const size_t ids_token_stride, const size_t dst_token_stride,
    const size_t src1_token_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const int token_idx  = item_ct1.get_group(0);
    const int expert_idx = item_ct1.get_group(1);
    const int32_t * ids_token = (const int32_t *) ((const char *) ids_dev + (size_t) token_idx * ids_token_stride);
    const int i02 = ids_token[expert_idx];

    const char * vx  = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
    const char * vy  = (const char *) vy_base + (size_t) token_idx * src1_token_stride + (size_t) expert_idx * src1_row_stride;
    float *      dst = (float *) ((char *) dst_base + (size_t) token_idx * dst_token_stride + (size_t) expert_idx * dst_row_stride);

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const auto sg = item_ct1.get_sub_group();

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;

        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);

        const int           iby            = i * block_type::block_to_q8_1_ratio();
        const int8_t *      q8_1_quant_ptr = (const int8_t *) vy + iby * QK8_1;
        const sycl::half2 * q8_1_ds_ptr    = (const sycl::half2 *) ((const char *) vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());
    if (sg.leader()) {
        dst[row] = sum;
    }
}

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_reorder_ordered(
    const void * __restrict__ vx_base,
    const void * __restrict__ vy_base,
    float * __restrict__ dst_base,
    const uint32_t * __restrict__ expert_offsets,
    const uint32_t * __restrict__ sorted_routes,
    const int ncols,
    const int nrows,
    const sycl::uint3 rows_fastdiv,      // wg_idx -> (expert, group_y) without an integer divide
    const sycl::uint3 experts_fastdiv,   // route -> (token, slot) without an integer divide
    const size_t expert_weight_stride,
    const size_t dst_row_stride,
    const size_t src1_row_stride,
    const size_t dst_token_stride,
    const size_t src1_token_stride,
    const sycl::nd_item<1> & item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const int wg_idx     = item.get_group(0);
    const sycl::uint2 eg = fast_div_modulo((uint32_t) wg_idx, rows_fastdiv);
    const int expert     = (int) eg.x();
    const int group_y    = (int) eg.y();
    const int sg_id      = item.get_local_id(0) / WARP_SIZE;
    const int row        = group_y * (item.get_local_range(0) / WARP_SIZE) + sg_id;
    const uint32_t begin = expert_offsets[expert];
    const uint32_t end   = expert_offsets[expert + 1];

    if (row >= nrows || begin == end) {
        return;
    }

    const auto sg = item.get_sub_group();
    const char * vx = (const char *) vx_base + (size_t) expert * expert_weight_stride;
    const int blocks_per_row = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int nblocks = nrows * blocks_per_row;

    for (uint32_t route_pos = begin; route_pos < end; ++route_pos) {
        const uint32_t route = sorted_routes[route_pos];
        const sycl::uint2 ts = fast_div_modulo(route, experts_fastdiv);
        const uint32_t token = ts.x();
        const uint32_t slot  = ts.y();
        const char * vy = (const char *) vy_base + (size_t) token * src1_token_stride + (size_t) slot * src1_row_stride;
        float * dst = (float *) ((char *) dst_base + (size_t) token * dst_token_stride + (size_t) slot * dst_row_stride);
        float sum = 0.0f;

        for (int i = sg.get_local_linear_id() / block_elements_per_subgroup;
             i < blocks_per_row; i += blocks_per_subgroup) {
            const int ibx = row * blocks_per_row + i;
            const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
            const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
            const int iby = i * block_type::block_to_q8_1_ratio();
            const int8_t * q8_1_quant_ptr = (const int8_t *) vy + iby * QK8_1;
            const sycl::half2 * q8_1_ds_ptr =
                (const sycl::half2 *) (vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
            for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
                const int iqs = elem + block_traits::vdr_mmvq *
                    (sg.get_local_linear_id() % block_elements_per_subgroup);
                sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }

        sum = sycl::reduce_over_group(sg, sum, std::plus<>());
        if (sg.leader()) {
            dst[row] = sum;
        }
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_reorder_ordered(
    const void * vx_base, const void * vy, float * dst_base,
    const ggml_sycl_moe_route_order & route_order,
    const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const size_t dst_token_stride,
    const size_t src1_token_stride, dpct::queue_ptr stream) {
    constexpr int rows_per_wg = 8;
    const int block_num_y = (nrows + rows_per_wg - 1) / rows_per_wg;
    const int total_wgs   = route_order.n_experts * block_num_y;
    const sycl::range<1> local_range(rows_per_wg * WARP_SIZE);
    const sycl::range<1> global_range(total_wgs * local_range[0]);
    const sycl::uint3 rows_fastdiv    = init_fastdiv_values((uint32_t) block_num_y);
    const sycl::uint3 experts_fastdiv = init_fastdiv_values((uint32_t) n_experts_used);

    stream->parallel_for(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            mul_mat_vec_q_moe_reorder_ordered<reorder_vec_dot_q_sycl>(
                vx_base, vy, dst_base, route_order.expert_offsets, route_order.sorted_routes,
                ncols, nrows, rows_fastdiv, experts_fastdiv, expert_weight_stride,
                dst_row_stride, src1_row_stride, dst_token_stride, src1_token_stride, item);
        });
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_reorder(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    float * dst_base, const int ncols, const int nrows, const int n_experts_used, const int n_tokens,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride, const size_t ids_token_stride,
    const size_t dst_token_stride, const size_t src1_token_stride,
    const ggml_sycl_moe_route_order * route_order,
    dpct::queue_ptr stream) {
    if (route_order != nullptr) {
        launch_mul_mat_vec_q_moe_reorder_ordered<reorder_vec_dot_q_sycl>(
            vx_base, vy, dst_base, *route_order, ncols, nrows, n_experts_used,
            expert_weight_stride, dst_row_stride, src1_row_stride,
            dst_token_stride, src1_token_stride, stream);
        return;
    }
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums((unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride,
                    ids_token_stride, dst_token_stride, src1_token_stride, item);
            });
    });
}

bool ggml_sycl_mul_mat_vec_q_id_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    int                n_tokens,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    size_t             ids_token_stride,
    size_t             dst_token_stride,
    size_t             src1_token_stride,
    const ggml_sycl_moe_route_order * route_order,
    dpct::queue_ptr    stream) {
    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, route_order, stream);
            return true;
        default:
            return false;
    }
}

template <typename reorder_vec_dot_q_sycl, int ncols_dst, int rows_per_sg>
static void launch_mul_mat_vec_q_reorder_glu_impl(const void * vx, const void * vgate, const void * vy, float * dst,
                                             const int ncols, const int nrows, const int stride_col_y_bytes,
                                             const int stride_col_dst, const ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;

    const int            block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups * rows_per_sg);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ true,
                                                        rows_per_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op,
                                 nd_item);
                         });
    });
}

template <typename reorder_vec_dot_q_sycl, int ncols_dst>
static void launch_mul_mat_vec_q_reorder_glu(const void * vx, const void * vgate, const void * vy, float * dst,
                                             const int ncols, const int nrows, const int stride_col_y_bytes,
                                             const int stride_col_dst, const ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    constexpr int rows_per_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 && ncols_dst <= 4
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, rows_per_sg>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
}

bool ggml_sycl_mul_mat_vec_q_glu_reorder(enum ggml_type src0_type, enum ggml_glu_op glu_op, const void * vx,
                                         const void * vgate, const void * vy, float * dst, int ncols, int nrows,
                                         int ncols_dst, int stride_col_y_bytes, int stride_col_dst,
                                         dpct::queue_ptr stream) {
    if (src0_type != GGML_TYPE_Q4_K) {
        return false;
    }
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }

    using vec_dot = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>;

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 1>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 2:
            if (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
            } else {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 1>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
            }
            return true;
        case 3:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 3>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 4:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 4>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 5:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 5>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 6:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 6>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 7:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 7>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        case 8:
            launch_mul_mat_vec_q_reorder_glu<vec_dot, 8>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                         stride_col_dst, glu_op, stream);
            return true;
        default:
            return false;
    }
}
