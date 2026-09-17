#include "fused-gemm.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace mx = sycl::ext::oneapi::experimental::matrix;

// XMX f16 tile: 8x16 (A) times 16x16 (B) into an 8x16 f32 accumulator
static constexpr int FG_TM = 8;
static constexpr int FG_TN = 16;
static constexpr int FG_TK = 16;

// a sub-group owns 16 rows (2 A tiles), stages one iq4_nl block per lane per step and
// walks its own K range; the sub-groups of a work-group are summed at the end
static constexpr int FG_SG_ROWS = 2 * FG_TM;
static constexpr int FG_BK      = QK4_NL;
static constexpr int FG_BN      = 2 * FG_TN;
static constexpr int FG_KSPLIT  = 4;
static constexpr int FG_WG_SIZE = FG_KSPLIT * WARP_SIZE;
static_assert(FG_SG_ROWS == WARP_SIZE, "the A stage maps one lane to one row");

static bool fused_gemm_f16_supported(dpct::queue_ptr stream) {
    try {
        const auto combinations = stream->get_device().get_info<
            sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        for (const auto & c : combinations) {
            if (c.atype == mx::matrix_type::fp16 && c.btype == mx::matrix_type::fp16 &&
                c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
                (c.max_msize >= FG_TM || c.msize == FG_TM) &&
                (c.max_nsize >= FG_TN || c.nsize == FG_TN) &&
                (c.max_ksize >= FG_TK || c.ksize == FG_TK)) {
                return true;
            }
        }
    } catch (const sycl::exception &) {
    }
    return false;
}

// src1 [N][K] -> VNNI packed [K/2][Npad][2] so B tiles load straight from global memory
static void fused_gemm_pack_b(const sycl::half * y, sycl::half * packed, int N, int Npad, int K, dpct::queue_ptr stream) {
    const int kpairs = K / 2;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kpairs), [=](sycl::id<1> id) {
        const int idx = id[0];
        const int n   = idx / kpairs;
        const int kp  = idx - n * kpairs;
        sycl::half v0 = (sycl::half) 0.0f;
        sycl::half v1 = (sycl::half) 0.0f;
        if (n < N) {
            const sycl::half * src = y + (size_t) n * K + 2 * kp;
            v0 = src[0];
            v1 = src[1];
        }
        sycl::half * out = packed + ((size_t) kp * Npad + n) * 2;
        out[0] = v0;
        out[1] = v1;
    });
}

[[sycl::reqd_sub_group_size(WARP_SIZE)]]
static void fused_dequant_gemm_iq4_nl(
    const block_iq4_nl * __restrict__ x,
    const sycl::half * __restrict__ packed_b,
    float * __restrict__ dst,
    const int M, const int N, const int Npad, const int K, const int ldd,
    sycl::local_accessor<sycl::half, 1> tile_a,
    sycl::local_accessor<float, 1> tile_c,
    const sycl::nd_item<2> & item) {
    const auto sg     = item.get_sub_group();
    const int  sg_id  = sg.get_group_id()[0];
    const int  lane   = sg.get_local_id()[0];
    const int  n0     = item.get_group(0) * FG_BN;
    const int  m0     = item.get_group(1) * FG_SG_ROWS;
    const int  nblk   = K / QK4_NL;
    const int  a_base = sg_id * FG_SG_ROWS * FG_BK;
    const int  c_base = sg_id * FG_SG_ROWS * FG_BN;

    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, FG_TM, FG_TN> acc[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
        }
    }

    const int  row    = m0 + lane;
    const bool row_ok = row < M;
    const block_iq4_nl * xrow = x + (size_t) (row_ok ? row : 0) * nblk;
    sycl::half2 * a = (sycl::half2 *) &tile_a[a_base + lane * FG_BK];

    const auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                sycl::access::decorated::no>(packed_b);
    const int b_stride = Npad * 2;

    const int kb_begin = (sg_id * nblk) / FG_KSPLIT;
    const int kb_end   = ((sg_id + 1) * nblk) / FG_KSPLIT;
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        // A: lane = row, one block of 32 values with the scale folded in
        if (row_ok) {
            const block_iq4_nl blk = xrow[kb];
            const float d = (float) blk.d;
#pragma unroll
            for (int j = 0; j < QK4_NL / 2; j += 2) {
                const uint8_t q0 = blk.qs[j];
                const uint8_t q1 = blk.qs[j + 1];
                a[j / 2]     = sycl::half2((sycl::half) (d * kvalues_iq4nl[q0 & 0xf]), (sycl::half) (d * kvalues_iq4nl[q1 & 0xf]));
                a[j / 2 + 8] = sycl::half2((sycl::half) (d * kvalues_iq4nl[q0 >> 4]),  (sycl::half) (d * kvalues_iq4nl[q1 >> 4]));
            }
        } else {
#pragma unroll
            for (int j = 0; j < FG_BK / 2; ++j) {
                a[j] = sycl::half2((sycl::half) 0.0f, (sycl::half) 0.0f);
            }
        }
        sycl::group_barrier(sg);

#pragma unroll
        for (int kt = 0; kt < FG_BK / FG_TK; ++kt) {
            const int kp0 = (kb * FG_BK + kt * FG_TK) / 2;
            mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, FG_TK, FG_TN, mx::layout::ext_intel_packed> sub_b[2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (size_t) kp0 * b_stride + (n0 + nt * FG_TN) * 2, b_stride);
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, FG_TM, FG_TK, mx::layout::row_major> sub_a;
                mx::joint_matrix_load(sg, sub_a,
                    tile_a.get_multi_ptr<sycl::access::decorated::no>() + a_base + (mt * FG_TM) * FG_BK + kt * FG_TK,
                    FG_BK);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                }
            }
        }
        // the next step overwrites tile_a
        sycl::group_barrier(sg);
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            mx::joint_matrix_store(sg, acc[mt][nt],
                tile_c.get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * FG_TM) * FG_BN + nt * FG_TN,
                FG_BN, mx::layout::row_major);
        }
    }
    sycl::group_barrier(item.get_group());

    // sum the K splits; consecutive lanes write consecutive rows of one dst column
    for (int idx = item.get_local_linear_id(); idx < FG_SG_ROWS * FG_BN; idx += FG_WG_SIZE) {
        const int r = idx % FG_SG_ROWS;
        const int c = idx / FG_SG_ROWS;
        const int m = m0 + r;
        const int n = n0 + c;
        if (m < M && n < N) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < FG_KSPLIT; ++s) {
                sum += tile_c[s * FG_SG_ROWS * FG_BN + r * FG_BN + c];
            }
            dst[(size_t) n * ldd + m] = sum;
        }
    }
}

bool ggml_sycl_fused_dequant_gemm_f16(ggml_type src0_type, const void * src0, const sycl::half * src1_f16, float * dst,
                                      int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                                      dpct::queue_ptr stream) {
    if (src0_type != GGML_TYPE_IQ4_NL || K % QK4_NL != 0 || M <= 0 || N <= 0 || K <= 0) {
        return false;
    }
    // every FG_BN columns dequantize A again, so wide N is left to the library GEMM
    if (N > 2 * FG_BN) {
        return false;
    }
    if (M > INT32_MAX || N > INT32_MAX || K > INT32_MAX || ldd > INT32_MAX) {
        return false;
    }
    if (!fused_gemm_f16_supported(stream)) {
        return false;
    }

    const int64_t groups_n = (N + FG_BN - 1) / FG_BN;
    const int64_t groups_m = (M + FG_SG_ROWS - 1) / FG_SG_ROWS;
    const int     Npad     = (int) (groups_n * FG_BN);

    ggml_sycl_pool_alloc<sycl::half> packed_b(pool, (size_t) K * Npad);
    fused_gemm_pack_b(src1_f16, packed_b.get(), (int) N, Npad, (int) K, stream);

    const sycl::half * packed = packed_b.get();
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> tile_a(FG_KSPLIT * FG_SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>      tile_c(FG_KSPLIT * FG_SG_ROWS * FG_BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * FG_WG_SIZE), sycl::range<2>(1, FG_WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                fused_dequant_gemm_iq4_nl((const block_iq4_nl *) src0, packed, dst,
                                          (int) M, (int) N, Npad, (int) K, (int) ldd, tile_a, tile_c, item);
            });
    });
    return true;
}
