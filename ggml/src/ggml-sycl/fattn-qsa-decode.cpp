#include "fattn-qsa-decode.hpp"
#include "kq-mask-bits.hpp"
#include "kv-soa.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// A QSA query row attends to the ~2k cells its indexer listed, whatever the context. The dense
// path gets there by writing the list into a full-width mask, gathering or converting K/V to f16
// and running the tile kernel, which on qwen4exp (24 query heads on 2 KV heads, D=256) reads every
// K/V row three times (four of the twelve query heads of a KV head per work-group), re-reads it
// from shared memory once per head and spills: ~55-88 us per layer at decode, any context. Here a
// work-group takes one KV head and R listed slots of one row, reads each K/V row once as stored,
// and leaves one partial (max, sum, D outputs) per query head; a second launch merges the
// partials of a row. A q8_0 cache runs on the matrix engines (qsa_dec_attend_xmx_kernel, ~11 us
// at a 2051-cell list), an f16 one on the vector units (qsa_dec_attend_kernel).
//
// Semantics are the dense graph's: a listed cell counts if it is a cell (the list may hold the
// n_kv sentinel and the dump rows past it) and the causal mask lets the row see it, and its score
// gets the mask value added. The list names a cell at most once (build_qsa_sel maps every dead
// slot to a row of its own past n_kv). A row that sees nothing gives 0, as the CPU reference does.

namespace syclexp = sycl::ext::oneapi::experimental;

// how a head row of the cache is stored
enum qsa_dec_kv {
    QSA_DEC_F16,
    QSA_DEC_Q8_0,     // canonical blocks: [d qs[32]] x 8, a block's quants 2-byte aligned
    QSA_DEC_Q8_0_SOA, // one GGML_SYCL_KV_SOA_SPAN: [qs[256]][d x 8], everything 16-byte aligned
};

static constexpr int QSA_DEC_R        = 64;  // listed slots per work-group, vector kernel
static constexpr int QSA_DEC_R_XMX    = 64;  // and matrix-engine kernel: one 8-cell tile per thread
static constexpr int QSA_DEC_MAX_ROWS = 8;  // decode, MTP verify batches
static constexpr int QSA_DEC_PART     = 4;  // floats after the D outputs of a partial: max, sum, pad to 16 B

struct qsa_dec_args {
    const char * Q;
    size_t       q_nb1, q_nb2, q_nb3;
    const char * K;
    size_t       k_nb1, k_nb2, k_nb3;
    const char * V;
    size_t       v_nb1, v_nb2, v_nb3;
    const char * mask;
    size_t       m_nb1, m_nb3; // f16 mask; a packed one uses m_row
    size_t       m_row;
    int          m_ne1; // rows of one stream in the mask, the token count padded
    const char * idx;
    size_t       i_nb1, i_nb2;
    float *      part;
    float *      dst;
    int          n_kv;
    int          width;
    int          n_tps;
    int          n_head_kv;
    int          n_chunks;
    float        scale;
};

template <int D, int G, int R, int KV, bool MBITS>
struct qsa_dec_attend_kernel {
    qsa_dec_args a;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        constexpr int NT  = D / 32;
        constexpr int SPT = R / NT;
        constexpr bool KQ8 = KV != QSA_DEC_F16;
        constexpr bool SOA = KV == QSA_DEC_Q8_0_SOA;
        constexpr int  QB  = 34 * D / 32; // bytes of one q8_0 head row
        // [G][D] q as half | [G][R] scores | [R] cells, -1 where the slot does not count
        constexpr int SLM_Q  = 0;
        constexpr int SLM_S  = SLM_Q + G * D * 2;
        constexpr int SLM_C  = SLM_S + G * R * 4;
        constexpr int SLM_SZ = SLM_C + R * 4;
        slm_init<SLM_SZ>();

        const int i     = it.get_local_id(0);
        int       w     = it.get_group(0);
        const int chunk = w % a.n_chunks;
        w /= a.n_chunks;
        const int g = w % a.n_head_kv;
        w /= a.n_head_kv;
        const int t = w % a.n_tps;
        const int s = w / a.n_tps;
        const int r = s * a.n_tps + t;

        // q of the G heads, features [32i, 32i+32) per thread, into shared memory as half
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float *       qp = (const float *) (a.Q + t * a.q_nb1 + (size_t) (g * G + h) * a.q_nb2 + s * a.q_nb3);
            simd<sycl::half, 32> qh = block_load<float, 32>(qp + 32 * i, element_aligned_tag{});
            slm_block_store<sycl::half, 32>(SLM_Q + (h * D + 32 * i) * 2, qh);
        }

        // the list entries of this thread's slots, and whether each one counts
        const int          slot0 = chunk * R + i * SPT;
        const int32_t *    lst   = (const int32_t *) (a.idx + t * a.i_nb1 + s * a.i_nb2);
        simd<uint32_t, SPT> soff(0, sizeof(int32_t));
        simd<int32_t, SPT>  sl(0, 1);
        simd_mask<SPT>      in_list = (sl + slot0) < a.width;
        simd<int32_t, SPT>  cell    = gather<int32_t, SPT>(lst + slot0, soff, in_list);
        simd_mask<SPT>      ok      = in_list & (cell >= 0) & (cell < a.n_kv);
        cell.merge(simd<int32_t, SPT>(0), !ok);
        simd<float, SPT> madd = 0.0f;
        if constexpr (MBITS) {
            const uint32_t *    mrow = (const uint32_t *) (a.mask + (size_t) (s * a.m_ne1 + t) * a.m_row);
            simd<uint32_t, SPT> word = gather<uint32_t, SPT>(mrow, simd<uint32_t, SPT>(cell >> 5) * 4, ok);
            ok &= ((word >> simd<uint32_t, SPT>(cell & 31)) & 1) != 0;
        } else {
            const sycl::half *    mrow = (const sycl::half *) (a.mask + t * a.m_nb1 + s * a.m_nb3);
            simd<sycl::half, SPT> mh   = gather<sycl::half, SPT>(mrow, simd<uint32_t, SPT>(cell) * 2, ok);
            madd                      = mh;
            ok &= madd > -INFINITY;
            madd.merge(simd<float, SPT>(0.0f), !ok);
        }

        barrier();

        simd<sycl::half, D> q[G];
#pragma unroll
        for (int h = 0; h < G; ++h) {
#pragma unroll
            for (int u = 0; u < D / 128; ++u) {
                q[h].template select<128, 1>(128 * u) = slm_block_load<sycl::half, 128>(SLM_Q + (h * D + 128 * u) * 2);
            }
        }

        simd<float, SPT> sc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            sc[h] = -INFINITY;
        }
        // every K row of this thread's slots in flight at once; a slot that does not count reads
        // cell 0 and is never scored
        constexpr int KW = KQ8 ? QB / 4 : D / 2; // dwords of one head row
        simd<uint32_t, KW> krow[SPT];
#pragma unroll
        for (int j = 0; j < SPT; ++j) {
            const uint32_t * kp = (const uint32_t *) (a.K + (size_t) cell[j] * a.k_nb1 + (size_t) g * a.k_nb2 +
                                                      (size_t) s * a.k_nb3);
            if constexpr (KQ8) {
                static_assert(QB == 272, "one 256-wide q8_0 head row: 64 + 4 dwords");
                krow[j].template select<64, 1>(0) = block_load<uint32_t, 64>(kp, overaligned_tag<16>{});
                krow[j].template select<4, 1>(64) = block_load<uint32_t, 4>(kp + 64, overaligned_tag<16>{});
            } else {
#pragma unroll
                for (int u = 0; u < KW / 64; ++u) {
                    krow[j].template select<64, 1>(64 * u) = block_load<uint32_t, 64>(kp + 64 * u, overaligned_tag<16>{});
                }
            }
        }
#pragma unroll
        for (int j = 0; j < SPT; ++j) {
            if (!ok[j]) {
                continue;
            }
            simd<float, 32> acc[G];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                acc[h] = 0.0f;
            }
#pragma unroll
            for (int b = 0; b < D / 32; ++b) {
                simd<float, 32> kf;
                if constexpr (SOA) {
                    auto                bytes = krow[j].template bit_cast_view<uint8_t>();
                    simd<uint16_t, 1>   db    = krow[j].template bit_cast_view<uint16_t>().template select<1, 1>(D / 2 + b);
                    simd<sycl::half, 1> dh    = db.template bit_cast_view<sycl::half>();
                    const float         d     = simd<float, 1>(dh)[0];
                    simd<int8_t, 32>    qs    = bytes.template select<32, 1>(32 * b).template bit_cast_view<int8_t>();
                    kf                        = simd<float, 32>(qs) * d;
                } else if constexpr (KQ8) {
                    auto                bytes = krow[j].template bit_cast_view<uint8_t>();
                    simd<uint8_t, 2>    db    = bytes.template select<2, 1>(34 * b);
                    simd<sycl::half, 1> dh    = db.template bit_cast_view<sycl::half>();
                    const float         d     = simd<float, 1>(dh)[0];
                    simd<int8_t, 32>    qs    = bytes.template select<32, 1>(34 * b + 2).template bit_cast_view<int8_t>();
                    kf                        = simd<float, 32>(qs) * d;
                } else {
                    kf = krow[j].template bit_cast_view<sycl::half>().template select<32, 1>(32 * b);
                }
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    acc[h] += simd<float, 32>(q[h].template select<32, 1>(32 * b)) * kf;
                }
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                sc[h][j] = reduce<float>(acc[h], std::plus<>()) * a.scale + madd[j];
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            slm_block_store<float, SPT>(SLM_S + (h * R + i * SPT) * 4, sc[h]);
        }
        simd<int32_t, SPT> cl = cell;
        cl.merge(simd<int32_t, SPT>(-1), !ok);
        slm_block_store<int32_t, SPT>(SLM_C + i * SPT * 4, cl);

        barrier();

        // phase 2: features [32i, 32i+32) of V for all G heads over the R slots
        simd<int32_t, R> cells = slm_block_load<int32_t, R>(SLM_C);
        simd<float, R>   p[G];
        float            mx[G], sm[G];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            simd<float, R> sh = slm_block_load<float, R>(SLM_S + h * R * 4);
            mx[h]             = hmax<float>(sh);
            if (mx[h] == -INFINITY) {
                p[h] = 0.0f;
            } else {
                p[h] = exp(sh - mx[h]);
            }
            sm[h] = reduce<float>(p[h], std::plus<>());
        }

        simd<float, 32> o[G];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            o[h] = 0.0f;
        }
        // 17 halves: the scale and 32 quants; 32 lanes, the tail ones repeat the last
        simd<uint32_t, 32> voff(0, 2);
        voff.merge(simd<uint32_t, 32>(32), voff > 32);
        // 16 slots of V in flight at a time; a slot that does not count reads cell 0 and adds 0
        constexpr int VB = 16;
#pragma unroll
        for (int j0 = 0; j0 < R; j0 += VB) {
            simd<int32_t, VB> cv = cells.template select<VB, 1>(j0);
            simd_mask<VB>     vv = cv >= 0;
            cv.merge(simd<int32_t, VB>(0), !vv);
            simd<uint16_t, 32> vraw[VB];
#pragma unroll
            for (int jj = 0; jj < VB; ++jj) {
                const char * vp = a.V + (size_t) cv[jj] * a.v_nb1 + (size_t) g * a.v_nb2 + (size_t) s * a.v_nb3;
                if constexpr (SOA) {
                    // quants of block i, then all eight scales; the scale is picked below
                    vraw[jj].template select<16, 1>(0) =
                        block_load<uint16_t, 16>((const uint16_t *) (vp + 32 * i), overaligned_tag<16>{});
                    vraw[jj].template select<8, 1>(16) = block_load<uint16_t, 8>((const uint16_t *) (vp + D), overaligned_tag<16>{});
                } else if constexpr (KQ8) {
                    vraw[jj] = gather<uint16_t, 32>((const uint16_t *) (vp + 34 * i), voff);
                } else {
                    vraw[jj] = block_load<uint16_t, 32>((const uint16_t *) vp + 32 * i, overaligned_tag<16>{});
                }
            }
#pragma unroll
            for (int jj = 0; jj < VB; ++jj) {
                simd<float, 32> vf;
                if constexpr (SOA) {
                    simd<uint16_t, 8>   d8 = vraw[jj].template select<8, 1>(16);
                    simd<sycl::half, 8> dh = d8.template bit_cast_view<sycl::half>();
                    const float         d  = simd<float, 8>(dh)[i];
                    simd<int8_t, 32>    qs = vraw[jj].template select<16, 1>(0).template bit_cast_view<int8_t>();
                    vf                     = simd<float, 32>(qs) * d;
                } else if constexpr (KQ8) {
                    simd<sycl::half, 1> dh = vraw[jj].template select<1, 1>(0).template bit_cast_view<sycl::half>();
                    const float         d  = simd<float, 1>(dh)[0];
                    simd<int8_t, 32>    qs = vraw[jj].template select<16, 1>(1).template bit_cast_view<int8_t>();
                    vf                     = simd<float, 32>(qs) * d;
                } else {
                    vf = vraw[jj].template bit_cast_view<sycl::half>();
                }
                // an uncounted slot read whatever cell 0 holds, which may not be finite
                if (!vv[jj]) {
                    vf = 0.0f;
                }
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    o[h] += (float) p[h][j0 + jj] * vf;
                }
            }
        }

        // partial (row r, head g*G + h, chunk): D outputs, then max and sum
        const int n_head = a.n_head_kv * G;
#pragma unroll
        for (int h = 0; h < G; ++h) {
            float * pp = a.part + ((size_t) (r * n_head + g * G + h) * a.n_chunks + chunk) * (D + QSA_DEC_PART);
            block_store<float, 32>(pp + 32 * i, o[h]);
            if (i == 0) {
                simd<float, 4> ml = 0.0f;
                ml[0]             = mx[h];
                ml[1]             = sm[h];
                block_store<float, 4>(pp + D, ml);
            }
        }
    }

    auto get(syclexp::properties_tag) const { return syclexp::properties{ sycl::ext::intel::experimental::grf_size<256> }; }
};

// the V slice [32i, 32i+32) of 16 listed cells, as stored; a negative cell reads cell 0
template <int D, bool SOA>
ESIMD_INLINE void qsa_dec_load_v(sycl::ext::intel::esimd::simd<uint16_t, 32> (&vr)[16],
                                 sycl::ext::intel::esimd::simd<int32_t, 16> cv, const qsa_dec_args & a, int g, int s,
                                 int i, const sycl::ext::intel::esimd::simd<uint32_t, 32> & voff) {
    using namespace sycl::ext::intel::esimd;
    cv.merge(simd<int32_t, 16>(0), cv < 0);
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        const char * vp = a.V + (size_t) cv[c] * a.v_nb1 + (size_t) g * a.v_nb2 + (size_t) s * a.v_nb3;
        if constexpr (SOA) {
            vr[c].template select<16, 1>(0) = block_load<uint16_t, 16>((const uint16_t *) (vp + 32 * i), overaligned_tag<16>{});
            vr[c].template select<8, 1>(16) = block_load<uint16_t, 8>((const uint16_t *) (vp + D), overaligned_tag<16>{});
        } else {
            vr[c] = gather<uint16_t, 32>((const uint16_t *) (vp + 34 * i), voff);
        }
    }
}

// The q8_0 cache on the matrix engines. A work-group of NT = D/32 threads takes R = 16*NT listed
// slots of one row and one KV head:
//   staging: thread i quantizes block i of the G query heads to q8_0, as ggml quantizes the
//            activation of a q8_0 dot, and stores it in the VNNI order of a dpas B operand,
//            the heads padded to HP = 16
//   phase 1: thread i scores cells [16i, 16i+16): per block, two int8 dpas of 8 cells x 32
//            features against the HP heads, the cells' quants being the A operand as stored;
//            the int32 dots are scaled by the cell's and the head's block scale
//   phase 2: thread i takes features [32i, 32i+32): per 16 cells, V in f16 in VNNI pairs of
//            cells times the softmax weights of heads 0-7 and 8-(G-1), four f16 dpas
template <int D, int G, int R, bool SOA, bool MBITS>
struct qsa_dec_attend_xmx_kernel {
    qsa_dec_args a;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        namespace xmx = sycl::ext::intel::esimd::xmx;
        constexpr int NT  = D / 32;
        constexpr int NB  = D / 32;
        constexpr int HP  = 16;
        constexpr int CPT = R / NT;
        constexpr int G1  = G - 8; // heads of the second dpas row block
        constexpr int QB  = 34 * NB;
        constexpr int NTG = CPT / 8; // 8-cell dpas tiles per thread in phase 1
        static_assert(CPT % 8 == 0 && R % 16 == 0, "phase 1 runs 8-cell dpas tiles, phase 2 16-cell ones");
        static_assert(G > 8 && G <= HP, "two dpas row blocks of 8 and G-8 query heads");
        static_assert(QB == 272, "one 256-wide q8_0 head row: 64 + 4 dwords");
        // [NB][8][HP] dwords of int8 q in VNNI order | [NB][HP] q scales | [G][R] scores | [R] cells
        constexpr int SLM_QT = 0;
        constexpr int SLM_DQ = SLM_QT + NB * 8 * HP * 4;
        constexpr int SLM_S  = SLM_DQ + NB * HP * 4;
        constexpr int SLM_C  = SLM_S + G * R * 4;
        constexpr int SLM_SZ = SLM_C + R * 4;
        slm_init<SLM_SZ>();

        const int i     = it.get_local_id(0);
        int       w     = it.get_group(0);
        const int chunk = w % a.n_chunks;
        w /= a.n_chunks;
        const int g = w % a.n_head_kv;
        w /= a.n_head_kv;
        const int t = w % a.n_tps;
        const int s = w / a.n_tps;
        const int r = s * a.n_tps + t;

        // staging: block i of every head, quantized as quantize_row_q8_0 does (round half away
        // from zero with 1/d, the dot scaled by d rounded to f16)
        {
            const simd<uint32_t, 8> kk_off(0, HP * 4);
#pragma unroll
            for (int h = 0; h < HP; ++h) {
                simd<uint32_t, 8> qd = 0;
                float             dq = 0.0f;
                if (h < G) {
                    const float *   qp = (const float *) (a.Q + t * a.q_nb1 + (size_t) (g * G + h) * a.q_nb2 + s * a.q_nb3);
                    simd<float, 32> qf = block_load<float, 32>(qp + 32 * i, element_aligned_tag{});
                    const float     d  = hmax<float>(abs(qf)) / 127.0f;
                    const float     id = d != 0.0f ? 1.0f / d : 0.0f;
                    simd<float, 32> x  = qf * id;
                    simd<float, 32> hf = 0.5f;
                    hf.merge(simd<float, 32>(-0.5f), x < 0.0f);
                    simd<int32_t, 32> qi = x + hf; // truncates: round half away from zero
                    simd<int8_t, 32>  q8 = qi;
                    qd                   = q8.template bit_cast_view<uint32_t>();
                    simd<sycl::half, 1> dh = d;
                    dq                     = simd<float, 1>(dh)[0];
                }
                slm_scatter<uint32_t, 8>(SLM_QT + (i * 8 * HP + h) * 4 + kk_off, qd);
                slm_scalar_store<float>(SLM_DQ + (i * HP + h) * 4, dq);
            }
        }

        // the list entries of this thread's cells, and whether each one counts
        const int           slot0 = chunk * R + i * CPT;
        const int32_t *     lst   = (const int32_t *) (a.idx + t * a.i_nb1 + s * a.i_nb2);
        simd<uint32_t, CPT> soff(0, sizeof(int32_t));
        simd<int32_t, CPT>  sl(0, 1);
        simd_mask<CPT>      in_list = (sl + slot0) < a.width;
        simd<int32_t, CPT>  cell    = gather<int32_t, CPT>(lst + slot0, soff, in_list);
        simd_mask<CPT>      ok      = in_list & (cell >= 0) & (cell < a.n_kv);
        cell.merge(simd<int32_t, CPT>(0), !ok);
        simd<float, CPT> madd = 0.0f;
        if constexpr (MBITS) {
            const uint32_t *    mrow = (const uint32_t *) (a.mask + (size_t) (s * a.m_ne1 + t) * a.m_row);
            simd<uint32_t, CPT> word = gather<uint32_t, CPT>(mrow, simd<uint32_t, CPT>(cell >> 5) * 4, ok);
            ok &= ((word >> simd<uint32_t, CPT>(cell & 31)) & 1) != 0;
        } else {
            const sycl::half *    mrow = (const sycl::half *) (a.mask + t * a.m_nb1 + s * a.m_nb3);
            simd<sycl::half, CPT> mh   = gather<sycl::half, CPT>(mrow, simd<uint32_t, CPT>(cell) * 2, ok);
            madd                      = mh;
            ok &= madd > -INFINITY;
            madd.merge(simd<float, CPT>(0.0f), !ok);
        }

        // every K row of this thread's cells in flight at once; a cell that does not count reads
        // cell 0 and is never scored
        simd<uint32_t, QB / 4> krow[CPT];
#pragma unroll
        for (int n = 0; n < CPT; ++n) {
            const uint32_t * kp = (const uint32_t *) (a.K + (size_t) cell[n] * a.k_nb1 + (size_t) g * a.k_nb2 +
                                                      (size_t) s * a.k_nb3);
            krow[n].template select<64, 1>(0) = block_load<uint32_t, 64>(kp, overaligned_tag<16>{});
            krow[n].template select<4, 1>(64) = block_load<uint32_t, 4>(kp + 64, overaligned_tag<16>{});
        }

        barrier();

        // phase 1: [cell][head] scores of this thread's 16 cells
        simd<float, CPT * HP> S = 0.0f;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            simd<uint32_t, 8 * HP> bq;
            bq.template select<64, 1>(0)  = slm_block_load<uint32_t, 64>(SLM_QT + b * 8 * HP * 4);
            bq.template select<64, 1>(64) = slm_block_load<uint32_t, 64>(SLM_QT + b * 8 * HP * 4 + 256);
            simd<int8_t, 32 * HP> B  = bq.template bit_cast_view<int8_t>();
            simd<float, HP>       dq = slm_block_load<float, HP>(SLM_DQ + b * HP * 4);
#pragma unroll
            for (int tg = 0; tg < NTG; ++tg) {
                simd<int8_t, 8 * 32> A;
                simd<float, 8>       dk;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    auto bytes = krow[tg * 8 + m].template bit_cast_view<uint8_t>();
                    auto u16   = krow[tg * 8 + m].template bit_cast_view<uint16_t>();
                    if constexpr (SOA) {
                        A.template select<32, 1>(32 * m) = bytes.template select<32, 1>(32 * b).template bit_cast_view<int8_t>();
                        simd<uint16_t, 1> db = u16.template select<1, 1>(D / 2 + b);
                        dk[m] = simd<float, 1>(db.template bit_cast_view<sycl::half>())[0];
                    } else {
                        A.template select<32, 1>(32 * m) = bytes.template select<32, 1>(34 * b + 2).template bit_cast_view<int8_t>();
                        simd<uint16_t, 1> db = u16.template select<1, 1>(17 * b);
                        dk[m] = simd<float, 1>(db.template bit_cast_view<sycl::half>())[0];
                    }
                }
                simd<int32_t, 8 * HP> C = xmx::dpas<8, 8, int32_t>(B, A);
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    S.template select<HP, 1>((tg * 8 + m) * HP) += simd<float, HP>(C.template select<HP, 1>(m * HP)) * (dq * (float) dk[m]);
                }
            }
        }

        // scores to shared memory as [head][slot], -inf where the cell does not count
        {
            simd<uint32_t, HP> hoff(0, R * 4);
            simd_mask<HP>      hm = simd<int32_t, HP>(0, 1) < G;
#pragma unroll
            for (int n = 0; n < CPT; ++n) {
                simd<float, HP> row = S.template select<HP, 1>(n * HP) * a.scale + (float) madd[n];
                if (!ok[n]) {
                    row = -INFINITY;
                }
                slm_scatter<float, HP>(SLM_S + (i * CPT + n) * 4 + hoff, row, hm);
            }
            simd<int32_t, CPT> cl = cell;
            cl.merge(simd<int32_t, CPT>(-1), !ok);
            slm_block_store<int32_t, CPT>(SLM_C + i * CPT * 4, cl);
        }

        barrier();

        // phase 2: features [32i, 32i+32) of V for all G heads over the R slots
        simd<int32_t, R> cells;
#pragma unroll
        for (int u = 0; u < R / 64; ++u) {
            cells.template select<64, 1>(64 * u) = slm_block_load<int32_t, 64>(SLM_C + 256 * u);
        }
        simd<float, R> p[G];
        float          mx[G], sm[G];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            simd<float, R> sh;
#pragma unroll
            for (int u = 0; u < R / 64; ++u) {
                sh.template select<64, 1>(64 * u) = slm_block_load<float, 64>(SLM_S + h * R * 4 + 256 * u);
            }
            mx[h] = hmax<float>(sh);
            if (mx[h] == -INFINITY) {
                p[h] = 0.0f;
            } else {
                p[h] = exp(sh - mx[h]);
            }
            sm[h] = reduce<float>(p[h], std::plus<>());
        }

        simd<float, 8 * 16>  O0[2];
        simd<float, G1 * 16> O1[2];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            O0[nb] = 0.0f;
            O1[nb] = 0.0f;
        }
        simd<uint32_t, 32> voff(0, 2);
        voff.merge(simd<uint32_t, 32>(32), voff > 32);
        // V of 16 cells per step; the next step's loads are issued before this step's math
        simd<uint16_t, 32> vraw[2][16];
        qsa_dec_load_v<D, SOA>(vraw[0], cells.template select<16, 1>(0), a, g, s, i, voff);
#pragma unroll
        for (int st = 0; st < R / 16; ++st) {
            const int n0 = 16 * st;
            if (st + 1 < R / 16) {
                qsa_dec_load_v<D, SOA>(vraw[(st + 1) & 1], cells.template select<16, 1>(n0 + 16), a, g, s, i, voff);
            }
            simd<int32_t, 16> cv = cells.template select<16, 1>(n0);
            simd_mask<16>     vv = cv >= 0;
            // V of the 16 cells in f16, in VNNI pairs of cells for the two 16-feature halves
            simd<sycl::half, 256> Bv[2];
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                float            d;
                simd<int8_t, 32> qs;
                if constexpr (SOA) {
                    simd<uint16_t, 8> d8 = vraw[st & 1][c].template select<8, 1>(16);
                    d  = simd<float, 8>(d8.template bit_cast_view<sycl::half>())[i];
                    qs = vraw[st & 1][c].template select<16, 1>(0).template bit_cast_view<int8_t>();
                } else {
                    simd<uint16_t, 1> d1 = vraw[st & 1][c].template select<1, 1>(0);
                    d  = simd<float, 1>(d1.template bit_cast_view<sycl::half>())[0];
                    qs = vraw[st & 1][c].template select<16, 1>(1).template bit_cast_view<int8_t>();
                }
                // an uncounted cell read whatever cell 0 holds, which may not be finite
                if (!vv[c]) {
                    d  = 0.0f;
                    qs = 0;
                }
                simd<sycl::half, 32> vh = simd<float, 32>(qs) * d;
#pragma unroll
                for (int nb = 0; nb < 2; ++nb) {
                    Bv[nb].template select<16, 2>((c / 2) * 32 + (c % 2)) = vh.template select<16, 1>(16 * nb);
                }
            }
            simd<sycl::half, 8 * 16>  A0;
            simd<sycl::half, G1 * 16> A1;
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                A0.template select<16, 1>(16 * m) = p[m].template select<16, 1>(n0);
            }
#pragma unroll
            for (int m = 0; m < G1; ++m) {
                A1.template select<16, 1>(16 * m) = p[8 + m].template select<16, 1>(n0);
            }
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                O0[nb] = xmx::dpas<8, 8, float, float>(O0[nb], Bv[nb], A0);
                O1[nb] = xmx::dpas<8, G1, float, float>(O1[nb], Bv[nb], A1);
            }
        }

        // partial (row r, head g*G + h, chunk): D outputs, then max and sum
        const int n_head = a.n_head_kv * G;
#pragma unroll
        for (int h = 0; h < G; ++h) {
            float * pp = a.part + ((size_t) (r * n_head + g * G + h) * a.n_chunks + chunk) * (D + QSA_DEC_PART);
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
                simd<float, 16> o = h < 8 ? simd<float, 16>(O0[nb].template select<16, 1>(16 * h))
                                          : simd<float, 16>(O1[nb].template select<16, 1>(16 * (h - 8)));
                block_store<float, 16>(pp + 32 * i + 16 * nb, o);
            }
            if (i == 0) {
                simd<float, 4> ml = 0.0f;
                ml[0]             = mx[h];
                ml[1]             = sm[h];
                block_store<float, 4>(pp + D, ml);
            }
        }
    }

    auto get(syclexp::properties_tag) const { return syclexp::properties{ sycl::ext::intel::experimental::grf_size<256> }; }
};

// one thread per (row, head, 32 outputs): the chunk maxima and sums in two gathers, every chunk's
// outputs loaded at once, so the merge costs one round trip
static constexpr int QSA_DEC_CMAX = 64;

template <int D>
static void qsa_dec_combine(const float * part, float * dst, int n_rows, int n_head, int n_chunks,
                            dpct::queue_ptr stream) {
    constexpr int NT = D / 32;
    stream->parallel_for(sycl::nd_range<1>((size_t) n_rows * n_head * NT, NT), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        constexpr int CM = QSA_DEC_CMAX;
        constexpr int ST = D + QSA_DEC_PART; // floats per partial
        const int     i  = it.get_local_id(0);
        const int     rh = it.get_group(0); // r * n_head + head
        const float * pp = part + (size_t) rh * n_chunks * ST;

        simd<float, CM> m, l;
#pragma unroll
        for (int u = 0; u < CM / 32; ++u) {
            simd<uint32_t, 32> c(32 * u, 1);
            simd_mask<32>      in = c < n_chunks;
            simd<uint32_t, 32> off = c * (ST * 4);
            simd<float, 32>    mu  = gather<float, 32>(pp + D, off, in);
            simd<float, 32>    lu  = gather<float, 32>(pp + D + 1, off, in);
            mu.merge(simd<float, 32>(-INFINITY), !in);
            lu.merge(simd<float, 32>(0.0f), !in);
            m.template select<32, 1>(32 * u) = mu;
            l.template select<32, 1>(32 * u) = lu;
        }
        const float     mx = hmax<float>(m);
        simd<float, CM> w  = 0.0f;
        if (mx != -INFINITY) {
            w = exp(m - mx); // exp(-inf) = 0 for an empty chunk
        }
        const float sum = reduce<float>(w * l, std::plus<>());

        simd<float, 32> o = 0.0f;
#pragma unroll
        for (int c = 0; c < CM; ++c) {
            if (c < n_chunks) {
                o += (float) w[c] * block_load<float, 32>(pp + (size_t) c * ST + 32 * i, overaligned_tag<16>{});
            }
        }
        if (sum > 0.0f) {
            o = o / sum;
        } else {
            o = 0.0f;
        }
        block_store<float, 32>(dst + (size_t) rh * D + 32 * i, o);
    });
}

template <int D, int G, int R, int KV, bool MBITS>
static void qsa_dec_launch(const qsa_dec_args & a, int n_rows, dpct::queue_ptr stream) {
    const size_t n_wg = (size_t) n_rows * a.n_head_kv * a.n_chunks;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n_wg * (D / 32), D / 32), qsa_dec_attend_kernel<D, G, R, KV, MBITS>{ a });
    });
}

template <int D, int G, int R, bool SOA, bool MBITS>
static void qsa_dec_launch_xmx(const qsa_dec_args & a, int n_rows, dpct::queue_ptr stream) {
    const size_t n_wg = (size_t) n_rows * a.n_head_kv * a.n_chunks;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n_wg * (D / 32), D / 32), qsa_dec_attend_xmx_kernel<D, G, R, SOA, MBITS>{ a });
    });
}

static int qsa_decode_enabled() {
    static const int on = ggml_sycl_get_env("GGML_SYCL_QSA_DECODE", 1);
    return on;
}

static bool aligned16(const void * p, std::initializer_list<size_t> strides) {
    if (((uintptr_t) p) % 16 != 0) {
        return false;
    }
    for (size_t st : strides) {
        if (st % 16 != 0) {
            return false;
        }
    }
    return true;
}

bool ggml_sycl_qsa_decode_fa(ggml_backend_sycl_context & ctx, ggml_tensor * fa, const ggml_tensor * mask,
                             const ggml_tensor * idx) {
    if (!qsa_decode_enabled()) {
        return false;
    }
    const ggml_tensor * Q = fa->src[0];
    const ggml_tensor * K = fa->src[1];
    const ggml_tensor * V = fa->src[2];
    if (fa->src[4] != nullptr) {
        return false; // sinks
    }
    float scale = 0.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale,    (const float *) fa->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) fa->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    constexpr int D = 256;
    constexpr int G = 12;
    if (Q->ne[0] != D || K->ne[0] != D || V->ne[0] != D || Q->type != GGML_TYPE_F32 || fa->type != GGML_TYPE_F32 ||
        Q->nb[0] != sizeof(float) || !ggml_is_contiguous(fa)) {
        return false;
    }
    if (K->ne[2] == 0 || Q->ne[2] != (int64_t) G * K->ne[2] || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1] ||
        K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > QSA_DEC_MAX_ROWS) {
        return false;
    }
    if (K->type != V->type || (K->type != GGML_TYPE_Q8_0 && K->type != GGML_TYPE_F16)) {
        return false;
    }
    // a head row must be exactly one SoA span (D == GGML_SYCL_KV_SOA_SPAN), and K and V agree
    const bool soa = ggml_sycl_kv_is_soa(K);
    if (soa != ggml_sycl_kv_is_soa(V) || (soa && D != GGML_SYCL_KV_SOA_SPAN)) {
        return false;
    }
    if (K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type) ||
        !aligned16(K->data, { K->nb[1], K->nb[2], K->nb[3] }) || !aligned16(V->data, { V->nb[1], V->nb[2], V->nb[3] })) {
        return false;
    }
    const bool mbits = ggml_sycl_kq_mask_is_bits(mask);
    if (!mbits && (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(sycl::half))) {
        return false;
    }
    if (K->ne[1] > INT32_MAX || idx->ne[0] > INT32_MAX) {
        return false;
    }

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    dpct::queue_ptr stream = ctx.stream();

    qsa_dec_args a = {};
    a.Q         = (const char *) Q->data;
    a.q_nb1     = Q->nb[1];
    a.q_nb2     = Q->nb[2];
    a.q_nb3     = Q->nb[3];
    a.K         = (const char *) K->data;
    a.k_nb1     = K->nb[1];
    a.k_nb2     = K->nb[2];
    a.k_nb3     = K->nb[3];
    a.V         = (const char *) V->data;
    a.v_nb1     = V->nb[1];
    a.v_nb2     = V->nb[2];
    a.v_nb3     = V->nb[3];
    a.mask      = (const char *) mask->data;
    a.m_nb1     = mask->nb[1];
    a.m_nb3     = mask->nb[3];
    a.m_row     = ggml_sycl_kq_mask_row_bytes(mask->ne[0]);
    a.m_ne1     = (int) mask->ne[1];
    a.idx       = (const char *) idx->data;
    a.i_nb1     = idx->nb[1];
    a.i_nb2     = idx->nb[2];
    a.dst       = (float *) fa->data;
    a.n_kv      = (int) K->ne[1];
    a.width     = (int) idx->ne[0];
    a.n_tps     = (int) Q->ne[1];
    a.n_head_kv = (int) K->ne[2];
    // the q8_0 cache runs on the matrix engines (GGML_SYCL_QSA_DECODE_XMX=0: the vector kernel)
    static const int xmx_on = ggml_sycl_get_env("GGML_SYCL_QSA_DECODE_XMX", 1);
    const bool       q8     = K->type == GGML_TYPE_Q8_0;
    const bool       xmx    = q8 && xmx_on;
    const int        R      = xmx ? QSA_DEC_R_XMX : QSA_DEC_R;
    a.n_chunks  = (a.width + R - 1) / R;
    if (a.n_chunks > QSA_DEC_CMAX) {
        return false; // the combine holds at most QSA_DEC_CMAX chunks per output
    }
    a.scale     = scale;

    const int n_rows = (int) (Q->ne[1] * Q->ne[3]);
    const int n_head = (int) Q->ne[2];
    ggml_sycl_pool_alloc<float> part(ctx.pool(), (size_t) n_rows * n_head * a.n_chunks * (D + QSA_DEC_PART));
    a.part = part.get();

    static std::atomic<int> trace_left{ getenv("GGML_SYCL_QSA_DECODE_TRACE") ? atoi(getenv("GGML_SYCL_QSA_DECODE_TRACE")) : 0 };
    if (trace_left.fetch_sub(1) > 0) {
        fprintf(stderr, "[QSADEC] n_kv=%d width=%d rows=%d chunks=%d %s %s\n", a.n_kv, a.width, n_rows, a.n_chunks,
                ggml_type_name(K->type), mbits ? "bits" : "f16");
    }
    if (xmx) {
        if (soa) {
            mbits ? qsa_dec_launch_xmx<D, G, QSA_DEC_R_XMX, true, true>(a, n_rows, stream)
                  : qsa_dec_launch_xmx<D, G, QSA_DEC_R_XMX, true, false>(a, n_rows, stream);
        } else {
            mbits ? qsa_dec_launch_xmx<D, G, QSA_DEC_R_XMX, false, true>(a, n_rows, stream)
                  : qsa_dec_launch_xmx<D, G, QSA_DEC_R_XMX, false, false>(a, n_rows, stream);
        }
    } else if (q8 && soa) {
        mbits ? qsa_dec_launch<D, G, QSA_DEC_R, QSA_DEC_Q8_0_SOA, true>(a, n_rows, stream)
              : qsa_dec_launch<D, G, QSA_DEC_R, QSA_DEC_Q8_0_SOA, false>(a, n_rows, stream);
    } else if (q8) {
        mbits ? qsa_dec_launch<D, G, QSA_DEC_R, QSA_DEC_Q8_0, true>(a, n_rows, stream)
              : qsa_dec_launch<D, G, QSA_DEC_R, QSA_DEC_Q8_0, false>(a, n_rows, stream);
    } else {
        mbits ? qsa_dec_launch<D, G, QSA_DEC_R, QSA_DEC_F16, true>(a, n_rows, stream)
              : qsa_dec_launch<D, G, QSA_DEC_R, QSA_DEC_F16, false>(a, n_rows, stream);
    }
    qsa_dec_combine<D>(a.part, a.dst, n_rows, n_head, a.n_chunks, stream);
    return true;
}
