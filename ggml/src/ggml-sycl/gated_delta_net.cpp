#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <atomic>
#include <cmath>
#include <type_traits>

// The operands of one token for the WIDE kernel: this lane's shard of the k and q rows (and of the
// KDA g row), and the scalars. The rows come from one sub-group block load each, whose lane/element
// layout is exactly the shard's: element r of lane l is row element r * warp_size + l.
template <int rows_per_lane, bool KDA> struct gdn_step {
    float k[rows_per_lane];
    float q[rows_per_lane];
    float g[KDA ? rows_per_lane : 1];
    float v;
    float beta;
    float kf; // L2 norm factors of k and q, 1 when the rows come normed
    float qf;
};

// one sub-group block load of N floats per lane: element r of lane l is p[r * sub-group size + l]
template <int N> static __dpct_inline__ void gdn_block_load(const sycl::sub_group & sg, const float * p, float * out) {
    namespace syclex = sycl::ext::oneapi::experimental;
    syclex::group_load(sg, p, sycl::span<float, N>(out, N), syclex::properties(syclex::data_placement_striped));
}

// store this lane's shard of a state column: element r of lane l goes to p[r * warp_size + l].
// UNCACHED: the lines go to VRAM now and are not left dirty in L2, where the next kernel would pay their writeback
template <int rows_per_lane, int warp_size, bool UNCACHED>
static __dpct_inline__ void gdn_store_state(float * p, const float * s_shard, int lane) {
    if constexpr (UNCACHED) {
        namespace syclex = sycl::ext::oneapi::experimental;
        namespace intelex = sycl::ext::intel::experimental;
        using props = decltype(syclex::properties(intelex::write_hint<intelex::cache_control<
            intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L2, syclex::cache_level::L3>>));
        syclex::annotated_ptr<float, props> ap(p);
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            ap[r * warp_size + lane] = s_shard[r];
        }
    } else {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            p[r * warp_size + lane] = s_shard[r];
        }
    }
}

// factor of the fused L2 norm (rms_norm + scale) of a row whose shard sum of squares is ss
template <int S_v, int warp_size> static __dpct_inline__ float gdn_qk_norm_factor(float ss, float qk_eps, float qk_scale) {
    return sycl::rsqrt(warp_reduce_sum<warp_size>(ss) / S_v + qk_eps) * qk_scale;
}

template <int S_v, int warp_size, int rows_per_lane, bool KDA>
static __dpct_inline__ gdn_step<rows_per_lane, KDA> gdn_load_step(const float * q_t, const float * k_t, const float * v_t,
                                                                  const float * g_t, const float * beta_t, int col,
                                                                  const sycl::sub_group & sg, float qk_eps, float qk_scale) {
    gdn_step<rows_per_lane, KDA> st;
    gdn_block_load<rows_per_lane>(sg, k_t, st.k);
    gdn_block_load<rows_per_lane>(sg, q_t, st.q);
    if constexpr (KDA) {
        gdn_block_load<rows_per_lane>(sg, g_t, st.g);
    } else {
        st.g[0] = *g_t;
    }
    st.v    = v_t[col];
    st.beta = *beta_t;
    st.kf   = 1.0f;
    st.qf   = 1.0f;
    if (qk_scale != 0.0f) {
        float ssk = 0.0f;
        float ssq = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            ssk += st.k[r] * st.k[r];
            ssq += st.q[r] * st.q[r];
        }
        st.kf = gdn_qk_norm_factor<S_v, warp_size>(ssk, qk_eps, qk_scale);
        st.qf = gdn_qk_norm_factor<S_v, warp_size>(ssq, qk_eps, qk_scale);
    }
    return st;
}


template <int S_v, bool KDA, bool keep_rs_t, bool WIDE = false, bool UNCACHED = false>
void gated_delta_net_sycl(const float *     q,
                          const float *     k,
                          const float *     v,
                          const float *     g,
                          const float *     beta,
                          const float *     curr_state,
                          float *           dst,
                          float *           state,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          const sycl::uint3 neqk1_magic,
                          const sycl::uint3 rq3_magic,
                          float             scale,
                          int64_t           state_slot_stride,
                          int               K,
                          const int32_t *   s_ids,
                          int64_t           s_row_stride,
                          float             qk_eps,
                          float             qk_scale) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = item_ct1.get_local_id(2);
    const int      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // with s_ids, s0 is read straight from the state cache: row s_ids[sequence] of stride s_row_stride
    const int64_t state_in_offset      = (s_ids ? s_ids[sequence] * s_row_stride : sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    if constexpr (WIDE) {
        // The operands of token t + 1 are loaded before the dependent chain of token t runs, so their
        // latency hides behind it. Two named operand sets take turns, so nothing is copied between
        // tokens. Every token does the same operations in the same order as the loop below.
        const auto sg = item_ct1.get_sub_group();
        auto load_step = [&](int t) {
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            const float * beta_t = beta + gb_offset;
            const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);
            return gdn_load_step<S_v, warp_size, rows_per_lane, KDA>(q_t, k_t, v_t, g_t, beta_t, col, sg, qk_eps, qk_scale);
        };
        auto step = [&](const gdn_step<rows_per_lane, KDA> & cur, int t) {
            const float beta_val = cur.beta;

            if constexpr (!KDA) {
                const float g_val = sycl::native::exp(cur.g[0]);

                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[r] * cur.k[r];
                }
                float kv_col = warp_reduce_sum<warp_size>(kv_shard) * cur.kf;

                float delta_col = (cur.v - g_val * kv_col) * beta_val;
                const float delta_k = delta_col * cur.kf;

                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r]  = g_val * s_shard[r] + cur.k[r] * delta_k;
                    attn_partial += s_shard[r] * cur.q[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial) * cur.qf;

                if (lane == 0) {
                    attn_data[col] = attn_col * scale;
                }
            } else {
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += sycl::native::exp(cur.g[r]) * s_shard[r] * cur.k[r];
                }

                float kv_col = warp_reduce_sum<warp_size>(kv_shard) * cur.kf;

                float delta_col = (cur.v - kv_col) * beta_val;
                const float delta_k = delta_col * cur.kf;

                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r]  = sycl::native::exp(cur.g[r]) * s_shard[r] + cur.k[r] * delta_k;
                    attn_partial += s_shard[r] * cur.q[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial) * cur.qf;

                if (lane == 0) {
                    attn_data[col] = attn_col * scale;
                }
            }

            attn_data += S_v * H;

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    gdn_store_state<rows_per_lane, warp_size, UNCACHED>(state + target_slot * state_slot_stride + col * S_v,
                                                                     s_shard, lane);
                }
            }
        };

        if (n_tokens > 0) {
            gdn_step<rows_per_lane, KDA> a = load_step(0);
            gdn_step<rows_per_lane, KDA> b;
            int t = 0;
            for (; t + 1 < n_tokens; t += 2) {
                b = load_step(t + 1);
                step(a, t);
                if (t + 2 < n_tokens) {
                    a = load_step(t + 2);
                }
                step(b, t + 1);
            }
            if (t < n_tokens) {
                step(a, t);
            }
        }

        if constexpr (!keep_rs_t) {
            gdn_store_state<rows_per_lane, warp_size, UNCACHED>(state + col * S_v, s_shard, lane);
        }
        return;
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        float kf = 1.0f;
        float qf = 1.0f;
        if (qk_scale != 0.0f) {
            float ssk = 0.0f;
            float ssq = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                ssk += k_t[i] * k_t[i];
                ssq += q_t[i] * q_t[i];
            }
            kf = gdn_qk_norm_factor<S_v, warp_size>(ssk, qk_eps, qk_scale);
            qf = gdn_qk_norm_factor<S_v, warp_size>(ssq, qk_eps, qk_scale);
        }

        if constexpr (!KDA) {
            const float g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard) * kf;

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
            const float delta_k = delta_col * kf;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_k;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial) * qf;

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard) * kf;

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;
            const float delta_k = delta_col * kf;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_k;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial) * qf;

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                gdn_store_state<rows_per_lane, warp_size, UNCACHED>(state + target_slot * state_slot_stride + col * S_v,
                                                                 s_shard, lane);
            }
        }
    }

    if constexpr (!keep_rs_t) {
        gdn_store_state<rows_per_lane, warp_size, UNCACHED>(state + col * S_v, s_shard, lane);
    }
}

template <bool KDA, bool keep_rs_t, bool WIDE, bool UNCACHED>
static void launch_gated_delta_net(const float *   q_d,
                                   const float *   k_d,
                                   const float *   v_d,
                                   const float *   g_d,
                                   const float *   b_d,
                                   const float *   s_d,
                                   float *         dst_d,
                                   float *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         neqk1,
                                   int64_t         rq3,
                                   float           scale,
                                   int64_t         state_slot_stride,
                                   int             K,
                                   const int32_t * s_ids,
                                   int64_t         s_row_stride,
                                   float           qk_eps,
                                   float           qk_scale,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    const int num_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            {
                constexpr int sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t, WIDE, UNCACHED>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, qk_eps, qk_scale);
                                     });
            }
            break;
        case 32:
            {
                constexpr int sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t, WIDE, UNCACHED>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, qk_eps, qk_scale);
                                     });
            }
            break;
        case 64: {
            {
                constexpr int sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t, WIDE, UNCACHED>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, qk_eps, qk_scale);
                                        });
            }
            break;
        }
        case 128: {
            {
                constexpr int sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t, WIDE, UNCACHED>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, qk_eps, qk_scale);
                                        });
            }
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// ---------------------------------------------------------------------------------------------
// Chunked prefill (scalar gate only)
//
// The recurrence S_t = g_t S_{t-1} + k_t d_t^T, d_t = b_t (v_t - g_t S_{t-1}^T k_t), o_t = scale S_t^T q_t
// over a chunk of C tokens, with G_r the gate's log summed over the chunk up to row r and S0 the
// state before it, is (the WY form of the delta rule)
//   (I + A) D = diag(b) V - diag(b e^G) K S0,  A[r][s] = b_r e^(G_r - G_s) k_r.k_s  (s < r)
//   D = U - W S0,  U = (I + A)^-1 diag(b) V,  W = (I + A)^-1 diag(b e^G) K
//   O = scale (diag(e^G) Q S0 + P D),         P[r][s] = e^(G_r - G_s) q_r.k_s      (s <= r)
//   S_C = e^(G_C) S0 + sum_s e^(G_C - G_s) k_s d_s^T
// U, W and P depend on the chunk alone, so they are built for every chunk at once; only the last
// pass walks the chunks in order, one sub-group per state column as the recurrent kernel does.
// Rows past n_tokens are zero (b = 0, gate 0), so they change nothing.
//
// Stages, all on the stream's queue:
//   1. per chunk: the q/k L2 norm factors (1 when the rows come normed), b, and G
//   2. per chunk: A (local memory), P, then U and W by forward substitution
//   3. per state column: the chunks in order
// Scratch (pool): KF, QF, BE, GC [bh][TP], U, W [bh][TP][S_v], P [bh][NC][C][C], bh = n_seqs * H.
//
// With XMX (GGML_SYCL_GDN_CHUNKED_XMX, on by default where f16 8x16x16 tiles exist) stage 3 is a work-group
// per 16 state columns doing the products as f16 tiles with f32 accumulation:
//   D = U + (-W) S0,  O = Qe S0 + P D,  S_C = e^(G_C) S0 + Kd D
// with Qe = diag(e^G qf) Q and Kd[i][s] = e^(G_C - G_s) kf_s k_s[i]; stage 2 also writes -W, Qe, P and
// Kd as f16 for it. The state stays f32 between chunks; only the operands of each product are f16.
// ---------------------------------------------------------------------------------------------

static constexpr int GDN_CHUNK = 64;

struct gdn_chunk_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    int64_t       sq1, sq2, sq3;  // q and k share strides
    int64_t       sv1, sv2, sv3;
    int64_t       sb1, sb2, sb3;  // g (scalar) and beta
    int64_t       neqk1;
    int64_t       rq3;
    int64_t       H;
    int64_t       n_tokens;
    int64_t       TP;  // n_tokens padded to whole chunks
    int64_t       NC;
    float         qk_eps;
    float         qk_scale;
    float *       KF;
    float *       QF;
    float *       BE;
    float *       GC;
    float *       U;
    float *       W;
    float *       P;
    sycl::half *  Wh;   // -W, [bh][TP][S_v]          (XMX only)
    sycl::half *  Qh;   // Qe, [bh][TP][S_v]
    sycl::half *  Ph;   // P,  [bh][NC][C][C]
    sycl::half *  KTh;  // Kd, [bh][NC][S_v][C]

    const float * qrow(int64_t seq, int64_t h, int64_t t) const {
        return q + (seq / rq3) * sq3 + t * sq2 + (h % neqk1) * sq1;
    }
    const float * krow(int64_t seq, int64_t h, int64_t t) const {
        return k + (seq / rq3) * sq3 + t * sq2 + (h % neqk1) * sq1;
    }
    const float * vrow(int64_t seq, int64_t h, int64_t t) const { return v + seq * sv3 + t * sv2 + h * sv1; }
    int64_t       gb(int64_t seq, int64_t h, int64_t t) const { return seq * sb3 + t * sb2 + h * sb1; }
};

// stage 1: one sub-group per (chunk, head, sequence)
template <int S_v, int warp_size>
static void gdn_chunk_scalars(const gdn_chunk_args a, const sycl::nd_item<3> & it) {
    const int64_t seq  = it.get_group(0);
    const int64_t h    = it.get_group(1);
    const int64_t c    = it.get_group(2);
    const int     lane = it.get_local_id(2);
    const int64_t base = (seq * a.H + h) * a.TP + c * GDN_CHUNK;

    float acc = 0.0f;
    for (int r = 0; r < GDN_CHUNK; ++r) {
        const int64_t t  = c * GDN_CHUNK + r;
        float         kf = 0.0f, qf = 0.0f, be = 0.0f, gg = 0.0f;
        if (t < a.n_tokens) {
            kf = 1.0f;
            qf = 1.0f;
            if (a.qk_scale != 0.0f) {
                const float * kr  = a.krow(seq, h, t);
                const float * qr  = a.qrow(seq, h, t);
                float         ssk = 0.0f, ssq = 0.0f;
#pragma unroll
                for (int i = lane; i < S_v; i += warp_size) {
                    ssk += kr[i] * kr[i];
                    ssq += qr[i] * qr[i];
                }
                kf = gdn_qk_norm_factor<S_v, warp_size>(ssk, a.qk_eps, a.qk_scale);
                qf = gdn_qk_norm_factor<S_v, warp_size>(ssq, a.qk_eps, a.qk_scale);
            }
            const int64_t o = a.gb(seq, h, t);
            be              = a.beta[o];
            gg              = a.g[o];
        }
        acc += gg;
        if (lane == 0) {
            a.KF[base + r] = kf;
            a.QF[base + r] = qf;
            a.BE[base + r] = be;
            a.GC[base + r] = acc;
        }
    }
}

// stage 2: one work-group of S_v items per (chunk, head, sequence)
template <int S_v>
static void gdn_chunk_wy(const gdn_chunk_args a, const sycl::nd_item<3> & it, float * A, float * X, float * sc) {
    constexpr int C    = GDN_CHUNK;
    const int64_t seq  = it.get_group(0);
    const int64_t h    = it.get_group(1);
    const int64_t c    = it.get_group(2);
    const int     lid  = it.get_local_id(2);
    const int64_t bh   = seq * a.H + h;
    const int64_t base = bh * a.TP + c * C;
    float *       G    = sc;          // [C]
    float *       BE   = sc + C;      // [C]
    float *       KF   = sc + 2 * C;  // [C]
    float *       QF   = sc + 3 * C;  // [C]

    for (int r = lid; r < C; r += S_v) {
        G[r]  = a.GC[base + r];
        BE[r] = a.BE[base + r];
        KF[r] = a.KF[base + r];
        QF[r] = a.QF[base + r];
    }
    // the chunk's k rows, read once: row s at s * (S_v + 1), so items reading different rows at the
    // same i hit different banks. X holds them until the forward substitution below
    constexpr int KS = S_v + 1;
    float *       Ks = X;
    for (int r = 0; r < C; ++r) {
        const int64_t t = c * C + r;
        Ks[r * KS + lid] = t < a.n_tokens ? a.krow(seq, h, t)[lid] : 0.0f;
    }
    sycl::group_barrier(it.get_group());

    float * Pc = a.P + (bh * a.NC + c) * C * C;
    for (int idx = lid; idx < C * C; idx += S_v) {
        const int     r  = idx / C;
        const int     s  = idx - r * C;
        const int64_t tr = c * C + r;
        const int64_t ts = c * C + s;
        float         av = 0.0f, pv = 0.0f;
        if (s <= r && tr < a.n_tokens) {
            // neighbouring items share r: the q row is one broadcast read, the k rows come from local memory
            const float * kr = Ks + r * KS;
            const float * ks = Ks + s * KS;
            const float * qr = a.qrow(seq, h, tr);
            float         kk = 0.0f, qk = 0.0f;
#pragma unroll 8
            for (int i = 0; i < S_v; ++i) {
                kk += kr[i] * ks[i];
                qk += qr[i] * ks[i];
            }
            const float dec = sycl::exp(G[r] - G[s]);
            if (s < r) {
                av = BE[r] * dec * KF[r] * KF[s] * kk;
            }
            pv = dec * QF[r] * KF[s] * qk;
        }
        A[idx]  = av;
        Pc[idx] = pv;
        if (a.Ph) {
            a.Ph[(bh * a.NC + c) * C * C + idx] = (sycl::half) pv;
        }
    }
    // every item is done with Ks before X is reused
    sycl::group_barrier(it.get_group());

    // item lid owns column lid of U, then of W; X[r * S_v + lid] is only ever touched by it
    for (int pass = 0; pass < 2; ++pass) {
        for (int r = 0; r < C; ++r) {
            const int64_t t   = c * C + r;
            float         acc = 0.0f;
            if (t < a.n_tokens) {
                acc = pass == 0 ? BE[r] * a.vrow(seq, h, t)[lid] : BE[r] * sycl::exp(G[r]) * KF[r] * a.krow(seq, h, t)[lid];
            }
            for (int s = 0; s < r; ++s) {
                acc -= A[r * C + s] * X[s * S_v + lid];
            }
            X[r * S_v + lid] = acc;
        }
        float * out = (pass == 0 ? a.U : a.W) + base * S_v;
        for (int r = 0; r < C; ++r) {
            out[r * S_v + lid] = X[r * S_v + lid];
            if (pass == 1 && a.Wh) {
                a.Wh[(base + r) * S_v + lid] = (sycl::half) -X[r * S_v + lid];
            }
        }
    }

    if (a.Qh) {
        // the operands of the XMX pass: Qe rows, and the decayed keys transposed (row lid of Kd)
        sycl::half * kd = a.KTh + ((bh * a.NC + c) * S_v + lid) * C;
        for (int r = 0; r < C; ++r) {
            const int64_t t  = c * C + r;
            float         qe = 0.0f, kv = 0.0f;
            if (t < a.n_tokens) {
                qe = sycl::exp(G[r]) * QF[r] * a.qrow(seq, h, t)[lid];
                kv = sycl::exp(G[C - 1] - G[r]) * KF[r] * a.krow(seq, h, t)[lid];
            }
            a.Qh[(base + r) * S_v + lid] = (sycl::half) qe;
            kd[r]                        = (sycl::half) kv;
        }
    }
}

namespace gdn_mx = sycl::ext::oneapi::experimental::matrix;

static constexpr int GDN_XMX_TM = 8;
static constexpr int GDN_XMX_TN = 16;
static constexpr int GDN_XMX_TK = 16;
static constexpr int GDN_XMX_SG = 4;

static bool gdn_xmx_supported_query(dpct::queue_ptr stream) {
    try {
        const auto combinations = stream->get_device().get_info<
            sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        for (const auto & cb : combinations) {
            if (cb.atype == gdn_mx::matrix_type::fp16 && cb.btype == gdn_mx::matrix_type::fp16 &&
                cb.ctype == gdn_mx::matrix_type::fp32 && cb.dtype == gdn_mx::matrix_type::fp32 &&
                (cb.max_msize >= GDN_XMX_TM || cb.msize == GDN_XMX_TM) &&
                (cb.max_nsize >= GDN_XMX_TN || cb.nsize == GDN_XMX_TN) &&
                (cb.max_ksize >= GDN_XMX_TK || cb.ksize == GDN_XMX_TK)) {
                return true;
            }
        }
    } catch (const sycl::exception &) {
    }
    return false;
}

// asked once per device: the matrix combinations query is not cheap and the answer cannot change
static bool gdn_xmx_supported(int device, dpct::queue_ptr stream) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_GDN_CHUNKED_XMX", 1);
    if (!on || WARP_SIZE != GDN_XMX_TN || device < 0 || device >= GGML_SYCL_MAX_DEVICES) {
        return false;
    }
    static std::atomic<int> cached[GGML_SYCL_MAX_DEVICES] = {};  // 0 unknown, 1 no, 2 yes
    int v = cached[device].load(std::memory_order_relaxed);
    if (v == 0) {
        v = gdn_xmx_supported_query(stream) ? 2 : 1;
        cached[device].store(v, std::memory_order_relaxed);
    }
    return v == 2;
}

// stage 3 on XMX: a work-group of GDN_XMX_SG sub-groups per (sequence, head, 16 state columns)
template <int S_v>
static void gdn_chunk_pass_xmx_launch(dpct::queue_ptr stream, const gdn_chunk_args a, int64_t n_seqs, const float * curr_state,
                                      float * dst, float * state, const int32_t * s_ids, int64_t s_row_stride, float scale) {
    constexpr int C  = GDN_CHUNK;
    constexpr int TN = GDN_XMX_TN;
    constexpr int TM = GDN_XMX_TM;
    constexpr int TK = GDN_XMX_TK;
    constexpr int NI = GDN_XMX_SG * WARP_SIZE;
    static_assert(S_v % TK == 0 && C % TK == 0 && C % TM == 0 && S_v % TM == 0, "whole tiles");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1>      Sf(S_v * TN, cgh);  // the state columns, f32, [S_v][16]
        sycl::local_accessor<sycl::half, 1> Sh(S_v * TN, cgh);  // the same, f16 packed B operand
        sycl::local_accessor<float, 1>      Df(C * TN, cgh);    // d of the chunk, [C][16]
        sycl::local_accessor<sycl::half, 1> Dh(C * TN, cgh);    // the same, f16 packed B operand
        sycl::local_accessor<float, 1>      Of(C * TN, cgh);    // outputs of the chunk, [C][16]
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(n_seqs, a.H, (size_t) (S_v / TN) * NI), sycl::range<3>(1, 1, NI)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const auto    sg    = it.get_sub_group();
                const int     sg_id = sg.get_group_linear_id();
                const int     lid   = it.get_local_linear_id();
                const int64_t seq   = it.get_group(0);
                const int64_t h     = it.get_group(1);
                const int     j0    = it.get_group(2) * TN;
                const int64_t bh    = seq * a.H + h;
                const auto    wg    = it.get_group();

                const auto sf_p = Sf.get_multi_ptr<sycl::access::decorated::no>();
                const auto sh_p = Sh.get_multi_ptr<sycl::access::decorated::no>();
                const auto df_p = Df.get_multi_ptr<sycl::access::decorated::no>();
                const auto dh_p = Dh.get_multi_ptr<sycl::access::decorated::no>();
                const auto of_p = Of.get_multi_ptr<sycl::access::decorated::no>();

                // state column j of s0 is S0[:, j]: element i at col * S_v + i
                const int64_t in_base = (s_ids ? s_ids[seq] * s_row_stride : seq * a.H * S_v * S_v) + h * S_v * S_v;
                for (int idx = lid; idx < S_v * TN; idx += NI) {
                    const int i = idx / TN;
                    const int n = idx % TN;
                    Sf[idx]     = curr_state[in_base + (int64_t) (j0 + n) * S_v + i];
                }
                sycl::group_barrier(wg);

                for (int64_t c = 0; c < a.NC; ++c) {
                    const int64_t base = bh * a.TP + c * C;

                    // B operand layout (ext_intel_packed): element (k, n) at ((k / 2) * 16 + n) * 2 + k % 2
                    for (int idx = lid; idx < S_v * TN; idx += NI) {
                        const int i                        = idx / TN;
                        const int n                        = idx % TN;
                        Sh[((i >> 1) * TN + n) * 2 + (i & 1)] = (sycl::half) Sf[idx];
                    }
                    sycl::group_barrier(wg);

                    // D = U + (-W) S0
                    for (int mt = sg_id; mt < C / TM; mt += GDN_XMX_SG) {
                        gdn_mx::joint_matrix<sycl::sub_group, float, gdn_mx::use::accumulator, TM, TN> acc;
                        gdn_mx::joint_matrix_load(
                            sg, acc,
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                a.U + (base + mt * TM) * S_v + j0),
                            S_v, gdn_mx::layout::row_major);
#pragma unroll
                        for (int kk = 0; kk < S_v / TK; ++kk) {
                            gdn_mx::joint_matrix<sycl::sub_group, sycl::half, gdn_mx::use::a, TM, TK, gdn_mx::layout::row_major> ma;
                            gdn_mx::joint_matrix_load(
                                sg, ma,
                                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                    a.Wh + (base + mt * TM) * S_v + kk * TK),
                                S_v);
                            gdn_mx::joint_matrix<sycl::sub_group, sycl::half, gdn_mx::use::b, TK, TN, gdn_mx::layout::ext_intel_packed> mb;
                            gdn_mx::joint_matrix_load(sg, mb, sh_p + kk * TK * TN, TN * 2);
                            gdn_mx::joint_matrix_mad(sg, acc, ma, mb, acc);
                        }
                        gdn_mx::joint_matrix_store(sg, acc, df_p + mt * TM * TN, TN, gdn_mx::layout::row_major);
                    }
                    sycl::group_barrier(wg);

                    for (int idx = lid; idx < C * TN; idx += NI) {
                        const int r                        = idx / TN;
                        const int n                        = idx % TN;
                        Dh[((r >> 1) * TN + n) * 2 + (r & 1)] = (sycl::half) Df[idx];
                    }
                    sycl::group_barrier(wg);

                    // O = Qe S0 + P D
                    for (int mt = sg_id; mt < C / TM; mt += GDN_XMX_SG) {
                        gdn_mx::joint_matrix<sycl::sub_group, float, gdn_mx::use::accumulator, TM, TN> acc;
                        gdn_mx::joint_matrix_fill(sg, acc, 0.0f);
#pragma unroll
                        for (int kk = 0; kk < S_v / TK; ++kk) {
                            gdn_mx::joint_matrix<sycl::sub_group, sycl::half, gdn_mx::use::a, TM, TK, gdn_mx::layout::row_major> ma;
                            gdn_mx::joint_matrix_load(
                                sg, ma,
                                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                    a.Qh + (base + mt * TM) * S_v + kk * TK),
                                S_v);
                            gdn_mx::joint_matrix<sycl::sub_group, sycl::half, gdn_mx::use::b, TK, TN, gdn_mx::layout::ext_intel_packed> mb;
                            gdn_mx::joint_matrix_load(sg, mb, sh_p + kk * TK * TN, TN * 2);
                            gdn_mx::joint_matrix_mad(sg, acc, ma, mb, acc);
                        }
#pragma unroll
                        for (int kk = 0; kk < C / TK; ++kk) {
                            gdn_mx::joint_matrix<sycl::sub_group, sycl::half, gdn_mx::use::a, TM, TK, gdn_mx::layout::row_major> ma;
                            gdn_mx::joint_matrix_load(
                                sg, ma,
                                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                    a.Ph + (bh * a.NC + c) * C * C + mt * TM * C + kk * TK),
                                C);
                            gdn_mx::joint_matrix<sycl::sub_group, sycl::half, gdn_mx::use::b, TK, TN, gdn_mx::layout::ext_intel_packed> mb;
                            gdn_mx::joint_matrix_load(sg, mb, dh_p + kk * TK * TN, TN * 2);
                            gdn_mx::joint_matrix_mad(sg, acc, ma, mb, acc);
                        }
                        gdn_mx::joint_matrix_store(sg, acc, of_p + mt * TM * TN, TN, gdn_mx::layout::row_major);
                    }
                    sycl::group_barrier(wg);

                    const float eg = sycl::exp(a.GC[base + C - 1]);
                    for (int idx = lid; idx < C * TN; idx += NI) {
                        const int     r = idx / TN;
                        const int     n = idx % TN;
                        const int64_t t = c * C + r;
                        if (t < a.n_tokens) {
                            dst[((seq * a.n_tokens + t) * a.H + h) * S_v + j0 + n] = Of[idx] * scale;
                        }
                    }
                    for (int idx = lid; idx < S_v * TN; idx += NI) {
                        Sf[idx] *= eg;
                    }
                    sycl::group_barrier(wg);

                    // S_C = e^(G_C) S0 + Kd D
                    for (int mt = sg_id; mt < S_v / TM; mt += GDN_XMX_SG) {
                        gdn_mx::joint_matrix<sycl::sub_group, float, gdn_mx::use::accumulator, TM, TN> acc;
                        gdn_mx::joint_matrix_load(sg, acc, sf_p + mt * TM * TN, TN, gdn_mx::layout::row_major);
#pragma unroll
                        for (int kk = 0; kk < C / TK; ++kk) {
                            gdn_mx::joint_matrix<sycl::sub_group, sycl::half, gdn_mx::use::a, TM, TK, gdn_mx::layout::row_major> ma;
                            gdn_mx::joint_matrix_load(
                                sg, ma,
                                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                    a.KTh + ((bh * a.NC + c) * S_v + mt * TM) * C + kk * TK),
                                C);
                            gdn_mx::joint_matrix<sycl::sub_group, sycl::half, gdn_mx::use::b, TK, TN, gdn_mx::layout::ext_intel_packed> mb;
                            gdn_mx::joint_matrix_load(sg, mb, dh_p + kk * TK * TN, TN * 2);
                            gdn_mx::joint_matrix_mad(sg, acc, ma, mb, acc);
                        }
                        gdn_mx::joint_matrix_store(sg, acc, sf_p + mt * TM * TN, TN, gdn_mx::layout::row_major);
                    }
                    sycl::group_barrier(wg);
                }

                for (int idx = lid; idx < S_v * TN; idx += NI) {
                    const int i = idx / TN;
                    const int n = idx % TN;
                    state[bh * S_v * S_v + (int64_t) (j0 + n) * S_v + i] = Sf[idx];
                }
            });
    });
}

// stage 3: one sub-group per state column, the chunks in order
template <int S_v, int warp_size, bool UNCACHED>
static void gdn_chunk_pass(const gdn_chunk_args a, const sycl::nd_item<3> & it, const float * curr_state, float * dst,
                           float * state, const int32_t * s_ids, int64_t s_row_stride, float scale, float * sm) {
    constexpr int C             = GDN_CHUNK;
    constexpr int rows_per_lane = S_v / warp_size;
    const auto    sg            = it.get_sub_group();
    const int64_t seq           = it.get_group(0);
    const int64_t h             = it.get_group(1);
    const int     sgid          = it.get_local_id(2) / warp_size;
    const int     lane          = it.get_local_id(2) % warp_size;
    const int     n_sg          = it.get_local_range(2) / warp_size;
    const int     col           = it.get_group(2) * n_sg + sgid;
    const int64_t bh            = seq * a.H + h;
    float *       sd            = sm + sgid * 2 * C;  // d of this column, [C]
    float *       sq            = sd + C;              // q_r . S0[:, col] * qf_r, [C]

    const int64_t in_off = (s_ids ? s_ids[seq] * s_row_stride : seq * a.H * S_v * S_v) + h * S_v * S_v + col * S_v;
    float         s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        s_shard[r] = curr_state[in_off + r * warp_size + lane];
    }

    for (int64_t c = 0; c < a.NC; ++c) {
        const int64_t base = bh * a.TP + c * C;
        const float * Wc   = a.W + base * S_v;
        const float * Uc   = a.U + base * S_v;
        for (int r = 0; r < C; ++r) {
            const int64_t t  = c * C + r;
            float         pd = 0.0f, pq = 0.0f;
#pragma unroll
            for (int rr = 0; rr < rows_per_lane; ++rr) {
                pd += Wc[r * S_v + rr * warp_size + lane] * s_shard[rr];
            }
            if (t < a.n_tokens) {
                const float * qr = a.qrow(seq, h, t);
#pragma unroll
                for (int rr = 0; rr < rows_per_lane; ++rr) {
                    pq += qr[rr * warp_size + lane] * s_shard[rr];
                }
            }
            const float dr = warp_reduce_sum<warp_size>(pd);
            const float qr = warp_reduce_sum<warp_size>(pq);
            if (lane == 0) {
                sd[r] = Uc[r * S_v + col] - dr;
                sq[r] = qr * a.QF[base + r];
            }
        }
        sycl::group_barrier(sg);

        const float * Pc = a.P + (bh * a.NC + c) * C * C;
        for (int r = lane; r < C; r += warp_size) {
            const int64_t t = c * C + r;
            if (t >= a.n_tokens) {
                continue;
            }
            float o = sycl::exp(a.GC[base + r]) * sq[r];
            for (int s = 0; s <= r; ++s) {
                o += Pc[r * C + s] * sd[s];
            }
            dst[((seq * a.n_tokens + t) * a.H + h) * S_v + col] = o * scale;
        }

        const float gl = a.GC[base + C - 1];
        const float eg = sycl::exp(gl);
#pragma unroll
        for (int rr = 0; rr < rows_per_lane; ++rr) {
            s_shard[rr] *= eg;
        }
        for (int s = 0; s < C; ++s) {
            const int64_t t = c * C + s;
            if (t >= a.n_tokens) {
                break;
            }
            const float   coef = sycl::exp(gl - a.GC[base + s]) * a.KF[base + s] * sd[s];
            const float * ks   = a.krow(seq, h, t);
#pragma unroll
            for (int rr = 0; rr < rows_per_lane; ++rr) {
                s_shard[rr] += coef * ks[rr * warp_size + lane];
            }
        }
        // the next chunk overwrites sd and sq
        sycl::group_barrier(sg);
    }

    gdn_store_state<rows_per_lane, warp_size, UNCACHED>(state + (seq * a.H + h) * S_v * S_v + col * S_v, s_shard, lane);
}

template <int S_v, bool UNCACHED>
static void gdn_chunked_launch(ggml_backend_sycl_context & ctx, gdn_chunk_args a, int64_t n_seqs, const float * s_d,
                               float * dst_d, float * state_d, const int32_t * s_ids, int64_t s_row_stride, float scale) {
    constexpr int ws   = WARP_SIZE;
    constexpr int C    = GDN_CHUNK;
    constexpr int n_sg = 4;
    static_assert(S_v % ws == 0 && S_v % n_sg == 0, "a state column is whole lanes");
    dpct::queue_ptr stream = ctx.stream();

    const int64_t bh = n_seqs * a.H;
    ggml_sycl_pool_alloc<float> scal(ctx.pool(), (size_t) 4 * bh * a.TP);
    ggml_sycl_pool_alloc<float> uw(ctx.pool(), (size_t) 2 * bh * a.TP * S_v);
    ggml_sycl_pool_alloc<float> pm(ctx.pool(), (size_t) bh * a.NC * C * C);
    a.KF = scal.get();
    a.QF = a.KF + bh * a.TP;
    a.BE = a.QF + bh * a.TP;
    a.GC = a.BE + bh * a.TP;
    a.U  = uw.get();
    a.W  = a.U + bh * a.TP * S_v;
    a.P  = pm.get();

    const bool xmx = gdn_xmx_supported(ctx.device, stream);
    ggml_sycl_pool_alloc<sycl::half> hs(ctx.pool());
    if (xmx) {
        // -W, Qe and Kd ([S_v][C] per chunk is TP * S_v per head as well), then P
        sycl::half * p = hs.alloc((size_t) bh * (3 * a.TP * S_v + a.NC * C * C));
        a.Wh  = p;
        a.Qh  = a.Wh + bh * a.TP * S_v;
        a.KTh = a.Qh + bh * a.TP * S_v;
        a.Ph  = a.KTh + bh * a.TP * S_v;
    }

    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(n_seqs, a.H, a.NC * ws), sycl::range<3>(1, 1, ws)),
                         [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             gdn_chunk_scalars<S_v, ws>(a, it);
                         });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> A(C * C, cgh);
        sycl::local_accessor<float, 1> X(C * (S_v + 1), cgh);  // the chunk's k rows first, then U and W
        sycl::local_accessor<float, 1> sc(4 * C, cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(n_seqs, a.H, a.NC * S_v), sycl::range<3>(1, 1, S_v)),
                         [=](sycl::nd_item<3> it) {
                             gdn_chunk_wy<S_v>(a, it, A.get_multi_ptr<sycl::access::decorated::no>().get(),
                                               X.get_multi_ptr<sycl::access::decorated::no>().get(),
                                               sc.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });

    if (xmx) {
        // the f16 state and d round to 11 bits per product; the state itself stays f32 between chunks
        gdn_chunk_pass_xmx_launch<S_v>(stream, a, n_seqs, s_d, dst_d, state_d, s_ids, s_row_stride, scale);
        return;
    }

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sm(n_sg * 2 * C, cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(n_seqs, a.H, (int64_t) S_v * ws), sycl::range<3>(1, 1, n_sg * ws)),
                         [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             gdn_chunk_pass<S_v, ws, UNCACHED>(a, it, s_d, dst_d, state_d, s_ids, s_row_stride, scale,
                                                               sm.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

// GGML_SYCL_GDN_CHUNKED=N runs prefill of at least N tokens through the chunked path; 0 (default) never
static int gdn_chunked_min_tokens() {
    static const int n = ggml_sycl_get_env("GGML_SYCL_GDN_CHUNKED", 0);
    return n;
}

static void ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              const ggml_sycl_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    // the q/k L2 norms of this op were skipped: read the raw rows and norm them in the kernel
    float qk_eps   = 0.0f;
    float qk_scale = 0.0f;
    if (ctx.gdn_qk_norm.gdn == dst) {
        src_q    = const_cast<ggml_tensor *>(ctx.gdn_qk_norm.q);
        src_k    = const_cast<ggml_tensor *>(ctx.gdn_qk_norm.k);
        qk_eps   = ctx.gdn_qk_norm.eps;
        qk_scale = ctx.gdn_qk_norm.scale;
        ctx.gdn_qk_norm = {};
    }

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // the state gather of this op was skipped (GGML_SYCL_GDN_STATE_GATHER): read s0 from the cache rows
    const int32_t * s_ids        = nullptr;
    int64_t         s_row_stride = 0;
    if (ctx.gdn_gather.gdn == dst) {
        s_d          = ctx.gdn_gather.states;
        s_ids        = ctx.gdn_gather.ids;
        s_row_stride = ctx.gdn_gather.row_stride;
        ctx.gdn_gather = {};
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // block loads of the k, q (and KDA g) rows want every row 16 B aligned; rows are whole shards
    // only when the sub-group spans them (S_v >= warp size)
    const int  warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;
    const bool wide      = (g_ggml_sycl_wide_loads & GGML_SYCL_WIDE_GDN) && S_v >= warp_size &&
                      ((uintptr_t) q_d | (uintptr_t) k_d | (sq1 | sq2 | sq3) * sizeof(float)) % 16 == 0 &&
                      (!kda || ((uintptr_t) g_d | S_v * sizeof(float) | nbb1 * S_v | nbb2 * S_v | nbb3 * S_v) % 16 == 0);

    // prefill: the chunked path when asked for. It writes the final state only, so not with rollback
    // snapshots (K > 1): delta-net-base.cpp copies min(n_tokens, K) slots into the cache whatever the batch
    const int chunk_min = gdn_chunked_min_tokens();
    if (chunk_min > 0 && !kda && n_tokens >= chunk_min && K == 1 && S_v % WARP_SIZE == 0 &&
        (S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128)) {
        gdn_chunk_args a = {};
        a.q        = q_d;
        a.k        = k_d;
        a.v        = v_d;
        a.g        = g_d;
        a.beta     = b_d;
        a.sq1      = sq1;
        a.sq2      = sq2;
        a.sq3      = sq3;
        a.sv1      = sv1;
        a.sv2      = sv2;
        a.sv3      = sv3;
        a.sb1      = sb1;
        a.sb2      = sb2;
        a.sb3      = sb3;
        a.neqk1    = neqk1;
        a.rq3      = rq3;
        a.H        = H;
        a.n_tokens = n_tokens;
        a.NC       = (n_tokens + GDN_CHUNK - 1) / GDN_CHUNK;
        a.TP       = a.NC * GDN_CHUNK;
        a.qk_eps   = qk_eps;
        a.qk_scale = qk_scale;
        const bool uncached = g_ggml_sycl_gdn_state_store && cache != nullptr;
        auto run = [&](auto sv_c) {
            constexpr int sv = decltype(sv_c)::value;
            if constexpr (sv % WARP_SIZE == 0) {
                if (uncached) {
                    gdn_chunked_launch<sv, true>(ctx, a, n_seqs, s_d, dst_d, state_d, s_ids, s_row_stride, scale);
                } else {
                    gdn_chunked_launch<sv, false>(ctx, a, n_seqs, s_d, dst_d, state_d, s_ids, s_row_stride, scale);
                }
            }
        };
        switch (S_v) {
            case 16:  run(std::integral_constant<int, 16>{});  break;
            case 32:  run(std::integral_constant<int, 32>{});  break;
            case 64:  run(std::integral_constant<int, 64>{});  break;
            default:  run(std::integral_constant<int, 128>{}); break;
        }
        return;
    }

    const auto launch = [&](auto kda_c, auto keep_c, auto wide_c, auto store_c) {
        launch_gated_delta_net<decltype(kda_c)::value, decltype(keep_c)::value, decltype(wide_c)::value,
                               decltype(store_c)::value>(
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, qk_eps, qk_scale, stream);
    };
    const auto launch_s = [&](auto kda_c, auto keep_c, auto wide_c) {
        // only for the cache: a plain dst tail is read right away by the cpy, from L2
        if (g_ggml_sycl_gdn_state_store && cache != nullptr) {
            launch(kda_c, keep_c, wide_c, std::true_type{});
        } else {
            launch(kda_c, keep_c, wide_c, std::false_type{});
        }
    };
    const auto launch_w = [&](auto kda_c, auto keep_c) {
        if (wide) {
            launch_s(kda_c, keep_c, std::true_type{});
        } else {
            launch_s(kda_c, keep_c, std::false_type{});
        }
    };

    if (kda) {
        if (keep_rs) { launch_w(std::true_type{}, std::true_type{}); }
        else         { launch_w(std::true_type{}, std::false_type{}); }
    } else {
        if (keep_rs) { launch_w(std::false_type{}, std::true_type{}); }
        else         { launch_w(std::false_type{}, std::false_type{}); }
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
