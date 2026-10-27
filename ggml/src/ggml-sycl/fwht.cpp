#include "fwht.hpp"
#include "kv-soa.hpp"

#include <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
static constexpr float H12[12][12] = {
    { P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, P, P, P, N, N, N, P, N },
    { P, N, N, P, N, P, P, P, N, N, N, P },
    { P, P, N, N, P, N, P, P, P, N, N, N },
    { P, N, P, N, N, P, N, P, P, P, N, N },
    { P, N, N, P, N, N, P, N, P, P, P, N },
    { P, N, N, N, P, N, N, P, N, P, P, P },
    { P, P, N, N, N, P, N, N, P, N, P, P },
    { P, P, P, N, N, N, P, N, N, P, N, P },
    { P, P, P, P, N, N, N, P, N, N, P, N },
    { P, N, P, P, P, N, N, N, P, N, N, P },
    { P, P, N, P, P, P, N, N, N, P, N, N }
};

static constexpr float H20[20][20] = {
    { P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N },
    { P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P },
    { P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P },
    { P, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N },
    { P, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N },
    { P, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N },
    { P, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N },
    { P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P },
    { P, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N },
    { P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P },
    { P, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N },
    { P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P },
    { P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P },
    { P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P },
    { P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P },
    { P, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N },
    { P, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N },
    { P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P },
    { P, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N }
};

#undef P
#undef N

template <int N, typename T>
static void fwht_kernel(const T * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = static_cast<float>(src[i * WARP_SIZE + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <int N, typename T>
static void launch_fwht(const T * src, float * dst, const int64_t n_rows, const float scale,
                        dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <int N, int NT, typename T>
static void fwht_kernel_wide(const T * __restrict__ src,
                             float * __restrict__ dst,
                             const int64_t            n_rows,
                             const float              scale,
                             const sycl::nd_item<2> & item,
                             float *                  smem) {
    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    const int tid = item.get_local_id(1);

    float reg[el_w];
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = static_cast<float>(src[i * NT + tid]) * scale;
    }

    const sycl::sub_group sg   = item.get_sub_group();
    const int             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    for (int h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        sycl::group_barrier(item.get_group());
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        sycl::group_barrier(item.get_group());
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
        for (int j = 0; j < el_w; j += 2 * step) {
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <int N, int NT, typename T>
static void launch_fwht_wide(const T *       src,
                             float *         dst,
                             const int64_t   n_rows,
                             const float     scale,
                             dpct::queue_ptr stream) {
    const sycl::range<2> global(n_rows, NT);
    const sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <int N, int m, typename T>
static void kronecker_kernel(const T * __restrict__ src,
                             float * __restrict__ dst,
                             const int64_t            n_rows,
                             const float              scale,
                             const sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int blocks_per_group = N / m;
    constexpr int el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    float     reg[el_w * m];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * WARP_SIZE + lane;

#pragma unroll
        for (int j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<float>(src[b_idx * m + j]) * scale;
        }
    }

#pragma unroll
    for (int b = 0; b < el_w; ++b) {
        float z[m] = { 0.0f };

#pragma unroll
        for (int i = 0; i < m; ++i) {
#pragma unroll
            for (int j = 0; j < m; ++j) {
                const float h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        for (int i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
#pragma unroll
            for (int k = 0; k < m; ++k) {
                const float val  = reg[j * m + k];
                const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    for (int h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int s = 0; s < step; ++s) {
#pragma unroll
                for (int k = 0; k < m; ++k) {
                    const float x = reg[(j + s) * m + k];
                    const float y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * WARP_SIZE + lane;
#pragma unroll
        for (int k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <int N, int m, typename T>
static void launch_kronecker(const T *       src,
                             float *         dst,
                             const int64_t   n_rows,
                             const float     scale,
                             dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_rows, scale, item);
                         });
}

template <typename T>
static bool ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const int     n    = (int) src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const T *       src_d  = (const T *) src->data;
    float *         dst_d  = (float *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    const float scale = 1.0f / std::sqrt((float) n);

    switch (n) {
        case 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            return true;
        case 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            return true;
        case 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            return true;
        case 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            return true;
        case 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            return true;
        case 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        default:
            return false;
    }
}

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    switch (src->type) {
        case GGML_TYPE_F32:
            return ggml_sycl_op_fwht_impl<float>(ctx, src, dst);
        case GGML_TYPE_F16:
            return ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        default:
            return false;
    }
}

// GGML_SYCL_FUSE_FWHT_SET_ROWS: a Hadamard MUL_MAT whose only reader is a SET_ROWS into a q8_0 cache (qwen4exp:
// the rotated V rows), in one launch: one work-group per row rotates it with the butterflies above (same order)
// and quantizes its blocks straight into the cache row. The rotated tensor is never written.
static int32_t fwht_uses(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

// The SET_ROWS into a q8_0 cache that is the only reader of the Hadamard MUL_MAT at i (through single-use
// views), or -1. It may come some nodes later (qwen4exp: V is rotated before the K norm, rope, rotation and
// write, and written after them); nothing in between touches that cache, so it is written at the MUL_MAT.
static constexpr int FWHT_SET_ROWS_WINDOW = 64;

static int fwht_set_rows_match(const ggml_cgraph * cgraph, int i) {
    static const bool on = ggml_sycl_get_env("GGML_SYCL_FUSE_FWHT_SET_ROWS", 1) != 0;
    const ggml_tensor * mm = cgraph->nodes[i];
    if (!on || !g_ggml_sycl_enable_fusion || mm->op != GGML_OP_MUL_MAT ||
        ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD || mm->type != GGML_TYPE_F32 ||
        fwht_uses(cgraph, mm) != 1) {
        return -1;
    }
    const ggml_tensor * x = mm->src[1];
    const int64_t       N = x->ne[0];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_are_same_shape(x, mm) || N < 32 || N > 1024 ||
        (N & (N - 1)) != 0 || N % QK8_0 != 0) {
        return -1;
    }
    for (int k = i + 1; k < cgraph->n_nodes && k <= i + FWHT_SET_ROWS_WINDOW; ++k) {
        const ggml_tensor * sr = cgraph->nodes[k];
        if (sr->op != GGML_OP_SET_ROWS) {
            continue;
        }
        const ggml_tensor * sv   = sr->src[0];
        const ggml_tensor * root = sv;
        while (root && root != mm && (root->op == GGML_OP_RESHAPE || root->op == GGML_OP_VIEW) && fwht_uses(cgraph, root) == 1) {
            root = root->src[0];
        }
        if (root != mm) {
            continue;
        }
        if (sr->type != GGML_TYPE_Q8_0 || ggml_sycl_kv_is_soa(sr) || (sr->view_src && ggml_sycl_kv_is_soa(sr->view_src)) ||
            !ggml_is_contiguous(sv) || sv->ne[0] % N != 0 || ggml_nrows(x) != (sv->ne[0] / N) * ggml_nrows(sv) ||
            sr->ne[0] != sv->ne[0] || sr->nb[0] != sizeof(block_q8_0) || sr->ne[2] != 1 || sr->ne[3] != 1 || !sr->src[1] ||
            !ggml_is_contiguous(sr->src[1]) || (sr->src[1]->type != GGML_TYPE_I64 && sr->src[1]->type != GGML_TYPE_I32) ||
            ggml_nelements(sr->src[1]) != ggml_nrows(sv)) {
            return -1;
        }
        // the cache rows are written at the MUL_MAT: nothing up to the SET_ROWS may touch that cache or its indices
        const ggml_tensor * cache = sr->view_src ? sr->view_src : sr;
        for (int q = i + 1; q < k; ++q) {
            const ggml_tensor * n = cgraph->nodes[q];
            if (n == sr->src[1] || n->view_src == cache || n == cache) {
                return -1;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (n->src[s] && (n->src[s] == cache || n->src[s]->view_src == cache)) {
                    return -1;
                }
            }
        }
        return k;
    }
    return -1;
}

bool ggml_sycl_fwht_set_rows_done(const ggml_cgraph * cgraph, int k) {
    if (cgraph->nodes[k]->op != GGML_OP_SET_ROWS) {
        return false;
    }
    for (int i = k - 1; i >= 0 && i >= k - FWHT_SET_ROWS_WINDOW; --i) {
        if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT && fwht_set_rows_match(cgraph, i) == k) {
            return true;
        }
    }
    return false;
}

int ggml_sycl_fwht_set_rows(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int i) {
    const int k = fwht_set_rows_match(cgraph, i);
    if (k < 0) {
        return 0;
    }
    const ggml_tensor * mm = cgraph->nodes[i];
    const ggml_tensor * x  = mm->src[1];
    const int64_t       N  = x->ne[0];
    const ggml_tensor * sr = cgraph->nodes[k];
    const ggml_tensor * sv = sr->src[0];
    scope_op_debug_print scope_dbg_print(__func__, sr, 2, " (hadamard -> q8_0 cache)");

    const float * xd    = (const float *) x->data;
    char *        cache = (char *) sr->data;
    const size_t  crow  = sr->nb[1];
    const void *  cidx  = sr->src[1]->data;
    const bool    i64   = sr->src[1]->type == GGML_TYPE_I64;
    const int     n     = (int) N;
    const int     per   = (int) (sv->ne[0] / N);  // rows per cache row
    const int     rows  = (int) ggml_nrows(x);
    const float   hs    = 1.0f / sycl::sqrt((float) N);

    ctx.stream()->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> zs(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) rows * N, N), [=](sycl::nd_item<1> it) {
            const int t = (int) it.get_local_id(0);
            const int r = (int) it.get_group(0);
            float     v = xd[(size_t) r * n + t] * hs;
            for (int h = 1; h < n; h <<= 1) {
                sycl::group_barrier(it.get_group());
                zs[t] = v;
                sycl::group_barrier(it.get_group());
                const float q = zs[t ^ h];
                v             = (t & h) == 0 ? v + q : q - v;
            }
            sycl::group_barrier(it.get_group());
            zs[t] = v;
            sycl::group_barrier(it.get_group());
            if (t % QK8_0 == 0) {
                float amax = 0.0f;
                for (int j = 0; j < QK8_0; ++j) {
                    amax = sycl::fmax(amax, sycl::fabs(zs[t + j]));
                }
                const float   d   = amax / ((1 << 7) - 1);
                const float   id  = d ? 1.0f / d : 0.0f;
                const int     tok = r / per;
                const int64_t row = i64 ? ((const int64_t *) cidx)[tok] : ((const int32_t *) cidx)[tok];
                char * blk = cache + (size_t) row * crow + (size_t) (((r % per) * n + t) / QK8_0) * sizeof(block_q8_0);
                *(sycl::half *) blk = (sycl::half) d;
                int8_t * qs = (int8_t *) (blk + sizeof(sycl::half));
                for (int j = 0; j < QK8_0; ++j) {
                    qs[j] = (int8_t) sycl::round(zs[t + j] * id);
                }
            }
        });
    });
    return 1;  // the SET_ROWS is skipped where it stands, see ggml_sycl_fwht_set_rows_done
}
