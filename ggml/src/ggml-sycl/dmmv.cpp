#include <atomic>
#include <cstdio>

#include "convert.hpp"
#include "dmmv.hpp"
#include "dequantize.hpp"
#include "presets.hpp"

#if defined(__INTEL_LLVM_COMPILER)
    #if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #include <sycl/ext/oneapi/bfloat16.hpp>
        #define GGML_SYCL_DMMV_HAS_BF16
    #endif
    #include <sycl/ext/intel/esimd.hpp>
    #include "esimd.hpp"
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

static void convert_f16(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const sycl::half *x = (const sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#ifdef GGML_SYCL_DMMV_HAS_BF16
static void convert_bf16(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const sycl::ext::oneapi::bfloat16 *x = (const sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#endif

static void convert_f32(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const float * x = (const float *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static void dequantize_mul_mat_vec(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows,
                                   const sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);

    const int iter_stride = 2*GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_SYCL_F16

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#else
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#else
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

template <int qk, int qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
static void dequantize_mul_mat_vec_reorder(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows,
                                   const sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);


    const int ncols_left = ncols % (QK4_0*WARP_SIZE);
    const int ncols_align = ncols - ncols_left;
    const int iter_stride = 8*2*GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_SYCL_F16
    const char *d_ptr = (const char*)vx+ncols*nrows/2;
    int i=0;
    for (i = 0; i < ncols_align; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((const void *)d_ptr, ib, (const void *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#else
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    for (; i < ncols; i += iter_stride) {
        if (tid>=ncols_left/QK4_0) continue;
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((const void *)d_ptr, ib, (const void *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#else
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#else
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

static void convert_mul_mat_vec_f16_sycl(const void *vx, const dfloat *y,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, item_ct1);
            });
    }
}

#ifdef GGML_SYCL_DMMV_HAS_BF16
static void convert_mul_mat_vec_bf16_sycl(const void *vx, const dfloat *y,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16>(vx, y, dst, ncols,
                                                           nrows, item_ct1);
            });
    }
}
#endif

static void dequantize_mul_mat_vec_q2_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int in = tid % step;                           // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int s_offset = 8*im;
            const int y_offset = 128*im + l0;

            const float   * y = yy + i * QK_K + y_offset;
            const uint8_t * q = x[i].qs + q_offset;

            const uint32_t * a = (const uint32_t *)(x[i].scales + s_offset);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            float sum1 = 0, sum2 = 0;
            for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
                sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                      + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                      + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                      + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                      + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                      + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                      + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                      +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
                sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                      + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

            }
            tmp += dall * sum1 - dmin * sum2;
        }

    }
#else
    const int tid = item_ct1.get_local_id(2) /
                    (2 * K_QUANTS_PER_ITERATION); // 0...15 or 0...7
    const int ix = item_ct1.get_local_id(2) %
                   (2 * K_QUANTS_PER_ITERATION); // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    const uint8_t * d = (const uint8_t *)uaux;


    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint32_t * s = (const uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        const sycl::float2 dall =
            x[i].dm.convert<float, sycl::rounding_mode::automatic>();

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#endif

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

static void dequantize_mul_mat_vec_q2_k_reorder(const void *__restrict__ vx,
                                                const float *__restrict__ yy,
                                                float *__restrict__ dst,
                                                const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t     * qs_base     = (const uint8_t *)vx;
    const uint8_t     * scales_base = qs_base + (size_t)nb * (QK_K / 4);
    const sycl::half2 * dm_base     = (const sycl::half2 *)(scales_base + (size_t)nb * (QK_K / 16));

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int in = tid % step;                           // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const sycl::half2 dm_val = dm_base[bi];
        const float dall = dm_val[0];
        const float dmin = dm_val[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int s_offset = 8*im;
            const int y_offset = 128*im + l0;

            const float   * y = yy + i * QK_K + y_offset;
            const uint8_t * q = qs_base + bi * (QK_K / 4) + q_offset;

            const uint32_t * a = (const uint32_t *)(scales_base + bi * (QK_K / 16) + s_offset);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            float sum1 = 0, sum2 = 0;
            for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
                sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                      + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                      + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                      + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                      + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                      + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                      + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                      +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
                sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                      + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

            }
            tmp += dall * sum1 - dmin * sum2;
        }
    }
#else
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(ncols);
    GGML_UNUSED(item_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#endif

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

static void dequantize_mul_mat_vec_q3_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int in = tid % step;                           // 0...15 or 0...7

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const uint8_t * h = x[i].hmask + l0;
        const float d = x[i].d;

        for (int im = 0; im < 2; ++im) {
            const int q_offset =  32*im + l0;
            const int y_offset = 128*im + l0;
            const uint16_t s_shift = 4*im;
            const uint8_t m = 1 << (4*im);

            const float   * y  = yy + i * QK_K + y_offset;
            const uint8_t * q = x[i].qs + q_offset;

            const uint16_t * a = (const uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

            float sum = 0;
            for (int l = 0; l < n; ++l) {
                sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * sum;
        }

    }
#else

    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    const int in = offset/8;                                 // 0 or 1
    const int im = offset%8;                                 // 0...7

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint8_t * s = x[i].scales;

        const float dall = (float)x[i].d;

        float sum = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t hl = x[i].hmask[im+l] >> in;
            const uint8_t ql = q[l];
            sum += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += sum;
    }
#endif

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

static void dequantize_mul_mat_vec_q3_k_reorder(const void *__restrict__ vx,
                                                const float *__restrict__ yy,
                                                float *__restrict__ dst,
                                                const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t   * qs_base     = (const uint8_t *)vx;
    const uint8_t   * hmask_base  = qs_base + (size_t)nb * (QK_K / 4);
    const uint8_t   * scales_base = hmask_base + (size_t)nb * (QK_K / 8);
    const sycl::half * d_base     = (const sycl::half *)(scales_base + (size_t)nb * 12);

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int in = tid % step;                           // 0...15 or 0...7

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const uint8_t * h  = hmask_base + bi * (QK_K / 8) + l0;

        const float d = d_base[bi];

        for (int im = 0; im < 2; ++im) {
            const int q_offset =  32*im + l0;
            const int y_offset = 128*im + l0;
            const uint16_t s_shift = 4*im;
            const uint8_t m = 1 << (4*im);

            const float   * y  = yy + i * QK_K + y_offset;
            const uint8_t * q  = qs_base + bi * (QK_K / 4) + q_offset;

            const uint16_t * a = (const uint16_t *)(scales_base + bi * 12);
            utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

            float sum = 0;
            for (int l = 0; l < n; ++l) {
                sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * sum;
        }
    }
#else
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(ncols);
    GGML_UNUSED(item_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#endif

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

static void dequantize_mul_mat_vec_q4_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il_base = tid/step;                        // 0 or 1 (was 0...3)
    const int ir  = tid - step*il_base;                  // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int in = il_base%2;

    const int l0 = n*(2*ir + in);

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int y_offset = 64*im + l0;

            const float   * y1 = yy + i*QK_K + y_offset;
            const float   * y2 = y1 + 128;

            const uint16_t * a = (const uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
            const uint32_t * q1 = (const uint32_t *)(x[i].qs + q_offset);
            const uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            for (int l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#else
            const uint16_t * q1 = (const uint16_t *)(x[i].qs + q_offset);
            const uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            for (int l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#endif
        }

    }
#else
    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const float d = (float)x[i].dm[0];
        const float m = (float)x[i].dm[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q4_k_reorder(const void *__restrict__ vx,
                                                const float *__restrict__ yy,
                                                float *__restrict__ dst,
                                                const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t     * qs_base     = (const uint8_t *)vx;
    const uint8_t     * scales_base = qs_base + (size_t)nb * (QK_K / 2);
    const sycl::half2 * dm_base     = (const sycl::half2 *)(scales_base + (size_t)nb * K_SCALE_SIZE);

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il_base = tid/step;                        // 0 or 1 (was 0...3)
    const int ir  = tid - step*il_base;                  // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int in = il_base%2;

    const int l0 = n*(2*ir + in);

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const sycl::half2 dm_val = dm_base[bi];
        const float dall = dm_val[0];
        const float dmin = dm_val[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int y_offset = 64*im + l0;

            const float   * y1 = yy + i*QK_K + y_offset;
            const float   * y2 = y1 + 128;

            const uint16_t * a = (const uint16_t *)(scales_base + bi * K_SCALE_SIZE);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
            const uint32_t * q1 = (const uint32_t *)(qs_base + bi * (QK_K / 2) + q_offset);
            const uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            for (int l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#else
            const uint16_t * q1 = (const uint16_t *)(qs_base + bi * (QK_K / 2) + q_offset);
            const uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            for (int l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#endif
        }

    }
#else
    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const uint8_t * q = qs_base + bi * (QK_K / 2) + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)(scales_base + bi * K_SCALE_SIZE);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const sycl::half2 dm_val = dm_base[bi];
        const float d = (float)dm_val[0];
        const float m = (float)dm_val[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q5_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = item_ct1.get_local_id(2) / 2; // 0...7
    const int ix = item_ct1.get_local_id(2) % 2;

    const int il_base = tid/4;     // 0 or 1 (was 0...3)
    const int ir  = tid - 4*il_base;// 0...3
    const int n   = 2;

    const int in = il_base%2;

    const int l0 = n*(2*ir + in);

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * qh  = x[i].qh + l0;
        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int y_offset = 64*im + l0;

            const uint8_t hm1  = 1 << (2*im);
            const uint8_t hm2  = hm1 << 4;

            const uint8_t * ql1 = x[i].qs + q_offset;
            const float   * y1  = yy + i*QK_K + y_offset;
            const float   * y2  = y1 + 128;

            const uint16_t * a = (const uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            const uint16_t * q1 = (const uint16_t *)ql1;
            const uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            for (int l = 0; l < n; ++l) {
                sum.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                sum.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                sum.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                sum.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (sum.x() * sc[0] + sum.y() * sc[1] + sum.z() * sc[4] +
                           sum.w() * sc[5]) -
                   dmin * smin;
        }
    }

#else
    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);
    const int step = tid * K_QUANTS_PER_ITERATION;
    const int im = step/8;
    const int in = step%8;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const int8_t  * s = x[i].scales;
        const float   * y = yy + i*QK_K + step;
        const float     d = x[i].d;
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            const uint8_t h = x[i].qh[in+j] >> im;
            sum += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += sum;
    }
#endif

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

static void dequantize_mul_mat_vec_q5_k_reorder(const void *__restrict__ vx,
                                                const float *__restrict__ yy,
                                                float *__restrict__ dst,
                                                const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t     * qs_base     = (const uint8_t *)vx;
    const uint8_t     * qh_base     = qs_base + (size_t)nb * (QK_K / 2);
    const uint8_t     * scales_base = qh_base + (size_t)nb * (QK_K / 8);
    const sycl::half2 * dm_base     = (const sycl::half2 *)(scales_base + (size_t)nb * K_SCALE_SIZE);

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = item_ct1.get_local_id(2) / 2; // 0...15
    const int ix = item_ct1.get_local_id(2) % 2;

    const int il_base = tid/4;      // 0...3
    const int ir  = tid - 4*il_base;// 0...3
    const int n   = 2;

    const int in = il_base%2;

    const int l0 = n*(2*ir + in);

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {
        const int bi = ib0 + i;

        const uint8_t * qh  = qh_base + bi * (QK_K / 8) + l0;
        const sycl::half2 dm_val = dm_base[bi];
        const float dall = dm_val[0];
        const float dmin = dm_val[1];

        for (int im = 0; im < 2; ++im) {
            const int q_offset = 32*im + l0;
            const int y_offset = 64*im + l0;

            const uint8_t hm1  = 1 << (2*im);
            const uint8_t hm2  = hm1 << 4;

            const uint8_t * ql1 = qs_base + bi * (QK_K / 2) + q_offset;
            const float   * y1  = yy + i*QK_K + y_offset;
            const float   * y2  = y1 + 128;

            const uint16_t * a = (const uint16_t *)(scales_base + bi * K_SCALE_SIZE);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
            float smin = 0;
            const uint16_t * q1 = (const uint16_t *)ql1;
            const uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            for (int l = 0; l < n; ++l) {
                sum.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                sum.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                sum.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                sum.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (sum.x() * sc[0] + sum.y() * sc[1] + sum.z() * sc[4] +
                           sum.w() * sc[5]) -
                   dmin * smin;
        }
    }
#else
    // The reordered Q5_K layout is only produced for QK_K == 256.
#endif

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

static void dequantize_mul_mat_vec_q6_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

#if QK_K == 256

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int in = tid % step;                           // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float d = x[i].d;

        for (int im = 0; im < 2; ++im) {
            const int ql_offset = 64*im + l0;
            const int qh_offset = 32*im + l0;
            const int s_offset  =  8*im + is;
            const int y_offset = 128*im + l0;

            const float   * y  = yy + i * QK_K + y_offset;
            const uint8_t * ql = x[i].ql + ql_offset;
            const uint8_t * qh = x[i].qh + qh_offset;
            const int8_t  * s  = x[i].scales + s_offset;

#if K_QUANTS_PER_ITERATION == 1
            float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += sum;
#else
            float sum = 0;
            for (int l = 0; l < 4; ++l) {
                sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += sum;
#endif
        }

    }

#else

    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = x[i].ql + step;
        const uint8_t * qh = x[i].qh + step;
        const int8_t  * s  = x[i].scales;

        const float d = x[i+0].d;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q6_k_reorder(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows,
                                                const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row >= nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    const int nb = nrows * num_blocks_per_row;
    const uint8_t   * ql_base     = (const uint8_t *)vx;
    const uint8_t   * qh_base     = ql_base + (size_t)nb * (QK_K / 2);
    const int8_t    * scales_base = (const int8_t *)(qh_base + (size_t)nb * (QK_K / 4));
    const sycl::half * d_base     = (const sycl::half *)((const uint8_t *)scales_base + (size_t)nb * (QK_K / 16));

#if QK_K == 256

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int in = tid % step;                           // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const float d = d_base[bi];

        for (int im = 0; im < 2; ++im) {
            const int ql_offset = 64*im + l0;
            const int qh_offset = 32*im + l0;
            const int s_offset  =  8*im + is;
            const int y_offset = 128*im + l0;

            const float   * y  = yy + i * QK_K + y_offset;
            const uint8_t * ql = ql_base + bi * (QK_K / 2) + ql_offset;
            const uint8_t * qh = qh_base + bi * (QK_K / 4) + qh_offset;
            const int8_t  * s  = scales_base + bi * (QK_K / 16) + s_offset;

#if K_QUANTS_PER_ITERATION == 1
            float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += sum;
#else
            float sum = 0;
            for (int l = 0; l < 4; ++l) {
                sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += sum;
#endif
        }

    }

#else

    const int tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const int bi = ib0 + i;

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = ql_base + bi * (QK_K / 2) + step;
        const uint8_t * qh = qh_base + bi * (QK_K / 4) + step;
        const int8_t  * s  = scales_base + bi * (QK_K / 16);

        const float d = d_base[bi];

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q4_0_sycl_reorder(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}


static void dequantize_mul_mat_vec_q4_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q1_0_sycl_reorder(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_reorder<QK1_0, QR1_0, dequantize_q1_0_reorder>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q1_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK1_0, QR1_0, dequantize_q1_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q4_1_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q5_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q5_1_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl_reorder(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                // Q8_0 reorder layout: [all qs (ncols*nrows bytes)][all d values]
                // Cannot reuse dequantize_mul_mat_vec_reorder template because it has
                // Q4_0-specific constants hardcoded (d_ptr offset and qs stride).
                const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
                if (row >= nrows) return;

                const int tid = item_ct1.get_local_id(2);
                const int iter_stride = 8*2*GGML_SYCL_DMMV_X;
                const int vals_per_iter = iter_stride / WARP_SIZE;
                const int ncols_left = ncols % (QK8_0*WARP_SIZE);
                const int ncols_align = ncols - ncols_left;

#ifdef GGML_SYCL_F16
                sycl::half2 tmp = {0.0f, 0.0f};
#else
                float tmp = 0.0f;
#endif
                const char *d_ptr = (const char*)vx + ncols*nrows;  // d after all qs

                int i = 0;
                for (i = 0; i < ncols_align; i += iter_stride) {
                    const int col = i + vals_per_iter*tid;
                    const int ib = (row*ncols + col)/QK8_0;
                    const int iqs = col % QK8_0;

#pragma unroll
                    for (int j = 0; j < vals_per_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((const void *)d_ptr, ib, (const void *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ifdef GGML_SYCL_F16
                        dfloat2 t1{y[col + j + 0], y[col + j + 1]};
                        tmp += v * t1;
#else
                        tmp += v.x() * y[col + j + 0];
                        tmp += v.y() * y[col + j + 1];
#endif
                    }
                }

                // handle remaining columns
                for (; i < ncols; i += iter_stride) {
                    if (tid >= ncols_left/QK8_0) continue;
                    const int col = i + vals_per_iter*tid;
                    const int ib = (row*ncols + col)/QK8_0;
                    const int iqs = col % QK8_0;

#pragma unroll
                    for (int j = 0; j < vals_per_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((const void *)d_ptr, ib, (const void *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ifdef GGML_SYCL_F16
                        dfloat2 t1{y[col + j + 0], y[col + j + 1]};
                        tmp += v * t1;
#else
                        tmp += v.x() * y[col + j + 0];
                        tmp += v.y() * y[col + j + 1];
#endif
                    }
                }

                // reduce
                const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
                for (int mask = mask_start; mask > 0; mask >>= 1) {
                    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
                }

                if (tid == 0) {
#ifdef GGML_SYCL_F16
                    dst[row] = tmp.x() + tmp.y();
#else
                    dst[row] = tmp;
#endif
                }
            });
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q2_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q2_K_sycl_reorder(const void *vx, const float *y,
                                                     float *dst, const int ncols,
                                                     const int nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q2_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q3_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q3_K_sycl_reorder(const void *vx, const float *y,
                                                     float *dst, const int ncols,
                                                     const int nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q3_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q4_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q5_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q6_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
using ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_SIZE;

// generic reordered dequantize-matvec: each work-group owns a pair of
// consecutive output rows and updates one 32-wide accumulator per row
template <ggml_type T>
ESIMD_INLINE void dequantize_mul_mat_vec_reorder_esimd(
        const void * vx, const float * y, float * dst,
        const int ncols, const int nrows,
        sycl::local_accessor<float, 1> lmem,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    using traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    const int    num_blocks_per_row = ncols / QK_K;
    const size_t nb = (size_t) nrows * num_blocks_per_row;
    const auto   ps = traits::make_ptrs(vx, nb);

    const int  tid      = it.get_local_id(0);
    const int  row_pair = it.get_group(0);
    const int  row0     = row_pair * 2; // two consecutive output rows
    const bool has_row1 = row0 + 1 < nrows;

    // one 32-wide accumulator per output row (small footprint, no spill)
    simd<float, 32> acc0 = 0.0f;
    simd<float, 32> acc1 = 0.0f;

    for (int ib = tid; ib < num_blocks_per_row; ib += GGML_SYCL_DMMV_ESIMD_WG_SIZE) {
        simd<float, 256> y_vec = block_load<float, 256>(y + (size_t) ib * QK_K);

        const size_t bi0 = (size_t) (row0 + 0) * num_blocks_per_row + ib;
        const size_t bi1 = (size_t) (row0 + 1) * num_blocks_per_row + ib;

        traits::mac_pair(ps, bi0, ps, bi1, has_row1, y_vec, acc0, acc1);
    }

    lmem[tid * 2 + 0] = reduce<float>(acc0, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<float>(acc1, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_SIZE; ++p) {
            sum0 += lmem[p * 2 + 0];
            sum1 += lmem[p * 2 + 1];
        }
        dst[row0 + 0] = sum0;
        if (has_row1) {
            dst[row0 + 1] = sum1;
        }
    }
}

static void dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    vx, y, dst, ncols, nrows, lmem, it);
            });
    });
}

static void dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    vx, y, dst, ncols, nrows, lmem, it);
            });
    });
}

static void dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    vx, y, dst, ncols, nrows, lmem, it);
            });
    });
}

static void dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    vx, y, dst, ncols, nrows, lmem, it);
            });
    });
}

static void dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q6_K>(
                    vx, y, dst, ncols, nrows, lmem, it);
            });
    });
}

// Q8_0 SOA reorder layout: [qs: nb*QK8_0] [d: nb*sizeof(half)].
// Process eight blocks per stripe and process remaining blocks one at a time.
template <int NBLK>
ESIMD_INLINE void q8_0_mac_stripe(
        const int8_t * qs_a, const int8_t * qs_b,
        const sycl::half * d_a, const sycl::half * d_b, bool has_b,
        sycl::ext::intel::esimd::simd<float, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<float, 32> & acc_a,
        sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
    using namespace sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = block_load<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = block_load<sycl::half, NBLK>(d_a, element_aligned_tag{});
    simd<sycl::half, NBLK>  db = 0;
    if (has_b) {
        qb = block_load<int8_t, 32 * NBLK>(qs_b);
        db = block_load<sycl::half, NBLK>(d_b, element_aligned_tag{});
    }

    simd<float, NBLK> da_f = convert<float>(da);
    simd<float, NBLK> db_f = convert<float>(db);

#pragma unroll
    for (int s = 0; s < NBLK; ++s) {
        simd<float, 32>  y_s  = y_vec.template select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.template select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.template select<32, 1>(s * 32);
        const float sa = da_f[s];
        const float sb = db_f[s];
        acc_a += y_s * (convert<float>(qa_s) * sa);
        acc_b += y_s * (convert<float>(qb_s) * sb);
    }
}

// TAIL: after the full stripes, take the leftover blocks in stripes of 4 and then 2 per thread
// before falling back to one block at a time (GGML_SYCL_Q8_0_MMV_TAIL). With K=2560 the 80 blocks of a row
// leave 16 after one WG*STRIPE pass, which otherwise cost two rounds of 32-byte loads.
template <int WG, bool TAIL = false>
ESIMD_INLINE void dequantize_mul_mat_vec_q8_0_reorder_esimd(
        const void * vx, const float * y, float * dst,
        const int ncols, const int nrows,
        sycl::local_accessor<float, 1> lmem,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;

    constexpr int STRIPE = 8;

    const int          nblk_row = ncols / QK8_0;
    const size_t       nb       = (size_t) nrows * nblk_row;
    const int8_t *     qs       = (const int8_t *) vx;
    const sycl::half * d        = (const sycl::half *) (qs + nb * QK8_0);

    const int  tid      = it.get_local_id(0);
    const int  row_pair = it.get_group(0);
    const int  row0     = row_pair * 2;
    const bool has_row1 = row0 + 1 < nrows;

    const size_t base0 = (size_t) row0 * nblk_row;
    const size_t base1 = has_row1 ? (size_t) (row0 + 1) * nblk_row : base0;

    simd<float, 32> acc0 = 0.0f;
    simd<float, 32> acc1 = 0.0f;

    // Each thread processes one contiguous stripe.
    int ib = 0;
    for (; ib + WG * STRIPE <= nblk_row; ib += WG * STRIPE) {
        const int b = ib + tid * STRIPE;
        simd<float, 256> y_vec = block_load<float, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs + (base0 + b) * QK8_0, qs + (base1 + b) * QK8_0,
                                d + base0 + b, d + base1 + b, has_row1, y_vec, acc0, acc1);
    }

    if constexpr (TAIL) {
        for (; ib + WG * 4 <= nblk_row; ib += WG * 4) {
            const int b = ib + tid * 4;
            simd<float, 128> y_vec = block_load<float, 128>(y + (size_t) b * QK8_0);
            q8_0_mac_stripe<4>(qs + (base0 + b) * QK8_0, qs + (base1 + b) * QK8_0,
                               d + base0 + b, d + base1 + b, has_row1, y_vec, acc0, acc1);
        }
        for (; ib + WG * 2 <= nblk_row; ib += WG * 2) {
            const int b = ib + tid * 2;
            simd<float, 64> y_vec = block_load<float, 64>(y + (size_t) b * QK8_0);
            q8_0_mac_stripe<2>(qs + (base0 + b) * QK8_0, qs + (base1 + b) * QK8_0,
                               d + base0 + b, d + base1 + b, has_row1, y_vec, acc0, acc1);
        }
    }

    // Distribute remaining blocks across the work-group.
    for (int b = ib + tid; b < nblk_row; b += WG) {
        simd<float, 32> y_vec = block_load<float, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs + (base0 + b) * QK8_0, qs + (base1 + b) * QK8_0,
                           d + base0 + b, d + base1 + b, has_row1, y_vec, acc0, acc1);
    }

    lmem[tid * 2 + 0] = reduce<float>(acc0, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<float>(acc1, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int p = 0; p < WG; ++p) {
            sum0 += lmem[p * 2 + 0];
            sum1 += lmem[p * 2 + 1];
        }
        dst[row0 + 0] = sum0;
        if (has_row1) {
            dst[row0 + 1] = sum1;
        }
    }
}

template <int WG, bool TAIL>
static void q8_0_esimd_launch_tail(const void * vx, const float * y, float * dst, const int ncols,
                                   const int nrows, dpct::queue_ptr stream) {
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG, TAIL>(vx, y, dst, ncols, nrows, lmem, it);
            });
    });
}

template <int WG>
static void q8_0_esimd_launch(const void * vx, const float * y, float * dst, const int ncols,
                              const int nrows, dpct::queue_ptr stream) {
    if (g_ggml_sycl_q8_0_mmv_tail) {
        q8_0_esimd_launch_tail<WG, true>(vx, y, dst, ncols, nrows, stream);
    } else {
        q8_0_esimd_launch_tail<WG, false>(vx, y, dst, ncols, nrows, stream);
    }
}

// GGML_SYCL_Q8_0_MMV_SHAPES: the same stripes over NR rows per work-group of WG threads, picked by row length.
// Streamed from VRAM (an L2 flush before each launch), K=2560 runs 2.4-16% faster with 4 rows on 4 threads, and
// K=6144 5% faster with 2 rows on 16 threads. Not the arithmetic of the multi-column kernels below, which keep the
// 2-row decomposition, so N=1 and N>1 agree to float rounding only.

// NR rows of a work-group, one NBLK-block stripe of each: all the weight loads are issued before the MACs
template <int NBLK, int NR>
ESIMD_INLINE void q8_0_mac_stripe_rows(
        const int8_t * qs, const sycl::half * d, const size_t * base, const int b, const float * y,
        sycl::ext::intel::esimd::simd<float, 32> * acc) {
    using namespace sycl::ext::intel::esimd;

    simd<float, 32 * NBLK> y_vec = block_load<float, 32 * NBLK>(y + (size_t) b * QK8_0);
    simd<int8_t, 32 * NBLK> q[NR];
    simd<sycl::half, NBLK>  dh[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        q[r]  = block_load<int8_t, 32 * NBLK>(qs + (base[r] + b) * QK8_0);
        dh[r] = block_load<sycl::half, NBLK>(d + base[r] + b, element_aligned_tag{});
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        simd<float, NBLK> df = convert<float>(dh[r]);
#pragma unroll
        for (int s = 0; s < NBLK; ++s) {
            simd<float, 32>  y_s = y_vec.template select<32, 1>(s * 32);
            simd<int8_t, 32> q_s = q[r].template select<32, 1>(s * 32);
            const float      sc  = df[s];
            acc[r] += y_s * (convert<float>(q_s) * sc);
        }
    }
}

template <int WG, int NR>
ESIMD_INLINE void dequantize_mul_mat_vec_q8_0_reorder_esimd_rows(
        const void * vx, const float * y, float * dst,
        const int ncols, const int nrows,
        sycl::local_accessor<float, 1> lmem,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;

    constexpr int STRIPE = 8;

    const int          nblk_row = ncols / QK8_0;
    const size_t       nb       = (size_t) nrows * nblk_row;
    const int8_t *     qs       = (const int8_t *) vx;
    const sycl::half * d        = (const sycl::half *) (qs + nb * QK8_0);

    const int tid  = it.get_local_id(0);
    const int row0 = it.get_group(0) * NR;

    // rows past the end read the last row again and are not stored
    size_t base[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        base[r] = (size_t) (row0 + r < nrows ? row0 + r : nrows - 1) * nblk_row;
    }

    simd<float, 32> acc[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        acc[r] = 0.0f;
    }

    int ib = 0;
    for (; ib + WG * STRIPE <= nblk_row; ib += WG * STRIPE) {
        q8_0_mac_stripe_rows<STRIPE, NR>(qs, d, base, ib + tid * STRIPE, y, acc);
    }
    for (; ib + WG * 4 <= nblk_row; ib += WG * 4) {
        q8_0_mac_stripe_rows<4, NR>(qs, d, base, ib + tid * 4, y, acc);
    }
    for (; ib + WG * 2 <= nblk_row; ib += WG * 2) {
        q8_0_mac_stripe_rows<2, NR>(qs, d, base, ib + tid * 2, y, acc);
    }
    for (int b = ib + tid; b < nblk_row; b += WG) {
        q8_0_mac_stripe_rows<1, NR>(qs, d, base, b, y, acc);
    }

#pragma unroll
    for (int r = 0; r < NR; ++r) {
        lmem[tid * NR + r] = reduce<float>(acc[r], std::plus<>{});
    }
    it.barrier(sycl::access::fence_space::local_space);

    if (tid < NR && row0 + tid < nrows) {
        float sum = 0.0f;
        for (int p = 0; p < WG; ++p) {
            sum += lmem[p * NR + tid];
        }
        dst[row0 + tid] = sum;
    }
}

template <int WG, int NR>
static void q8_0_esimd_launch_rows(const void * vx, const float * y, float * dst, const int ncols,
                                   const int nrows, dpct::queue_ptr stream) {
    const int workgroups = (nrows + NR - 1) / NR;
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(WG * NR), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd_rows<WG, NR>(vx, y, dst, ncols, nrows, lmem, it);
            });
    });
}

static void dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(const void *vx, const float *y,
                                                           float *dst, const int ncols,
                                                           const int nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    // Scale the work-group with the number of blocks per row.
    const int nblk_row = ncols / QK8_0;
    if (g_ggml_sycl_q8_0_mmv_shapes && nblk_row >= 128) {
        q8_0_esimd_launch_rows<16, 2>(vx, y, dst, ncols, nrows, stream);
    } else if (g_ggml_sycl_q8_0_mmv_shapes && nblk_row >= 64) {
        q8_0_esimd_launch_rows<4, 4>(vx, y, dst, ncols, nrows, stream);
    } else if (nblk_row >= 64) {
        q8_0_esimd_launch<8>(vx, y, dst, ncols, nrows, stream);
    } else if (nblk_row >= 32) {
        q8_0_esimd_launch<4>(vx, y, dst, ncols, nrows, stream);
    } else if (nblk_row >= 16) {
        q8_0_esimd_launch<2>(vx, y, dst, ncols, nrows, stream);
    } else {
        q8_0_esimd_launch<1>(vx, y, dst, ncols, nrows, stream);
    }
}


// Multi-column variants of the two reordered ESIMD kernels above, for 2..GGML_SYCL_ESIMD_MAX_NCOLS activation columns
// (MTP verification, small-batch decode). Each work-group keeps the N=1 decomposition: the same
// row pair, the same WG threads, the same stripes, the same per-lane accumulation order and the
// same lane-0 reduction. Each column agrees with the N=1 kernel to float rounding (measured
// max|diff| / max|y| <= 2.8e-7 on the qwen4exp shapes; the compiler contracts the products into
// FMAs differently, so not bitwise), where the q8_1 MMVQ path it replaces differs by ~4e-3.
// What changes is that a weight stripe is loaded and dequantized once and then MACed against
// every column, so the weight bytes - which are what bound these kernels - are read once for all
// columns. The activations are read as f32, like the N=1 kernels: no q8_1 quantize launch.
// Kept separate from the N=1 kernels so that path stays the code it was.

// dequantize one q8_0 stripe of a row into floats, with the arithmetic of q8_0_mac_stripe
template <int NBLK>
ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 32 * NBLK> q8_0_dequant_stripe(
        const int8_t * qs, const sycl::half * d, bool valid) {
    using namespace sycl::ext::intel::esimd;
    simd<int8_t, 32 * NBLK> q  = 0;
    simd<sycl::half, NBLK>  dh = 0;
    if (valid) {
        q  = block_load<int8_t, 32 * NBLK>(qs);
        dh = block_load<sycl::half, NBLK>(d, element_aligned_tag{});
    }
    simd<float, NBLK>       df = convert<float>(dh);
    simd<float, 32 * NBLK>  w;
#pragma unroll
    for (int s = 0; s < NBLK; ++s) {
        simd<int8_t, 32> q_s = q.template select<32, 1>(s * 32);
        const float      sc  = df[s];
        w.template select<32, 1>(s * 32) = convert<float>(q_s) * sc;
    }
    return w;
}

template <int NBLK, int NCOLS>
ESIMD_INLINE void q8_0_mac_stripe_ncols(
        const int8_t * qs_a, const int8_t * qs_b,
        const sycl::half * d_a, const sycl::half * d_b, bool has_b,
        const float * y, const int stride_col_y,
        sycl::ext::intel::esimd::simd<float, 32 * NCOLS> & acc_a,
        sycl::ext::intel::esimd::simd<float, 32 * NCOLS> & acc_b) {
    using namespace sycl::ext::intel::esimd;

    simd<float, 32 * NBLK> wa = q8_0_dequant_stripe<NBLK>(qs_a, d_a, true);
    simd<float, 32 * NBLK> wb = q8_0_dequant_stripe<NBLK>(qs_b, d_b, has_b);

#pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        simd<float, 32 * NBLK> y_vec = block_load<float, 32 * NBLK>(y + (size_t) c * stride_col_y);
        simd<float, 32> a = acc_a.template select<32, 1>(c * 32);
        simd<float, 32> b = acc_b.template select<32, 1>(c * 32);
#pragma unroll
        for (int s = 0; s < NBLK; ++s) {
            simd<float, 32> y_s = y_vec.template select<32, 1>(s * 32);
            a += y_s * simd<float, 32>(wa.template select<32, 1>(s * 32));
            b += y_s * simd<float, 32>(wb.template select<32, 1>(s * 32));
        }
        acc_a.template select<32, 1>(c * 32) = a;
        acc_b.template select<32, 1>(c * 32) = b;
    }
}

// lmem layout [thread][column][row of the pair]; lane 0 sums the threads in the N=1 order
template <int WG, int NCOLS>
ESIMD_INLINE void esimd_ncols_epilogue(
        sycl::ext::intel::esimd::simd<float, 32 * NCOLS> & acc0,
        sycl::ext::intel::esimd::simd<float, 32 * NCOLS> & acc1,
        float * dst, const int row0, const bool has_row1, const int stride_col_dst,
        sycl::local_accessor<float, 1> lmem, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    const int tid = it.get_local_id(0);
#pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        lmem[(tid * NCOLS + c) * 2 + 0] = reduce<float>(simd<float, 32>(acc0.template select<32, 1>(c * 32)), std::plus<>{});
        lmem[(tid * NCOLS + c) * 2 + 1] = reduce<float>(simd<float, 32>(acc1.template select<32, 1>(c * 32)), std::plus<>{});
    }
    it.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            for (int p = 0; p < WG; ++p) {
                sum0 += lmem[(p * NCOLS + c) * 2 + 0];
                sum1 += lmem[(p * NCOLS + c) * 2 + 1];
            }
            dst[(size_t) c * stride_col_dst + row0 + 0] = sum0;
            if (has_row1) {
                dst[(size_t) c * stride_col_dst + row0 + 1] = sum1;
            }
        }
    }
}

template <int WG, int NCOLS>
ESIMD_INLINE void dequantize_mul_mat_vec_q8_0_reorder_esimd_ncols(
        const void * vx, const float * y, float * dst,
        const int ncols, const int nrows, const int stride_col_y, const int stride_col_dst,
        sycl::local_accessor<float, 1> lmem,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;

    constexpr int STRIPE = 8;

    const int          nblk_row = ncols / QK8_0;
    const size_t       nb       = (size_t) nrows * nblk_row;
    const int8_t *     qs       = (const int8_t *) vx;
    const sycl::half * d        = (const sycl::half *) (qs + nb * QK8_0);

    const int  tid      = it.get_local_id(0);
    const int  row0     = it.get_group(0) * 2;
    const bool has_row1 = row0 + 1 < nrows;

    const size_t base0 = (size_t) (row0 + 0) * nblk_row;
    const size_t base1 = (size_t) (row0 + 1) * nblk_row;

    simd<float, 32 * NCOLS> acc0 = 0.0f;
    simd<float, 32 * NCOLS> acc1 = 0.0f;

    int ib = 0;
    for (; ib + WG * STRIPE <= nblk_row; ib += WG * STRIPE) {
        const int b = ib + tid * STRIPE;
        q8_0_mac_stripe_ncols<STRIPE, NCOLS>(qs + (base0 + b) * QK8_0, qs + (base1 + b) * QK8_0,
                                             d + base0 + b, d + base1 + b, has_row1,
                                             y + (size_t) b * QK8_0, stride_col_y, acc0, acc1);
    }
    for (int b = ib + tid; b < nblk_row; b += WG) {
        q8_0_mac_stripe_ncols<1, NCOLS>(qs + (base0 + b) * QK8_0, qs + (base1 + b) * QK8_0,
                                        d + base0 + b, d + base1 + b, has_row1,
                                        y + (size_t) b * QK8_0, stride_col_y, acc0, acc1);
    }

    esimd_ncols_epilogue<WG, NCOLS>(acc0, acc1, dst, row0, has_row1, stride_col_dst, lmem, it);
}

template <int WG, int NCOLS>
static void q8_0_esimd_ncols_launch(const void * vx, const float * y, float * dst, const int ncols,
                                    const int nrows, const int stride_col_y, const int stride_col_dst,
                                    dpct::queue_ptr stream) {
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(WG * NCOLS * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd_ncols<WG, NCOLS>(
                    vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, lmem, it);
            });
    });
}

template <int NCOLS>
static void q8_0_esimd_ncols_launch_wg(const void * vx, const float * y, float * dst, const int ncols,
                                       const int nrows, const int stride_col_y, const int stride_col_dst,
                                       dpct::queue_ptr stream) {
    // the same work-group choice as the N=1 launcher, so the per-column arithmetic matches it
    const int nblk_row = ncols / QK8_0;
    if (nblk_row >= 64) {
        q8_0_esimd_ncols_launch<8, NCOLS>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
    } else if (nblk_row >= 32) {
        q8_0_esimd_ncols_launch<4, NCOLS>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
    } else if (nblk_row >= 16) {
        q8_0_esimd_ncols_launch<2, NCOLS>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
    } else {
        q8_0_esimd_ncols_launch<1, NCOLS>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
    }
}

template <int NCOLS>
ESIMD_INLINE void dequantize_mul_mat_vec_q6_K_reorder_esimd_ncols(
        const void * vx, const float * y, float * dst,
        const int ncols, const int nrows, const int stride_col_y, const int stride_col_dst,
        sycl::local_accessor<float, 1> lmem,
        const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    using traits = ggml_sycl_esimd::esimd_reorder_q_traits<GGML_TYPE_Q6_K>;
    constexpr int WG = GGML_SYCL_DMMV_ESIMD_WG_SIZE;

    const int    num_blocks_per_row = ncols / QK_K;
    const size_t nb = (size_t) nrows * num_blocks_per_row;
    const auto   ps = traits::make_ptrs(vx, nb);

    const int  tid      = it.get_local_id(0);
    const int  row0     = it.get_group(0) * 2;
    const bool has_row1 = row0 + 1 < nrows;

    simd<float, 32 * NCOLS> acc0 = 0.0f;
    simd<float, 32 * NCOLS> acc1 = 0.0f;

    for (int ib = tid; ib < num_blocks_per_row; ib += WG) {
        const size_t bi0 = (size_t) (row0 + 0) * num_blocks_per_row + ib;
        const size_t bi1 = (size_t) (row0 + 1) * num_blocks_per_row + ib;

        simd<float, 256> deq_a;
        simd<float, 256> deq_b;
        traits::dequant_pair(ps, bi0, bi1, has_row1, deq_a, deq_b);

#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            simd<float, 256> y_vec = block_load<float, 256>(y + (size_t) c * stride_col_y + (size_t) ib * QK_K);
            simd<float, 32> a = acc0.template select<32, 1>(c * 32);
            simd<float, 32> b = acc1.template select<32, 1>(c * 32);
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                simd<float, 32> y_g = y_vec.template select<32, 1>(32 * g);
                a += y_g * simd<float, 32>(deq_a.template select<32, 1>(32 * g));
                b += y_g * simd<float, 32>(deq_b.template select<32, 1>(32 * g));
            }
            acc0.template select<32, 1>(c * 32) = a;
            acc1.template select<32, 1>(c * 32) = b;
        }
    }

    esimd_ncols_epilogue<WG, NCOLS>(acc0, acc1, dst, row0, has_row1, stride_col_dst, lmem, it);
}

template <int NCOLS>
static void q6_K_esimd_ncols_launch(const void * vx, const float * y, float * dst, const int ncols,
                                    const int nrows, const int stride_col_y, const int stride_col_dst,
                                    dpct::queue_ptr stream) {
    constexpr int WG = GGML_SYCL_DMMV_ESIMD_WG_SIZE;
    const int workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>(WG * NCOLS * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q6_K_reorder_esimd_ncols<NCOLS>(
                    vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, lmem, it);
            });
    });
}

// 2..GGML_SYCL_ESIMD_MAX_NCOLS columns of a reordered q8_0 or q6_K weight. Returns false for
// anything it does not cover, so the caller can assert rather than silently compute garbage.
static bool dequantize_mul_mat_vec_reorder_esimd_ncols(ggml_type type, const void * vx, const float * y,
                                                       float * dst, const int ncols, const int nrows,
                                                       const int ncols_dst, const int stride_col_y,
                                                       const int stride_col_dst, dpct::queue_ptr stream) {
    switch (type) {
        case GGML_TYPE_Q8_0:
            GGML_ASSERT(ncols % QK8_0 == 0);
            switch (ncols_dst) {
                case 2: q8_0_esimd_ncols_launch_wg<2>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 3: q8_0_esimd_ncols_launch_wg<3>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 4: q8_0_esimd_ncols_launch_wg<4>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 5: q8_0_esimd_ncols_launch_wg<5>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 6: q8_0_esimd_ncols_launch_wg<6>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 7: q8_0_esimd_ncols_launch_wg<7>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 8: q8_0_esimd_ncols_launch_wg<8>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                default: return false;
            }
        case GGML_TYPE_Q6_K:
            GGML_ASSERT(ncols % QK_K == 0);
            switch (ncols_dst) {
                case 2: q6_K_esimd_ncols_launch<2>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 3: q6_K_esimd_ncols_launch<3>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 4: q6_K_esimd_ncols_launch<4>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 5: q6_K_esimd_ncols_launch<5>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 6: q6_K_esimd_ncols_launch<6>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 7: q6_K_esimd_ncols_launch<7>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                case 8: q6_K_esimd_ncols_launch<8>(vx, y, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); return true;
                default: return false;
            }
        default:
            return false;
    }
}

// kvalues_iq4nl[n] as a degree-5 minimax polynomial in n - 7.5, evaluated in half and rounded.
// Exact for all 16 n: the half Horner result is off by at most 0.375. A register table lookup
// (simd::iselect) costs about one instruction per element and made the kernel 2x slower.
template <int N>
ESIMD_INLINE sycl::ext::intel::esimd::simd<float, N> iq4_nl_values(sycl::ext::intel::esimd::simd<uint16_t, N> n) {
    using namespace sycl::ext::intel::esimd;
    using h = sycl::half;
    simd<h, N> x = n;
    x -= h(7.5f);
    simd<h, N> p = h(1.447913936e-04f);
    p = p * x + h(1.084357675e-03f);
    p = p * x + h(6.978526453e-02f);
    p = p * x + h(-1.134981564e-01f);
    p = p * x + h(1.161644985e+01f);
    p = p * x + h(-4.417674999e+00f);
    simd<float, N> pf = p;
    return rnde<float, N>(pf);
}

// MoE expert mat-vec over reordered IQ4_NL weights, for decode. Per expert slice the layout is
// [qs: nrows*ncols/2] [d: nrows*ncols/QK4_NL halves], Q4_0 nibble order. One thread computes R
// consecutive rows of one routed expert over the whole row, so there is no cross-thread reduction.
// The R rows share each activation stripe, which is read as f32 (no q8_1 quantize).
template <int NBLK, int R, bool Y_SLM = false>
ESIMD_INLINE void iq4_nl_moe_mac_stripe(
        const uint8_t * qs, const sycl::half * d, const int row0, const int nrows, const int ncols,
        const int b, const float * y, sycl::ext::intel::esimd::simd<float, 16 * R> & acc) {
    using namespace sycl::ext::intel::esimd;

    const int blocks_per_row = ncols / QK4_NL;

    simd<uint8_t, 16 * NBLK * R> q;
    simd<sycl::half, NBLK * R>   dh;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        // a tail row past nrows recomputes the last row and is not stored
        const int    row = row0 + r < nrows ? row0 + r : nrows - 1;
        const size_t ib  = (size_t) row * blocks_per_row + b;
        q.template select<16 * NBLK, 1>(r * 16 * NBLK) = block_load<uint8_t, 16 * NBLK>(qs + ib * (QK4_NL / 2));
        dh.template select<NBLK, 1>(r * NBLK) = block_load<sycl::half, NBLK>(d + ib, element_aligned_tag{});
    }
    simd<float, 32 * NBLK> y_vec;
    if constexpr (Y_SLM) {
        y_vec = slm_block_load<float, 32 * NBLK>(b * QK4_NL * sizeof(float));
    } else {
        y_vec = block_load<float, 32 * NBLK>(y + (size_t) b * QK4_NL);
    }
    simd<float, NBLK * R>  df    = convert<float>(dh);

#pragma unroll
    for (int r = 0; r < R; ++r) {
        simd<uint16_t, 16 * NBLK> qr = q.template select<16 * NBLK, 1>(r * 16 * NBLK);
        simd<float, 16 * NBLK>    lo = iq4_nl_values<16 * NBLK>(qr & 0xF);
        simd<float, 16 * NBLK>    hi = iq4_nl_values<16 * NBLK>(qr >> 4);
#pragma unroll
        for (int s = 0; s < NBLK; ++s) {
            simd<float, 16> t = y_vec.template select<16, 1>(s * 32) * lo.template select<16, 1>(s * 16) +
                                y_vec.template select<16, 1>(s * 32 + 16) * hi.template select<16, 1>(s * 16);
            const float     sc = df[r * NBLK + s];
            acc.template select<16, 1>(r * 16) += t * sc;
        }
    }
}

template <int R>
ESIMD_INLINE void mul_mat_vec_iq4_nl_moe_reorder_esimd(
        const void * vx_base, const void * y_base, float * dst_base, const int32_t * ids_dev,
        const int ncols, const int nrows, const size_t expert_weight_stride, const size_t dst_row_stride,
        const size_t src1_row_stride, const size_t ids_token_stride, const size_t dst_token_stride,
        const size_t src1_token_stride, const sycl::nd_item<3> & it) {
    using namespace sycl::ext::intel::esimd;

    const int row0 = (int) (it.get_group(2) * it.get_local_range(2) + it.get_local_id(2)) * R;
    if (row0 >= nrows) {
        return;
    }

    const int token_idx  = it.get_group(0);
    const int expert_idx = it.get_group(1);
    const int i02 = *(const int32_t *) ((const char *) ids_dev + (size_t) token_idx * ids_token_stride + expert_idx * sizeof(int32_t));

    const uint8_t *    qs  = (const uint8_t *) vx_base + (size_t) i02 * expert_weight_stride;
    const sycl::half * d   = (const sycl::half *) (qs + (size_t) nrows * (ncols / 2));
    const float *      y   = (const float *) ((const char *) y_base + (size_t) token_idx * src1_token_stride + (size_t) expert_idx * src1_row_stride);
    float *            dst = (float *) ((char *) dst_base + (size_t) token_idx * dst_token_stride + (size_t) expert_idx * dst_row_stride);

    simd<float, 16 * R> acc = 0.0f;

    const int blocks_per_row = ncols / QK4_NL;
    int b = 0;
    for (; b + 8 <= blocks_per_row; b += 8) {
        iq4_nl_moe_mac_stripe<8, R>(qs, d, row0, nrows, ncols, b, y, acc);
    }
    if (b + 4 <= blocks_per_row) {
        iq4_nl_moe_mac_stripe<4, R>(qs, d, row0, nrows, ncols, b, y, acc);
        b += 4;
    }
    if (b + 2 <= blocks_per_row) {
        iq4_nl_moe_mac_stripe<2, R>(qs, d, row0, nrows, ncols, b, y, acc);
        b += 2;
    }
    if (b < blocks_per_row) {
        iq4_nl_moe_mac_stripe<1, R>(qs, d, row0, nrows, ncols, b, y, acc);
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r < nrows) {
            simd<float, 16> acc_r = acc.template select<16, 1>(r * 16);
            dst[row0 + r] = reduce<float>(acc_r, std::plus<>{});
        }
    }
}

template <int R, int WG>
static void iq4_nl_moe_esimd_launch(const void * vx_base, const float * y, const int32_t * ids_dev, float * dst_base,
                                    const int ncols, const int nrows, const int n_experts_used, const int n_tokens,
                                    const size_t expert_weight_stride, const size_t dst_row_stride,
                                    const size_t src1_row_stride, const size_t ids_token_stride,
                                    const size_t dst_token_stride, const size_t src1_token_stride,
                                    dpct::queue_ptr stream) {
    const int            block_num = ceil_div(nrows, R * WG);
    const sycl::range<3> block_nums((unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num);
    const sycl::range<3> block_dims(1, 1, WG);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> it) [[intel::sycl_explicit_simd]] {
            mul_mat_vec_iq4_nl_moe_reorder_esimd<R>(vx_base, y, dst_base, ids_dev, ncols, nrows,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, it);
        });
}

// GGML_SYCL_MOE_ESIMD=5: rows of exactly NB blocks take them in one stripe, so a thread issues all of its weight
// loads at once instead of in 3 dependent rounds (8 + 8 + 4 blocks at 640 columns).
template <int R, int NB>
ESIMD_INLINE void mul_mat_vec_iq4_nl_moe_reorder_row_esimd(
        const void * vx_base, const void * y_base, float * dst_base, const int32_t * ids_dev,
        const int ncols, const int nrows, const size_t expert_weight_stride, const size_t dst_row_stride,
        const size_t src1_row_stride, const size_t ids_token_stride, const size_t dst_token_stride,
        const size_t src1_token_stride, const sycl::nd_item<3> & it) {
    using namespace sycl::ext::intel::esimd;

    const int row0 = (int) (it.get_group(2) * it.get_local_range(2) + it.get_local_id(2)) * R;
    if (row0 >= nrows) {
        return;
    }

    const int token_idx  = it.get_group(0);
    const int expert_idx = it.get_group(1);
    const int i02 = *(const int32_t *) ((const char *) ids_dev + (size_t) token_idx * ids_token_stride + expert_idx * sizeof(int32_t));

    const uint8_t *    qs  = (const uint8_t *) vx_base + (size_t) i02 * expert_weight_stride;
    const sycl::half * d   = (const sycl::half *) (qs + (size_t) nrows * (ncols / 2));
    const float *      y   = (const float *) ((const char *) y_base + (size_t) token_idx * src1_token_stride + (size_t) expert_idx * src1_row_stride);
    float *            dst = (float *) ((char *) dst_base + (size_t) token_idx * dst_token_stride + (size_t) expert_idx * dst_row_stride);

    simd<float, 16 * R> acc = 0.0f;
    iq4_nl_moe_mac_stripe<NB, R>(qs, d, row0, nrows, ncols, 0, y, acc);

#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r < nrows) {
            simd<float, 16> acc_r = acc.template select<16, 1>(r * 16);
            dst[row0 + r] = reduce<float>(acc_r, std::plus<>{});
        }
    }
}

template <int R, int WG, int NB>
static void iq4_nl_moe_row_esimd_launch(const void * vx_base, const float * y, const int32_t * ids_dev, float * dst_base,
                                        const int ncols, const int nrows, const int n_experts_used, const int n_tokens,
                                        const size_t expert_weight_stride, const size_t dst_row_stride,
                                        const size_t src1_row_stride, const size_t ids_token_stride,
                                        const size_t dst_token_stride, const size_t src1_token_stride,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols == NB * QK4_NL);
    const int            block_num = ceil_div(nrows, R * WG);
    const sycl::range<3> block_nums((unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num);
    const sycl::range<3> block_dims(1, 1, WG);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> it) [[intel::sycl_explicit_simd]] {
            mul_mat_vec_iq4_nl_moe_reorder_row_esimd<R, NB>(vx_base, y, dst_base, ids_dev, ncols, nrows,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, it);
        });
}

// One reordered IQ3_S block of one row against a 256-float activation stripe. Per expert slice the
// layout is [qs: nb*64] [qh: nb*8] [signs: nb*32] [{d, scales}: nb*6]. Grid entry k holds the
// magnitudes of elements 4k..4k+3, and bit e%8 of signs[e/8] is the sign of element e.
ESIMD_INLINE void iq3_s_moe_mac_block(const uint8_t * vx, const size_t nb, const size_t ib,
                                      sycl::ext::intel::esimd::simd<float, QK_K> & y_vec,
                                      sycl::ext::intel::esimd::simd<float, 16> & acc) {
    using namespace sycl::ext::intel::esimd;

    simd<uint8_t, 64>  qs   = block_load<uint8_t, 64>(vx + ib * 64);
    simd<uint8_t, 8>   qh   = block_load<uint8_t, 8>(vx + nb * 64 + ib * 8);
    simd<uint8_t, 32>  sg   = block_load<uint8_t, 32>(vx + nb * 72 + ib * 32);
    simd<uint16_t, 3>  meta = block_load<uint16_t, 3>((const uint16_t *) (vx + nb * 104 + ib * 6), element_aligned_tag{});

    simd<uint16_t, 1>   d_bits = meta.template select<1, 1>(0);
    simd<sycl::half, 1> d_h    = d_bits.template bit_cast_view<sycl::half>();
    simd<float, 1>      d_f    = d_h;
    const float         d      = d_f[0];
    simd<uint16_t, 2> sc_raw = meta.template select<2, 1>(1);
    simd<uint8_t, 4>  sc_b   = sc_raw.template bit_cast_view<uint8_t>();
    simd<float, 8>    sc;
    sc.template select<4, 2>(0) = sc_b & 0xf;
    sc.template select<4, 2>(1) = sc_b >> 4;
    sc = (sc * 2.0f + 1.0f) * d;

    // grid index k takes its 9th bit from bit k%8 of qh[k/8]
    const simd<uint16_t, 64> lane64(0, 1);
    simd<uint16_t, 64> qh64 = qh.template replicate_vs_w_hs<8, 1, 8, 0>(0);
    simd<uint32_t, 64> idx  = qs | (((qh64 >> (lane64 & 7)) & 1) << 8);
    simd<uint32_t, 64> grid = gather<uint32_t, 64>(iq3s_grid, idx * (uint32_t) sizeof(uint32_t));

    simd<uint8_t, QK_K>  mag  = grid.template bit_cast_view<uint8_t>();
    simd<float, QK_K>    w    = mag;
    const simd<uint32_t, QK_K> lane256(0, 1);
    simd<uint32_t, QK_K> sgn  = sg.template replicate_vs_w_hs<32, 1, 8, 0>(0);
    sgn = (sgn << (31 - (lane256 & 7))) & 0x80000000u;
    w.template bit_cast_view<uint32_t>() ^= sgn;

#pragma unroll
    for (int s = 0; s < 8; ++s) {
        simd<float, 16> t = y_vec.template select<16, 1>(s * 32) * w.template select<16, 1>(s * 32) +
                            y_vec.template select<16, 1>(s * 32 + 16) * w.template select<16, 1>(s * 32 + 16);
        const float sc_s = sc[s];
        acc += t * sc_s;
    }
}

// MoE gate and up mat-vec over reordered IQ3_S weights with the SWIGLU folded in, for decode. Same
// thread layout as the IQ4_NL kernel above: R consecutive rows of one routed expert per thread, both
// weights, one activation stripe per QK_K block shared by the 2*R rows.
template <int R>
ESIMD_INLINE void mul_mat_vec_iq3_s_moe_glu_esimd(
        const void * vx_gate_base, const void * vx_up_base, const void * y_base, float * dst_base,
        const int32_t * ids_dev, const int ncols, const int nrows, const size_t expert_weight_stride,
        const size_t dst_row_stride, const size_t src1_row_stride, const size_t ids_token_stride,
        const size_t dst_token_stride, const size_t src1_token_stride, const sycl::nd_item<3> & it) {
    using namespace sycl::ext::intel::esimd;

    const int row0 = (int) (it.get_group(2) * it.get_local_range(2) + it.get_local_id(2)) * R;
    if (row0 >= nrows) {
        return;
    }

    const int token_idx  = it.get_group(0);
    const int expert_idx = it.get_group(1);
    const int i02 = *(const int32_t *) ((const char *) ids_dev + (size_t) token_idx * ids_token_stride + expert_idx * sizeof(int32_t));

    const uint8_t * vg  = (const uint8_t *) vx_gate_base + (size_t) i02 * expert_weight_stride;
    const uint8_t * vu  = (const uint8_t *) vx_up_base   + (size_t) i02 * expert_weight_stride;
    const float *   y   = (const float *) ((const char *) y_base + (size_t) token_idx * src1_token_stride + (size_t) expert_idx * src1_row_stride);
    float *         dst = (float *) ((char *) dst_base + (size_t) token_idx * dst_token_stride + (size_t) expert_idx * dst_row_stride);

    const int    blocks_per_row = ncols / QK_K;
    const size_t nb             = (size_t) nrows * blocks_per_row;

    simd<float, 16 * R> acc_g = 0.0f;
    simd<float, 16 * R> acc_u = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        simd<float, QK_K> y_vec = block_load<float, QK_K>(y + (size_t) b * QK_K);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            // a tail row past nrows recomputes the last row and is not stored
            const int    row = row0 + r < nrows ? row0 + r : nrows - 1;
            const size_t ib  = (size_t) row * blocks_per_row + b;
            simd<float, 16> ag = acc_g.template select<16, 1>(r * 16);
            simd<float, 16> au = acc_u.template select<16, 1>(r * 16);
            iq3_s_moe_mac_block(vg, nb, ib, y_vec, ag);
            iq3_s_moe_mac_block(vu, nb, ib, y_vec, au);
            acc_g.template select<16, 1>(r * 16) = ag;
            acc_u.template select<16, 1>(r * 16) = au;
        }
    }

    simd<float, R> gate;
    simd<float, R> up;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        simd<float, 16> ag = acc_g.template select<16, 1>(r * 16);
        simd<float, 16> au = acc_u.template select<16, 1>(r * 16);
        gate[r] = reduce<float>(ag, std::plus<>{});
        up[r]   = reduce<float>(au, std::plus<>{});
    }
    simd<float, R> out = up * gate / (1.0f + exp(-gate));
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r < nrows) {
            dst[row0 + r] = out[r];
        }
    }
}

template <int R, int WG>
static void iq3_s_moe_glu_esimd_launch(const void * vx_gate_base, const void * vx_up_base, const float * y,
                                       const int32_t * ids_dev, float * dst_base, const int ncols, const int nrows,
                                       const int n_experts_used, const int n_tokens, const size_t expert_weight_stride,
                                       const size_t dst_row_stride, const size_t src1_row_stride,
                                       const size_t ids_token_stride, const size_t dst_token_stride,
                                       const size_t src1_token_stride, dpct::queue_ptr stream) {
    const int            block_num = ceil_div(nrows, R * WG);
    const sycl::range<3> block_nums((unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num);
    const sycl::range<3> block_dims(1, 1, WG);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> it) [[intel::sycl_explicit_simd]] {
            mul_mat_vec_iq3_s_moe_glu_esimd<R>(vx_gate_base, vx_up_base, y, dst_base, ids_dev, ncols, nrows,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, it);
        });
}

// Integer variant of the kernel above (GGML_SYCL_MOE_ESIMD=3). The work-group quantizes its activation
// row into SLM as int8 with one scale per 32 values (the IQ3_S sub-block) and stages iq3s_grid there.
// A block is then 64 packed 4-byte lanes: an SLM grid gather, the signs folded into the grid bytes and
// one dp4a per lane, instead of 256 f32 lanes.
#define IQ3_S_MOE_DP4A_MAX_COLS 4096
#define IQ3_S_MOE_DP4A_YD_SLM   (512 * sizeof(uint32_t))
#define IQ3_S_MOE_DP4A_YQ_SLM   (IQ3_S_MOE_DP4A_YD_SLM + IQ3_S_MOE_DP4A_MAX_COLS / 32 * sizeof(float))

// NB consecutive blocks of one row, so the weight loads are NB blocks wide
template <int NB>
ESIMD_INLINE void iq3_s_moe_dp4a_blocks(const uint8_t * vx, const size_t nb, const size_t ib,
                                        const sycl::ext::intel::esimd::simd<int32_t, 64 * NB> & yq,
                                        const sycl::ext::intel::esimd::simd<float, 8 * NB> & yd,
                                        sycl::ext::intel::esimd::simd<float, 8> & acc) {
    using namespace sycl::ext::intel::esimd;

    simd<uint8_t, 64 * NB> qs   = block_load<uint8_t, 64 * NB>(vx + ib * 64);
    simd<uint8_t, 8 * NB>  qh   = block_load<uint8_t, 8 * NB>(vx + nb * 64 + ib * 8);
    simd<uint8_t, 32 * NB> sg   = block_load<uint8_t, 32 * NB>(vx + nb * 72 + ib * 32);
    simd<uint16_t, 3 * NB> meta = block_load<uint16_t, 3 * NB>((const uint16_t *) (vx + nb * 104 + ib * 6), element_aligned_tag{});

    simd<float, 8 * NB> sc;
    simd<float, 8 * NB> d;
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        simd<uint16_t, 1>   d_bits = meta.template select<1, 1>(3 * k);
        simd<sycl::half, 1> d_h    = d_bits.template bit_cast_view<sycl::half>();
        simd<float, 1>      d_f    = d_h;
        const float         d_k    = d_f[0];
        simd<uint16_t, 2>   sc_raw = meta.template select<2, 1>(3 * k + 1);
        simd<uint8_t, 4>    sc_b   = sc_raw.template bit_cast_view<uint8_t>();
        sc.template select<4, 2>(8 * k)     = sc_b & 0xf;
        sc.template select<4, 2>(8 * k + 1) = sc_b >> 4;
        d.template select<8, 1>(8 * k)      = d_k;
    }
    sc = (sc * 2.0f + 1.0f) * d * yd;

    // lane k holds elements 4k..4k+3: grid entry k, and sign bits 4(k%2)..4(k%2)+3 of signs[k/2]
    const simd<uint16_t, 64 * NB> lane(0, 1);
    simd<uint16_t, 64 * NB> qh64 = qh.template replicate_vs_w_hs<8 * NB, 1, 8, 0>(0);
    simd<uint32_t, 64 * NB> off  = (qs | (((qh64 >> (lane & 7)) & 1) << 8)) * (uint32_t) sizeof(uint32_t);
    simd<uint32_t, 64 * NB> grid;
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        grid.template select<64, 1>(64 * k) = slm_gather<uint32_t, 64>(off.template select<64, 1>(64 * k));
    }

    // spread each sign bit to the low bit of its byte and negate those bytes: grid bytes are 1..15,
    // so (g ^ 0xff) + 1 never carries into the next byte
    simd<uint32_t, 64 * NB> sgn = sg.template replicate_vs_w_hs<32 * NB, 1, 2, 0>(0);
    sgn = (sgn >> ((lane & 1) * 4)) & 0xf;
    simd<uint32_t, 64 * NB> m01 = (sgn * 0x00204081u) & 0x01010101u;
    simd<uint32_t, 64 * NB> w   = (grid ^ (m01 * 0xffu)) + m01;

    // 8 lanes per 32-value sub-block
    simd<int32_t, 64 * NB> p   = dp4a<int32_t, int32_t, int32_t, int32_t, 64 * NB>(
        simd<int32_t, 64 * NB>(0), yq, w.template bit_cast_view<int32_t>());
    simd<int32_t, 32 * NB> p32 = p.template select<32 * NB, 2>(0) + p.template select<32 * NB, 2>(1);
    simd<int32_t, 16 * NB> p16 = p32.template select<16 * NB, 2>(0) + p32.template select<16 * NB, 2>(1);
    simd<int32_t, 8 * NB>  p8  = p16.template select<8 * NB, 2>(0) + p16.template select<8 * NB, 2>(1);
    simd<float, 8 * NB>    f   = convert<float>(p8) * sc;
#pragma unroll
    for (int k = 0; k < NB; ++k) {
        acc += f.template select<8, 1>(8 * k);
    }
}

template <int NB, int R>
ESIMD_INLINE void iq3_s_moe_glu_dp4a_step(const uint8_t * vg, const uint8_t * vu, const int row0, const int nrows,
                                          const int blocks_per_row, const int b,
                                          sycl::ext::intel::esimd::simd<float, 8 * R> & acc_g,
                                          sycl::ext::intel::esimd::simd<float, 8 * R> & acc_u) {
    using namespace sycl::ext::intel::esimd;

    const size_t nb = (size_t) nrows * blocks_per_row;
    simd<int32_t, 64 * NB> yq = slm_block_load<int32_t, 64 * NB>(IQ3_S_MOE_DP4A_YQ_SLM + b * QK_K);
    simd<float, 8 * NB>    yd = slm_block_load<float, 8 * NB>(IQ3_S_MOE_DP4A_YD_SLM + b * 8 * sizeof(float));
#pragma unroll
    for (int r = 0; r < R; ++r) {
        // a tail row past nrows recomputes the last row and is not stored
        const int    row = row0 + r < nrows ? row0 + r : nrows - 1;
        const size_t ib  = (size_t) row * blocks_per_row + b;
        simd<float, 8> ag = acc_g.template select<8, 1>(r * 8);
        simd<float, 8> au = acc_u.template select<8, 1>(r * 8);
        iq3_s_moe_dp4a_blocks<NB>(vg, nb, ib, yq, yd, ag);
        iq3_s_moe_dp4a_blocks<NB>(vu, nb, ib, yq, yd, au);
        acc_g.template select<8, 1>(r * 8) = ag;
        acc_u.template select<8, 1>(r * 8) = au;
    }
}

template <int R, int WG, int NB>
ESIMD_INLINE void mul_mat_vec_iq3_s_moe_glu_dp4a_esimd(
        const void * vx_gate_base, const void * vx_up_base, const void * y_base, float * dst_base,
        const int32_t * ids_dev, const int ncols, const int nrows, const size_t expert_weight_stride,
        const size_t dst_row_stride, const size_t src1_row_stride, const size_t ids_token_stride,
        const size_t dst_token_stride, const size_t src1_token_stride, const sycl::nd_item<3> & it) {
    using namespace sycl::ext::intel::esimd;

    // SLM: [iq3s_grid] [f32 activation scale per 32 values] [int8 activation]
    slm_init<IQ3_S_MOE_DP4A_YQ_SLM + IQ3_S_MOE_DP4A_MAX_COLS>();

    const int tid        = it.get_local_id(2);
    const int token_idx  = it.get_group(0);
    const int expert_idx = it.get_group(1);
    const float * y = (const float *) ((const char *) y_base + (size_t) token_idx * src1_token_stride + (size_t) expert_idx * src1_row_stride);

    constexpr int grid_per_thread = 512 / WG;
    slm_block_store<uint32_t, grid_per_thread>(tid * grid_per_thread * sizeof(uint32_t),
        block_load<uint32_t, grid_per_thread>(iq3s_grid + tid * grid_per_thread));
    for (int j = tid; j < ncols / 32; j += WG) {
        simd<float, 32>  v    = block_load<float, 32>(y + j * 32);
        const float      amax = hmax<float>(abs(v));
        const float      id   = amax > 0.0f ? 127.0f / amax : 0.0f;
        simd<int8_t, 32> q    = convert<int8_t>(rnde<float, 32>(v * id));
        slm_block_store<int8_t, 32>(IQ3_S_MOE_DP4A_YQ_SLM + j * 32, q);
        slm_scalar_store<float>(IQ3_S_MOE_DP4A_YD_SLM + j * sizeof(float), amax / 127.0f);
    }
    barrier();

    const int row0 = (int) (it.get_group(2) * WG + tid) * R;
    if (row0 >= nrows) {
        return;
    }

    const int i02 = *(const int32_t *) ((const char *) ids_dev + (size_t) token_idx * ids_token_stride + expert_idx * sizeof(int32_t));
    const uint8_t * vg  = (const uint8_t *) vx_gate_base + (size_t) i02 * expert_weight_stride;
    const uint8_t * vu  = (const uint8_t *) vx_up_base   + (size_t) i02 * expert_weight_stride;
    float *         dst = (float *) ((char *) dst_base + (size_t) token_idx * dst_token_stride + (size_t) expert_idx * dst_row_stride);

    const int blocks_per_row = ncols / QK_K;

    simd<float, 8 * R> acc_g = 0.0f;
    simd<float, 8 * R> acc_u = 0.0f;
    int b = 0;
    for (; b + NB <= blocks_per_row; b += NB) {
        iq3_s_moe_glu_dp4a_step<NB, R>(vg, vu, row0, nrows, blocks_per_row, b, acc_g, acc_u);
    }
    for (; b < blocks_per_row; ++b) {
        iq3_s_moe_glu_dp4a_step<1, R>(vg, vu, row0, nrows, blocks_per_row, b, acc_g, acc_u);
    }

    simd<float, R> gate;
    simd<float, R> up;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        simd<float, 8> ag = acc_g.template select<8, 1>(r * 8);
        simd<float, 8> au = acc_u.template select<8, 1>(r * 8);
        gate[r] = reduce<float>(ag, std::plus<>{});
        up[r]   = reduce<float>(au, std::plus<>{});
    }
    simd<float, R> out = up * gate / (1.0f + exp(-gate));
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r < nrows) {
            dst[row0 + r] = out[r];
        }
    }
}

template <int R, int WG, int NB>
static void iq3_s_moe_glu_dp4a_esimd_launch(const void * vx_gate_base, const void * vx_up_base, const float * y,
                                            const int32_t * ids_dev, float * dst_base, const int ncols, const int nrows,
                                            const int n_experts_used, const int n_tokens, const size_t expert_weight_stride,
                                            const size_t dst_row_stride, const size_t src1_row_stride,
                                            const size_t ids_token_stride, const size_t dst_token_stride,
                                            const size_t src1_token_stride, dpct::queue_ptr stream) {
    static_assert(512 % (16 * WG) == 0, "each thread stages a multiple of 16 grid entries");
    GGML_ASSERT(ncols % QK_K == 0 && ncols <= IQ3_S_MOE_DP4A_MAX_COLS);
    const int            block_num = ceil_div(nrows, R * WG);
    const sycl::range<3> block_nums((unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num);
    const sycl::range<3> block_dims(1, 1, WG);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> it) [[intel::sycl_explicit_simd]] {
            mul_mat_vec_iq3_s_moe_glu_dp4a_esimd<R, WG, NB>(vx_gate_base, vx_up_base, y, dst_base, ids_dev, ncols, nrows,
                expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, it);
        });
}

// GGML_SYCL_MOE_ESIMD=4: the level-3 kernel with each row's blocks split across KS threads. A thread walks NB
// blocks, then KS - 1 more steps away, so a row of KS * NB blocks takes one step per thread instead of a 4-block
// loop and a 1-block tail; the partial sums meet in SLM. Each thread's chain of dependent loads is KS times
// shorter, and RG row groups per work-group spread the per-work-group prologue (activation quantization and the
// grid staging) over more rows. The expert id is read first, so its latency overlaps the prologue.
template <int R, int RG, int KS, int NB>
ESIMD_INLINE void mul_mat_vec_iq3_s_moe_glu_dp4a_ks_esimd(
        const void * vx_gate_base, const void * vx_up_base, const void * y_base, float * dst_base,
        const int32_t * ids_dev, const int ncols, const int nrows, const size_t expert_weight_stride,
        const size_t dst_row_stride, const size_t src1_row_stride, const size_t ids_token_stride,
        const size_t dst_token_stride, const size_t src1_token_stride, const sycl::nd_item<3> & it) {
    using namespace sycl::ext::intel::esimd;

    constexpr int WG  = RG * KS;
    // SLM: [iq3s_grid] [f32 activation scale per 32 values] [int8 activation] [partial sums]
    constexpr int RED = IQ3_S_MOE_DP4A_YQ_SLM + IQ3_S_MOE_DP4A_MAX_COLS;
    slm_init<RED + WG * 2 * R * sizeof(float)>();

    const int tid        = it.get_local_id(2);
    const int token_idx  = it.get_group(0);
    const int expert_idx = it.get_group(1);
    const float * y = (const float *) ((const char *) y_base + (size_t) token_idx * src1_token_stride + (size_t) expert_idx * src1_row_stride);

    const int i02 = *(const int32_t *) ((const char *) ids_dev + (size_t) token_idx * ids_token_stride + expert_idx * sizeof(int32_t));

    for (int i = tid; i < 512 / 16; i += WG) {
        slm_block_store<uint32_t, 16>(i * 16 * sizeof(uint32_t), block_load<uint32_t, 16>(iq3s_grid + i * 16));
    }
    for (int j = tid; j < ncols / 32; j += WG) {
        simd<float, 32>  v    = block_load<float, 32>(y + j * 32);
        const float      amax = hmax<float>(abs(v));
        const float      id   = amax > 0.0f ? 127.0f / amax : 0.0f;
        simd<int8_t, 32> q    = convert<int8_t>(rnde<float, 32>(v * id));
        slm_block_store<int8_t, 32>(IQ3_S_MOE_DP4A_YQ_SLM + j * 32, q);
        slm_scalar_store<float>(IQ3_S_MOE_DP4A_YD_SLM + j * sizeof(float), amax / 127.0f);
    }
    barrier();

    const int rg   = tid / KS;
    const int ks   = tid % KS;
    // a row group past nrows still takes part in the reduction barrier
    const int row0 = (int) (it.get_group(2) * RG + rg) * R;

    const uint8_t * vg  = (const uint8_t *) vx_gate_base + (size_t) i02 * expert_weight_stride;
    const uint8_t * vu  = (const uint8_t *) vx_up_base   + (size_t) i02 * expert_weight_stride;
    float *         dst = (float *) ((char *) dst_base + (size_t) token_idx * dst_token_stride + (size_t) expert_idx * dst_row_stride);

    const int blocks_per_row = ncols / QK_K;

    simd<float, 8 * R> acc_g = 0.0f;
    simd<float, 8 * R> acc_u = 0.0f;
    if (row0 < nrows) {
        for (int b = ks * NB; b < blocks_per_row; b += KS * NB) {
            iq3_s_moe_glu_dp4a_step<NB, R>(vg, vu, row0, nrows, blocks_per_row, b, acc_g, acc_u);
        }
    }

    simd<float, 2 * R> part;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        simd<float, 8> ag = acc_g.template select<8, 1>(r * 8);
        simd<float, 8> au = acc_u.template select<8, 1>(r * 8);
        part[r]     = reduce<float>(ag, std::plus<>{});
        part[R + r] = reduce<float>(au, std::plus<>{});
    }
    slm_block_store<float, 2 * R>(RED + tid * 2 * R * sizeof(float), part);
    barrier();

    if (ks != 0 || row0 >= nrows) {
        return;
    }
    simd<float, 2 * R> sum = 0.0f;
#pragma unroll
    for (int k = 0; k < KS; ++k) {
        sum += slm_block_load<float, 2 * R>(RED + (tid + k) * 2 * R * sizeof(float));
    }
    simd<float, R> gate = sum.template select<R, 1>(0);
    simd<float, R> up   = sum.template select<R, 1>(R);
    simd<float, R> out  = up * gate / (1.0f + exp(-gate));
#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (row0 + r < nrows) {
            dst[row0 + r] = out[r];
        }
    }
}

template <int R, int RG, int KS, int NB>
static void iq3_s_moe_glu_dp4a_ks_esimd_launch(const void * vx_gate_base, const void * vx_up_base, const float * y,
                                               const int32_t * ids_dev, float * dst_base, const int ncols, const int nrows,
                                               const int n_experts_used, const int n_tokens, const size_t expert_weight_stride,
                                               const size_t dst_row_stride, const size_t src1_row_stride,
                                               const size_t ids_token_stride, const size_t dst_token_stride,
                                               const size_t src1_token_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % (QK_K * KS * NB) == 0 && ncols <= IQ3_S_MOE_DP4A_MAX_COLS);
    const int            block_num = ceil_div(nrows, R * RG);
    const sycl::range<3> block_nums((unsigned) n_tokens, (unsigned) n_experts_used, (unsigned) block_num);
    const sycl::range<3> block_dims(1, 1, RG * KS);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> it) [[intel::sycl_explicit_simd]] {
            mul_mat_vec_iq3_s_moe_glu_dp4a_ks_esimd<R, RG, KS, NB>(vx_gate_base, vx_up_base, y, dst_base, ids_dev, ncols,
                nrows, expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride,
                dst_token_stride, src1_token_stride, it);
        });
}

#endif // GGML_SYCL_DMMV_HAS_ESIMD

static void dequantize_mul_mat_vec_q4_K_sycl_reorder(const void *vx, const float *y,
                                                     float *dst, const int ncols,
                                                     const int nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q4_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q5_K_sycl_reorder(const void *vx, const float *y,
                                                     float *dst, const int ncols,
                                                     const int nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q5_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q6_K_sycl_reorder(const void *vx, const float *y,
                                                     float *dst, const int ncols,
                                                     const int nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q6_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // several columns: only the reordered ESIMD q8_0 / q6_K kernels take them, and
    // ggml_sycl_mul_mat() only sends them here when they will
    if (src1_ncols > 1) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
        const auto * extra = (const ggml_tensor_extra_gpu *) src0->extra;
        GGML_ASSERT(extra && extra->optimized_feature.is_reordered());
        const bool ok = dequantize_mul_mat_vec_reorder_esimd_ncols(
            src0->type, src0_dd_i, src1_ddf_i, dst_dd_i, (int) ne00, (int) row_diff, (int) src1_ncols,
            /*stride_col_y=*/(int) src1->ne[0], /*stride_col_dst=*/(int) dst->ne[0], stream);
        GGML_ASSERT(ok);
        return;
#else
        GGML_ABORT("multi-column dequantize_mul_mat_vec needs ESIMD");
#endif
    }

    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = nullptr; // dfloat == half

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    const bool q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd && g_ggml_sycl_esimd_q8_0 &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.is_reordered();
#else
    const bool q8_0_esimd = false;
#endif

    bool src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    if (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#else
    const dfloat * src1_dfloat = (const dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_SYCL_F16

    switch (src0->type) {
        case GGML_TYPE_Q1_0:
            if ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.is_reordered()) {
                dequantize_mul_mat_vec_q1_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } else {
                dequantize_mul_mat_vec_q1_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q4_0:
            if ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.is_reordered()) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } else {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q8_0:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.is_reordered()) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd && g_ggml_sycl_esimd_q8_0) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                } else
#endif
                {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q2_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.is_reordered()) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q3_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.is_reordered()) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q4_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.is_reordered()) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q5_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.is_reordered()) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_Q6_K:
            if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.is_reordered()) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                if (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                else
#endif
                {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } else {
                dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            break;
        case GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
#ifdef GGML_SYCL_DMMV_HAS_BF16
        case GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
#endif
        default:
            printf("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
    GGML_UNUSED(ctx);
}

bool ggml_sycl_mul_mat_vec_q_id_reorder_esimd(
    enum ggml_type src0_type, const void * vx_base, const float * y, const int32_t * ids_dev,
    float * dst_base, int ncols, int nrows, int n_experts_used, int n_tokens,
    size_t expert_weight_stride, size_t dst_row_stride, size_t src1_row_stride,
    size_t ids_token_stride, size_t dst_token_stride, size_t src1_token_stride,
    dpct::queue_ptr stream) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    if (src0_type != GGML_TYPE_IQ4_NL || ncols % QK4_NL != 0) {
        return false;
    }
    if (g_ggml_sycl_moe_esimd >= 5 && ncols == 20 * QK4_NL) {
        // GGML_SYCL_IQ4_NL_MOE_R / _WG pick rows per thread and threads per work-group for tuning
        static const int r  = ggml_sycl_get_env("GGML_SYCL_IQ4_NL_MOE_R", 2);
        static const int wg = ggml_sycl_get_env("GGML_SYCL_IQ4_NL_MOE_WG", 8);
#define IQ4_NL_ROW_LAUNCH(R_, WG_) \
        iq4_nl_moe_row_esimd_launch<R_, WG_, 20>(vx_base, y, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens, \
            expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride, dst_token_stride, \
            src1_token_stride, stream); \
        return true
        if (r == 1) {
            IQ4_NL_ROW_LAUNCH(1, 8);
        }
        if (r == 4) {
            if (wg == 16) { IQ4_NL_ROW_LAUNCH(4, 16); }
            IQ4_NL_ROW_LAUNCH(4, 8);
        }
        if (wg == 16) { IQ4_NL_ROW_LAUNCH(2, 16); }
        IQ4_NL_ROW_LAUNCH(2, 8);
#undef IQ4_NL_ROW_LAUNCH
    }
    // 4 rows per thread and 8 threads per work-group measured best on Arc Pro B60 (2, 8 rows: 10-18% slower)
    iq4_nl_moe_esimd_launch<4, 8>(vx_base, y, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
        expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride, dst_token_stride, src1_token_stride, stream);
    return true;
#else
    GGML_UNUSED_VARS(src0_type, vx_base, y, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
        expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride, dst_token_stride, src1_token_stride, stream);
    return false;
#endif
}

bool ggml_sycl_mul_mat_vec_q_id_reorder_glu_esimd(
    enum ggml_type src0_type, const void * vx_gate_base, const void * vx_up_base, const float * y,
    const int32_t * ids_dev, float * dst_base, int ncols, int nrows, int n_experts_used, int n_tokens,
    size_t expert_weight_stride, size_t dst_row_stride, size_t src1_row_stride,
    size_t ids_token_stride, size_t dst_token_stride, size_t src1_token_stride,
    ggml_glu_op glu_op, dpct::queue_ptr stream) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    if (src0_type != GGML_TYPE_IQ3_S || glu_op != GGML_GLU_OP_SWIGLU || ncols % QK_K != 0) {
        return false;
    }
    if (g_ggml_sycl_moe_esimd >= 4 && ncols <= IQ3_S_MOE_DP4A_MAX_COLS) {
        // GGML_SYCL_IQ3_S_DP4A_KS picks the split for tuning: 2 threads x 5 blocks or 5 threads x 2 blocks per row,
        // with GGML_SYCL_IQ3_S_DP4A_RG row groups per work-group
        static const int ks = ggml_sycl_get_env("GGML_SYCL_IQ3_S_DP4A_KS", 2);
        static const int rg = ggml_sycl_get_env("GGML_SYCL_IQ3_S_DP4A_RG", 16);
        const int bpr = ncols / QK_K;
#define IQ3_S_KS_LAUNCH(RG_, KS_, NB_) \
        iq3_s_moe_glu_dp4a_ks_esimd_launch<2, RG_, KS_, NB_>(vx_gate_base, vx_up_base, y, ids_dev, dst_base, ncols, nrows, \
            n_experts_used, n_tokens, expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride, \
            dst_token_stride, src1_token_stride, stream); \
        return true
        if (ks == 2 && bpr % 10 == 0) {
            if (rg == 4)  { IQ3_S_KS_LAUNCH(4, 2, 5); }
            if (rg == 16) { IQ3_S_KS_LAUNCH(16, 2, 5); }
            IQ3_S_KS_LAUNCH(8, 2, 5);
        }
        if (ks == 5 && bpr % 10 == 0) {
            if (rg == 4) { IQ3_S_KS_LAUNCH(4, 5, 2); }
            IQ3_S_KS_LAUNCH(8, 5, 2);
        }
        if (bpr % 2 == 0) {
            if (rg == 16) { IQ3_S_KS_LAUNCH(16, 2, 1); }
            IQ3_S_KS_LAUNCH(8, 2, 1);
        }
#undef IQ3_S_KS_LAUNCH
    }
    if (g_ggml_sycl_moe_esimd >= 3 && ncols <= IQ3_S_MOE_DP4A_MAX_COLS) {
        // 2 rows x 4 blocks per thread measured best on Arc Pro B60 (4 rows x 1 block: same at 8 tokens, 6% slower at 1)
        iq3_s_moe_glu_dp4a_esimd_launch<2, 8, 4>(vx_gate_base, vx_up_base, y, ids_dev, dst_base, ncols, nrows, n_experts_used,
            n_tokens, expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride, dst_token_stride,
            src1_token_stride, stream);
        return true;
    }
    // 1 and 4 rows per thread and 16 threads per work-group measured no better
    iq3_s_moe_glu_esimd_launch<2, 8>(vx_gate_base, vx_up_base, y, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
        expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride, dst_token_stride, src1_token_stride, stream);
    return true;
#else
    GGML_UNUSED_VARS(src0_type, vx_gate_base, vx_up_base, y, ids_dev, dst_base, ncols, nrows, n_experts_used, n_tokens,
        expert_weight_stride, dst_row_stride, src1_row_stride, ids_token_stride, dst_token_stride, src1_token_stride,
        glu_op, stream);
    return false;
#endif
}
