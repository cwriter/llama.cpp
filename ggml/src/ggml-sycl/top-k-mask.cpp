#include "top-k-mask.hpp"
#include "kq-mask-bits.hpp"

static constexpr int SYCL_TOP_K_MASK_WG = 256;

bool ggml_sycl_top_k_mask_supported(const ggml_tensor * op) {
    const ggml_tensor * mask   = op->src[0];
    const ggml_tensor * top_k  = op->src[1];
    const ggml_tensor * score  = op->src[2];
    const ggml_tensor * blocks = op->src[3];
    const ggml_tensor * extra  = op->src[4];
    if ((mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) || op->type != mask->type ||
        mask->nb[0] != ggml_type_size(mask->type) || op->nb[0] != ggml_type_size(op->type) ||
        top_k->type != GGML_TYPE_I32 || (score && score->type != GGML_TYPE_F32) ||
        (blocks && blocks->type != GGML_TYPE_I32) || (extra && extra->type != GGML_TYPE_I32)) {
        return false;
    }
    // The k-pool form (qwen4exp, glm5-next). With it off those models build the scattered selection
    // list instead, which qsa-mask.cpp also runs as sparse flash attention; this form needs no
    // scatter, dump rows or contiguous copy, so it is the default.
    static const bool blocks_on = ggml_sycl_get_env("GGML_SYCL_TOP_K_MASK_BLOCKS", 1) != 0;
    return !blocks || blocks_on;
}

// One work-group per row: fill the row with -inf, then copy the mask cells of the live picks and
// the extra cells. A cell kept twice gets the same value twice.
template <typename T, bool BITS>
static void top_k_mask_sycl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mask   = dst->src[0];
    const ggml_tensor * top_k  = dst->src[1];
    const ggml_tensor * score  = dst->src[2];
    const ggml_tensor * blocks = dst->src[3];
    const ggml_tensor * extra  = dst->src[4];

    const int64_t n_kv    = mask->ne[0];
    const int64_t ne1     = dst->ne[1];
    const int64_t ne2     = dst->ne[2];
    const int64_t n_rows  = ggml_nrows(dst);
    const int64_t m1      = mask->ne[1];
    const int64_t m2      = mask->ne[2];
    const int64_t m3      = mask->ne[3];
    const int64_t n_top   = top_k->ne[0];
    const int64_t n_blk   = blocks ? blocks->ne[0] : 1;
    const int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    const int64_t n_extra = extra ? extra->ne[0] : 0;

    // a packed mask keeps its f16 shape but stores ggml_sycl_kq_mask_row_bytes() per row, rows back to back
    const size_t rbyte = ggml_sycl_kq_mask_row_bytes(n_kv);
    const size_t sm1   = BITS ? rbyte : mask->nb[1];
    const size_t sm2   = BITS ? rbyte * m1 : mask->nb[2];
    const size_t sm3   = BITS ? rbyte * m1 * m2 : mask->nb[3];

    const size_t sd1 = dst->nb[1], sd2 = dst->nb[2], sd3 = dst->nb[3];
    const size_t sk0 = top_k->nb[0], sk1 = top_k->nb[1], sk2 = top_k->nb[2], sk3 = top_k->nb[3];
    const size_t ss0 = score ? score->nb[0] : 0, ss1 = score ? score->nb[1] : 0;
    const size_t ss2 = score ? score->nb[2] : 0, ss3 = score ? score->nb[3] : 0;
    const size_t sb0 = blocks ? blocks->nb[0] : 0, sb1 = blocks ? blocks->nb[1] : 0;
    const size_t se0 = extra ? extra->nb[0] : 0, se1 = extra ? extra->nb[1] : 0;
    const size_t se2 = extra ? extra->nb[2] : 0, se3 = extra ? extra->nb[3] : 0;

    const char * mask_d   = (const char *) mask->data;
    const char * top_k_d  = (const char *) top_k->data;
    // after the tiled indexer the score tensor holds its staging, and the picks' scores are kept apart
    const float * picked_d = score ? ggml_sycl_indexer_picked(ctx, top_k) : nullptr;
    const char *  score_d  = score && !picked_d ? (const char *) score->data : nullptr;
    const char * blocks_d = blocks ? (const char *) blocks->data : nullptr;
    const char * extra_d  = extra ? (const char *) extra->data : nullptr;
    char *       dst_d    = (char *) dst->data;

    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(n_rows * SYCL_TOP_K_MASK_WG), sycl::range<1>(SYCL_TOP_K_MASK_WG)),
        [=](sycl::nd_item<1> it) {
            const int64_t ir  = (int64_t) it.get_group(0);
            const int64_t lid = (int64_t) it.get_local_id(0);
            const int64_t i1  = ir % ne1;
            const int64_t i2  = (ir / ne1) % ne2;
            const int64_t i3  = ir / (ne1 * ne2);

            const char * mask_row  = mask_d + (i1 % m1) * sm1 + (i2 % m2) * sm2 + (i3 % m3) * sm3;
            T *          dst_row   = (T *) (dst_d + i1 * sd1 + i2 * sd2 + i3 * sd3);
            const char * top_k_row = top_k_d + i1 * sk1 + i2 * sk2 + i3 * sk3;

            const T ninf = T(-INFINITY);
            for (int64_t c = lid; c < n_kv; c += SYCL_TOP_K_MASK_WG) {
                dst_row[c] = ninf;
            }

            // the scatter below writes cells other items filled
            sycl::group_barrier(it.get_group());

            auto keep = [&](int32_t c) {
                if (c < 0 || c >= n_kv) {
                    return;
                }
                if constexpr (BITS) {
                    const uint32_t w = ((const uint32_t *) mask_row)[c >> 5];
                    dst_row[c]       = ((w >> (c & 31)) & 1u) ? T(0.0f) : ninf;
                } else {
                    dst_row[c] = ((const T *) mask_row)[c];
                }
            };

            for (int64_t i = lid; i < n_top + n_extra; i += SYCL_TOP_K_MASK_WG) {
                if (i >= n_top) {
                    keep(*(const int32_t *) (extra_d + (i - n_top) * se0 + i1 * se1 + i2 * se2 + i3 * se3));
                    continue;
                }
                const int32_t p = *(const int32_t *) (top_k_row + i * sk0);
                if (p < 0 || p >= n_pick) {
                    continue;
                }
                if (picked_d) {
                    if (!(picked_d[ir * n_top + i] > -INFINITY)) {
                        continue;
                    }
                } else if (score_d) {
                    const float s = *(const float *) (score_d + p * ss0 + i1 * ss1 + i2 * ss2 + i3 * ss3);
                    if (!(s > -INFINITY)) {
                        continue;
                    }
                }
                if (!blocks_d) {
                    keep(p);
                    continue;
                }
                for (int64_t k = 0; k < n_blk; ++k) {
                    keep(*(const int32_t *) (blocks_d + k * sb0 + p * sb1));
                }
            }
        });
}

void ggml_sycl_op_top_k_mask(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);
    GGML_ASSERT(dst->type == dst->src[0]->type && dst->src[1]->type == GGML_TYPE_I32);

    if (ggml_sycl_kq_mask_is_bits(dst->src[0])) {
        // only the whole mask is packed (kq_mask_reader_taught), so its rows are back to back
        GGML_ASSERT(dst->type == GGML_TYPE_F16);
        top_k_mask_sycl<sycl::half, true>(ctx, dst);
        ggml_sycl_kq_mask_taught(ctx, dst);
        return;
    }
    switch (dst->type) {
        case GGML_TYPE_F16:
            top_k_mask_sycl<sycl::half, false>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            top_k_mask_sycl<float, false>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
