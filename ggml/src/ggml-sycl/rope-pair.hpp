#ifndef GGML_SYCL_ROPE_PAIR_HPP
#define GGML_SYCL_ROPE_PAIR_HPP

#include "common.hpp"

// The angle of one rotated pair of a forward NEOX or multi-section (MROPE, IMROPE) rope, for kernels
// that fuse a rope into their own work: the same arithmetic as rope_neox()/rope_multi() and rope_yarn()
// in rope.cpp. Pair j rotates channels n_offs + j and n_offs + j + n_dims/2.
struct ggml_sycl_rope_pair_params {
    int   n_dims, n_offs, sections[4];
    bool  multi, is_imrope;
    float freq_scale, ext_factor, attn_factor, theta_scale, corr_dims[2];
};

// true if rope is one these kernels take: forward NEOX or multi-section, not vision
static inline bool ggml_sycl_rope_pair_supported(const ggml_tensor * rope) {
    const int mode   = ggml_get_op_params_i32(rope, 2);
    const int n_dims = ggml_get_op_params_i32(rope, 1);
    const int n_offs = ggml_get_op_params_i32(rope, 15);
    const bool multi = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    if (mode == GGML_ROPE_TYPE_VISION || (!multi && mode != GGML_ROPE_TYPE_NEOX)) {
        return false;
    }
    return n_dims > 0 && n_dims % 2 == 0 && n_offs >= 0 && n_offs % 2 == 0 && n_offs + n_dims <= rope->ne[0];
}

static inline ggml_sycl_rope_pair_params ggml_sycl_rope_pair_init(const ggml_tensor * rope) {
    ggml_sycl_rope_pair_params p;
    float freq_base, beta_fast, beta_slow;
    const int mode       = ggml_get_op_params_i32(rope, 2);
    const int n_ctx_orig = ggml_get_op_params_i32(rope, 4);
    p.n_dims = ggml_get_op_params_i32(rope, 1);
    p.n_offs = ggml_get_op_params_i32(rope, 15);
    memcpy(&freq_base, (const int32_t *) rope->op_params + 5, sizeof(float));
    memcpy(&p.freq_scale, (const int32_t *) rope->op_params + 6, sizeof(float));
    memcpy(&p.ext_factor, (const int32_t *) rope->op_params + 7, sizeof(float));
    memcpy(&p.attn_factor, (const int32_t *) rope->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (const int32_t *) rope->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (const int32_t *) rope->op_params + 10, sizeof(float));
    memcpy(p.sections, (const int32_t *) rope->op_params + 11, sizeof(int) * 4);
    p.multi       = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    p.is_imrope   = mode == GGML_ROPE_TYPE_IMROPE;
    p.theta_scale = powf(freq_base, -2.0f / p.n_dims);
    ggml_rope_yarn_corr_dims(p.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, p.corr_dims);
    return p;
}

// cos and sin of pair j for position row i2 of n_pos rows (multi-section reads pos[i2 + n_pos*k])
static inline void ggml_sycl_rope_pair_cs(const ggml_sycl_rope_pair_params & p, const int32_t * pos, int n_pos,
                                          int i2, int j, const float * ff, float & cos_theta, float & sin_theta) {
    const int   iw = 2 * j;
    const float pw = sycl::pow(p.theta_scale, iw / 2.0f);

    float theta_base = 0.0f;
    if (!p.multi) {
        theta_base = pos[i2] * pw;
    } else {
        const int sect_dims = p.sections[0] + p.sections[1] + p.sections[2] + p.sections[3];
        const int sec_w     = p.sections[1] + p.sections[0];
        const int sector    = j % sect_dims;
        if (p.is_imrope) {
            if (sector % 3 == 1 && sector < 3 * p.sections[1]) {
                theta_base = pos[i2 + n_pos * 1] * pw;
            } else if (sector % 3 == 2 && sector < 3 * p.sections[2]) {
                theta_base = pos[i2 + n_pos * 2] * pw;
            } else if (sector % 3 == 0 && sector < 3 * p.sections[0]) {
                theta_base = pos[i2] * pw;
            } else {
                theta_base = pos[i2 + n_pos * 3] * pw;
            }
        } else {
            if (sector < p.sections[0]) {
                theta_base = pos[i2] * pw;
            } else if (sector >= p.sections[0] && sector < sec_w) {
                theta_base = pos[i2 + n_pos * 1] * pw;
            } else if (sector >= sec_w && sector < sec_w + p.sections[2]) {
                theta_base = pos[i2 + n_pos * 2] * pw;
            } else if (sector >= sec_w + p.sections[2]) {
                theta_base = pos[i2 + n_pos * 3] * pw;
            }
        }
    }
    const float theta_extrap = theta_base / (ff ? ff[j] : 1.0f);

    float theta  = p.freq_scale * theta_extrap;
    float mscale = p.attn_factor;
    if (p.ext_factor != 0.0f) {
        const float y        = (iw / 2 - p.corr_dims[0]) / sycl::max(0.001f, p.corr_dims[1] - p.corr_dims[0]);
        const float ramp_mix = (1.0f - sycl::min(1.0f, sycl::max(0.0f, y))) * p.ext_factor;
        theta                = theta * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * sycl::log(1.0f / p.freq_scale);
    }
    cos_theta = sycl::cos(theta) * mscale;
    sin_theta = sycl::sin(theta) * mscale;
}

#endif // GGML_SYCL_ROPE_PAIR_HPP
