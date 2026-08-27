#include <stdint.h>
#include <riscv_vector.h>

void kernel_linear_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int M, int K, int N,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    if (M <= 0 || N <= 0)
        return;

    if (K <= 0 ||
        input_offset < -32640 || input_offset > 32640 ||
        filter_offset < -32640 || filter_offset > 32640) {
        for (int m = 0; m < M; ++m) {
            const int8_t *in_row = input + (long)m * (long)K;
            int8_t *out_row = output + (long)m * (long)N;

            for (int n = 0; n < N; ++n) {
                int32_t acc = bias ? bias[n] : 0;
                const int8_t *w = weight + (long)n * (long)K;

                for (int k = 0; k < K; ++k) {
                    int32_t in_v = (int32_t)in_row[k] + input_offset;
                    int32_t w_v = (int32_t)w[k] + filter_offset;
                    acc += in_v * w_v;
                }

                int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                prod = (prod + (1LL << 30)) >> 31;
                int32_t scaled = (int32_t)prod;

                if (output_shift > 0) {
                    scaled = (int32_t)(((int64_t)scaled +
                              ((int64_t)1 << (output_shift - 1))) >>
                              output_shift);
                } else if (output_shift < 0) {
                    scaled = scaled << (-output_shift);
                }

                scaled += output_offset;
                if (scaled < activation_min) scaled = activation_min;
                if (scaled > activation_max) scaled = activation_max;
                out_row[n] = (int8_t)scaled;
            }
        }
        return;
    }

    const unsigned long vlmax = __riscv_vsetvlmax_e8m1();
    const int full_chunks = K / (int)vlmax;
    const unsigned long tail = (unsigned long)(K % (int)vlmax);

    for (int m = 0; m < M; ++m) {
        const int8_t *in_row = input + (long)m * (long)K;
        int8_t *out_row = output + (long)m * (long)N;
        int n = 0;

        for (; n + 3 < N; n += 4) {
            const int8_t *w0 = weight + (long)(n + 0) * (long)K;
            const int8_t *w1 = weight + (long)(n + 1) * (long)K;
            const int8_t *w2 = weight + (long)(n + 2) * (long)K;
            const int8_t *w3 = weight + (long)(n + 3) * (long)K;

            uint32_t a0 = bias ? (uint32_t)bias[n + 0] : 0u;
            uint32_t a1 = bias ? (uint32_t)bias[n + 1] : 0u;
            uint32_t a2 = bias ? (uint32_t)bias[n + 2] : 0u;
            uint32_t a3 = bias ? (uint32_t)bias[n + 3] : 0u;

            if (full_chunks != 0) {
                vint32m4_t s0 = __riscv_vmv_v_x_i32m4(0, vlmax);
                vint32m4_t s1 = __riscv_vmv_v_x_i32m4(0, vlmax);
                vint32m4_t s2 = __riscv_vmv_v_x_i32m4(0, vlmax);
                vint32m4_t s3 = __riscv_vmv_v_x_i32m4(0, vlmax);

                for (int q = 0; q < full_chunks; ++q) {
                    long k = (long)q * (long)vlmax;
                    vint8m1_t in8 =
                        __riscv_vle8_v_i8m1(in_row + k, vlmax);
                    vint16m2_t in16 =
                        __riscv_vsext_vf2_i16m2(in8, vlmax);
                    in16 = __riscv_vadd_vx_i16m2(
                        in16, (int16_t)input_offset, vlmax);

                    vint8m1_t w8 =
                        __riscv_vle8_v_i8m1(w0 + k, vlmax);
                    vint16m2_t w16 =
                        __riscv_vsext_vf2_i16m2(w8, vlmax);
                    w16 = __riscv_vadd_vx_i16m2(
                        w16, (int16_t)filter_offset, vlmax);
                    s0 = __riscv_vwmacc_vv_i32m4(s0, in16, w16, vlmax);

                    w8 = __riscv_vle8_v_i8m1(w1 + k, vlmax);
                    w16 = __riscv_vsext_vf2_i16m2(w8, vlmax);
                    w16 = __riscv_vadd_vx_i16m2(
                        w16, (int16_t)filter_offset, vlmax);
                    s1 = __riscv_vwmacc_vv_i32m4(s1, in16, w16, vlmax);

                    w8 = __riscv_vle8_v_i8m1(w2 + k, vlmax);
                    w16 = __riscv_vsext_vf2_i16m2(w8, vlmax);
                    w16 = __riscv_vadd_vx_i16m2(
                        w16, (int16_t)filter_offset, vlmax);
                    s2 = __riscv_vwmacc_vv_i32m4(s2, in16, w16, vlmax);

                    w8 = __riscv_vle8_v_i8m1(w3 + k, vlmax);
                    w16 = __riscv_vsext_vf2_i16m2(w8, vlmax);
                    w16 = __riscv_vadd_vx_i16m2(
                        w16, (int16_t)filter_offset, vlmax);
                    s3 = __riscv_vwmacc_vv_i32m4(s3, in16, w16, vlmax);
                }

                vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vlmax);
                vint32m1_t r;

                r = __riscv_vredsum_vs_i32m4_i32m1(s0, z, vlmax);
                a0 += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
                r = __riscv_vredsum_vs_i32m4_i32m1(s1, z, vlmax);
                a1 += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
                r = __riscv_vredsum_vs_i32m4_i32m1(s2, z, vlmax);
                a2 += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
                r = __riscv_vredsum_vs_i32m4_i32m1(s3, z, vlmax);
                a3 += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
            }

            if (tail != 0) {
                long k = (long)full_chunks * (long)vlmax;
                vint8m1_t in8 = __riscv_vle8_v_i8m1(in_row + k, tail);
                vint16m2_t in16 =
                    __riscv_vsext_vf2_i16m2(in8, tail);
                in16 = __riscv_vadd_vx_i16m2(
                    in16, (int16_t)input_offset, tail);

                vint32m4_t s0 = __riscv_vmv_v_x_i32m4(0, tail);
                vint32m4_t s1 = __riscv_vmv_v_x_i32m4(0, tail);
                vint32m4_t s2 = __riscv_vmv_v_x_i32m4(0, tail);
                vint32m4_t s3 = __riscv_vmv_v_x_i32m4(0, tail);
                vint8m1_t w8;
                vint16m2_t w16;

                w8 = __riscv_vle8_v_i8m1(w0 + k, tail);
                w16 = __riscv_vsext_vf2_i16m2(w8, tail);
                w16 = __riscv_vadd_vx_i16m2(
                    w16, (int16_t)filter_offset, tail);
                s0 = __riscv_vwmacc_vv_i32m4(s0, in16, w16, tail);

                w8 = __riscv_vle8_v_i8m1(w1 + k, tail);
                w16 = __riscv_vsext_vf2_i16m2(w8, tail);
                w16 = __riscv_vadd_vx_i16m2(
                    w16, (int16_t)filter_offset, tail);
                s1 = __riscv_vwmacc_vv_i32m4(s1, in16, w16, tail);

                w8 = __riscv_vle8_v_i8m1(w2 + k, tail);
                w16 = __riscv_vsext_vf2_i16m2(w8, tail);
                w16 = __riscv_vadd_vx_i16m2(
                    w16, (int16_t)filter_offset, tail);
                s2 = __riscv_vwmacc_vv_i32m4(s2, in16, w16, tail);

                w8 = __riscv_vle8_v_i8m1(w3 + k, tail);
                w16 = __riscv_vsext_vf2_i16m2(w8, tail);
                w16 = __riscv_vadd_vx_i16m2(
                    w16, (int16_t)filter_offset, tail);
                s3 = __riscv_vwmacc_vv_i32m4(s3, in16, w16, tail);

                vint32m1_t z = __riscv_vmv_v_x_i32m1(0, tail);
                vint32m1_t r;

                r = __riscv_vredsum_vs_i32m4_i32m1(s0, z, tail);
                a0 += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
                r = __riscv_vredsum_vs_i32m4_i32m1(s1, z, tail);
                a1 += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
                r = __riscv_vredsum_vs_i32m4_i32m1(s2, z, tail);
                a2 += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
                r = __riscv_vredsum_vs_i32m4_i32m1(s3, z, tail);
                a3 += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
            }

            int32_t accumulators[4];
            accumulators[0] = (int32_t)a0;
            accumulators[1] = (int32_t)a1;
            accumulators[2] = (int32_t)a2;
            accumulators[3] = (int32_t)a3;

            for (int j = 0; j < 4; ++j) {
                int64_t prod =
                    (int64_t)accumulators[j] * (int64_t)output_multiplier;
                prod = (prod + (1LL << 30)) >> 31;
                int32_t scaled = (int32_t)prod;

                if (output_shift > 0) {
                    scaled = (int32_t)(((int64_t)scaled +
                              ((int64_t)1 << (output_shift - 1))) >>
                              output_shift);
                } else if (output_shift < 0) {
                    scaled = scaled << (-output_shift);
                }

                scaled += output_offset;
                if (scaled < activation_min) scaled = activation_min;
                if (scaled > activation_max) scaled = activation_max;
                out_row[n + j] = (int8_t)scaled;
            }
        }

        for (; n < N; ++n) {
            const int8_t *w = weight + (long)n * (long)K;
            uint32_t a = bias ? (uint32_t)bias[n] : 0u;

            if (full_chunks != 0) {
                vint32m4_t sum = __riscv_vmv_v_x_i32m4(0, vlmax);

                for (int q = 0; q < full_chunks; ++q) {
                    long k = (long)q * (long)vlmax;
                    vint8m1_t in8 =
                        __riscv_vle8_v_i8m1(in_row + k, vlmax);
                    vint8m1_t w8 =
                        __riscv_vle8_v_i8m1(w + k, vlmax);
                    vint16m2_t in16 =
                        __riscv_vsext_vf2_i16m2(in8, vlmax);
                    vint16m2_t w16 =
                        __riscv_vsext_vf2_i16m2(w8, vlmax);

                    in16 = __riscv_vadd_vx_i16m2(
                        in16, (int16_t)input_offset, vlmax);
                    w16 = __riscv_vadd_vx_i16m2(
                        w16, (int16_t)filter_offset, vlmax);
                    sum = __riscv_vwmacc_vv_i32m4(
                        sum, in16, w16, vlmax);
                }

                vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vlmax);
                vint32m1_t r =
                    __riscv_vredsum_vs_i32m4_i32m1(sum, z, vlmax);
                a += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
            }

            if (tail != 0) {
                long k = (long)full_chunks * (long)vlmax;
                vint8m1_t in8 =
                    __riscv_vle8_v_i8m1(in_row + k, tail);
                vint8m1_t w8 =
                    __riscv_vle8_v_i8m1(w + k, tail);
                vint16m2_t in16 =
                    __riscv_vsext_vf2_i16m2(in8, tail);
                vint16m2_t w16 =
                    __riscv_vsext_vf2_i16m2(w8, tail);

                in16 = __riscv_vadd_vx_i16m2(
                    in16, (int16_t)input_offset, tail);
                w16 = __riscv_vadd_vx_i16m2(
                    w16, (int16_t)filter_offset, tail);

                vint32m4_t sum = __riscv_vmv_v_x_i32m4(0, tail);
                sum = __riscv_vwmacc_vv_i32m4(
                    sum, in16, w16, tail);

                vint32m1_t z = __riscv_vmv_v_x_i32m1(0, tail);
                vint32m1_t r =
                    __riscv_vredsum_vs_i32m4_i32m1(sum, z, tail);
                a += (uint32_t)__riscv_vmv_x_s_i32m1_i32(r);
            }

            int32_t acc = (int32_t)a;
            int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
            prod = (prod + (1LL << 30)) >> 31;
            int32_t scaled = (int32_t)prod;

            if (output_shift > 0) {
                scaled = (int32_t)(((int64_t)scaled +
                          ((int64_t)1 << (output_shift - 1))) >>
                          output_shift);
            } else if (output_shift < 0) {
                scaled = scaled << (-output_shift);
            }

            scaled += output_offset;
            if (scaled < activation_min) scaled = activation_min;
            if (scaled > activation_max) scaled = activation_max;
            out_row[n] = (int8_t)scaled;
        }
    }
}