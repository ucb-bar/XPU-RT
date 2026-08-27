#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>

void kernel_linear_s8_elu_s8_mlp_control(
    const int8_t *input, const int8_t *weight,
    const int32_t *bias, int8_t *output,
    int M, int K, int N,
    int input_offset, int filter_offset, int linear_output_offset,
    int output_multiplier, int output_shift,
    int linear_activation_min, int linear_activation_max,
    float scale_linear_out, float scale_final_out,
    int activation_min, int activation_max, float alpha)
{
    const size_t vlmax = __riscv_vsetvlmax_e8m1();
    const vint16m2_t zero_products =
        __riscv_vmv_v_x_i16m2(0, vlmax);
    const vint16m1_t zero_i16 =
        __riscv_vmv_v_x_i16m1(0, 1);
    const vint32m1_t zero_i32 =
        __riscv_vmv_v_x_i32m1(0, 1);
    const uint32_t k_count = (uint32_t)(K > 0 ? K : 0);

    for (int m = 0; m < M; ++m) {
        const int8_t *in_row = input;
        uint32_t input_sum_bits = 0;

        if (K > 0)
            in_row = input + (ptrdiff_t)m * (ptrdiff_t)K;

        if (filter_offset != 0 && K > 0) {
            int k = 0;
            while (k < K) {
                size_t vl = __riscv_vsetvl_e8m1((size_t)(K - k));
                vint8m1_t xv = __riscv_vle8_v_i8m1(in_row + k, vl);
                vint16m1_t sumv =
                    __riscv_vwredsum_vs_i8m1_i16m1(xv, zero_i16, vl);
                int16_t chunk_sum =
                    __riscv_vmv_x_s_i16m1_i16(sumv);

                input_sum_bits += (uint32_t)(int32_t)chunk_sum;
                k += (int)vl;
            }
        }

        for (int n = 0; n < N; n += 4) {
            int nb = N - n;
            uint32_t acc_bits[4];
            uint32_t weight_sum_bits[4] = {0, 0, 0, 0};

            if (nb > 4)
                nb = 4;

#pragma GCC unroll 4
            for (int j = 0; j < nb; ++j)
                acc_bits[j] = bias ? (uint32_t)bias[n + j] : 0u;

            if (K > 0) {
                int k = 0;

                while (k < K) {
                    size_t vl = __riscv_vsetvl_e8m1((size_t)(K - k));
                    vint8m1_t xv =
                        __riscv_vle8_v_i8m1(in_row + k, vl);

#pragma GCC unroll 4
                    for (int j = 0; j < nb; ++j) {
                        const int8_t *w_row =
                            weight + (ptrdiff_t)(n + j) * (ptrdiff_t)K;
                        vint8m1_t wv =
                            __riscv_vle8_v_i8m1(w_row + k, vl);
                        vint16m2_t products =
                            __riscv_vwmacc_vv_i16m2(
                                zero_products, xv, wv, vl);
                        vint32m1_t dotv =
                            __riscv_vwredsum_vs_i16m2_i32m1(
                                products, zero_i32, vl);
                        int32_t chunk_dot =
                            __riscv_vmv_x_s_i32m1_i32(dotv);

                        acc_bits[j] += (uint32_t)chunk_dot;

                        if (input_offset != 0) {
                            vint16m1_t sumv =
                                __riscv_vwredsum_vs_i8m1_i16m1(
                                    wv, zero_i16, vl);
                            int16_t chunk_sum =
                                __riscv_vmv_x_s_i16m1_i16(sumv);
                            weight_sum_bits[j] +=
                                (uint32_t)(int32_t)chunk_sum;
                        }
                    }

                    k += (int)vl;
                }
            }

#pragma GCC unroll 4
            for (int j = 0; j < nb; ++j) {
                int32_t acc;
                int32_t scaled;
                int8_t linear_int8;
                uint32_t u = acc_bits[j];

                u += (uint32_t)input_offset * weight_sum_bits[j];
                u += (uint32_t)filter_offset * input_sum_bits;
                u += k_count *
                     (uint32_t)input_offset *
                     (uint32_t)filter_offset;
                memcpy(&acc, &u, sizeof(acc));

                {
                    int64_t prod =
                        (int64_t)acc * (int64_t)output_multiplier;
                    prod = (prod + (1LL << 30)) >> 31;
                    scaled = (int32_t)prod;
                }

                if (output_shift > 0) {
                    scaled = (int32_t)(
                        ((int64_t)scaled +
                         ((int64_t)1 << (output_shift - 1))) >>
                        output_shift);
                } else if (output_shift < 0) {
                    scaled = scaled << (-output_shift);
                }

                scaled += linear_output_offset;
                if (scaled < linear_activation_min)
                    scaled = linear_activation_min;
                if (scaled > linear_activation_max)
                    scaled = linear_activation_max;

                linear_int8 = (int8_t)scaled;

                {
                    float f = (float)linear_int8 * scale_linear_out;
                    float y = (f > 0.0f)
                                  ? f
                                  : alpha * (expf(f) - 1.0f);
                    int32_t v =
                        (int32_t)roundf(y / scale_final_out);

                    if (v < activation_min)
                        v = activation_min;
                    if (v > activation_max)
                        v = activation_max;

                    output[(ptrdiff_t)m * (ptrdiff_t)N + n + j] =
                        (int8_t)v;
                }
            }
        }
    }
}