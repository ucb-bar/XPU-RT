// Copyright 2026 CompGen Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portable float32 op primitives for CompGen's ahead-of-time runtime.
// See compgen/ops.h for contract / scope notes.

#include "compgen/ops.h"

void cg_conv2d_f32(
    const float *in, float *out,
    int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int kH, int kW, int sH, int sW, int pH, int pW,
    const float *weight, const float *bias) {
  const int in_hw = Hin * Win;
  const int out_hw = Hout * Wout;
  const int wk_per_oc = Cin * kH * kW;

  for (int oc = 0; oc < Cout; ++oc) {
    const float b = bias ? bias[oc] : 0.0f;
    float *out_ch = out + oc * out_hw;
    const float *wt_oc = weight + oc * wk_per_oc;
    for (int oy = 0; oy < Hout; ++oy) {
      for (int ox = 0; ox < Wout; ++ox) {
        float acc = b;
        for (int ic = 0; ic < Cin; ++ic) {
          const float *in_ch = in + ic * in_hw;
          const float *wt_ic = wt_oc + ic * (kH * kW);
          for (int ky = 0; ky < kH; ++ky) {
            const int iy = oy * sH - pH + ky;
            if ((unsigned)iy >= (unsigned)Hin) continue;
            for (int kx = 0; kx < kW; ++kx) {
              const int ix = ox * sW - pW + kx;
              if ((unsigned)ix >= (unsigned)Win) continue;
              acc += in_ch[iy * Win + ix] * wt_ic[ky * kW + kx];
            }
          }
        }
        out_ch[oy * Wout + ox] = acc;
      }
    }
  }
}

void cg_relu_f32(float *x, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (x[i] < 0.0f) x[i] = 0.0f;
  }
}

void cg_add_f32(const float *a, const float *b, float *out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void cg_global_avgpool_f32(const float *in, float *out, int C, int H, int W) {
  const int hw = H * W;
  const float inv = 1.0f / (float)hw;
  for (int c = 0; c < C; ++c) {
    const float *ch = in + c * hw;
    float acc = 0.0f;
    for (int i = 0; i < hw; ++i) acc += ch[i];
    out[c] = acc * inv;
  }
}

void cg_linear_f32(
    const float *in, float *out,
    int Cin, int Cout,
    const float *weight, const float *bias) {
  for (int oc = 0; oc < Cout; ++oc) {
    float acc = bias ? bias[oc] : 0.0f;
    const float *w_row = weight + oc * Cin;
    for (int ic = 0; ic < Cin; ++ic) acc += w_row[ic] * in[ic];
    out[oc] = acc;
  }
}
