// Copyright 2026 CompGen Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portable float32 op primitives for CompGen's ahead-of-time runtime.
//
// These are the always-correct fallback implementations the foundational
// runtime ships. Target packs (e.g. Saturn OPU's VOPACC mmt4d, Triton,
// vendor BLAS) expose their specialised variants under distinct symbols
// and the code generator emits the appropriate call based on the
// target's capability spec. The portable versions below are what the
// runtime executes when no specialisation applies.
//
// Tensors are NCHW, N=1 (batch-1 inference). Shapes are explicit
// arguments — no opaque tensor struct — so the ABI is trivially
// testable on the host via ctypes.

#ifndef COMPGEN_OPS_H_
#define COMPGEN_OPS_H_

#include "compgen/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// conv2d: zero-padded, explicit stride, weight layout
// [Cout, Cin, kH, kW], bias [Cout] (may be NULL).
void cg_conv2d_f32(
    const float *in, float *out,
    int Cin, int Hin, int Win,
    int Cout, int Hout, int Wout,
    int kH, int kW, int sH, int sW, int pH, int pW,
    const float *weight, const float *bias);

void cg_relu_f32(float *x, size_t n);
void cg_add_f32(const float *a, const float *b, float *out, size_t n);

void cg_global_avgpool_f32(const float *in, float *out,
                           int C, int H, int W);

void cg_linear_f32(
    const float *in, float *out,
    int Cin, int Cout,
    const float *weight, const float *bias);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // COMPGEN_OPS_H_
