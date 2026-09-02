#!/usr/bin/env python3
"""The size ladders. One dimension varied at a time, others held nominal.

Structured rather than random so a heatmap has axes that mean something: each
op contributes a set of (varied dimension, ladder) series, and the nominal point
is shared between series so they intersect. Ladders are roughly geometric
because the thing being separated -- a fixed dispatch cost from a linear
compute term -- is only identifiable across decades, not across a narrow band.
Every ladder therefore starts far below any plausible break-even (a 64-MAC conv
is deliberately in the conv2d set) and ends above the largest instance the real
models contain.

`points()` yields (op, params_tuple, axis, value) so a plot can group by axis.
"""
from __future__ import annotations

CH = [8, 16, 32, 64, 128, 256, 512, 1024]
HW = [4, 8, 16, 32, 64, 128, 256]
TOK = [1, 4, 16, 50, 113, 256, 1024]
ELEM = [64, 1024, 16384, 262144, 1048576, 3145728, 12582912]
K = [1, 3, 5, 7]


def points():
    out = []
    def add(op, prm, axis, val): out.append((op, tuple(prm), axis, val))

    # conv2d: channels at fixed spatial; spatial at fixed channels; kernel
    for c in CH:            add("conv2d", [c, c, 32, 3], "channels", c)
    for h in HW:            add("conv2d", [64, 64, h, 3], "spatial", h)
    for k in K:             add("conv2d", [64, 64, 32, k], "kernel", k)
    for c in CH[:6]:        add("conv2d", [3, c, 224, 3], "out_ch@224", c)   # stem shapes

    for c in CH:            add("depthwise_conv2d", [c, 32, 3], "channels", c)
    for h in HW:            add("depthwise_conv2d", [64, h, 3], "spatial", h)

    # conv1x1 and linear at MATCHED arithmetic -- the kernel-choice comparison
    for c in CH:            add("conv1x1", [c, c, 50], "channels", c)
    for s in TOK:           add("conv1x1", [720, 2048, s], "tokens", s)
    for c in CH:            add("linear", [50, c, c], "channels", c)
    for s in TOK:           add("linear", [s, 720, 2048], "tokens", s)

    for b in [1, 4, 8, 15, 32]:  add("matmul_dyn", [b, 113, 64, 163], "heads", b)
    for s in TOK:                add("matmul_dyn", [15, s, 64, s], "seq", s)

    for r in [1, 8, 64, 512, 4096]:   add("softmax", [r, 1024], "rows", r)
    for c in [64, 256, 1024, 4096]:   add("softmax", [113, c], "cols", c)
    for r in [1, 8, 64, 512, 4096]:   add("layernorm", [r, 768], "rows", r)
    for c in CH[2:]:                  add("layernorm", [113, c], "width", c)

    for op in ("relu", "sigmoid", "tanh", "gelu", "elu", "add", "mul"):
        for n in ELEM:      add(op, [n], "elements", n)

    for c in CH:            add("maxpool2d", [c, 32, 2], "channels", c)
    for h in HW[1:]:        add("maxpool2d", [64, h, 2], "spatial", h)
    for c in CH:            add("avgpool_global", [c, 32], "channels", c)

    for r in [8, 64, 512, 4096]:      add("transpose", [r, 1024], "rows", r)
    for c in CH:                      add("concat_c", [c, 113, 2], "channels", c)
    return out


# int8 needs a quantized DLC; fp32/fp16 do not. DSP and HTA reject a float DLC
# outright ("Input[0] has incorrect Datatype 0x508") so those pairs are not
# attempted -- the sweep records why rather than emitting a blank cell.
PRECISION_BACKENDS = {
    "int8": ["Cpu", "Dsp", "Hta", "Gpu"],
    "fp32": ["Cpu", "Gpu"],
    "fp16": ["Gpu"],
}

if __name__ == "__main__":
    from collections import Counter
    p = points()
    c = Counter(x[0] for x in p)
    n_meas = sum(len(v) for v in PRECISION_BACKENDS.values())
    print(f"  {len(p)} size points over {len(c)} ops")
    for k, v in sorted(c.items()):
        print(f"    {k:<20} {v:>3}")
    print(f"  x {n_meas} (precision,backend) pairs = {len(p)*n_meas} measurements max")
