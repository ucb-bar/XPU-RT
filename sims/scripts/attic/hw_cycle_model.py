"""HW cycle model for the Model DSE — grounds latency in the real SoC (task #24).

Replaces dse_pareto.py's x86 wall-clock proxy with a per-op **cycle** estimate on
the actual target (Loren's spec: 1-hart Rocket + Saturn RVV-128 FP16 + a 32x32
weight-stationary int8 Gemmini w/ Q0.31 requant, at ~60 MHz — 100 MHz was the old
assumption, Gemmini won't close it).

Three cycle sources, tagged per-op so measured never masquerades as estimated:

  * ``firesim``      — a directly-measured FireSim cycle count for THIS op shape
                       (from gen/profile/<hw>/firesim_rocket_saturn/**/results.csv;
                       FireSim's target is 1 GHz so 1 cycle == 1 ns == mean_time_ns).
  * ``loren_est``    — Loren's per-shape int8 conv/GEMM cycle estimate.
  * ``extrapolated`` — MACs x a measured efficiency anchor (int8 GEMM/conv on
                       Gemmini) or a measured per-element cost (FP16 elementwise on
                       RVV: softmax / LayerNorm / GELU / LSTM gating), for ops with
                       no measured shape (the whole ViT attention/norm stack).

The int8 conv/linear MACs map to Gemmini; softmax/LayerNorm/GELU/LSTM-gating are
FP16-only on the Saturn/scalar path (Gemmini is int8-GEMM only) — the DSE's whole
point is that these float ops are the un-accelerated wall. We report an OPTIMISTIC
bound (Loren's ~35 MAC/cyc peak int8 util) and a REALISTIC bound (the measured
DroNet blended efficiency, which bakes in real im2col/DMA/tiny-shape overhead), so
the 10 Hz feasibility verdict is a range, not false precision.
"""

from __future__ import annotations

import csv
import glob
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # XPU-RT
PROFILE_ROOT = os.path.join(REPO, "gen", "profile")

# FireSim target clock: 1 cycle == 1 ns (see docstring).
FIRESIM_HZ = 1_000_000_000

# ---- Loren's per-shape int8 conv/GEMM cycle estimates (Slack, "should be remeasured") ----
# Peak observed int8 utilisation ~35 MAC/cycle on a well-shaped conv (23.0M MACs / 654,633 cyc).
LOREN_PEAK_MAC_PER_CYC = 35.1
# FP16 elementwise cost on RVV, calibrated from measured mlp_control elu (26,356 cyc / 256 el).
RVV_CYC_PER_ELEM = 103.0
# Pass-count multipliers for the compound FP16 ops (relative to a 1-pass elementwise like elu).
FP16_OP_PASSES = {"softmax": 3.0, "layernorm": 4.0, "gelu": 3.0, "sigmoid": 1.0, "tanh": 1.0}


def _macs_from_shape(op: str, shape: str) -> int:
    """MACs implied by a profile CSV shape string, for the efficiency calibration."""
    d = dict(kv.split("=") for kv in shape.split(";") if "=" in kv)
    try:
        if op in ("conv2d_s8", "conv2d_batchnorm2d_s8", "conv2d_batchnorm2d_silu_s8"):
            return (int(d["OH"]) * int(d["OW"]) * int(d["OC"]) *
                    int(d["IC"]) * int(d["KH"]) * int(d["KW"]))
        if op in ("linear_s8", "linear"):
            return int(d["M"]) * int(d["K"]) * int(d["N"])
    except (KeyError, ValueError):
        return 0
    return 0


def load_measured():
    """Return {(backend, op, shape): cycles} (source=firesim only — directly measured)
    and {(backend, model, variant): total_cycles}.

    Each model appears twice per backend: ``dronet.int8`` (non-fused, source=firesim —
    real per-op measurements) and ``dronet.int8_fused`` (source=firesim_sum_constituents
    — derived by summing constituents). We keep them SEPARATE (variant = fused|nonfused),
    keep only ``topo_0`` (skip ``topo_0_1`` reruns), and feed only the measured
    (source=firesim) rows into the per-op join/efficiency anchor."""
    per_op, per_model = {}, {}
    for csv_path in glob.glob(os.path.join(PROFILE_ROOT, "*", "firesim_rocket_saturn",
                                            "*", "*", "*", "*", "results.csv")):
        parts = csv_path.split(os.sep)
        if parts[-2] != "topo_0":            # skip topo_0_1 / other reruns
            continue
        backend = parts[parts.index("profile") + 1]
        model = parts[parts.index("firesim_rocket_saturn") + 1]
        basename = parts[parts.index("firesim_rocket_saturn") + 2]  # dronet.int8[_fused]
        variant = "fused" if basename.endswith("_fused") else "nonfused"
        total = 0
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ns = row.get("mean_time_ns", "")
                if not ns or not ns[0].isdigit():
                    continue
                cyc = float(ns)  # 1 ns == 1 cycle
                total += cyc
                op, shape, source = row.get("op", ""), row.get("shape", ""), row.get("source", "")
                if op and shape and source == "firesim":  # measured only
                    per_op[(backend, op, shape)] = cyc
        per_model[(backend, model, variant)] = total
    return per_op, per_model


_GEMM_OPS = ("conv2d_s8", "conv2d_batchnorm2d_s8", "conv2d_batchnorm2d_silu_s8",
             "linear_s8")


def gemmini_gemm_eff(per_op) -> float:
    """MAC/cycle achieved by measured int8 conv/GEMM ops on gemmini_q31 (the realistic
    anchor). Divides conv/linear MACs by conv/linear CYCLES only — it captures real
    im2col/DMA/tiny-shape tiling overhead but excludes the BN/pool/add sideshows (those
    are separate ops accounted elsewhere), so it is a fair int8-GEMM throughput anchor.
    Sits well below Loren's 35 MAC/cyc peak because the measured shapes have small
    channel counts that underfill the 32x32 array."""
    macs = cyc = 0
    for (backend, op, shape), c in per_op.items():
        if backend != "gemmini_q31" or op not in _GEMM_OPS:
            continue
        m = _macs_from_shape(op, shape)
        if m:
            macs += m
            cyc += c
    return macs / cyc if cyc else 1.0


# ---- per-op cycle estimate for a MODELED op (op dict from MacCounter record mode) ----
def _elems(shape: str, keys) -> int:
    d = dict(kv.split("=") for kv in shape.split(";") if "=" in kv)
    n = 1
    for k in keys:
        n *= int(d.get(k, 1))
    return n


def op_cycles(op: dict, per_op, eff_realistic: float):
    """Return (optimistic_cyc, realistic_cyc, backend, source) for one recorded op.

    op has {op, shape, macs, groups}. conv2d/linear -> Gemmini int8; lstm split into
    its gate GEMM (Gemmini) + gating elementwise (RVV FP16)."""
    kind = op["op"]
    macs = op["macs"]
    if kind in ("conv2d", "linear"):
        # measured exact-shape hit? (join against gemmini_q31 int8 shapes)
        prof_op = "conv2d_s8" if kind == "conv2d" else "linear_s8"
        hit = per_op.get(("gemmini_q31", prof_op, op["shape"]))
        if hit is not None:
            return hit, hit, "gemmini", "firesim"
        opt = macs / LOREN_PEAK_MAC_PER_CYC
        real = macs / eff_realistic
        # Grouped/depthwise conv underfills a weight-stationary systolic array: the
        # contraction (K) dim per output is only (IC/groups) channels, so the 32-tall
        # array runs ~32/cpg empty. Penalty = clamp(32/cpg, 1, 8) (capped; real HW has
        # some depthwise datapath help). cpg = in_channels/groups.
        if kind == "conv2d" and op["groups"] > 1:
            d = dict(kv.split("=") for kv in op["shape"].split(";") if "=" in kv)
            cpg = max(1, int(d.get("IC", op["groups"])) // op["groups"])
            pen = max(1.0, min(32.0 / cpg, 8.0))
            opt *= pen
            real *= pen
        return opt, real, "gemmini", "extrapolated"
    if kind == "lstm":
        opt = macs / LOREN_PEAK_MAC_PER_CYC
        real = macs / eff_realistic
        hid = _elems(op["shape"], ["HID"]) * _elems(op["shape"], ["LAYERS"])
        gate = 4 * hid * RVV_CYC_PER_ELEM  # sigmoid/tanh/Hadamard gating on RVV
        return opt + gate, real + gate, "gemmini+rvv", "extrapolated"
    return 0.0, 0.0, "?", "extrapolated"


def fp16_op_cycles(kind: str, n_elems: int):
    """Cycles for a float elementwise/compound op on the RVV FP16 path (no Gemmini)."""
    passes = FP16_OP_PASSES.get(kind, 1.0)
    c = n_elems * RVV_CYC_PER_ELEM * passes
    return c, c, "rvv", "extrapolated"


def ms_at(cycles: float, mhz: float) -> float:
    return cycles / (mhz * 1e3)  # cycles / (MHz * 1e6) * 1e3 ms
