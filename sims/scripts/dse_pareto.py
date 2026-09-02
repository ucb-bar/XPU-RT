"""Model DSE — quality-vs-cost Pareto explorer for the nav model zoo (task #26 core).

Consolidates every nav model we have into one design-space table: compute cost
(params, MACs, CPU latency) on one axis, task quality (closed-loop flight and/or
offline accuracy) on the other, and marks the Pareto frontier — the models where
you can't get more quality without paying more compute. This is the co-design
instrument: it says which model to ship for a given latency budget (the 10 Hz /
100 MHz SoC target) and is the scaffold that measured HW latency (once the
profiling data is trustworthy, #24) and int8/QAT variants (#25) slot into.

MACs are hook-counted from a real forward pass (zero-skip aware for the fused
model). Latency here is x86 CPU wall-clock — a RELATIVE proxy; the absolute SoC
numbers come from the ModelBlaster/FireSim profiles once #24 lands. Quality is
read from the closed-loop eval JSONs + the offline held-out numbers.

    <env_isaaclab py> sims/scripts/dse_pareto.py

Writes dse_pareto.json + dse_pareto.png to the scratchpad.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

# .../XPU-RT/sims/scripts/dse_pareto.py → REPO = XPU-RT
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for hw_cycle_model
sys.path.insert(0, os.path.abspath(os.path.join(REPO, "..", "vitfly", "models")))
import torch.nn as nn  # noqa: E402
from ablate_fused_compute import MacCounter  # noqa: E402  (hook-based MAC counter)
import hw_cycle_model as HW  # noqa: E402  (measured-FireSim-grounded per-op cycle model)


class FloatOpCounter:
    """Records the float ops MacCounter skips (LayerNorm / Softmax / GELU / sigmoid /
    tanh) with their output element counts — the FP16-on-RVV cost the cycle model needs.
    These are the ops Gemmini cannot touch (int8 GEMM only)."""

    _KIND = {nn.LayerNorm: "layernorm", nn.Softmax: "softmax", nn.GELU: "gelu",
             nn.Sigmoid: "sigmoid", nn.Tanh: "tanh"}

    def __init__(self, model):
        self.model = model
        self.ops = []
        self._handles = []

    def _hook(self, mod, inp, out):
        kind = self._KIND.get(type(mod))
        if kind is None:
            return
        n = 1
        for d in out.shape:
            n *= int(d)
        self.ops.append({"kind": kind, "elems": n})

    def __enter__(self):
        self.ops = []
        for m in self.model.modules():
            if type(m) in self._KIND:
                self._handles.append(m.register_forward_hook(self._hook))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []

_SCRATCH = ("/tmp/claude-2621/-scratch-agustin-projects-DIMA/"
            "057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad")

# Per-MAC energy ESTIMATE (pJ/MAC), rough order-of-magnitude at a ~16-28nm edge node.
# LABELED ESTIMATE — real energy needs the ModelBlaster/Accelergy path (audit #24, P3: energy
# does not exist anywhere in the profiling data). int8 MACs are ~6-8x cheaper than fp32.
PJ_PER_MAC = {"fp32": 1.5, "fp16": 0.55, "int8": 0.23}
# int8 latency speedup ESTIMATE on an int8-capable datapath (Gemmini Q0.31). Real number needs
# measured SoC profiles (audit #24, P0/P1 — the .fp32 profiles under int8 backends were mislabeled).
INT8_LAT_SPEEDUP = 3.5


# ---- build each model + a representative input; return (module, input, forward_fn) ----
def _dronet(size, head):
    from qnn_models.dronet import DronetTorch
    s = 112 if size == "small" else 224
    m = DronetTorch(img_dims=(s, s), img_channels=1, output_dim=3 if head == "classifier" else 1,
                    small=(size == "small"), head=head).eval()
    x = torch.rand(1, 1, s, s)
    return m, lambda: m(x)


def _vit(head):
    from trail_vit import TrailViT
    m = TrailViT(head=head).eval()
    x = torch.rand(1, 1, 112, 112)
    return m, lambda: m(x)


def _fused(mask=None, vision_encoder="vit"):
    from fused_model import FusedSensorNet
    m = FusedSensorNet(out_dim=2, vision_encoder=vision_encoder).eval()
    B = 1
    inp = {"front_grey": torch.rand(B, 1, 60, 90), "tof_cross": torch.rand(B, 4, 8, 8),
           "optical_flow": torch.randn(B, 2), "down_tof": torch.rand(B, 1), "baro": torch.randn(B, 2),
           "quat": torch.randn(B, 4), "body_rates": torch.randn(B, 3), "desired_vel": torch.randn(B, 3),
           "flags": torch.ones(B, 6)}
    return m, lambda: m(inp, mask=mask)


def measure(build_fn, iters=30):
    m, fwd = build_fn()
    params = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        with MacCounter(m, record=True) as mc, FloatOpCounter(m) as fc:
            fwd()
        macs, mac_ops, float_ops = mc.total, list(mc.ops), list(fc.ops)
        for _ in range(3):
            fwd()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); fwd(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"params_M": round(params / 1e6, 3), "macs_M": round(macs / 1e6, 2),
            "latency_ms": round(ts[len(ts) // 2], 3),
            "_mac_ops": mac_ops, "_float_ops": float_ops}


# ---- HW cycle model: sum measured/estimated per-op cycles → SoC latency at 60/100 MHz ----
_MEASURED_OP, _MEASURED_MODEL = HW.load_measured()
_GEMMINI_EFF = HW.gemmini_gemm_eff(_MEASURED_OP)  # measured int8-GEMM MAC/cyc (realistic anchor)
NAV_BUDGET_MS = 100.0   # 10 Hz local-planner budget (100 Hz flight loop is the low-level ctrl, not us)


def hw_latency(cost):
    """Sum per-op cycles for one model → optimistic/realistic cycles, ms at 60/100 MHz,
    source mix, and the 10 Hz PASS/FAIL verdict. Optimistic = Loren 35 MAC/cyc peak int8;
    realistic = measured DroNet blended efficiency (bakes in real overhead)."""
    opt = real = 0.0
    src = {"firesim": 0, "loren_est": 0, "extrapolated": 0}
    bk = {}
    for op in cost["_mac_ops"]:
        o, r, backend, source = HW.op_cycles(op, _MEASURED_OP, _GEMMINI_EFF)
        opt += o; real += r
        src[source] = src.get(source, 0) + 1
        bk[backend] = bk.get(backend, 0.0) + r
    for fop in cost["_float_ops"]:
        o, r, backend, source = HW.fp16_op_cycles(fop["kind"], fop["elems"])
        opt += o; real += r
        src["extrapolated"] += 1
        bk[backend] = bk.get(backend, 0.0) + r
    gemm = bk.get("gemmini", 0.0) + bk.get("gemmini+rvv", 0.0)
    rvv = bk.get("rvv", 0.0)
    return {
        "cyc_opt_M": round(opt / 1e6, 2), "cyc_real_M": round(real / 1e6, 2),
        "ms60_opt": round(HW.ms_at(opt, 60), 1), "ms60_real": round(HW.ms_at(real, 60), 1),
        "ms100_opt": round(HW.ms_at(opt, 100), 1), "ms100_real": round(HW.ms_at(real, 100), 1),
        # breakdown (realistic, @60 MHz): int8 GEMM/conv on Gemmini vs float on RVV
        "gemm_ms60_real": round(HW.ms_at(gemm, 60), 1), "rvv_ms60_real": round(HW.ms_at(rvv, 60), 1),
        "fits_10hz_60MHz": HW.ms_at(real, 60) <= NAV_BUDGET_MS,
        "fits_10hz_100MHz": HW.ms_at(real, 100) <= NAV_BUDGET_MS,
        "source_mix": src,
        "rvv_frac_real": round(rvv / real, 3) if real else 0.0,
    }


# ---- quality: closed-loop flight (mean lateral offset, lower=better → invert to a score)
#      and offline held-out accuracy/sign-agree. Filled from the eval JSONs where present. ----
def _load_offset(name):
    """Return mean closed-loop offset (m) averaged over straight+curved if the JSONs exist."""
    files = {
        "dronet-cls": ["forestnav_dronet_cls_straight.json", "forestnav_dronet_cls_curved.json"],
        "dronet-reg": ["forestnav_dronet_reg_straight.json", "forestnav_dronet_reg_curved.json"],
        "vit-cls": ["forestnav_vit_cls_straight.json", "forestnav_vit_cls_curved.json"],
        "vit-reg": ["forestnav_vit_reg_straight.json", "forestnav_vit_reg_curved.json"],
        # fused goal-conditioned GATE nav (the headline model): Stage-1 mapped goal.
        "fused-gate (mapped goal)": ["fused_gate_eval.json"],
        # Stage-2 vision goal (desired_vel masked → gate inferred from camera, no YOLO). Same cost.
        "fused-gate (vision goal)": ["fused_gate_vision_eval.json"],
    }.get(name)
    if not files:
        return None
    offs = []
    for f in files:
        p = os.path.join(_SCRATCH, f)
        if os.path.exists(p):
            offs.append(json.load(open(p))["agg"]["mean_offset"])
    return round(sum(offs) / len(offs), 3) if offs else None


# Model zoo: name -> (builder, offline quality). Closed-loop offset loaded from JSONs.
ZOO = [
    ("dronet-cls", lambda: _dronet("small", "classifier"), {"offline_acc": 0.543, "task": "trail"}),
    ("dronet-reg", lambda: _dronet("small", "regression"), {"offline_signagree": 0.756, "task": "trail"}),
    ("dronet-large-cls", lambda: _dronet("large", "classifier"), {"offline_acc": 0.586, "task": "trail"}),
    ("vit-cls", lambda: _vit("classifier"), {"offline_acc": 0.615, "task": "trail"}),
    ("vit-reg", lambda: _vit("regression"), {"offline_signagree": 0.759, "task": "trail"}),
    ("fused-gate (mapped goal)", lambda: _fused(None), {"task": "gate", "note": "Stage-1 goal from map"}),
    ("fused-gate (vision goal)", lambda: _fused(None), {"task": "gate", "note": "Stage-2 gate from camera, no YOLO"}),
    ("fused-gate (CNN vision)", lambda: _fused(None, vision_encoder="cnn"),
     {"task": "gate", "note": "CNN vision stem — no attention/softmax/LN, Gemmini-friendly"}),
    ("fused (cam-off zero-skip)", lambda: _fused({"front_grey": False}),
     {"task": "gate", "note": "ToF+state only, 32x cheaper — FAILS gate nav 0% (#62: camera essential)"}),
]


def pareto_front(points):
    """points: list of (cost, quality_score, name). Higher quality, lower cost = better.
    Return the set of non-dominated names (maximize quality, minimize cost)."""
    front = []
    for c, q, n in points:
        dominated = any((c2 <= c and q2 >= q and (c2 < c or q2 > q)) for c2, q2, _ in points)
        if not dominated:
            front.append(n)
    return set(front)


def main():
    rows = []
    for name, build, qual in ZOO:
        cost = measure(build)
        offset = _load_offset(name)
        # quality score: closed-loop → 1/(1+offset) (higher=better); else offline metric; else NaN
        if offset is not None:
            score = round(1.0 / (1.0 + offset), 3); qmetric = f"flight_offset={offset}m"
        elif "offline_acc" in qual:
            score = qual["offline_acc"]; qmetric = f"acc={qual['offline_acc']}"
        elif "offline_signagree" in qual:
            score = qual["offline_signagree"]; qmetric = f"sign_agree={qual['offline_signagree']}"
        else:
            score = None; qmetric = qual.get("note", "pending")
        # energy ESTIMATE (µJ/inference) at fp32 and int8; int8 latency ESTIMATE.
        e_fp32 = round(cost["macs_M"] * 1e6 * PJ_PER_MAC["fp32"] / 1e6, 2)   # pJ→µJ
        e_int8 = round(cost["macs_M"] * 1e6 * PJ_PER_MAC["int8"] / 1e6, 2)
        lat_int8 = round(cost["latency_ms"] / INT8_LAT_SPEEDUP, 3)
        hw = hw_latency(cost)
        cost.pop("_mac_ops", None); cost.pop("_float_ops", None)  # bulky, not for JSON
        rows.append({"name": name, **cost, "quality_score": score, "quality": qmetric,
                     "task": qual["task"],
                     "energy_fp32_uJ_est": e_fp32, "energy_int8_uJ_est": e_int8,
                     "latency_int8_ms_est": lat_int8, **hw})

    # Pareto over rows with a numeric quality score (cost = MACs).
    scored = [(r["macs_M"], r["quality_score"], r["name"]) for r in rows if r["quality_score"] is not None]
    front = pareto_front(scored)
    for r in rows:
        r["pareto"] = r["name"] in front

    print(f"\n=== MODEL DSE — quality vs SoC latency (nav zoo) ===")
    print(f"  Target: 1-hart Rocket + Saturn RVV-128 FP16 + 32x32 int8 Gemmini(Q0.31), ~60 MHz.")
    print(f"  10 Hz local-planner budget = {NAV_BUDGET_MS:.0f} ms.  Realistic anchor = measured "
          f"DroNet blended {_GEMMINI_EFF:.2f} MAC/cyc on gemmini_q31.")
    print(f"  [SoC latency = per-op cycles: measured FireSim where shape-matched, else "
          f"Loren-peak(opt)/measured-eff(real) extrapolation.]")
    print(f"{'model':28s} {'MACs(M)':>8s} {'ms@60 opt–real':>15s} {'GEMM|RVV @60':>13s} "
          f"{'ms@100':>7s} {'10Hz 60/100':>12s} {'quality':>20s} {'Par':>4s}")
    for r in sorted(rows, key=lambda r: r["macs_M"]):
        star = " ★" if r["pareto"] else ""
        v = ("Y" if r["fits_10hz_60MHz"] else "N") + "/" + ("Y" if r["fits_10hz_100MHz"] else "N")
        print(f"{r['name']:28s} {r['macs_M']:>8.2f} "
              f"{r['ms60_opt']:>7.1f}–{r['ms60_real']:<7.1f} "
              f"{r['gemm_ms60_real']:>5.0f}|{r['rvv_ms60_real']:<5.0f} {r['ms100_real']:>7.1f} "
              f"{v:>12s} {r['quality']:>20s}{star}")
    print("  ms@60 opt–real = SoC latency range (Loren-peak 35 MAC/cyc … measured-eff).  "
          "GEMM|RVV = realistic @60 split: int8 conv/GEMM on Gemmini | float softmax/LN/GELU/LSTM on RVV.")
    print("  10Hz 60/100 = fits 100 ms budget at 60 / 100 MHz (realistic).")
    print("  NOTE: modeled CNN latency counts conv/linear/LSTM only (no BN/pool/add) — it is a "
          "GEMM lower bound; the MEASURED dronet anchor below (BN/pool/add included) is authoritative.")

    print(f"\n=== measured FireSim anchors (1 cyc = 1 ns; nonfused=source firesim, fused=sum-constituents) ===")
    for (backend, model, variant), cyc in sorted(_MEASURED_MODEL.items()):
        print(f"  {backend:14s} {model+'.'+variant:22s} {cyc/1e6:>8.2f} Mcyc  "
              f"({HW.ms_at(cyc,60):>8.1f} ms@60MHz, {HW.ms_at(cyc,100):>8.1f} ms@100MHz)")

    out = os.path.join(_SCRATCH, "dse_pareto.json")
    json.dump({"rows": rows, "pareto_front": sorted(front),
               "notes": "latency=x86 CPU proxy; measured SoC latency pending #24; int8/QAT pending #25"},
              open(out, "w"), indent=2)
    print(f"\n[dse] wrote {out}")

    # plot (matplotlib Agg — headless safe)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in rows:
            if r["quality_score"] is None:
                continue
            col = "#d1495b" if r["pareto"] else "#5b7db1"
            ax.scatter(r["macs_M"], r["quality_score"], c=col, s=70, zorder=3)
            ax.annotate(r["name"], (r["macs_M"], r["quality_score"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        pts = sorted([(r["macs_M"], r["quality_score"]) for r in rows if r["pareto"]])
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "--", c="#d1495b", alpha=0.6, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("MACs (M) — compute cost  [log]"); ax.set_ylabel("quality score (higher=better)")
        ax.set_title("Model DSE: nav zoo quality vs cost (★ = Pareto frontier)")
        ax.grid(alpha=0.3, zorder=0)
        png = os.path.join(_SCRATCH, "dse_pareto.png")
        fig.tight_layout(); fig.savefig(png, dpi=120)
        print(f"[dse] wrote {png}")
    except Exception as e:
        print(f"[dse] plot skipped: {e}")


if __name__ == "__main__":
    main()
