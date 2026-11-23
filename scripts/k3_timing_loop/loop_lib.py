#!/usr/bin/env python3
"""Read walker runs that carry the two extra stamps (got_cores, kernel_end) and cut each layer into pieces.

  ready    the latest of: ends of its inputs, end of the previous entry of the same worker, end of
           its ordering-edge entry; for a frame's first layer its start gate (planned start)
  gap      start - ready       (frame-first layers that waited for their gate are left out)
  grab     got_cores - start   (waiting for a core another layer holds; in neither gap nor factor)
  run      end - got_cores     (compared with the planned duration; factor = run / planned)

A key is (network, layer IR id, core kind, kernel type, number of cores); a frame is (network, instance);
the entry a layer waited for last can belong to another model or to an earlier frame.

The join trace <-> dispatch table <-> placement <-> IR id is walker_run.py's.

Environment:
  K3TL_NETS          comma list of the workload's models (default mlp_control,dronet,yolov8_nano)
  K3TL_MODELBLASTER  the ModelBlaster checkout (default <this repo>/zephyr-chipyard-sw/modelblaster); its
                     cores/spacemit_k3.json is the registry for the placement check
  K3TL_GOLDEN_C_DIR  goldens made with C-style rounding (golden_c.sh), one folder per network
                     (default: <folder that holds the raw run folder>/_golden_check/c_float)
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../.."))          # the XPU-RT checkout this file is in
sys.path.insert(0, HERE)
import walker_run as L  # noqa: E402
from walker_run import check_output, output_block  # noqa: E402

#: the models of the workload. Default: the three of the mix. K3TL_NETS=yolov8_nano runs the same scripts on one model.
NETS = tuple(x for x in os.environ.get("K3TL_NETS", "mlp_control,dronet,yolov8_nano").split(",") if x)
MODELBLASTER = os.path.abspath(os.environ.get("K3TL_MODELBLASTER") or os.path.join(REPO, "zephyr-chipyard-sw/modelblaster"))
NS_PER_MS = 1e6
SCHEDULE_TOL_MS = 2e-4     # C float start_time_ms: spacing 1.2e-4 ms between 1024 and 2048 ms (dronet's horizon)
REGISTRY = os.path.join(MODELBLASTER, "cores/spacemit_k3.json")
CHECKER = os.path.join(REPO, "scripts/check_k3_placement.py")
BACKENDS = "x100=ime_k3_x100,a100=ime_k3_a100"
IMPL = "x100:rvv=rvv_k3,x100:ime=ime_k3_x100,a100:rvv=rvv_k3_a100,a100:ime=ime_k3_a100"


def golden_c_dir(raw: str) -> str:
    """Goldens made with C-style rounding (golden_c.sh), one folder per network."""
    return os.environ.get("K3TL_GOLDEN_C_DIR") or os.path.join(os.path.dirname(os.path.abspath(raw)), "_golden_check", "c_float")


#: (network, layer IR id, core kind, kernel type, number of cores)
Key = Tuple[str, int, str, str, int]
KIND_NAME = {"x100": "X100", "a100": "A100"}


@dataclass(frozen=True)
class Piece:
    rep: int
    net: str
    frame: int
    eid: int
    irid: int
    name: str
    op: str
    kind: str
    impl: str
    width: int
    planned_start: float
    planned: float
    ready: float
    start: float
    got: Optional[float]
    kend: Optional[float]
    end: float
    binding: str
    binding_eid: int
    handoff: str          # "" for a gated frame-first layer

    @property
    def key(self) -> Key:
        return (self.net, self.irid, self.kind, self.impl, self.width)

    @property
    def gated(self) -> bool:
        return self.binding == "gate"

    @property
    def gap(self) -> float:
        return self.start - self.ready

    @property
    def grab(self) -> float:
        return self.got - self.start

    @property
    def run(self) -> float:
        return self.end - self.got

    @property
    def kernel(self) -> float:
        return self.kend - self.got

    @property
    def release(self) -> float:
        return self.end - self.kend

    @property
    def planned_end(self) -> float:
        return self.planned_start + self.planned

    @property
    def finish_error(self) -> float:
        return self.end - self.planned_end


@dataclass(frozen=True)
class FrameTotals:
    rep: int
    net: str
    frame: int
    planned_ms: float        # planned end of the last layer - release (instance x period)
    actual_ms: float         # actual end of the last layer - release
    deadline_ms: float       # window
    gate_offset_ms: float    # planned start of the frame's first layer - release (the walker holds the frame until then)
    layers: int
    sum_gap: float
    sum_grab: float
    sum_run_extra: float
    handoffs: int            # layers whose last wait was for an entry on another worker thread
    handoffs_other_model: int
    # exact split of (actual - planned) along the chain that ended the frame:
    #   chain_gap + chain_grab + chain_run_extra - chain_planned_idle + chain_inherited
    chain_gap: float
    chain_grab: float
    chain_run_extra: float
    chain_planned_idle: float
    chain_inherited: float   # how late the entry of ANOTHER model / an earlier frame, where the chain left this frame, finished
    chain_left_to: str       # "" (reached the frame's own start), "other model" or "earlier frame"
    chain_len: int
    residual: float

    @property
    def extra(self) -> float:
        return self.actual_ms - self.planned_ms

    @property
    def planned_run_ms(self) -> float:
        """Planned frame time: planned end of the last layer - planned start of the first."""
        return self.planned_ms - self.gate_offset_ms

    @property
    def actual_run_ms(self) -> float:
        """Board frame time: end of the last layer - planned start of the first (its start gate)."""
        return self.actual_ms - self.gate_offset_ms

    @property
    def late(self) -> bool:
        return self.actual_ms > self.deadline_ms

    @property
    def lateness(self) -> float:
        return self.actual_ms - self.deadline_ms

    @property
    def planned_late(self) -> bool:
        return self.planned_ms > self.deadline_ms + 1e-6


@dataclass(frozen=True)
class RunData:
    rep: int
    run: "L.Run"
    pieces: Tuple[Piece, ...]
    has_stamps: bool
    checks: dict


def quantile(xs: Sequence[float], q: float) -> float:
    v = sorted(xs)
    if not v:
        return float("nan")
    pos = q * (len(v) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def handoff_type(e, b) -> str:
    """Where the entry `e` waited for last (`b`) ran: same worker thread, or another one (by core kinds);
    with a note when it belongs to another model."""
    t = "same thread" if (b.kind, b.hart) == (e.kind, e.hart) else f"{KIND_NAME[b.kind]} -> {KIND_NAME[e.kind]}"
    return t + (", other model" if b.net != e.net else "")


def output_checks(stdout_path: str, raw: str) -> dict:
    """Per network: the board's output block against the build's own golden and against the
    C-rounding golden (used only if that golden was made from a byte-identical graph)."""
    text = open(stdout_path, errors="replace").read()
    out = {}
    for net in NETS:
        block = output_block(text, net)
        d_ok, d_why = check_output(block, np.load(f"{raw}/golden/{net}.io.npz")["output"])
        c_ok, c_why = False, "no C-rounding golden made from this graph"
        c_npz, c_graph = os.path.join(golden_c_dir(raw), net, "io.npz"), os.path.join(golden_c_dir(raw), net, "graph.json")
        if os.path.exists(c_npz) and os.path.exists(c_graph) and md5(c_graph) == md5(f"{raw}/ir/{net}.graph.json"):
            c_ok, c_why = check_output(block, np.load(c_npz)["output"])
        have_c = os.path.exists(c_npz) and os.path.exists(c_graph) and md5(c_graph) == md5(f"{raw}/ir/{net}.graph.json")
        # the pass/fail of a run: the C-rounding golden when one exists for this graph; the build's own golden
        # otherwise
        out[net] = dict(ok=bool(c_ok) if have_c else bool(d_ok), used=("C-rounding golden" if have_c else "build golden"),
                        c_ok=bool(c_ok), detail=c_why, default_ok=bool(d_ok), default_detail=d_why)
    first = text.splitlines()[0] if text else ""
    return dict(nets=out, fatal="FATAL" in text, policy_line=first, ok=all(v["ok"] for v in out.values()),
                default_ok=all(v["default_ok"] for v in out.values()))


def placement_check(placement_csv: str) -> str:
    r = subprocess.run([sys.executable, CHECKER, "--placement", placement_csv, "--registry", REGISTRY,
                        "--backends", BACKENDS, "--impl-backends", IMPL], capture_output=True, text=True)
    lines = (r.stdout or r.stderr).strip().splitlines()
    return lines[-1] if lines else "no output"


def load(raw: str, rep: int, checks: bool = True) -> RunData:
    """One run from a raw folder written by job.sh (a run of a walker without the stamps also loads: no grab)."""
    trace = f"{raw}/trace_r{rep}.csv"
    run = L.load_run(f"{os.path.basename(raw)}_r{rep}", trace, f"{raw}/placement_r{rep}.csv", f"{raw}/table.c",
                     [f"{n}={raw}/ir/{n}.graph.json" for n in NETS], f"{raw}/config.json", f"{raw}/schedule.json")
    with open(trace, newline="") as f:
        rows = {int(r["entry_id"]): r for r in csv.DictReader(f)}
    has = bool(rows) and "got_cores_cycles" in next(iter(rows.values()))
    E = run.entries
    pieces: List[Piece] = []
    bad_order = 0
    for eid in sorted(E):
        e = E[eid]
        if not e.real:
            continue
        got = kend = None
        if has:
            got = int(rows[eid]["got_cores_cycles"]) / NS_PER_MS
            kend = int(rows[eid]["kernel_end_cycles"]) / NS_PER_MS
            bad_order += not (e.rs <= got <= kend <= e.ae)
        hand = "" if e.binding in ("gate", "start") or e.binding_eid < 0 else handoff_type(e, E[e.binding_eid])
        pieces.append(Piece(rep=rep, net=e.net, frame=e.inst, eid=eid, irid=e.irid, name=e.name, op=e.op, kind=e.kind,
                            impl=e.impl, width=len(e.harts), planned_start=e.ps, planned=e.pdur, ready=e.ready,
                            start=e.rs, got=got, kend=kend, end=e.ae, binding=e.binding, binding_eid=e.binding_eid,
                            handoff=hand))
    info: dict = dict(audit=run.audit, stamps=has, stamps_out_of_order=bad_order, layers=len(pieces))
    if checks:
        info["output"] = output_checks(f"{raw}/stdout_r{rep}.txt", raw)
        info["placement"] = placement_check(f"{raw}/placement_r{rep}.csv")
        rc_path = f"{raw}/stdout_r{rep}.rc"
        info["rc"] = open(rc_path).read().strip() if os.path.exists(rc_path) else "0 (runner)"
        au = run.audit
        info["ok"] = bool(
            info["output"]["ok"] and not info["output"]["fatal"] and info["placement"].startswith("PASS")
            and info["rc"].startswith("0") and bad_order == 0 and au["name_mismatch"] == 0 and au["op_mismatch"] == 0
            and au["unmapped"] == 0 and au["table_mismatch"] == 0 and au.get("schedule_missing", 0) == 0
            and au.get("schedule_max_diff_ms", 0.0) < SCHEDULE_TOL_MS)
    return RunData(rep=rep, run=run, pieces=tuple(pieces), has_stamps=has, checks=info)


def frame_totals(rd: RunData) -> List[FrameTotals]:
    """Per (network, frame). Frame time is measured from the frame's release (instance x period).
    Needs the stamps for the chain split; without them the grab/run parts are folded into run."""
    E = rd.run.entries
    by_eid = {p.eid: p for p in rd.pieces}
    groups: Dict[Tuple[str, int], List[Piece]] = {}
    for p in rd.pieces:
        groups.setdefault((p.net, p.frame), []).append(p)
    out: List[FrameTotals] = []
    for (net, fr), ps in sorted(groups.items()):
        f = rd.run.frames[(net, fr)]
        release, window = f.release, f.deadline - f.release
        last = max(ps, key=lambda p: p.end)
        planned = max(p.planned_end for p in ps) - release
        actual = last.end - release
        cg = cb = cr = work = 0.0
        n, left = 0, ""
        origin_act = origin_pln = None
        eid = last.eid
        while eid >= 0:
            e = E[eid]
            if (e.net, e.inst) != (net, fr):            # the chain left the frame
                origin_act, origin_pln = e.done, e.pe
                left = "other model" if e.net != net else "earlier frame"
                break
            p = by_eid.get(eid)
            if p is not None:
                n += 1
                work += p.planned
                if p.got is not None:
                    cb += p.grab
                    cr += p.run - p.planned
                else:
                    cr += (p.end - p.start) - p.planned
                if p.gated:
                    cg += p.start - p.ready
                    origin_act = origin_pln = p.ready
                    break
                cg += p.gap
            if e.binding in ("gate", "start") or e.binding_eid < 0:
                origin_act = origin_pln = e.ready
                break
            eid = e.binding_eid
        if origin_act is None:
            origin_act = origin_pln = release
        idle = planned - work - (origin_pln - release)
        inherited = origin_act - origin_pln
        hand = [p for p in ps if p.handoff and not p.handoff.startswith("same thread")]
        out.append(FrameTotals(
            rep=rd.rep, net=net, frame=fr, planned_ms=planned, actual_ms=actual, deadline_ms=window,
            gate_offset_ms=min(p.planned_start for p in ps) - release, layers=len(ps),
            sum_gap=sum(p.gap for p in ps if not p.gated),
            sum_grab=sum(p.grab for p in ps) if rd.has_stamps else float("nan"),
            sum_run_extra=sum((p.run if rd.has_stamps else p.end - p.start) - p.planned for p in ps),
            handoffs=len(hand), handoffs_other_model=sum(1 for p in hand if p.handoff.endswith("other model")),
            chain_gap=cg, chain_grab=cb, chain_run_extra=cr, chain_planned_idle=idle, chain_inherited=inherited,
            chain_left_to=left, chain_len=n, residual=(actual - planned) - (cg + cb + cr - idle + inherited)))
    return out


# ---------------------------------------------------------------- the profile tree
BACKEND_DIR = {("x100", "rvv"): ("rvv_k3_x100", "spacemit_x100"), ("x100", "ime"): ("ime_k3_x100", "spacemit_x100"),
               ("a100", "rvv"): ("rvv_k3_a100", "spacemit_a100"), ("a100", "ime"): ("ime_k3_a100", "spacemit_a100")}
WIDTH_TOPO = {1: "topo_0", 2: "topo_0_1", 4: "topo_0_1_2_3", 8: "topo_0_1_2_3_4_5_6_7"}
SENTINEL_MS = 1e6
#: bookkeeping files apply.py writes into a corrected tree (names kept from the experiment that produced the first
#: trees, so that those trees stay readable and a rebuilt tree is byte-identical to a recorded one)
TREE_ENTRIES, TREE_REPORT = "t023_tree_entries.csv", "t023_tree_report.json"


def profile_csv(root: str, model: str, kind: str, impl: str, width: int) -> str:
    hw, target = BACKEND_DIR[(kind, impl)]
    return os.path.join(root, "profile", hw, target, model, f"{model}.int8",
                        f"{model}_{target}_{hw}_{model}.int8", WIDTH_TOPO[width], "results.csv")


def read_profile(root: str) -> Dict[Key, dict]:
    """Every profile entry of the workload's models."""
    out: Dict[Key, dict] = {}
    for model in NETS:
        for (kind, impl) in BACKEND_DIR:
            for width in WIDTH_TOPO:
                path = profile_csv(root, model, kind, impl, width)
                if not os.path.exists(path):
                    continue
                with open(path, newline="") as f:
                    for row in csv.DictReader(f):
                        if (row.get("mean_unit") or "ms") != "ms":
                            raise ValueError(f"{path}: unit {row.get('mean_unit')!r}")
                        out[(model, int(row["dispatch_id"]), kind, impl, width)] = dict(
                            ms=float(row["mean_time"]), op=row.get("op", ""), path=path)
    return out


def group_by_key(pieces: Iterable[Piece]) -> Dict[Key, List[Piece]]:
    out: Dict[Key, List[Piece]] = {}
    for p in pieces:
        out.setdefault(p.key, []).append(p)
    return out


def med(xs) -> float:
    xs = list(xs)
    return statistics.median(xs) if xs else float("nan")
