"""Helpers for auditing the QRB5165 cost table for inconsistencies.

The helpers here parse the canonical ``qrb5165_costs.json`` key shape, run
basic-sanity / ordering / outlier checks, and (when shape-parameterized
families are present) build a Z3 expression suitable for
``prove_cost_monotonicity``.

The module is intentionally read-only — it never mutates the cost table.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# Key form examples:
#   Conv2d@1x320x320x3->1x160x160x16,g1,k3,s2@uint8::HTA::0
#   elementwise@3x320x320+3x322x322->3x322x322@int8::CPU::0
#   conv@?+?+?+?+?+?->32x6400@fp32::CPU::0
#   dispatch::dispatch_24::CPU::0  (no shape, used for infeasible markers)
_KEY_RE = re.compile(r"^(?P<op>[^@:]+)@(?P<shape>.+?)@(?P<dtype>[^@:]+)::(?P<backend>[^:]+)::(?P<dev>\d+)$")
_DISPATCH_RE = re.compile(r"^dispatch::(?P<name>[^:]+)::(?P<backend>[^:]+)::(?P<dev>\d+)$")


@dataclass(frozen=True)
class CostEntry:
    """Parsed cost-table entry."""

    key: str
    op: str
    shape: str | None
    dtype: str | None
    backend: str
    device: str
    mean_us: float | None
    infeasible: bool
    reason: str | None


@dataclass(frozen=True)
class Finding:
    """A single audit finding written to ``results.jsonl``."""

    check_kind: str
    severity: str
    op_or_family: str
    backends: list[str]
    detail: dict[str, Any]


def parse_entry(key: str, value: dict[str, Any]) -> CostEntry | None:
    """Parse a cost-table key/value into a :class:`CostEntry`.

    Returns ``None`` if the key does not match any recognized shape; the
    caller should record that as a separate audit finding.
    """

    m = _KEY_RE.match(key)
    if m is not None:
        return CostEntry(
            key=key,
            op=m.group("op"),
            shape=m.group("shape"),
            dtype=m.group("dtype"),
            backend=m.group("backend"),
            device=m.group("dev"),
            mean_us=value.get("mean_us"),
            infeasible=bool(value.get("infeasible", False)),
            reason=value.get("reason"),
        )
    m = _DISPATCH_RE.match(key)
    if m is not None:
        return CostEntry(
            key=key,
            op=f"dispatch::{m.group('name')}",
            shape=None,
            dtype=None,
            backend=m.group("backend"),
            device=m.group("dev"),
            mean_us=value.get("mean_us"),
            infeasible=bool(value.get("infeasible", False)),
            reason=value.get("reason"),
        )
    return None


def parse_table(table: dict[str, Any]) -> tuple[list[CostEntry], list[str]]:
    """Parse the ``execute`` map. Returns (entries, unparseable_keys)."""

    entries: list[CostEntry] = []
    unparseable: list[str] = []
    for key, value in table.items():
        parsed = parse_entry(key, value)
        if parsed is None:
            unparseable.append(key)
        else:
            entries.append(parsed)
    return entries, unparseable


def check_basic_sanity(entries: list[CostEntry]) -> list[Finding]:
    """Flag non-finite, negative, or zero costs and duplicate (op, backend, shape)."""

    findings: list[Finding] = []
    seen: dict[tuple[str, str, str | None, str | None], CostEntry] = {}
    for e in entries:
        c = e.mean_us
        # Only inspect entries that *claim* to be measured (not infeasible).
        if not e.infeasible and c is not None:
            if math.isnan(c) or math.isinf(c):
                findings.append(
                    Finding(
                        check_kind="non_finite_cost",
                        severity="high",
                        op_or_family=e.op,
                        backends=[e.backend],
                        detail={"key": e.key, "mean_us": c},
                    )
                )
            elif c < 0:
                findings.append(
                    Finding(
                        check_kind="negative_cost",
                        severity="high",
                        op_or_family=e.op,
                        backends=[e.backend],
                        detail={"key": e.key, "mean_us": c},
                    )
                )
            elif c == 0:
                findings.append(
                    Finding(
                        check_kind="zero_cost",
                        severity="medium",
                        op_or_family=e.op,
                        backends=[e.backend],
                        detail={"key": e.key, "mean_us": c},
                    )
                )
        # Duplicates.
        sig = (e.op, e.backend, e.shape, e.dtype)
        prior = seen.get(sig)
        if prior is not None:
            findings.append(
                Finding(
                    check_kind="duplicate_entry",
                    severity="high",
                    op_or_family=e.op,
                    backends=[e.backend],
                    detail={"key_a": prior.key, "key_b": e.key},
                )
            )
        else:
            seen[sig] = e
    return findings


def check_backend_coverage(entries: list[CostEntry]) -> list[Finding]:
    """For ops measured on >=2 backends, flag the *missing* backends.

    We treat any op that has measurements on >=2 of {CPU, GPU, HTA, HTA2} as
    a candidate for full coverage; missing entries are recorded as low
    severity (the table is acknowledged not to be exhaustive).
    """

    findings: list[Finding] = []
    target_backends = {"CPU", "GPU", "HTA", "HTA2"}
    by_op: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        if e.mean_us is None or e.infeasible:
            continue
        by_op[e.op].add(e.backend)
    for op, observed in by_op.items():
        if len(observed) < 2:
            continue
        missing = sorted(target_backends - observed)
        if missing:
            findings.append(
                Finding(
                    check_kind="missing_backend_coverage",
                    severity="low",
                    op_or_family=op,
                    backends=sorted(observed),
                    detail={"missing": missing, "observed": sorted(observed)},
                )
            )
    return findings


def check_magnitude_outliers(entries: list[CostEntry], factor: float = 100.0) -> list[Finding]:
    """Flag costs > ``factor`` x the per-backend median.

    Mis-priced "unsupported" markers usually appear as a giant cost so the
    scheduler avoids them; we surface them so a human can decide whether
    they should be flagged ``infeasible`` instead.
    """

    findings: list[Finding] = []
    by_backend: dict[str, list[float]] = defaultdict(list)
    for e in entries:
        if e.mean_us is not None and not e.infeasible and e.mean_us > 0:
            by_backend[e.backend].append(e.mean_us)
    medians = {b: statistics.median(v) for b, v in by_backend.items() if v}
    for e in entries:
        if e.mean_us is None or e.infeasible or e.mean_us <= 0:
            continue
        med = medians.get(e.backend)
        if med is None or med <= 0:
            continue
        ratio = e.mean_us / med
        if ratio > factor:
            findings.append(
                Finding(
                    check_kind="magnitude_outlier",
                    severity="medium",
                    op_or_family=e.op,
                    backends=[e.backend],
                    detail={
                        "key": e.key,
                        "mean_us": e.mean_us,
                        "backend_median_us": med,
                        "ratio": ratio,
                    },
                )
            )
    return findings


def _ordering_index(entries: list[CostEntry]) -> dict[str, dict[str, float]]:
    """Build a ``backend -> op -> cost`` index for ordering checks.

    Only one entry per (op, backend) is kept (the cheapest measured one);
    this approximates "characteristic cost" so the ordering check is
    independent of shape variants. Ops without a measurement on the
    backend are absent.
    """

    out: dict[str, dict[str, float]] = defaultdict(dict)
    for e in entries:
        if e.mean_us is None or e.infeasible or e.mean_us <= 0:
            continue
        cur = out[e.backend].get(e.op)
        if cur is None or e.mean_us < cur:
            out[e.backend][e.op] = e.mean_us
    return out


def check_pairwise_ordering_stability(
    entries: list[CostEntry], sample_fraction: float = 1.0, seed: int = 0
) -> tuple[list[Finding], dict[str, Any]]:
    """For every backend pair, count op-pair ordering flips.

    Returns (findings, stats). Findings are aggregated per backend-pair
    (not per op-pair) since individual flips are not informative — we
    care about the *rate*. ``sample_fraction`` < 1.0 samples op-pairs.
    """

    import random

    rng = random.Random(seed)
    idx = _ordering_index(entries)
    backends = sorted(idx.keys())
    findings: list[Finding] = []
    stats: dict[str, Any] = {"backend_pairs": []}
    for i, b1 in enumerate(backends):
        for b2 in backends[i + 1 :]:
            common = sorted(set(idx[b1].keys()) & set(idx[b2].keys()))
            if len(common) < 2:
                continue
            pairs = [(a, b) for ai, a in enumerate(common) for b in common[ai + 1 :]]
            if sample_fraction < 1.0:
                k = max(1, int(len(pairs) * sample_fraction))
                pairs = rng.sample(pairs, k)
            flips = 0
            ties = 0
            for a, b in pairs:
                ca1, cb1 = idx[b1][a], idx[b1][b]
                ca2, cb2 = idx[b2][a], idx[b2][b]
                if ca1 == cb1 or ca2 == cb2:
                    ties += 1
                    continue
                if (ca1 < cb1) != (ca2 < cb2):
                    flips += 1
            counted = len(pairs) - ties
            rate = flips / counted if counted else 0.0
            pair_stat = {
                "backend_a": b1,
                "backend_b": b2,
                "common_ops": len(common),
                "pairs_examined": len(pairs),
                "flips": flips,
                "ties": ties,
                "flip_rate": rate,
            }
            stats["backend_pairs"].append(pair_stat)
            # Severity: > 30% flip rate is suspicious; > 50% means the
            # ordering between these backends is effectively random.
            if rate > 0.5 and counted >= 10:
                severity = "high"
            elif rate > 0.3 and counted >= 10:
                severity = "medium"
            else:
                severity = "low"
            findings.append(
                Finding(
                    check_kind="ordering_flip_rate",
                    severity=severity,
                    op_or_family=f"{b1}_vs_{b2}",
                    backends=[b1, b2],
                    detail=pair_stat,
                )
            )
    return findings, stats


# --------------------------- Shape-monotonicity ---------------------------

# Conv2d shape grammar (HTA / GPU keys):
#   <in_n>x<in_h>x<in_w>x<in_c>-><out_n>x<out_h>x<out_w>x<out_c>,gG,kK,sS
_CONV2D_SHAPE_RE = re.compile(
    r"^(?P<in_n>\d+)x(?P<in_h>\d+)x(?P<in_w>\d+)x(?P<in_c>\d+)"
    r"->(?P<out_n>\d+)x(?P<out_h>\d+)x(?P<out_w>\d+)x(?P<out_c>\d+),"
    r"g(?P<g>\d+),k(?P<k>\d+),s(?P<s>\d+)$"
)


@dataclass(frozen=True)
class ParsedConv2d:
    """Conv2d shape decomposed into the three Z3-tracked dimensions.

    For Conv2d cost we need three variables that *jointly* dominate FLOP
    count. Output H and W give us spatial extent, but channels matter
    via ``in_c * out_c`` (the dot-product accumulator). We therefore map
    Z3's (m, n, k) to (out_h, out_w, in_c * out_c) so that "shape_a
    dominates shape_b" implies dominated MAC count.
    """

    out_h: int
    out_w: int
    in_c: int
    out_c: int
    mean_us: float
    key: str

    @property
    def mac_channels(self) -> int:
        """Pseudo-dimension proportional to MAC channels (``in_c * out_c``)."""

        return self.in_c * self.out_c


def parse_conv2d_family(entries: list[CostEntry], backend: str, dtype: str) -> list[ParsedConv2d]:
    """Return parsed Conv2d entries for one (backend, dtype) slice."""

    out: list[ParsedConv2d] = []
    for e in entries:
        if e.op != "Conv2d" or e.backend != backend or e.dtype != dtype:
            continue
        if e.shape is None or e.mean_us is None or e.infeasible:
            continue
        m = _CONV2D_SHAPE_RE.match(e.shape)
        if not m:
            continue
        out.append(
            ParsedConv2d(
                out_h=int(m.group("out_h")),
                out_w=int(m.group("out_w")),
                in_c=int(m.group("in_c")),
                out_c=int(m.group("out_c")),
                mean_us=e.mean_us,
                key=e.key,
            )
        )
    return out


def build_conv2d_cost_expr(family: list[ParsedConv2d]) -> Callable[[Any, Any, Any], Any]:
    """Build a Z3 cost callable encoding the empirical Conv2d table.

    The returned function takes Z3 ``Int`` variables ``(m, n, k)`` mapped
    to ``(out_h, out_w, out_c)`` and returns a piecewise constant Z3
    expression: if the inputs equal a measured shape, the expression
    yields that shape's measured cost (rounded to integer microseconds);
    otherwise it is unconstrained-but-bounded by the family's max+1.

    The piecewise form lets ``prove_cost_monotonicity`` directly probe
    whether component-wise-dominated shapes have non-decreasing costs.
    """

    import z3

    # Round to integer microseconds: source mean_us already has > 1us noise.
    # Z3 vars map to (out_h, out_w, in_c * out_c) so larger MAC shapes
    # dominate smaller ones.
    rounded = [(p.out_h, p.out_w, p.mac_channels, int(round(p.mean_us)), p.key) for p in family]
    measured_keys = {(h, w, c) for h, w, c, _, _ in rounded}
    # Restrict Z3 to ONLY measured shapes: otherwise the un-measured
    # "default" cost we'd assign creates spurious counterexamples. We
    # express this by adding an extra "domain" constraint via z3.Or in
    # the returned callable wrapper.
    _ = measured_keys  # exposed via cost_expr.measured_keys below.
    default = max((c for *_, c, _ in rounded), default=1) + 1

    def cost_expr(m: Any, n: Any, k: Any) -> Any:
        expr: Any = z3.IntVal(default)
        for h, w, c, cost_val, _ in rounded:
            expr = z3.If(z3.And(m == h, n == w, k == c), z3.IntVal(cost_val), expr)
        return expr

    # Stash measured-shape coordinates on the callable so the caller can
    # add a domain restriction to the solver (Z3 will then only consider
    # measured shapes, eliminating the default-cost spurious flips).
    cost_expr.measured_shapes = tuple(measured_keys)  # type: ignore[attr-defined]
    return cost_expr


def shape_param_families(entries: list[CostEntry]) -> dict[tuple[str, str, str], int]:
    """Return ``(op, backend, dtype) -> count`` for shape-parameterized families."""

    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for e in entries:
        if e.shape is None or e.mean_us is None or e.infeasible:
            continue
        if e.dtype is None:
            continue
        counts[(e.op, e.backend, e.dtype)] += 1
    return {k: v for k, v in counts.items() if v >= 2}


def finding_to_dict(f: Finding) -> dict[str, Any]:
    """Serialize a Finding for JSONL output."""

    return {
        "check_kind": f.check_kind,
        "severity": f.severity,
        "op_or_family": f.op_or_family,
        "backends": f.backends,
        "detail": f.detail,
    }


# ----------------------- Per-op cost matrix audits -----------------------
#
# The helpers below operate on the simpler ``per-op profile matrix`` shape:
#
#     {workload_id: {op_id: {backend_name: latency_us, ...}}}
#
# produced by ``scripts/experiments/_parse_qnn_profile.py``. They are
# intentionally separate from the qrb5165-table helpers above because the
# matrix uses raw op IDs (no shape/dtype embedded in the key) and a wider
# set of backends (CPU/GPU/DSP).

# Op IDs in the matrix come in two flavors:
#   1. QNN-normalized (yolov8n): "convolution_0", "elementwiseneuron_5",
#      "elementwise_product_2", "strided_slice_1", "pad_3".
#   2. ONNX-path (dronet): "/conv_modules.0/Conv", "/Add_1", "/relu_modules.0/Relu",
#      "/Add_2_output_0.nchw", "/sigmoid1/Sigmoid".
#
# The two flavors are normalized to a common "op_family" name for cross-
# workload comparison via ``ONNX_TO_QNN_FAMILY``.

_QNN_NORM_RE = re.compile(r"^(?P<family>[a-zA-Z_]+?)_(?P<idx>\d+)$")

# ONNX-typename -> QNN family. None means "no QNN-side equivalent in the
# workloads we have"; the helper still groups the dronet entries under the
# ONNX family name so they show up in per-workload stats.
ONNX_TO_QNN_FAMILY: dict[str, str] = {
    "Conv": "convolution",
    "Relu": "elementwiseneuron",
    "Add": "elementwise_sum",
    "MaxPool": "pool",
    "Sigmoid": "elementwiseneuron",
    "Gemm": "fullyconnected",
}


def matrix_op_family(op_id: str) -> str:
    """Extract the op-family from a matrix op_id (both naming flavors).

    Examples:
        ``convolution_12`` -> ``convolution``
        ``elementwise_product_3`` -> ``elementwise_product``
        ``/conv_modules.5/Conv`` -> ``Conv``
        ``/Add_1`` -> ``Add``
        ``/Add_2_output_0.nchw`` -> ``Add`` (trailing tensor-name suffix dropped)
        ``Misc accelerator time`` -> ``Misc accelerator time`` (unchanged)
    """

    if op_id.startswith("/"):
        # ONNX-path style: take the last `/`-segment (or the bare name if it's
        # a top-level node like ``/Add_1``). Then strip the ``_<digit>`` suffix
        # and any ``_output_*`` / ``.nchw`` trailing tag.
        last = op_id.rsplit("/", 1)[-1]
        # Remove ``_output_...`` and ``.<anything>`` trailers.
        last = last.split("_output_", 1)[0]
        last = last.split(".", 1)[0]
        # Strip a trailing ``_<N>`` ordinal (Add_1 -> Add).
        m = re.match(r"^(?P<base>[A-Za-z][A-Za-z0-9]*)(?:_\d+)?$", last)
        if m:
            return m.group("base")
        return last
    m2 = _QNN_NORM_RE.match(op_id)
    if m2 is not None:
        return m2.group("family")
    return op_id


def canonical_family(family: str) -> str:
    """Map ONNX-typename families to their QNN equivalent (where known).

    Returns the input unchanged if no mapping is known — callers can then
    decide whether the entry is cross-workload-comparable.
    """

    return ONNX_TO_QNN_FAMILY.get(family, family)


@dataclass(frozen=True)
class MatrixOp:
    """One row of the per-op profile matrix.

    ``costs`` maps backend name (e.g. ``"CPU"``, ``"GPU"``, ``"DSP"``) to
    measured latency in microseconds. Missing backends mean "unsupported /
    not measured" — they are absent from the dict, not zero.
    """

    workload: str
    op_id: str
    family: str
    costs: dict[str, float]

    def has_all(self, backends: tuple[str, ...]) -> bool:
        return all(b in self.costs and self.costs[b] is not None for b in backends)


def load_cost_matrix(path: Any) -> dict[str, list[MatrixOp]]:
    """Load a ``qnn_cost_matrix.json`` file into per-workload op rows.

    The ``_meta`` workload key is skipped. Workloads with no ops are
    omitted from the returned dict.
    """

    import json
    from pathlib import Path

    p = Path(path)
    raw = json.loads(p.read_text())
    out: dict[str, list[MatrixOp]] = {}
    for wl, ops in raw.items():
        if wl == "_meta" or not isinstance(ops, dict):
            continue
        rows: list[MatrixOp] = []
        for op_id, costs in ops.items():
            if not isinstance(costs, dict):
                continue
            # Keep only numeric backend entries.
            clean: dict[str, float] = {
                str(k): float(v) for k, v in costs.items() if isinstance(v, (int, float))
            }
            rows.append(
                MatrixOp(
                    workload=wl,
                    op_id=op_id,
                    family=matrix_op_family(op_id),
                    costs=clean,
                )
            )
        if rows:
            out[wl] = rows
    return out


def cross_backend_flip_rate(
    rows: list[MatrixOp], backends: tuple[str, ...] = ("CPU", "GPU", "DSP")
) -> list[dict[str, Any]]:
    """For every backend pair, count op-pair flip rate over rows with all backends.

    For each pair ``(op_a, op_b)`` where both ops have measurements on the
    two backends in the pair, count it as a *flip* if the cost ordering on
    backend_a is opposite to backend_b. Returns one dict per backend pair.
    """

    full = [r for r in rows if r.has_all(backends)]
    stats: list[dict[str, Any]] = []
    n = len(full)
    for i, b1 in enumerate(backends):
        for b2 in backends[i + 1 :]:
            flips = 0
            ties = 0
            total = 0
            for ai in range(n):
                for bi in range(ai + 1, n):
                    a, b = full[ai], full[bi]
                    ca1, cb1 = a.costs[b1], b.costs[b1]
                    ca2, cb2 = a.costs[b2], b.costs[b2]
                    total += 1
                    if ca1 == cb1 or ca2 == cb2:
                        ties += 1
                        continue
                    if (ca1 < cb1) != (ca2 < cb2):
                        flips += 1
            counted = total - ties
            stats.append(
                {
                    "backend_a": b1,
                    "backend_b": b2,
                    "ops_compared": n,
                    "pairs_examined": total,
                    "ties": ties,
                    "flips": flips,
                    "flip_rate": (flips / counted) if counted else 0.0,
                }
            )
    return stats


def argmin_backend(row: MatrixOp, backends: tuple[str, ...]) -> str | None:
    """Return the backend with the smallest cost among ``backends``.

    Returns ``None`` if the row doesn't have all the requested backends.
    """

    if not row.has_all(backends):
        return None
    return min(backends, key=lambda b: row.costs[b])


def family_backend_specialty(
    rows: list[MatrixOp], backends: tuple[str, ...] = ("CPU", "GPU", "DSP")
) -> dict[str, dict[str, Any]]:
    """Per op-family, count argmin-backend frequency.

    Returns ``{family: {"n": int, "argmin": {backend: count}, "fastest": str}}``
    restricted to rows that have all three backends measured.
    """

    by_fam: dict[str, list[MatrixOp]] = defaultdict(list)
    for r in rows:
        if r.has_all(backends):
            by_fam[r.family].append(r)
    out: dict[str, dict[str, Any]] = {}
    for fam, fam_rows in by_fam.items():
        counts: dict[str, int] = defaultdict(int)
        for r in fam_rows:
            argm = argmin_backend(r, backends)
            if argm is not None:
                counts[argm] += 1
        fastest = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "n/a"
        out[fam] = {
            "n": len(fam_rows),
            "argmin": dict(counts),
            "fastest": fastest,
        }
    return out


def pathological_ratios(
    rows: list[MatrixOp],
    backends: tuple[str, ...] = ("CPU", "GPU", "DSP"),
    top_n: int = 5,
    min_min_us: float = 1.0,
) -> list[dict[str, Any]]:
    """Return the ``top_n`` ops by ``max/min`` cost ratio.

    Ops with ``min < min_min_us`` are skipped because tiny-latency rows
    produce explosive ratios that dominate the listing without being
    informative.
    """

    items: list[dict[str, Any]] = []
    for r in rows:
        if not r.has_all(backends):
            continue
        vals = [r.costs[b] for b in backends]
        lo = min(vals)
        hi = max(vals)
        if lo < min_min_us:
            continue
        items.append(
            {
                "workload": r.workload,
                "op_id": r.op_id,
                "family": r.family,
                "costs": {b: r.costs[b] for b in backends},
                "min_us": lo,
                "max_us": hi,
                "ratio": hi / lo,
                "fastest_backend": min(backends, key=lambda b: r.costs[b]),
                "slowest_backend": max(backends, key=lambda b: r.costs[b]),
            }
        )
    items.sort(key=lambda d: d["ratio"], reverse=True)
    return items[:top_n]


def index_correlation(
    rows: list[MatrixOp], backends: tuple[str, ...] = ("CPU", "GPU", "DSP")
) -> dict[str, dict[str, Any]]:
    """Spearman-style rank correlation between op-suffix index and cost.

    For each (family, backend), if there are >= 3 rows whose op_id ends
    in ``_<int>``, compute Spearman rho between the suffix index and the
    measured cost on that backend. Returns
    ``{family: {backend: {"n": int, "rho": float}}}``.

    Implemented inline (no scipy) to keep the audit zero-extra-deps. Ties
    are broken by average rank, which matches the standard definition.
    """

    def _ranks(xs: list[float]) -> list[float]:
        # Average-rank for ties.
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    by_fam_be: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        m = _QNN_NORM_RE.match(r.op_id)
        if m is None:
            continue
        idx = int(m.group("idx"))
        for b in backends:
            c = r.costs.get(b)
            if c is None:
                continue
            by_fam_be[(r.family, b)].append((idx, c))

    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for (fam, b), data in by_fam_be.items():
        if len(data) < 3:
            continue
        idxs = [float(d[0]) for d in data]
        costs = [d[1] for d in data]
        ri = _ranks(idxs)
        rc = _ranks(costs)
        n = len(data)
        mean_i = sum(ri) / n
        mean_c = sum(rc) / n
        num = sum((ri[k] - mean_i) * (rc[k] - mean_c) for k in range(n))
        den_i = math.sqrt(sum((x - mean_i) ** 2 for x in ri))
        den_c = math.sqrt(sum((x - mean_c) ** 2 for x in rc))
        rho = num / (den_i * den_c) if den_i > 0 and den_c > 0 else 0.0
        out[fam][b] = {"n": n, "rho": rho}
    return dict(out)


def cross_workload_consistency(
    matrices: dict[str, list[MatrixOp]],
    backends: tuple[str, ...] = ("CPU", "GPU", "DSP"),
) -> dict[str, dict[str, Any]]:
    """Compare per-family ``argmin_backend`` distributions across workloads.

    For each canonical family (ONNX->QNN folded), build a histogram of
    fastest-backend counts per workload. Returns
    ``{canonical_family: {workload: {"n": int, "argmin": {backend: count}, "fastest": str}}}``.
    """

    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for wl, rows in matrices.items():
        by_fam: dict[str, list[MatrixOp]] = defaultdict(list)
        for r in rows:
            if not r.has_all(backends):
                continue
            fam_canon = canonical_family(r.family)
            by_fam[fam_canon].append(r)
        for fam_canon, fam_rows in by_fam.items():
            counts: dict[str, int] = defaultdict(int)
            for r in fam_rows:
                argm = argmin_backend(r, backends)
                if argm is not None:
                    counts[argm] += 1
            fastest = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "n/a"
            out[fam_canon][wl] = {
                "n": len(fam_rows),
                "argmin": dict(counts),
                "fastest": fastest,
            }
    # Filter to families that appear in >=2 workloads — the cross-workload
    # comparison is only meaningful then.
    return {fam: per_wl for fam, per_wl in out.items() if len(per_wl) >= 2}
