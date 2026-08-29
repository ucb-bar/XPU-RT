"""Co-runner contention multipliers, measured on hardware.

A solo profile is necessary and not sufficient. `runtime/scripts/k1_contention_mb.py`
measures the *same* dispatch with and without a co-runner pinned to a chosen
other core, and writes the ratio to an artifact (default
``artifacts/k1_run/contention.json``). This module is the read side: it loads
that artifact and answers

    contention_factor(op_or_dispatch, placement) -> float

which is a multiplier (>= 0) on a solo service time.

Two invariants this module exists to protect:

1. **The multiplier is never folded into the solo profile.** The contention
   artifact is a separate file from the profile; the scheduler multiplies at
   duration-lookup time so that a solo cost stays a solo cost on disk. If the
   two were merged, the next re-profile would silently double-count.

2. **A missing artifact is a no-op.** :func:`load` returns ``None`` when the
   file is absent, and the scheduler wiring is off unless a model is explicitly
   installed. Nothing changes for anyone who has not run the measurement.

What the measurement said, ON THE IREE PATH, and why that qualifier now
matters (K1 / SpaceMiT X60, 8 harts, 2 clusters of 4 sharing a 512K L2 each):

    median slowdown, co-runner on the SAME cluster  : 1.043x
    median slowdown, co-runner on the OTHER cluster : 1.185x

That is the *opposite* of the shared-L2 intuition -- cross-cluster co-running
worse than sharing an L2, so "spread the work across clusters" would be the
wrong default on this part.

**IT DOES NOT REPRODUCE ON THE PATH WE SHIP.** Those numbers come from
`iree-benchmark-module` running `.vmfb` files, and the IREE path is retired;
every kernel on this board today comes out of ModelBlaster's curated tree.
Re-measured with `runtime/scripts/k1_contention_mb.py` under a paired design
(solo re-taken immediately before each arm), one co-runner gives:

    same cluster,  4 samples : 0.999  1.012  1.010  1.051
    other cluster, 4 samples : 1.061  0.995  1.002  1.004

Two distributions that straddle 1.0 and overlap completely. Arms with three
and four co-runners land inside the same band and are not monotonic in
co-runner count, which cannot be physical. So contention is BELOW THIS
MEASUREMENT'S RESOLUTION at these co-runner counts, and neither artifact
should be installed as a model today.

`artifacts/k1_run/CONTENTION_FINDINGS.md` has the full account, including the
three ways the re-measurement was wrong before it was right -- a co-runner
that was not pinned where it claimed, a survivor check that counted itself,
and an unpaired design whose drift was the size of its effect.

Nothing here changes by default: :func:`load` returns ``None`` when the
artifact is absent and the scheduler wiring is off unless a model is
explicitly installed.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

# Default artifact location, relative to the repo root.
DEFAULT_PATH = "artifacts/k1_run/contention.json"

SCHEMA = "xpurt.contention/v2"

# Canonical placement names. `same_cluster` / `other_cluster` are produced by
# the default sweep; experiment-specific runs use their own keys (e.g.
# "same_cluster_IME_x1") and are looked up verbatim.
SOLO = "solo"
SAME_CLUSTER = "same_cluster"
OTHER_CLUSTER = "other_cluster"

# Machine name -> cluster id, used only to *derive* a placement from a machine
# combination when the caller does not name one. On the K1 the two RVV clusters
# are modelled as CPU_P (harts 0-3) and CPU_E (harts 4-7).
DEFAULT_CLUSTER_OF_MACHINE = {"CPU_P": 0, "CPU_E": 1}

_MODULE_RE = re.compile(r"module_([A-Za-z0-9_.]+?)\$async_dispatch_(\d+)")
_DISPATCH_RE = re.compile(r"^(.*?)_dispatch_(\d+)$")
_TRAILING_INSTANCE_RE = re.compile(r"^(.*?)(\d+)$")


def canonical_key(op_or_dispatch: Any) -> str:
    """Reduce the many spellings of "one dispatch" to a single key.

    Accepts an :class:`Operation` (uses ``operation_name``), a benchmark module
    filename, a trace ``dispatch_key``, or a plain string, and maps

        module_dronet$async_dispatch_12_embedded_elf_riscv_64_benchmark.vmfb
        dronet0_dispatch_12
        dronet_dispatch_12

    all to ``"dronet:12"``. The trailing instance index on the model name
    (``dronet0`` -> ``dronet``) is stripped because contention is a property of
    the kernel, not of which periodic instance is running it.

    Anything unrecognised comes back lowercased and untouched, so an exact
    string match still works for hand-written keys.
    """
    name = getattr(op_or_dispatch, "operation_name", None)
    if name is None:
        name = op_or_dispatch
    if name is None:
        return ""
    name = str(name)

    m = _MODULE_RE.search(name)
    if m:
        return f"{_strip_instance(m.group(1))}:{int(m.group(2))}"

    base = os.path.basename(name)
    m = _DISPATCH_RE.match(base)
    if m:
        return f"{_strip_instance(m.group(1))}:{int(m.group(2))}"

    return base.lower()


def _strip_instance(model: str) -> str:
    """`dronet0` -> `dronet`, `yolov8` -> `yolov8` (only strip when what is left
    is still a name, so version digits survive)."""
    model = model.lower()
    m = _TRAILING_INSTANCE_RE.match(model)
    if m and m.group(1) and not m.group(1).endswith(("v", "_")):
        return m.group(1)
    return model


class ContentionModel:
    """Read-only view over a measured contention artifact."""

    def __init__(self, data: dict, path: str | None = None):
        self.path = path
        self.raw = data
        self.measurements: dict[str, dict] = dict(data.get("measurements") or {})
        self.cluster_of_machine: dict[str, int] = dict(
            data.get("cluster_of_machine") or DEFAULT_CLUSTER_OF_MACHINE
        )
        # Per-placement lookup tables, built once.
        self._by_key: dict[str, dict[str, float]] = {}
        self._median: dict[str, float] = {}
        for placement, meas in self.measurements.items():
            table: dict[str, float] = {}
            for module, entry in (meas.get("per_module") or {}).items():
                ratio = entry.get("ratio") if isinstance(entry, dict) else entry
                if ratio is None:
                    continue
                table[canonical_key(module)] = float(ratio)
            self._by_key[placement] = table
            med = meas.get("median_ratio")
            if med is None and table:
                vals = sorted(table.values())
                med = vals[len(vals) // 2]
            self._median[placement] = float(med) if med is not None else 1.0

    # -- queries ---------------------------------------------------------

    def placements(self) -> list[str]:
        return sorted(self.measurements)

    def median_factor(self, placement: str) -> float:
        """Median measured slowdown for a placement, or 1.0 if unmeasured."""
        if placement in (None, SOLO):
            return 1.0
        return self._median.get(placement, 1.0)

    def contention_factor(self, op_or_dispatch: Any, placement: str) -> float:
        """Multiplier to apply to `op_or_dispatch`'s SOLO service time when it
        runs in `placement`.

        Falls back, in order, to the per-dispatch measurement, the placement's
        median, and finally 1.0 (unknown placement / no data => no change).
        Never raises: an unknown op is not an error, it is just unmeasured.
        """
        if placement in (None, SOLO):
            return 1.0
        table = self._by_key.get(placement)
        if table:
            key = canonical_key(op_or_dispatch)
            if key in table:
                return table[key]
        return self.median_factor(placement)

    def n_co_runners(self, placement: str) -> int:
        meas = self.measurements.get(placement) or {}
        return int(meas.get("n_co_runners") or 0)

    def co_runner(self, placement: str) -> dict:
        meas = self.measurements.get(placement) or {}
        return dict(meas.get("co_runner") or {})

    # -- placement derivation --------------------------------------------

    def placement_for_combination(self, combination: Iterable[str]) -> str:
        """Guess a placement from a machine combination.

        A combination that spans two clusters is cross-cluster work; one that
        stays inside a cluster is same-cluster. Returns ``None`` when the
        combination names machines this model has no cluster mapping for, so
        the caller leaves the duration alone.
        """
        combo = [c for c in (combination or [])]
        clusters = {
            self.cluster_of_machine[m] for m in combo if m in self.cluster_of_machine
        }
        if not clusters:
            return None
        if len(clusters) > 1:
            return OTHER_CLUSTER if OTHER_CLUSTER in self.measurements else None
        return SAME_CLUSTER if SAME_CLUSTER in self.measurements else None


# -- loading -------------------------------------------------------------


def _normalise(data: dict) -> dict:
    """Accept both the v1 artifact (flat ``median_same_cluster_ratio`` plus a
    ``results`` list) and the v2 keyed-measurement artifact, and return v2.

    v1 is kept readable because the first measured run — the one that found the
    cross-cluster inversion — was written in that shape and is the evidence the
    regression test pins.
    """
    if data.get("measurements") is not None:
        return data

    results = data.get("results") or []
    out = {
        "schema": SCHEMA,
        "upgraded_from": data.get("schema", "xpurt.contention/v1"),
        "cpu_under_test": data.get("cpu"),
        "measurements": {},
    }
    for placement, ms_key, ratio_key, cpu_key in (
        (SAME_CLUSTER, "same_cluster_ms", "same_ratio", "same_cluster_cpu"),
        (OTHER_CLUSTER, "other_cluster_ms", "other_ratio", "other_cluster_cpu"),
    ):
        per_module = {}
        for r in results:
            if r.get(ratio_key) is None:
                continue
            per_module[r["module"]] = {
                "solo_ms": r.get("solo_ms"),
                "co_ms": r.get(ms_key),
                "ratio": r.get(ratio_key),
            }
        if not per_module:
            continue
        co_cpu = data.get(cpu_key)
        out["measurements"][placement] = {
            "placement": placement,
            "cpu_under_test": data.get("cpu"),
            "co_cpus": [co_cpu] if co_cpu is not None else [],
            "n_co_runners": 1 if co_cpu is not None else 0,
            "co_runner": {"remote_dir": data.get("remote_dir"), "build": None},
            "per_module": per_module,
            "median_ratio": data.get(
                "median_same_cluster_ratio"
                if placement == SAME_CLUSTER
                else "median_other_cluster_ratio"
            ),
        }
    return out


def load(path: str | None = None) -> ContentionModel | None:
    """Load a contention artifact. Returns ``None`` if it is not there.

    A missing artifact must be a no-op for every caller — that is the whole
    contract that lets this be wired in additively.
    """
    path = path or DEFAULT_PATH
    if not os.path.isabs(path):
        # Resolve relative to the repo root (this file lives in <root>/xpu-rt/).
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(root, path)
        path = cand if os.path.exists(cand) else path
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return ContentionModel(_normalise(data), path=path)


def load_if_enabled(config: dict | None) -> ContentionModel | None:
    """Config-flag entry point.

    ``config["contention"] = {"enabled": true, "path": "..."}`` (or a bare
    ``config["contention_enabled"]``) turns the model on. Absent or false, this
    returns ``None`` and nothing downstream changes.
    """
    if not config:
        return None
    section = config.get("contention")
    if isinstance(section, dict):
        if not section.get("enabled", False):
            return None
        return load(section.get("path"))
    if config.get("contention_enabled"):
        return load(config.get("contention_path"))
    return None


def contention_factor(
    op_or_dispatch: Any,
    placement: str,
    model: ContentionModel | None = None,
    autoload: bool = False,
) -> float:
    """Module-level convenience wrapper.

    Off by default in the strongest sense: with `model=None` this returns 1.0
    without touching the filesystem. Pass `autoload=True` (or a model) to opt
    into the artifact — nothing should read a measured multiplier by accident.
    """
    if model is None and autoload:
        model = load()
    if model is None:
        return 1.0
    return model.contention_factor(op_or_dispatch, placement)
