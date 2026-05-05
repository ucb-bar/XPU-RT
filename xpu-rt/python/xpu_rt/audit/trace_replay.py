"""Trace replay + decision-id discipline (M-31A.3).

Every agent decision must be deterministically reproducible from artifacts
alone. This module emits a ``DecisionTrace`` per agent decision and
verifies that re-deriving the inputs from disk yields the same hashes —
proving the decision was *really* made on the artifacts, not on hidden
chat context.

Trace schema (``agent_decision_trace_v2``)::

    {
      "schema_version": "agent_decision_trace_v2",
      "decision_id": "16-char hex",
      "commit": "<git commit>",
      "run_id": "<run id>",
      "region_id": "<region or empty>",
      "decision_index": 0,
      "input_hashes": {
        "agent_decision_request":  "sha256...",
        "llm_graph_view":          "sha256...",
        "candidate_actions":       "sha256...",
        "promotion_library_state": "sha256..."
      },
      "chosen_action": {
        "kind": "select_candidate" | "skip" | "trace_replay",
        "candidate_id": "...",
        "rationale": {...}
      },
      "rationale_paths": ["llm_graph_view.regions[0]....", ...],
      "output_hashes": {
        "agent_decision_response": "sha256...",
        "agent_decision_record":   "sha256..."
      }
    }

The ``decision_id`` is deterministic: ``sha256[:16](run_id ":" region_id
":" decision_index ":" agent_decision_request_hash)``. Two runs with the
same inputs must therefore produce the same ``decision_id``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compgen.audit.errors import (
    DecisionIdMismatch,
    ReplayHashMismatch,
)

TRACE_SCHEMA_VERSION = "agent_decision_trace_v2"


def _sha256_text(text: str | bytes) -> str:
    if isinstance(text, str):
        text = text.encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return _sha256_text(path.read_bytes())


def _sha256_dir_contents(path: Path) -> str:
    """Stable hash over a directory tree (sorted relative paths + content)."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    for sub in sorted(path.rglob("*")):
        if not sub.is_file():
            continue
        rel = sub.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(sub.read_bytes())
        h.update(b"\x01")
    return h.hexdigest()


def compute_decision_id(
    *,
    run_id: str,
    region_id: str,
    decision_index: int,
    request_hash: str,
) -> str:
    """Deterministic 16-char decision_id derived from run inputs."""
    payload = f"{run_id}:{region_id}:{decision_index}:{request_hash}"
    return _sha256_text(payload)[:16]


# --------------------------------------------------------------------------- #
# Trace dataclass
# --------------------------------------------------------------------------- #


@dataclass
class DecisionTrace:
    """One agent-decision trace, replayable from disk."""

    schema_version: str = TRACE_SCHEMA_VERSION
    decision_id: str = ""
    commit: str = ""
    run_id: str = ""
    region_id: str = ""
    decision_index: int = 0
    input_hashes: dict[str, str] = field(default_factory=dict)
    chosen_action: dict[str, Any] = field(default_factory=dict)
    rationale_paths: list[str] = field(default_factory=list)
    output_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "commit": self.commit,
            "run_id": self.run_id,
            "region_id": self.region_id,
            "decision_index": self.decision_index,
            "input_hashes": dict(self.input_hashes),
            "chosen_action": dict(self.chosen_action),
            "rationale_paths": list(self.rationale_paths),
            "output_hashes": dict(self.output_hashes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionTrace:
        return cls(
            schema_version=str(data.get("schema_version", TRACE_SCHEMA_VERSION)),
            decision_id=str(data.get("decision_id", "")),
            commit=str(data.get("commit", "")),
            run_id=str(data.get("run_id", "")),
            region_id=str(data.get("region_id", "")),
            decision_index=int(data.get("decision_index", 0)),
            input_hashes=dict(data.get("input_hashes") or {}),
            chosen_action=dict(data.get("chosen_action") or {}),
            rationale_paths=list(data.get("rationale_paths") or []),
            output_hashes=dict(data.get("output_hashes") or {}),
        )


# --------------------------------------------------------------------------- #
# Build / verify
# --------------------------------------------------------------------------- #


def _input_paths(run_dir: Path) -> dict[str, Path]:
    """Map of trace input names to artifact paths."""
    rp = run_dir / "03_recipe_planning"
    return {
        "agent_decision_request": rp / "agent_decision_request.json",
        "llm_graph_view": rp / "llm_graph_view.json",
        "candidate_actions": rp / "candidate_actions.json",
    }


def compute_input_hashes(
    run_dir: Path,
    *,
    promotion_library: Path | None = None,
) -> dict[str, str]:
    """Hash every artifact the agent reads to make its decision."""
    hashes = {
        name: _sha256_file(path)
        for name, path in _input_paths(run_dir).items()
    }
    if promotion_library is None:
        promotion_library = Path(".compgen_cache") / "recipes"
    hashes["promotion_library_state"] = _sha256_dir_contents(promotion_library)
    return hashes


def _output_paths(run_dir: Path) -> dict[str, Path]:
    rp = run_dir / "03_recipe_planning"
    return {
        "agent_decision_response": rp / "agent_decision_response.json",
        "agent_decision_record": rp / "agent_decision_record.json",
    }


def compute_output_hashes(run_dir: Path) -> dict[str, str]:
    return {
        name: _sha256_file(path)
        for name, path in _output_paths(run_dir).items()
    }


def build_trace(
    run_dir: Path,
    *,
    run_id: str,
    region_id: str = "",
    decision_index: int = 0,
    chosen_action: dict[str, Any] | None = None,
    rationale_paths: list[str] | None = None,
    commit: str = "",
    promotion_library: Path | None = None,
) -> DecisionTrace:
    """Construct a :class:`DecisionTrace` from a finished run directory."""
    input_hashes = compute_input_hashes(run_dir, promotion_library=promotion_library)
    request_hash = input_hashes.get("agent_decision_request", "")
    decision_id = compute_decision_id(
        run_id=run_id,
        region_id=region_id,
        decision_index=decision_index,
        request_hash=request_hash,
    )
    if chosen_action is None:
        # Best-effort: read the response if it's on disk.
        resp_path = run_dir / "03_recipe_planning" / "agent_decision_response.json"
        if resp_path.exists():
            try:
                resp = json.loads(resp_path.read_text())
                chosen_action = {
                    "kind": "select_candidate",
                    "candidate_id": resp.get("selected_candidate_id", ""),
                    "rationale": resp.get("rationale", {}),
                }
            except json.JSONDecodeError:
                chosen_action = {"kind": "unknown"}
        else:
            chosen_action = {"kind": "unknown"}
    return DecisionTrace(
        decision_id=decision_id,
        commit=commit,
        run_id=run_id,
        region_id=region_id,
        decision_index=decision_index,
        input_hashes=input_hashes,
        chosen_action=chosen_action,
        rationale_paths=list(rationale_paths or []),
        output_hashes=compute_output_hashes(run_dir),
    )


def write_trace(trace: DecisionTrace, *, run_dir: Path) -> Path:
    """Emit ``<run_dir>/agent_decision_trace_<index>.json``."""
    out_path = run_dir / f"agent_decision_trace_{trace.decision_index:04d}.json"
    out_path.write_text(
        json.dumps(trace.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def load_trace(path: Path) -> DecisionTrace:
    return DecisionTrace.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


@dataclass
class ReplayReport:
    """Result of a trace replay."""

    trace_path: Path
    run_dir: Path
    decision_id_match: bool
    input_hashes_match: bool
    output_hashes_match: bool
    input_deltas: dict[str, tuple[str, str]]  # {name: (expected, actual)}
    output_deltas: dict[str, tuple[str, str]]

    @property
    def all_match(self) -> bool:
        return (
            self.decision_id_match
            and self.input_hashes_match
            and self.output_hashes_match
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_path": str(self.trace_path),
            "run_dir": str(self.run_dir),
            "decision_id_match": self.decision_id_match,
            "input_hashes_match": self.input_hashes_match,
            "output_hashes_match": self.output_hashes_match,
            "input_deltas": {k: list(v) for k, v in self.input_deltas.items()},
            "output_deltas": {k: list(v) for k, v in self.output_deltas.items()},
            "all_match": self.all_match,
        }


def replay(
    *,
    trace_path: Path,
    run_dir: Path,
    promotion_library: Path | None = None,
    strict: bool = True,
) -> ReplayReport:
    """Re-derive hashes from ``run_dir`` and compare against ``trace_path``.

    Args:
        trace_path: Path to a saved ``agent_decision_trace_<n>.json``.
        run_dir: Run directory whose artifacts should reproduce the trace.
            For a same-run replay this is the run that emitted the trace;
            for cross-run replay it is a regenerated run dir.
        promotion_library: Path to the recipe library whose state should
            match. Defaults to ``.compgen_cache/recipes/``.
        strict: When True, raises :class:`ReplayHashMismatch` on any
            mismatch. When False, returns the report and lets the caller
            inspect deltas.

    Returns:
        :class:`ReplayReport` with per-field match status and deltas.
    """
    trace = load_trace(trace_path)
    actual_inputs = compute_input_hashes(run_dir, promotion_library=promotion_library)
    actual_outputs = compute_output_hashes(run_dir)

    input_deltas = {
        name: (expected, actual_inputs.get(name, ""))
        for name, expected in trace.input_hashes.items()
        if expected != actual_inputs.get(name, "")
    }
    output_deltas = {
        name: (expected, actual_outputs.get(name, ""))
        for name, expected in trace.output_hashes.items()
        if expected != actual_outputs.get(name, "")
    }

    actual_decision_id = compute_decision_id(
        run_id=trace.run_id,
        region_id=trace.region_id,
        decision_index=trace.decision_index,
        request_hash=actual_inputs.get("agent_decision_request", ""),
    )
    decision_id_match = actual_decision_id == trace.decision_id

    report = ReplayReport(
        trace_path=trace_path,
        run_dir=run_dir,
        decision_id_match=decision_id_match,
        input_hashes_match=not input_deltas,
        output_hashes_match=not output_deltas,
        input_deltas=input_deltas,
        output_deltas=output_deltas,
    )
    if strict and not report.all_match:
        details: list[str] = []
        if not decision_id_match:
            details.append(
                f"decision_id mismatch: trace={trace.decision_id} "
                f"actual={actual_decision_id}"
            )
        for name, (exp, act) in input_deltas.items():
            details.append(f"input {name}: expected={exp[:16]} actual={act[:16]}")
        for name, (exp, act) in output_deltas.items():
            details.append(f"output {name}: expected={exp[:16]} actual={act[:16]}")
        raise ReplayHashMismatch(
            f"replay {trace_path.name} on {run_dir} mismatched:\n  "
            + "\n  ".join(details)
        )
    return report


# --------------------------------------------------------------------------- #
# Decision-id validation hook (used by validate_agent_decision_response)
# --------------------------------------------------------------------------- #


def assert_decision_ids_match(
    *,
    request: dict[str, Any],
    response: dict[str, Any],
) -> None:
    """If the request carries a ``decision_id``, the response must echo it."""
    req_id = request.get("decision_id")
    if not req_id:
        return  # additive: pre-M-31A requests have no decision_id
    resp_id = response.get("decision_id")
    if resp_id != req_id:
        raise DecisionIdMismatch(
            f"agent_decision_response.decision_id={resp_id!r} does not match "
            f"agent_decision_request.decision_id={req_id!r}"
        )
