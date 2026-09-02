"""Evaluate an original-vs-feedback solver matrix without changing the rules.

The compiler loop and the solver sweep already produce schedules.  This module
is the missing experiment-level contract: it proves that every cell contains
the same instances, scores every validated schedule through
``schedule_scoring``, excludes timeouts from ranking, applies the three
eligibility gates, and asks whether the best feedback schedule beats *every*
validated original-graph schedule under ``candidate_objective``.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import candidate_objective as objective
import job_names
import schedule_scoring
import workload_spec

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import check_schedule_feasibility as feasibility  # noqa: E402

VALIDATED = "validated"
NON_RESULTS = {"timeout", "infeasible", "error", "unavailable"}
REQUIRED_GATES = ("correctness", "hardware_legality", "profile_validity")


class ManifestError(ValueError):
    """The experiment is not comparable and therefore has no verdict."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(repo: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo, path)


def _outcome_dict(out: objective.CandidateOutcome) -> dict:
    d = asdict(out)
    d.update({
        "total_deadline_misses": out.total_misses(),
        "max_lateness_ms": out.worst_lateness(),
        "frequency_shortfall": out.worst_frequency_shortfall(),
        "critical_p99_ms": out.worst_p99(),
    })
    return d


def _check_feasible(label: str, schedule: dict) -> dict:
    dispatches = schedule.get("dispatches") or {}
    checks = {
        "overlaps": feasibility.find_overlaps(
            feasibility.intervals_by_machine(dispatches), 1e-6),
        "dependency_violations": feasibility.find_dependency_violations(
            dispatches, 1e-6),
        "forward_edges": feasibility.find_forward_edges(dispatches, 1e-6),
        "out_of_range_targets": feasibility.find_out_of_range_targets(
            dispatches, feasibility.K1_HARTS_PER_CLUSTER),
        "illegal_implementations": feasibility.find_illegal_implementations(
            dispatches),
    }
    counts = {name: len(rows) for name, rows in checks.items()}
    if any(counts.values()):
        raise ManifestError(f"{label}: schedule marked validated is infeasible: "
                            f"{counts}")
    return counts


def _gate_statuses(manifest: dict) -> Tuple[bool, List[dict]]:
    gates = manifest.get("gates") or {}
    rows = []
    for name in REQUIRED_GATES:
        gate = gates.get(name) or {}
        status = str(gate.get("status", "missing"))
        rows.append({"gate": name, "status": status,
                     "evidence": gate.get("evidence")})
    return all(r["status"] == "pass" for r in rows), rows


def _feedback_transformation(repo: str, transformation: dict | None,
                             known: Iterable[str]) -> dict | None:
    """Validate and fingerprint the scheduler feedback behind a rewrite.

    A prose ``source`` is useful to a reader, but it is not provenance.  When
    a transformation names a feedback artifact, require the actual XPU-RT
    schema and require every transformed model to have emitted
    ``prefer_finer``.  The source-schedule hash is checked later, once all
    matrix cells have been loaded.
    """
    if transformation is None:
        return None
    out = dict(transformation)
    feedback_rel = out.get("feedback_artifact")
    if not feedback_rel:
        return out
    feedback_path = _resolve(repo, feedback_rel)
    if not os.path.isfile(feedback_path):
        raise ManifestError(f"feedback: missing provenance {feedback_path}")
    with open(feedback_path) as f:
        feedback = json.load(f)
    if feedback.get("schema_version") != 1:
        raise ManifestError("feedback provenance must use schema_version 1")

    finer_models = set()
    dispatches = feedback.get("dispatches") or {}
    for dispatch_id, row in dispatches.items():
        if "prefer_finer" in (row.get("hints") or ()):
            job = str(dispatch_id).rsplit("_dispatch_", 1)[0]
            finer_models.add(job_names.model_of(job, known))
    missing = set(out.get("targets") or ()) - finer_models
    if missing:
        raise ManifestError(
            "feedback transformation targets without prefer_finer evidence: "
            f"{sorted(missing)}")
    out["feedback_sha256"] = sha256_file(feedback_path)
    out["feedback_dispatches_with_hints"] = len(dispatches)
    return out


def _flatten_workload(value, prefix=()) -> dict:
    """Flatten semantic workload fields; ``_comment`` is never a contract."""
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if key == "_comment":
                continue
            out.update(_flatten_workload(child, prefix + (str(key),)))
        return out
    return {".".join(prefix): value}


def _workload_changes(original: dict, feedback: dict,
                      transformation: dict | None) -> List[dict]:
    """Require every phase-to-phase input change to be declared exactly."""
    before = _flatten_workload(original)
    after = _flatten_workload(feedback)
    observed = []
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        if old != new:
            observed.append({"path": path, "from": old, "to": new})
    expected = sorted(
        (transformation or {}).get("allowed_workload_changes") or [],
        key=lambda row: row.get("path", ""))
    if observed != expected:
        raise ManifestError(
            "workload changes do not match transformation contract: "
            f"observed={observed}, allowed={expected}")
    return observed


def _validated_cells(manifest: dict, repo: str) -> Tuple[List[dict], dict]:
    common = manifest.get("common") or {}
    critical = tuple(common.get("critical_models") or ())
    heavy = common.get("heavy_model")
    cells: List[dict] = []
    phase_meta: Dict[str, dict] = {}

    phases = manifest.get("phases") or {}
    if set(phases) != {"original", "feedback"}:
        raise ManifestError("phases must contain exactly original and feedback")

    expected_solvers = list(common.get("solvers") or ())
    if not expected_solvers:
        raise ManifestError("common.solvers must not be empty")

    for phase_name in ("original", "feedback"):
        phase = phases[phase_name]
        workload_path = _resolve(repo, phase["networks_json"])
        if not os.path.isfile(workload_path):
            raise ManifestError(f"{phase_name}: missing workload {workload_path}")
        with open(workload_path) as f:
            workload = json.load(f)
        windows, known = workload_spec.windows_and_names(workload)
        declared_periods = workload_spec.periods_ms(workload)
        transformation = phase.get("transformation")
        if phase_name == "feedback":
            transformation = _feedback_transformation(
                repo, transformation, known)
        phase_meta[phase_name] = {
            "networks_json": phase["networks_json"],
            "networks_sha256": sha256_file(workload_path),
            "transformation": transformation,
            "_workload": workload,
        }

        by_solver = {str(c.get("solver")): c for c in phase.get("cells") or ()}
        if set(by_solver) != set(expected_solvers):
            raise ManifestError(
                f"{phase_name}: solver cells {sorted(by_solver)} do not match "
                f"common.solvers {sorted(expected_solvers)}")
        for solver in expected_solvers:
            src = by_solver[solver]
            status = str(src.get("status"))
            if status != VALIDATED:
                if status not in NON_RESULTS:
                    raise ManifestError(
                        f"{phase_name}/{solver}: unknown status {status!r}")
                cells.append({"phase": phase_name, "solver": solver,
                              "status": status, "detail": src.get("detail", ""),
                              "wall_s": src.get("wall_s")})
                continue

            schedule_path = _resolve(repo, src["schedule"])
            if not os.path.isfile(schedule_path):
                raise ManifestError(
                    f"{phase_name}/{solver}: missing schedule {schedule_path}")
            with open(schedule_path) as f:
                schedule = json.load(f)
            label = f"{phase_name}/{solver}"
            feasibility_counts = _check_feasible(label, schedule)
            summary, outcome, _ = schedule_scoring.score(
                label, schedule, windows, critical, heavy, known,
                declared_periods)
            cells.append({
                "phase": phase_name,
                "solver": solver,
                "status": VALIDATED,
                "schedule": src["schedule"],
                "schedule_sha256": sha256_file(schedule_path),
                "pdb_hash": (schedule.get("metadata") or {}).get("pdb_hash"),
                "wall_s": src.get("wall_s"),
                "instances": schedule_scoring.instances_per_model(schedule, known),
                "n_dispatches": len(schedule.get("dispatches") or {}),
                "feasibility": feasibility_counts,
                "summary": summary,
                "terms": _outcome_dict(outcome),
                "_outcome": outcome,
            })
    return cells, phase_meta


def _same_work(cells: Iterable[dict]) -> dict:
    validated = [c for c in cells if c["status"] == VALIDATED]
    if not validated:
        raise ManifestError("the experiment contains no validated schedules")
    reference = validated[0]["instances"]
    for cell in validated[1:]:
        if cell["instances"] != reference:
            raise ManifestError(
                "instance counts differ: "
                f"{validated[0]['phase']}/{validated[0]['solver']}={reference} "
                f"vs {cell['phase']}/{cell['solver']}={cell['instances']}")
    return reference


def _best(cells: List[dict]) -> Tuple[dict, List[dict]]:
    """Return a stable representative and every cell tied for best."""
    ranked = objective.rank([c["_outcome"] for c in cells])
    best_out = ranked[0]
    representative = next(c for c in cells if c["_outcome"] is best_out)
    co_best = [c for c in cells
               if objective.compare(c["_outcome"], best_out)[0] == 0]
    return representative, co_best


def evaluate(manifest: dict, repo: str) -> dict:
    """Return the self-contained, JSON-serialisable benchmark result."""
    if manifest.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    cells, phase_meta = _validated_cells(manifest, repo)
    changes = _workload_changes(
        phase_meta["original"].pop("_workload"),
        phase_meta["feedback"].pop("_workload"),
        phase_meta["feedback"].get("transformation"))
    instances = _same_work(cells)
    gates_pass, gates = _gate_statuses(manifest)

    originals = [c for c in cells
                 if c["phase"] == "original" and c["status"] == VALIDATED]
    candidates = [c for c in cells
                  if c["phase"] == "feedback" and c["status"] == VALIDATED]
    if not originals or not candidates:
        raise ManifestError("both phases need at least one validated schedule")
    transformation = phase_meta["feedback"].get("transformation") or {}
    feedback_source_hash = transformation.get("source_schedule_sha256")
    if feedback_source_hash and feedback_source_hash not in {
            c["schedule_sha256"] for c in originals}:
        raise ManifestError(
            "feedback provenance source_schedule_sha256 does not identify a "
            "validated original schedule")
    if (manifest.get("common") or {}).get("require_distinct_pdb"):
        original_hashes = {c.get("pdb_hash") for c in originals}
        candidate_hashes = {c.get("pdb_hash") for c in candidates}
        if None in original_hashes | candidate_hashes:
            raise ManifestError("require_distinct_pdb needs a pdb_hash in every "
                                "validated schedule")
        if original_hashes & candidate_hashes:
            raise ManifestError("original and feedback phases share a pdb_hash; "
                                "the feedback costs/design space did not change")

    original_best, original_co_best = _best(originals)
    candidate_best, candidate_co_best = _best(candidates)
    comparisons = []
    beats_all = True
    for base in originals:
        accepted, why = objective.accept(candidate_best["_outcome"],
                                         base["_outcome"])
        beats_all = beats_all and accepted
        comparisons.append({"baseline_solver": base["solver"],
                            "accepted": accepted, "why": why})

    result_cells = []
    for cell in cells:
        result_cells.append({k: v for k, v in cell.items() if k != "_outcome"})
    accepted = gates_pass and beats_all
    reason = ("feedback candidate beats every validated original solver"
              if accepted else
              "feedback candidate did not pass every gate and comparison")
    return {
        "schema_version": 1,
        "experiment_id": manifest.get("experiment_id"),
        "description": manifest.get("description", ""),
        "common": manifest.get("common") or {},
        "numbers_are": "predicted schedules using measured K1 dispatch profiles",
        "objective": "xpu-rt/candidate_objective.py nine-term lexicographic order",
        "instances": instances,
        "workload_changes": changes,
        "gates": gates,
        "phases": phase_meta,
        "cells": result_cells,
        "original_best": original_best["solver"],
        "original_co_best": [c["solver"] for c in original_co_best],
        "feedback_best": candidate_best["solver"],
        "feedback_co_best": [c["solver"] for c in candidate_co_best],
        "comparisons": comparisons,
        "accepted": accepted,
        "verdict": ("ACCEPT" if accepted else "REJECT"),
        "why": reason,
    }
