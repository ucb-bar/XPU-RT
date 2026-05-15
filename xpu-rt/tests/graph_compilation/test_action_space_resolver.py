"""Acceptance tests for Action Space Resolver.

Asserts that ``resolve_candidate`` honours the canonical IR
(``02_graph_analysis/action_space.mlir``) as the source of truth and
rejects every class of mismatch:

- nonexistent candidate_id
- illegal candidate (without --allow-illegal)
- JSON ``recipe_delta`` tampered without re-emitting the IR
- ``action_space_ir_sha256`` mismatch in any projection
- opaque-region tiling/fusion candidates do not exist (so cannot be selected)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from compgen.graph_compilation.action_space_resolver import (
    CandidateNotFoundError,
    HashMismatchError,
    IllegalCandidateError,
    RecipeDeltaMismatchError,
    resolve_candidate,
)
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

RESOLVER_MODELS: tuple[str, ...] = (
    "tiny_mlp",
    "tiny_attention",
    "tiny_conv_block",
    "proxy_vlm",
    "proxy_vla",
    "custom_unsupported_op",
)


@pytest.fixture(scope="module")
def resolver_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("resolver_runs")
    out: dict[str, Path] = {}
    for model_id in RESOLVER_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="graph-analysis",
            run_id=f"res_{model_id}",
        )
        out[model_id] = run_dir
    return out


def _candidates(run: Path) -> list[dict]:
    return json.loads(
        (run / "02_graph_analysis" / "candidate_actions.json").read_text()
    )["candidates"]


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", RESOLVER_MODELS)
def test_resolve_a_legal_candidate(
    model_id: str, resolver_runs: dict[str, Path]
) -> None:
    run = resolver_runs[model_id]
    legal = next(
        (c for c in _candidates(run) if c["legality"]["ok"]),
        None,
    )
    assert legal is not None, f"{model_id}: no legal candidates"
    resolved, report = resolve_candidate(run, legal["candidate_id"])
    assert report.overall == "pass"
    assert resolved.legality_ok is True
    assert resolved.recipe_delta == legal["recipe_delta"]
    # Every check must pass on the happy path.
    for c in report.checks:
        assert c["status"] == "pass", c


@pytest.mark.parametrize("model_id", RESOLVER_MODELS)
def test_resolve_writes_outputs(
    model_id: str, resolver_runs: dict[str, Path], tmp_path: Path
) -> None:
    """write_outputs=True must emit the resolver report + candidate_selection
    + selected_recipe_delta.mlir."""
    src = resolver_runs[model_id]
    work = tmp_path / model_id
    shutil.copytree(src, work)
    legal = next(c for c in _candidates(work) if c["legality"]["ok"])
    resolved, _ = resolve_candidate(
        work, legal["candidate_id"],
        write_outputs=True,
        selection_mode="test_smoke",
        rationale={"primary_reason": "test", "evidence": ["legal"]},
    )
    rep = work / "02_graph_analysis" / "action_space_resolver_report.json"
    sel = work / "03_recipe_planning" / "candidate_selection.json"
    delta = work / "03_recipe_planning" / "selected_recipe_delta.mlir"
    assert rep.exists()
    assert sel.exists()
    assert delta.exists()
    sel_obj = json.loads(sel.read_text())
    assert sel_obj["selected_candidate_id"] == resolved.candidate_id
    assert sel_obj["selection_mode"] == "test_smoke"
    assert sel_obj["source"]["action_space_ir_sha256"] == resolved.source["action_space_ir_sha256"]
    # selected_recipe_delta.mlir must contain the verbatim candidate block.
    text = delta.read_text()
    assert resolved.candidate_id in text
    assert resolved.source["action_space_ir_sha256"] in text


# --------------------------------------------------------------------------- #
# Negative path: nonexistent
# --------------------------------------------------------------------------- #


def test_nonexistent_candidate_raises(resolver_runs: dict[str, Path]) -> None:
    run = resolver_runs["tiny_mlp"]
    with pytest.raises(CandidateNotFoundError):
        resolve_candidate(run, "cand_does_not_exist_xxxx")


# --------------------------------------------------------------------------- #
# Negative path: illegal
# --------------------------------------------------------------------------- #


def test_illegal_fp8_candidate_rejected_by_default(
    resolver_runs: dict[str, Path],
) -> None:
    run = resolver_runs["tiny_mlp"]
    illegal = next(
        c for c in _candidates(run)
        if c["kind"] == "quantize_fp8" and not c["legality"]["ok"]
    )
    with pytest.raises(IllegalCandidateError):
        resolve_candidate(run, illegal["candidate_id"])


def test_allow_illegal_flag_bypasses_gate(
    resolver_runs: dict[str, Path],
) -> None:
    run = resolver_runs["tiny_mlp"]
    illegal = next(
        c for c in _candidates(run)
        if c["kind"] == "quantize_fp8" and not c["legality"]["ok"]
    )
    resolved, report = resolve_candidate(
        run, illegal["candidate_id"], allow_illegal=True
    )
    assert resolved.legality_ok is False
    # The legality_gate check still records the bypass, but overall passes.
    assert report.overall == "pass"


# --------------------------------------------------------------------------- #
# Negative path: hash mismatch
# --------------------------------------------------------------------------- #


def test_action_space_ir_sha256_mismatch_raises(
    resolver_runs: dict[str, Path], tmp_path: Path,
) -> None:
    """If candidate_actions.json's action_space_ir_sha256 disagrees with
    the actual sha256 of action_space.mlir, the resolver must reject."""
    src = resolver_runs["tiny_mlp"]
    work = tmp_path / "sha_mismatch"
    shutil.copytree(src, work)
    cas_path = work / "02_graph_analysis" / "candidate_actions.json"
    obj = json.loads(cas_path.read_text())
    obj["source"]["action_space_ir_sha256"] = "sha256:" + "a" * 64
    cas_path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    legal = next(c for c in obj["candidates"] if c["legality"]["ok"])
    with pytest.raises(HashMismatchError):
        resolve_candidate(work, legal["candidate_id"])


def test_tampered_json_recipe_delta_rejected(
    resolver_runs: dict[str, Path], tmp_path: Path,
) -> None:
    """Mutating recipe_delta in the JSON projection without re-emitting
    action_space.mlir must be caught — JSON is a projection only."""
    src = resolver_runs["tiny_mlp"]
    work = tmp_path / "tampered"
    shutil.copytree(src, work)
    cas_path = work / "02_graph_analysis" / "candidate_actions.json"
    obj = json.loads(cas_path.read_text())
    legal = next(c for c in obj["candidates"] if c["legality"]["ok"] and c["kind"] == "set_tile_params")
    legal["recipe_delta"][0]["region"] = "TAMPERED_REGION"
    cas_path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    with pytest.raises(RecipeDeltaMismatchError):
        resolve_candidate(work, legal["candidate_id"])


def test_decision_sites_sha_mismatch_raises(
    resolver_runs: dict[str, Path], tmp_path: Path,
) -> None:
    """The resolver checks ALL three projections' sha — not just
    candidate_actions."""
    src = resolver_runs["tiny_mlp"]
    work = tmp_path / "ds_mismatch"
    shutil.copytree(src, work)
    ds_path = work / "02_graph_analysis" / "decision_sites.json"
    obj = json.loads(ds_path.read_text())
    obj["source"]["action_space_ir_sha256"] = "sha256:" + "0" * 64
    ds_path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    legal = next(c for c in _candidates(work) if c["legality"]["ok"])
    with pytest.raises(HashMismatchError):
        resolve_candidate(work, legal["candidate_id"])


# --------------------------------------------------------------------------- #
# Negative path: opaque-region tiling cannot be selected
# --------------------------------------------------------------------------- #


def test_opaque_region_tiling_candidate_does_not_exist(
    resolver_runs: dict[str, Path],
) -> None:
    """Opaque regions never get tiling candidates (invariant), so an
    LLM cannot select a tile on them. We assert the absence rather than
    expecting a specific error class."""
    run = resolver_runs["custom_unsupported_op"]  # has crgtoy.affine_gelu opaque
    rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
    opaque_regions = {r["region_id"] for r in rm["regions"] if r["kind"].startswith("opaque_")}
    cas = _candidates(run)
    bad = [
        c for c in cas
        if c["kind"] == "set_tile_params" and c["region_id"] in opaque_regions
    ]
    assert not bad, f"opaque regions got tiling candidates: {[c['candidate_id'] for c in bad]}"


# --------------------------------------------------------------------------- #
# IR cross-check: parsed recipe_delta from IR matches JSON for every legal candidate
# --------------------------------------------------------------------------- #


def test_every_legal_candidate_resolves_for_every_model(
    resolver_runs: dict[str, Path],
) -> None:
    """End-to-end smoke: every legal candidate across every model
    resolves and round-trips via the IR parser."""
    failures: list[tuple[str, str, str]] = []
    for model_id, run in resolver_runs.items():
        for c in _candidates(run):
            if not c["legality"]["ok"]:
                continue
            try:
                resolve_candidate(run, c["candidate_id"])
            except Exception as exc:  # pragma: no cover — diagnostics only
                failures.append((model_id, c["candidate_id"], f"{type(exc).__name__}: {exc}"))
    assert not failures, failures[:5]
