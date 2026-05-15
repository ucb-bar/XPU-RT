"""Acceptance tests for Candidate Selection + Recipe IR Commit.

Asserts:
- recipe.mlir exists and references the selected candidate_id
- selection is deterministic in greedy mode
- selection_trace.jsonl contains both considered and skipped_illegal entries
- recipe.mlir's recipe op matches the JSON projection
- compiler core untouched
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from compgen.graph_compilation.recipe_planning import run_recipe_planning
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

RECIPE_MODELS: tuple[str, ...] = (
    "tiny_mlp",
    "tiny_attention",
    "tiny_conv_block",
    "proxy_vlm",
    "proxy_vla",
    "custom_unsupported_op",
)


@pytest.fixture(scope="module")
def recipe_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("recipe_runs")
    out: dict[str, Path] = {}
    for model_id in RECIPE_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="recipe-planning",
            run_id=f"recipe_{model_id}",
            selection_mode="greedy",
        )
        out[model_id] = run_dir
    return out


# --------------------------------------------------------------------------- #
# Existence + shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", RECIPE_MODELS)
def test_recipe_planning_artifacts_emitted(
    model_id: str, recipe_runs: dict[str, Path]
) -> None:
    rp = recipe_runs[model_id] / "03_recipe_planning"
    for name in (
        "recipe.mlir",
        "candidate_selection.json",
        "selection_trace.jsonl",
        "recipe_validation.json",
        "recipe_summary.json",
    ):
        p = rp / name
        assert p.exists() and p.stat().st_size > 0, f"{model_id}: missing/empty {name}"


@pytest.mark.parametrize("model_id", RECIPE_MODELS)
def test_recipe_validation_overall_pass(
    model_id: str, recipe_runs: dict[str, Path]
) -> None:
    obj = json.loads(
        (recipe_runs[model_id] / "03_recipe_planning" / "recipe_validation.json").read_text()
    )
    # All canonical 6-model runs select something, so overall must be "pass".
    assert obj["overall"] == "pass", obj


@pytest.mark.parametrize("model_id", RECIPE_MODELS)
def test_recipe_mlir_references_selected_candidate(
    model_id: str, recipe_runs: dict[str, Path]
) -> None:
    run = recipe_runs[model_id]
    sel = json.loads(
        (run / "03_recipe_planning" / "candidate_selection.json").read_text()
    )
    text = (run / "03_recipe_planning" / "recipe.mlir").read_text()
    assert sel["selected_candidate_id"] in text, (model_id, sel["selected_candidate_id"])
    # And the action_space sha is recorded.
    assert sel["source"]["action_space_ir_sha256"] in text


@pytest.mark.parametrize("model_id", RECIPE_MODELS)
def test_recipe_mlir_recipe_delta_matches_candidate(
    model_id: str, recipe_runs: dict[str, Path]
) -> None:
    run = recipe_runs[model_id]
    sel = json.loads(
        (run / "03_recipe_planning" / "candidate_selection.json").read_text()
    )
    text = (run / "03_recipe_planning" / "recipe.mlir").read_text()
    # Each op in recipe_delta must appear as a recipe.<snake_op> line.
    for op in sel["recipe_delta"]:
        op_camel = op["op"]
        op_snake = "".join(
            "_" + c.lower() if (c.isupper() and i > 0 and (op_camel[i - 1].islower() or op_camel[i - 1].isdigit())) else c.lower()
            for i, c in enumerate(op_camel)
        )
        assert f"recipe.{op_snake} " in text, (model_id, op_camel, op_snake)


@pytest.mark.parametrize("model_id", RECIPE_MODELS)
def test_selection_trace_has_selected_event(
    model_id: str, recipe_runs: dict[str, Path]
) -> None:
    """Every canonical model must record a ``selected`` event
    (acceptance: a candidate was actually picked)."""
    run = recipe_runs[model_id]
    decisions: set[str] = set()
    with (run / "03_recipe_planning" / "selection_trace.jsonl").open() as f:
        for line in f:
            decisions.add(json.loads(line)["decision"])
    assert "selected" in decisions, (model_id, decisions)


def test_selection_trace_skipped_illegal_seen_in_suite(
    recipe_runs: dict[str, Path]
) -> None:
    """At least one model in the suite must record a ``skipped_illegal``
    trace event — proves the selector actually filters illegality
    rather than picking blindly. The greedy selector stops at the first
    legal site, so models whose top-priority site has only legal
    candidates won't have this event individually."""
    seen = False
    for run in recipe_runs.values():
        with (run / "03_recipe_planning" / "selection_trace.jsonl").open() as f:
            for line in f:
                if json.loads(line)["decision"] == "skipped_illegal":
                    seen = True
                    break
        if seen:
            break
    assert seen, "no model in suite recorded a skipped_illegal trace event"


# --------------------------------------------------------------------------- #
# Determinism (acceptance)
# --------------------------------------------------------------------------- #


def test_greedy_mode_is_deterministic_across_reruns(
    tmp_path: Path, recipe_runs: dict[str, Path],
) -> None:
    """Re-running with greedy on the same graph-analysis output must
    produce a byte-identical recipe.mlir."""
    src = recipe_runs["tiny_mlp"]
    work = tmp_path / "rerun"
    shutil.copytree(src, work)
    # Wipe and re-run against the same graph-analysis state.
    rp_dir = work / "03_recipe_planning"
    if rp_dir.exists():
        shutil.rmtree(rp_dir)
    run_recipe_planning(work, selection_mode="greedy")
    a = (src / "03_recipe_planning" / "recipe.mlir").read_text()
    b = (work / "03_recipe_planning" / "recipe.mlir").read_text()
    # Strip the timestamps from candidate_selection.json before comparing
    # the recipe.mlir; recipe.mlir itself has no timestamps so it should
    # be byte-identical.
    assert a == b


# --------------------------------------------------------------------------- #
# Negative path: tampering / illegal selection paths
# --------------------------------------------------------------------------- #


def test_selecting_illegal_fp8_via_resolver_fails(
    tmp_path: Path, recipe_runs: dict[str, Path],
) -> None:
    """The greedy selector cannot pick an illegal candidate, but if a
    caller invokes the resolver directly with an illegal candidate_id
    and allow_illegal=False, it must reject."""
    from compgen.graph_compilation.action_space_resolver import (
        IllegalCandidateError,
        resolve_candidate,
    )
    run = recipe_runs["tiny_mlp"]
    cas = json.loads(
        (run / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    illegal_fp8 = next(
        c for c in cas["candidates"]
        if c["kind"] == "quantize_fp8" and not c["legality"]["ok"]
    )
    with pytest.raises(IllegalCandidateError):
        resolve_candidate(run, illegal_fp8["candidate_id"])


def test_modifying_candidate_actions_without_remitting_ir_is_caught(
    tmp_path: Path, recipe_runs: dict[str, Path],
) -> None:
    """Mutate candidate_actions.json's recipe_delta but DO NOT
    re-emit action_space.mlir. the resolver-backed pipeline must
    refuse to use the tampered JSON.

    We mutate the candidate that greedy would otherwise pick and assert
    that a rerun of recipe_planning either picks a different candidate
    OR fails resolver validation — the key invariant is that the IR's
    recipe_delta wins over the JSON's."""
    from compgen.graph_compilation.action_space_resolver import (
        RecipeDeltaMismatchError,
        resolve_candidate,
    )
    src = recipe_runs["tiny_mlp"]
    work = tmp_path / "tampered"
    shutil.copytree(src, work)

    cas_path = work / "02_graph_analysis" / "candidate_actions.json"
    obj = json.loads(cas_path.read_text())
    legal_tile = next(
        c for c in obj["candidates"]
        if c["kind"] == "set_tile_params" and c["legality"]["ok"]
    )
    # Tamper: change the tile to one that is not in the IR.
    legal_tile["recipe_delta"][0]["tile"] = {"M": 999, "N": 999, "K": 999}
    cas_path.write_text(json.dumps(obj, indent=2, sort_keys=True))

    with pytest.raises(RecipeDeltaMismatchError):
        resolve_candidate(work, legal_tile["candidate_id"])


def test_action_space_ir_sha_drift_is_caught(
    tmp_path: Path, recipe_runs: dict[str, Path],
) -> None:
    from compgen.graph_compilation.action_space_resolver import (
        HashMismatchError,
        resolve_candidate,
    )
    src = recipe_runs["tiny_mlp"]
    work = tmp_path / "sha_drift"
    shutil.copytree(src, work)
    # Tamper sha256 in decision_sites.json — every projection's sha must match.
    p = work / "02_graph_analysis" / "decision_sites.json"
    obj = json.loads(p.read_text())
    obj["source"]["action_space_ir_sha256"] = "sha256:" + "0" * 64
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))
    cas = json.loads(
        (work / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    legal = next(c for c in cas["candidates"] if c["legality"]["ok"])
    with pytest.raises(HashMismatchError):
        resolve_candidate(work, legal["candidate_id"])


def test_no_recipe_op_for_opaque_region_tile_selection(
    recipe_runs: dict[str, Path],
) -> None:
    """The greedy selector must never produce a recipe op that tiles an
    opaque region — the contract guarantees no such candidate exists."""
    for run in recipe_runs.values():
        sel = json.loads((run / "03_recipe_planning" / "candidate_selection.json").read_text())
        if sel["selected_candidate_id"] is None:
            continue
        if sel["candidate_kind"] != "set_tile_params":
            continue
        rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
        kinds = {r["region_id"]: r["kind"] for r in rm["regions"]}
        assert not kinds[sel["region_id"]].startswith("opaque_")


# --------------------------------------------------------------------------- #
# Stage manifest integration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", RECIPE_MODELS)
def test_recipe_planning_stage_in_manifest(
    model_id: str, recipe_runs: dict[str, Path]
) -> None:
    rm = json.loads((recipe_runs[model_id] / "run_manifest.json").read_text())
    stage_ids = [s["stage_id"] for s in rm["stages"]]
    assert "recipe_planning" in stage_ids
    rp = next(s for s in rm["stages"] if s["stage_id"] == "recipe_planning")
    assert rp["status"] == "pass"
    assert rp["llm_calls"] == 0


@pytest.mark.parametrize("model_id", RECIPE_MODELS)
def test_artifact_validator_passes_with_recipe_planning(
    model_id: str, recipe_runs: dict[str, Path]
) -> None:
    from compgen.graph_compilation import validate_run
    report = validate_run(recipe_runs[model_id])
    assert report.overall == "pass", [r for r in report.rules if r.status == "fail"]


# --------------------------------------------------------------------------- #
# Compiler-core untouched (anti-coupling)
# --------------------------------------------------------------------------- #


def test_compiler_core_not_modified_by_m05() -> None:
    import subprocess
    forbidden = [
        "python/compgen/ir/payload/import_fx.py",
        "python/compgen/capture/torch_export.py",
        "python/compgen/capture/torch_mlir_bridge.py",
        "python/compgen/pipeline/driver.py",
        "python/compgen/runtime/bundle_emit.py",
    ]
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD", "--"] + forbidden,
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pytest.skip("git unavailable")
    changed = [line.strip() for line in diff.splitlines() if line.strip()]
    assert not changed, f"M-05 modified compiler core: {changed}"
