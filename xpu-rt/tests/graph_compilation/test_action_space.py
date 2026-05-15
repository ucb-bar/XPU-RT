"""Acceptance tests for Decision Sites + Candidate Actions.

Asserts the acceptance checklist on the canonical 6-model suite,
including target sensitivity (default vs. discovered profile must
produce different priorities or costs).

Per the project's anti-mock policy, every test runs the real
``run_graph_compilation`` pipeline — no fake dossiers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from compgen.graph_compilation.run import run_graph_compilation
from compgen.graph_compilation.target_discovery import build_target_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

ACTION_SPACE_MODELS: tuple[str, ...] = (
    "tiny_mlp",
    "tiny_attention",
    "tiny_conv_block",
    "proxy_vlm",
    "proxy_vla",
    "custom_unsupported_op",
)


@pytest.fixture(scope="module")
def action_space_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("action_space_runs")
    out: dict[str, Path] = {}
    for model_id in ACTION_SPACE_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="graph-analysis",
            run_id=f"as_{model_id}",
        )
        out[model_id] = run_dir
    return out


@pytest.fixture(scope="module")
def discovered_target_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Same suite, run with the auto-discovered host target. Used for
    target-sensitivity assertions."""
    base = tmp_path_factory.mktemp("action_space_runs_disc")
    target_yaml = base / "discovered_target.yaml"
    build_target_yaml(out_path=target_yaml, target_id="discovered_test")
    out: dict[str, Path] = {}
    for model_id in ACTION_SPACE_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=target_yaml,
            out_dir=run_dir,
            stop_after="graph-analysis",
            run_id=f"as_disc_{model_id}",
        )
        out[model_id] = run_dir
    return out


# --------------------------------------------------------------------------- #
# Existence + IR cross-link
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_required_artifacts_emitted(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    ga = action_space_runs[model_id] / "02_graph_analysis"
    for name in (
        "action_space.mlir",
        "decision_sites.json",
        "candidate_actions.json",
        "llm_action_space.json",
        "action_space_validation.json",
    ):
        p = ga / name
        assert p.exists() and p.stat().st_size > 0, f"{model_id}: missing/empty {name}"


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_action_space_validation_passes(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    p = (
        action_space_runs[model_id]
        / "02_graph_analysis"
        / "action_space_validation.json"
    )
    obj = json.loads(p.read_text())
    failed = [c for c in obj["checks"] if c["status"] == "fail"]
    assert obj["overall"] == "pass", failed


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_json_projections_share_action_space_ir_sha256(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    """Every JSON projection must carry the same action_space_ir_sha256
    AND it must match the actual sha256 of action_space.mlir bytes."""
    ga = action_space_runs[model_id] / "02_graph_analysis"
    actual_sha = (
        "sha256:"
        + hashlib.sha256((ga / "action_space.mlir").read_bytes()).hexdigest()
    )
    for name in (
        "decision_sites.json",
        "candidate_actions.json",
        "llm_action_space.json",
        "action_space_validation.json",
    ):
        obj = json.loads((ga / name).read_text())
        assert obj["source"]["action_space_ir_sha256"] == actual_sha, (
            model_id, name, obj["source"]["action_space_ir_sha256"], actual_sha,
        )


# --------------------------------------------------------------------------- #
# Referential integrity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_every_decision_site_references_valid_region(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    run = action_space_runs[model_id]
    rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
    region_ids = {r["region_id"] for r in rm["regions"]}
    sites = json.loads(
        (run / "02_graph_analysis" / "decision_sites.json").read_text()
    )["sites"]
    for s in sites:
        assert s["region_id"] in region_ids, (model_id, s)


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_every_candidate_references_valid_site_and_region(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    run = action_space_runs[model_id]
    rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
    region_ids = {r["region_id"] for r in rm["regions"]}
    ds = json.loads(
        (run / "02_graph_analysis" / "decision_sites.json").read_text()
    )
    site_ids = {s["site_id"] for s in ds["sites"]}
    cas = json.loads(
        (run / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    for c in cas["candidates"]:
        assert c["region_id"] in region_ids, (model_id, c["candidate_id"])
        assert c["site_id"] in site_ids, (model_id, c["candidate_id"])


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_every_candidate_has_recipe_delta(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    cas = json.loads(
        (
            action_space_runs[model_id]
            / "02_graph_analysis"
            / "candidate_actions.json"
        ).read_text()
    )
    for c in cas["candidates"]:
        assert c["recipe_delta"], (model_id, c["candidate_id"])
        for op in c["recipe_delta"]:
            assert "op" in op, (model_id, op)


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_illegal_candidates_have_reason(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    cas = json.loads(
        (
            action_space_runs[model_id]
            / "02_graph_analysis"
            / "candidate_actions.json"
        ).read_text()
    )
    for c in cas["candidates"]:
        if c["legality"]["ok"]:
            continue
        assert c["legality"].get("reason"), (model_id, c["candidate_id"])


# --------------------------------------------------------------------------- #
# llm_action_space hides illegal candidates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_llm_action_space_only_legal(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    run = action_space_runs[model_id]
    cas = json.loads(
        (run / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    illegal_ids = {c["candidate_id"] for c in cas["candidates"] if not c["legality"]["ok"]}
    las = json.loads(
        (run / "02_graph_analysis" / "llm_action_space.json").read_text()
    )
    for site in las["ranked_sites"]:
        for c in site["legal_candidates"]:
            assert c["candidate_id"] not in illegal_ids, (
                model_id, site["site_id"], c["candidate_id"],
            )


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_llm_action_space_summary_consistent(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    run = action_space_runs[model_id]
    cas = json.loads(
        (run / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    las = json.loads(
        (run / "02_graph_analysis" / "llm_action_space.json").read_text()
    )
    total = len(cas["candidates"])
    legal = sum(1 for c in cas["candidates"] if c["legality"]["ok"])
    illegal = total - legal
    s = las["summary"]
    assert s["candidate_count_total"] == total
    assert s["candidate_count_legal"] == legal
    assert s["hidden_illegal_candidates"] == illegal


# --------------------------------------------------------------------------- #
# Family-specific invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_tile_sizes_only_from_working_set_curve(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    run = action_space_runs[model_id]
    cas = json.loads(
        (run / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    gd = json.loads(
        (run / "02_graph_analysis" / "graph_dossier_v2.json").read_text()
    )
    for c in cas["candidates"]:
        if c["kind"] != "set_tile_params":
            continue
        d = json.loads((run / gd["region_dossiers"][c["region_id"]]).read_text())
        curve_tiles = [t["tile"] for t in d["working_set_curve"]]
        ours = c["recipe_delta"][0]["tile"]
        assert ours in curve_tiles, (model_id, c["candidate_id"], ours)


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_opaque_regions_get_no_tiling_or_fusion_or_numerics(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    run = action_space_runs[model_id]
    rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
    opaque = {r["region_id"] for r in rm["regions"] if r["kind"].startswith("opaque_")}
    cas = json.loads(
        (run / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    forbidden_kinds = {
        "set_tile_params", "fuse_producer_consumer",
        "set_accumulator_fp16", "quantize_fp8", "enable_fast_math",
    }
    for c in cas["candidates"]:
        if c["region_id"] in opaque and c["kind"] in forbidden_kinds:
            pytest.fail(
                f"{model_id}: opaque region {c['region_id']} got "
                f"{c['kind']} candidate {c['candidate_id']}"
            )


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_fp8_candidates_obey_numerical_sensitivity(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    run = action_space_runs[model_id]
    gd = json.loads(
        (run / "02_graph_analysis" / "graph_dossier_v2.json").read_text()
    )
    cas = json.loads(
        (run / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    for c in cas["candidates"]:
        if c["kind"] != "quantize_fp8":
            continue
        d = json.loads((run / gd["region_dossiers"][c["region_id"]]).read_text())
        st = d["numerical_sensitivity"]["fp8_e4m3"]["status"]
        if c["legality"]["ok"]:
            assert st == "safe", (model_id, c["candidate_id"], st)
        else:
            assert st in ("risky", "exceeds_budget", "requires_reference"), (
                model_id, c["candidate_id"], st,
            )


@pytest.mark.parametrize("model_id", ACTION_SPACE_MODELS)
def test_fusion_candidates_obey_use_def_invariants(
    model_id: str, action_space_runs: dict[str, Path]
) -> None:
    run = action_space_runs[model_id]
    use_def = json.loads(
        (run / "02_graph_analysis" / "tensor_use_def_graph.json").read_text()
    )
    cas = json.loads(
        (run / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    tensor_lookup = {t["tensor_id"]: t for t in use_def["tensors"]}
    for c in cas["candidates"]:
        if c["kind"] != "fuse_producer_consumer":
            continue
        tid = c["recipe_delta"][0]["via_tensor"]
        t = tensor_lookup.get(tid)
        assert t is not None, (model_id, tid)
        assert t["consumer_count"] == 1, (model_id, tid, t["consumer_count"])
        assert t["producer_lifetime_class"] == "transient", (model_id, tid)


# --------------------------------------------------------------------------- #
# Suite-level acceptance bars
# --------------------------------------------------------------------------- #


def _aggregate(action_space_runs: dict[str, Path]) -> dict[str, int]:
    agg = {
        "legal_tile": 0, "illegal_tile": 0,
        "illegal_fp8": 0,
        "fusion": 0,
        "extension_closure": 0,
    }
    for run in action_space_runs.values():
        cas = json.loads(
            (run / "02_graph_analysis" / "candidate_actions.json").read_text()
        )
        for c in cas["candidates"]:
            if c["kind"] == "set_tile_params":
                agg["legal_tile" if c["legality"]["ok"] else "illegal_tile"] += 1
            elif c["kind"] == "quantize_fp8" and not c["legality"]["ok"]:
                agg["illegal_fp8"] += 1
            elif c["kind"] == "fuse_producer_consumer":
                agg["fusion"] += 1
            elif c["kind"] in ("create_payload_lowering_extension",
                               "create_kernel_contract", "keep_as_fallback"):
                agg["extension_closure"] += 1
    return agg


def test_suite_has_at_least_one_legal_tiling_candidate(
    action_space_runs: dict[str, Path]
) -> None:
    assert _aggregate(action_space_runs)["legal_tile"] >= 1


def test_suite_has_at_least_one_illegal_tiling_candidate(
    action_space_runs: dict[str, Path]
) -> None:
    assert _aggregate(action_space_runs)["illegal_tile"] >= 1


def test_suite_has_at_least_one_illegal_fp8_candidate(
    action_space_runs: dict[str, Path]
) -> None:
    assert _aggregate(action_space_runs)["illegal_fp8"] >= 1


def test_suite_has_at_least_one_fusion_candidate(
    action_space_runs: dict[str, Path]
) -> None:
    assert _aggregate(action_space_runs)["fusion"] >= 1


def test_suite_has_at_least_one_extension_closure_candidate(
    action_space_runs: dict[str, Path]
) -> None:
    assert _aggregate(action_space_runs)["extension_closure"] >= 1


# --------------------------------------------------------------------------- #
# Target sensitivity (default host_cpu vs auto-discovered)
# --------------------------------------------------------------------------- #


def test_discovered_target_changes_priorities_or_costs(
    action_space_runs: dict[str, Path],
    discovered_target_runs: dict[str, Path],
) -> None:
    """Acceptance: an auto-discovered profile must change at least one
    decision-site priority OR one candidate cost preview vs the default
    host_cpu profile. Otherwise the action space isn't responding to
    target inputs."""
    sensitive = False
    for model_id in ACTION_SPACE_MODELS:
        a_run = action_space_runs[model_id]
        b_run = discovered_target_runs[model_id]
        a_sites = json.loads(
            (a_run / "02_graph_analysis" / "decision_sites.json").read_text()
        )
        b_sites = json.loads(
            (b_run / "02_graph_analysis" / "decision_sites.json").read_text()
        )
        a_priorities = {s["site_id"]: s["priority"] for s in a_sites["sites"]}
        b_priorities = {s["site_id"]: s["priority"] for s in b_sites["sites"]}
        for sid in a_priorities:
            if sid in b_priorities and a_priorities[sid] != b_priorities[sid]:
                sensitive = True
                break
        if sensitive:
            break
    assert sensitive, (
        "discovered profile produced identical priorities to host_cpu; "
        "action space is not target-sensitive"
    )


# --------------------------------------------------------------------------- #
# Compiler-core untouched (anti-coupling)
# --------------------------------------------------------------------------- #


def test_compiler_core_not_modified_by_m04() -> None:
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
    assert not changed, f"M-04 modified compiler core: {changed}"
