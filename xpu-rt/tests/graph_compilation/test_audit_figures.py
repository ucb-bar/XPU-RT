"""Acceptance tests for Trust Audit Figures + Summary.

Asserts the audit script reads the suite output and emits four PNGs +
``trust_audit_summary.md`` + ``trust_audit_tables.json`` with the
shape required by the acceptance checklist.

The script does not change compiler behavior; these tests only verify
that the reviewer-facing evidence pack is well-formed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

# Make the script importable without touching its argparse main.
SCRIPT_PATH = REPO_ROOT / "scripts" / "dev" / "render_graph_compilation_audit.py"


@pytest.fixture(scope="module")
def render_module() -> object:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "render_audit_module", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_audit_module"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit_artifacts(
    render_module: object, tmp_path_factory: pytest.TempPathFactory
) -> dict:
    """Run the canonical 6-model suite + render audit figures into a
    temp dir. Module-scoped so the heavy run is shared across tests."""
    from compgen.graph_compilation.run import run_graph_compilation
    base = tmp_path_factory.mktemp("audit")
    suite_root = base / "suite"
    for model_id in (
        "tiny_mlp",
        "tiny_attention",
        "tiny_conv_block",
        "proxy_vlm",
        "proxy_vla",
        "custom_unsupported_op",
    ):
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=suite_root / model_id,
            stop_after="recipe-verification",
            run_id=f"audit_{model_id}",
            selection_mode="greedy",
        )

    out = base / "audit_figures"
    artifacts = render_module.render_audit(suite_root, out)  # type: ignore[attr-defined]
    return {"suite": suite_root, "out": out, "artifacts": artifacts}


# --------------------------------------------------------------------------- #
# File presence + non-empty
# --------------------------------------------------------------------------- #


def test_all_six_artifacts_emitted(audit_artifacts: dict) -> None:
    out = audit_artifacts["out"]
    expected = [
        "payload_coverage_stacked_bar.png",
        "candidate_legality_heatmap.png",
        "region_roofline_scatter.png",
        "refinement_histogram.png",
        "trust_audit_summary.md",
        "trust_audit_tables.json",
    ]
    for name in expected:
        p = out / name
        assert p.exists(), f"missing {name}"
        assert p.stat().st_size > 0, f"empty {name}"


def test_pngs_have_png_magic(audit_artifacts: dict) -> None:
    out = audit_artifacts["out"]
    png_magic = b"\x89PNG\r\n\x1a\n"
    for name in (
        "payload_coverage_stacked_bar.png",
        "candidate_legality_heatmap.png",
        "region_roofline_scatter.png",
        "refinement_histogram.png",
    ):
        head = (out / name).read_bytes()[:8]
        assert head == png_magic, f"{name}: not a PNG"


# --------------------------------------------------------------------------- #
# trust_audit_tables.json structure
# --------------------------------------------------------------------------- #


def test_tables_json_schema(audit_artifacts: dict) -> None:
    out = audit_artifacts["out"]
    obj = json.loads((out / "trust_audit_tables.json").read_text())
    assert obj["schema_version"] == "trust_audit_tables_v1"
    assert "models" in obj and len(obj["models"]) == 6
    for key in (
        "aggregate", "diversity", "payload_coverage",
        "candidate_legality", "region_roofline", "refinement_declarations",
    ):
        assert key in obj, f"missing top-level key {key}"


def test_aggregate_consistent_with_suite(audit_artifacts: dict) -> None:
    """Cross-check aggregate numbers against the underlying JSON
    artifacts, so the audit script can't accidentally fabricate counts."""
    out = audit_artifacts["out"]
    suite = audit_artifacts["suite"]
    obj = json.loads((out / "trust_audit_tables.json").read_text())
    agg = obj["aggregate"]

    # Recompute the same numbers from the suite for the canonical 6 models.
    expected_call_function = 0
    expected_decomposed = 0
    for m in obj["models"]:
        s = json.loads(
            (suite / m / "01_payload_lowering" / "fx_to_payload_accounting.json").read_text()
        )["summary"]
        expected_call_function += s["call_function_nodes"]
        expected_decomposed += s["decomposed_structured"]
    assert agg["fx_call_function_total"] == expected_call_function
    assert agg["decomposed_with_payload_ops"] == expected_decomposed


def test_diversity_table_one_row_per_model(audit_artifacts: dict) -> None:
    obj = json.loads(
        (audit_artifacts["out"] / "trust_audit_tables.json").read_text()
    )
    assert len(obj["diversity"]) == len(obj["models"])
    for row in obj["diversity"]:
        assert row["model"]
        assert row["selected_candidate_id"]
        assert row["candidate_kind"]
        assert row["declared_refinement"]


def test_refinement_table_at_least_two_distinct_types(
    audit_artifacts: dict,
) -> None:
    """Acceptance: every checked recipe op is counted exactly once,
    and the suite produces at least two distinct refinement types
    (is not a constant pass)."""
    obj = json.loads(
        (audit_artifacts["out"] / "trust_audit_tables.json").read_text()
    )
    counts = obj["refinement_declarations"]["by_refinement"]
    per_model = obj["refinement_declarations"]["per_model"]
    assert sum(counts.values()) == len([
        r for r in per_model if r.get("recipe_op_id")
    ])
    assert len(counts) >= 2, counts


def test_candidate_legality_heatmap_has_three_families(
    audit_artifacts: dict,
) -> None:
    """Acceptance: at least three candidate families appear with
    non-zero counts somewhere in the suite."""
    obj = json.loads(
        (audit_artifacts["out"] / "trust_audit_tables.json").read_text()
    )
    rows = obj["candidate_legality"]["rows"]
    families: set[str] = set()
    for r in rows:
        for k, v in r.items():
            if k in ("model", "candidate_total"):
                continue
            if v > 0:
                # Strip the trailing _legal/_illegal to get the family name.
                fam = k.rsplit("_legal", 1)[0].rsplit("_illegal", 1)[0]
                families.add(fam)
    assert len(families) >= 3, families


def test_candidate_legality_has_illegal_fp8_and_legal_tile(
    audit_artifacts: dict,
) -> None:
    obj = json.loads(
        (audit_artifacts["out"] / "trust_audit_tables.json").read_text()
    )
    rows = obj["candidate_legality"]["rows"]
    illegal_fp8 = sum(r.get("quantize_fp8_illegal", 0) for r in rows)
    legal_tile = sum(r.get("set_tile_params_legal", 0) for r in rows)
    assert illegal_fp8 >= 1, illegal_fp8
    assert legal_tile >= 1, legal_tile


def test_region_roofline_has_compute_and_memory_bound_points(
    audit_artifacts: dict,
) -> None:
    """Roofline scatter must include both bottleneck classes — otherwise
    the dossier cost model collapsed to a single regime."""
    obj = json.loads(
        (audit_artifacts["out"] / "trust_audit_tables.json").read_text()
    )
    points = obj["region_roofline"]["points"]
    bottlenecks = {p["bottleneck"] for p in points}
    assert "compute" in bottlenecks, bottlenecks
    assert "memory" in bottlenecks, bottlenecks
    # Each point links back to model + region_id.
    for p in points:
        assert p["model"] and p["region_id"]
    # Excluded list is well-formed.
    excluded = obj["region_roofline"]["excluded"]
    for e in excluded:
        assert e["reason"] in {"opaque_fallback", "structural_only_no_compute"}


# --------------------------------------------------------------------------- #
# trust_audit_summary.md content
# --------------------------------------------------------------------------- #


def test_summary_markdown_has_all_required_sections(
    audit_artifacts: dict,
) -> None:
    text = (audit_artifacts["out"] / "trust_audit_summary.md").read_text()
    for section in (
        "## 1. Suite command",
        "## 2. Test count",
        "## 3. Per-stage pass/fail summary",
        "## 4. Cross-suite aggregate numbers",
        "## 5. Diversity table",
        "## 6. What this proves",
        "## 7. What this does NOT prove yet",
        "## 8. Figures",
    ):
        assert section in text, f"missing section: {section}"


def test_summary_records_caveats(audit_artifacts: dict) -> None:
    """The user explicitly required honest caveats. Verify each is present."""
    text = (audit_artifacts["out"] / "trust_audit_summary.md").read_text()
    for caveat in (
        "No Payload transform",
        "obligations are declared, not discharged",
        "planning estimate",
        "No real LLM call",
        "No benchmark, profiler, or kernel codegen",
    ):
        assert caveat in text, f"missing caveat: {caveat!r}"


def test_summary_lists_each_model_in_diversity(
    audit_artifacts: dict,
) -> None:
    text = (audit_artifacts["out"] / "trust_audit_summary.md").read_text()
    for m in (
        "tiny_mlp", "tiny_attention", "tiny_conv_block",
        "proxy_vlm", "proxy_vla", "custom_unsupported_op",
    ):
        assert m in text, f"missing model in summary: {m}"


# --------------------------------------------------------------------------- #
# Read-only against compiler core
# --------------------------------------------------------------------------- #


def test_audit_does_not_modify_compiler_core() -> None:
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
    assert not changed, f"M-06.5 modified compiler core: {changed}"
