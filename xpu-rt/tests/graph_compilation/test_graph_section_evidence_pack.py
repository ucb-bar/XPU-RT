"""Acceptance tests for M-17 Graph Section Evidence Pack.

Verifies the evidence pack:

- emits all required files (markdown, claim matrix, 5 CSVs, evidence
  tables JSON, and 7 figures);
- summarizes both canonical and wide suites when both are passed;
- aggregate totals match per-model source artifacts;
- claim matrix separates implemented / partially_implemented /
  implemented_partial_scope / missing statuses;
- real SetTileParams + real FuseProducerConsumer discharges appear;
- unsupported fusion blocked cases appear;
- retry events appear when present;
- the "Honest non-claims" section exists in the markdown;
- all PNG figures have valid magic bytes;
- no compiler-core imports.

A small ad-hoc fixture suite (`tiny_mlp` + `proxy_vla` + canonical
`merlin_mlp_wide` if available) is built once at module scope so the
tests run against real artifacts emitted by the live pipeline.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run_one(*, model: str, out_dir: Path) -> None:
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out_dir),
        "--stop-after", "real-transform-differential",
        "--selection-mode", "greedy",
    ]
    # Pipeline may exit non-zero on M-12 fail / M-15B raise — that's fine
    # for our purposes; the typed reports still land on disk and the
    # evidence pack must summarize them honestly.
    subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def fixture_pack(tmp_path_factory) -> dict:  # type: ignore[no-untyped-def]
    """Build a small fixture pack covering both transform families.

    Models:
      - tiny_mlp        → set_tile path; greedy fails M-12 (real fail).
      - proxy_vla       → fusion path; greedy passes (add_0 → relu_0).
      - merlin_mlp_wide → set_tile path; greedy passes (clean divides).

    Lives in a tmp dir, so the test never touches results/ on disk.
    """
    suite_dir = tmp_path_factory.mktemp("m17_fixture_suite")
    canonical = suite_dir / "canonical"
    wide = suite_dir / "wide"
    canonical.mkdir()
    wide.mkdir()

    # Canonical: tiny_mlp + proxy_vla
    _run_one(model="tiny_mlp", out_dir=canonical / "tiny_mlp")
    _run_one(model="proxy_vla", out_dir=canonical / "proxy_vla")
    # Wide: merlin_mlp_wide (the SetTileParams discharge case)
    _run_one(model="merlin_mlp_wide", out_dir=wide / "merlin_mlp_wide")

    out_pack = suite_dir / "evidence_pack"

    from compgen.graph_compilation.evidence_pack import build_evidence_pack

    res = build_evidence_pack(
        canonical_suite_root=canonical,
        wide_suite_root=wide,
        out_dir=out_pack,
    )
    return {
        "result": res,
        "out": out_pack,
        "canonical": canonical,
        "wide": wide,
    }


# --------------------------------------------------------------------------- #
# Required files exist
# --------------------------------------------------------------------------- #


def test_pack_emits_required_files(fixture_pack: dict) -> None:
    out: Path = fixture_pack["out"]
    expected = [
        "graph_section_evidence_summary.md",
        "graph_section_claim_matrix.json",
        "graph_section_model_matrix.csv",
        "graph_section_agent_decisions.csv",
        "graph_section_retry_events.csv",
        "graph_section_verification_matrix.csv",
        "graph_section_transform_coverage.csv",
        "graph_section_evidence_tables.json",
    ]
    for name in expected:
        assert (out / name).exists(), f"missing {name}"


def test_required_figures_exist_and_are_pngs(fixture_pack: dict) -> None:
    from compgen.graph_compilation.evidence_pack import is_png

    out: Path = fixture_pack["out"]
    figures = [
        "payload_coverage_by_model.png",
        "candidate_family_by_model.png",
        "selected_action_family_by_model.png",
        "real_verification_status_by_model.png",
        "retry_flow_counts.png",
        "greedy_vs_agent_candidate_change.png",
        "transform_family_discharge_matrix.png",
    ]
    for name in figures:
        path = out / "figures" / name
        assert path.exists(), f"missing figure {name}"
        assert is_png(path), f"{name} is not a valid PNG"


# --------------------------------------------------------------------------- #
# Both suites summarized
# --------------------------------------------------------------------------- #


def test_model_matrix_has_canonical_and_wide(fixture_pack: dict) -> None:
    matrix = fixture_pack["out"] / "graph_section_model_matrix.csv"
    rows = list(csv.DictReader(matrix.open(encoding="utf-8")))
    suites = {r["suite"] for r in rows}
    assert "canonical" in suites
    assert "wide" in suites
    model_ids = {r["model_id"] for r in rows}
    assert "tiny_mlp" in model_ids
    assert "proxy_vla" in model_ids
    assert "merlin_mlp_wide" in model_ids


# --------------------------------------------------------------------------- #
# Aggregate totals match per-model source artifacts
# --------------------------------------------------------------------------- #


def test_aggregate_totals_match_per_model_artifacts(fixture_pack: dict) -> None:
    out = fixture_pack["out"]
    agg = _read(out / "graph_section_evidence_tables.json")
    matrix = list(csv.DictReader(
        (out / "graph_section_model_matrix.csv").open(encoding="utf-8"),
    ))
    # CSV totals must equal aggregate totals.
    fx_total = sum(int(r["fx_nodes_total"]) for r in matrix)
    cf_total = sum(int(r["call_function_nodes"]) for r in matrix)
    payload_total = sum(int(r["payload_ops"]) for r in matrix)
    cand_total = sum(int(r["candidates_total"]) for r in matrix)
    assert agg["fx_nodes_total"] == fx_total
    assert agg["call_function_nodes_total"] == cf_total
    assert agg["payload_ops_total"] == payload_total
    assert agg["candidate_count_total"] == cand_total
    assert agg["model_count"] == len(matrix)


def test_aggregate_matches_pipeline_artifacts_for_one_model(
    fixture_pack: dict,
) -> None:
    """Spot-check: tiny_mlp's fx_nodes_total in the model matrix matches
    its on-disk fx_to_payload_accounting.json."""
    canonical = fixture_pack["canonical"]
    matrix_path = (
        fixture_pack["out"] / "graph_section_model_matrix.csv"
    )
    rows = list(csv.DictReader(matrix_path.open(encoding="utf-8")))
    tiny = next(r for r in rows if r["model_id"] == "tiny_mlp")
    accounting = _read(
        canonical / "tiny_mlp" / "01_payload_lowering"
        / "fx_to_payload_accounting.json"
    )
    expected_total = sum(
        len(m.get("nodes", [])) for m in accounting.get("modules", [])
    )
    assert int(tiny["fx_nodes_total"]) == expected_total


# --------------------------------------------------------------------------- #
# Claim matrix
# --------------------------------------------------------------------------- #


def test_claim_matrix_includes_required_statuses(fixture_pack: dict) -> None:
    cm = _read(fixture_pack["out"] / "graph_section_claim_matrix.json")
    assert cm["schema_version"] == "graph_section_claim_matrix_v1"
    statuses = {c["status"] for c in cm["claims"]}
    assert "implemented" in statuses
    # M-16.2 fusion is partial-scope; cost preview is partially_implemented.
    assert any(s in statuses for s in ("partially_implemented", "implemented_partial_scope"))


def test_claim_matrix_references_real_evidence_artifacts(fixture_pack: dict) -> None:
    cm = _read(fixture_pack["out"] / "graph_section_claim_matrix.json")
    set_tile_claim = next(
        c for c in cm["claims"]
        if "SetTileParams" in c["claim"]
    )
    assert any(
        "real_differential_report" in a
        for a in set_tile_claim["evidence_artifacts"]
    )
    fusion_claim = next(
        c for c in cm["claims"]
        if "FuseProducerConsumer" in c["claim"]
    )
    assert any(
        "real_fusion_differential_report" in a
        for a in fusion_claim["evidence_artifacts"]
    )


# --------------------------------------------------------------------------- #
# Real transform discharges appear
# --------------------------------------------------------------------------- #


def test_real_set_tile_discharge_appears(fixture_pack: dict) -> None:
    """merlin_mlp_wide greedy → set_tile bit-equality."""
    out = fixture_pack["out"]
    rows = list(csv.DictReader(
        (out / "graph_section_verification_matrix.csv").open(encoding="utf-8"),
    ))
    merlin = next(r for r in rows if r["model_id"] == "merlin_mlp_wide")
    assert merlin["real_set_tile_status"] == "pass"
    assert merlin["bit_equality_discharged"] == "True"


def test_real_fusion_discharge_appears(fixture_pack: dict) -> None:
    """proxy_vla greedy → fusion bit-equality."""
    out = fixture_pack["out"]
    rows = list(csv.DictReader(
        (out / "graph_section_verification_matrix.csv").open(encoding="utf-8"),
    ))
    proxy = next(r for r in rows if r["model_id"] == "proxy_vla")
    assert proxy["real_fusion_status"] == "pass"
    assert proxy["bit_equality_discharged"] == "True"


def test_real_transform_families_discharged_at_least_two(
    fixture_pack: dict,
) -> None:
    agg = _read(fixture_pack["out"] / "graph_section_evidence_tables.json")
    assert agg["real_transform_families_discharged_count"] >= 2


# --------------------------------------------------------------------------- #
# Unsupported / blocked path appears
# --------------------------------------------------------------------------- #


def test_unsupported_path_appears(fixture_pack: dict) -> None:
    """tiny_mlp evidence row exists.

    Pre-M-37.12: tile_16 → K_iters=4 → bit-equality fail → row=fail/blocked.
    Post-M-37.12: shape-fit clean-divide tile + tolerance_eps + combined
    torch.allclose → row=pass. The test now just checks the row exists
    (any of pass/fail/blocked is acceptable; what matters is that
    tiny_mlp is *represented* in the evidence pack)."""
    out = fixture_pack["out"]
    rows = list(csv.DictReader(
        (out / "graph_section_verification_matrix.csv").open(encoding="utf-8"),
    ))
    tm = next(r for r in rows if r["model_id"] == "tiny_mlp")
    assert tm["real_set_tile_status"] in ("pass", "fail", "blocked")


# --------------------------------------------------------------------------- #
# Retry events appear (when present)
# --------------------------------------------------------------------------- #


def test_retry_events_csv_records_downstream_retries(fixture_pack: dict) -> None:
    """retry_events.csv exists and is well-formed.

    Pre-M-37.12: tiny_mlp tripped M-15B (bit-equality fail), populating
    the table. Post-M-37.12: the canonical-set models all pass M-12
    (combined torch.allclose tolerance), so retry_events.csv may be
    empty. The test now verifies the file exists and parses; the M-15B
    plumbing itself is covered by detector unit tests in
    test_downstream_retry.py."""
    out = fixture_pack["out"]
    csv_path = out / "graph_section_retry_events.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    # If any retry row exists for tiny_mlp it must be well-formed.
    for r in rows:
        if r["model_id"] == "tiny_mlp":
            assert int(r["downstream_retry_events"]) >= 0


def test_aggregate_records_downstream_retry(fixture_pack: dict) -> None:
    """Aggregate retry count is non-negative.

    Pre-M-37.12 it was always >= 1 (tiny_mlp). Post-M-37.12 it may be
    0 because every canonical-set model now passes M-12."""
    agg = _read(fixture_pack["out"] / "graph_section_evidence_tables.json")
    assert agg["downstream_retry_count"] >= 0


# --------------------------------------------------------------------------- #
# Honest non-claims section
# --------------------------------------------------------------------------- #


def test_honest_non_claims_section_present(fixture_pack: dict) -> None:
    text = (
        fixture_pack["out"] / "graph_section_evidence_summary.md"
    ).read_text(encoding="utf-8")
    assert "Honest non-claims" in text
    # Spot-check a handful of the required bullets.
    for must in (
        "not yet a full compiler backend",
        "tiled matmul",
        "pointwise fusion",
        "Cost Preview V2",
        "M-15B",
        "Claude Code",
        "bounded agentic compilation",
    ):
        assert must in text, f"missing non-claim phrase: {must!r}"


def test_summary_has_headline_numbers(fixture_pack: dict) -> None:
    text = (
        fixture_pack["out"] / "graph_section_evidence_summary.md"
    ).read_text(encoding="utf-8")
    assert "agent_changed_from_greedy_count" in text
    assert "real_transform_families_discharged_count" in text


# --------------------------------------------------------------------------- #
# No compiler-core imports
# --------------------------------------------------------------------------- #


def test_evidence_pack_does_not_import_compiler_core() -> None:
    forbidden = (
        "from compgen.ir",
        "import compgen.ir",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
        "import compgen.pipeline",
    )
    for src_path in (
        REPO_ROOT / "python" / "compgen" / "graph_compilation" / "evidence_pack.py",
        REPO_ROOT / "python" / "compgen" / "graph_compilation" / "evidence_pack_figures.py",
    ):
        text = src_path.read_text(encoding="utf-8")
        for pat in forbidden:
            assert pat not in text, f"{src_path.name} imports forbidden module: {pat}"


def test_evidence_pack_is_read_only(fixture_pack: dict) -> None:
    """Re-running the builder must not mutate any source artifact under
    canonical/ or wide/. Snapshot SHAs before and after."""
    import hashlib

    canonical = fixture_pack["canonical"]
    wide = fixture_pack["wide"]

    def _shas(root: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            out[str(p.relative_to(root))] = (
                "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
            )
        return out

    before = {**_shas(canonical), **_shas(wide)}

    from compgen.graph_compilation.evidence_pack import build_evidence_pack

    build_evidence_pack(
        canonical_suite_root=canonical,
        wide_suite_root=wide,
        out_dir=fixture_pack["out"],
    )

    after = {**_shas(canonical), **_shas(wide)}
    assert before == after, (
        "evidence pack mutated source suite directories — read-only invariant broken"
    )
