"""Acceptance tests Merlin-Dronet Strict-Gate Cleanup.

Verifies:

- Every model now emits a typed ``<model_id>_strict_gate_report.json``
  whose ``status`` is exactly ``pass`` or ``blocked`` — never a silent
  warning.
- For ``merlin_dronet`` specifically: the report is ``blocked`` with
  ``root_cause.category == "unsupported_op"`` and the diagnostic points
  at the silent-drop / unsupported-ops artifacts (the FX→Payload
  importer at ``compgen/ir/payload/import_fx.py`` cannot infer
  ``tensor_meta`` for conv2d / batch_norm / max_pool2d / relu).
- For clean models (``tiny_mlp``): ``status == "pass"`` with
  ``root_cause.category == "unknown"`` (no drops).
- For ``merlin_mlp_wide`` (5 drops): ``status == "pass"`` with
  ``root_cause.category == "lowering_accounting"`` — informational.
- The strict gate is **not** weakened: the underlying
  ``lowering_summary::status`` field is preserved (unchanged source of
  truth); the report sits alongside it.
- ``root_cause.category`` is always one of the allowed enum values.
- Evidence paths exist on disk.
- The evidence pack ingests the new fields
  (``strict_gate_report_status`` + ``strict_gate_root_cause`` columns).
- No compiler-core files modified (the report module imports nothing
  from ``compgen.ir`` / ``compgen.capture`` / ``compgen.pipeline``).
still pass (regression).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


_ALLOWED_ROOT_CAUSE_CATEGORIES: set[str] = {
    "graph_break", "unsupported_op", "lowering_accounting",
    "adapter_issue", "dynamic_shape", "external_dependency", "unknown",
}


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run(model: str, out_dir: Path, stop_after: str = "payload-lowering") -> int:
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out_dir),
        "--stop-after", stop_after,
        "--selection-mode", "greedy",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return res.returncode


# --------------------------------------------------------------------------- #
# Module-scope fixture: one run per model, reused across tests
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def dronet_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m161_dronet") / "run"
    _run("merlin_dronet", out)
    return out


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m161_clean") / "run"
    _run("tiny_mlp", out)
    return out


@pytest.fixture(scope="module")
def partial_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m161_partial") / "run"
    _run("merlin_mlp_wide", out)
    return out


# --------------------------------------------------------------------------- #
# Report exists for merlin_dronet
# --------------------------------------------------------------------------- #


def test_dronet_report_emitted(dronet_run: Path) -> None:
    rep = dronet_run / "01_payload_lowering" / "merlin_dronet_strict_gate_report.json"
    assert rep.exists()
    summ = dronet_run / "01_payload_lowering" / "merlin_dronet_strict_gate_summary.md"
    assert summ.exists()


def test_dronet_status_is_pass_or_blocked_never_silent(dronet_run: Path) -> None:
    rep = _read(
        dronet_run / "01_payload_lowering" / "merlin_dronet_strict_gate_report.json"
    )
    assert rep["status"] in ("pass", "blocked")
    # Specifically merlin_dronet should be blocked under the current
    # importer behavior.
    assert rep["status"] == "blocked"


def test_dronet_root_cause_is_typed_unsupported_op(dronet_run: Path) -> None:
    rep = _read(
        dronet_run / "01_payload_lowering" / "merlin_dronet_strict_gate_report.json"
    )
    rc = rep["root_cause"]
    assert rc["category"] in _ALLOWED_ROOT_CAUSE_CATEGORIES
    assert rc["category"] == "unsupported_op"
    assert rc["summary"]              # non-empty
    # Either points at unsupported_ops.json or silent_drop_audit.json.
    assert rc["source_artifact"] in (
        "01_payload_lowering/unsupported_ops.json",
        "01_payload_lowering/silent_drop_audit.json",
    )


def test_dronet_evidence_paths_exist(dronet_run: Path) -> None:
    rep = _read(
        dronet_run / "01_payload_lowering" / "merlin_dronet_strict_gate_report.json"
    )
    for key, rel in rep["evidence"].items():
        path = dronet_run / rel
        assert path.exists(), f"evidence[{key!r}] points at missing path {rel}"


def test_dronet_counts_match_silent_drop_audit(dronet_run: Path) -> None:
    rep = _read(
        dronet_run / "01_payload_lowering" / "merlin_dronet_strict_gate_report.json"
    )
    audit = _read(
        dronet_run / "01_payload_lowering" / "silent_drop_audit.json"
    )
    audit_drops = int(
        (audit.get("totals") or {}).get("dropped_auxiliary_output", 0)
    )
    assert rep["counts"]["dropped_auxiliary_output"] == audit_drops
    assert audit_drops > 0  # this is the case we're highlighting


# --------------------------------------------------------------------------- #
# Clean baseline + partial-success path
# --------------------------------------------------------------------------- #


def test_clean_model_passes_strict_gate(clean_run: Path) -> None:
    rep = _read(
        clean_run / "01_payload_lowering" / "tiny_mlp_strict_gate_report.json"
    )
    assert rep["status"] == "pass"
    assert rep["root_cause"]["category"] == "unknown"
    assert rep["counts"]["dropped_auxiliary_output"] == 0


def test_partial_success_passes_with_lowering_accounting_note(
    partial_run: Path,
) -> None:
    rep = _read(
        partial_run / "01_payload_lowering"
        / "merlin_mlp_wide_strict_gate_report.json"
    )
    # partial_success means downstream proceeds — strict gate passes.
    assert rep["status"] == "pass"
    # Drops > 0 are surfaced informationally.
    assert rep["counts"]["dropped_auxiliary_output"] > 0
    assert rep["root_cause"]["category"] == "lowering_accounting"


# --------------------------------------------------------------------------- #
# Strict gate is not globally weakened
# --------------------------------------------------------------------------- #


def test_underlying_lowering_summary_unchanged(dronet_run: Path) -> None:
    """The report must NOT mutate the existing
    ``lowering_summary.json`` source of truth. The pre-existing
    ``status="fail"`` field stays intact."""
    ls = _read(dronet_run / "01_payload_lowering" / "lowering_summary.json")
    assert ls["status"] == "fail"   # honest baseline preserved


def test_underlying_silent_drop_audit_unchanged(dronet_run: Path) -> None:
    audit = _read(dronet_run / "01_payload_lowering" / "silent_drop_audit.json")
    # Audit lists the dropped nodes — must still match what import_fx
    # actually dropped.
    assert int(
        (audit.get("totals") or {}).get("dropped_auxiliary_output", 0)
    ) > 0


def test_strict_gate_report_does_not_modify_payload_artifacts(
    dronet_run: Path,
) -> None:
    """Snapshot every payload_lowering file SHA before vs after a
    re-build of the strict-gate report. Nothing else should change."""
    import hashlib

    pl = dronet_run / "01_payload_lowering"

    def _shas() -> dict[str, str]:
        out: dict[str, str] = {}
        for p in sorted(pl.iterdir()):
            if not p.is_file():
                continue
            if p.name.startswith("merlin_dronet_strict_gate_"):
                continue   # the report files themselves are allowed to change
            out[p.name] = "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    before = _shas()

    from compgen.graph_compilation.strict_gate_report import (
        build_strict_gate_report,
    )
    build_strict_gate_report(dronet_run)

    after = _shas()
    assert before == after, (
        "strict_gate_report mutated payload_lowering source artifacts"
    )


# --------------------------------------------------------------------------- #
# Evidence pack ingests the new fields
# --------------------------------------------------------------------------- #


def test_evidence_pack_includes_strict_gate_status(
    dronet_run: Path, clean_run: Path, partial_run: Path, tmp_path: Path,
) -> None:
    """The evidence pack model matrix must include both
    ``strict_gate_report_status`` and ``strict_gate_root_cause`` columns,
    populated from the new report."""
    suite = tmp_path / "fixture_suite"
    canonical = suite / "canonical"
    wide = suite / "wide"
    canonical.mkdir(parents=True)
    wide.mkdir(parents=True)

    import shutil
    shutil.copytree(clean_run, canonical / "tiny_mlp")
    shutil.copytree(partial_run, canonical / "merlin_mlp_wide")
    shutil.copytree(dronet_run, wide / "merlin_dronet")

    pack_out = suite / "evidence_pack"

    from compgen.graph_compilation.evidence_pack import build_evidence_pack
    build_evidence_pack(
        canonical_suite_root=canonical, wide_suite_root=wide,
        out_dir=pack_out, skip_figures=True,
    )

    rows = list(csv.DictReader(
        (pack_out / "graph_section_model_matrix.csv").open(encoding="utf-8"),
    ))
    by_id = {r["model_id"]: r for r in rows}
    assert by_id["merlin_dronet"]["strict_gate_report_status"] == "blocked"
    assert by_id["merlin_dronet"]["strict_gate_root_cause"] == "unsupported_op"
    assert by_id["tiny_mlp"]["strict_gate_report_status"] == "pass"
    assert by_id["merlin_mlp_wide"]["strict_gate_report_status"] == "pass"

    agg = _read(pack_out / "graph_section_evidence_tables.json")
    assert agg["strict_gate_pass_count"] >= 2     # tiny_mlp + merlin_mlp_wide
    assert agg["strict_gate_blocked_count"] >= 1  # merlin_dronet
    assert "unsupported_op" in agg["strict_gate_root_causes"]


def test_evidence_pack_claim_matrix_records_strict_gate(
    dronet_run: Path, clean_run: Path, tmp_path: Path,
) -> None:
    suite = tmp_path / "claim_fixture"
    canonical = suite / "canonical"
    canonical.mkdir(parents=True)
    import shutil
    shutil.copytree(clean_run, canonical / "tiny_mlp")
    shutil.copytree(dronet_run, canonical / "merlin_dronet")

    from compgen.graph_compilation.evidence_pack import build_evidence_pack
    res = build_evidence_pack(
        canonical_suite_root=canonical, wide_suite_root=None,
        out_dir=suite / "evidence_pack", skip_figures=True,
    )
    cm = _read(res.claim_matrix)
    sg_claim = next(
        c for c in cm["claims"]
        if "Strict payload-lowering gate is typed" in c["claim"]
    )
    assert sg_claim["status"] == "implemented"
    obs = sg_claim["observed_metric"]
    assert obs["strict_gate_pass"] >= 1
    assert obs["strict_gate_blocked"] >= 1


# --------------------------------------------------------------------------- #
# No compiler-core changes
# --------------------------------------------------------------------------- #


def test_strict_gate_report_does_not_import_compiler_core() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "strict_gate_report.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir",
        "import compgen.ir",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
        "import compgen.pipeline",
    )
    for pat in forbidden:
        assert pat not in src, f"strict_gate_report imports forbidden module: {pat}"
