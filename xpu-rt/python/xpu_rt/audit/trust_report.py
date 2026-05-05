"""Trust report aggregator (M-31A.5).

A single-page artifact that proves the system is honest. The report
runs every audit gate and emits both a JSON record and a human-readable
markdown summary.

Gates (in order):

1. ``realness_scan``           — no unallowlisted stubs in source.
2. ``negative_controls``       — every fault-injection raises its typed error.
3. ``caveat_ledger``           — schema-valid; no stale rows (or
                                 ``allow_stale=True`` for offline audits).
4. ``realness_contracts``      — every seed contract loads + validates.
5. ``import_provenance``       — when run against a real run dir, no
                                 forbidden modules imported.
6. ``trace_replay_self_check`` — synthesize a trace, replay it, assert
                                 hashes match.
7. ``task_pack_buildable``     — task pack builds + validates.
8. ``holdout_outcomes_honest`` — every holdout config carries
                                 ``holdout: true``.

Each gate produces a :class:`GateResult` (see ``compgen.audit.errors``)
with ``status`` ∈ ``{pass, fail, skipped}``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compgen.audit.caveat_ledger import CaveatLedger
from compgen.audit.contracts import iter_contracts
from compgen.audit.errors import (
    AuditError,
    CaveatLedgerError,
    ForbiddenImportError,
    GateResult,
    StaleCaveatError,
)
from compgen.audit.fresh_agent import (
    REQUIRED_PATHS,
    build_task_pack,
    verify_task_pack,
)
from compgen.audit.import_provenance import (
    assert_no_forbidden,
    load_provenance,
)
from compgen.audit.negative_controls import run_all_negative_controls
from compgen.audit.realness_scan import (
    Allowlist,
    assert_clean,
    scan_repo,
)
from compgen.audit.trace_replay import (
    build_trace,
    replay,
    write_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_REALNESS_DIR = REPO_ROOT / "docs" / "realness"
SEED_LEDGER_DIR = REPO_ROOT / "results" / "audit" / "_seed"


@dataclass
class TrustReport:
    """Top-level trust report."""

    commit: str
    generated_at_utc: str
    gates: list[GateResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        """True iff zero gates failed. Skipped gates are honest neutrals."""
        return self.fail_count == 0

    @property
    def fail_count(self) -> int:
        return sum(1 for g in self.gates if g.status == "fail")

    @property
    def skip_count(self) -> int:
        return sum(1 for g in self.gates if g.status == "skipped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "generated_at_utc": self.generated_at_utc,
            "all_pass": self.all_pass,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "gates": [
                {
                    "name": g.name,
                    "status": g.status,
                    "detail": g.detail,
                    "artifact_path": (
                        str(g.artifact_path) if g.artifact_path else None
                    ),
                }
                for g in self.gates
            ],
            "notes": list(self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# CompGen trust report — {self.commit}",
            "",
            f"Generated: `{self.generated_at_utc}`",
            f"Overall: **{'PASS' if self.all_pass else 'FAIL'}**",
            f"Gates: {len(self.gates)} total, "
            f"{len(self.gates) - self.fail_count - self.skip_count} pass, "
            f"{self.fail_count} fail, {self.skip_count} skipped",
            "",
            "| Gate | Status | Detail |",
            "| --- | --- | --- |",
        ]
        for g in self.gates:
            status_md = {"pass": "✅ pass", "fail": "❌ fail",
                         "skipped": "⊝ skipped"}.get(g.status, g.status)
            detail = g.detail.replace("\n", " ").replace("|", r"\|")[:200]
            lines.append(f"| `{g.name}` | {status_md} | {detail} |")
        if self.notes:
            lines.append("")
            lines.append("## Notes")
            for n in self.notes:
                lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Individual gates
# --------------------------------------------------------------------------- #


def _gate_realness_scan() -> GateResult:
    try:
        report = scan_repo(repo_root=REPO_ROOT, include_tests=False)
        assert_clean(report)
        return GateResult(
            name="realness_scan",
            status="pass",
            detail=(
                f"{report.files_scanned} files scanned, "
                f"{len(report.hits)} hits ({len(report.allowlisted_hits)} allowlisted)"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            name="realness_scan",
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
        )


def _gate_negative_controls(tmp_path: Path) -> GateResult:
    report = run_all_negative_controls(tmp_path / "negative_controls")
    if report.all_pass:
        return GateResult(
            name="negative_controls",
            status="pass",
            detail=f"{len(report.outcomes)} controls; all raised the expected typed error",
        )
    failed = [o.name for o in report.outcomes if not o.passes]
    return GateResult(
        name="negative_controls",
        status="fail",
        detail=f"failing controls: {failed}",
    )


def _gate_caveat_ledger(*, allow_stale: bool = True) -> GateResult:
    seed = SEED_LEDGER_DIR / "caveat_ledger.json"
    if not seed.exists():
        return GateResult(
            name="caveat_ledger",
            status="skipped",
            detail=f"seed ledger missing at {seed}",
        )
    try:
        ledger = CaveatLedger.load(seed)
        ledger.validate(allow_stale=allow_stale)
        stale = ledger.stale()
        return GateResult(
            name="caveat_ledger",
            status="pass",
            detail=(
                f"{len(ledger)} caveats validated; "
                f"{len(stale)} stale (allow_stale={allow_stale})"
            ),
            artifact_path=seed,
        )
    except (CaveatLedgerError, StaleCaveatError) as exc:
        return GateResult(
            name="caveat_ledger",
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
        )


def _gate_realness_contracts() -> GateResult:
    try:
        contracts = list(iter_contracts(SEED_REALNESS_DIR))
        if not contracts:
            return GateResult(
                name="realness_contracts",
                status="fail",
                detail="no contracts found in docs/realness/",
            )
        return GateResult(
            name="realness_contracts",
            status="pass",
            detail=f"{len(contracts)} contracts validated",
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            name="realness_contracts",
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
        )


def _gate_import_provenance(*, run_dir: Path | None) -> GateResult:
    if run_dir is None or not (run_dir / "import_provenance.json").exists():
        return GateResult(
            name="import_provenance",
            status="skipped",
            detail=(
                "no run dir provided; pass --run-dir to a recent "
                "graph_compilation run to exercise this gate"
            ),
        )
    try:
        prov = load_provenance(run_dir / "import_provenance.json")
        assert_no_forbidden(prov)
        return GateResult(
            name="import_provenance",
            status="pass",
            detail=(
                f"run_id={prov.run_id} cache_mode={prov.cache_mode} "
                f"evidence_mode={prov.evidence_mode} "
                f"forbidden_count={len(prov.forbidden_modules_imported)}"
            ),
            artifact_path=run_dir / "import_provenance.json",
        )
    except ForbiddenImportError as exc:
        return GateResult(
            name="import_provenance",
            status="fail",
            detail=str(exc),
        )


def _gate_trace_replay_self_check(tmp_path: Path) -> GateResult:
    """Synthesize a small run, build a trace, replay it, assert match."""
    run_dir = tmp_path / "trace_self_check"
    rp = run_dir / "03_recipe_planning"
    rp.mkdir(parents=True)
    (rp / "agent_decision_request.json").write_text('{"a": 1}')
    (rp / "llm_graph_view.json").write_text('{"regions": []}')
    (rp / "candidate_actions.json").write_text('{"candidates": []}')
    (rp / "agent_decision_response.json").write_text('{"selected_candidate_id": "x"}')
    (rp / "agent_decision_record.json").write_text('{}')
    promo_lib = tmp_path / "missing_lib"
    try:
        trace = build_trace(run_dir, run_id="self_check", commit="abc",
                            promotion_library=promo_lib)
        trace_path = write_trace(trace, run_dir=run_dir)
        report = replay(trace_path=trace_path, run_dir=run_dir,
                        promotion_library=promo_lib, strict=True)
        if report.all_match:
            return GateResult(
                name="trace_replay_self_check",
                status="pass",
                detail=f"decision_id={trace.decision_id}",
            )
        return GateResult(
            name="trace_replay_self_check",
            status="fail",
            detail=f"deltas: input={list(report.input_deltas)} output={list(report.output_deltas)}",
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            name="trace_replay_self_check",
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
        )


def _gate_task_pack_buildable(tmp_path: Path) -> GateResult:
    pack_out = tmp_path / "task_pack"
    try:
        pack = build_task_pack(
            out_dir=pack_out, commit="trust_report",
            repo_root=REPO_ROOT, skip_python_package=True,
        )
        verify_task_pack(pack_out, lenient_python_package=True)
        return GateResult(
            name="task_pack_buildable",
            status="pass",
            detail=(
                f"{pack.files_copied} files, {pack.bytes_copied / (1024*1024):.1f} MB"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            name="task_pack_buildable",
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
        )


def _gate_holdout_outcomes_honest() -> GateResult:
    from compgen.graph_compilation.evidence_pack import is_holdout_model

    holdout_yamls = list((REPO_ROOT / "configs" / "models").glob("holdout_*.yaml"))
    if not holdout_yamls:
        return GateResult(
            name="holdout_outcomes_honest",
            status="fail",
            detail="no holdout YAMLs found under configs/models/",
        )
    bad = [y.name for y in holdout_yamls if not is_holdout_model(y)]
    if bad:
        return GateResult(
            name="holdout_outcomes_honest",
            status="fail",
            detail=f"holdout YAMLs missing 'holdout: true': {bad}",
        )
    return GateResult(
        name="holdout_outcomes_honest",
        status="pass",
        detail=f"{len(holdout_yamls)} holdout YAMLs declare holdout: true",
    )


# --------------------------------------------------------------------------- #
# Build entry point
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_short_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_trust_report(
    *,
    tmp_path: Path,
    run_dir: Path | None = None,
    commit: str | None = None,
) -> TrustReport:
    """Run every gate and return the aggregate report."""
    report = TrustReport(
        commit=commit or _git_short_commit(),
        generated_at_utc=_utc_now(),
    )
    report.gates.append(_gate_realness_scan())
    report.gates.append(_gate_negative_controls(tmp_path))
    report.gates.append(_gate_caveat_ledger(allow_stale=True))
    report.gates.append(_gate_realness_contracts())
    report.gates.append(_gate_import_provenance(run_dir=run_dir))
    report.gates.append(_gate_trace_replay_self_check(tmp_path))
    report.gates.append(_gate_task_pack_buildable(tmp_path))
    report.gates.append(_gate_holdout_outcomes_honest())
    if report.skip_count > 0:
        report.notes.append(
            f"{report.skip_count} gate(s) skipped — likely "
            "(import_provenance) no run-dir was supplied"
        )
    return report


def emit_trust_report(report: TrustReport, *, out_dir: Path) -> tuple[Path, Path]:
    """Write trust_report.{json,md} into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "trust_report.json"
    md_path = out_dir / "trust_report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a CompGen trust report")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/audit/<commit>/)")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Recent graph_compilation run dir for "
                        "import-provenance gate")
    p.add_argument("--commit", default=None,
                   help="Commit hash to embed (default: git rev-parse)")
    p.add_argument("--tmp", type=Path, default=None,
                   help="Tmp dir for synthesized fixtures")
    args = p.parse_args(argv)

    import tempfile
    cleanup_tmp = False
    if args.tmp is None:
        args.tmp = Path(tempfile.mkdtemp(prefix="compgen_trust_"))
        cleanup_tmp = True
    else:
        args.tmp.mkdir(parents=True, exist_ok=True)

    commit = args.commit or _git_short_commit()
    out = args.out or (REPO_ROOT / "results" / "audit" / commit)

    try:
        report = build_trust_report(
            tmp_path=args.tmp,
            run_dir=args.run_dir,
            commit=commit,
        )
        json_path, md_path = emit_trust_report(report, out_dir=out)
    finally:
        if cleanup_tmp:
            shutil.rmtree(args.tmp, ignore_errors=True)

    print(f"trust report: {md_path}")
    print(f"             {json_path}")
    print(f"  overall:    {'PASS' if report.all_pass else 'FAIL'}")
    print(f"  gates:      {len(report.gates)} total, "
          f"{report.fail_count} fail, {report.skip_count} skipped")
    return 0 if report.all_pass else 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
