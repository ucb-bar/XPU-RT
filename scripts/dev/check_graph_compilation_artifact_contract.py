#!/usr/bin/env python3
"""Black-box acceptance test for graph_compilation artifact contract.

Constructs a valid graph compilation run directory using ONLY the Python standard
library (json, hashlib, pathlib, shutil) — no imports from
``compgen.graph_compilation`` and no imports from
``tests/graph_compilation/_synth.py``. This is deliberate: the validator
must not be tested only with the same helpers that produce its
fixtures, otherwise a coordinated bug in the writer + validator would
go undetected.

The script then materialises 15 tampered variants and invokes the
validator CLI as a subprocess. For each variant it asserts:

- the expected exit code (0 / 1 / 2)
- the validation_report.json names the expected failed rule

Usage::

    python scripts/dev/check_graph_compilation_artifact_contract.py --out /tmp/graph_compilation_review

Exit codes:

- 0 — every case behaved as expected.
- 1 — at least one unexpected pass (a tampered run validated cleanly)
      or unexpected failure (a valid run was rejected, or the wrong
      rule was named).

The script prints a JSON summary on stdout::

    {
      "valid_cases_passed": 1,
      "tampered_cases_rejected": 15,
      "unexpected_passes": 0,
      "unexpected_failures": 0
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


# --------------------------------------------------------------------------- #
# Independent run-directory builder (NO imports from compgen.graph_compilation)
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(root: Path) -> str:
    """Independently re-implement the documented tree-hash contract."""
    files: list[tuple[str, Path]] = []
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            files.append((p.relative_to(root).as_posix(), p))
    files.sort(key=lambda t: t[0])
    h = hashlib.sha256()
    for rel, fpath in files:
        h.update(f"{rel}\0{_sha256_file(fpath)}\n".encode())
    return h.hexdigest()


def _write_file(path: Path, content: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": "",  # caller fills
        "sha256": _sha256_file(path),
        "size_bytes": len(content),
        "kind": "file",
    }


def build_valid_run(run_dir: Path) -> None:
    """Build a 3-stage well-formed run from scratch."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- stage0 ---
    capture = run_dir / "00_graph_capture"
    capture.mkdir()
    payload0 = b"\x80\x02blackbox-exported-program"
    s0_artifact = _write_file(capture / "exported_program.pt2", payload0)
    s0_artifact["path"] = "00_graph_capture/exported_program.pt2"
    (capture / "capture_report.json").write_text(
        json.dumps({"schema_version": "graph_capture_report_v1", "status": "pass"}),
        encoding="utf-8",
    )
    s0_input_hash = hashlib.sha256(b"blackbox-stage0-seed").hexdigest()
    s0_output_hash = _sha256_tree(capture)

    # --- stage1 ---
    lower = run_dir / "01_payload_lowering"
    lower.mkdir()
    s1_artifact = _write_file(lower / "payload.mlir", b"module { func.func @main() { return } }\n")
    s1_artifact["path"] = "01_payload_lowering/payload.mlir"
    (lower / "lowering_report.json").write_text(
        json.dumps({"schema_version": "payload_lowering_report_v1", "status": "pass"}),
        encoding="utf-8",
    )
    s1_input_hash = s0_output_hash
    s1_output_hash = _sha256_tree(lower)

    # --- stage2 ---
    analyze = run_dir / "02_gap_discovery"
    analyze.mkdir()
    s2_artifact = _write_file(
        analyze / "gap_analysis.json",
        json.dumps({"schema_version": "gap_analysis_v1", "regions_total": 1, "ops": []}).encode(),
    )
    s2_artifact["path"] = "02_gap_discovery/gap_analysis.json"
    (analyze / "gap_report.json").write_text(
        json.dumps({"schema_version": "gap_discovery_summary_v1", "status": "pass"}),
        encoding="utf-8",
    )
    s2_input_hash = s1_output_hash
    s2_output_hash = _sha256_tree(analyze)

    def stage_record(stage_id: str, outputs, report_path, input_hash, output_hash, inputs=None):
        return {
            "stage_id": stage_id,
            "status": "pass",
            "inputs": inputs or [],
            "outputs": outputs,
            "report_path": report_path,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "llm_calls": 0,
            "started_at_utc": "2026-04-30T00:00:00Z",
            "finished_at_utc": "2026-04-30T00:00:01Z",
        }

    s0_artifact_input_for_s1 = {
        "path": "00_graph_capture/exported_program.pt2",
        "sha256": s0_artifact["sha256"],
        "size_bytes": s0_artifact["size_bytes"],
        "kind": "file",
    }
    s1_artifact_input_for_s2 = {
        "path": "01_payload_lowering/payload.mlir",
        "sha256": s1_artifact["sha256"],
        "size_bytes": s1_artifact["size_bytes"],
        "kind": "file",
    }

    manifest = {
        "schema_version": "run_manifest_v1",
        "run_id": "blackbox_run_001",
        "created_at_utc": "2026-04-30T00:00:00Z",
        "git_commit": "b" * 40,
        "model": {
            "config_path": "configs/models/blackbox.yaml",
            "model_id": "blackbox_tiny",
            "config_sha256": "0" * 64,
        },
        "target": {
            "config_path": "configs/targets/host_cpu.yaml",
            "target_id": "host_cpu",
            "config_sha256": "0" * 64,
        },
        "seed": 0,
        "stages": [
            stage_record("graph_capture", [s0_artifact], "00_graph_capture/capture_report.json", s0_input_hash, s0_output_hash),
            stage_record(
                "payload_lowering",
                [s1_artifact],
                "01_payload_lowering/lowering_report.json",
                s1_input_hash,
                s1_output_hash,
                inputs=[s0_artifact_input_for_s1],
            ),
            stage_record(
                "gap_discovery",
                [s2_artifact],
                "02_gap_discovery/gap_report.json",
                s2_input_hash,
                s2_output_hash,
                inputs=[s1_artifact_input_for_s2],
            ),
        ],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    ledger_lines = []
    for sid in ("graph_capture", "payload_lowering", "gap_discovery"):
        for ev in ("start", "finish"):
            ledger_lines.append(
                json.dumps(
                    {
                        "schema_version": "stage_event_v1",
                        "stage_id": sid,
                        "event": ev,
                        "artifact_path": None,
                        "sha256": None,
                        "timestamp_utc": "2026-04-30T00:00:00Z",
                        "note": None,
                    }
                )
            )
    (run_dir / "stage_ledger.jsonl").write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Tamper recipes
# --------------------------------------------------------------------------- #


def _mutate_manifest(run_dir: Path, fn: Callable[[dict], None]) -> None:
    path = run_dir / "run_manifest.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    fn(obj)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def tamper_remove_manifest(run_dir: Path) -> None:
    (run_dir / "run_manifest.json").unlink()


def tamper_malformed_manifest(run_dir: Path) -> None:
    (run_dir / "run_manifest.json").write_text("{not valid json", encoding="utf-8")


def tamper_remove_ledger(run_dir: Path) -> None:
    (run_dir / "stage_ledger.jsonl").unlink()


def tamper_malformed_ledger(run_dir: Path) -> None:
    path = run_dir / "stage_ledger.jsonl"
    text = path.read_text(encoding="utf-8") + "this is not json\n"
    path.write_text(text, encoding="utf-8")


def tamper_delete_artifact(run_dir: Path) -> None:
    (run_dir / "00_graph_capture" / "exported_program.pt2").unlink()


def tamper_byte_flip_artifact(run_dir: Path) -> None:
    target = run_dir / "00_graph_capture" / "exported_program.pt2"
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))


def tamper_pass_with_no_outputs(run_dir: Path) -> None:
    def fn(obj):
        obj["stages"][0]["outputs"] = []

    _mutate_manifest(run_dir, fn)


def tamper_stage_order(run_dir: Path) -> None:
    def fn(obj):
        obj["stages"][0], obj["stages"][1] = obj["stages"][1], obj["stages"][0]

    _mutate_manifest(run_dir, fn)


def tamper_duplicate_stage_id(run_dir: Path) -> None:
    def fn(obj):
        obj["stages"].insert(1, obj["stages"][0])
        # Fix up the duplicate's input_hash so R009 isn't the first to fail.
        obj["stages"][1]["input_hash"] = obj["stages"][0]["output_hash"]

    _mutate_manifest(run_dir, fn)


def tamper_hash_chain_break(run_dir: Path) -> None:
    def fn(obj):
        obj["stages"][1]["input_hash"] = "f" * 64

    _mutate_manifest(run_dir, fn)


def tamper_missing_report(run_dir: Path) -> None:
    (run_dir / "00_graph_capture" / "capture_report.json").unlink()


def tamper_path_traversal(run_dir: Path) -> None:
    def fn(obj):
        obj["stages"][0]["outputs"][0]["path"] = "../etc/passwd"

    _mutate_manifest(run_dir, fn)


def tamper_symlink_escape(run_dir: Path) -> None:
    outside = run_dir.parent / "blackbox_outside_payload.mlir"
    outside.write_bytes(b"attacker-controlled\n")
    target = run_dir / "01_payload_lowering" / "payload.mlir"
    target.unlink()
    target.symlink_to(outside)


def tamper_llm_calls(run_dir: Path) -> None:
    def fn(obj):
        obj["stages"][0]["llm_calls"] = 7

    _mutate_manifest(run_dir, fn)


def tamper_remove_run_dir(run_dir: Path) -> None:
    """Delete the run directory entirely so the validator hits the exit-2 path."""
    shutil.rmtree(run_dir)


# --------------------------------------------------------------------------- #
# Case definitions
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    name: str
    expected_exit: int  # 0 (valid) | 1 (fail) | 2 (validator/external error)
    tamper: Callable[[Path], None] | None
    expected_failed_rule: str | None  # rule_id we expect to flip to fail; None for valid


CASES: list[Case] = [
    Case("valid_run", 0, None, None),
    Case("missing_run_manifest", 1, tamper_remove_manifest, "R001_manifest_schema"),
    Case("malformed_run_manifest", 1, tamper_malformed_manifest, "R001_manifest_schema"),
    Case("missing_stage_ledger", 1, tamper_remove_ledger, "R002_ledger_schema"),
    Case("malformed_ledger_line", 1, tamper_malformed_ledger, "R002_ledger_schema"),
    Case("deleted_artifact", 1, tamper_delete_artifact, "R004_artifact_paths"),
    Case("byte_flipped_artifact", 1, tamper_byte_flip_artifact, "R005_artifact_hashes"),
    Case("pass_with_empty_outputs", 1, tamper_pass_with_no_outputs, "R007_pass_outputs"),
    Case("stage_order_changed", 1, tamper_stage_order, "R003_stage_order"),
    Case("duplicate_stage_id", 1, tamper_duplicate_stage_id, "R012_unique_stage_ids"),
    Case("hash_chain_broken", 1, tamper_hash_chain_break, "R009_hash_chain"),
    Case("missing_report_path", 1, tamper_missing_report, "R006_report_paths"),
    Case("path_traversal_dotdot", 1, tamper_path_traversal, "R004_artifact_paths"),
    Case("path_traversal_symlink", 1, tamper_symlink_escape, None),
    # ^ symlink escape: either R004 or R005 catches it; we accept either.
    Case("llm_calls_nonzero", 1, tamper_llm_calls, "R008_no_llm_calls"),
    # Final case: the run dir itself is gone — validator must exit 2 (external/internal),
    # distinct from a clean fail (1). This proves the CLI distinguishes "broken run"
    # from "missing run".
    Case("missing_run_directory", 2, tamper_remove_run_dir, None),
]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _run_validator(run_dir: Path, *, env: dict[str, str] | None = None) -> tuple[int, dict | None]:
    cmd = [str(VENV_PY), "-m", "compgen.graph_compilation", "validate", "--run", str(run_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    report_path = run_dir / "validation" / "artifact_validation.json"
    report: dict | None = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = None
    return proc.returncode, report


def _check_case(out_root: Path, source_valid: Path, case: Case) -> tuple[bool, str]:
    """Run one case, return (ok, detail)."""
    target = out_root / case.name
    if target.exists():
        shutil.rmtree(target)
    if case.tamper is None:
        # Valid case: copy clean source.
        shutil.copytree(source_valid, target)
    else:
        shutil.copytree(source_valid, target)
        case.tamper(target)

    rc, report = _run_validator(target)
    if rc != case.expected_exit:
        return False, f"expected exit {case.expected_exit}, got {rc}; report={report!r}"

    if case.expected_failed_rule is None:
        # Either valid case (need overall=pass) or a tampered case where any
        # specific rule may flip — we just need a coherent outcome.
        if case.expected_exit == 0:
            if report is None or report.get("overall") != "pass":
                return False, f"expected overall=pass, report={report!r}"
        elif case.expected_exit == 2:
            # CLI exited 2 (external error like a missing run dir). No
            # validation report is expected.
            if report is not None:
                return False, f"expected no report on exit 2, got {report!r}"
        else:
            # exit 1 with no specific rule expectation: ≥1 failed rule.
            if report is None:
                return False, "no validation report written"
            fails = [r for r in report["rules"] if r["status"] == "fail"]
            if not fails:
                return False, f"expected ≥1 failed rule, got {report['rules']}"
        return True, "ok"

    # Tampered case with a named expected rule.
    if report is None:
        return False, "tampered run produced no validation report"
    flagged = next((r for r in report["rules"] if r["rule_id"] == case.expected_failed_rule), None)
    if flagged is None:
        return False, f"expected rule {case.expected_failed_rule} not present in report"
    if flagged["status"] != "fail":
        return False, f"expected rule {case.expected_failed_rule} to fail, got {flagged['status']}"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="graph_compilation artifact contract black-box acceptance check.")
    parser.add_argument("--out", required=True, type=Path, help="Working directory for synthetic runs.")
    args = parser.parse_args(argv)

    if not VENV_PY.exists():
        print(f"error: expected venv interpreter at {VENV_PY}", file=sys.stderr)
        return 2

    out_root: Path = args.out.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    # Build the canonical valid run that all cases derive from.
    valid_src = out_root / "_source_valid"
    build_valid_run(valid_src)

    # First, validate the source on its own as a sanity step.
    rc, report = _run_validator(valid_src)
    if rc != 0 or report is None or report.get("overall") != "pass":
        print(
            json.dumps(
                {
                    "fatal": "source valid run did not validate",
                    "exit": rc,
                    "report": report,
                },
                indent=2,
            )
        )
        return 1

    valid_passed = 0
    tampered_rejected = 0
    unexpected_passes = 0
    unexpected_failures = 0
    failures: list[dict[str, str]] = []

    for case in CASES:
        ok, detail = _check_case(out_root, valid_src, case)
        if ok:
            if case.expected_exit == 0:
                valid_passed += 1
            else:
                tampered_rejected += 1
            print(f"  [ok ] {case.name}")
        else:
            if case.expected_exit == 0:
                unexpected_failures += 1
            else:
                # Could be: tampered run validated cleanly, or wrong rule named.
                # We classify both as unexpected_passes if the validator returned 0,
                # otherwise as unexpected_failures (validator failed but for the wrong reason).
                if "got 0" in detail:
                    unexpected_passes += 1
                else:
                    unexpected_failures += 1
            failures.append({"case": case.name, "detail": detail})
            print(f"  [FAIL] {case.name}: {detail}")

    summary = {
        "valid_cases_passed": valid_passed,
        "tampered_cases_rejected": tampered_rejected,
        "unexpected_passes": unexpected_passes,
        "unexpected_failures": unexpected_failures,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if (unexpected_passes == 0 and unexpected_failures == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
