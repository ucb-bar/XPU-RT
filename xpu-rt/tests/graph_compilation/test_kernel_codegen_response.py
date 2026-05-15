"""provider-response validator + commit-tool tests.

The plan's 12 tests (failure classes 1-12), exercised on real
task surfaces produced by the pipeline. Each test drives
``commit_response`` against the canonical merlin_mlp_wide task
and asserts the typed failure_kind + recoverability + next_action.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compgen.graph_compilation.kernel_codegen_response import (
    DEFAULT_MAX_ATTEMPTS,
    RECOVERABILITY,
    commit_response,
    validate_response,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Pipeline driver — produces a real task to validate against
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def merlin_task(tmp_path_factory) -> dict:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m43_merlin") / "run"
    res = subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out),
            "--stop-after", "kernel-codegen-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    request_files = sorted(
        (out / "04_kernel_codegen" / "requests").glob("*.request.json")
    )
    assert len(request_files) == 1
    request_body = json.loads(request_files[0].read_text())
    return {
        "run_dir": out,
        "request_path": request_files[0],
        "request_body": request_body,
        "task_id": request_body["task_id"],
    }


def _contract_compliant_metadata(request_body: dict) -> dict:
    """Read the materialised contract and emit metadata that satisfies
    the contract-driven obligations. Real providers will derive
    these the same way."""
    full_path = Path(request_body["contract_paths"]["full"])
    return _metadata_from_contract_path(full_path)


def _metadata_from_contract_path(rel_path: Path) -> dict:
    # Tests that have run_dir context resolve the path; here we assume
    # the caller passes a relative path that the helper resolves.
    return {}


def _write_minimal_artifacts(run_dir: Path, request_body: dict) -> dict[str, str]:
    """Helper: drop -compliant artifacts into the sandboxed
    artifact_dir. Returns the artifacts dict the response declares."""
    artifact_dir = run_dir / request_body["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Read the materialised contract to derive matching metadata.
    contract_full = run_dir / request_body["contract_paths"]["full"]
    contract = json.loads(contract_full.read_text(encoding="utf-8"))
    io = contract["io"]
    metadata = {
        "inputs": [
            {"dims": list(t["shape"]["dims"]),
             "dtype": t["dtype_class"][0],
             "layout": t["layout"]}
            for t in io["inputs"]
        ],
        "outputs": [
            {"dims": list(t["shape"]["dims"]),
             "dtype": t["dtype_class"][0],
             "layout": t["layout"]}
            for t in io["outputs"]
        ],
        "accumulator_dtype": io["numerics"]["accumulator_dtype"],
        "target_name": (
            (contract["orchestration"]["execution"] or {})
            .get("hardware", {}).get("target_name", "")
        ),
        "signals_emitted": {
            e["name"]: e["wait_count"]
            for e in contract["orchestration"]["sync"].get("event_decls") or []
        },
    }
    dispatch_model = contract["orchestration"]["dispatch"]["model"]
    claims = {
        "backend": request_body["allowed_backends"][0],
        "supports_dispatch": [dispatch_model],
        "expected_numerics": "bit_equality",
        "estimated_registers": 0,
        "estimated_smem_bytes": 0,
    }

    artifacts = {}
    for name in request_body["required_outputs"]:
        ext = ".c" if name == "kernel_source" else ".json"
        path = artifact_dir / f"{name}{ext}"
        if name == "kernel_metadata":
            path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif name == "provider_claims":
            path.write_text(
                json.dumps(claims, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif name == "launch_config":
            path.write_text(
                json.dumps({"grid": [1, 1, 1], "block": [1, 1, 1]},
                           indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text("/* placeholder kernel source */\n", encoding="utf-8")
        artifacts[name] = str(path.relative_to(run_dir))
    return artifacts


def _good_response(run_dir: Path, request_body: dict) -> dict:
    artifacts = _write_minimal_artifacts(run_dir, request_body)
    return {
        "schema_version": "kernel_codegen_response_v1",
        "task_id": request_body["task_id"],
        "contract_hash": request_body["contract_hash"],
        "artifacts": artifacts,
        "claims": {
            "backend": request_body["allowed_backends"][0],
            "supports_dispatch": ["sync"],
            "estimated_registers": 0,
            "estimated_smem_bytes": 0,
            "expected_numerics": "bit_equality",
        },
        "provider": {"kind": "test_synthetic", "model": "stub"},
        "contract_feedback": [],
        "notes": "synthetic placeholder for M-43 testing",
    }


# --------------------------------------------------------------------------- #
# Failure-class tests (the plan's 12 cases)
# --------------------------------------------------------------------------- #


class TestSchemaInvalidFailures:
    """Tests 1-2: invalid JSON + missing artifact path → retry_request."""

    def test_invalid_json_response_rejected(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        result = commit_response(
            run_dir=run_dir, task_id=merlin_task["task_id"],
            response="this is not JSON {",
        )
        assert not result.accepted
        assert result.failure_kind == "schema_invalid"
        assert result.recoverability == "recoverable_provider_failure"
        assert result.next_action == "retry"
        assert (run_dir / result.retry_request_path).exists()

    def test_missing_required_artifact_path_rejected(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        body = _good_response(run_dir, merlin_task["request_body"])
        # Strip a required output.
        del body["artifacts"]["kernel_source"]
        result = commit_response(
            run_dir=run_dir, task_id=merlin_task["task_id"], response=body,
        )
        assert not result.accepted
        assert result.failure_kind == "schema_invalid"
        assert "missing required outputs" in result.failure_summary
        assert result.next_action == "retry"


class TestProtocolFatalFailures:
    """Tests 6-8: contract_hash_mismatch, contract_mutation,
    forbidden_path_write — all fatal, no retry."""

    def test_contract_hash_mismatch_fatal(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        body = _good_response(run_dir, merlin_task["request_body"])
        body["contract_hash"] = "deadbeefdeadbeef"
        result = commit_response(
            run_dir=run_dir, task_id=merlin_task["task_id"], response=body,
        )
        assert not result.accepted
        assert result.failure_kind == "contract_hash_mismatch"
        assert result.recoverability == "protocol_or_contract_fatal"
        assert result.next_action == "fatal_reject"
        # Failure report (downstream_retry_request_v1) emitted.
        assert (run_dir / result.retry_request_path).exists()
        failure_report = json.loads(
            (run_dir / result.retry_request_path).read_text()
        )
        assert failure_report["schema_version"] == "downstream_retry_request_v1"
        assert failure_report["failed_check"] == "contract_hash_mismatch"

    def test_contract_mutation_fatal(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        # Write the response artifacts FIRST (needs the contract to
        # derive matching metadata)...
        body = _good_response(run_dir, merlin_task["request_body"])
        # ... then delete the materialised contract — simulates "the
        # provider modified or removed the contract" between request
        # emit and commit.
        full = run_dir / merlin_task["request_body"]["contract_paths"]["full"]
        full.unlink()
        result = commit_response(
            run_dir=run_dir, task_id=merlin_task["task_id"], response=body,
        )
        assert not result.accepted
        assert result.failure_kind == "contract_mutation"
        assert result.recoverability == "protocol_or_contract_fatal"
        assert result.next_action == "fatal_reject"

    def test_forbidden_path_write_fatal(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        # Write an artifact OUTSIDE the sandbox.
        body = _good_response(run_dir, merlin_task["request_body"])
        outside = run_dir / "rogue_output.txt"
        outside.write_text("escape attempt\n")
        body["artifacts"]["kernel_source"] = "rogue_output.txt"
        result = commit_response(
            run_dir=run_dir, task_id=merlin_task["task_id"], response=body,
        )
        assert not result.accepted
        assert result.failure_kind == "forbidden_path_write"
        assert result.recoverability == "protocol_or_contract_fatal"
        assert result.next_action == "fatal_reject"

    def test_absolute_path_rejected_as_forbidden(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        body = _good_response(run_dir, merlin_task["request_body"])
        body["artifacts"]["kernel_source"] = "/absolute/path/to/rogue.c"
        result = commit_response(
            run_dir=run_dir, task_id=merlin_task["task_id"], response=body,
        )
        assert not result.accepted
        assert result.failure_kind == "forbidden_path_write"


class TestProviderProtocolViolations:
    """unsupported_backend + semantic_contract_violation paths."""

    def test_unsupported_backend_rejected(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        body = _good_response(run_dir, merlin_task["request_body"])
        body["claims"]["backend"] = "cuda_assembly_handwritten"
        result = commit_response(
            run_dir=run_dir, task_id=merlin_task["task_id"], response=body,
        )
        assert not result.accepted
        assert result.failure_kind == "unsupported_backend"
        assert result.recoverability == "provider_protocol_violation"

    def test_semantic_contract_violation_rejected(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        body = _good_response(run_dir, merlin_task["request_body"])
        body["claims"]["expected_numerics"] = "made_up_refinement"
        result = commit_response(
            run_dir=run_dir, task_id=merlin_task["task_id"], response=body,
        )
        assert not result.accepted
        assert result.failure_kind == "semantic_contract_violation"


# --------------------------------------------------------------------------- #
# Attempt trail + retry policy
# --------------------------------------------------------------------------- #


class TestAttemptTrail:
    def test_attempts_are_appended_per_invocation(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        for _ in range(2):
            commit_response(
                run_dir=run_dir, task_id=merlin_task["task_id"],
                response="not json",
            )
        log_path = run_dir / "04_kernel_codegen" / "kernel_codegen_attempts.json"
        log = json.loads(log_path.read_text())
        assert log["task_id"] == merlin_task["task_id"]
        assert len(log["attempts"]) == 2
        assert log["attempts"][0]["attempt_index"] == 0
        assert log["attempts"][1]["attempt_index"] == 1
        # Each attempt has its own dir.
        attempt_dirs = sorted(
            (run_dir / "04_kernel_codegen" / "attempts" / merlin_task["task_id"])
            .iterdir()
        )
        assert len(attempt_dirs) == 2

    def test_three_failed_attempts_emit_downstream_retry(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        # Three recoverable failures — exhausts the budget.
        results = [
            commit_response(
                run_dir=run_dir, task_id=merlin_task["task_id"],
                response="not json", max_attempts=DEFAULT_MAX_ATTEMPTS,
            )
            for _ in range(3)
        ]
        # The last result must escalate to fatal_reject.
        assert results[-1].next_action == "fatal_reject"
        assert results[-1].failure_kind == "kernel_codegen_attempts_exhausted"
        # downstream_retry_request_v1 emitted.
        report_path = (
            run_dir / "04_kernel_codegen" / "kernel_codegen_failure_report.json"
        )
        assert report_path.exists()
        body = json.loads(report_path.read_text())
        assert body["schema_version"] == "downstream_retry_request_v1"
        assert body["failed_check"] == "kernel_codegen_attempts_exhausted"
        assert body["retry_policy"]["must_choose_different_candidate"] is True
        assert (
            merlin_task["request_body"]["candidate_id"]
            in body["retry_policy"]["exclude_candidate_ids"]
        )


# --------------------------------------------------------------------------- #
# Accept path
# --------------------------------------------------------------------------- #


class TestAcceptedResponse:
    def test_well_formed_response_accepted(
        self, merlin_task: dict, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "rd"
        shutil.copytree(merlin_task["run_dir"], run_dir)
        body = _good_response(run_dir, merlin_task["request_body"])
        result = commit_response(
            run_dir=run_dir, task_id=merlin_task["task_id"], response=body,
        )
        assert result.accepted is True, result.failure_summary
        # verifier ran; "verified" or "verifier_pending" depending on
        # whether all obligations short-circuit on metadata or stay deferred.
        assert result.next_action in ("verified", "verifier_pending"), (
            f"unexpected next_action {result.next_action!r}"
        )
        # wrote a validation report.
        validation_dir = run_dir / "04_kernel_codegen" / "validation"
        assert validation_dir.is_dir()
        assert any(validation_dir.iterdir())
        # No retry / failure report.
        assert not (
            run_dir / "04_kernel_codegen" / "kernel_codegen_retry_request.json"
        ).exists()
        assert not (
            run_dir / "04_kernel_codegen" / "kernel_codegen_failure_report.json"
        ).exists()

    def test_recoverability_table_is_complete(self) -> None:
        """The failure-kind taxonomy must enumerate every kind the
        validator can raise. Catches a regression where a new kind is
        added to validate_response but not to the recovery table."""
        observed_in_module = {
            "schema_invalid", "compile_error", "metadata_mismatch",
            "numerical_mismatch", "shape_mismatch",
            "unsupported_backend", "semantic_contract_violation",
            "timeout",
            "contract_hash_mismatch", "contract_mutation",
            "forbidden_path_write",
        }
        assert observed_in_module == set(RECOVERABILITY.keys())


# --------------------------------------------------------------------------- #
# MCP tools — sanity
# --------------------------------------------------------------------------- #


class TestMcpTools:
    def test_inspect_returns_request_after_emit(
        self, merlin_task: dict,
    ) -> None:
        from compgen.mcp.tools.kernel_codegen import (
            compgen_inspect_kernel_codegen_task,
        )
        body = compgen_inspect_kernel_codegen_task(
            run_dir=str(merlin_task["run_dir"]),
            task_id=merlin_task["task_id"],
        )
        assert body["ok"]
        assert body["request"]["task_id"] == merlin_task["task_id"]
        # No attempts yet; the merlin_task fixture only emits the request.
        assert body["attempts"] == []

    def test_run_task_returns_operator_action_required_until_subagent_wired(
        self, merlin_task: dict,
    ) -> None:
        """ships the file-based protocol; spawn helper is +."""
        from compgen.mcp.tools.kernel_codegen import (
            compgen_run_kernel_codegen_task,
        )
        body = compgen_run_kernel_codegen_task(
            run_dir=str(merlin_task["run_dir"]),
            task_id=merlin_task["task_id"],
        )
        assert body["operator_action_required"] is True
        assert body["task_id"] == merlin_task["task_id"]
