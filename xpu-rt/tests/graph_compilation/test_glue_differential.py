"""M-49 glue-differential tests — paper-facing milestone.

End-to-end:
- merlin_mlp_wide (K_iters=1, declared bit_equality) → 8/8 cases bit-equal,
  refinement_status=discharged_bit_equality.
- tiny_mlp (K_iters>1, declared tolerance_eps) → all cases within Higham bound,
  refinement_status=discharged_tolerance_eps.
- Tampered kernel (returns torch.zeros) → status=fail, M-15B retry trips.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _drive_pipeline_with_provider_response(model: str, out: Path) -> None:
    res = subprocess.run([
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out),
        "--stop-after", "glue-emit",
        "--selection-mode", "greedy",
    ], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    req = json.loads(
        next((out / "04_kernel_codegen" / "requests").glob("*.request.json")).read_text()
    )
    contract = json.loads((out / req["contract_paths"]["full"]).read_text())
    io_b = contract["io"]
    metadata = {
        "inputs": [
            {"dims": list(t["shape"]["dims"]), "dtype": t["dtype_class"][0],
             "layout": t["layout"]} for t in io_b["inputs"]
        ],
        "outputs": [
            {"dims": list(t["shape"]["dims"]), "dtype": t["dtype_class"][0],
             "layout": t["layout"]} for t in io_b["outputs"]
        ],
        "accumulator_dtype": io_b["numerics"]["accumulator_dtype"],
        "target_name": (contract["orchestration"]["execution"] or {})
            .get("hardware", {}).get("target_name", ""),
        "signals_emitted": {
            e["name"]: e["wait_count"]
            for e in contract["orchestration"]["sync"]["event_decls"]
        },
    }
    claims = {
        "backend": req["allowed_backends"][0],
        "supports_dispatch": [contract["orchestration"]["dispatch"]["model"]],
        "expected_numerics": "bit_equality",
        "estimated_registers": 0, "estimated_smem_bytes": 0,
    }
    sandbox = out / req["artifact_dir"]
    sandbox.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in req["required_outputs"]:
        ext = ".c" if name == "kernel_source" else ".json"
        p = sandbox / f"{name}{ext}"
        if name == "kernel_metadata":
            p.write_text(json.dumps(metadata, sort_keys=True))
        elif name == "provider_claims":
            p.write_text(json.dumps(claims, sort_keys=True))
        elif name == "launch_config":
            p.write_text("{}")
        else:
            p.write_text("/* synthetic */\n")
        artifacts[name] = str(p.relative_to(out))
    response = {
        "schema_version": "kernel_codegen_response_v1",
        "task_id": req["task_id"], "contract_hash": req["contract_hash"],
        "artifacts": artifacts, "claims": claims,
        "provider": {"kind": "test_synthetic"},
    }
    from compgen.graph_compilation.kernel_codegen_response import commit_response
    commit_response(run_dir=out, task_id=req["task_id"], response=response)
    from compgen.graph_compilation.execution_plan_emit import emit_execution_plan
    from compgen.runtime.glue_emit import emit_python_sync_executor
    emit_execution_plan(out)
    emit_python_sync_executor(out)


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


def test_merlin_mlp_wide_discharges_bit_equality(tmp_path: Path) -> None:
    """K_iters=1 (K=16, tile_K=16) → tiled output exactly equals eager."""
    from compgen.graph_compilation.glue_differential import run_glue_differential
    out = tmp_path / "run"
    _drive_pipeline_with_provider_response("merlin_mlp_wide", out)
    result = run_glue_differential(out, num_cases=8)
    assert result.status == "pass", result.failure_summary
    assert result.refinement_status == "discharged_bit_equality"
    assert result.cases_passed == 8
    # All cases bit-exact.
    for c in result.case_records:
        assert c.max_abs_error == 0.0


def test_tiny_mlp_discharges_tolerance_eps(tmp_path: Path) -> None:
    """tiny_mlp picks tile_M4_N16_K16 on K=64 → K_iters=4 → tolerance_eps.
    Every case must be within the Higham bound."""
    from compgen.graph_compilation.glue_differential import run_glue_differential
    out = tmp_path / "run"
    _drive_pipeline_with_provider_response("tiny_mlp", out)
    result = run_glue_differential(out, num_cases=8)
    assert result.status == "pass", result.failure_summary
    assert result.refinement_status == "discharged_tolerance_eps"
    assert result.cases_passed == 8
    # Every case within the Higham bound.
    for c in result.case_records:
        assert c.max_abs_error <= c.higham_bound, (
            f"{c.case_id}: max_abs={c.max_abs_error:.3e} > "
            f"bound={c.higham_bound:.3e}"
        )


# --------------------------------------------------------------------------- #
# Tampered kernel — M-15B downstream-retry surface
# --------------------------------------------------------------------------- #


def test_tampered_kernel_triggers_failure(tmp_path: Path) -> None:
    """A buggy kernel that returns torch.zeros instead of A@B fails the
    differential with status=fail. The M-15B detector picks this up via
    the downstream-retry table."""
    from compgen.graph_compilation.glue_differential import run_glue_differential
    out = tmp_path / "run"
    _drive_pipeline_with_provider_response("merlin_mlp_wide", out)

    # Custom resolver that returns wrong output for any region.
    def _broken_kernel_resolver(contract: dict):
        import torch
        def _wrong(*args, **kwargs):
            # Return zeros of the right shape — passes the assert_plan
            # checks but fails the differential.
            return torch.zeros(args[0].shape[0], args[1].shape[1],
                               dtype=args[0].dtype)
        return _wrong

    result = run_glue_differential(
        out, num_cases=4, kernel_resolver=_broken_kernel_resolver,
    )
    assert result.status == "fail"
    assert result.refinement_status == "fail_refinement_mismatch"
    assert result.cases_passed == 0


def test_glue_differential_picks_up_in_m15b_table() -> None:
    """The M-15B downstream-retry detector must include
    'glue_differential' so a fail status surfaces as a typed retry."""
    from compgen.graph_compilation.downstream_retry import _DOWNSTREAM_REPORTS
    stage_ids = [row[0] for row in _DOWNSTREAM_REPORTS]
    assert "glue_differential" in stage_ids


def test_m15b_detects_glue_differential_failure(tmp_path: Path) -> None:
    """End-to-end: a tampered glue differential is detected as a
    downstream failure by the M-15B retry table."""
    from compgen.graph_compilation.glue_differential import run_glue_differential
    from compgen.graph_compilation.downstream_retry import (
        detect_downstream_failure,
    )
    out = tmp_path / "run"
    _drive_pipeline_with_provider_response("merlin_mlp_wide", out)

    def _broken(contract):
        import torch
        def _f(*args, **kwargs):
            return torch.zeros(args[0].shape[0], args[1].shape[1],
                               dtype=args[0].dtype)
        return _f

    run_glue_differential(out, num_cases=4, kernel_resolver=_broken)
    failure = detect_downstream_failure(out)
    assert failure is not None
    assert failure.failed_stage == "glue_differential"
    assert failure.failed_check == "glue_differential_check"


# --------------------------------------------------------------------------- #
# Plan-not-emitted skip
# --------------------------------------------------------------------------- #


def test_skips_when_no_plan(tmp_path: Path) -> None:
    """No plan → status=skipped, not crash."""
    from compgen.graph_compilation.glue_differential import run_glue_differential
    out = tmp_path / "empty"
    out.mkdir()
    result = run_glue_differential(out)
    assert result.status == "skipped"
