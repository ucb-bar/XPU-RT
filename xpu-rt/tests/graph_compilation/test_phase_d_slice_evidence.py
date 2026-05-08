"""M-65 — Phase D vertical-slice evidence tests.

Coverage:

- ``TestSlice2`` — proxy_vla on host_cpu (fusion path). The selected
  candidate is a fusion (``fuse_*``) which M-42 routes to
  ``not_applicable``; the slice evidence records this as
  ``honest_gap`` rather than papering over it.
- ``TestSlice3Deferred`` — merlin_mlp_wide on cuda_sm75; no
  configs/targets/cuda_sm75.yaml ships locally, so the slice is
  deferred. The evidence honestly records the deferral.
- ``TestEvidenceSchema`` — the emitted JSON matches schema_version
  ``phase_d_slice_evidence_v1`` with all five summary blocks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke_pipeline(
    *, model: str, out_dir: Path, stop_after: str = "execution-plan-emit",
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", stop_after,
            "--selection-mode", "greedy",
            "--auction-mode", "multi-bidder",
            "--kernel-coverage-mode", "both",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


# --------------------------------------------------------------------------- #
# Slice 2 — proxy_vla fusion path
# --------------------------------------------------------------------------- #


class TestSlice2:
    def test_proxy_vla_fusion_path(self, tmp_path: Path) -> None:
        """Run the pipeline + emit slice evidence. proxy_vla's
        candidate is a fusion (set_tile_params is M-42's only
        supported kind today), so the request_kind comes back as
        ``not_applicable``; the slice evidence records this as an
        honest gap."""
        result = _invoke_pipeline(
            model="proxy_vla", out_dir=tmp_path / "run",
            stop_after="kernel-auction",
        )
        # The pipeline runs through kernel-auction even when the
        # M-42 task is not_applicable (the auction stage short-
        # circuits). It should not error.
        assert result.returncode == 0, result.stderr

        from compgen.graph_compilation.phase_d_slice_evidence import (
            emit_slice_evidence,
        )

        run_dir = tmp_path / "run"
        evidence_path = emit_slice_evidence(
            run_dir=run_dir, slice_id="2", slice_name="proxy_vla_fusion",
            model="proxy_vla", target="host_cpu",
            overall="honest_gap",
            overall_reason=(
                "M-42 supports only candidate_kind='set_tile_params' today; "
                "proxy_vla's recipe planner selects fusion candidates "
                "which route to not_applicable. Auction never bids; "
                "fusion-archetype contract registry expansion is the next "
                "milestone."
            ),
            notes="See 04_kernel_codegen/kernel_codegen_summary.json::not_applicable_reason",
        )
        body = json.loads(evidence_path.read_text())

        assert body["schema_version"] == "phase_d_slice_evidence_v1"
        assert body["slice_id"] == "2"
        assert body["overall"] == "honest_gap"
        # Auction didn't run (not_applicable request).
        assert body["auction_summary"]["ran"] is False


# --------------------------------------------------------------------------- #
# Slice 3 — cuda_sm75 deferred
# --------------------------------------------------------------------------- #


class TestSlice3Deferred:
    def test_cuda_sm75_target_not_present_locally(
        self, tmp_path: Path,
    ) -> None:
        """No configs/targets/cuda_sm75.yaml ships in the repo today;
        slice 3 must record the deferral honestly."""
        cuda_yaml = REPO_ROOT / "configs" / "targets" / "cuda_sm75.yaml"
        # Documented residual: cuda_sm75 not configured locally.
        assert not cuda_yaml.exists()

        from compgen.graph_compilation.phase_d_slice_evidence import (
            emit_deferred_slice_evidence,
        )

        evidence_path = emit_deferred_slice_evidence(
            out_dir=tmp_path,
            slice_id="3",
            slice_name="merlin_mlp_wide_cuda_sm75",
            model="merlin_mlp_wide",
            target="cuda_sm75",
            deferred_reason=(
                "configs/targets/cuda_sm75.yaml does not ship with the "
                "repo; running CUDA-bound providers (TritonTemplate, "
                "Claude-Code GPU codegen) requires a CUDA-capable host "
                "and a target YAML. M-66 covers the CUDA path under a "
                "GPU-host-conditional run."
            ),
        )
        body = json.loads(evidence_path.read_text())
        assert body["schema_version"] == "phase_d_slice_evidence_v1"
        assert body["slice_id"] == "3"
        assert body["overall"] == "deferred"
        assert "configs/targets/cuda_sm75.yaml" in body["overall_reason"]


# --------------------------------------------------------------------------- #
# Evidence schema
# --------------------------------------------------------------------------- #


class TestEvidenceSchema:
    def test_evidence_carries_five_summary_blocks(self, tmp_path: Path) -> None:
        from compgen.graph_compilation.phase_d_slice_evidence import (
            emit_deferred_slice_evidence,
        )

        path = emit_deferred_slice_evidence(
            out_dir=tmp_path, slice_id="schema_test",
            slice_name="schema_test", model="x", target="y",
            deferred_reason="schema-only test",
        )
        body = json.loads(path.read_text())
        for k in (
            "auction_summary",
            "coverage_summary",
            "specialization_summary",
            "bindings_summary",
            "contract_versioning_summary",
        ):
            assert k in body, f"missing summary block {k}"
            assert isinstance(body[k], dict)
        assert body["schema_version"] == "phase_d_slice_evidence_v1"
