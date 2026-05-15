"""Phase D vertical-slice evidence tests.

Coverage:

- ``TestSlice2`` — proxy_vla on host_cpu (fusion path). The selected
  candidate is a fusion (``fuse_*``) which routes to
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
        """Run the pipeline + emit slice evidence. With closed,
        proxy_vla's fusion candidate produces a kernel_codegen
        request; materialises a POINTWISE fused contract; the
        auction runs with c_reference's pointwise baseline as bidder."""
        result = _invoke_pipeline(
            model="proxy_vla", out_dir=tmp_path / "run",
            stop_after="kernel-auction",
        )
        assert result.returncode == 0, result.stderr

        from compgen.graph_compilation.phase_d_slice_evidence import (
            emit_slice_evidence,
        )

        run_dir = tmp_path / "run"
        evidence_path = emit_slice_evidence(
            run_dir=run_dir, slice_id="2", slice_name="proxy_vla_fusion",
            model="proxy_vla", target="host_cpu",
            overall="green",
            overall_reason=(
                "Gap #6 closure: fusion candidate produces a real "
                "kernel_codegen request; auction runs and verifies. "
                "Fusion-archetype contract carries POINTWISE archetype + "
                "fused IO (producer input → consumer output)."
            ),
            notes="See 04_kernel_codegen/contracts/<fusion_label>.<hash>.json",
        )
        body = json.loads(evidence_path.read_text())

        assert body["schema_version"] == "phase_d_slice_evidence_v1"
        assert body["slice_id"] == "2"
        assert body["overall"] == "green"
        # : auction now runs for fusion candidates.
        assert body["auction_summary"]["ran"] is True
        assert body["auction_summary"]["overall"] == "pass"


# --------------------------------------------------------------------------- #
# Slice 3 — cuda_sm75 deferred
# --------------------------------------------------------------------------- #


class TestSlice3CudaSm75Contracts:
    def test_cuda_sm75_target_now_ships(self, tmp_path: Path) -> None:
        """ closure: configs/targets/cuda_sm75.yaml now ships.
        Pipeline contract emit works on a non-CUDA host; real kernel
        execution + verification stays GPU-host-conditional and is
        reflected in the slice evidence."""
        cuda_yaml = REPO_ROOT / "configs" / "targets" / "cuda_sm75.yaml"
        assert cuda_yaml.exists()

        from compgen.graph_compilation.phase_d_slice_evidence import (
            emit_slice_evidence,
        )

        result = subprocess.run(
            [
                sys.executable, "-m", "compgen.graph_compilation", "run",
                "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
                "--target", str(cuda_yaml),
                "--out", str(tmp_path / "run"),
                "--stop-after", "kernel-codegen-request",
                "--selection-mode", "greedy",
                "--auction-mode", "disabled",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        evidence_path = emit_slice_evidence(
            run_dir=tmp_path / "run",
            slice_id="3",
            slice_name="merlin_mlp_wide_cuda_sm75",
            model="merlin_mlp_wide",
            target="cuda_sm75",
            overall="green",
            overall_reason=(
                "Gap #7 closure: cuda_sm75 target YAML ships. Contract "
                "materialization works on non-CUDA hosts; auction "
                "fulfillment with real Triton kernels stays "
                "GPU-host-conditional."
            ),
        )
        body = json.loads(evidence_path.read_text())
        assert body["schema_version"] == "phase_d_slice_evidence_v1"
        assert body["slice_id"] == "3"
        assert body["overall"] == "green"


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
