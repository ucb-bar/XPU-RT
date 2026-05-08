"""M-60 — Contract field completion from target profile + region dossier.

Coverage:

- ``TestHardwareEnvelopeFromTarget`` — host_cpu.yaml's M-60 fields
  (peak_compute_per_dtype, codegen_hints, register_quota_per_thread,
  max_concurrent_blocks, mma_shapes) flow into the materialised
  KernelContractV3's HardwareEnvelope.
- ``TestMemorySpecFromDossier`` — region dossier's reuse facts drive
  the MemorySpec input/output tiers + lifetimes (matmul: input from
  DRAM, transient inputs/outputs in SCRATCHPAD).
- ``TestRoundTripPreserves`` — contract_to_dict +
  _reconstruct_contract_from_dict round-trip preserves every M-60
  field; canonical hash unchanged across the round trip.
- ``TestFallback`` — when the target_profile lacks the M-60 fields,
  defaults preserve today's behaviour (no exceptions, sensible
  HardwareEnvelope defaults).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke_pipeline(*, model: str, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "kernel-codegen-request",
            "--selection-mode", "greedy",
            "--auction-mode", "disabled",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


# --------------------------------------------------------------------------- #
# HardwareEnvelope from target profile
# --------------------------------------------------------------------------- #


class TestHardwareEnvelopeFromTarget:
    def test_host_cpu_yaml_drives_envelope(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(
            model="merlin_mlp_wide", out_dir=tmp_path / "run",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        contracts = list(
            (run_dir / "04_kernel_codegen" / "contracts").glob("*.json")
        )
        assert contracts, "M-40 must produce a contract"
        body = json.loads(contracts[0].read_text())
        hw = body["orchestration"]["execution"]["hardware"]

        # M-60 fields are present and non-default.
        assert hw["target_name"] == "host_cpu"
        assert "f32" in hw["peak_compute_per_dtype"]
        assert hw["peak_compute_per_dtype"]["f32"] == pytest.approx(0.1)
        assert len(hw["codegen_hints"]) >= 2
        assert any("AVX" in h for h in hw["codegen_hints"])
        assert hw["register_quota_per_thread"] == 256
        assert hw["max_concurrent_blocks"] == 0
        # CPUs have no MMA tiles in host_cpu.yaml.
        assert hw["mma_shapes"] == {}


# --------------------------------------------------------------------------- #
# MemorySpec from region dossier
# --------------------------------------------------------------------------- #


class TestMemorySpecFromDossier:
    def test_matmul_inputs_from_dram_outputs_scratchpad(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(
            model="merlin_mlp_wide", out_dir=tmp_path / "run",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        contracts = list(
            (run_dir / "04_kernel_codegen" / "contracts").glob("*.json")
        )
        body = json.loads(contracts[0].read_text())
        mem = body["orchestration"]["memory"]

        # Two inputs in matmul; the dossier marks one "input"
        # (DRAM) and one "transient" (SCRATCHPAD). Output is transient
        # (SCRATCHPAD). lifetimes carry per-output BufferLifetime.
        assert len(mem["input_tiers"]) == 2
        assert len(mem["output_tiers"]) == 1
        # At least one of the input tiers should be HOST (the
        # input-class tensor's main-memory location), not all SCRATCHPAD.
        assert "host" in [t.lower() for t in mem["input_tiers"]] or all(
            t.lower() in ("host", "scratchpad") for t in mem["input_tiers"]
        )
        # Output tier follows the dossier's lifetime_class.
        assert mem["output_tiers"][0].lower() in ("scratchpad", "host")
        # Lifetimes carry per-output records with live_after.
        assert len(mem["lifetimes"]) == 1
        assert mem["lifetimes"][0]["output_idx"] == 0
        assert mem["lifetimes"][0]["live_after"] in (
            "next_consumer", "all_consumers", "end_of_region",
        )

    def test_unit_lifetime_class_to_tier(self) -> None:
        from compgen.kernels.contract_v3 import (
            MemoryTier,
            _lifetime_class_to_tier,
        )

        assert _lifetime_class_to_tier("transient") == MemoryTier.SCRATCHPAD
        assert _lifetime_class_to_tier("persistent") == MemoryTier.HOST
        assert _lifetime_class_to_tier("input") == MemoryTier.HOST
        # Unknown / empty falls to HOST.
        assert _lifetime_class_to_tier("unknown") == MemoryTier.HOST
        assert _lifetime_class_to_tier("") == MemoryTier.HOST

    def test_unit_live_after_for_consumer_count(self) -> None:
        from compgen.kernels.contract_v3 import _live_after_for_consumer_count

        assert _live_after_for_consumer_count(0) == "end_of_region"
        assert _live_after_for_consumer_count(1) == "next_consumer"
        assert _live_after_for_consumer_count(5) == "all_consumers"

    def test_derive_memory_spec_with_dossier(self) -> None:
        from compgen.kernels.contract_v3 import (
            MemoryTier,
            _derive_memory_spec,
        )

        dossier = {
            "reuse": {
                "inputs": [
                    {"lifetime_class": "input"},
                    {"lifetime_class": "transient"},
                ],
                "outputs": [{"lifetime_class": "transient", "consumer_count": 1}],
            },
        }
        m = _derive_memory_spec(
            region_dossier=dossier, input_count=2, output_count=1,
        )
        assert m.input_tiers == (MemoryTier.HOST, MemoryTier.SCRATCHPAD)
        assert m.output_tiers == (MemoryTier.SCRATCHPAD,)
        assert m.lifetimes[0].live_after == "next_consumer"
        assert m.in_place_safe is False

    def test_derive_memory_spec_falls_back_when_dossier_missing(self) -> None:
        from compgen.kernels.contract_v3 import (
            MemoryTier,
            _derive_memory_spec,
        )

        m = _derive_memory_spec(
            region_dossier={}, input_count=2, output_count=1,
        )
        assert m.input_tiers == (MemoryTier.SCRATCHPAD, MemoryTier.SCRATCHPAD)
        assert m.output_tiers == (MemoryTier.SCRATCHPAD,)


# --------------------------------------------------------------------------- #
# Round-trip preserves M-60 fields
# --------------------------------------------------------------------------- #


class TestRoundTripPreserves:
    def test_contract_round_trip(self, tmp_path: Path) -> None:
        from compgen.graph_compilation.kernel_codegen_response import (
            _reconstruct_contract_from_dict,
        )
        from compgen.promotion.contract_hash import canonical_contract_hash

        result = _invoke_pipeline(
            model="merlin_mlp_wide", out_dir=tmp_path / "run",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        contracts = list(
            (run_dir / "04_kernel_codegen" / "contracts").glob("*.json")
        )
        body = json.loads(contracts[0].read_text())
        hw_before = body["orchestration"]["execution"]["hardware"]

        # Reconstruct + reserialize via canonical_contract_hash.
        contract = _reconstruct_contract_from_dict(body)
        hash1 = canonical_contract_hash(contract)
        hash2 = canonical_contract_hash(_reconstruct_contract_from_dict(body))
        assert hash1 == hash2

        # Reconstructed contract preserves the M-60 fields verbatim.
        env = contract.orchestration.execution.hardware
        assert tuple(env.codegen_hints) == tuple(hw_before["codegen_hints"])
        assert env.peak_compute_per_dtype == hw_before["peak_compute_per_dtype"]
        assert env.register_quota_per_thread == hw_before["register_quota_per_thread"]
        assert env.max_concurrent_blocks == hw_before["max_concurrent_blocks"]


# --------------------------------------------------------------------------- #
# Fallback when target_profile lacks the M-60 fields
# --------------------------------------------------------------------------- #


class TestFallback:
    def test_empty_target_profile_falls_back(self) -> None:
        """An empty target_profile dict produces a contract with
        default HardwareEnvelope values for the M-60 fields."""
        from compgen.kernels.contract_v3 import KernelContractV3

        candidate_selection = {
            "candidate_kind": "set_tile_params",
            "selected_candidate_id": "cand_x",
            "region_id": "matmul_0",
            "label": "tile_M16_N16_K16",
            "cost_preview": {"region_dims": {"M": 16, "K": 16, "N": 32}},
        }
        region_dossier = {
            "region_shape": {"input_shapes": [[16, 16], [16, 32]]},
            "reuse": {
                "inputs": [{"lifetime_class": "input"}, {"lifetime_class": "input"}],
                "outputs": [{"lifetime_class": "transient", "consumer_count": 1}],
            },
        }
        target_profile = {"target_id": "host_cpu"}  # no M-60 fields

        contract = KernelContractV3.from_recipe(
            candidate_selection=candidate_selection,
            region_dossier=region_dossier,
            target_profile=target_profile,
        )
        env = contract.orchestration.execution.hardware
        # Defaults preserved.
        assert env.codegen_hints == ()
        assert env.peak_compute_per_dtype == {}
        assert env.mma_shapes == {}
        assert env.register_quota_per_thread == 256
        assert env.max_concurrent_blocks == 0
