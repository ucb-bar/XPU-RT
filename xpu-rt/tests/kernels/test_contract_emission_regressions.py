"""Phase D gap closure — Batch C regression tests.

Covers gaps #7, #9, #14:

- #7: ``configs/targets/cuda_sm75.yaml`` ships; M-42 emits a
  contract for cuda_sm75 even on a non-CUDA host.
- #9: ``ShapeClass.divisibility`` populated by ``from_recipe`` from
  tile dims; canonical hash uses ``divisibility`` to abstract dims
  into ``{"mod": k}`` form. Two regions with different concrete
  dims both divisible by the same k share canonical hash.
- #14: ``COMPGEN_SHAPE_POLICY=class`` (or ``shape_policy="class"``
  kwarg) substitutes concrete dims with ``None``; canonical hash
  collapses to a single dynamic-shape canonical kernel covering any
  concrete instantiation under the declared divisibility.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Gap #7 — cuda_sm75 target
# --------------------------------------------------------------------------- #


class TestGap7CudaSm75TargetShipsContract:
    def test_target_yaml_exists(self) -> None:
        cfg = REPO_ROOT / "configs" / "targets" / "cuda_sm75.yaml"
        assert cfg.exists()

    def test_pipeline_emits_cuda_sm75_contract(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "compgen.graph_compilation", "run",
                "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
                "--target", str(REPO_ROOT / "configs/targets/cuda_sm75.yaml"),
                "--out", str(tmp_path / "run"),
                "--stop-after", "kernel-codegen-request",
                "--selection-mode", "greedy",
                "--auction-mode", "disabled",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        contracts = list(
            (tmp_path / "run" / "04_kernel_codegen" / "contracts").glob("*.json")
        )
        assert contracts
        body = json.loads(contracts[0].read_text())
        hw = body["orchestration"]["execution"]["hardware"]
        assert hw["target_name"] == "cuda_sm75"
        # cuda_sm75.yaml ships mma_shapes for f16/bf16/fp16.
        assert hw["mma_shapes"]
        assert "f16" in hw["mma_shapes"]
        assert hw["mma_shapes"]["f16"] == [16, 16, 16]
        # peak_compute_per_dtype.f16 = 65 TFLOPS Tensor Core.
        assert hw["peak_compute_per_dtype"]["f16"] == 65.0
        # Tensor Core codegen hint present.
        assert any("Tensor Core" in h for h in hw["codegen_hints"])


# --------------------------------------------------------------------------- #
# Gap #9 — ShapeClass.divisibility canonical-hash abstraction
# --------------------------------------------------------------------------- #


class TestGap9ShapeClassDivisibility:
    def test_divisibility_populated_by_from_recipe(self) -> None:
        from compgen.kernels.contract_v3 import KernelContractV3

        cs = {
            "candidate_kind": "set_tile_params",
            "selected_candidate_id": "x",
            "region_id": "r",
            "label": "tile_M16_N32_K16",
            "cost_preview": {"region_dims": {"M": 16, "K": 16, "N": 32}},
        }
        dossier = {"region_shape": {"input_shapes": [[16, 16], [16, 32]]}}
        c = KernelContractV3.from_recipe(
            candidate_selection=cs, region_dossier=dossier,
            target_profile={"target_id": "host_cpu"},
        )
        # lhs has divisibility (tile_M=16, tile_K=16).
        assert c.io.inputs[0].shape.divisibility == (16, 16)
        # rhs has divisibility (tile_K=16, tile_N=32).
        assert c.io.inputs[1].shape.divisibility == (16, 32)
        # out has divisibility (tile_M=16, tile_N=32).
        assert c.io.outputs[0].shape.divisibility == (16, 32)

    def test_two_regions_share_canonical_hash_under_divisibility(self) -> None:
        from compgen.kernels.contract_v3 import KernelContractV3
        from compgen.promotion.contract_hash import (
            canonical_contract_hash,
            instance_contract_hash,
        )

        # Same archetype/dtype/layout, different concrete dims, both
        # divisible by tile=16.
        c_a = KernelContractV3.from_recipe(
            candidate_selection={
                "candidate_kind": "set_tile_params",
                "selected_candidate_id": "a",
                "region_id": "r",
                "label": "tile_M16_N32_K16",
                "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
            },
            region_dossier={"region_shape": {"input_shapes": [[16, 16], [16, 32]]}},
            target_profile={"target_id": "host_cpu"},
        )
        c_b = KernelContractV3.from_recipe(
            candidate_selection={
                "candidate_kind": "set_tile_params",
                "selected_candidate_id": "b",
                "region_id": "r",
                "label": "tile_M16_N32_K16",
                "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 32}},
            },
            region_dossier={"region_shape": {"input_shapes": [[16, 32], [32, 32]]}},
            target_profile={"target_id": "host_cpu"},
        )
        assert instance_contract_hash(c_a) != instance_contract_hash(c_b), \
            "instance hashes should differ — concrete dims differ"
        assert canonical_contract_hash(c_a) == canonical_contract_hash(c_b), \
            "canonical hashes should match — both K%16==0"


# --------------------------------------------------------------------------- #
# Gap #14 — dynamic-shape mode
# --------------------------------------------------------------------------- #


class TestGap14DynamicShapeMode:
    def test_class_mode_substitutes_dims_with_none(self) -> None:
        from compgen.kernels.contract_v3 import KernelContractV3

        c = KernelContractV3.from_recipe(
            candidate_selection={
                "candidate_kind": "set_tile_params",
                "selected_candidate_id": "x",
                "region_id": "r",
                "label": "tile_M16_N32_K16",
                "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
            },
            region_dossier={"region_shape": {"input_shapes": [[16, 16], [16, 32]]}},
            target_profile={"target_id": "host_cpu"},
            shape_policy="class",
        )
        # All dims are None in class mode.
        for tio in c.io.inputs + c.io.outputs:
            assert all(d is None for d in tio.shape.dims), \
                f"dims should be all None in class mode; got {tio.shape.dims}"
        # divisibility is still populated.
        assert c.io.inputs[0].shape.divisibility == (16, 16)

    def test_class_mode_canonical_hash_is_single_per_archetype(self) -> None:
        """Two contracts from totally different concrete shapes
        produce the SAME canonical hash under class mode (since both
        are dynamic + same archetype + dtype + layout + target)."""
        from compgen.kernels.contract_v3 import KernelContractV3
        from compgen.promotion.contract_hash import canonical_contract_hash

        c_a = KernelContractV3.from_recipe(
            candidate_selection={
                "candidate_kind": "set_tile_params",
                "selected_candidate_id": "a",
                "region_id": "r",
                "label": "tile_M16_N32_K16",
                "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
            },
            region_dossier={"region_shape": {"input_shapes": [[16, 16], [16, 32]]}},
            target_profile={"target_id": "host_cpu"},
            shape_policy="class",
        )
        c_b = KernelContractV3.from_recipe(
            candidate_selection={
                "candidate_kind": "set_tile_params",
                "selected_candidate_id": "b",
                "region_id": "r",
                "label": "tile_M16_N32_K16",
                "cost_preview": {"region_dims": {"M": 64, "N": 128, "K": 256}},
            },
            region_dossier={"region_shape": {"input_shapes": [[64, 256], [256, 128]]}},
            target_profile={"target_id": "host_cpu"},
            shape_policy="class",
        )
        assert canonical_contract_hash(c_a) == canonical_contract_hash(c_b)

    def test_unknown_shape_policy_raises(self) -> None:
        from compgen.kernels.contract_v3 import KernelContractV3

        with pytest.raises(ValueError, match="unknown shape_policy"):
            KernelContractV3.from_recipe(
                candidate_selection={
                    "candidate_kind": "set_tile_params",
                    "selected_candidate_id": "x",
                    "region_id": "r",
                    "label": "tile_M16_N32_K16",
                    "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
                },
                region_dossier={"region_shape": {"input_shapes": [[16, 16], [16, 32]]}},
                target_profile={"target_id": "host_cpu"},
                shape_policy="invalid_mode",
            )

    def test_env_var_drives_shape_policy(self, tmp_path: Path, monkeypatch) -> None:
        """COMPGEN_SHAPE_POLICY=class makes the materializer
        substitute dims with None."""
        monkeypatch.setenv("COMPGEN_SHAPE_POLICY", "class")
        result = subprocess.run(
            [
                sys.executable, "-m", "compgen.graph_compilation", "run",
                "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
                "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
                "--out", str(tmp_path / "run"),
                "--stop-after", "kernel-codegen-request",
                "--selection-mode", "greedy",
                "--auction-mode", "disabled",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
            env={**os.environ, "COMPGEN_SHAPE_POLICY": "class"},
        )
        assert result.returncode == 0, result.stderr

        contracts = list(
            (tmp_path / "run" / "04_kernel_codegen" / "contracts").glob("*.json")
        )
        body = json.loads(contracts[0].read_text())
        # All input/output dims are None.
        for tio_kind in ("inputs", "outputs"):
            for tio in body["io"][tio_kind]:
                dims = tio["shape"]["dims"]
                assert all(d is None for d in dims), \
                    f"dims should be None in class mode; got {dims}"
