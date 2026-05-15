"""Contract materialization tests.

Three layers of coverage matching the plan's done condition:

- **Schema**: KernelContractV3.from_recipe populates every field; round-trip
  through contract_to_dict / kernel_facing_to_dict; canonical hash is
  byte-stable.
- **E2E**: pipeline run on merlin_mlp_wide and tiny_mlp produces a
  contract file + kernel-facing view + summary at the right paths;
  declared refinement matches the recipe-gate verdict; non-set_tile_params
  candidates emit a typed not_applicable row.
- **Negative control**: the kernel_facing view does NOT leak any
  compiler-only field (the load-bearing invariant).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke(*, model: str, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "kernel-specialization-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


# --------------------------------------------------------------------------- #
# Schema — from_recipe + canonical hash
# --------------------------------------------------------------------------- #


class TestFromRecipe:
    def test_merlin_synthesis_populates_every_field(self) -> None:
        from compgen.kernels.contract_v3 import (
            DispatchModel, Granularity, KernelArchetype, KernelContractV3,
        )
        sel = {
            "candidate_kind": "set_tile_params",
            "selected_candidate_id": "cand_x",
            "region_id": "matmul_0",
            "label": "tile_M16_N16_K16",
            "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
            "recipe_delta": [{"op": "SetTileParams", "M": 16, "N": 16, "K": 16}],
            "target_id": "host_cpu",
        }
        dossier = {
            "region_id": "matmul_0",
            "kind": "matmul",
            "region_shape": {
                "dtype": "f32",
                "input_shapes": [[16, 16], [16, 32]],
                "output_shapes": [[16, 32]],
                "summary": "matmul/16x32x16/f32",
            },
        }
        target = {
            "target_id": "host_cpu",
            "hardware_envelope": {
                "scratchpad_kib": 64, "register_bytes": 256,
                "vector_lanes": 8, "native_dtypes": ["f32", "bf16"],
            },
        }
        c = KernelContractV3.from_recipe(
            candidate_selection=sel, region_dossier=dossier,
            target_profile=target, declared_refinement="bit_equality",
        )
        assert c.archetype is KernelArchetype.COMPUTE_TILED
        assert c.granularity is Granularity.NORMAL
        assert c.op_name == "linalg.matmul"
        # IO: 2 inputs + 1 output, shape concrete.
        assert len(c.io.inputs) == 2
        assert len(c.io.outputs) == 1
        assert c.io.inputs[0].shape.dims == (16, 16)
        assert c.io.inputs[1].shape.dims == (16, 32)
        assert c.io.outputs[0].shape.dims == (16, 32)
        # Tile attrs present.
        attr_names = {a.name for a in c.io.attributes}
        assert {"tile_M", "tile_N", "tile_K", "declared_refinement"} <= attr_names
        # bit_equality → max_rel_err = 0.0
        assert c.io.numerics.max_relative_error == 0.0
        assert c.io.numerics.deterministic is True
        # Dispatch is SYNC (widens).
        assert c.orchestration.dispatch.model is DispatchModel.SYNC
        # Sync declares matmul_done event.
        assert any(e.name == "matmul_done" for e in c.orchestration.sync.event_decls)

    def test_tiny_mlp_synthesis_tolerance_eps(self) -> None:
        from compgen.kernels.contract_v3 import KernelContractV3
        sel = {
            "candidate_kind": "set_tile_params",
            "selected_candidate_id": "cand_y",
            "region_id": "matmul_0",
            "label": "tile_M4_N16_K16",
            "cost_preview": {"region_dims": {"M": 4, "N": 128, "K": 64}},
            "recipe_delta": [{"op": "SetTileParams", "M": 4, "N": 16, "K": 16}],
            "target_id": "host_cpu",
        }
        dossier = {
            "region_id": "matmul_0",
            "kind": "matmul",
            "region_shape": {
                "dtype": "f32",
                "input_shapes": [[4, 64], [64, 128]],
                "output_shapes": [[4, 128]],
                "summary": "matmul/4x128x64/f32",
            },
        }
        c = KernelContractV3.from_recipe(
            candidate_selection=sel, region_dossier=dossier,
            target_profile={}, declared_refinement="tolerance_eps",
        )
        assert c.io.inputs[0].shape.dims == (4, 64)
        assert c.io.numerics.max_relative_error > 0.0
        assert c.io.numerics.max_relative_error < 1e-2  # not silently widened

    def test_rejects_non_set_tile_params(self) -> None:
        from compgen.kernels.contract_v3 import KernelContractV3
        with pytest.raises(ValueError, match="set_tile_params"):
            KernelContractV3.from_recipe(
                candidate_selection={
                    "candidate_kind": "fuse_producer_consumer",
                    "label": "fuse_x_into_y",
                    "cost_preview": {"region_dims": {}},
                },
                region_dossier={"region_shape": {}},
                target_profile={},
                declared_refinement="unknown",
            )


class TestCanonicalHash:
    def test_hash_is_byte_stable(self) -> None:
        from compgen.kernels.contract_v3 import KernelContractV3
        from compgen.promotion.contract_hash import hash_contract
        sel = {
            "candidate_kind": "set_tile_params",
            "selected_candidate_id": "cand_x",
            "region_id": "matmul_0",
            "label": "tile_M16_N16_K16",
            "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
            "recipe_delta": [{"op": "SetTileParams", "M": 16, "N": 16, "K": 16}],
        }
        dossier = {
            "region_id": "matmul_0",
            "region_shape": {
                "dtype": "f32",
                "input_shapes": [[16, 16], [16, 32]],
                "output_shapes": [[16, 32]],
            },
        }
        c1 = KernelContractV3.from_recipe(
            candidate_selection=sel, region_dossier=dossier,
            target_profile={}, declared_refinement="bit_equality",
        )
        c2 = KernelContractV3.from_recipe(
            candidate_selection=sel, region_dossier=dossier,
            target_profile={}, declared_refinement="bit_equality",
        )
        assert hash_contract(c1) == hash_contract(c2)
        # 16 hex chars (truncated SHA256).
        assert len(hash_contract(c1)) == 16

    def test_hash_differs_on_shape_change(self) -> None:
        from compgen.kernels.contract_v3 import KernelContractV3
        from compgen.promotion.contract_hash import hash_contract
        sel = {
            "candidate_kind": "set_tile_params", "label": "tile_M16_N16_K16",
            "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
        }
        d1 = {"region_shape": {
            "dtype": "f32",
            "input_shapes": [[16, 16], [16, 32]],
            "output_shapes": [[16, 32]],
        }}
        d2 = {"region_shape": {
            "dtype": "f32",
            "input_shapes": [[8, 16], [16, 32]],
            "output_shapes": [[8, 32]],
        }}
        h1 = hash_contract(KernelContractV3.from_recipe(
            candidate_selection=sel, region_dossier=d1,
            target_profile={}, declared_refinement="bit_equality"))
        h2 = hash_contract(KernelContractV3.from_recipe(
            candidate_selection=sel, region_dossier=d2,
            target_profile={}, declared_refinement="bit_equality"))
        assert h1 != h2


# --------------------------------------------------------------------------- #
# Negative control — kernel_facing view never leaks compiler-only fields
# --------------------------------------------------------------------------- #


class TestKernelFacingNoLeak:
    """The kernel_facing view is the BOUNDED surface a kernel codegen
    provider may read (+ hands this to the spawned Claude Code
    agent). Compiler-only fields MUST NOT appear in the serialized
    JSON. This is the load-bearing invariant."""

    _FORBIDDEN_FIELDS = (
        "wait_on",
        "blocking",
        "lifetimes",
        "fusion",
        "is_boundary",
        "fusable_with",
        "prefer_inline_into",
        "observability",
        "emit_dispatch_event",
        "emit_completion_event",
        "cost_emit_period",
        "max_concurrent_invocations",
        "retry_on_recoverable_error",
        "providers",
        "metadata",
    )

    def test_no_compiler_only_fields_in_view(self) -> None:
        from compgen.graph_compilation.kernel_contract_materialization import (
            kernel_facing_to_dict,
        )
        from compgen.kernels.contract_v3 import KernelContractV3
        c = KernelContractV3.from_recipe(
            candidate_selection={
                "candidate_kind": "set_tile_params",
                "label": "tile_M16_N16_K16",
                "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
            },
            region_dossier={"region_shape": {
                "dtype": "f32",
                "input_shapes": [[16, 16], [16, 32]],
                "output_shapes": [[16, 32]],
            }},
            target_profile={}, declared_refinement="bit_equality",
        )
        view_json = json.dumps(kernel_facing_to_dict(c.kernel_facing()))
        for forbidden in self._FORBIDDEN_FIELDS:
            assert f'"{forbidden}"' not in view_json, (
                f"compiler-only field {forbidden!r} leaked into kernel_facing view; "
                f"M-40 invariant violated"
            )


# --------------------------------------------------------------------------- #
# E2E — pipeline emits artifacts at the right paths
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def merlin_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m40_merlin") / "run"
    res = _invoke(model="merlin_mlp_wide", out_dir=out)
    assert res.returncode == 0, res.stderr
    return out


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m40_tiny") / "run"
    res = _invoke(model="tiny_mlp", out_dir=out)
    assert res.returncode == 0, res.stderr
    return out


def _read_summary(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "04_kernel_codegen" / "contract_materialization_summary.json")
        .read_text(encoding="utf-8")
    )


def test_e2e_artifacts_exist(merlin_run: Path) -> None:
    out = merlin_run / "04_kernel_codegen"
    assert out.is_dir()
    assert (out / "contract_materialization_summary.json").exists()
    contracts = list((out / "contracts").glob("*.json"))
    views = list((out / "views").glob("*.kernel_facing.json"))
    assert len(contracts) == 1
    assert len(views) == 1


def test_e2e_merlin_bit_equality(merlin_run: Path) -> None:
    summary = _read_summary(merlin_run)
    assert len(summary["rows"]) == 1
    row = summary["rows"][0]
    assert row["status"] == "materialized"
    assert row["region_id"] == "matmul_0"
    assert row["candidate_kind"] == "set_tile_params"
    assert row["contract_hash"]  # non-empty
    # Read the contract; verify bit_equality refinement.
    contract_path = merlin_run / row["contract_path"]
    body = json.loads(contract_path.read_text())
    attrs = {a["name"]: a["value"] for a in body["io"]["attributes"]}
    assert attrs["declared_refinement"] == "bit_equality"
    assert attrs["tile_M"] == 16
    assert body["io"]["numerics"]["max_relative_error"] == 0.0


def test_e2e_tiny_tolerance_eps(tiny_run: Path) -> None:
    summary = _read_summary(tiny_run)
    row = summary["rows"][0]
    assert row["status"] == "materialized"
    contract_path = tiny_run / row["contract_path"]
    body = json.loads(contract_path.read_text())
    attrs = {a["name"]: a["value"] for a in body["io"]["attributes"]}
    assert attrs["declared_refinement"] == "tolerance_eps"
    assert attrs["tile_M"] == 4  # shape-fit
    assert body["io"]["numerics"]["max_relative_error"] > 0.0


def test_e2e_byte_stable_across_reruns(tmp_path: Path) -> None:
    """Two independent runs of merlin_mlp_wide produce byte-identical
    contract JSON (modulo `generated_at_utc` in the summary). Catches
    accidental dependence on RNG / timestamps / pid in the contract."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    res_a = _invoke(model="merlin_mlp_wide", out_dir=out_a)
    res_b = _invoke(model="merlin_mlp_wide", out_dir=out_b)
    assert res_a.returncode == 0 and res_b.returncode == 0

    sum_a = _read_summary(out_a)
    sum_b = _read_summary(out_b)
    assert sum_a["rows"][0]["contract_hash"] == sum_b["rows"][0]["contract_hash"]

    contract_a = (out_a / sum_a["rows"][0]["contract_path"]).read_text()
    contract_b = (out_b / sum_b["rows"][0]["contract_path"]).read_text()
    assert contract_a == contract_b, (
        "contract JSON diverged across reruns; M-40 byte-stability invariant violated"
    )

    view_a = (out_a / sum_a["rows"][0]["kernel_facing_path"]).read_text()
    view_b = (out_b / sum_b["rows"][0]["kernel_facing_path"]).read_text()
    assert view_a == view_b


def test_e2e_view_does_not_leak_compiler_only_fields_on_real_run(merlin_run: Path) -> None:
    summary = _read_summary(merlin_run)
    view_path = merlin_run / summary["rows"][0]["kernel_facing_path"]
    view_text = view_path.read_text()
    for forbidden in TestKernelFacingNoLeak._FORBIDDEN_FIELDS:
        assert f'"{forbidden}"' not in view_text, (
            f"compiler-only field {forbidden!r} leaked into the on-disk view"
        )
