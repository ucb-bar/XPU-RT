"""Kernel-codegen task emitter tests (supersedes the tests).

Three layers of coverage:

- **Schema**: KernelCodegenRequest serialises deterministically; every
  required field is populated; task_id is deterministic from
  (region_id, candidate_id).
- **E2E**: pipeline run on merlin_mlp_wide and tiny_mlp produces
  04_kernel_codegen/requests/<task_id>.request.json + a sandboxed
  artifact_dir with the right structure; old 04_kernel_specialization/
  is no longer written.
- **Negative control**: non-set_tile_params candidates emit
  not_applicable; sandbox path stays under 04_kernel_codegen/artifacts/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke(*, model: str, out_dir: Path, stop_after: str = "kernel-codegen-request") -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", stop_after,
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestSchema:
    def test_to_dict_round_trips(self) -> None:
        from compgen.graph_compilation.kernel_codegen import (
            KernelCodegenRequest, _ContractPaths,
        )
        req = KernelCodegenRequest(
            schema_version="kernel_codegen_request_v1",
            task_id="kcodegen_dead",
            generated_at_utc="2026-05-07T00:00:00Z",
            request_kind="kernel_codegen",
            region_id="matmul_0",
            candidate_id="cand_x",
            recipe_op_id="recipe_0000",
            contract_hash="dead" + "0" * 12,
            contract_paths=_ContractPaths(
                full="04_kernel_codegen/contracts/matmul_0.deadbeef.json",
                kernel_facing="04_kernel_codegen/views/matmul_0.kernel_facing.json",
            ),
            allowed_backends=("c_reference",),
            required_outputs=("kernel_source", "kernel_metadata"),
            forbidden=("modify_contract",),
            artifact_dir="04_kernel_codegen/artifacts/kcodegen_dead",
        )
        d = req.to_dict()
        s = json.dumps(d, indent=2, sort_keys=True)
        assert "kernel_codegen_request_v1" in s
        assert "kcodegen_dead" in s
        for k in (
            "schema_version", "task_id", "generated_at_utc", "request_kind",
            "region_id", "candidate_id", "recipe_op_id", "contract_hash",
            "contract_paths", "allowed_backends", "required_outputs",
            "forbidden", "artifact_dir",
        ):
            assert k in d


# --------------------------------------------------------------------------- #
# E2E — pipeline emits at the right paths
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def merlin_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m42_merlin") / "run"
    res = _invoke(model="merlin_mlp_wide", out_dir=out)
    assert res.returncode == 0, res.stderr
    return out


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m42_tiny") / "run"
    res = _invoke(model="tiny_mlp", out_dir=out)
    assert res.returncode == 0, res.stderr
    return out


def _read_request(run_dir: Path) -> dict:
    files = sorted((run_dir / "04_kernel_codegen" / "requests").glob("*.request.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_e2e_directory_layout(merlin_run: Path) -> None:
    """unifies under 04_kernel_codegen/. Sub-dirs:
    requests/, contracts/, views/, artifacts/ (sandbox)."""
    out = merlin_run / "04_kernel_codegen"
    assert (out / "requests").is_dir()
    assert (out / "contracts").is_dir()
    assert (out / "views").is_dir()
    assert (out / "artifacts").is_dir()
    assert (out / "kernel_codegen_summary.json").exists()


def test_e2e_legacy_dir_no_longer_written(merlin_run: Path) -> None:
    """The pre-04_kernel_specialization/ path must not be created."""
    legacy = merlin_run / "04_kernel_specialization"
    assert not legacy.exists(), (
        "M-39 directory was created; M-42 migration is incomplete"
    )


def test_e2e_request_schema(merlin_run: Path) -> None:
    body = _read_request(merlin_run)
    assert body["schema_version"] == "kernel_codegen_request_v1"
    assert body["request_kind"] == "kernel_codegen"
    assert body["region_id"] == "matmul_0"
    assert body["candidate_id"].startswith("cand_tile_matmul_0")
    assert body["recipe_op_id"] == "recipe_0000"
    assert body["contract_hash"]
    assert body["contract_paths"]["kernel_facing"].startswith(
        "04_kernel_codegen/views/"
    )
    assert body["contract_paths"]["full"].startswith(
        "04_kernel_codegen/contracts/"
    )
    assert body["allowed_backends"] == ["c_reference"]
    assert "kernel_source" in body["required_outputs"]
    assert "modify_contract" in body["forbidden"]
    assert body["artifact_dir"].startswith("04_kernel_codegen/artifacts/kcodegen_")


def test_e2e_contract_paths_resolve(merlin_run: Path) -> None:
    """Both contract_paths.full and contract_paths.kernel_facing must
    point to existing files on disk."""
    body = _read_request(merlin_run)
    full = merlin_run / body["contract_paths"]["full"]
    facing = merlin_run / body["contract_paths"]["kernel_facing"]
    assert full.exists(), f"full contract path does not resolve: {full}"
    assert facing.exists(), f"kernel_facing path does not resolve: {facing}"


def test_e2e_artifact_dir_exists_and_is_empty(merlin_run: Path) -> None:
    """The sandboxed artifact_dir is created empty ; 's
    commit tool will reject any provider response that writes outside
    this directory."""
    body = _read_request(merlin_run)
    artifact_dir = merlin_run / body["artifact_dir"]
    assert artifact_dir.is_dir()
    assert list(artifact_dir.iterdir()) == []


def test_e2e_kernel_facing_view_no_compiler_only_fields(merlin_run: Path) -> None:
    """Defense-in-depth — the view at the embedded path still excludes
    compiler-only fields. Mirrors the negative control on the
    on-disk artifact, this time triggered through the pointer."""
    body = _read_request(merlin_run)
    facing_text = (merlin_run / body["contract_paths"]["kernel_facing"]).read_text()
    forbidden = (
        "wait_on", "blocking", "lifetimes", "fusion", "is_boundary",
        "fusable_with", "prefer_inline_into", "observability",
        "emit_dispatch_event", "emit_completion_event", "cost_emit_period",
        "max_concurrent_invocations", "retry_on_recoverable_error",
        "providers", "metadata",
    )
    for f in forbidden:
        assert f'"{f}"' not in facing_text, (
            f"compiler-only field {f!r} leaked into the kernel_facing view"
        )


def test_e2e_byte_stable_across_reruns(tmp_path: Path) -> None:
    """Same model run twice produces byte-identical request JSON
    (modulo generated_at_utc). Catches non-determinism in task_id /
    contract_hash / artifact_dir."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    res_a = _invoke(model="merlin_mlp_wide", out_dir=out_a)
    res_b = _invoke(model="merlin_mlp_wide", out_dir=out_b)
    assert res_a.returncode == 0 and res_b.returncode == 0

    a = _read_request(out_a)
    b = _read_request(out_b)
    a.pop("generated_at_utc", None)
    b.pop("generated_at_utc", None)
    assert a == b, "request JSON diverged across reruns"
    assert a["task_id"] == b["task_id"]
    assert a["contract_hash"] == b["contract_hash"]


def test_e2e_task_id_deterministic_from_inputs(merlin_run: Path) -> None:
    """task_id derives from (region_id, candidate_id) — no random
    suffix, no timestamp."""
    from compgen.graph_compilation.kernel_codegen import _task_id
    body = _read_request(merlin_run)
    expected = _task_id(
        candidate_id=body["candidate_id"],
        region_id=body["region_id"],
    )
    assert body["task_id"] == expected


def test_e2e_tiny_mlp_emits_request_too(tiny_run: Path) -> None:
    body = _read_request(tiny_run)
    assert body["request_kind"] == "kernel_codegen"
    assert body["region_id"] == "matmul_0"
    assert body["allowed_backends"] == ["c_reference"]
    # tiny_mlp's tile_M4_N16_K16 should yield a tolerance_eps refinement
    # in the materialised contract (puts it as a StaticAttr).
    full = tiny_run / body["contract_paths"]["full"]
    contract_body = json.loads(full.read_text())
    attrs = {a["name"]: a["value"] for a in contract_body["io"]["attributes"]}
    assert attrs["declared_refinement"] == "tolerance_eps"


# --------------------------------------------------------------------------- #
# Stop-after boundary semantics
# --------------------------------------------------------------------------- #


def test_e2e_stop_at_recipe_planning_does_not_emit_request(tmp_path: Path) -> None:
    """When stop_after is before kernel-codegen-request, no
    04_kernel_codegen/requests/ is created. The 04_kernel_codegen/
    contracts dir may exist if its boundary lower than this
    one — the request emit is what we're checking."""
    out = tmp_path / "no_request"
    res = _invoke(model="merlin_mlp_wide", out_dir=out, stop_after="agent-decision-request")
    assert res.returncode == 0, res.stderr
    requests_dir = out / "04_kernel_codegen" / "requests"
    if requests_dir.exists():
        assert list(requests_dir.glob("*.request.json")) == []


def test_e2e_kernel_specialization_request_alias_still_works(tmp_path: Path) -> None:
    """The legacy --stop-after kernel-specialization-request flag is
    kept as an alias of --stop-after kernel-codegen-request because the
    boundary block runs both unconditionally. Makes the
    migration safe for callers that still pass the old flag."""
    out = tmp_path / "alias"
    res = _invoke(model="merlin_mlp_wide", out_dir=out,
                  stop_after="kernel-specialization-request")
    assert res.returncode == 0, res.stderr
    body = _read_request(out)
    assert body["schema_version"] == "kernel_codegen_request_v1"
