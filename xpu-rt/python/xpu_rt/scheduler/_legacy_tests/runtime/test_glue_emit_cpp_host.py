"""emitted C++ host (CUDA) plan-executor tests.

Coverage parallels the test suite plus a **differential** check
that the C++ emit dispatches the same regions in the same order as
the C11 emit when given the same plan (the D6
plan-refinement gate generalises this).

The fully-built differential against the Python CUDA emitter
on real CUDA hardware lives in
``tests/native/test_cpp_host_cuda_integration.py`` (gated on the
``cuda`` driver being available at build time).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from compgen.runtime.execution_plan import (
    DependencyEdge,
    ExecutionPlan,
    RegionKernelBinding,
    RegionPlacement,
    Resource,
)
from compgen.runtime.glue_emit import (
    emit_c11_baremetal_executor,
    emit_cpp_host_executor,
)
from compgen.runtime.glue_emit.c11_baremetal import _PLAN_VIOLATION_CODES


# --------------------------------------------------------------------------- #
# Helpers (same shape as 's) #
# --------------------------------------------------------------------------- #


def _emit_certs(run_dir: Path, bindings: list[RegionKernelBinding]) -> None:
    for b in bindings:
        cert_path = run_dir / b.certificate_path
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        cert_path.write_text(json.dumps({
            "schema_version": "kernel_certificate_v1",
            "contract_hash": b.contract_hash,
            "task_id": "t", "region_id": b.region_id, "candidate_id": "c",
            "accepted_at_utc": "x", "artifact_hashes": {},
            "artifact_paths": {}, "verifier_report_path": "",
            "verifier_report_hash": "", "claims": {},
        }))


def _write_plan(run_dir: Path, plan: ExecutionPlan) -> None:
    plan_dir = run_dir / "05_execution_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_dict = plan.to_dict()
    try:
        import yaml  # type: ignore[import-untyped]
        (plan_dir / "execution_plan.yaml").write_text(
            yaml.safe_dump(plan_dict, sort_keys=True), encoding="utf-8",
        )
    except ImportError:
        (plan_dir / "execution_plan.json").write_text(
            json.dumps(plan_dict, sort_keys=True), encoding="utf-8",
        )


def _make_run_dir(
    tmp_path: Path,
    bindings: list[RegionKernelBinding],
    placements: list[RegionPlacement] | None = None,
    edges: list[DependencyEdge] | None = None,
    target: str = "cuda",
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _emit_certs(run_dir, bindings)
    placements = placements or [
        RegionPlacement(region_id=b.region_id, device=target, queue="q")
        for b in bindings
    ]
    plan = ExecutionPlan(
        workload="test", target=target,
        resources=[Resource(id="q", kind="compute", device=target)],
        region_placement=placements,
        dependency_edges=edges or [],
        region_kernel_bindings=bindings,
    )
    plan.validate()
    _write_plan(run_dir, plan)
    return run_dir


def _find_cxx() -> str | None:
    for cxx in ("c++", "g++", "clang++"):
        path = shutil.which(cxx)
        if path:
            return path
    return None


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #


class TestEmitProducesArtifacts:
    def test_cpp_executor_header_manifest_emitted(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_cpp_host_executor(run_dir)
        assert result.executor_path.name == "generated_plan_executor.cpp"
        assert result.header_path.name == "generated_plan_executor.h"
        assert result.manifest_path.name == "plan_executor_cpp_host_manifest.json"
        assert result.executor_path.exists()
        assert result.header_path.exists()
        assert result.manifest_path.exists()

    def test_manifest_carries_cuda_driver(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_cpp_host_executor(run_dir)
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["executor_kind"] == "cpp_host"
        assert manifest["abi"]["driver_name"] == "cuda"
        assert manifest["abi"]["uses_only_cg_rt"] is True
        assert manifest["abi"]["push_constants_layout"] == (
            "u32[6]:grid_xyz,block_xyz"
        )


# --------------------------------------------------------------------------- #
# Byte stability                                                              #
# --------------------------------------------------------------------------- #


class TestByteStability:
    def test_two_emits_match(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        r1 = emit_cpp_host_executor(run_dir)
        c1 = r1.executor_path.read_bytes()
        r2 = emit_cpp_host_executor(run_dir)
        c2 = r2.executor_path.read_bytes()
        assert c1 == c2


# --------------------------------------------------------------------------- #
# C++ syntax (compiler-gated)                                                 #
# --------------------------------------------------------------------------- #


class TestCppSyntax:
    @pytest.mark.skipif(_find_cxx() is None, reason="no C++ compiler in PATH")
    def test_emit_parses_with_cxx_fsyntax_only(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_cpp_host_executor(run_dir)
        cxx = _find_cxx()
        assert cxx is not None
        repo_root = Path(__file__).resolve().parents[2]
        rt_include = repo_root / "runtime" / "native" / "libcompgen_rt" / "include"
        emit_dir = result.executor_path.parent
        proc = subprocess.run(
            [
                cxx, "-std=c++17", "-Wall", "-Wextra",
                "-fsyntax-only",
                f"-I{rt_include}", f"-I{emit_dir}",
                str(result.executor_path),
            ],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"emitted .cpp failed c++ -fsyntax-only:\n"
            f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
        )


# --------------------------------------------------------------------------- #
# ABI lint                                                                    #
# --------------------------------------------------------------------------- #


class TestAbiLint:
    def test_only_cg_rt_externs_called(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_cpp_host_executor(run_dir)
        src = result.executor_path.read_text()
        src_no_block = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
        src_no_line = re.sub(r"//[^\n]*", "", src_no_block)
        # Strip C string literals so prose like "foo failed (cuda)"
        # doesn't false-positive as a call to ``failed``.
        src_clean = re.sub(r'"([^"\\]|\\.)*"', '""', src_no_line)
        call_re = re.compile(r"(?<![.>:])\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
        called = {m.group(1) for m in call_re.finditer(src_clean)}
        allowed = (
            "sizeof", "if", "for", "while", "return", "switch", "case",
            "compgen_run", "static_cast", "reinterpret_cast", "const_cast",
        )
        for name in called:
            if name.startswith("cg_rt_"):
                continue
            if name.startswith("compgen_kernel_"):
                continue
            if name in allowed:
                continue
            if name in ("memset", "memcpy"):
                continue
            # Explicitly catch the things the gate is designed to forbid.
            forbidden = (
                "cudaMalloc", "cuLaunchKernel", "cuMemcpy",
                "hipMalloc", "hipLaunchKernel",
                "vkCreateBuffer", "vkCmdDispatch",
            )
            if name in forbidden:
                raise AssertionError(
                    f"emit calls forbidden vendor primitive {name!r}; "
                    f"M-91 ABI-conformance gate would reject this"
                )
            raise AssertionError(
                f"unknown extern {name!r} in C++ emit; add to allowlist "
                f"only with a clear reason"
            )


# --------------------------------------------------------------------------- #
# Differential vs C11 emit (same plan → same region order) #
# --------------------------------------------------------------------------- #


class TestDifferentialAgainstC11Emit:
    def test_same_region_order_as_c11_emit(self, tmp_path: Path) -> None:
        """Both emitters consume the same plan; both must dispatch
        the same regions in the same order. This is a pre-check for
        the D6 plan-refinement gate."""
        bindings = [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
            RegionKernelBinding(
                region_id="r1", contract_hash="h1",
                certificate_path="04_kernel_codegen/certificates/h1.json",
            ),
        ]
        run_dir = _make_run_dir(
            tmp_path, bindings,
            edges=[DependencyEdge(from_region="r0", to_region="r1")],
        )
        c11 = emit_c11_baremetal_executor(run_dir)
        cpp = emit_cpp_host_executor(run_dir)
        c11_manifest = json.loads(c11.manifest_path.read_text())
        cpp_manifest = json.loads(cpp.manifest_path.read_text())
        assert c11_manifest["region_order"] == cpp_manifest["region_order"]
        assert c11_manifest["bound_regions"] == cpp_manifest["bound_regions"]
        assert (
            c11_manifest["plan_violation_codes"]
            == cpp_manifest["plan_violation_codes"]
        )

    def test_dispatch_count_matches_bound_regions(self, tmp_path: Path) -> None:
        bindings = [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
            RegionKernelBinding(
                region_id="r1", contract_hash="h1",
                certificate_path="04_kernel_codegen/certificates/h1.json",
            ),
            RegionKernelBinding(
                region_id="r2", contract_hash="h2",
                certificate_path="04_kernel_codegen/certificates/h2.json",
            ),
        ]
        run_dir = _make_run_dir(tmp_path, bindings)
        result = emit_cpp_host_executor(run_dir)
        src = result.executor_path.read_text()
        # One dispatch per region.
        assert src.count("cg_rt_command_buffer_dispatch(") == 3


# --------------------------------------------------------------------------- #
# Unbound region                                                              #
# --------------------------------------------------------------------------- #


class TestUnboundRegion:
    def test_unbound_region_emits_typed_violation(
        self, tmp_path: Path,
    ) -> None:
        run_dir = _make_run_dir(
            tmp_path, bindings=[],
            placements=[RegionPlacement(
                region_id="r0", device="cuda", queue="q",
            )],
        )
        result = emit_cpp_host_executor(run_dir)
        src = result.executor_path.read_text()
        assert "COMPGEN_PLAN_VIOLATION_UNBOUND_REGION" in src
        assert result.overall == "skipped"


# --------------------------------------------------------------------------- #
# Header coexistence with #
# --------------------------------------------------------------------------- #


class TestHeaderCoexistsWithC11Emit:
    def test_both_emit_into_same_06_glue_emit(self, tmp_path: Path) -> None:
        """Both the C emit and the C++ emit share the same
        header (the ABI is the same); the source files coexist so
        an operator can pick either backend at build time."""
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        c11 = emit_c11_baremetal_executor(run_dir)
        cpp = emit_cpp_host_executor(run_dir)
        assert c11.header_path == cpp.header_path
        assert c11.executor_path.parent == cpp.executor_path.parent
        # Both source files exist.
        assert c11.executor_path.exists()
        assert cpp.executor_path.exists()
        # Manifests are siblings.
        assert c11.manifest_path.parent == cpp.manifest_path.parent
