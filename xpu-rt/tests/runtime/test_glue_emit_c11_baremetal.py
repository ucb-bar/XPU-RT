"""emitted C11 baremetal plan-executor tests.

Coverage:

- Schema     : emit produces .c, .h, and manifest at the right paths.
- Byte stab. : two consecutive emits are byte-identical (sorted-keys
                manifest, fixed dispatch order, no embedded timestamps
                in the source).
- C syntax   : the emit parses with ``cc -fsyntax-only`` when a system
                C compiler is available; skipped otherwise.
- ABI lint   : the emit calls only ``cg_rt_*`` external symbols (the
                D6 ABI-conformance gate's static pre-check).
- Topology   : multi-region emits respect the dependency-edge order
                that already enforces.
- Unbound    : a plan with no bindings emits the unbound check that
                returns ``COMPGEN_PLAN_VIOLATION_UNBOUND_REGION``.
- Codes      : ``COMPGEN_PLAN_VIOLATION_<KIND>`` codes are present in
                the header and the manifest, with stable integer
                values.
- Assertions : contract dtype/shape/bytes/layout all generate matching
                C-level checks; predicate kinds (mod_eq, byte_size_le,
                dtype_in) emit named checks with correct codes.

The C compilation acceptance — actually building the emitted ``.c``
against ``libcompgen_rt`` and matching the Python SYNC executor
bit-for-bit on ``proxy_vla`` — runs as an integration test in
``tests/native/test_c11_baremetal_integration.py`` (gated behind a
local C toolchain + libcompgen_rt headers being on the include path).
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
from compgen.runtime.glue_emit import emit_c11_baremetal_executor
from compgen.runtime.glue_emit.c11_baremetal import _PLAN_VIOLATION_CODES


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _emit_certs(run_dir: Path, bindings: list[RegionKernelBinding]) -> None:
    if not bindings:
        return
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


def _emit_contract_with_inputs(
    run_dir: Path,
    region_id: str,
    contract_hash: str,
    inputs: list[dict],
    preconditions: list[dict] | None = None,
) -> None:
    contracts_dir = run_dir / "04_kernel_codegen" / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": "kernel_contract_v3",
        "region_id": region_id,
        "contract_hash": contract_hash,
        "io": {"inputs": inputs, "outputs": []},
        "preconditions": preconditions or [],
    }
    path = contracts_dir / f"{region_id}.{contract_hash}.json"
    path.write_text(json.dumps(body))


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
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _emit_certs(run_dir, bindings)
    placements = placements or [
        RegionPlacement(region_id=b.region_id, device="host_cpu", queue="q")
        for b in bindings
    ]
    plan = ExecutionPlan(
        workload="test", target="host_cpu",
        resources=[Resource(id="q", kind="compute", device="host_cpu")],
        region_placement=placements,
        dependency_edges=edges or [],
        region_kernel_bindings=bindings,
    )
    plan.validate()
    _write_plan(run_dir, plan)
    return run_dir


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #


class TestEmitProducesArtifacts:
    def test_executor_header_and_manifest_emitted(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_c11_baremetal_executor(run_dir)
        assert result.executor_path.name == "generated_plan_executor.c"
        assert result.header_path.name == "generated_plan_executor.h"
        assert result.manifest_path.name == "plan_executor_c11_manifest.json"
        assert result.executor_path.exists()
        assert result.header_path.exists()
        assert result.manifest_path.exists()

    def test_manifest_schema_fields_present(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_c11_baremetal_executor(run_dir)
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["schema_version"] == "plan_executor_c11_manifest_v1"
        assert manifest["executor_kind"] == "c11_baremetal"
        assert manifest["bound_regions"] == ["r0"]
        assert manifest["region_order"] == ["r0"]
        assert manifest["abi"]["driver_name"] == "cpu_sync"
        assert manifest["abi"]["uses_only_cg_rt"] is True
        assert manifest["abi"]["kernel_extern_prefix"] == "compgen_kernel_"
        # All PLAN_VIOLATION codes are present and have integer values.
        codes = manifest["plan_violation_codes"]
        for name, val in _PLAN_VIOLATION_CODES:
            assert codes[name] == val

    def test_missing_plan_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="execution plan not found"):
            emit_c11_baremetal_executor(tmp_path)


# --------------------------------------------------------------------------- #
# Byte stability                                                              #
# --------------------------------------------------------------------------- #


class TestByteStability:
    def test_two_emits_match_byte_for_byte(self, tmp_path: Path) -> None:
        """Source files are deterministic across reruns. The manifest
        carries a timestamp, so it is stripped from the comparison."""
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        r1 = emit_c11_baremetal_executor(run_dir)
        c1 = r1.executor_path.read_bytes()
        h1 = r1.header_path.read_bytes()
        m1 = json.loads(r1.manifest_path.read_text())
        m1.pop("generated_at_utc", None)

        r2 = emit_c11_baremetal_executor(run_dir)
        c2 = r2.executor_path.read_bytes()
        h2 = r2.header_path.read_bytes()
        m2 = json.loads(r2.manifest_path.read_text())
        m2.pop("generated_at_utc", None)

        assert c1 == c2, "executor.c must be byte-stable across reruns"
        assert h1 == h2, "executor.h must be byte-stable across reruns"
        assert m1 == m2, "manifest payload must be byte-stable across reruns"


# --------------------------------------------------------------------------- #
# C syntax (compiler-gated)                                                   #
# --------------------------------------------------------------------------- #


def _find_cc() -> str | None:
    for cc in ("cc", "gcc", "clang"):
        path = shutil.which(cc)
        if path:
            return path
    return None


class TestCSyntax:
    @pytest.mark.skipif(_find_cc() is None, reason="no C compiler in PATH")
    def test_emit_parses_with_cc_fsyntax_only(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_c11_baremetal_executor(run_dir)
        cc = _find_cc()
        assert cc is not None
        # Point at the real libcompgen_rt header; if it's not there the
        # test still passes provided the .c is syntactically valid
        # against a stub. Try the real include path first.
        repo_root = Path(__file__).resolve().parents[2]
        rt_include = repo_root / "runtime" / "native" / "libcompgen_rt" / "include"
        emit_dir = result.executor_path.parent
        proc = subprocess.run(
            [
                cc, "-std=c11", "-Wall", "-Wextra",
                "-fsyntax-only",
                f"-I{rt_include}", f"-I{emit_dir}",
                str(result.executor_path),
            ],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"emitted .c failed cc -fsyntax-only:\n"
            f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
        )


# --------------------------------------------------------------------------- #
# ABI lint — pre-check D6 #
# --------------------------------------------------------------------------- #


class TestAbiLint:
    def test_only_cg_rt_externs_called(self, tmp_path: Path) -> None:
        """Static lint: every function-call symbol in the emit is
        either ``cg_rt_*`` (libcompgen_rt) or a local/builtin name.
        The D6 ABI-conformance gate enforces this end-to-end."""
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_c11_baremetal_executor(run_dir)
        src = result.executor_path.read_text()
        # Strip C block comments + line comments + string literals so
        # prose like ``the cg_rt_* ABI (which calls...)`` and
        # ``"foo failed (cuda)"`` don't false-positive as calls.
        src_no_block = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
        src_no_line = re.sub(r"//[^\n]*", "", src_no_block)
        src_clean = re.sub(r'"([^"\\]|\\.)*"', '""', src_no_line)
        # Match ``identifier(`` not preceded by ``.`` (struct field) or
        # ``->`` (pointer field). This is a coarse lint but sufficient
        # for the gate.
        call_re = re.compile(r"(?<![.>])\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
        called = {m.group(1) for m in call_re.finditer(src_clean)}
        # Allowlist: cg_rt_*, compgen_run, kernel externs, libc, control
        # flow, sizeof, etc. The ABI gate's intent is "no vendor primitive
        # bypasses libcompgen_rt"; cudaMalloc, hipMalloc, vkCreate*, etc.
        # would appear in this set if present.
        allowed = (
            "sizeof", "if", "for", "while", "return", "switch", "case",
            "compgen_run",  # the entry point itself
        )
        for name in called:
            if name.startswith("cg_rt_"):
                continue
            if name.startswith("compgen_kernel_"):
                continue  # kernel pack externs
            if name in allowed:
                continue
            # Bare libc names we don't actually call but the parser may
            # see in macro expansions; tighten as needed.
            if name in ("memset", "memcpy"):
                continue
            raise AssertionError(
                f"emit calls non-cg_rt extern {name!r}; the M-91 "
                f"ABI-conformance gate would reject this. "
                f"Add an allowlist entry only with a clear reason."
            )


# --------------------------------------------------------------------------- #
# Topology                                                                    #
# --------------------------------------------------------------------------- #


class TestTopology:
    def test_multi_region_dispatch_in_topological_order(
        self, tmp_path: Path,
    ) -> None:
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
        result = emit_c11_baremetal_executor(run_dir)
        src = result.executor_path.read_text()
        # r0 must dispatch before r1.
        idx0 = src.find("compgen_kernel_r0")
        idx1 = src.find("compgen_kernel_r1")
        assert idx0 != -1 and idx1 != -1, src
        assert idx0 < idx1, "r0 must be dispatched before r1"

        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["region_order"] == ["r0", "r1"]


# --------------------------------------------------------------------------- #
# Unbound region                                                              #
# --------------------------------------------------------------------------- #


class TestUnboundRegion:
    def test_unbound_region_emits_typed_violation_path(
        self, tmp_path: Path,
    ) -> None:
        run_dir = _make_run_dir(
            tmp_path, bindings=[],
            placements=[RegionPlacement(
                region_id="r0", device="host_cpu", queue="q",
            )],
        )
        result = emit_c11_baremetal_executor(run_dir)
        src = result.executor_path.read_text()
        assert "COMPGEN_PLAN_VIOLATION_UNBOUND_REGION" in src
        assert result.overall == "skipped"
        assert result.unbound_regions == ("r0",)


# --------------------------------------------------------------------------- #
# Plan-violation codes                                                        #
# --------------------------------------------------------------------------- #


class TestPlanViolationCodes:
    def test_codes_in_header_and_manifest(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_c11_baremetal_executor(run_dir)
        header = result.header_path.read_text()
        manifest = json.loads(result.manifest_path.read_text())
        for name, val in _PLAN_VIOLATION_CODES:
            assert f"#define COMPGEN_PLAN_VIOLATION_{name} ({val})" in header
            assert manifest["plan_violation_codes"][name] == val

    def test_io_null_path_present(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_c11_baremetal_executor(run_dir)
        src = result.executor_path.read_text()
        assert "COMPGEN_PLAN_VIOLATION_IO_NULL" in src


# --------------------------------------------------------------------------- #
# Contract-driven assertions                                                  #
# --------------------------------------------------------------------------- #


class TestContractAssertions:
    def test_dtype_shape_bytes_layout_checks_emitted(
        self, tmp_path: Path,
    ) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        _emit_contract_with_inputs(
            run_dir, "r0", "h0",
            [{
                "name": "x",
                "shape": {"dims": [4, 8]},
                "dtype_class": ["f32"],
                "layout": "row_major",
            }],
        )
        result = emit_c11_baremetal_executor(run_dir)
        src = result.executor_path.read_text()
        assert "COMPGEN_PLAN_VIOLATION_INPUT_DTYPE" in src
        assert "COMPGEN_PLAN_VIOLATION_INPUT_SHAPE" in src
        assert "COMPGEN_PLAN_VIOLATION_INPUT_BYTES" in src
        assert "COMPGEN_PLAN_VIOLATION_LAYOUT" in src
        # 4*8*4 = 128 bytes for f32.
        assert "(size_t)128" in src
        # rank = 2.
        assert "rank != 2" in src

    def test_mod_eq_precondition_emitted(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        _emit_contract_with_inputs(
            run_dir, "r0", "h0",
            [{
                "name": "x",
                "shape": {"dims": [4, 64]},
                "dtype_class": ["f32"],
                "layout": "row_major",
            }],
            preconditions=[{"kind": "mod_eq", "arg_dim": "K", "k": 16}],
        )
        result = emit_c11_baremetal_executor(run_dir)
        src = result.executor_path.read_text()
        assert "COMPGEN_PLAN_VIOLATION_PRECONDITION_MOD_EQ" in src
        assert "% 16 != 0" in src

    def test_byte_size_le_precondition_emitted(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        _emit_contract_with_inputs(
            run_dir, "r0", "h0",
            [{
                "name": "x",
                "shape": {"dims": [4, 8]},
                "dtype_class": ["f32"],
                "layout": "row_major",
            }],
            preconditions=[
                {"kind": "byte_size_le", "arg": "x", "max_bytes": 1024},
            ],
        )
        result = emit_c11_baremetal_executor(run_dir)
        src = result.executor_path.read_text()
        assert "COMPGEN_PLAN_VIOLATION_PRECONDITION_BYTE_SIZE_LE" in src
        assert "> (size_t)1024" in src

    def test_dtype_in_precondition_emitted(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        _emit_contract_with_inputs(
            run_dir, "r0", "h0",
            [{
                "name": "x",
                "shape": {"dims": [4, 8]},
                "dtype_class": ["f32", "bf16"],
                "layout": "row_major",
            }],
            preconditions=[
                {"kind": "dtype_in", "arg": "x",
                 "dtype_set": ["f32", "bf16"]},
            ],
        )
        result = emit_c11_baremetal_executor(run_dir)
        src = result.executor_path.read_text()
        assert "COMPGEN_PLAN_VIOLATION_PRECONDITION_DTYPE_IN" in src
        # Should reference both f32 and bf16 dtype enums.
        assert "COMPGEN_DTYPE_F32" in src
        assert "COMPGEN_DTYPE_BF16" in src


# --------------------------------------------------------------------------- #
# Negative control — fault injection                                          #
# --------------------------------------------------------------------------- #


class TestNegativeControls:
    def test_corrupted_emit_missing_dispatch_is_detectable(
        self, tmp_path: Path,
    ) -> None:
        """Negative control for the D6 plan-refinement gate: if a
        post-hoc edit removes a dispatch call from the emit, the count
        of dispatched regions no longer matches the plan's bound
        region count. The gate consumes this signal; just
        exposes the count."""
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
            RegionKernelBinding(
                region_id="r1", contract_hash="h1",
                certificate_path="04_kernel_codegen/certificates/h1.json",
            ),
        ])
        result = emit_c11_baremetal_executor(run_dir)
        src = result.executor_path.read_text()
        # Count dispatch sites.
        n_dispatch = src.count("cg_rt_command_buffer_dispatch(")
        assert n_dispatch == 2

        # Simulate corruption: drop one dispatch call.
        corrupted = src.replace(
            "cg_rt_command_buffer_dispatch(command_buffer, "
            "compgen_kernel_r1,", "/* dropped */ (void)(", 1,
        )
        assert corrupted.count("cg_rt_command_buffer_dispatch(") == 1, (
            "fault injection should reduce the dispatch-call count by one"
        )
