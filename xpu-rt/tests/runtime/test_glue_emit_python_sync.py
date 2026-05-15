"""emitted Python SYNC plan executor tests.

Coverage:
- Schema: emit_python_sync_executor produces both files at the right paths.
- Syntactic validity: the emitted module parses + imports cleanly.
- Behaviour: PLAN_VIOLATION_UNBOUND_REGION fires when no kernel is
  bound; well-formed dispatch loops over bound regions.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from compgen.runtime.execution_plan import (
    ExecutionPlan,
    RegionKernelBinding,
    RegionPlacement,
    Resource,
)
from compgen.runtime.glue_emit import emit_python_sync_executor


def _make_run_dir(
    tmp_path: Path,
    bindings: list[RegionKernelBinding],
    placements: list[RegionPlacement] | None = None,
) -> Path:
    """Build a synthetic run_dir with an execution plan and (when
    bindings are non-empty) the matching certificate files."""
    run_dir = tmp_path / "run"
    plan_dir = run_dir / "05_execution_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    # Emit the certs the bindings reference.
    if bindings:
        cert_dir = run_dir / "04_kernel_codegen" / "certificates"
        cert_dir.mkdir(parents=True, exist_ok=True)
        for b in bindings:
            cert_path = run_dir / b.certificate_path
            cert_path.parent.mkdir(parents=True, exist_ok=True)
            cert_path.write_text(json.dumps({
                "schema_version": "kernel_certificate_v1",
                "contract_hash": b.contract_hash,
                "task_id": "t", "region_id": b.region_id, "candidate_id": "c",
                "accepted_at_utc": "x", "artifact_hashes": {}, "artifact_paths": {},
                "verifier_report_path": "", "verifier_report_hash": "",
                "claims": {},
            }))

    placements = placements or [
        RegionPlacement(region_id=b.region_id, device="host_cpu", queue="q")
        for b in bindings
    ]
    plan = ExecutionPlan(
        workload="test", target="host_cpu",
        resources=[Resource(id="q", kind="compute", device="host_cpu")],
        region_placement=placements,
        region_kernel_bindings=bindings,
    )
    plan.validate()
    plan_dict = plan.to_dict()
    try:
        import yaml  # type: ignore[import-untyped]
        (plan_dir / "execution_plan.yaml").write_text(
            yaml.safe_dump(plan_dict, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
    except ImportError:
        (plan_dir / "execution_plan.json").write_text(
            json.dumps(plan_dict, sort_keys=True),
            encoding="utf-8",
        )
    return run_dir


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestEmitProducesArtifacts:
    def test_executor_and_manifest_emitted(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_python_sync_executor(run_dir)
        assert result.executor_path.exists()
        assert result.manifest_path.exists()
        assert result.executor_path.name == "generated_plan_executor.py"
        assert result.manifest_path.name == "plan_executor_manifest.json"
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["executor_kind"] == "python_sync"
        assert manifest["bound_regions"] == ["r0"]

    def test_emit_with_no_plan_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="execution plan not found"):
            emit_python_sync_executor(tmp_path)


# --------------------------------------------------------------------------- #
# Syntactic validity
# --------------------------------------------------------------------------- #


class TestSyntacticValidity:
    def test_emitted_module_parses_as_python(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_python_sync_executor(run_dir)
        source = result.executor_path.read_text()
        ast.parse(source)  # raises SyntaxError if invalid

    def test_emitted_module_imports_cleanly(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_python_sync_executor(run_dir)
        spec = importlib.util.spec_from_file_location(
            "test_generated_executor", result.executor_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        # Public surface: PLAN_*, KERNEL_BINDINGS, compgen_run, assert_plan, PlanViolation.
        assert module.PLAN_TARGET == "host_cpu"
        assert "r0" in module.KERNEL_BINDINGS
        assert callable(module.compgen_run)
        assert callable(module.assert_plan)
        assert issubclass(module.PlanViolation, RuntimeError)

    def test_emitted_module_imports_no_test_only_modules(
        self, tmp_path: Path,
    ) -> None:
        """The realness scan enforces no-mocks at the audit level. Here
        we double-check that the emitted source string mentions no
        mock/stub/dummy keywords."""
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        result = emit_python_sync_executor(run_dir)
        source = result.executor_path.read_text()
        for forbidden in ("mock_client", "fake_provider", "dummy_kernel",
                           "synthetic_profile", "placeholder_pass"):
            assert forbidden not in source, (
                f"emitted executor mentions forbidden module/keyword "
                f"{forbidden!r}; M-47's no-mocks invariant violated"
            )


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


class TestBehaviour:
    def _import_module(self, run_dir: Path):
        result = emit_python_sync_executor(run_dir)
        # Use a unique module name per test to avoid cache hits.
        module_name = f"_gen_executor_{run_dir.name}"
        spec = importlib.util.spec_from_file_location(
            module_name, result.executor_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def test_unbound_region_raises_plan_violation(
        self, tmp_path: Path,
    ) -> None:
        # Plan declares a region but binds nothing → PLAN_VIOLATION_UNBOUND_REGION.
        run_dir = _make_run_dir(
            tmp_path, bindings=[],
            placements=[RegionPlacement(region_id="r0", device="host_cpu", queue="q")],
        )
        module = self._import_module(run_dir)
        with pytest.raises(module.PLAN_VIOLATION_UNBOUND_REGION):
            module.compgen_run({"x": 1}, {}, runtime=_StubRuntime())
        # The typed subclass is also a PlanViolation.
        assert issubclass(
            module.PLAN_VIOLATION_UNBOUND_REGION, module.PlanViolation,
        )

    def test_bound_region_dispatches_through_runtime(
        self, tmp_path: Path,
    ) -> None:
        run_dir = _make_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
            ),
        ])
        module = self._import_module(run_dir)
        runtime = _StubRuntime()
        # Caller-supplied kernel callable.
        kernels = {"r0": lambda *args, **kwargs: ("kernel_r0_output",)}
        out = module.compgen_run({"x": 1}, kernels, runtime=runtime)
        assert runtime.dispatch_calls == 1, (
            f"runtime.dispatch should fire once per bound region; "
            f"got {runtime.dispatch_calls}"
        )
        assert runtime.synchronize_called is True
        # The DispatchResult.outputs come back from the stub.
        assert out == ("kernel_r0_output",)

    def test_multi_region_dispatches_in_topological_order(
        self, tmp_path: Path,
    ) -> None:
        from compgen.runtime.execution_plan import DependencyEdge
        # r0 → r1 (r0 must dispatch first).
        run_dir = tmp_path / "run"
        plan_dir = run_dir / "05_execution_plan"
        plan_dir.mkdir(parents=True)
        cert_dir = run_dir / "04_kernel_codegen" / "certificates"
        cert_dir.mkdir(parents=True)
        for h in ("h0", "h1"):
            (cert_dir / f"{h}.json").write_text(json.dumps({
                "schema_version": "kernel_certificate_v1",
                "contract_hash": h, "task_id": "t",
                "region_id": "r", "candidate_id": "c",
                "accepted_at_utc": "x", "artifact_hashes": {},
                "artifact_paths": {}, "verifier_report_path": "",
                "verifier_report_hash": "", "claims": {},
            }))
        plan = ExecutionPlan(
            workload="multi", target="host_cpu",
            resources=[Resource(id="q", kind="compute", device="host_cpu")],
            region_placement=[
                RegionPlacement(region_id="r0", device="host_cpu", queue="q"),
                RegionPlacement(region_id="r1", device="host_cpu", queue="q"),
            ],
            dependency_edges=[
                DependencyEdge(from_region="r0", to_region="r1"),
            ],
            region_kernel_bindings=[
                RegionKernelBinding(
                    region_id="r0", contract_hash="h0",
                    certificate_path="04_kernel_codegen/certificates/h0.json",
                ),
                RegionKernelBinding(
                    region_id="r1", contract_hash="h1",
                    certificate_path="04_kernel_codegen/certificates/h1.json",
                ),
            ],
        )
        try:
            import yaml
            (plan_dir / "execution_plan.yaml").write_text(
                yaml.safe_dump(plan.to_dict(), sort_keys=True),
                encoding="utf-8",
            )
        except ImportError:
            (plan_dir / "execution_plan.json").write_text(
                json.dumps(plan.to_dict(), sort_keys=True),
                encoding="utf-8",
            )
        module = self._import_module(run_dir)
        assert module.PLAN_REGION_ORDER == ["r0", "r1"]


# --------------------------------------------------------------------------- #
# Stub runtime adapter — returns a predictable DispatchResult
# --------------------------------------------------------------------------- #


class _StubRuntime:
    """Minimal RuntimeAdapter that records calls and returns the
    callable's output as a DispatchResult."""

    name = "stub"

    def __init__(self) -> None:
        self.dispatch_calls = 0
        self.synchronize_called = False

    def dispatch(self, *, contract, callable_kernel, args, kwargs):
        self.dispatch_calls += 1

        class _Result:
            def __init__(self, output):
                self.output = output

        out = callable_kernel(*args, **kwargs)
        return _Result(out)

    def synchronize(self) -> None:
        self.synchronize_called = True

    def supports(self, contract) -> bool:
        return True
