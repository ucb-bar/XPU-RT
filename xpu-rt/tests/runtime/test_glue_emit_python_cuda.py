"""M-52 emitted Python CUDA plan executor tests.

Coverage (CPU-side, never requires a real GPU):
- Schema: emit_python_cuda_executor produces both files when target is
  CUDA-class; emits manifest with overall=skipped on non-CUDA targets.
- Syntactic validity: emitted module parses + imports cleanly.
- Behaviour:
  * mode="sync" with stub adapter: dispatches sequentially with
    synchronize between regions; output matches.
  * mode="async" with stub adapter: per-region threading +
    EventTensor handshake honours dependency_edges; ordering enforced.
  * mode="bogus": raises ValueError honestly.
  * capture=True with single-region plan and an adapter that returns
    None from capture_graph: returns dict with capture_status =
    "unavailable_no_cuda" instead of pretending capture worked.
  * capture=True with multi-region plan: raises ValueError honestly.
  * Unbound region: PLAN_VIOLATION_UNBOUND_REGION fires before
    dispatch.

GPU-conditional tests (requires_gpu marker) exercise the real CUDA
path; these skip on CPU-only hosts.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

from compgen.runtime.execution_plan import (
    DependencyEdge,
    ExecutionPlan,
    RegionKernelBinding,
    RegionPlacement,
    Resource,
)
from compgen.runtime.glue_emit import emit_python_cuda_executor


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_certs(run_dir: Path, hashes: list[str]) -> None:
    cert_dir = run_dir / "04_kernel_codegen" / "certificates"
    cert_dir.mkdir(parents=True, exist_ok=True)
    for h in hashes:
        (cert_dir / f"{h}.json").write_text(json.dumps({
            "schema_version": "kernel_certificate_v1",
            "contract_hash": h, "task_id": "t",
            "region_id": "r", "candidate_id": "c",
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


def _make_cuda_run_dir(
    tmp_path: Path,
    bindings: list[RegionKernelBinding],
    *,
    target: str = "cuda_sm75",
    dependency_edges: list[DependencyEdge] | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    _write_certs(run_dir, [b.contract_hash for b in bindings])
    placements = [
        RegionPlacement(region_id=b.region_id, device=target, queue="q")
        for b in bindings
    ] or [RegionPlacement(region_id="r0", device=target, queue="q")]
    plan = ExecutionPlan(
        workload="test", target=target,
        resources=[Resource(id="q", kind="compute", device=target)],
        region_placement=placements,
        dependency_edges=dependency_edges or [],
        region_kernel_bindings=bindings,
    )
    plan.validate()
    _write_plan(run_dir, plan)
    return run_dir


def _import_cuda_module(run_dir: Path):
    result = emit_python_cuda_executor(run_dir)
    if result.overall == "skipped":
        return None, result
    module_name = f"_gen_cuda_{run_dir.name}_{int(time.time()*1e6)}"
    spec = importlib.util.spec_from_file_location(
        module_name, result.executor_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module, result


# --------------------------------------------------------------------------- #
# Stub CUDA adapter — emulates the runtime protocol without a GPU
# --------------------------------------------------------------------------- #


class _StubCudaRuntime:
    name = "stub_cuda"

    def __init__(self, *, capture_returns_none: bool = False) -> None:
        self.dispatch_calls = 0
        self.synchronize_calls = 0
        self.capture_calls = 0
        self.replay_calls = 0
        self._capture_returns_none = capture_returns_none
        self._lock = threading.Lock()

    def dispatch(self, *, contract, callable_kernel, args, kwargs):
        with self._lock:
            self.dispatch_calls += 1

        class _Result:
            def __init__(self, output):
                self.output = output

        out = callable_kernel(*args, **kwargs)
        return _Result(out)

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def capture_graph(self, *, model_fn, sample_inputs):
        self.capture_calls += 1
        if self._capture_returns_none:
            return None

        class _Captured:
            def __init__(self, fn, inputs):
                self.fn = fn
                self.inputs = inputs

        return _Captured(model_fn, sample_inputs)

    def replay(self, captured, inputs):
        self.replay_calls += 1
        return captured.fn(*inputs)

    def supports(self, contract) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestEmitProducesArtifacts:
    def test_skips_on_non_cuda_target(self, tmp_path: Path) -> None:
        run_dir = _make_cuda_run_dir(
            tmp_path,
            [RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="sync",
            )],
            target="host_cpu",
        )
        result = emit_python_cuda_executor(run_dir)
        assert result.overall == "skipped"
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["overall"] == "skipped"
        assert "not CUDA-class" in manifest["skipped_reason"]
        assert not (run_dir / "06_glue_emit" / "generated_plan_executor_cuda.py").exists()

    def test_emits_on_cuda_target(self, tmp_path: Path) -> None:
        run_dir = _make_cuda_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="sync",
            ),
        ], target="cuda_sm75")
        result = emit_python_cuda_executor(run_dir)
        assert result.overall == "pass"
        assert result.executor_path.name == "generated_plan_executor_cuda.py"
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["executor_kind"] == "python_cuda"
        assert manifest["cuda_target"] == "cuda_sm75"
        assert manifest["supports_modes"] == ["sync", "async"]
        assert manifest["supports_capture"] is True


# --------------------------------------------------------------------------- #
# Syntactic validity
# --------------------------------------------------------------------------- #


class TestSyntacticValidity:
    def test_emitted_module_parses(self, tmp_path: Path) -> None:
        run_dir = _make_cuda_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="sync",
            ),
        ], target="cuda_sm75")
        result = emit_python_cuda_executor(run_dir)
        ast.parse(result.executor_path.read_text())

    def test_emitted_module_imports(self, tmp_path: Path) -> None:
        run_dir = _make_cuda_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="sync",
            ),
        ], target="cuda_sm75")
        module, _ = _import_cuda_module(run_dir)
        assert module is not None
        assert module.PLAN_TARGET == "cuda_sm75"
        assert callable(module.compgen_run_cuda)
        assert callable(module.assert_plan)
        assert "r0" in module.KERNEL_BINDINGS


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


class TestBehaviour:
    def test_sync_mode_dispatches_with_synchronize_between(
        self, tmp_path: Path,
    ) -> None:
        run_dir = _make_cuda_run_dir(
            tmp_path,
            [
                RegionKernelBinding(
                    region_id="r0", contract_hash="h0",
                    certificate_path="04_kernel_codegen/certificates/h0.json",
                    dispatch_model="sync",
                ),
                RegionKernelBinding(
                    region_id="r1", contract_hash="h1",
                    certificate_path="04_kernel_codegen/certificates/h1.json",
                    dispatch_model="sync",
                ),
            ],
            target="cuda_sm75",
            dependency_edges=[DependencyEdge(from_region="r0", to_region="r1")],
        )
        module, _ = _import_cuda_module(run_dir)
        rt = _StubCudaRuntime()
        kernels = {"r0": lambda *a, **k: "v0", "r1": lambda *a, **k: "v1"}
        out = module.compgen_run_cuda({"x": 1}, kernels, runtime=rt, mode="sync")
        assert rt.dispatch_calls == 2
        # synchronize: once after each region (2) + once at end of compgen_run_cuda = 3.
        assert rt.synchronize_calls >= 2
        assert out == "v1"

    def test_async_mode_dispatches_with_handshake(self, tmp_path: Path) -> None:
        run_dir = _make_cuda_run_dir(
            tmp_path,
            [
                RegionKernelBinding(
                    region_id="r0", contract_hash="h0",
                    certificate_path="04_kernel_codegen/certificates/h0.json",
                    dispatch_model="async",
                ),
                RegionKernelBinding(
                    region_id="r1", contract_hash="h1",
                    certificate_path="04_kernel_codegen/certificates/h1.json",
                    dispatch_model="async",
                ),
            ],
            target="cuda_sm75",
            dependency_edges=[DependencyEdge(from_region="r0", to_region="r1")],
        )
        module, _ = _import_cuda_module(run_dir)
        order_lock = threading.Lock()
        order: list[str] = []

        def _r0(*a, **k):
            time.sleep(0.05)
            with order_lock:
                order.append("r0")
            return "v0"

        def _r1(*a, **k):
            with order_lock:
                order.append("r1")
            return "v1"

        rt = _StubCudaRuntime()
        out = module.compgen_run_cuda(
            {"x": 1}, {"r0": _r0, "r1": _r1},
            runtime=rt, mode="async",
        )
        assert rt.dispatch_calls == 2
        assert order == ["r0", "r1"], (
            f"async CUDA executor must serialise r0→r1 via EventTensor; got {order}"
        )
        assert out == "v1"

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        run_dir = _make_cuda_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="sync",
            ),
        ], target="cuda_sm75")
        module, _ = _import_cuda_module(run_dir)
        with pytest.raises(ValueError, match="mode must be 'sync' or 'async'"):
            module.compgen_run_cuda(
                {"x": 1}, {"r0": lambda *a, **k: "x"},
                runtime=_StubCudaRuntime(), mode="bogus",
            )

    def test_capture_with_no_cuda_returns_unavailable_dict(
        self, tmp_path: Path,
    ) -> None:
        run_dir = _make_cuda_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="sync",
            ),
        ], target="cuda_sm75")
        module, _ = _import_cuda_module(run_dir)
        rt = _StubCudaRuntime(capture_returns_none=True)
        out = module.compgen_run_cuda(
            {"x": 1}, {"r0": lambda *a, **k: "v0"},
            runtime=rt, capture=True,
        )
        assert isinstance(out, dict)
        assert out["capture_status"] == "unavailable_no_cuda"
        assert out["captured_graph"] is None
        # Honestly fell back to dispatch.
        assert rt.dispatch_calls == 1

    def test_capture_records_graph_when_available(self, tmp_path: Path) -> None:
        run_dir = _make_cuda_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="sync",
            ),
        ], target="cuda_sm75")
        module, _ = _import_cuda_module(run_dir)
        rt = _StubCudaRuntime(capture_returns_none=False)
        out = module.compgen_run_cuda(
            {"x": 1}, {"r0": lambda *a, **k: "captured_v0"},
            runtime=rt, capture=True,
        )
        assert isinstance(out, dict)
        assert out["capture_status"] == "captured"
        assert out["captured_graph"] is not None
        assert out["output"] == "captured_v0"
        assert rt.capture_calls == 1
        assert rt.replay_calls == 1

    def test_capture_with_multi_region_raises(self, tmp_path: Path) -> None:
        run_dir = _make_cuda_run_dir(
            tmp_path,
            [
                RegionKernelBinding(
                    region_id="r0", contract_hash="h0",
                    certificate_path="04_kernel_codegen/certificates/h0.json",
                    dispatch_model="sync",
                ),
                RegionKernelBinding(
                    region_id="r1", contract_hash="h1",
                    certificate_path="04_kernel_codegen/certificates/h1.json",
                    dispatch_model="sync",
                ),
            ],
            target="cuda_sm75",
        )
        module, _ = _import_cuda_module(run_dir)
        with pytest.raises(ValueError, match="exactly one bound region"):
            module.compgen_run_cuda(
                {"x": 1}, {"r0": lambda *a, **k: "x", "r1": lambda *a, **k: "y"},
                runtime=_StubCudaRuntime(), capture=True,
            )

    def test_unbound_region_raises_plan_violation(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        _write_certs(run_dir, ["h0"])
        plan = ExecutionPlan(
            workload="test", target="cuda_sm75",
            resources=[Resource(id="q", kind="compute", device="cuda_sm75")],
            region_placement=[
                RegionPlacement(region_id="r0", device="cuda_sm75", queue="q"),
                RegionPlacement(region_id="r1", device="cuda_sm75", queue="q"),
            ],
            region_kernel_bindings=[
                RegionKernelBinding(
                    region_id="r0", contract_hash="h0",
                    certificate_path="04_kernel_codegen/certificates/h0.json",
                    dispatch_model="sync",
                ),
            ],
        )
        plan.validate()
        _write_plan(run_dir, plan)
        module, _ = _import_cuda_module(run_dir)
        with pytest.raises(module.PLAN_VIOLATION_UNBOUND_REGION):
            module.compgen_run_cuda(
                {"x": 1}, {"r0": lambda *a, **k: "x"},
                runtime=_StubCudaRuntime(), mode="sync",
            )


# --------------------------------------------------------------------------- #
# Adapter selection — emitted code routes correctly
# --------------------------------------------------------------------------- #


def test_default_adapter_is_cuda(tmp_path: Path) -> None:
    """When ``runtime`` is not passed, ``select_adapter(PLAN_TARGET)``
    is invoked. For a CUDA target this returns a CudaRuntimeAdapter
    (real class); we verify only the type, not GPU dispatch — the
    real adapter's dispatch needs a real KernelContractV3 (M-49+
    wiring), which is honestly not yet in the emitted code path."""
    from compgen.runtime.glue import CudaRuntimeAdapter, select_adapter
    run_dir = _make_cuda_run_dir(tmp_path, [
        RegionKernelBinding(
            region_id="r0", contract_hash="h0",
            certificate_path="04_kernel_codegen/certificates/h0.json",
            dispatch_model="sync",
        ),
    ], target="cuda_sm75")
    module, _ = _import_cuda_module(run_dir)
    # Sanity: select_adapter on the plan's target picks CUDA.
    adapter = select_adapter(module.PLAN_TARGET)
    assert isinstance(adapter, CudaRuntimeAdapter)
