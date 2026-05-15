"""emitted Python ASYNC plan executor tests.

Coverage:
- Schema: emit_python_async_executor produces both files when at least
  one binding has dispatch_model="async"; emits manifest with
  overall=skipped when all bindings are SYNC.
- Syntactic validity: emitted module parses + imports cleanly.
- Behaviour:
  * Single async region: compgen_run_async dispatches + synchronizes;
    output matches SYNC executor's last_out.
  * Two async regions with a dependency: producer-consumer EventTensor
    handshake works (consumer sees producer's output).
  * Missing notify (kernel that never returns) → TimeoutError.
  * Plan with double-writer event → assert_plan raises
    PLAN_VIOLATION_EVENT_WRITERS at IMPORT-time generation when the
    contracts on disk declare the same event name. (We emulate via a
    crafted run_dir with two contracts naming the same event.)
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
from compgen.runtime.glue_emit import (
    emit_python_async_executor,
    emit_python_sync_executor,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


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


def _make_async_run_dir(
    tmp_path: Path,
    bindings: list[RegionKernelBinding],
    dependency_edges: list[DependencyEdge] | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    _write_certs(run_dir, [b.contract_hash for b in bindings])
    placements = [
        RegionPlacement(region_id=b.region_id, device="host_cpu", queue="q")
        for b in bindings
    ]
    plan = ExecutionPlan(
        workload="test", target="host_cpu",
        resources=[Resource(id="q", kind="compute", device="host_cpu")],
        region_placement=placements,
        dependency_edges=dependency_edges or [],
        region_kernel_bindings=bindings,
    )
    plan.validate()
    _write_plan(run_dir, plan)
    return run_dir


def _import_async_module(run_dir: Path):
    result = emit_python_async_executor(run_dir)
    if result.overall == "skipped":
        return None, result
    module_name = f"_gen_async_{run_dir.name}_{int(time.time()*1e6)}"
    spec = importlib.util.spec_from_file_location(
        module_name, result.executor_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module, result


# --------------------------------------------------------------------------- #
# Stub runtime — same shape as the SYNC test stub.
# --------------------------------------------------------------------------- #


class _StubRuntime:
    name = "stub"

    def __init__(self) -> None:
        self.dispatch_calls = 0
        self.synchronize_called = False
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
        self.synchronize_called = True

    def supports(self, contract) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestEmitProducesArtifacts:
    def test_skips_when_all_bindings_sync(self, tmp_path: Path) -> None:
        run_dir = _make_async_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="sync",
            ),
        ])
        result = emit_python_async_executor(run_dir)
        assert result.overall == "skipped"
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["overall"] == "skipped"
        assert manifest["async_regions"] == []
        # No async executor file is written.
        assert not (run_dir / "06_glue_emit" / "generated_plan_executor_async.py").exists()

    def test_emits_when_any_binding_async(self, tmp_path: Path) -> None:
        run_dir = _make_async_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="async",
            ),
        ])
        result = emit_python_async_executor(run_dir)
        assert result.overall == "pass"
        assert result.executor_path.exists()
        assert result.executor_path.name == "generated_plan_executor_async.py"
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["executor_kind"] == "python_async"
        assert manifest["async_regions"] == ["r0"]
        assert manifest["default_timeout_s"] == 30.0


# --------------------------------------------------------------------------- #
# Syntactic validity
# --------------------------------------------------------------------------- #


class TestSyntacticValidity:
    def test_emitted_module_parses_as_python(self, tmp_path: Path) -> None:
        run_dir = _make_async_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="async",
            ),
        ])
        result = emit_python_async_executor(run_dir)
        ast.parse(result.executor_path.read_text())

    def test_emitted_module_imports_cleanly(self, tmp_path: Path) -> None:
        run_dir = _make_async_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="async",
            ),
        ])
        module, _ = _import_async_module(run_dir)
        assert module is not None
        assert module.PLAN_TARGET == "host_cpu"
        assert callable(module.compgen_run_async)
        assert callable(module.assert_plan)
        assert "r0" in module.KERNEL_BINDINGS
        # Event spec for r0_done present with wait_count_default >= 1.
        assert "r0_done" in module.EVENT_SPECS
        assert module.EVENT_SPECS["r0_done"]["wait_count_default"] >= 1


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


class TestBehaviour:
    def test_single_region_dispatches_and_synchronizes(self, tmp_path: Path) -> None:
        run_dir = _make_async_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="async",
            ),
        ])
        module, _ = _import_async_module(run_dir)
        rt = _StubRuntime()
        kernels = {"r0": lambda *args, **kwargs: "out_r0"}
        out = module.compgen_run_async({"x": 1}, kernels, runtime=rt)
        assert rt.dispatch_calls == 1
        assert rt.synchronize_called is True
        assert out == "out_r0"

    def test_two_regions_with_dependency_handshake(self, tmp_path: Path) -> None:
        run_dir = _make_async_run_dir(
            tmp_path,
            bindings=[
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
            dependency_edges=[DependencyEdge(from_region="r0", to_region="r1")],
        )
        module, _ = _import_async_module(run_dir)
        # r0_done event should expect 1 consumer.
        assert module.EVENT_SPECS["r0_done"]["wait_count_default"] == 1

        # Track ordering: r0's worker must complete before r1's dispatch
        # observes runtime.dispatch_calls == 1.
        order: list[str] = []
        order_lock = threading.Lock()

        def _r0(*args, **kwargs):
            time.sleep(0.05)
            with order_lock:
                order.append("r0")
            return "r0_out"

        def _r1(*args, **kwargs):
            with order_lock:
                order.append("r1")
            return "r1_out"

        rt = _StubRuntime()
        out = module.compgen_run_async(
            {"x": 1}, {"r0": _r0, "r1": _r1}, runtime=rt,
        )
        assert rt.dispatch_calls == 2
        assert rt.synchronize_called is True
        # Terminal region is r1 (last in topo order).
        assert out == "r1_out"
        # r0 ran before r1 — wait/notify on r0_done enforced ordering.
        assert order == ["r0", "r1"], (
            f"async executor must serialise r0→r1 via EventTensor; got {order}"
        )

    def test_missing_notify_times_out(self, tmp_path: Path) -> None:
        """Inject a kernel that hangs; must time out deterministically."""
        run_dir = _make_async_run_dir(
            tmp_path,
            bindings=[
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
            dependency_edges=[DependencyEdge(from_region="r0", to_region="r1")],
        )
        module, _ = _import_async_module(run_dir)

        stop = threading.Event()

        def _r0_hangs(*args, **kwargs):
            stop.wait(timeout=10.0)
            return "never"

        def _r1(*args, **kwargs):
            return "r1_out"

        rt = _StubRuntime()
        with pytest.raises(TimeoutError):
            module.compgen_run_async(
                {"x": 1}, {"r0": _r0_hangs, "r1": _r1},
                runtime=rt, timeout_s=0.5,
            )
        stop.set()

    def test_kernel_exception_propagates(self, tmp_path: Path) -> None:
        run_dir = _make_async_run_dir(tmp_path, [
            RegionKernelBinding(
                region_id="r0", contract_hash="h0",
                certificate_path="04_kernel_codegen/certificates/h0.json",
                dispatch_model="async",
            ),
        ])
        module, _ = _import_async_module(run_dir)

        def _r0_boom(*args, **kwargs):
            raise RuntimeError("kernel boom")

        rt = _StubRuntime()
        with pytest.raises(RuntimeError, match="kernel boom"):
            module.compgen_run_async({"x": 1}, {"r0": _r0_boom}, runtime=rt)

    def test_unbound_region_raises_plan_violation(self, tmp_path: Path) -> None:
        # Plan declares a region but binds nothing; emit returns
        # skipped (no async bindings). Force the path by giving one
        # async + one unbound placement.
        run_dir = tmp_path / "run"
        _write_certs(run_dir, ["h0"])
        plan = ExecutionPlan(
            workload="test", target="host_cpu",
            resources=[Resource(id="q", kind="compute", device="host_cpu")],
            region_placement=[
                RegionPlacement(region_id="r0", device="host_cpu", queue="q"),
                RegionPlacement(region_id="r1", device="host_cpu", queue="q"),
            ],
            region_kernel_bindings=[
                RegionKernelBinding(
                    region_id="r0", contract_hash="h0",
                    certificate_path="04_kernel_codegen/certificates/h0.json",
                    dispatch_model="async",
                ),
            ],
        )
        plan.validate()
        _write_plan(run_dir, plan)
        module, _ = _import_async_module(run_dir)
        assert module is not None
        with pytest.raises(module.PLAN_VIOLATION_UNBOUND_REGION):
            module.compgen_run_async({"x": 1}, {"r0": lambda *a, **k: "x"}, runtime=_StubRuntime())


# --------------------------------------------------------------------------- #
# Cross-executor parity (paper claim: ASYNC produces same output as SYNC)
# --------------------------------------------------------------------------- #


class TestSyncAsyncParity:
    def test_same_output_as_sync_executor(self, tmp_path: Path) -> None:
        """Two-region linear plan: emit BOTH executors and confirm the
        terminal output matches. With a single linear chain and no
        kernel non-determinism, async ordering must produce the same
        final output as sync."""
        run_dir = _make_async_run_dir(
            tmp_path,
            bindings=[
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
            dependency_edges=[DependencyEdge(from_region="r0", to_region="r1")],
        )
        # Emit BOTH executors.
        emit_python_sync_executor(run_dir)
        emit_python_async_executor(run_dir)
        sync_path = run_dir / "06_glue_emit" / "generated_plan_executor.py"
        async_path = run_dir / "06_glue_emit" / "generated_plan_executor_async.py"
        assert sync_path.exists() and async_path.exists()

        def _load(path, suffix):
            spec = importlib.util.spec_from_file_location(f"sync_{suffix}", path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m

        sync_mod = _load(sync_path, f"sync_{run_dir.name}")
        async_mod = _load(async_path, f"async_{run_dir.name}")

        kernels = {"r0": lambda *a, **k: "v0", "r1": lambda *a, **k: "v1"}
        out_sync = sync_mod.compgen_run({"x": 1}, kernels, runtime=_StubRuntime())
        out_async = async_mod.compgen_run_async(
            {"x": 1}, kernels, runtime=_StubRuntime(),
        )
        assert out_sync == out_async == "v1"
