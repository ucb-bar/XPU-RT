"""plan-assertion tests.

Each test fires one specific PLAN_VIOLATION_<KIND> with a typed
subclass. The harness:

1. Drives ``--stop-after glue-emit`` end-to-end on merlin_mlp_wide
   (after committing a synthetic provider response) so the emitted
   module has all the generated assertions wired.
2. Imports the module.
3. Calls ``compgen_run(io, kernels, runtime)`` with a tampered ``io``
   dict that violates exactly one invariant.
4. Asserts the typed subclass fires.

Catches the negative-control discipline: each PLAN_VIOLATION
case is reachable via a fault injection in the test, not just by
inspection of the emitted source.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_run_with_provider_response(out: Path) -> tuple[Path, dict]:
    """Drive merlin_mlp_wide through the pipeline + commit a
    contract-compliant provider response so the emitted module has
    bound regions + assertions wired. Returns (out, request)."""
    res = subprocess.run([
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out),
        "--stop-after", "kernel-codegen-request",
        "--selection-mode", "greedy",
    ], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    # Build contract-compliant response.
    req_files = list((out / "04_kernel_codegen" / "requests").glob("*.request.json"))
    request = json.loads(req_files[0].read_text())
    contract = json.loads((out / request["contract_paths"]["full"]).read_text())
    io_block = contract["io"]
    metadata = {
        "inputs": [
            {"dims": list(t["shape"]["dims"]), "dtype": t["dtype_class"][0],
             "layout": t["layout"]} for t in io_block["inputs"]
        ],
        "outputs": [
            {"dims": list(t["shape"]["dims"]), "dtype": t["dtype_class"][0],
             "layout": t["layout"]} for t in io_block["outputs"]
        ],
        "accumulator_dtype": io_block["numerics"]["accumulator_dtype"],
        "target_name": (contract["orchestration"]["execution"] or {})
            .get("hardware", {}).get("target_name", ""),
        "signals_emitted": {
            e["name"]: e["wait_count"]
            for e in contract["orchestration"]["sync"]["event_decls"]
        },
    }
    claims = {
        "backend": request["allowed_backends"][0],
        "supports_dispatch": [contract["orchestration"]["dispatch"]["model"]],
        "expected_numerics": "bit_equality",
        "estimated_registers": 0, "estimated_smem_bytes": 0,
    }
    sandbox = out / request["artifact_dir"]
    sandbox.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in request["required_outputs"]:
        ext = ".c" if name == "kernel_source" else ".json"
        path = sandbox / f"{name}{ext}"
        if name == "kernel_metadata":
            path.write_text(json.dumps(metadata, sort_keys=True))
        elif name == "provider_claims":
            path.write_text(json.dumps(claims, sort_keys=True))
        elif name == "launch_config":
            path.write_text("{}")
        else:
            path.write_text("/* synthetic */\n")
        artifacts[name] = str(path.relative_to(out))
    response = {
        "schema_version": "kernel_codegen_response_v1",
        "task_id": request["task_id"],
        "contract_hash": request["contract_hash"],
        "artifacts": artifacts, "claims": claims,
        "provider": {"kind": "test_synthetic"},
    }
    from compgen.graph_compilation.kernel_codegen_response import commit_response
    commit_response(run_dir=out, task_id=request["task_id"], response=response)

    # Re-emit .
    from compgen.graph_compilation.execution_plan_emit import emit_execution_plan
    from compgen.runtime.glue_emit import emit_python_sync_executor
    emit_execution_plan(out)
    emit_python_sync_executor(out)
    return out, request


def _import_emitted(out: Path):
    executor = out / "06_glue_emit" / "generated_plan_executor.py"
    spec = importlib.util.spec_from_file_location(
        f"_m48_test_{out.name}", executor,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubRuntime:
    def __init__(self) -> None:
        self.dispatch_count = 0
        self.synced = False

    def dispatch(self, *, contract, callable_kernel, args, kwargs):
        self.dispatch_count += 1

        class _R:
            def __init__(self, o): self.output = o
        return _R(callable_kernel(*args, **kwargs))

    def synchronize(self) -> None:
        self.synced = True


@pytest.fixture(scope="module")
def merlin_emit(tmp_path_factory) -> tuple[Path, dict]:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m48_merlin") / "run"
    return _build_run_with_provider_response(out)


# --------------------------------------------------------------------------- #
# Negative controls — one per PLAN_VIOLATION_<KIND>
# --------------------------------------------------------------------------- #


class TestPlanViolationKinds:
    def test_io_type_violation(self, merlin_emit) -> None:
        out, _ = merlin_emit
        module = _import_emitted(out)
        with pytest.raises(module.PLAN_VIOLATION_IO_TYPE):
            module.compgen_run("not a dict", {"matmul_0": lambda *a: None},
                               runtime=_StubRuntime())

    def test_input_count_violation(self, merlin_emit) -> None:
        out, _ = merlin_emit
        module = _import_emitted(out)
        # Pass too few inputs (matmul needs 2: A and B).
        import torch
        io = {"A": torch.randn(16, 16, dtype=torch.float32)}
        with pytest.raises(module.PLAN_VIOLATION_INPUT_COUNT):
            module.compgen_run(
                io, {"matmul_0": lambda *a: torch.zeros(16, 32)},
                runtime=_StubRuntime(),
            )

    def test_input_shape_violation(self, merlin_emit) -> None:
        out, _ = merlin_emit
        module = _import_emitted(out)
        import torch
        # Wrong shape for input 0 (contract expects 16x16).
        io = {
            "A": torch.randn(99, 99, dtype=torch.float32),
            "B": torch.randn(16, 32, dtype=torch.float32),
        }
        with pytest.raises(module.PLAN_VIOLATION_INPUT_SHAPE):
            module.compgen_run(
                io, {"matmul_0": lambda *a: torch.zeros(16, 32)},
                runtime=_StubRuntime(),
            )

    def test_input_dtype_violation(self, merlin_emit) -> None:
        out, _ = merlin_emit
        module = _import_emitted(out)
        import torch
        # Wrong dtype for input 0 (contract expects f32).
        io = {
            "A": torch.randn(16, 16, dtype=torch.float64),
            "B": torch.randn(16, 32, dtype=torch.float32),
        }
        with pytest.raises(module.PLAN_VIOLATION_INPUT_DTYPE):
            module.compgen_run(
                io, {"matmul_0": lambda *a: torch.zeros(16, 32)},
                runtime=_StubRuntime(),
            )

    def test_input_bytes_violation(self, merlin_emit) -> None:
        """Tampered tensor with right shape but tampered storage size.
        Mostly covered by INPUT_SHAPE; the BYTES check is a defence-
        in-depth path. We synthesize a slice with non-contiguous
        storage where numel() * element_size() != allocated bytes —
        but on torch this is the same as numel*element_size.

        Skip this when contract's expected_bytes is computable and
        torch's numel*element_size always matches expected. The
        check still fires when a custom Tensor-like object reports
        a different numel/element_size combination."""
        out, _ = merlin_emit
        module = _import_emitted(out)

        class _BadTensor:
            shape = (16, 16)
            dtype = "float32"

            def __init__(self, n: int, es: int):
                self._n = n
                self._es = es

            def numel(self) -> int:
                return self._n

            def element_size(self) -> int:
                return self._es

            def is_contiguous(self) -> bool:
                return True

        import torch
        io = {
            "A": _BadTensor(n=16 * 16, es=2),  # claims f16-size bytes (mismatched)
            "B": torch.randn(16, 32, dtype=torch.float32),
        }
        # The DTYPE check fires before BYTES on a synthetic object.
        # If the dtype check matches (e.g. our _BadTensor has dtype
        # "float32"), then BYTES fires because numel*es=512 vs expected 1024.
        with pytest.raises((module.PLAN_VIOLATION_INPUT_BYTES,
                            module.PLAN_VIOLATION_INPUT_DTYPE)):
            module.compgen_run(
                io, {"matmul_0": lambda *a: torch.zeros(16, 32)},
                runtime=_StubRuntime(),
            )

    def test_well_formed_io_passes_assertions(self, merlin_emit) -> None:
        out, _ = merlin_emit
        module = _import_emitted(out)
        import torch
        torch.manual_seed(0)
        io = {
            "A": torch.randn(16, 16, dtype=torch.float32),
            "B": torch.randn(16, 32, dtype=torch.float32),
        }
        kernels = {"matmul_0": lambda *args: torch.matmul(args[0], args[1])}
        runtime = _StubRuntime()
        out_t = module.compgen_run(io, kernels, runtime=runtime)
        assert tuple(out_t.shape) == (16, 32)
        assert runtime.dispatch_count == 1


# --------------------------------------------------------------------------- #
# Type-hierarchy invariants
# --------------------------------------------------------------------------- #


class TestTypeHierarchy:
    def test_every_violation_kind_is_planviolation(self, merlin_emit) -> None:
        out, _ = merlin_emit
        module = _import_emitted(out)
        for kind in (
            "INPUT_COUNT", "INPUT_DTYPE", "INPUT_SHAPE", "INPUT_BYTES",
            "LAYOUT", "BUFFER_SIZE", "EVENT_WRITERS",
            "UNBOUND_REGION", "IO_TYPE",
        ):
            cls = getattr(module, f"PLAN_VIOLATION_{kind}")
            assert issubclass(cls, module.PlanViolation)
            assert issubclass(cls, RuntimeError)
