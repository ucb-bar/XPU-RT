"""End-to-end test for Phase A item 6 — extended artefact emission.

Compiles a tiny model through :func:`compgen.compile_model`, then
re-loads and re-executes the resulting bundle through
:mod:`compgen.runtime.bundle_runner`. This is the flagship
"recipe-library re-execution" test that proves the artefact-emission
wiring actually closes the ``compile_model`` → ``bundle_runner`` loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compgen.api import compile_model, device
from compgen.runtime.bundle_runner import load_bundle, run_bundle

EXEMPLAR_DIR = Path(__file__).parent.parent / "targetgen" / "exemplars"


class _TinyMLP(nn.Module):
    """Realistic 1-layer MLP with bias + ReLU.

    Exercises matmul, ``aten_bias_add`` (from linear's bias), and
    ``aten_relu`` on the CPU executor — all three are covered by
    :data:`compgen.runtime.cpu_executor._ATEN_DISPATCH`. If a future
    refactor drops any of them the round-trip test will fail here
    loudly instead of silently reducing coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).relu()


@pytest.fixture
def compiled_bundle_dir(tmp_path: Path) -> tuple[Path, _TinyMLP, tuple[torch.Tensor, ...]]:
    """Compile a TinyMLP and return (bundle_dir, model, inputs).

    Shared by several tests to avoid paying the compile cost multiple
    times per fixture invocation.
    """
    torch.manual_seed(0)
    model = _TinyMLP()
    inputs = (torch.randn(4, 32),)

    dev = device(
        EXEMPLAR_DIR / "test_gpu_simt.yaml",
        output_dir=tmp_path / "out",
    )
    compiled = compile_model(model, dev, sample_inputs=inputs)

    bundle_dir_str = compiled.pipeline_result.all_artifacts.get("bundle_dir")
    assert bundle_dir_str is not None, "bundle stage should have emitted bundle_dir"
    bundle_dir = Path(bundle_dir_str)
    assert bundle_dir.is_dir()

    return bundle_dir, model, inputs


def test_bundle_contains_extended_artefacts(compiled_bundle_dir) -> None:
    """After a successful ``compile_model``, the bundle directory
    must contain the extended artefacts (exported_program.pt2,
    golden_inputs.pt, golden_outputs.pt, compile_baseline.json,
    graph_breaks.json)."""
    bundle_dir, _, _ = compiled_bundle_dir

    for name in (
        "payload.mlir",
        "manifest.json",
        "exported_program.pt2",
        "golden_inputs.pt",
        "golden_outputs.pt",
        "compile_baseline.json",
        "graph_breaks.json",
    ):
        assert (bundle_dir / name).is_file(), f"missing expected artefact: {name}"


def test_manifest_lists_extended_artefacts(compiled_bundle_dir) -> None:
    """The manifest's ``artifacts`` block must enumerate every
    on-disk extended artefact so downstream consumers can find them
    via the manifest rather than globbing."""
    bundle_dir, _, _ = compiled_bundle_dir
    manifest = json.loads((bundle_dir / "manifest.json").read_text())

    artefacts = manifest.get("artifacts", {})
    assert isinstance(artefacts, dict)
    for key in (
        "payload",
        "exported_program",
        "golden_inputs",
        "golden_outputs",
        "compile_baseline",
        "graph_breaks",
    ):
        assert key in artefacts, f"manifest missing artefact entry: {key}"
        path = bundle_dir / artefacts[key]
        assert path.is_file(), f"manifest points at missing file: {artefacts[key]}"


def test_bundle_runner_loads_compile_model_output(compiled_bundle_dir) -> None:
    """``bundle_runner.load_bundle`` must consume what ``compile_model``
    emitted and rehydrate all optional artefacts."""
    bundle_dir, _, _ = compiled_bundle_dir

    bundle = load_bundle(bundle_dir)
    assert bundle.payload_module is not None
    assert bundle.exported_program is not None
    assert bundle.golden_inputs is not None
    assert bundle.golden_output is not None


def test_round_trip_compile_then_run_bundle(compiled_bundle_dir) -> None:
    """Flagship test: compile a model, load the emitted bundle, run it,
    and check the output matches the golden within fp32 tolerance.

    This is the Phase-A "closing the loop" proof — recipe library to
    re-executable artefact without the LLM or the agent loop.
    """
    bundle_dir, _, _ = compiled_bundle_dir

    bundle = load_bundle(bundle_dir)
    output = run_bundle(bundle)

    assert bundle.golden_output is not None
    assert tuple(output.shape) == tuple(bundle.golden_output.shape)
    max_abs_diff = (output - bundle.golden_output).abs().max().item()
    # TinyMLP through the bridge + cpu_executor should be near-identical
    # to eager. Allow a tolerant fp32 bound; the bridge/executor path
    # may introduce tiny numerical wiggles.
    assert max_abs_diff < 1e-4, f"max_abs_diff={max_abs_diff}"


def test_compile_baseline_json_shape(compiled_bundle_dir) -> None:
    """``compile_baseline.json`` has the real ``BaselineReport`` fields —
    derived from an actual ``torch.compile`` pass during emission, not
    from ``DynamoReport`` diagnostics."""
    bundle_dir, _, _ = compiled_bundle_dir
    baseline_path = bundle_dir / "compile_baseline.json"
    if not baseline_path.is_file():
        pytest.skip("compile_baseline.json not emitted — torch.compile may have failed on this model")
    payload = json.loads(baseline_path.read_text())
    for k in ("backend", "cold_compile_ms", "warm_run_ms", "num_graph_breaks", "compiled_op_fraction"):
        assert k in payload, f"missing baseline field: {k}"
    # Sanity: timing fields are non-negative floats.
    assert isinstance(payload["cold_compile_ms"], (int, float))
    assert isinstance(payload["warm_run_ms"], (int, float))
    assert payload["cold_compile_ms"] >= 0
    assert payload["warm_run_ms"] >= 0


def test_graph_breaks_json_shape(compiled_bundle_dir) -> None:
    """graph_breaks.json has the canonical fields and lists are lists."""
    bundle_dir, _, _ = compiled_bundle_dir
    payload = json.loads((bundle_dir / "graph_breaks.json").read_text())
    assert isinstance(payload.get("graph_breaks"), list)
    assert isinstance(payload.get("warnings"), list)


def test_execution_plan_yaml_is_real(compiled_bundle_dir) -> None:
    """``execution_plan.yaml`` has the canonical planner schema:
    placements + execution_order + memory_plans with non-negative
    peak_bytes."""
    import yaml

    bundle_dir, _, _ = compiled_bundle_dir
    path = bundle_dir / "execution_plan.yaml"
    assert path.is_file(), "execution_plan.yaml should be emitted"

    plan = yaml.safe_load(path.read_text())
    assert isinstance(plan, dict)
    # Planner schema-required keys.
    for key in ("placements", "copies", "execution_order", "memory_plans"):
        assert key in plan, f"execution_plan missing key: {key}"
    # Single-device target → everything on device 0.
    for p in plan["placements"]:
        assert p["device"] == 0
    # Memory plan is realistic: at least one entry, peak_bytes >= 0.
    assert plan["memory_plans"], "single-device plan should have at least one memory plan"
    for mp in plan["memory_plans"]:
        assert mp["peak_bytes"] >= 0


def test_memory_plan_yaml_matches_execution_plan(compiled_bundle_dir) -> None:
    """``memory_plan.yaml`` (when present) is a per-device list and
    agrees with the ``execution_plan.yaml`` device count."""
    import yaml

    bundle_dir, _, _ = compiled_bundle_dir
    mp_path = bundle_dir / "memory_plan.yaml"
    if not mp_path.is_file():
        pytest.skip("memory_plan.yaml not emitted (planner returned no memory_plans)")
    mp = yaml.safe_load(mp_path.read_text())
    assert isinstance(mp, list)
    assert len(mp) >= 1
    devices = {int(entry["device"]) for entry in mp}
    ep = yaml.safe_load((bundle_dir / "execution_plan.yaml").read_text())
    ep_devices = {int(e["device"]) for e in ep["memory_plans"]}
    assert devices == ep_devices


def test_gap_analysis_json_is_real(compiled_bundle_dir) -> None:
    """``gap_analysis.json`` carries the real NetworkAnalysis payload
    — clusters, flops, bytes, opportunities — not just a bool."""
    bundle_dir, _, _ = compiled_bundle_dir
    ga_path = bundle_dir / "gap_analysis.json"
    assert ga_path.is_file(), "gap_analysis.json should be emitted"

    gap = json.loads(ga_path.read_text())
    for key in (
        "model_name",
        "total_params",
        "total_flops",
        "total_bytes",
        "clusters",
        "unclustered_ops",
        "data_flow",
        "bottleneck_clusters",
        "optimization_opportunities",
    ):
        assert key in gap, f"gap_analysis missing key: {key}"
    # Sanity: counts are non-negative.
    assert int(gap["total_params"]) >= 0
    assert int(gap["total_flops"]) >= 0
    assert isinstance(gap["clusters"], list)


def test_bundle_runner_exposes_plan_and_gap(compiled_bundle_dir) -> None:
    """``LoadedBundle`` surfaces the new plan + gap fields so consumers
    (e.g. the promotion pipeline) can introspect without re-parsing."""
    bundle_dir, _, _ = compiled_bundle_dir
    bundle = load_bundle(bundle_dir)
    # Present on CPU/GPU targets that emit an ExecutionPlan.
    assert bundle.execution_plan is not None
    assert "placements" in bundle.execution_plan
    # Gap analysis is always emitted for a compiled model.
    assert bundle.gap_analysis is not None
    assert bundle.gap_analysis["model_name"]


def test_kernel_contracts_dir_emitted(compiled_bundle_dir) -> None:
    """``kernel_contracts/`` contains one YAML per op that needs a
    kernel; each file has the real KernelContract fields (op_name,
    cost, layouts, perf_target_us) — not a stub."""
    import yaml

    bundle_dir, _, _ = compiled_bundle_dir
    contracts_dir = bundle_dir / "kernel_contracts"
    if not contracts_dir.is_dir():
        # If the payload module has no extractable contracts this is
        # legitimately empty — the TinyMLP linalg.matmul should produce
        # at least one, so fail loudly rather than skip.
        pytest.fail("kernel_contracts/ directory should exist for TinyMLP")

    yaml_files = sorted(contracts_dir.glob("*.yaml"))
    assert yaml_files, "at least one kernel contract YAML expected"

    # Inspect the first contract — enough to verify schema correctness.
    payload = yaml.safe_load(yaml_files[0].read_text())
    assert isinstance(payload, dict)
    for key in ("op_name", "cost", "supported_dtypes", "fusable", "priority"):
        assert key in payload, f"contract missing key: {key}"
    cost = payload["cost"]
    assert "flops" in cost and cost["flops"] >= 0
    assert "bytes_read" in cost and cost["bytes_read"] >= 0


def test_bundle_runner_exposes_kernel_contracts(compiled_bundle_dir) -> None:
    """``LoadedBundle.kernel_contracts`` mirrors the on-disk YAMLs so
    downstream consumers can pass them straight to kernel providers."""
    bundle_dir, _, _ = compiled_bundle_dir
    bundle = load_bundle(bundle_dir)
    assert bundle.kernel_contracts is not None
    assert len(bundle.kernel_contracts) >= 1
    # Each loaded contract is a dict with an op_name.
    for _stem, contract in bundle.kernel_contracts.items():
        assert isinstance(contract, dict)
        assert contract.get("op_name")


# ---------------------------------------------------------------------------
# Phase-1 error-surfacing contract
# ---------------------------------------------------------------------------


def test_emission_report_returned_not_thrown(tmp_path: Path) -> None:
    """A call with unsupported inputs returns a report — no broad
    ``except Exception`` swallow, no silent empty bundle."""
    from compgen.runtime.bundle_emit import emit_extended_artefacts
    from compgen.runtime.errors import BundleEmissionReport

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}))

    class _FakeCapture:
        exported_program = None
        diagnostics = None

    # Calling with no model → compile_baseline is skipped with reason
    # (not failed), golden_outputs is skipped (no eager forward source).
    report = emit_extended_artefacts(
        bundle_dir,
        capture_artifact=_FakeCapture(),
        sample_inputs=(torch.randn(2, 2),),
    )
    assert isinstance(report, BundleEmissionReport)
    names = {s.name for s in report.statuses}
    # Every contract slot must have a status entry.
    for slot in (
        "exported_program",
        "golden_inputs",
        "golden_outputs",
        "compile_baseline",
        "graph_breaks",
        "execution_plan",
        "memory_plan",
        "gap_analysis",
        "kernel_contracts",
        "transforms",
        "generated_kernels",
        "verification_report",
    ):
        assert slot in names, f"slot {slot!r} missing from report"
    # No failures in this benign scenario — everything is ok or skipped.
    assert not report.failed, f"unexpected failures: {[s.name for s in report.failed]}"


def test_emission_report_written_to_manifest(tmp_path: Path) -> None:
    """Per-artifact statuses land in ``manifest.json::extended_artifacts``."""
    from compgen.runtime.bundle_emit import emit_extended_artefacts

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}))

    class _FakeCapture:
        exported_program = None
        diagnostics = None

    emit_extended_artefacts(
        bundle_dir,
        capture_artifact=_FakeCapture(),
        sample_inputs=(torch.randn(2, 2),),
    )
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert "extended_artifacts" in manifest
    block = manifest["extended_artifacts"]
    for _name, entry in block.items():
        assert entry["status"] in {"ok", "skipped", "failed"}


def test_failure_surfaces_as_failed_status(tmp_path: Path) -> None:
    """Disk-write failure (sample_inputs with unserializable object)
    shows up as ``failed`` in the report — not swallowed."""
    from compgen.runtime.bundle_emit import emit_extended_artefacts

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}))

    class _FakeCapture:
        exported_program = None
        diagnostics = None

    class _Unserializable:
        def __reduce_ex__(self, _protocol):  # type: ignore[override]
            raise RuntimeError("deliberately unserializable for testing")

    report = emit_extended_artefacts(
        bundle_dir,
        capture_artifact=_FakeCapture(),
        sample_inputs=(_Unserializable(),),
    )
    failed_names = {s.name for s in report.failed}
    assert "golden_inputs" in failed_names
    # And the error message is carried along.
    gi = next(s for s in report.statuses if s.name == "golden_inputs")
    assert "deliberately unserializable" in (gi.error or "")
