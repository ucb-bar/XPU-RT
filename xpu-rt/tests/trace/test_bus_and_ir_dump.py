"""Integration tests for the trace bus, publishers, IR dump writer, and
graph digest tools.

These tests exercise the real code paths (no mocks) on a tiny PyTorch
model so any regression in the trace/dump hooks breaks loudly.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from compgen.agent.analyzer import NetworkAnalyzer
from compgen.analysis.graph_digest import (
    build_chunk_view,
    build_digest,
)
from compgen.capture.torch_export import capture_frontend_artifact
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.llm.recorder import LLMRecorder
from compgen.targets.schema import (
    ComputeUnit,
    DeviceSpec,
    MemoryLevel,
    TargetProfile,
)
from compgen.trace import (
    IRDumpWriter,
    PassPublisher,
    StagePublisher,
    TraceBus,
    TracingLLMRecorder,
    get_active_bus,
    install_bus,
    install_ir_dump_writer,
    set_active_bus,
)
from xdsl.dialects.builtin import ModuleOp


def _profile() -> TargetProfile:
    return TargetProfile(
        name="cuda-a100",
        devices=[
            DeviceSpec(
                device_type="gpu",
                name="A100",
                compute_units=[
                    ComputeUnit(
                        name="tensor_core",
                        count=432,
                        supported_dtypes={"bf16", "f16", "f32"},
                        peak_tflops=312.0,
                    )
                ],
                memory_hierarchy=[
                    MemoryLevel(
                        name="hbm",
                        size_bytes=80 * 1024**3,
                        bandwidth_gbps=1555.0,
                    ),
                    MemoryLevel(name="shared_memory", size_bytes=167936),
                ],
            )
        ],
    )


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(x)))


@pytest.fixture
def _reset_bus():
    prev = get_active_bus()
    set_active_bus(None)
    try:
        yield
    finally:
        set_active_bus(prev)


def _read_events(bus: TraceBus) -> list[dict]:
    return [json.loads(line) for line in bus.trace_path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# Trace bus basics
# ---------------------------------------------------------------------------


def test_bus_publishes_paired_span_with_parent_id(tmp_path, _reset_bus):
    bus = install_bus(output_dir=tmp_path, session_id="t_span")
    with PassPublisher.span(payload={"name": "p"}) as outer_id:
        with StagePublisher.span(payload={"stage": "s"}) as inner_id:
            assert inner_id
    events = _read_events(bus)
    assert len(events) == 4  # 2 starts + 2 ends
    kinds = [(e["kind"], e["phase"]) for e in events]
    assert kinds == [
        ("pass_run", "start"),
        ("stage_run", "start"),
        ("stage_run", "end"),
        ("pass_run", "end"),
    ]
    # Inner span's parent must equal outer span's event_id
    stage_start = next(e for e in events if e["kind"] == "stage_run" and e["phase"] == "start")
    assert stage_start["parent_event_id"] == outer_id


def test_bus_session_mirror_symlink(tmp_path, _reset_bus):
    mirror = tmp_path / "mirror" / "trace.jsonl"
    bus = install_bus(output_dir=tmp_path / "out", session_id="t_mirror", session_mirror=mirror)
    PassPublisher.emit(name="x")
    # Mirror exists via symlink or fallback pointer file.
    assert mirror.exists() or mirror.with_suffix(".jsonl.json").exists()
    # Ensure trace content is reachable through the mirror path.
    if mirror.is_symlink() or mirror.exists():
        assert bus.trace_path.read_text() == mirror.read_text()


# ---------------------------------------------------------------------------
# IR dump writer
# ---------------------------------------------------------------------------


def test_ir_dump_writer_emits_before_after_and_index(tmp_path, _reset_bus):
    install_bus(output_dir=tmp_path, session_id="t_dump")
    writer = IRDumpWriter(tmp_path, enabled=True)
    install_ir_dump_writer(writer)

    module = ModuleOp([])
    writer.dump(name="pass_a", phase="before", module=module)
    writer.dump(name="pass_a", phase="after", module=module)
    writer.write_final(module)

    dumps = sorted((tmp_path / "ir_dumps").glob("*.mlir"))
    names = [p.name for p in dumps]
    assert names == [
        "0001_pass_a_before.mlir",
        "0002_pass_a_after.mlir",
        "final.mlir",
    ]
    index = json.loads((tmp_path / "ir_dumps" / "index.json").read_text())
    assert index["count"] == 2
    assert index["entries"][0]["name"] == "pass_a"
    assert index["entries"][0]["phase"] == "before"
    # Each index entry records the SHA-256 hash.
    assert index["entries"][0]["ir_hash"].startswith("sha256:")
    install_ir_dump_writer(None)


def test_ir_dump_writer_disabled_emits_nothing(tmp_path, _reset_bus):
    writer = IRDumpWriter(tmp_path, enabled=False)
    install_ir_dump_writer(writer)
    module = ModuleOp([])
    path, h = writer.dump(name="pass_a", phase="before", module=module)
    assert path is None
    assert h == ""
    assert not (tmp_path / "ir_dumps").exists()
    install_ir_dump_writer(None)


# ---------------------------------------------------------------------------
# Tracing LLM recorder — no double-wrap
# ---------------------------------------------------------------------------


class _StubClient:
    def generate(self, request):  # noqa: ANN001
        raise NotImplementedError

    def generate_structured(self, request, schema):  # noqa: ANN001
        raise NotImplementedError


def test_tracing_llm_recorder_is_idempotent(tmp_path, _reset_bus):
    bus = install_bus(output_dir=tmp_path, session_id="t_idem")
    raw = LLMRecorder(wrapped=_StubClient(), log_dir=tmp_path / "llm")
    first = TracingLLMRecorder.wrap(raw, bus)
    second = TracingLLMRecorder.wrap(raw, bus)
    # ``wrap`` returns the underlying recorder on the second call so we
    # do not stack trace events.
    assert second is raw or second is first
    assert getattr(raw, "_compgen_trace_bus", None) is bus


# ---------------------------------------------------------------------------
# Graph digest — non-empty on a tiny model
# ---------------------------------------------------------------------------


def test_build_digest_populates_distributions_on_tiny_model(_reset_bus):
    model = _Tiny()
    inputs = (torch.randn(4, 64),)
    cap = capture_frontend_artifact(model, inputs)
    module, _ = fx_to_xdsl(cap.exported_program, **cap.strict_import_options())
    profile = _profile()
    analysis = NetworkAnalyzer().analyze(cap.exported_program, profile, model_name="Tiny")

    digest = build_digest(analysis, module=module, target_name=profile.name)
    # At minimum we expect the module carries some tensor results and we
    # classify their dtype + rank.
    assert digest.dtype_spectrum, "expected at least one dtype"
    assert digest.dim_spectrum.rank_histogram, "expected at least one rank entry"
    assert digest.memory_footprint_bytes > 0
    # The compact summary fits inside the target budget.
    summary = digest.to_prompt_summary(max_bytes=2048)
    assert len(summary) <= 2048
    assert "graph_digest" in summary


def test_focus_chunk_returns_knobs_and_dof_side_by_side(_reset_bus):
    model = _Tiny()
    inputs = (torch.randn(4, 64),)
    cap = capture_frontend_artifact(model, inputs)
    module, _ = fx_to_xdsl(cap.exported_program, **cap.strict_import_options())
    profile = _profile()
    analysis = NetworkAnalyzer().analyze(cap.exported_program, profile, model_name="Tiny")
    view = build_chunk_view(analysis, profile, {}, module=module)
    # Granularity options are structured dicts carrying source +
    # oracle_advisory flag from recommend_granularity.
    gran_vals = {entry["granularity"] for entry in view.decision_knobs.granularity_options}
    assert gran_vals == {"MICRO", "NORMAL", "MEGA"}
    assert any(entry.get("oracle_advisory") for entry in view.decision_knobs.granularity_options), (
        "no granularity carries oracle advisory"
    )
    assert all(entry.get("source", "").startswith("oracle:") for entry in view.decision_knobs.granularity_options), (
        "granularity entries missing source attribution"
    )
    assert view.decision_knobs.memory_tier_options  # non-empty
    # DoF: open-ended axes + archetype space.
    assert view.dof_description.archetypes  # non-empty
    assert view.dof_description.axes  # non-empty
    # Envelope facts carry the target name we asked about.
    assert view.envelope_facts.get("target") == profile.name
