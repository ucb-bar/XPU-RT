"""Tests for the P2 drive_loop hook in compile_model.

We avoid running the full compile_model path (expensive; requires a
real device). Instead we test the signature + dataclass extension +
behavior of the drive_loop parameter in isolation by patching the
internals.
"""

from __future__ import annotations

import inspect

from compgen.api import CompiledModel, compile_model


def test_signature_includes_drive_loop_params() -> None:
    sig = inspect.signature(compile_model)
    params = sig.parameters
    assert "drive_loop" in params
    assert params["drive_loop"].default is None
    assert "drive_loop_phases" in params
    assert params["drive_loop_phases"].default == (2, 3)


def test_compiled_model_has_drive_loop_result_field() -> None:
    from dataclasses import fields

    names = {f.name for f in fields(CompiledModel)}
    assert "drive_loop_result" in names


def test_drive_loop_default_none_preserves_legacy_shape() -> None:
    # Build a CompiledModel instance directly to confirm the new
    # field has a default (backward-compat with existing callers
    # that construct CompiledModel in tests).
    from dataclasses import fields

    for f in fields(CompiledModel):
        if f.name == "drive_loop_result":
            assert f.default is None
            return
    raise AssertionError("drive_loop_result field not found")


def test_drive_loop_object_with_run_is_called(monkeypatch) -> None:
    """Build a mock drive-loop object and confirm compile_model calls its ``run``.

    We patch the internals just enough that we don't do a real compile:
    we stop at the capture step and assert the drive_loop was invoked
    first (it happens AFTER fx_to_xdsl; we stub fx_to_xdsl to return a
    bare module + empty diagnostics, then stub downstream to no-op).
    """
    import compgen.api as api_mod
    from xdsl.dialects.builtin import ModuleOp

    class _MockDriveLoop:
        def __init__(self) -> None:
            self.run_called = False
            self.phases_seen: list[int] = []
            self.context = {"policy": lambda *a, **k: []}

        def run(self, *, phases, policy):
            self.run_called = True
            self.phases_seen = list(phases)
            # Mirror DriveLoopResult.__init__ shape
            class _Result:
                total_elapsed_ms = 1.0
                phase_summaries: list = []
            return _Result()

    mock_loop = _MockDriveLoop()

    # --- Stub the pipeline to avoid real heavy work ---
    class _StubArtifact:
        exported_program = None
        def strict_import_options(self):
            return {}

    class _StubAnalysis:
        dossier = None
        model_name = "stub"
        total_params = 0
        total_flops = 0
        total_bytes = 0
        clusters = ()
        bottleneck_clusters = ()
        optimization_opportunities = ()

    monkeypatch.setattr(api_mod, "capture_frontend_artifact", lambda *a, **k: _StubArtifact())
    monkeypatch.setattr(api_mod, "fx_to_xdsl", lambda *a, **k: (ModuleOp([]), []))
    monkeypatch.setattr(
        api_mod.NetworkAnalyzer, "analyze",
        lambda self, *a, **k: _StubAnalysis(),
    )
    monkeypatch.setattr(
        "compgen.ir.ukernel.annotate.annotate_ukernel_ops",
        lambda *a, **k: 0,
    )

    class _StubEqSat:
        changed = False
    monkeypatch.setattr(api_mod, "run_eqsat_pass", lambda *a, **k: _StubEqSat())

    class _StubPipelineResult:
        passed = True
        stages_run = 0
        stage_results = ()
        all_artifacts = {}
    monkeypatch.setattr(
        api_mod.StageRegistry, "run_pipeline",
        lambda self, *a, **k: _StubPipelineResult(),
    )

    # Build a minimal CompGenDevice-shaped stub
    from dataclasses import dataclass, field as _dc_field
    from compgen.stages.registry import TargetDialectStack

    @dataclass
    class _StubProfile:
        name: str = "stub"
        devices: list = _dc_field(default_factory=list)

    class _StubDevice:
        profile = _StubProfile()
        capabilities = None
        dialect_stack = TargetDialectStack(target_name="stub", stages=[])
        generated_target = None

    result = compile_model(
        model=None,  # capture_frontend_artifact stubbed → model arg unused
        target_device=_StubDevice(),
        drive_loop=mock_loop,
        drive_loop_phases=(2, 3),
    )

    assert mock_loop.run_called is True
    assert mock_loop.phases_seen == [2, 3]
    assert result.drive_loop_result is not None


def test_drive_loop_none_skips_loop(monkeypatch) -> None:
    """When drive_loop=None, drive_loop_result stays None."""
    import compgen.api as api_mod
    from xdsl.dialects.builtin import ModuleOp

    class _StubArtifact:
        exported_program = None
        def strict_import_options(self):
            return {}

    class _StubAnalysis:
        dossier = None
        model_name = "stub"
        total_params = 0
        total_flops = 0
        total_bytes = 0
        clusters = ()
        bottleneck_clusters = ()
        optimization_opportunities = ()

    monkeypatch.setattr(api_mod, "capture_frontend_artifact", lambda *a, **k: _StubArtifact())
    monkeypatch.setattr(api_mod, "fx_to_xdsl", lambda *a, **k: (ModuleOp([]), []))
    monkeypatch.setattr(
        api_mod.NetworkAnalyzer, "analyze",
        lambda self, *a, **k: _StubAnalysis(),
    )
    monkeypatch.setattr(
        "compgen.ir.ukernel.annotate.annotate_ukernel_ops",
        lambda *a, **k: 0,
    )

    class _StubEqSat:
        changed = False
    monkeypatch.setattr(api_mod, "run_eqsat_pass", lambda *a, **k: _StubEqSat())

    class _StubPipelineResult:
        passed = True
        stages_run = 0
        stage_results = ()
        all_artifacts = {}
    monkeypatch.setattr(
        api_mod.StageRegistry, "run_pipeline",
        lambda self, *a, **k: _StubPipelineResult(),
    )

    from dataclasses import dataclass, field as _dc_field
    from compgen.stages.registry import TargetDialectStack

    @dataclass
    class _StubProfile:
        name: str = "stub"
        devices: list = _dc_field(default_factory=list)

    class _StubDevice:
        profile = _StubProfile()
        capabilities = None
        dialect_stack = TargetDialectStack(target_name="stub", stages=[])
        generated_target = None

    result = compile_model(
        model=None, target_device=_StubDevice(), drive_loop=None,
    )
    assert result.drive_loop_result is None
