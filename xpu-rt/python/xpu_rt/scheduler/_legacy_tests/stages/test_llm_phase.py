"""Tests for the P14 ``llm_phase`` attribute on CompilationStage."""

from __future__ import annotations

from pathlib import Path

from compgen.stages.base import CompilationStage, StageContract
from compgen.stages.registry import StageRegistry, TargetDialectStack
from xdsl.dialects.builtin import ModuleOp


class _PhaseTagged(CompilationStage):
    """Minimal concrete stage for phase tests."""

    def __init__(self, label: str, phase: int | None) -> None:
        super().__init__()
        self._label = label
        self.llm_phase = phase

    @property
    def name(self) -> str:
        return self._label

    @property
    def description(self) -> str:
        return self._label

    def input_contract(self) -> StageContract:
        return StageContract(stage_name=self._label)

    def output_contract(self) -> StageContract:
        return StageContract(stage_name=self._label)

    def shared_passes(self, module: ModuleOp, target):  # type: ignore[override]
        return module

    @property
    def requirements_doc_path(self) -> Path:
        return Path("/tmp/none")


def test_default_phase_is_none() -> None:
    assert CompilationStage.llm_phase is None


def test_subclass_can_override_phase() -> None:
    s = _PhaseTagged("s", 3)
    assert s.llm_phase == 3


def test_phase_monotonicity_allows_increasing() -> None:
    r = StageRegistry()
    stack = TargetDialectStack(
        target_name="t",
        stages=[_PhaseTagged("a", 2), _PhaseTagged("b", 3), _PhaseTagged("c", 5)],
    )
    r.register_target_stack(stack)
    assert r.check_phase_monotonicity("t") == []


def test_phase_monotonicity_catches_regression() -> None:
    r = StageRegistry()
    stack = TargetDialectStack(
        target_name="t",
        stages=[_PhaseTagged("a", 4), _PhaseTagged("b", 2)],  # 2 after 4: regression
    )
    r.register_target_stack(stack)
    violations = r.check_phase_monotonicity("t")
    assert len(violations) == 1
    assert "'b'" in violations[0] and "'a'" in violations[0]


def test_phase_monotonicity_ignores_untagged() -> None:
    r = StageRegistry()
    stack = TargetDialectStack(
        target_name="t",
        stages=[
            _PhaseTagged("a", 2),
            _PhaseTagged("b", None),  # no opinion
            _PhaseTagged("c", 3),
        ],
    )
    r.register_target_stack(stack)
    assert r.check_phase_monotonicity("t") == []


def test_phase_monotonicity_missing_target() -> None:
    r = StageRegistry()
    assert "no stack" in r.check_phase_monotonicity("nope")[0]
