"""Verify that XpuRtBackend stopped mislabeling torch.compile as XPU-RT.

Before this refactor, ``compile_and_benchmark`` ran
``torch.compile(backend="inductor")`` and returned
``mode="xpu_rt_compiled"``. Now the baseline path is honest
(``mode="torch_compile_inductor"``) and the XPU-RT path is a
separate method that delegates to an
:class:`xpu_rt.runtime.executor.XpuRtExecutor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from xpu_rt.runtime.errors import AdapterUnavailableError
from xpu_rt.runtime.measured_cost import MeasuredCost, MeasurementSample
from xpu_rt.runtime.torch_backend import XpuRtBackend, CompileResult


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).relu()


def test_baseline_mode_label_is_truthful() -> None:
    """The baseline path labels itself as torch.compile, not XPU-RT."""
    backend = XpuRtBackend()
    result = backend.compile_and_benchmark_baseline(
        _TinyMLP(), (torch.randn(2, 8),),
        device="cpu", num_iterations=3, warmup=1,
    )
    assert isinstance(result, CompileResult)
    assert result.mode == "torch_compile_inductor"
    assert result.cycles_total is None
    assert result.correctness_vs_eager == "skipped"


@dataclass
class _StubExecutor:
    """Test double for XpuRtExecutor — no toolchain required."""

    name: str = "stub_executor"
    target_id: str = "stub_target"
    available: bool = True
    available_reason: str = ""
    cycles: int | None = 42
    correctness: str = "pass"

    def is_available(self) -> tuple[bool, str]:
        return self.available, self.available_reason

    def execute(
        self,
        bundle_dir: Path,
        *,
        run_dir: Path | None = None,
        sample_inputs: tuple[Any, ...] | None = None,
    ) -> MeasuredCost:
        return MeasuredCost(
            executor=self.name,
            target_id=self.target_id,
            cycles_total=self.cycles,
            correctness_vs_eager=self.correctness,  # type: ignore[arg-type]
            samples=(
                MeasurementSample(
                    region_id="r0",
                    op_family="matmul",
                    cycles=self.cycles,
                    correctness=self.correctness,  # type: ignore[arg-type]
                ),
            ),
        )


def test_xpu_rt_path_delegates_to_executor(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    backend = XpuRtBackend()
    executor = _StubExecutor()

    result = backend.compile_and_benchmark_xpu_rt(
        bundle, executor=executor, batch_size=4,
    )
    assert result.mode == "xpu_rt_stub_executor"
    assert result.device == "stub_target"
    assert result.cycles_total == 42
    assert result.correctness_vs_eager == "pass"
    # No wall-clock measured: latency stays 0; throughput stays 0.
    assert result.latency_median_us == 0.0
    assert result.throughput_samples_per_sec == 0.0


def test_xpu_rt_path_surfaces_unavailable_executor(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    backend = XpuRtBackend()
    executor = _StubExecutor(available=False, available_reason="toolchain missing")
    with pytest.raises(AdapterUnavailableError, match="toolchain missing"):
        backend.compile_and_benchmark_xpu_rt(bundle, executor=executor)


def test_legacy_alias_is_still_baseline_path() -> None:
    """compile_and_benchmark() used to mislabel; now it's an alias of the baseline."""
    backend = XpuRtBackend()
    result = backend.compile_and_benchmark(
        _TinyMLP(), (torch.randn(2, 8),),
        device="cpu", num_iterations=3, warmup=1,
    )
    assert result.mode == "torch_compile_inductor"
