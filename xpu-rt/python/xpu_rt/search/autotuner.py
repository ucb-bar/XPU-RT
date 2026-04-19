"""Autotuning harness over ``CompGenOptions``.

Searches a space of options configurations and records the
compilation outcome for each. Uses ``PipelineCache`` underneath so
repeated compilations are cheap.

Search strategies:

- ``"grid"``    -- dense cartesian product of all axis values.
- ``"random"``  -- uniform random samples (``n_trials`` of them).
- ``"baseline"`` -- just the base options (one trial; useful for
  regression reference).

Metric: the autotuner accepts a ``metric_fn(result) -> float`` that
takes a :class:`compgen.pipeline.PipelineResult` and returns a
scalar (lower is better). Defaults to opaque rate.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import structlog

from compgen.options import CompGenOptions
from compgen.pipeline import PipelineCache, PipelineResult

log = structlog.get_logger()


@dataclass(frozen=True)
class OptionsAxis:
    """One search dimension over a single CompGenOptions field.

    ``field_name`` is the dataclass attr to vary. ``values`` is the
    list of values to try.
    """

    field_name: str
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"OptionsAxis {self.field_name!r} needs ≥1 value")
        # Validate the field exists on CompGenOptions.
        import dataclasses
        fields = {f.name for f in dataclasses.fields(CompGenOptions)}
        if self.field_name not in fields:
            raise ValueError(
                f"{self.field_name!r} is not a CompGenOptions field; "
                f"known: {sorted(fields)[:10]}..."
            )


@dataclass
class AutotuneTrial:
    """One trial's outcome."""

    options: CompGenOptions
    result: PipelineResult
    metric: float
    index: int = 0


@dataclass
class AutotuneResult:
    """Final outcome of an autotune run."""

    trials: list[AutotuneTrial] = field(default_factory=list)
    best_index: int = -1

    @property
    def best_trial(self) -> AutotuneTrial | None:
        if self.best_index < 0 or self.best_index >= len(self.trials):
            return None
        return self.trials[self.best_index]

    @property
    def best_metric(self) -> float:
        bt = self.best_trial
        return bt.metric if bt is not None else float("inf")

    def summary(self) -> str:
        if not self.trials:
            return "(no trials)"
        lines = [f"{len(self.trials)} trials, best metric: {self.best_metric:.4f}"]
        for t in self.trials:
            mark = "*" if t.index == self.best_index else " "
            lines.append(
                f"  {mark} trial {t.index}: metric={t.metric:.4f} "
                f"stages_run={t.result.stages_run}"
            )
        return "\n".join(lines)


# --- default metric ---------------------------------------------------------


def _default_metric(result: PipelineResult) -> float:
    """Lower is better. Default: opaque rate + pipeline failure penalty."""
    if result is None or result.module is None:
        return float("inf")
    total = 0
    opaque = 0
    for op in result.module.walk():
        if op.name in {"builtin.module", "func.func", "func.return"}:
            continue
        total += 1
        if op.name == "func.call":
            callee = op.properties.get("callee")
            if callee is not None and "aten_" in str(callee):
                opaque += 1
    return (opaque / total) if total else 1.0


# --- the autotuner ----------------------------------------------------------


class Autotuner:
    """Search over ``CompGenOptions`` configurations."""

    def __init__(
        self,
        base: CompGenOptions,
        axes: Sequence[OptionsAxis],
        *,
        strategy: str = "grid",
        n_trials: int = 16,
        seed: int = 0,
        cache: PipelineCache | None = None,
        metric_fn: Callable[[PipelineResult], float] | None = None,
    ) -> None:
        if strategy not in ("grid", "random", "baseline"):
            raise ValueError(
                f"strategy must be grid/random/baseline, got {strategy!r}"
            )
        self.base = base
        self.axes = tuple(axes)
        self.strategy = strategy
        self.n_trials = n_trials
        self.seed = seed
        self.cache = cache if cache is not None else PipelineCache()
        self.metric_fn = metric_fn if metric_fn is not None else _default_metric

    def _enumerate(self) -> Iterable[CompGenOptions]:
        if self.strategy == "baseline":
            yield self.base
            return

        axes_values = [axis.values for axis in self.axes]
        if self.strategy == "grid":
            for combo in itertools.product(*axes_values):
                changes = {
                    axis.field_name: value
                    for axis, value in zip(self.axes, combo, strict=True)
                }
                yield self.base.replace(**changes)
            return

        # random
        rng = random.Random(self.seed)
        for _ in range(self.n_trials):
            changes = {
                axis.field_name: rng.choice(axis.values)
                for axis in self.axes
            }
            yield self.base.replace(**changes)

    def search(
        self,
        model: Any,
        example_inputs: tuple[Any, ...] | None = None,
        *,
        workload_name: str = "unnamed",
    ) -> AutotuneResult:
        result = AutotuneResult()
        best_idx = -1
        best_metric = float("inf")
        for idx, opts in enumerate(self._enumerate()):
            pr = self.cache.compile(
                model, example_inputs, options=opts,
                workload_name=workload_name,
            )
            metric = self.metric_fn(pr)
            trial = AutotuneTrial(options=opts, result=pr, metric=metric, index=idx)
            result.trials.append(trial)
            if metric < best_metric:
                best_metric = metric
                best_idx = idx
            log.info(
                "autotune.trial", index=idx, metric=metric,
                stages_run=pr.stages_run,
            )
        result.best_index = best_idx
        return result


__all__ = [
    "Autotuner",
    "AutotuneResult",
    "AutotuneTrial",
    "OptionsAxis",
]
