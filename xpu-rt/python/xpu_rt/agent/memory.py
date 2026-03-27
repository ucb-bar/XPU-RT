"""Persistent agent memory across compilation sessions.

Stores cost model calibration data, strategy history, and pass ordering
preferences. Updated by CalibrateAction after real hardware benchmarks.

The key value: after calibration, the roofline cost model becomes accurate
enough to guide decisions without running benchmarks every time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CostCalibration:
    """Per-device cost model calibration data.

    Maps (device_name, op_type) → correction_factor.
    The roofline estimate is multiplied by this factor to get
    a calibrated estimate.
    """

    # device_name → {op_type → correction_factor}
    factors: dict[str, dict[str, float]] = field(default_factory=dict)

    def get_factor(self, device_name: str, op_type: str) -> float:
        """Get correction factor. Returns 1.0 if uncalibrated."""
        return self.factors.get(device_name, {}).get(op_type, 1.0)

    def update(self, device_name: str, op_type: str, estimated_us: float, measured_us: float) -> None:
        """Update correction factor from a measurement."""
        if estimated_us <= 0:
            return
        new_factor = measured_us / estimated_us
        if device_name not in self.factors:
            self.factors[device_name] = {}

        old = self.factors[device_name].get(op_type, 1.0)
        # Exponential moving average (α=0.3) for stability
        self.factors[device_name][op_type] = 0.7 * old + 0.3 * new_factor


@dataclass
class StrategyRecord:
    """Record of one optimization strategy attempt."""

    model_name: str
    target_name: str
    pattern_type: str
    actions_taken: list[str]
    estimated_improvement: float
    actual_improvement: float
    success: bool


@dataclass
class AgentMemory:
    """Persistent state the agent accumulates across sessions."""

    cost_calibration: CostCalibration = field(default_factory=CostCalibration)
    strategy_history: list[StrategyRecord] = field(default_factory=list)
    pass_orderings: dict[str, list[str]] = field(default_factory=dict)
    session_count: int = 0

    def save(self, path: str | Path) -> None:
        """Persist memory to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cost_calibration": self.cost_calibration.factors,
            "strategy_history": [asdict(s) for s in self.strategy_history],
            "pass_orderings": self.pass_orderings,
            "session_count": self.session_count,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> AgentMemory:
        """Load memory from JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        mem = cls()
        mem.cost_calibration.factors = data.get("cost_calibration", {})
        mem.strategy_history = [
            StrategyRecord(**s) for s in data.get("strategy_history", [])
        ]
        mem.pass_orderings = data.get("pass_orderings", {})
        mem.session_count = data.get("session_count", 0)
        return mem

    def record_strategy(
        self,
        model_name: str,
        target_name: str,
        pattern_type: str,
        actions: list[str],
        estimated: float,
        actual: float,
        success: bool,
    ) -> None:
        """Record a strategy attempt."""
        self.strategy_history.append(StrategyRecord(
            model_name=model_name,
            target_name=target_name,
            pattern_type=pattern_type,
            actions_taken=actions,
            estimated_improvement=estimated,
            actual_improvement=actual,
            success=success,
        ))

    def best_pass_ordering(self, pattern_type: str) -> list[str] | None:
        """Get the best known pass ordering for a pattern type."""
        return self.pass_orderings.get(pattern_type)

    def record_pass_ordering(self, pattern_type: str, ordering: list[str]) -> None:
        """Record a successful pass ordering."""
        self.pass_orderings[pattern_type] = ordering


__all__ = ["AgentMemory", "CostCalibration", "StrategyRecord"]
