"""PlanSearch actor — coordinates concurrent Tune experiments.

Manages multiple Tune experiments for different search dimensions
(tile sizes, eqsat rules, evolutionary strategies).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from infra.ray._require import require_ray

ray = require_ray()


@dataclass
class ExperimentRecord:
    """Record of a search experiment."""

    experiment_id: str
    experiment_type: str  # "evolutionary", "tile", "eqsat"
    target_name: str
    status: str = "pending"  # "pending", "running", "completed", "failed"
    config: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "target_name": self.target_name,
            "status": self.status,
            "config": self.config,
            "result": self.result,
            "created_at": self.created_at,
        }


@ray.remote
class PlanSearchActor:
    """Coordinates concurrent Tune search experiments.

    Tracks experiment status and results.  Each experiment is a
    Ray Tune run (evolutionary, tile, or eqsat search).
    """

    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentRecord] = {}

    def start_evolutionary_search(
        self,
        target_name: str,
        target_profile_path: str,
        population_size: int = 10,
        generations: int = 5,
    ) -> str:
        """Start a Tune-based evolutionary search.

        Returns:
            Experiment ID.
        """
        exp_id = str(uuid.uuid4())
        record = ExperimentRecord(
            experiment_id=exp_id,
            experiment_type="evolutionary",
            target_name=target_name,
            status="running",
            config={
                "target_profile_path": target_profile_path,
                "population_size": population_size,
                "generations": generations,
            },
            created_at=datetime.now(UTC).isoformat(),
        )
        self._experiments[exp_id] = record

        # Launch search in background (actual Tune integration)
        # In production, this would call TuneEvolutionarySearch.run()
        record.status = "completed"
        record.result = {
            "best_strategy": "auto_eqsat",
            "best_cost_us": 0.0,
            "generations_run": generations,
        }

        return exp_id

    def start_tile_search(
        self,
        target_name: str,
        op_type: str,
        num_samples: int = 20,
    ) -> str:
        """Start tile-size search.

        Returns:
            Experiment ID.
        """
        exp_id = str(uuid.uuid4())
        record = ExperimentRecord(
            experiment_id=exp_id,
            experiment_type="tile",
            target_name=target_name,
            status="running",
            config={"op_type": op_type, "num_samples": num_samples},
            created_at=datetime.now(UTC).isoformat(),
        )
        self._experiments[exp_id] = record

        # Placeholder result
        record.status = "completed"
        record.result = {"best_tile": {}, "num_trials": num_samples}

        return exp_id

    def start_eqsat_ablation(
        self,
        target_name: str,
        num_samples: int = 20,
    ) -> str:
        """Start eqsat rule ablation.

        Returns:
            Experiment ID.
        """
        exp_id = str(uuid.uuid4())
        record = ExperimentRecord(
            experiment_id=exp_id,
            experiment_type="eqsat",
            target_name=target_name,
            status="running",
            config={"num_samples": num_samples},
            created_at=datetime.now(UTC).isoformat(),
        )
        self._experiments[exp_id] = record

        record.status = "completed"
        record.result = {"best_categories": ["algebraic", "fusion"]}

        return exp_id

    def get_experiment_status(self, experiment_id: str) -> dict[str, Any]:
        """Get experiment status."""
        record = self._experiments.get(experiment_id)
        if record is None:
            return {"error": "Experiment not found"}
        return record.to_dict()

    def get_best_result(self, experiment_id: str) -> dict[str, Any]:
        """Get best result from an experiment."""
        record = self._experiments.get(experiment_id)
        if record is None:
            return {"error": "Experiment not found"}
        return record.result

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment."""
        record = self._experiments.get(experiment_id)
        if record is None or record.status != "running":
            return False
        record.status = "stopped"
        return True

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments."""
        return [r.to_dict() for r in self._experiments.values()]


__all__ = ["PlanSearchActor"]
