"""Higher-level benchmark result envelope with JSON serialization.

Provides a frozen ``BenchmarkResult`` dataclass that captures the essential
metrics of a single benchmark run together with provenance metadata, plus
helpers for JSON round-trip and pairwise comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    """Frozen envelope for a single benchmark run's key metrics.

    This is a *higher-level* summary intended for cross-suite comparison and
    CI dashboards.  For the full verbose schema used inside the study harness
    see ``benchmarks.record.RunRecord``; for the normalised per-suite
    projection see ``compgen.benchmarks.common.results.NormalizedSuiteResult``.

    Attributes:
        suite: Benchmark suite identifier (e.g. ``"torchbench"``).
        workload: Workload / model name within the suite.
        capture_ok: Whether frontend capture succeeded.
        export_ok: Whether ``torch.export`` succeeded.
        correctness_ok: Whether correctness checks passed.
        compile_time_s: Total compilation wall-time in seconds.
        latency_ms_p50: Median (p50) inference latency in milliseconds.
        throughput: Throughput in samples per second.
        peak_memory_mb: Peak device memory usage in megabytes.
        unsupported_ops: Count of ops that could not be lowered.
        auto_translations_added: Automatic op translations applied.
        generated_kernels: Number of generated kernel specs.
        generated_passes: Number of generated transform passes.
        generated_guards: Number of generated guards.
        promoted_artifacts: Number of artifacts promoted after verification.
        run_id: Unique identifier for this run.
        timestamp: ISO-8601 timestamp string.
        tags: Arbitrary tags attached to this run.
        source_commit: Git commit hash of the source tree.
    """

    # --- core metrics ---
    suite: str
    workload: str
    capture_ok: bool
    export_ok: bool
    correctness_ok: bool
    compile_time_s: float
    latency_ms_p50: float
    throughput: float
    peak_memory_mb: float
    unsupported_ops: int
    auto_translations_added: int
    generated_kernels: int
    generated_passes: int
    generated_guards: int
    promoted_artifacts: int

    # --- provenance metadata ---
    run_id: str = ""
    timestamp: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_commit: str = ""

    # -- serialisation helpers --------------------------------------------------

    def write_json(self, out_path: str | Path) -> None:
        """Serialise this result to a JSON file.

        Args:
            out_path: Destination path.  Parent directories are created if
                they do not already exist.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        # ``tags`` is a tuple but ``asdict`` already converts it to a list,
        # which is the natural JSON representation.
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> BenchmarkResult:
    """Deserialise a ``BenchmarkResult`` from a JSON file.

    Args:
        path: Path to the JSON file previously written by
            :meth:`BenchmarkResult.write_json`.

    Returns:
        A new ``BenchmarkResult`` instance.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # ``tags`` is stored as a JSON list; convert back to a tuple.
    if "tags" in raw and isinstance(raw["tags"], list):
        raw["tags"] = tuple(raw["tags"])
    return BenchmarkResult(**raw)


def compare_results(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
) -> dict[str, float]:
    """Compute relative deltas between two benchmark results.

    Returns a dictionary with the following keys:

    * ``latency_delta`` -- absolute difference ``candidate - baseline`` in ms.
    * ``throughput_delta`` -- absolute difference ``candidate - baseline``.
    * ``memory_delta`` -- absolute difference ``candidate - baseline`` in MB.

    Negative latency / memory deltas mean the candidate improved; positive
    throughput delta means improvement.

    Args:
        baseline: The reference result.
        candidate: The result to compare against *baseline*.

    Returns:
        Mapping of metric name to signed delta value.
    """
    return {
        "latency_delta": candidate.latency_ms_p50 - baseline.latency_ms_p50,
        "throughput_delta": candidate.throughput - baseline.throughput,
        "memory_delta": candidate.peak_memory_mb - baseline.peak_memory_mb,
    }


__all__ = [
    "BenchmarkResult",
    "compare_results",
    "read_json",
]
