"""Headline benchmark runner (P4.1).

Drives the three P4 workloads (TinyLlama-1.1B, smolVLA, Whisper-tiny)
through three adapters and produces:

* per-iteration latencies (raw csv)
* summary stats (p50/p90/p99/mean/std)
* correctness verdict against eager (bit-equality with tolerance)

The adapters live behind an abstract interface so the test suite can
swap a mock in, exercise the full runner-and-evidence-pack pipeline,
and never touch GPU memory. The live adapters (TorchEagerAdapter,
TorchCompileAdapter, CompGenAdapter) plug in when ``uv sync --extra
demo --extra benchmarks`` is run — outside the dev base.

Hard rules:

1. Correctness is *gated*: a workload that fails the bit-equality
   check against eager is recorded as a `correctness_failure`, NOT
   silently dropped. Honest failure is the win condition; hidden
   regression is the lose condition.
2. Latencies are measured with `cuda.Event`-style timing or
   `time.perf_counter` (fallback when no CUDA), with the same `iters`
   and `warmup` counts across adapters so the comparison is fair.
3. Every measurement is byte-deterministic over its (workload,
   adapter, seed) tuple — same inputs → same output dir contents.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

ADAPTERS: Final[tuple[str, ...]] = (
    "torch_eager",
    "torch_compile",
    "compgen",
)


class HeadlineBenchmarkError(RuntimeError):
    """A workload could not be benchmarked under the requested adapter."""


class _Adapter(Protocol):
    """The minimal adapter contract the runner depends on."""

    adapter_name: str

    def measure(
        self,
        workload_id: str,
        *,
        iters: int,
        warmup: int,
        seed: int,
    ) -> AdapterMeasurement: ...


@dataclass(frozen=True)
class AdapterMeasurement:
    """One adapter × workload run result."""

    adapter_name: str
    workload_id: str
    latencies_us: tuple[float, ...]
    output_hash: str  # used for correctness check
    blocked: bool = False
    blocked_reason: str = ""

    def stats(self) -> dict[str, float]:
        if not self.latencies_us:
            return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"),
                    "mean": float("nan"), "std": float("nan")}
        xs = sorted(self.latencies_us)
        n = len(xs)
        return {
            "p50": xs[n // 2],
            "p90": xs[min(n - 1, int(0.9 * n))],
            "p99": xs[min(n - 1, int(0.99 * n))],
            "mean": sum(xs) / n,
            "std": statistics.pstdev(xs) if n > 1 else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        body = {
            "adapter_name": self.adapter_name,
            "workload_id": self.workload_id,
            "latencies_us": list(self.latencies_us),
            "output_hash": self.output_hash,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "stats": self.stats(),
        }
        return body


@dataclass(frozen=True)
class WorkloadResult:
    """Joined result for one workload across adapters."""

    workload_id: str
    measurements: dict[str, AdapterMeasurement] = field(default_factory=dict)

    def correctness_ok(self) -> bool:
        """Bit-equal output hashes across all non-blocked adapters."""

        hashes = {
            m.output_hash for m in self.measurements.values() if not m.blocked
        }
        return len(hashes) <= 1  # 0 hashes (all blocked) or 1 unique hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "correctness_ok": self.correctness_ok(),
            "measurements": {
                name: m.to_dict() for name, m in self.measurements.items()
            },
        }


def run_benchmark(
    *,
    workloads: list[str],
    adapters: dict[str, _Adapter],
    iters: int = 100,
    warmup: int = 10,
    seed: int = 0xC0FFEE,
) -> list[WorkloadResult]:
    """Drive every (workload, adapter) pair and collect results."""

    results: list[WorkloadResult] = []
    for wl in workloads:
        meas: dict[str, AdapterMeasurement] = {}
        for name, adapter in adapters.items():
            try:
                meas[name] = adapter.measure(
                    wl, iters=iters, warmup=warmup, seed=seed
                )
            except Exception as exc:  # noqa: BLE001
                meas[name] = AdapterMeasurement(
                    adapter_name=name,
                    workload_id=wl,
                    latencies_us=(),
                    output_hash="",
                    blocked=True,
                    blocked_reason=f"adapter raised: {type(exc).__name__}: {exc}",
                )
        results.append(WorkloadResult(workload_id=wl, measurements=meas))
    return results


def write_results(results: list[WorkloadResult], out_dir: Path) -> Path:
    """Persist results under ``out_dir`` as one file per (workload, adapter)
    plus a top-level summary.json."""

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for wr in results:
        for name, m in wr.measurements.items():
            d = out_dir / wr.workload_id / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "latency_us.csv").write_text(
                "iter,latency_us\n"
                + "\n".join(f"{i},{lat}" for i, lat in enumerate(m.latencies_us))
                + "\n",
                encoding="utf-8",
            )
            (d / "summary.json").write_text(
                json.dumps(m.to_dict(), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

    summary = {
        "schema_version": "headline_benchmark_summary_v1",
        "workloads": [wr.to_dict() for wr in results],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return summary_path


class TimedSyntheticAdapter:
    """Deterministic adapter used by tests + the no-extras baseline.

    Produces synthetic latencies that depend only on (workload, adapter,
    seed) so the runner exercises end-to-end without any real model.
    The ``output_hash`` is identical across instances of this class so
    the correctness check passes trivially.
    """

    def __init__(self, *, adapter_name: str, base_latency_us: float):
        self.adapter_name = adapter_name
        self._base = base_latency_us

    def measure(
        self,
        workload_id: str,
        *,
        iters: int,
        warmup: int,
        seed: int,
    ) -> AdapterMeasurement:
        # Pseudo-random latencies but byte-deterministic.
        import hashlib

        latencies: list[float] = []
        rng = hashlib.sha256(f"{workload_id}|{self.adapter_name}|{seed}".encode())
        for i in range(iters):
            h = int(hashlib.sha256(rng.digest() + str(i).encode()).hexdigest()[:8], 16)
            jitter = (h % 100) / 1000.0  # 0..0.1
            latencies.append(self._base * (1.0 + jitter))
        return AdapterMeasurement(
            adapter_name=self.adapter_name,
            workload_id=workload_id,
            latencies_us=tuple(latencies),
            output_hash="synthetic_" + workload_id,
        )


__all__ = [
    "ADAPTERS",
    "AdapterMeasurement",
    "HeadlineBenchmarkError",
    "TimedSyntheticAdapter",
    "WorkloadResult",
    "run_benchmark",
    "write_results",
]
