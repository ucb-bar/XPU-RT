"""Minimal job-id queue for tools whose wall-clock exceeds MCP's
stdio comfort zone.

MCP stdio works best for sub-5-second tool calls. CompGen has a
few paths (first Triton JIT for a new dim) that routinely exceed 30s.
Rather than block the pipe, those paths launch a ``Job`` whose
``job_id`` we hand back to the LLM; the LLM then polls via
``poll_job(job_id)`` until the job reports ``done``.

The implementation is intentionally tiny — one thread pool, an in-
memory dict, no persistence. Jobs live only as long as the server
process. If the job runs in < ``inline_threshold_s`` we inline the
result instead of returning a job-id, which is the common case for
everything but kernel JIT.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    """One outstanding asynchronous tool call."""

    job_id: str
    name: str
    status: str = "running"          # running | done | failed
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    _future: Future | None = None

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


class JobQueue:
    """Thread-pool-backed job queue.

    Thread-safe. The pool is sized by ``max_workers`` (default 2 —
    a single CompGen session rarely needs more concurrent heavy
    work than one Triton JIT + one gate run).
    """

    def __init__(
        self, *, max_workers: int = 2, inline_threshold_s: float = 5.0,
    ) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._inline_threshold_s = inline_threshold_s
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self, name: str, fn: Callable[[], dict[str, Any]],
    ) -> Job:
        """Submit ``fn`` for async execution. Returns the job record.

        The caller typically checks ``job.status`` and optionally calls
        :meth:`run_inline_or_async` to project small jobs to synchronous.
        """
        job = Job(job_id=f"job_{uuid.uuid4().hex[:10]}", name=name)
        future = self._pool.submit(self._runner, job, fn)
        job._future = future
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def run_inline_or_async(
        self, name: str, fn: Callable[[], dict[str, Any]],
        *, inline_threshold_s: float | None = None,
    ) -> dict[str, Any]:
        """Project to sync when fast; hand back a job-id when slow.

        This is the function MCP tool handlers call. A result dict is
        always returned; ``{"async": true, "job_id": "..."}`` means the
        LLM should poll, anything else is the final result.
        """
        threshold = (
            inline_threshold_s if inline_threshold_s is not None
            else self._inline_threshold_s
        )
        job = self.submit(name, fn)
        try:
            assert job._future is not None
            result = job._future.result(timeout=threshold)
            # finished in time — inline.
            return result
        except Exception as exc:   # includes TimeoutError from .result()
            # Timeout: hand back job-id. Real exceptions within fn are
            # already captured in job.status via _runner.
            if "timeout" in type(exc).__name__.lower():
                return {
                    "async": True,
                    "job_id": job.job_id,
                    "status": "running",
                    "started_at": job.started_at,
                }
            # Some other exception — fn raised directly and _runner
            # captured it; surface the failure synchronously.
            return {
                "async": False,
                "job_id": job.job_id,
                "status": job.status,
                "error": job.error or f"{type(exc).__name__}: {exc}",
            }

    def poll(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return {"status": "unknown", "job_id": job_id}
        if job.status in {"done", "failed"}:
            return {
                "status": job.status,
                "job_id": job_id,
                "result": job.result,
                "error": job.error,
                "elapsed_s": job.elapsed_s,
            }
        return {"status": "running", "job_id": job_id, "elapsed_s": job.elapsed_s}

    def _runner(self, job: Job, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            out = fn()
            job.result = out
            job.status = "done"
        except Exception as e:   # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
        finally:
            job.finished_at = time.time()
        return job.result or {}


__all__ = ["Job", "JobQueue"]
