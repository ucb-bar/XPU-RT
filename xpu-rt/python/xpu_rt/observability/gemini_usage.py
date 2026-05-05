"""Gemini API usage + cost tracker.

Persists every Gemini API call to an append-only JSONL log under
``<repo>/.compgen/gemini_usage/`` and maintains a derived summary
(cumulative + per-month buckets keyed by ``YYYY-MM``).

Design notes:
    * The tracker is **best-effort**: ``record_call`` swallows all errors
      so a tracking failure can never break a compile pipeline. Errors
      are logged via :mod:`structlog` for later inspection.
    * Concurrent writes are serialised with an ``fcntl`` advisory lock on
      a sidecar lockfile so multiple processes (CLI, pipeline, tests)
      can append safely.
    * The pricing table reflects published Google AI Studio rates as of
      2026-05. To override without touching code, drop a YAML file at
      ``configs/gemini_pricing.yaml`` (see :func:`load_pricing_overrides`).
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import errno
import fcntl
import functools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------
# USD per 1M tokens. ``input_long`` / ``output_long`` apply when the prompt
# exceeds ``long_threshold`` tokens. ``cached`` applies to the
# context-cache hit portion of the prompt (Gemini reports
# ``cached_content_token_count`` in usage_metadata).
#
# Rates verified against ai.google.dev/pricing as of 2026-05.

PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "input_long": 2.50,
        "output_long": 15.00,
        "long_threshold": 200_000,
        "cached": 0.31,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
        "cached": 0.075,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
        "cached": 0.025,
    },
    "gemini-1.5-pro": {
        "input": 1.25,
        "output": 5.00,
        "input_long": 2.50,
        "output_long": 10.00,
        "long_threshold": 128_000,
        "cached": 0.3125,
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "input_long": 0.15,
        "output_long": 0.60,
        "long_threshold": 128_000,
        "cached": 0.01875,
    },
    "gemini-1.5-flash-8b": {
        "input": 0.0375,
        "output": 0.15,
        "input_long": 0.075,
        "output_long": 0.30,
        "long_threshold": 128_000,
        "cached": 0.01,
    },
}

# Fallback used when a model id is unknown. We bias toward the
# mid-tier flash rate so unknown models don't silently report $0.
_FALLBACK_RATES = {"input": 0.30, "output": 2.50, "cached": 0.075}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Locate the CompGen repo root by walking up to find ``pyproject.toml``."""
    env_root = os.environ.get("COMPGEN_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "python" / "compgen").exists():
            return parent
    # Fallback: assume four levels up from this file
    # (python/compgen/observability/gemini_usage.py -> repo root)
    return here.parents[3]


def get_storage_dir() -> Path:
    """Return the directory holding usage events, summary, and budget."""
    override = os.environ.get("COMPGEN_GEMINI_USAGE_DIR")
    if override:
        path = Path(override)
    else:
        path = _repo_root() / ".compgen" / "gemini_usage"
    path.mkdir(parents=True, exist_ok=True)
    return path


def events_path() -> Path:
    return get_storage_dir() / "events.jsonl"


def summary_path() -> Path:
    return get_storage_dir() / "summary.json"


def budget_path() -> Path:
    return get_storage_dir() / "budget.json"


def _lock_path() -> Path:
    return get_storage_dir() / ".lock"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UsageEvent:
    """A single recorded API call."""

    timestamp: str  # ISO 8601 UTC
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float
    latency_ms: float
    source: str  # 'gemini_client', 'autocomp', 'manual', ...
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"))

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Budget:
    """Optional spending limits."""

    monthly_usd: float | None = None
    monthly_tokens: int | None = None
    cumulative_usd: float | None = None
    cumulative_tokens: int | None = None

    @classmethod
    def load(cls) -> Budget:
        path = budget_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("budget file unreadable: %s", exc)
            return cls()
        return cls(
            monthly_usd=data.get("monthly_usd"),
            monthly_tokens=data.get("monthly_tokens"),
            cumulative_usd=data.get("cumulative_usd"),
            cumulative_tokens=data.get("cumulative_tokens"),
        )

    def save(self) -> None:
        budget_path().write_text(json.dumps(dataclasses.asdict(self), indent=2))


@dataclass
class MonthBucket:
    month: str  # YYYY-MM
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageSummary:
    """Aggregate snapshot derived from the event log."""

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    by_month: dict[str, MonthBucket] = field(default_factory=dict)
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    first_event_at: str | None = None
    last_event_at: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def current_month(self, now: datetime | None = None) -> MonthBucket:
        key = (now or datetime.now(timezone.utc)).strftime("%Y-%m")
        return self.by_month.get(key, MonthBucket(month=key))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "first_event_at": self.first_event_at,
            "last_event_at": self.last_event_at,
            "by_month": {k: dataclasses.asdict(v) for k, v in sorted(self.by_month.items())},
            "by_model": self.by_model,
        }


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def _normalize_model(model: str) -> str:
    """Map a versioned model id ('gemini-2.5-flash-001') to a pricing key."""
    if not model:
        return ""
    m = model.lower().strip()
    # Strip provider prefix if present.
    m = m.removeprefix("models/")
    # Try exact then strip trailing -NNN / preview suffix.
    if m in PRICING:
        return m
    parts = m.split("-")
    while parts:
        candidate = "-".join(parts)
        if candidate in PRICING:
            return candidate
        parts.pop()
    return ""


def load_pricing_overrides() -> dict[str, dict[str, float]]:
    """Load optional ``configs/gemini_pricing.yaml`` overrides."""
    path = _repo_root() / "configs" / "gemini_pricing.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("PyYAML not installed; skipping pricing overrides")
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("pricing override file unreadable: %s", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def compute_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Compute USD cost for a single call.

    Cached tokens are billed at the cached rate and are assumed to be a
    subset of ``prompt_tokens`` (consistent with Gemini usage_metadata).
    """
    table = dict(PRICING)
    table.update(load_pricing_overrides())
    key = _normalize_model(model)
    rates = table.get(key, _FALLBACK_RATES)

    threshold = rates.get("long_threshold")
    if threshold is not None and prompt_tokens > threshold:
        in_rate = rates.get("input_long", rates["input"])
        out_rate = rates.get("output_long", rates["output"])
    else:
        in_rate = rates["input"]
        out_rate = rates["output"]
    cached_rate = rates.get("cached", in_rate * 0.25)

    billable_prompt = max(prompt_tokens - cached_tokens, 0)
    cost = (
        billable_prompt * in_rate / 1_000_000
        + cached_tokens * cached_rate / 1_000_000
        + completion_tokens * out_rate / 1_000_000
    )
    return round(cost, 8)


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _exclusive_lock() -> Iterator[None]:
    """Cross-process advisory lock guarding writes to events.jsonl."""
    lock_file = _lock_path()
    lock_file.touch(exist_ok=True)
    fd = os.open(lock_file, os.O_RDWR)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                break
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                raise
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------
# Recording + reading
# ---------------------------------------------------------------------------


def record_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached_tokens: int = 0,
    latency_ms: float = 0.0,
    source: str = "gemini_client",
    metadata: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> UsageEvent | None:
    """Record one API call. Returns the event, or ``None`` on failure.

    This function never raises — a tracking failure must not break a
    pipeline. Errors are logged.
    """
    try:
        ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        cost = compute_cost_usd(model, prompt_tokens, completion_tokens, cached_tokens)
        event = UsageEvent(
            timestamp=ts,
            model=model,
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            cached_tokens=int(cached_tokens or 0),
            cost_usd=cost,
            latency_ms=float(latency_ms or 0.0),
            source=source,
            metadata=dict(metadata or {}),
        )
        with _exclusive_lock():
            with events_path().open("a", encoding="utf-8") as f:
                f.write(event.to_json_line() + "\n")
            _refresh_summary_locked()
        return event
    except Exception:  # noqa: BLE001 — tracker must never raise
        logger.exception("gemini_usage.record_call failed")
        return None


def iter_events() -> Iterator[UsageEvent]:
    """Yield events in append order. Skips malformed lines."""
    path = events_path()
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield UsageEvent(**data)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("skipping malformed event line: %s", exc)


def build_summary() -> UsageSummary:
    """Recompute the summary from the event log."""
    summary = UsageSummary()
    for event in iter_events():
        summary.total_calls += 1
        summary.total_prompt_tokens += event.prompt_tokens
        summary.total_completion_tokens += event.completion_tokens
        summary.total_cached_tokens += event.cached_tokens
        summary.total_cost_usd += event.cost_usd

        month_key = event.timestamp[:7]
        bucket = summary.by_month.setdefault(month_key, MonthBucket(month=month_key))
        bucket.calls += 1
        bucket.prompt_tokens += event.prompt_tokens
        bucket.completion_tokens += event.completion_tokens
        bucket.cached_tokens += event.cached_tokens
        bucket.cost_usd += event.cost_usd

        model_bucket = summary.by_model.setdefault(
            event.model,
            {"calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "cost_usd": 0.0},
        )
        model_bucket["calls"] += 1
        model_bucket["prompt_tokens"] += event.prompt_tokens
        model_bucket["completion_tokens"] += event.completion_tokens
        model_bucket["cost_usd"] += event.cost_usd

        if summary.first_event_at is None:
            summary.first_event_at = event.timestamp
        summary.last_event_at = event.timestamp

    return summary


def _refresh_summary_locked() -> None:
    """Refresh ``summary.json`` from current events. Caller holds the lock."""
    summary = build_summary()
    summary_path().write_text(json.dumps(summary.to_dict(), indent=2))


def load_summary() -> UsageSummary:
    """Load the cached summary (recomputing if absent or stale)."""
    path = summary_path()
    events = events_path()
    if not path.exists() or (events.exists() and events.stat().st_mtime > path.stat().st_mtime):
        with _exclusive_lock():
            _refresh_summary_locked()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return build_summary()
    summary = UsageSummary(
        total_calls=data.get("total_calls", 0),
        total_prompt_tokens=data.get("total_prompt_tokens", 0),
        total_completion_tokens=data.get("total_completion_tokens", 0),
        total_cached_tokens=data.get("total_cached_tokens", 0),
        total_cost_usd=data.get("total_cost_usd", 0.0),
        first_event_at=data.get("first_event_at"),
        last_event_at=data.get("last_event_at"),
        by_model=data.get("by_model", {}),
    )
    summary.by_month = {k: MonthBucket(**v) for k, v in data.get("by_month", {}).items()}
    return summary


# ---------------------------------------------------------------------------
# Budget checks
# ---------------------------------------------------------------------------


@dataclass
class BudgetStatus:
    """Result of evaluating the current usage against a configured budget."""

    monthly_usd_pct: float | None
    monthly_tokens_pct: float | None
    cumulative_usd_pct: float | None
    cumulative_tokens_pct: float | None
    warnings: list[str] = field(default_factory=list)
    exceeded: list[str] = field(default_factory=list)


def evaluate_budget(summary: UsageSummary | None = None, budget: Budget | None = None) -> BudgetStatus:
    """Compare current usage against the configured budget."""
    summary = summary or load_summary()
    budget = budget or Budget.load()
    month = summary.current_month()

    def pct(used: float, limit: float | None) -> float | None:
        if limit is None or limit <= 0:
            return None
        return round(100.0 * used / limit, 2)

    status = BudgetStatus(
        monthly_usd_pct=pct(month.cost_usd, budget.monthly_usd),
        monthly_tokens_pct=pct(month.total_tokens, budget.monthly_tokens),
        cumulative_usd_pct=pct(summary.total_cost_usd, budget.cumulative_usd),
        cumulative_tokens_pct=pct(summary.total_tokens, budget.cumulative_tokens),
    )

    def check(name: str, pct_value: float | None) -> None:
        if pct_value is None:
            return
        if pct_value >= 100:
            status.exceeded.append(f"{name}: {pct_value:.1f}% of budget")
        elif pct_value >= 80:
            status.warnings.append(f"{name}: {pct_value:.1f}% of budget")

    check("monthly USD", status.monthly_usd_pct)
    check("monthly tokens", status.monthly_tokens_pct)
    check("cumulative USD", status.cumulative_usd_pct)
    check("cumulative tokens", status.cumulative_tokens_pct)
    return status


# ---------------------------------------------------------------------------
# Convenience for instrumentation hooks
# ---------------------------------------------------------------------------


def record_from_response(
    model: str,
    response: Any,
    *,
    started_at: float | None = None,
    source: str = "gemini_client",
    metadata: dict[str, Any] | None = None,
) -> UsageEvent | None:
    """Record a call given a google-genai response object.

    Tolerant to missing fields / older SDK versions.
    """
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    cached_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)
    latency_ms = (time.perf_counter() - started_at) * 1000.0 if started_at is not None else 0.0
    return record_call(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        latency_ms=latency_ms,
        source=source,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# SDK-level instrumentation
# ---------------------------------------------------------------------------
# A monkey-patch on google.genai's Models.generate_content (sync) and
# AsyncModels.generate_content (async) so every Gemini call routed through
# the SDK is recorded — regardless of whether the caller is our own
# GeminiClient, autocomp's LLMClient, or any other downstream consumer.
#
# Source attribution flows via a ContextVar so callers (autocomp adapter,
# GeminiClient.generate, etc.) can tag their calls without changing the
# patched function's signature.

_current_source: contextvars.ContextVar[str] = contextvars.ContextVar(
    "compgen_gemini_usage_source", default="genai_sdk"
)
_current_metadata: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "compgen_gemini_usage_metadata", default={}
)

_INSTRUMENTED_FLAG = "_compgen_usage_instrumented"


@contextlib.contextmanager
def tracking_source(source: str, **metadata: Any) -> Iterator[None]:
    """Tag any Gemini SDK calls made within this block with ``source``.

    Stacks via ContextVar so nested blocks restore the prior value. Safe
    to use across asyncio tasks (each task gets its own context copy).
    """
    src_token = _current_source.set(source)
    meta_token = _current_metadata.set(dict(metadata)) if metadata else None
    try:
        yield
    finally:
        _current_source.reset(src_token)
        if meta_token is not None:
            _current_metadata.reset(meta_token)


def _resolve_model_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Pull the ``model`` arg out of a generate_content call."""
    if "model" in kwargs:
        return str(kwargs["model"])
    # Position 0 is `self`. The first positional after that is `model`.
    if len(args) >= 2:
        return str(args[1])
    return ""


def _record_sdk_call(model: str, response: Any, started_at: float) -> None:
    record_from_response(
        model=model,
        response=response,
        started_at=started_at,
        source=_current_source.get(),
        metadata=dict(_current_metadata.get()),
    )


def install_genai_instrumentation() -> bool:
    """Monkey-patch ``google.genai`` to record every API call.

    Idempotent: subsequent calls are no-ops. Returns True if the SDK was
    found and patched (or already patched), False if google-genai is not
    importable.
    """
    try:
        from google.genai import models as genai_models  # type: ignore[import-not-found]
    except ImportError:
        return False

    sync_cls = getattr(genai_models, "Models", None)
    async_cls = getattr(genai_models, "AsyncModels", None)

    patched = False
    if sync_cls is not None and not getattr(sync_cls, _INSTRUMENTED_FLAG, False):
        original = sync_cls.generate_content

        @functools.wraps(original)
        def patched_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            response = original(self, *args, **kwargs)
            try:
                model = _resolve_model_arg((self, *args), kwargs)
                _record_sdk_call(model, response, t0)
            except Exception:  # noqa: BLE001
                logger.exception("usage tracking failed for sync generate_content")
            return response

        sync_cls.generate_content = patched_sync  # type: ignore[method-assign]
        setattr(sync_cls, _INSTRUMENTED_FLAG, True)
        patched = True

    if async_cls is not None and not getattr(async_cls, _INSTRUMENTED_FLAG, False):
        original_async = async_cls.generate_content

        @functools.wraps(original_async)
        async def patched_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            response = await original_async(self, *args, **kwargs)
            try:
                model = _resolve_model_arg((self, *args), kwargs)
                _record_sdk_call(model, response, t0)
            except Exception:  # noqa: BLE001
                logger.exception("usage tracking failed for async generate_content")
            return response

        async_cls.generate_content = patched_async  # type: ignore[method-assign]
        setattr(async_cls, _INSTRUMENTED_FLAG, True)
        patched = True

    if patched:
        logger.debug("compgen.observability: instrumented google.genai")
    return True


def is_genai_instrumented() -> bool:
    """Whether the google.genai SDK has been patched in this process."""
    try:
        from google.genai import models as genai_models  # type: ignore[import-not-found]
    except ImportError:
        return False
    sync_ok = getattr(getattr(genai_models, "Models", None), _INSTRUMENTED_FLAG, False)
    async_ok = getattr(getattr(genai_models, "AsyncModels", None), _INSTRUMENTED_FLAG, False)
    return bool(sync_ok and async_ok)
