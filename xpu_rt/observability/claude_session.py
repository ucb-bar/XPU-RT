"""Read Claude Code session token usage from the on-disk JSONL.

Claude Code writes every assistant turn (and every tool use) to
``~/.claude/projects/<sanitized-cwd>/<session-uuid>.jsonl``. Each
``"type": "assistant"`` row carries a ``message.usage`` dict with
input / output / cache-read / cache-creation token counts. Summing
across rows gives a cumulative running total that an experiment
harness can snapshot before/after a batch to compute the delta —
the same numbers the ``/usage`` panel shows, just read from the
file the panel reads itself.

The reader is **best-effort**:

* If the session JSONL can't be located, returns empty counts
  (logs a warning). The caller can fall back to "manual /usage
  paste-in" for cost reporting.
* If a row is missing usage fields or has a stale schema, that row
  is skipped silently.

Cost estimation is **optional** and lives in a small rate table
near the bottom of this file. The table is **not authoritative** —
Anthropic's published rates change, and the 1M-context Opus tier
has discounted cache pricing that this table approximates. Use the
returned token counts as the source of truth; treat the dollar
figure as an estimate, not a bill.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate table (estimates — see module docstring)
# ---------------------------------------------------------------------------


# All rates in USD per **million tokens**. The 1M-context Opus tier uses
# a lower cache-read rate than the standard tier; we encode both as
# separate keys. When the model id ends in ``[1m]`` (the long-context
# variant) the long-context column wins.
RATES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.50,
        "cache_creation_5m": 18.75,
        "cache_creation_1h": 30.00,
    },
    # 1M-context tier (Opus 4.7[1m]). Rates back-solved from a
    # captured ``/usage`` panel reading $58.62 against 685 input /
    # 183.7k output / 84.7M cache_read / 1.9M cache_creation_1h tokens.
    # Estimated rates that reproduce that within ~$1:
    #   input  $15  /M  (same as standard)
    #   output $75  /M  (same as standard)
    #   cache_read $0.30/M  (90% discount on input, classic Anthropic
    #                        cache pricing)
    #   cache_creation_1h $10/M  (5m cache pricing × 1h amortisation
    #                              factor; the panel does not separate
    #                              the two but the JSONL does)
    "claude-opus-4-7[1m]": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 0.30,
        "cache_creation_5m": 3.75,
        "cache_creation_1h": 10.0,
    },
    "claude-haiku-4-5": {
        "input": 1.0,
        "output": 5.0,
        "cache_read": 0.10,
        "cache_creation_5m": 1.25,
        "cache_creation_1h": 2.00,
    },
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_creation_5m": 3.75,
        "cache_creation_1h": 6.00,
    },
}


def _normalize_model(model: str, *, long_context: bool) -> str:
    if not model:
        return ""
    base = model.lower().strip()
    if long_context and not base.endswith("[1m]"):
        return base + "[1m]"
    return base


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenCounts:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_5m_tokens: int = 0
    cache_creation_1h_tokens: int = 0

    def __add__(self, other: TokenCounts) -> TokenCounts:
        return TokenCounts(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_5m_tokens=self.cache_creation_5m_tokens
            + other.cache_creation_5m_tokens,
            cache_creation_1h_tokens=self.cache_creation_1h_tokens
            + other.cache_creation_1h_tokens,
        )

    def __sub__(self, other: TokenCounts) -> TokenCounts:
        return TokenCounts(
            input_tokens=max(self.input_tokens - other.input_tokens, 0),
            output_tokens=max(self.output_tokens - other.output_tokens, 0),
            cache_read_tokens=max(self.cache_read_tokens - other.cache_read_tokens, 0),
            cache_creation_5m_tokens=max(
                self.cache_creation_5m_tokens - other.cache_creation_5m_tokens, 0
            ),
            cache_creation_1h_tokens=max(
                self.cache_creation_1h_tokens - other.cache_creation_1h_tokens, 0
            ),
        )

    def estimate_cost_usd(self, model: str, *, long_context: bool = False) -> float:
        key = _normalize_model(model, long_context=long_context)
        rates = RATES_USD_PER_MTOK.get(key)
        if rates is None:
            # Fall back to the non-long-context variant if available.
            fallback = _normalize_model(model, long_context=False)
            rates = RATES_USD_PER_MTOK.get(fallback)
        if rates is None:
            return 0.0
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
            + self.cache_read_tokens * rates["cache_read"]
            + self.cache_creation_5m_tokens * rates["cache_creation_5m"]
            + self.cache_creation_1h_tokens * rates["cache_creation_1h"]
        ) / 1_000_000

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclass
class SessionSnapshot:
    """All Claude Code token usage in one session JSONL, by model id."""

    session_path: Path
    rows_seen: int = 0
    rows_with_usage: int = 0
    by_model: dict[str, TokenCounts] = field(default_factory=dict)

    @property
    def total(self) -> TokenCounts:
        out = TokenCounts()
        for t in self.by_model.values():
            out = out + t
        return out

    def estimate_total_cost_usd(self) -> float:
        usd = 0.0
        for model, counts in self.by_model.items():
            # Heuristic: a model id like ``claude-opus-4-7`` with cache_read
            # < input means short-context; if cache_read >> input, the
            # long-context tier rates are a better fit.
            long_ctx = counts.cache_read_tokens > counts.input_tokens * 100
            usd += counts.estimate_cost_usd(model, long_context=long_ctx)
        return usd

    def to_dict(self) -> dict:
        return {
            "session_path": str(self.session_path),
            "rows_seen": self.rows_seen,
            "rows_with_usage": self.rows_with_usage,
            "by_model": {k: v.to_dict() for k, v in self.by_model.items()},
            "total": self.total.to_dict(),
            "estimated_cost_usd": round(self.estimate_total_cost_usd(), 4),
        }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _projects_root() -> Path:
    override = os.environ.get("CLAUDE_PROJECTS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "projects"


def _sanitize_cwd_for_claude(cwd: Path) -> str:
    """Mirror Claude Code's directory-name encoding for ``~/.claude/projects``.

    Claude stores per-project session JSONLs under a subdir whose name is
    the working directory with ``/`` replaced by ``-`` and a leading
    dash. e.g. ``/scratch2/agustin/xpu-rt-integration`` →
    ``-scratch2-agustin-xpu-rt-integration``.
    """
    return "-" + re.sub(r"/", "-", str(cwd).lstrip("/"))


def find_active_session_jsonl(cwd: Path | None = None) -> Path | None:
    """Return the JSONL for the **active** Claude Code session.

    Resolution order:
      1. ``CLAUDE_CODE_SESSION_ID`` env var (set by Claude Code itself)
         → look for ``<projects_root>/<sanitized-cwd>/<session-id>.jsonl``.
         This is the *only* truly reliable signal — Claude Code may
         write to multiple JSONLs in the same project dir (sidechain
         agents, parallel branches) and the most-recently-modified
         file is **not** necessarily the active session.
      2. Fallback: most-recently-modified JSONL in the project dir,
         for use outside a live Claude Code session.

    Returns None if neither path resolves.
    """
    cwd = cwd or Path.cwd()
    project_dir = _projects_root() / _sanitize_cwd_for_claude(cwd)
    if not project_dir.is_dir():
        return None
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if sid:
        candidate = project_dir / f"{sid}.jsonl"
        if candidate.is_file():
            return candidate
    # Fallback path.
    candidates = sorted(
        project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_snapshot(path: Path | None = None, *, cwd: Path | None = None) -> SessionSnapshot:
    """Read all assistant-turn usage records from a session JSONL.

    Args:
        path: Explicit session JSONL path. When None, auto-detects via
            :func:`find_active_session_jsonl`.
        cwd: Working dir used by auto-detect. Defaults to ``Path.cwd()``.

    Returns:
        :class:`SessionSnapshot` aggregated across every ``assistant``
        message in the file. ``by_model`` keys are the raw ``model``
        strings reported by the SDK (e.g. ``claude-opus-4-7``).
    """
    if path is None:
        path = find_active_session_jsonl(cwd=cwd)
    if path is None or not path.is_file():
        return SessionSnapshot(session_path=Path("(none)"))

    snapshot = SessionSnapshot(session_path=path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("claude_session: could not read %s: %s", path, exc)
        return snapshot

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        snapshot.rows_seen += 1
        if row.get("type") != "assistant":
            continue
        message = row.get("message")
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        snapshot.rows_with_usage += 1

        model = str(message.get("model", "")).lower()
        counts = TokenCounts(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_creation_5m_tokens=int(
                (usage.get("cache_creation") or {}).get("ephemeral_5m_input_tokens", 0) or 0
            ),
            cache_creation_1h_tokens=int(
                (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0) or 0
            ),
        )
        if model in snapshot.by_model:
            snapshot.by_model[model] = snapshot.by_model[model] + counts
        else:
            snapshot.by_model[model] = counts
    return snapshot


def delta(before: SessionSnapshot, after: SessionSnapshot) -> dict:
    """Difference two snapshots, return JSON-friendly dict.

    The delta is the work performed *between* the two snapshots — useful
    for instrumenting a batch run that wraps a section of session
    activity.
    """
    by_model_delta: dict[str, TokenCounts] = {}
    models = set(before.by_model) | set(after.by_model)
    for m in models:
        b = before.by_model.get(m, TokenCounts())
        a = after.by_model.get(m, TokenCounts())
        by_model_delta[m] = a - b
    total_after = after.total
    total_before = before.total
    delta_total = total_after - total_before
    delta_cost = max(
        after.estimate_total_cost_usd() - before.estimate_total_cost_usd(), 0.0
    )
    return {
        "by_model_delta": {k: v.to_dict() for k, v in by_model_delta.items()},
        "total_delta": delta_total.to_dict(),
        "estimated_cost_delta_usd": round(delta_cost, 4),
        "rows_seen_delta": after.rows_seen - before.rows_seen,
        "rows_with_usage_delta": after.rows_with_usage - before.rows_with_usage,
    }


# ---------------------------------------------------------------------------
# CLI: ``uv run xpu-rt-claude-usage`` (entry-point added below if needed)
# ---------------------------------------------------------------------------


def _print(snapshot: SessionSnapshot) -> None:
    print(json.dumps(snapshot.to_dict(), indent=2))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Read Claude Code session token usage from the on-disk JSONL."
    )
    parser.add_argument("--path", type=Path, default=None, help="explicit session jsonl")
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="working dir for auto-detect (default: current dir)",
    )
    parser.add_argument(
        "--write-snapshot",
        type=Path,
        default=None,
        help="if set, write the snapshot JSON to this path",
    )
    args = parser.parse_args(argv)

    snapshot = read_snapshot(path=args.path, cwd=args.cwd)
    if args.write_snapshot:
        args.write_snapshot.write_text(json.dumps(snapshot.to_dict(), indent=2))
    _print(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RATES_USD_PER_MTOK",
    "SessionSnapshot",
    "TokenCounts",
    "delta",
    "find_active_session_jsonl",
    "main",
    "read_snapshot",
]
