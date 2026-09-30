"""Where each profile tree lives, and the names specs and records use for it.

Workload specs select a profile tree by `hardware.profile.gen_root`, and schedules, sidecars and
recorded runs carry that value. The K1 trees live under `gen/`; their original top-level names are
kept here as aliases, so every recorded spec and schedule resolves unchanged.

    resolve_gen_root("gen_mb_shard")        -> "gen/mb_shard"
    resolve_path("gen_mb/vmfb/yolo/...")    -> "gen/mb/vmfb/yolo/..."

A name that is not an alias, including every temporary directory a test builds, is returned as is.
"""
from __future__ import annotations

import os

ALIASES = {
    "gen25": "gen/clk25",
    "gen_mb": "gen/mb",
    "gen_mb_cal": "gen/mb_cal",
    "gen_mb_force": "gen/mb_force",
    "gen_mb_shard": "gen/mb_shard",
    "gen_mb_shard_nav": "gen/mb_shard_nav",
}


def resolve_gen_root(gen_root: str) -> str:
    """The directory a `gen_root` value names, relative to the repository (or as given)."""
    if not gen_root:
        return gen_root
    stripped = gen_root.rstrip("/")
    return ALIASES.get(stripped, gen_root)


def resolve_path(path: str) -> str:
    """Rewrite a path whose first component (after an optional repo prefix) is an alias."""
    if not path:
        return path
    head, sep, rest = path.partition("/")
    if head in ALIASES:
        return ALIASES[head] + (sep + rest if sep else "")
    for alias, target in ALIASES.items():
        marker = os.sep + alias + os.sep
        i = path.find(marker)
        if i >= 0 and os.path.isdir(path[:i]) and not os.path.exists(path[:i] + marker):
            return path[:i] + os.sep + target + os.sep + path[i + len(marker):]
        if path.endswith(os.sep + alias) and not os.path.exists(path):
            return path[: -len(alias)] + target
    return path
