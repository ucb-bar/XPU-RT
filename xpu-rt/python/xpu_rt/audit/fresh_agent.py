"""Fresh-agent task pack builder (M-31A.4).

Builds a directory containing exactly the files a fresh Claude Code
session needs to run a CompGen compile task — and nothing more. The
contents are an explicit allowlist; everything else (chat transcripts,
project memory, scratch results, kernel cache) is excluded.

Why this matters: if a *current* Claude session can complete a task
that a *fresh* Claude session cannot, the system is overfit to the
conversation. The task pack is the operator-facing surface that
catches that.

The CI-runnable contract is two-part:

1. ``build_task_pack`` produces a directory whose contents match the
   allowlist exactly (no forbidden files; no missing required files).
2. The greedy/no-LLM baseline (``compgen.audit.fresh_agent_modes``)
   succeeds against the task pack on a holdout model. If the
   deterministic path can't reproduce the workflow with the public
   doc surface, neither can a fresh agent.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from compgen.audit.errors import TaskPackContaminated, TaskPackIncomplete

REPO_ROOT = Path(__file__).resolve().parents[3]


# Required paths the task pack MUST contain. Globs are relative to repo
# root and resolved at build time. Missing required paths raise
# :class:`TaskPackIncomplete`.
REQUIRED_PATHS: tuple[str, ...] = (
    "CLAUDE.md",
    "AGENT.md",
    ".claude/skills/compgen/SKILL.md",
    ".claude/skills/compgen-compile/SKILL.md",
    ".claude/skills/compgen-candidate-selection/SKILL.md",
    "docs/reference/cli.md",
    "docs/reference/mcp-tools.md",
    "docs/realness/m31a_audit_layer.yaml",
    "pyproject.toml",
    "python/compgen/__init__.py",
)

# Path globs to copy into the pack (relative to repo root). Order matters
# only for human readability of the resulting tree.
ALLOWLISTED_PATHS: tuple[str, ...] = (
    "CLAUDE.md",
    "AGENT.md",
    "pyproject.toml",
    "uv.lock",
    "scripts/bootstrap.sh",
    ".claude/skills/compgen/**",
    ".claude/skills/compgen-compile/**",
    ".claude/skills/compgen-candidate-selection/**",
    "docs/reference/**",
    "docs/architecture/**",
    "docs/concepts/**",
    "docs/realness/**",
    "docs/generated/**",
    "configs/**",
    # The compgen package itself; required because MCP tool schemas
    # are introspected from compgen.mcp.tools.ALL_TOOLS at runtime.
    "python/compgen/**",
    # Holdout model adapters live under tests/, but they are referenced
    # by the holdout YAMLs and need to be copyable from the task pack.
    "tests/graph_compilation/models/holdout_*.py",
    "tests/graph_compilation/models/__init__.py",
)

# Path globs that must NEVER appear in the pack. These are the failure
# modes the audit catches: private chat memory, scratch results, hidden
# kernel caches.
FORBIDDEN_PATHS: tuple[str, ...] = (
    ".claude/projects/**",
    ".claude/scheduled_tasks*",
    ".claude/worktrees/**",
    ".claude/settings.local.json",
    "results/**",
    ".compgen_cache/**",
    ".crg-artifacts/**",
    "tmp/**",
    ".compgen/**",
    ".git/**",
    ".venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "third_party/**",
    "user_extensions/**",
    "benchmarks/results/**",
)


@dataclass(frozen=True)
class TaskPack:
    """Metadata about a built task pack."""

    out_dir: Path
    commit: str
    created_at_utc: str
    files_copied: int
    bytes_copied: int
    required_paths_present: tuple[str, ...]
    forbidden_paths_blocked: tuple[str, ...]
    task_prompt_path: Path | None
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "commit": self.commit,
            "created_at_utc": self.created_at_utc,
            "files_copied": self.files_copied,
            "bytes_copied": self.bytes_copied,
            "required_paths_present": list(self.required_paths_present),
            "forbidden_paths_blocked": list(self.forbidden_paths_blocked),
            "task_prompt_path": (
                str(self.task_prompt_path) if self.task_prompt_path else None
            ),
        }


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_forbidden(rel_path: str, forbidden: Iterable[str]) -> bool:
    for pattern in forbidden:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def _matches_any_allowed(rel_path: str, allowed: Iterable[str]) -> bool:
    for pattern in allowed:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        # Handle directory-prefix globs like 'docs/reference/**'
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if rel_path == prefix or rel_path.startswith(prefix + "/"):
                return True
    return False


def _iter_repo_files(repo_root: Path) -> Iterable[Path]:
    """Walk repo files, skipping common heavy dirs early."""
    skip_root_dirs = {
        ".git",
        ".venv",
        "node_modules",
        ".compgen_cache",
        ".crg-artifacts",
        "results",
        "tmp",
        ".compgen",
        "user_extensions",
        "third_party",
    }
    for entry in repo_root.iterdir():
        if entry.name in skip_root_dirs:
            continue
        if entry.is_file():
            yield entry
            continue
        for path in entry.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def build_task_pack(
    *,
    out_dir: Path,
    commit: str,
    repo_root: Path | None = None,
    task_prompt: str | None = None,
    task_model: str = "holdout_mlp_odd_shapes",
    task_target: str = "host_cpu",
    skip_python_package: bool = False,
) -> TaskPack:
    """Copy allowlisted files from ``repo_root`` into ``out_dir``.

    Args:
        out_dir: Empty (or absent) directory to write the pack into. The
            caller owns ``out_dir``; ``build_task_pack`` will populate
            it but won't clean up on failure.
        commit: Git commit short hash that this pack snapshots. Used as
            metadata only; the actual file contents come from the
            current working tree.
        repo_root: Source root (defaults to the resolved package root).
        task_prompt: Override the bundled task prompt. When None, the
            default ``compile_holdout.md`` prompt is used.
        task_model: Holdout model id for the bundled prompt.
        task_target: Target id for the bundled prompt.
        skip_python_package: When True, omit ``python/compgen/**`` from
            the pack. Used by tests that only need to verify the
            allowlist plumbing — copying the entire package is slow.

    Returns:
        :class:`TaskPack` metadata. The pack manifest is written to
        ``<out_dir>/task_pack_manifest.json``.

    Raises:
        :class:`TaskPackIncomplete` if any required path is missing
            after the copy.
        :class:`TaskPackContaminated` if any forbidden path landed in
            the pack (private memory leak).
    """
    repo_root = (repo_root or REPO_ROOT).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed = list(ALLOWLISTED_PATHS)
    if skip_python_package:
        allowed = [p for p in allowed if p != "python/compgen/**"]

    files_copied = 0
    bytes_copied = 0
    for src in _iter_repo_files(repo_root):
        try:
            rel = src.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if _is_forbidden(rel, FORBIDDEN_PATHS):
            continue
        if not _matches_any_allowed(rel, allowed):
            continue
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files_copied += 1
        bytes_copied += dst.stat().st_size

    # Verify required paths present. When skip_python_package is True
    # (test mode), the python/compgen/* requirements are skipped — the
    # task pack is being verified for its allowlist plumbing, not for
    # full executability.
    required_to_check = list(REQUIRED_PATHS)
    if skip_python_package:
        required_to_check = [
            r for r in required_to_check
            if not r.startswith("python/compgen")
        ]
    missing: list[str] = []
    for required in required_to_check:
        if not (out_dir / required).exists():
            missing.append(required)
    if missing:
        raise TaskPackIncomplete(
            f"task pack at {out_dir} missing required paths: {missing}"
        )

    # Verify forbidden paths NOT present.
    contamination: list[str] = []
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(out_dir).as_posix()
        except ValueError:
            continue
        if _is_forbidden(rel, FORBIDDEN_PATHS):
            contamination.append(rel)
    if contamination:
        raise TaskPackContaminated(
            f"task pack at {out_dir} contains forbidden paths: {contamination[:10]}"
        )

    # Write the bundled task prompt.
    prompt_text = task_prompt or _default_task_prompt(
        model=task_model, target=task_target,
    )
    prompt_dst = out_dir / "TASK.md"
    prompt_dst.write_text(prompt_text)

    pack = TaskPack(
        out_dir=out_dir,
        commit=commit,
        created_at_utc=_utc_now(),
        files_copied=files_copied,
        bytes_copied=bytes_copied,
        required_paths_present=tuple(REQUIRED_PATHS),
        forbidden_paths_blocked=FORBIDDEN_PATHS,
        task_prompt_path=prompt_dst,
        manifest_path=out_dir / "task_pack_manifest.json",
    )

    pack.manifest_path.write_text(
        json.dumps(pack.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pack


def _default_task_prompt(*, model: str, target: str) -> str:
    return f"""\
# CompGen fresh-agent task

Compile **{model}** for **{target}** using only the public repo
artifacts in this task pack. The acceptance contract:

1. Use the MCP tools (or the `/compgen-compile` skill if you are
   running in Claude Code) to drive the compile.
2. Do not edit source code.
3. Do not invent candidate IDs, pass IDs, tile sizes, or summary
   ids. Every reference must resolve in the registry.
4. Read the realness contracts under `docs/realness/` to learn
   what each subsystem promises (M-26 through M-34).
5. Browse the available compiler passes via
   `docs/generated/pass_cards/INDEX.md` (60 cards across 12 families).
   Each card declares preconditions, invalidates, preserves_refinement,
   verification rungs, phase, requires_after / excludes contracts.
6. If you propose a multi-pass plan, ensure phase ordering is strict
   (canonicalize → analyze → optimize → verify → emit) and pair
   contracts hold. The validator will refuse a plan that breaks any
   invariant; you'll see typed errors in
   `agent_decision_validation.json`.
7. Reach a verified compile OR produce a typed-blocked outcome from
   `compgen.runtime.errors`. A silent partial pass is failure.

What the pipeline gives you (read these in this order):

- `agent_decision_request.json` — full inline pass cards + analysis
  summaries + bounded candidate list
- `llm_graph_view.json` — bounded view of legal candidates per region
- `cost_preview_v2.json` — predicted relative cost per candidate
- `analysis_summaries` block in the request — every summary's
  content_hash + dependency closure (M-32)

What you write back:

- `agent_decision_response.json` with either
  - `selected_candidate_id` (single-step path), or
  - `pass_plan: [{{pass_id, region_id, candidate_id, rationale}}, ...]`
    for an ordered multi-step plan (M-34.3).

When done, record the outcome in the caveat ledger via
`compgen.audit.fresh_agent_modes.record_manual_session_result(...)`
so the audit can pick it up.
"""


def verify_task_pack(out_dir: Path, *, lenient_python_package: bool = True) -> TaskPack:
    """Load + re-verify a previously built task pack.

    Args:
        out_dir: Path to a previously-built task pack.
        lenient_python_package: When True, missing ``python/compgen/*``
            files are tolerated (a caller may have built the pack with
            ``skip_python_package=True`` for testing). Set False for a
            fully-executable production pack.
    """
    out_dir = Path(out_dir).resolve()
    manifest_path = out_dir / "task_pack_manifest.json"
    if not manifest_path.exists():
        raise TaskPackIncomplete(
            f"task pack at {out_dir}: manifest missing"
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Re-check required paths.
    required_to_check = list(REQUIRED_PATHS)
    if lenient_python_package:
        required_to_check = [
            r for r in required_to_check
            if not r.startswith("python/compgen")
        ]
    missing = [p for p in required_to_check if not (out_dir / p).exists()]
    if missing:
        raise TaskPackIncomplete(
            f"task pack at {out_dir} missing required paths: {missing}"
        )
    # Re-check forbidden paths.
    contamination: list[str] = []
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(out_dir).as_posix()
        if _is_forbidden(rel, FORBIDDEN_PATHS):
            contamination.append(rel)
    if contamination:
        raise TaskPackContaminated(
            f"task pack at {out_dir} contains forbidden paths: {contamination[:10]}"
        )
    return TaskPack(
        out_dir=out_dir,
        commit=str(raw.get("commit", "")),
        created_at_utc=str(raw.get("created_at_utc", "")),
        files_copied=int(raw.get("files_copied", 0)),
        bytes_copied=int(raw.get("bytes_copied", 0)),
        required_paths_present=tuple(raw.get("required_paths_present") or REQUIRED_PATHS),
        forbidden_paths_blocked=tuple(raw.get("forbidden_paths_blocked") or FORBIDDEN_PATHS),
        task_prompt_path=(
            Path(raw["task_prompt_path"])
            if raw.get("task_prompt_path") else None
        ),
        manifest_path=manifest_path,
    )
