#!/usr/bin/env python
"""Build a fresh-agent task pack.

Usage::

    uv run python scripts/dev/fresh_agent_task_pack.py \\
        --out /tmp/compgen_task_pack \\
        [--task-model holdout_mlp_odd_shapes] \\
        [--task-target host_cpu]

Exit 0 iff the pack is built, allowlist-clean, and the manifest
verifies.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from compgen.audit.errors import TaskPackContaminated, TaskPackIncomplete
from compgen.audit.fresh_agent import build_task_pack


def _git_short_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a fresh-agent task pack")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (will be cleaned if --force is given)")
    p.add_argument("--force", action="store_true",
                   help="Remove --out before building")
    p.add_argument("--task-model", default="holdout_mlp_odd_shapes")
    p.add_argument("--task-target", default="host_cpu")
    p.add_argument("--skip-python-package", action="store_true",
                   help="Don't copy python/compgen/** (faster; pack won't run end-to-end)")
    p.add_argument("--repo-root", type=Path, default=None)
    args = p.parse_args(argv)

    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    if args.out.exists():
        if args.force:
            shutil.rmtree(args.out)
        else:
            print(f"FAIL: {args.out} exists; pass --force to overwrite",
                  file=sys.stderr)
            return 2
    args.out.mkdir(parents=True)

    commit = _git_short_commit(repo_root)
    try:
        pack = build_task_pack(
            out_dir=args.out,
            commit=commit,
            repo_root=repo_root,
            task_model=args.task_model,
            task_target=args.task_target,
            skip_python_package=args.skip_python_package,
        )
    except TaskPackIncomplete as exc:
        print(f"FAIL: task pack incomplete: {exc}", file=sys.stderr)
        return 1
    except TaskPackContaminated as exc:
        print(f"FAIL: task pack contaminated: {exc}", file=sys.stderr)
        return 1

    print(
        f"task pack built at {pack.out_dir}: "
        f"{pack.files_copied} files, "
        f"{pack.bytes_copied / (1024 * 1024):.1f} MB"
    )
    print(f"  manifest: {pack.manifest_path}")
    print(f"  task prompt: {pack.task_prompt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
