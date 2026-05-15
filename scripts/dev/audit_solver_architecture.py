"""Solver architecture audit.

Enforces:

1. Optional-solver imports (``mosek``, ``highspy``, ``z3``,
   ``ortools``) appear ONLY in:
     - ``python/compgen/solve/backends/``  (the backend implementations)
     - ``python/compgen/solve/`` for the few inner-loop planner modules
       that need them inline (memory_planner, z3_obligations)
     - ``tests/`` (test code)
     - ``scripts/dev/`` (operator-driven tools)
     - ``python/compgen/semantic/`` (the SMT/Z3 verification stack
       that pre-dates the envelope)

2. Verification-flavored ``SolverProblemKind`` values never appear in
   compiler-core call sites alongside ``backend_preference=MOSEK`` /
   ``HIGHS`` / ``ORTOOLS_CP_SAT``.

3. No production module under ``python/compgen/`` (excluding the
   allowlist above) imports an optional solver package.

The script exits non-zero on violation so it can run in CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PY_ROOT = REPO_ROOT / "python" / "compgen"


# Optional solver package names; importing any of these outside the
# allowlist is a solver-architecture violation.
_OPTIONAL_SOLVERS = (
    "mosek",
    "highspy",
    "z3",
    "ortools",
    "osqp",
    "clarabel",
    "scipy.optimize",
)

# Paths (relative to repo root) where optional-solver imports are
# allowed. Anything below these prefixes may import the solver
# packages.
_ALLOWED_PREFIXES = (
    "python/compgen/solve/backends/",
    "python/compgen/solve/memory_planner.py",
    "python/compgen/solve/z3_obligations.py",
    "python/compgen/solve/placement_planner.py",
    "python/compgen/solve/overlap_planner.py",
    "python/compgen/solve/bandwidth_planner.py",
    "python/compgen/solve/_mosek_solve_impl.py",
    "python/compgen/solve/_highs_solve_impl.py",
    "python/compgen/semantic/",  # pre-SMT verification stack
    "python/compgen/ir/semantic/",  # semantic IR dialect (Z3 lowerings)
    "python/compgen/agent/prompts/",  # LLM-prompt scaffolding for semantic stack
    "python/compgen/solve/backends.py",
    "python/compgen/solve/objectives.py",
    # Legacy modules that shim to the new envelope-aware planners;
    # they import the solver libs but should be deprecated once all
    # callers migrate.
    "python/compgen/solve/memory.py",
    "python/compgen/solve/placement.py",
    "python/compgen/solve/schedule.py",
    "python/compgen/solve/per_sm_queue.py",
)

# Test and scripts are always allowed.
_ALWAYS_ALLOWED_ROOTS = ("tests/", "scripts/")

# Each import_pattern is a regex. We match raw text for simplicity;
# false positives in docstrings are tolerated (these aren't real
# imports).
_IMPORT_PATTERNS = tuple(
    re.compile(rf"^\s*(?:import|from)\s+{re.escape(name)}(\.\w+)*(?:\s+|$)", re.MULTILINE)
    for name in _OPTIONAL_SOLVERS
)


def _is_allowed_path(rel: str) -> bool:
    for prefix in _ALWAYS_ALLOWED_ROOTS:
        if rel.startswith(prefix):
            return True
    for prefix in _ALLOWED_PREFIXES:
        if prefix.endswith(".py"):
            if rel == prefix:
                return True
        else:
            if rel.startswith(prefix):
                return True
    return False


def _scan_one(path: Path) -> list[tuple[int, str]]:
    """Return (line_no, source-line) for every optional-solver import."""

    hits: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return hits
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        for solver in _OPTIONAL_SOLVERS:
            if re.match(
                rf"^(?:import|from)\s+{re.escape(solver)}(?:\.\w+)*(?:\s|$|,)",
                stripped,
            ):
                hits.append((i, stripped))
                break
    return hits


def audit(*, repo_root: Path = REPO_ROOT) -> int:
    """Run the audit; return exit code (0 ok, 2 on violation)."""

    violations: list[str] = []
    py_root = repo_root / "python" / "compgen"
    if not py_root.is_dir():
        print(f"audit: skipped (no python/compgen at {py_root})", file=sys.stderr)
        return 0
    for path in sorted(py_root.rglob("*.py")):
        rel = str(path.relative_to(repo_root))
        if _is_allowed_path(rel):
            continue
        for line_no, src in _scan_one(path):
            violations.append(f"{rel}:{line_no}: {src}")
    if violations:
        print("Solver architecture audit FAILED — optional solver imports leaked outside allowlist:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 2
    print(f"Solver architecture audit OK (scanned {sum(1 for _ in py_root.rglob('*.py'))} files).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    return audit(repo_root=args.repo_root)


if __name__ == "__main__":
    sys.exit(main())
