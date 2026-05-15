"""H6 — side-effect audit.

Coverage:

1. A synthetic module that mutates ``recipe_module`` outside an
   allowed path is detected.
2. A synthetic module inside ``mcp/tools/`` is allowed even when it
   mutates session state.
3. ``scan_tree`` runs over the production tree without crashing.
4. The closed-enum forbidden attrs are exactly the documented six.
5. ``ALLOWED_PREFIXES`` does NOT contain ``tests/`` (test code is not
   allowed to mutate session state via this audit; the audit scopes
   to ``python/compgen/``).
"""

from __future__ import annotations

from pathlib import Path

from compgen.audit.side_effect_audit import (
    ALLOWED_PREFIXES,
    FORBIDDEN_ATTRS,
    scan_file,
    scan_tree,
)


def _make_repo(root: Path) -> Path:
    """Create a synthetic repo skeleton under ``root``."""

    (root / "python" / "compgen" / "naughty").mkdir(parents=True)
    (root / "python" / "compgen" / "mcp" / "tools").mkdir(parents=True)
    return root


def test_synthetic_violation_detected_outside_allowlist(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    bad = repo / "python" / "compgen" / "naughty" / "mutator.py"
    bad.write_text(
        "def bad(sm):\n"
        "    sm.recipe_module = None\n"
        "    sm.kernel_cache.update({'k': 1})\n"
    )
    violations = scan_file(bad, repo)
    attrs = {v.attr for v in violations}
    assert "recipe_module" in attrs
    assert "kernel_cache" in attrs


def test_mutation_inside_mcp_tools_is_allowed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    ok = repo / "python" / "compgen" / "mcp" / "tools" / "fine.py"
    ok.write_text(
        "def f(sm):\n"
        "    sm.decision_registry.set('k', 'v')\n"
    )
    assert scan_file(ok, repo) == []


def test_scan_tree_runs_without_crashing() -> None:
    # Resolve to the real repo root; the scan should produce a list
    # (possibly empty) of violations.
    repo_root = Path(__file__).resolve().parents[2]
    violations = scan_tree(repo_root)
    # Just assert it returns a list — the *count* is informational.
    assert isinstance(violations, list)


def test_forbidden_attrs_closed_enum() -> None:
    expected = {
        "recipe_module",
        "payload_module",
        "decision_registry",
        "kernel_cache",
        "bench_cache",
        "refinement_cache",
    }
    assert set(FORBIDDEN_ATTRS) == expected


def test_allowed_prefixes_scope_to_compgen() -> None:
    """Allowlist is for production code; tests/ is not in scope."""

    for p in ALLOWED_PREFIXES:
        assert p.startswith("python/compgen/"), p
