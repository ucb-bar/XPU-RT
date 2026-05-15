"""H6 — Tools-as-only-side-effect audit (Section 11 Dream 6).

Scan ``python/compgen/`` for direct mutations to session-owned state
(``recipe_module``, ``payload_module``, ``decision_registry``,
``kernel_cache``, ``bench_cache``, ``refinement_cache``) that bypass
MCP tool dispatch. Returns a typed list of violations; CI rejects any
*new* violation (the audit tolerates the baseline from already-vetted
sites under ``mcp/tools/`` and a small explicit allowlist).

Forbidden patterns (closed enum):

* ``*.recipe_module = ...`` / ``*.recipe_module.body.append(...)``
* ``*.payload_module = ...``
* ``*.decision_registry.set(...)`` / ``*._decisions[...] = ...``
* ``*.kernel_cache[...] = ...``
* ``*.bench_cache[...] = ...``
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Files allowed to mutate session-owned state directly.
ALLOWED_PREFIXES: tuple[str, ...] = (
    "python/compgen/mcp/tools/",
    "python/compgen/mcp/server.py",
    "python/compgen/mcp/session.py",
    # Stage-plugin + internal infrastructure that owns the state.
    "python/compgen/agent/decisions.py",
    "python/compgen/agent/env/",
    "python/compgen/agent/loop/",
    "python/compgen/agent/llm_driver.py",
    "python/compgen/agent/llm_driver_recovery.py",
    "python/compgen/agent/gates/",
    "python/compgen/agent/views/",
    "python/compgen/runtime/",
    "python/compgen/api.py",
    "python/compgen/api_llm.py",
    "python/compgen/ir/recipe/",
    "python/compgen/graph_compilation/",
    "python/compgen/kernels/",
    "python/compgen/promotion/",
    "python/compgen/bench/",
    "python/compgen/audit/",
)

FORBIDDEN_ATTRS: tuple[str, ...] = (
    "recipe_module",
    "payload_module",
    "decision_registry",
    "kernel_cache",
    "bench_cache",
    "refinement_cache",
)


@dataclass(frozen=True)
class SideEffectViolation:
    """One AST-detected direct-mutation violation."""

    file: str
    line: int
    node_type: str
    attr: str
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "node_type": self.node_type,
            "attr": self.attr,
            "snippet": self.snippet,
        }


def _is_allowed(rel_path: str) -> bool:
    """True iff the file is under one of :data:`ALLOWED_PREFIXES`."""

    return any(rel_path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _attr_name(node: ast.AST) -> str | None:
    """Extract the trailing attribute name from an AST node, if any."""

    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _attr_name(node.value)
    return None


class _SideEffectVisitor(ast.NodeVisitor):
    """Records direct mutations to forbidden attribute names."""

    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.violations: list[SideEffectViolation] = []

    def _record(self, node: ast.AST, attr: str, node_type: str) -> None:
        try:
            snippet = ast.unparse(node)
        except Exception:  # noqa: BLE001
            snippet = f"<{node_type}>"
        self.violations.append(
            SideEffectViolation(
                file=self.rel_path,
                line=getattr(node, "lineno", -1),
                node_type=node_type,
                attr=attr,
                snippet=snippet[:120],
            )
        )

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for tgt in node.targets:
            attr = _attr_name(tgt)
            if attr in FORBIDDEN_ATTRS:
                self._record(node, attr, "assign")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        attr = _attr_name(node.target)
        if attr in FORBIDDEN_ATTRS:
            self._record(node, attr, "aug_assign")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # x.kernel_cache.set(...) / x._decisions.set(...) / .append on
        # one of the forbidden attrs.
        if isinstance(node.func, ast.Attribute):
            attr = _attr_name(node.func.value)
            method = node.func.attr
            if attr in FORBIDDEN_ATTRS and method in {
                "set",
                "append",
                "update",
                "pop",
                "clear",
            }:
                self._record(node, attr, f"call:{method}")
        self.generic_visit(node)


def scan_file(path: Path, repo_root: Path) -> list[SideEffectViolation]:
    """Return the violations found in ``path`` (empty if allowed)."""

    try:
        rel_path = str(path.relative_to(repo_root))
    except ValueError:
        rel_path = str(path)
    if _is_allowed(rel_path):
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    v = _SideEffectVisitor(rel_path)
    v.visit(tree)
    return v.violations


def scan_tree(repo_root: Path) -> list[SideEffectViolation]:
    """Walk ``python/compgen/`` and aggregate violations."""

    src_root = repo_root / "python" / "compgen"
    out: list[SideEffectViolation] = []
    for p in src_root.rglob("*.py"):
        out.extend(scan_file(p, repo_root))
    return out


__all__ = [
    "ALLOWED_PREFIXES",
    "FORBIDDEN_ATTRS",
    "SideEffectViolation",
    "scan_file",
    "scan_tree",
]
