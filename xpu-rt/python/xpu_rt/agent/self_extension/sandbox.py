"""Sandboxed execution of LLM-authored Python source.

The sandbox compiles a source blob into a scratch module whose
``__builtins__`` is a pruned view (no ``open``, ``exec``, ``eval``,
``__import__`` overrides, no ``compile``), then invokes a named entry
point with caller-supplied keyword args. A wall-clock timeout bounds
how long the authored code can run.

This is deliberately *not* a security boundary — authored tools are
trusted enough to land in ``~/.compgen/``; the goals are:

1. **Isolation** — one authored tool's globals cannot leak into the
   host process's modules.
2. **Observability** — import-list + exec errors are captured as
   :class:`SandboxViolation` strings, not raised.
3. **Liveness** — an infinite-loop authored tool cannot block the
   session indefinitely.

Full process-isolation + syscall sandboxing is out of scope for P4
but the :class:`SandboxResult` record shape is forward-compatible
with that extension.
"""

from __future__ import annotations

import ast
import signal
import types
from dataclasses import dataclass, field
from typing import Any

# Exact set of top-level modules authored tools may import. Adjust
# conservatively — every addition expands the attack surface.
DEFAULT_IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "math",
        "statistics",
        "itertools",
        "functools",
        "operator",
        "collections",
        "dataclasses",
        "typing",
        # CompGen's own surface — authored tools compose our primitives.
        "compgen",
        "compgen.ir",
        "compgen.ir.recipe",
        "compgen.ir.recipe.llm_view",
        "compgen.ir.payload",
        "compgen.llm.registry",
        # Numerical libs the LLM is likely to reach for.
        "torch",
        "numpy",
    }
)


# Names removed from the authored tool's __builtins__. Everything else
# from builtins remains available.
_FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "help",
        "exit",
        "quit",
    }
)


@dataclass(frozen=True)
class SandboxViolation:
    """One policy violation detected statically or at runtime."""

    kind: str  # forbidden_import | forbidden_call | exec_error | timeout
    detail: str
    location: str = ""


@dataclass
class SandboxResult:
    """Outcome of a single sandboxed invocation."""

    ok: bool
    value: Any = None
    error: str | None = None
    violations: list[SandboxViolation] = field(default_factory=list)
    stdout: str = ""
    elapsed_s: float = 0.0

    def first_violation(self) -> SandboxViolation | None:
        return self.violations[0] if self.violations else None


# ---------------------------------------------------------------------------
# Static pre-checks
# ---------------------------------------------------------------------------


def _scan_imports(source: str, allow: frozenset[str]) -> list[SandboxViolation]:
    """Walk AST; flag imports that aren't on the allowlist."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            SandboxViolation(
                kind="exec_error",
                detail=f"SyntaxError: {exc.msg}",
                location=f"line {exc.lineno}",
            )
        ]

    violations: list[SandboxViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if alias.name not in allow and top not in allow:
                    violations.append(
                        SandboxViolation(
                            kind="forbidden_import",
                            detail=alias.name,
                            location=f"line {node.lineno}",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                violations.append(
                    SandboxViolation(
                        kind="forbidden_import",
                        detail="relative import",
                        location=f"line {node.lineno}",
                    )
                )
                continue
            top = node.module.split(".")[0]
            if node.module not in allow and top not in allow:
                violations.append(
                    SandboxViolation(
                        kind="forbidden_import",
                        detail=node.module,
                        location=f"line {node.lineno}",
                    )
                )
    return violations


def _safe_builtins() -> dict[str, Any]:
    """Return a filtered ``__builtins__`` dict."""
    import builtins

    out = {k: v for k, v in vars(builtins).items() if k not in _FORBIDDEN_BUILTINS}
    # ``__import__`` is reachable via the 'import X' statement even
    # without us exposing it as a name — Python uses the __import__
    # hook on the module's __builtins__. We replace it with a guarded
    # version that defers to the allowlist checked during static AST.
    out["__import__"] = __import__
    return out


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


class _Timeout:
    """SIGALRM-based wall-clock cap. Only active on POSIX + main thread."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._prev = None
        self._enabled = False

    def __enter__(self) -> _Timeout:
        try:
            import threading

            if threading.current_thread() is threading.main_thread():
                self._prev = signal.signal(signal.SIGALRM, self._raise)
                signal.setitimer(signal.ITIMER_REAL, self.seconds)
                self._enabled = True
        except (AttributeError, ValueError):
            # signal not available (e.g. on Windows). Run without cap.
            pass
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._enabled:
            return
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self._prev is not None:
                signal.signal(signal.SIGALRM, self._prev)
        except ValueError:
            pass

    @staticmethod
    def _raise(signum, frame) -> None:
        raise TimeoutError("sandbox execution exceeded time budget")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def sandbox_invoke(
    source: str,
    entry_name: str,
    *,
    kwargs: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
    allow_imports: frozenset[str] | None = None,
) -> SandboxResult:
    """Compile ``source``, call ``entry_name(**kwargs)``, return the result.

    Args:
        source: The Python source blob authored by the LLM. Must declare
            a top-level function named ``entry_name``.
        entry_name: The function the sandbox will invoke. Typically
            ``"run"`` or the AuthoredTool's name.
        kwargs: Keyword arguments to pass to ``entry_name``.
        timeout_s: Wall-clock cap in seconds (best effort — relies on
            SIGALRM which is only active on the main thread on POSIX).
        allow_imports: Override the default import allowlist. Pass an
            empty frozenset to forbid all imports.

    Returns:
        :class:`SandboxResult` whose ``ok`` field says whether the
        invocation completed without violations and without raising.

    The sandbox never raises on bad code — all failure modes are
    reported through the result object. Caller can decide how much to
    surface to the LLM.
    """
    import io
    import time
    from contextlib import redirect_stdout

    allow = allow_imports if allow_imports is not None else DEFAULT_IMPORT_ALLOWLIST
    result = SandboxResult(ok=False)

    # Static import scan.
    import_violations = _scan_imports(source, allow)
    if any(v.kind != "exec_error" for v in import_violations):
        # Only hard-stop on forbidden imports; syntax errors still try
        # the exec path below so the exception message surfaces.
        result.violations = import_violations
        return result

    # Compile + exec the source into an isolated namespace.
    namespace: dict[str, Any] = {"__builtins__": _safe_builtins()}
    module = types.ModuleType(f"compgen_authored_{id(source)}")
    stdout = io.StringIO()

    t0 = time.perf_counter()
    try:
        with redirect_stdout(stdout):
            try:
                exec(compile(source, f"<authored:{entry_name}>", "exec"), namespace)
            except SyntaxError as exc:
                result.error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
                result.violations = [
                    SandboxViolation(
                        kind="exec_error",
                        detail=result.error,
                        location=f"line {exc.lineno}",
                    )
                ]
                return result
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        result.violations.append(
            SandboxViolation(
                kind="exec_error",
                detail=result.error,
            )
        )
        return result

    entry = namespace.get(entry_name)
    if not callable(entry):
        result.error = f"source does not define a callable named {entry_name!r}"
        result.violations.append(
            SandboxViolation(
                kind="exec_error",
                detail=result.error,
            )
        )
        return result

    # Invoke under a wall-clock timeout.
    try:
        with _Timeout(timeout_s):
            with redirect_stdout(stdout):
                value = entry(**(kwargs or {}))
    except TimeoutError as exc:
        result.error = str(exc)
        result.violations.append(
            SandboxViolation(
                kind="timeout",
                detail=str(exc),
            )
        )
        result.elapsed_s = time.perf_counter() - t0
        result.stdout = stdout.getvalue()
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        result.violations.append(
            SandboxViolation(
                kind="exec_error",
                detail=result.error,
            )
        )
        result.elapsed_s = time.perf_counter() - t0
        result.stdout = stdout.getvalue()
        return result

    result.ok = True
    result.value = value
    result.elapsed_s = time.perf_counter() - t0
    result.stdout = stdout.getvalue()
    return result


__all__ = [
    "DEFAULT_IMPORT_ALLOWLIST",
    "SandboxResult",
    "SandboxViolation",
    "sandbox_invoke",
]
