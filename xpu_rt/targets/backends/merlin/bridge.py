"""``MerlinBridge`` — Python-API-first invocation of merlin's tools.

Merlin's compile / profile / chipyard drivers all live as argparse
scripts under ``$MERLIN_ROOT/tools/<name>.py``. Each script exposes
``setup_parser(argparse.ArgumentParser)`` and ``main(args) -> int``.

Resolution order for ``MerlinBridge.call(tool, ...)``:

1. **Python import.** ``from tools.<tool> import main, setup_parser``.
   Build an ``argparse.Namespace`` from the caller's ``cli_args`` via
   ``setup_parser`` so we honour merlin's defaults and required-arg
   checks, then call ``main(ns)``. Reuses the active Python process —
   no subprocess startup cost.
2. **Subprocess fallback.** ``python -m tools.<tool> <cli_args>`` with
   ``cwd=$MERLIN_ROOT`` and ``PYTHONPATH`` prepended with the same
   root. Used when ``merlin-dev`` is not on the active interpreter's
   ``sys.path`` (e.g., XPU-RT's venv differs from merlin's).
3. **Failure.** :class:`MerlinUnavailableError` carrying both the
   import error and the subprocess exit information.

The bridge is intentionally stateless apart from the resolved
``merlin_root`` — every call independently re-resolves so callers can
share one ``MerlinBridge`` instance across invocations safely.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class MerlinUnavailableError(RuntimeError):
    """Merlin is not reachable from this Python environment.

    Carries the merlin_root the bridge tried, the tool that was being
    invoked, the underlying ImportError text (if any), and the
    subprocess exit info (if any) so the caller can log a single
    actionable message.
    """

    def __init__(
        self,
        *,
        merlin_root: Path,
        tool: str,
        import_error: str | None,
        subprocess_returncode: int | None,
        subprocess_stderr: str | None,
    ) -> None:
        self.merlin_root = merlin_root
        self.tool = tool
        self.import_error = import_error
        self.subprocess_returncode = subprocess_returncode
        self.subprocess_stderr = subprocess_stderr
        parts = [f"merlin tool {tool!r} unavailable under merlin_root={merlin_root}"]
        if import_error:
            parts.append(f"import: {import_error}")
        if subprocess_returncode is not None:
            tail = (subprocess_stderr or "").strip().splitlines()[-3:]
            parts.append(
                f"subprocess: rc={subprocess_returncode}; "
                f"stderr_tail={tail}"
            )
        super().__init__(" | ".join(parts))


@dataclass(frozen=True)
class MerlinCallResult:
    """Outcome of one :meth:`MerlinBridge.call`.

    ``returncode`` is 0 on success (Python-API or subprocess path).
    ``via`` tells the caller which path produced the result so tests
    can pin behaviour. ``stdout`` / ``stderr`` are captured from
    both paths; the Python-API path captures via stream redirection.
    """

    tool: str
    returncode: int
    via: str  # "python-api" | "subprocess"
    stdout: str
    stderr: str
    cli_args: tuple[str, ...]


def _default_merlin_root() -> Path:
    return Path(os.environ.get("MERLIN_ROOT", "/scratch2/agustin/merlin"))


@dataclass(frozen=True)
class MerlinBridge:
    """Stateless resolver for merlin tool invocations.

    Attributes:
        merlin_root: Path to the merlin checkout. Defaults to
            ``$MERLIN_ROOT`` or ``/scratch2/agustin/merlin``.
        python: Interpreter used for the subprocess fallback.
            Defaults to ``sys.executable``.
        prefer_python_api: When False, skip the in-process path and go
            straight to subprocess. Useful for tests that want to pin
            behaviour, and as an escape hatch when the Python-API
            import drags in heavy deps (torch, iree-base-compiler).
    """

    merlin_root: Path = field(default_factory=_default_merlin_root)
    python: Path = field(default_factory=lambda: Path(sys.executable))
    prefer_python_api: bool = True

    def available(self) -> bool:
        """Cheap probe: does the merlin root look right on disk?"""
        return (self.merlin_root / "tools").is_dir()

    def call(
        self,
        tool: str,
        *,
        cli_args: list[str] | None = None,
        env_extra: dict[str, str] | None = None,
    ) -> MerlinCallResult:
        """Invoke ``tools/<tool>.py`` and return its outcome.

        Args:
            tool: Module name under ``$MERLIN_ROOT/tools/`` (e.g.
                ``"compile"``, ``"compile_dispatch_matrix"``,
                ``"chipyard"``, ``"onnx_to_mlir"``,
                ``"profile_dispatch_matrix"``).
            cli_args: Argv-style arguments passed both to the in-process
                ``setup_parser`` and to the subprocess fallback. The
                tool's own ``main()`` reads them from the resulting
                ``argparse.Namespace``.
            env_extra: Extra env vars merged into the subprocess
                environment. Ignored on the Python-API path (the caller
                must set them itself before invoking the bridge).

        Returns:
            :class:`MerlinCallResult` on success or non-zero exit.

        Raises:
            MerlinUnavailableError: If neither path can reach the tool.
        """
        cli_args = list(cli_args or [])
        import_err: str | None = None

        if self.prefer_python_api:
            try:
                fn, parser_setup = self._resolve_python_entry(tool)
            except ImportError as exc:
                import_err = f"{type(exc).__name__}: {exc}"
            else:
                ns = self._build_namespace(parser_setup, cli_args)
                return self._invoke_python(tool, fn, ns, cli_args)

        rc, stdout, stderr = self._invoke_subprocess(tool, cli_args, env_extra)
        if rc is None:
            raise MerlinUnavailableError(
                merlin_root=self.merlin_root,
                tool=tool,
                import_error=import_err,
                subprocess_returncode=None,
                subprocess_stderr=stderr,
            )
        return MerlinCallResult(
            tool=tool,
            returncode=rc,
            via="subprocess",
            stdout=stdout,
            stderr=stderr,
            cli_args=tuple(cli_args),
        )

    # ----- internals ------------------------------------------------- #

    def _resolve_python_entry(
        self, tool: str
    ) -> tuple[Callable[[argparse.Namespace], int], Callable[[argparse.ArgumentParser], None]]:
        """Import ``tools.<tool>`` and return ``(main, setup_parser)``.

        Adds ``merlin_root`` to ``sys.path`` if needed. We do NOT
        permanently mutate ``sys.path`` — the entry is appended,
        then left in place so subsequent calls hit the import cache.
        """
        import importlib

        root = str(self.merlin_root)
        if root not in sys.path:
            sys.path.append(root)
        module = importlib.import_module(f"tools.{tool}")
        if not hasattr(module, "main"):
            raise ImportError(
                f"tools.{tool} loaded but does not expose main(args) -> int"
            )
        if not hasattr(module, "setup_parser"):
            raise ImportError(
                f"tools.{tool} loaded but does not expose setup_parser(parser)"
            )
        return module.main, module.setup_parser  # type: ignore[return-value]

    @staticmethod
    def _build_namespace(
        setup_parser: Callable[[argparse.ArgumentParser], None],
        cli_args: list[str],
    ) -> argparse.Namespace:
        """Construct an ``argparse.Namespace`` exactly the way the tool
        would receive it from its own ``__main__`` block."""
        parser = argparse.ArgumentParser()
        setup_parser(parser)
        return parser.parse_args(cli_args)

    def _invoke_python(
        self,
        tool: str,
        fn: Callable[[argparse.Namespace], int],
        ns: argparse.Namespace,
        cli_args: list[str],
    ) -> MerlinCallResult:
        """Run ``main(ns)`` in-process, capturing stdout / stderr."""
        out_buf, err_buf = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                rc = fn(ns)
        except SystemExit as exc:
            rc = int(exc.code) if isinstance(exc.code, int) else 1
        except Exception as exc:  # noqa: BLE001 - report to caller
            logger.exception("merlin tool %s raised", tool)
            err_buf.write(f"\nuncaught: {type(exc).__name__}: {exc}\n")
            rc = 1
        return MerlinCallResult(
            tool=tool,
            returncode=int(rc or 0),
            via="python-api",
            stdout=out_buf.getvalue(),
            stderr=err_buf.getvalue(),
            cli_args=tuple(cli_args),
        )

    def _invoke_subprocess(
        self,
        tool: str,
        cli_args: list[str],
        env_extra: dict[str, str] | None,
    ) -> tuple[int | None, str, str]:
        """Run ``python -m tools.<tool>`` under ``merlin_root``.

        Returns ``(returncode, stdout, stderr)``. ``returncode`` is
        ``None`` if the bridge couldn't even start the subprocess
        (e.g., ``merlin_root`` is missing on disk).
        """
        if not self.available():
            return None, "", f"merlin_root not on disk: {self.merlin_root}"
        env = os.environ.copy()
        py_path = str(self.merlin_root)
        env["PYTHONPATH"] = (
            f"{py_path}{os.pathsep}{env['PYTHONPATH']}"
            if "PYTHONPATH" in env
            else py_path
        )
        if env_extra:
            env.update(env_extra)
        cmd = [str(self.python), "-m", f"tools.{tool}", *cli_args]
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.merlin_root), env=env,
                capture_output=True, text=True,
            )
        except OSError as exc:
            return None, "", f"subprocess failed to start: {exc}"
        return proc.returncode, proc.stdout, proc.stderr
