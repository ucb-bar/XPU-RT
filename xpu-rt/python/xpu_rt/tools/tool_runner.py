"""ToolRunner — generic dispatcher for ToolCard-declared tools.

Responsibilities, in order:

1. Resolve the card's ``entrypoints.python`` (``module:attr``) to a
   callable.
2. Validate the caller's ``request`` against the card's
   ``input_schema`` using JSONSchema.
3. Invoke the entrypoint with ``request`` and ``out_dir``.
4. Validate the returned dict against the card's ``output_schema``.
5. Verify any artifact paths the entrypoint reports under
   ``artifacts`` resolve under one of the card's ``writes.allowed_roots``.
6. Compute deterministic SHA-256 hashes of canonical-JSON
   serialisations of input + output so audits can replay.
7. Write ``out_dir/result.json`` (final ToolResult) and
   ``out_dir/trace.jsonl`` (one event line — start, end, optional
   error). The trace is append-only so multiple runner invocations
   in one ``out_dir`` accumulate.

Hard rules (enforced; not suggestions):

* If output_schema validation fails, the runner raises
  :class:`ToolOutputSchemaError` and refuses to write ``result.json``.
  A T2+ tool that hits this in its positive fixture has a real bug.
* If the entrypoint raises, the runner catches it and writes a
  ``status=error`` trace event with the typed reason, then re-raises
  :class:`ToolRunError` so callers can distinguish from a
  successful ``status=blocked`` return.
* Tools are pure-ish: the runner refuses to inspect the working
  directory afterwards. Writes outside ``out_dir`` are detected only
  if the tool reports them in its ``artifacts`` field. Audits
  do a stricter import + path probe.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from compgen.tools.errors import (
    ToolEntrypointError,
    ToolInputSchemaError,
    ToolOutputSchemaError,
    ToolRunError,
)
from compgen.tools.tool_card import TOOL_STATUSES, ToolCard


def resolve_python_entrypoint(spec: str) -> Callable[..., Any]:
    """Resolve a ``module.path:attribute`` string to a callable.

    Raises :class:`compgen.tools.errors.ToolEntrypointError` if the
    module cannot be imported, the attribute does not exist, or the
    attribute is not callable. The original error is chained via
    ``__cause__`` so callers can see the underlying ImportError.
    """

    if ":" not in spec:
        raise ToolEntrypointError(
            f"python entrypoint {spec!r} must be 'module.path:attribute'"
        )
    module_path, attr = spec.split(":", 1)
    if not module_path or not attr:
        raise ToolEntrypointError(
            f"python entrypoint {spec!r} has empty module or attribute"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ToolEntrypointError(
            f"python entrypoint {spec!r}: cannot import module {module_path!r}"
        ) from exc
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise ToolEntrypointError(
            f"python entrypoint {spec!r}: module {module_path!r} has no "
            f"attribute {attr!r}"
        ) from exc
    if not callable(target):
        raise ToolEntrypointError(
            f"python entrypoint {spec!r}: {module_path}.{attr} is not callable"
        )
    return target


def _canonical_json(payload: Any) -> bytes:
    """Canonical-JSON bytes for hashing.

    ``sort_keys`` + ``separators`` together fix every quoting and
    ordering ambiguity, so identical requests/outputs hash identically
    across runs and machines.
    """

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validate_against(schema: dict[str, Any], instance: Any, *, kind: str, tool_id: str) -> None:
    try:
        jsonschema.validate(instance=instance, schema=schema)
    except jsonschema.ValidationError as exc:
        if kind == "input":
            raise ToolInputSchemaError(
                f"tool {tool_id!r} input failed schema validation: {exc.message} "
                f"(path={list(exc.absolute_path)})"
            ) from exc
        raise ToolOutputSchemaError(
            f"tool {tool_id!r} output failed schema validation: {exc.message} "
            f"(path={list(exc.absolute_path)})"
        ) from exc


def _check_artifact_under_allowed_roots(
    path: Path, allowed_roots: tuple[str, ...], out_dir: Path, tool_id: str
) -> None:
    resolved = path.resolve()
    candidates = []
    for root in allowed_roots:
        expanded = root.replace("${run_dir}", str(out_dir.resolve()))
        candidates.append(Path(expanded).resolve())
    for cand in candidates:
        try:
            resolved.relative_to(cand)
            return
        except ValueError:
            continue
    raise ToolRunError(
        f"tool {tool_id!r} wrote artifact {resolved} outside allowed_roots="
        f"{[str(c) for c in candidates]}"
    )


@dataclass(frozen=True)
class ToolResult:
    """Typed return from :meth:`ToolRunner.run`.

    Fields:

    * ``tool_id`` — copy of the card's ``tool_id``.
    * ``status`` — closed enum subset of
      :data:`compgen.tools.tool_card.TOOL_STATUSES`. The runner
      copies this from the tool's output dict and re-validates against
      the card's output_schema.
    * ``result`` — the raw output dict (already validated against
      ``output_schema``).
    * ``artifacts`` — paths the tool reported writing; the runner
      verifies each lives under one of ``writes.allowed_roots``.
    * ``input_hash`` / ``output_hash`` — SHA-256 of canonical-JSON
      serialisation of input/output; used by replay / fresh-agent
      grading to assert determinism.
    * ``duration_ms`` — wall-clock time for the entrypoint call (not
      including schema validation).
    * ``trace_path`` — filesystem path to the appended trace.jsonl.
    """

    tool_id: str
    status: str
    result: dict[str, Any]
    artifacts: tuple[str, ...]
    input_hash: str
    output_hash: str
    duration_ms: float
    trace_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "status": self.status,
            "result": self.result,
            "artifacts": list(self.artifacts),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "duration_ms": self.duration_ms,
            "trace_path": self.trace_path,
        }


@dataclass
class _TraceWriter:
    path: Path

    def append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


@dataclass
class ToolRunner:
    """Runs a tool declared by a :class:`ToolCard`.

    The runner is stateless across invocations; pass a fresh
    instance (or reuse — both are safe). The only persistent state
    is the trace file under each ``out_dir``.

    Example::

        from compgen.tools import ToolRunner, load_tool_card
        card = load_tool_card(Path("python/compgen/tools/cards/echo.yaml"))
        result = ToolRunner().run(card, request={"text": "hi"}, out_dir=Path("/tmp/echo"))
        assert result.status == "ok"
    """

    write_result_json: bool = True

    def run(
        self,
        card: ToolCard,
        *,
        request: dict[str, Any],
        out_dir: Path,
    ) -> ToolResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trace = _TraceWriter(out_dir / "trace.jsonl")

        # 1. Input validation.
        _validate_against(card.input_schema, request, kind="input", tool_id=card.tool_id)
        input_hash = _sha256(request)

        # 2. Resolve entrypoint.
        try:
            fn = resolve_python_entrypoint(card.entrypoints.python)
        except ToolEntrypointError as exc:
            trace.append(
                {
                    "event": "entrypoint_error",
                    "tool_id": card.tool_id,
                    "spec": card.entrypoints.python,
                    "error": str(exc),
                }
            )
            raise

        trace.append(
            {
                "event": "start",
                "tool_id": card.tool_id,
                "maturity": card.maturity,
                "phase": card.phase,
                "input_hash": input_hash,
                "entrypoint": card.entrypoints.python,
                "out_dir": str(out_dir.resolve()),
            }
        )

        # 3. Invoke.
        start = time.perf_counter()
        try:
            raw_output = fn(request, out_dir=out_dir)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace.append(
                {
                    "event": "error",
                    "tool_id": card.tool_id,
                    "duration_ms": duration_ms,
                    "exc_type": type(exc).__name__,
                    "exc_message": str(exc),
                }
            )
            raise ToolRunError(
                f"tool {card.tool_id!r} entrypoint raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        duration_ms = (time.perf_counter() - start) * 1000.0

        if not isinstance(raw_output, dict):
            raise ToolOutputSchemaError(
                f"tool {card.tool_id!r} returned {type(raw_output).__name__}; "
                f"must return a dict matching output_schema"
            )

        # 4. Output validation.
        _validate_against(
            card.output_schema, raw_output, kind="output", tool_id=card.tool_id
        )
        status = raw_output.get("status")
        if status not in TOOL_STATUSES:
            raise ToolOutputSchemaError(
                f"tool {card.tool_id!r} returned status={status!r}; must be "
                f"one of {TOOL_STATUSES}"
            )

        # 5. Artifact verification.
        artifacts_raw = raw_output.get("artifacts", ()) or ()
        if not isinstance(artifacts_raw, (list, tuple)):
            raise ToolOutputSchemaError(
                f"tool {card.tool_id!r} returned artifacts="
                f"{type(artifacts_raw).__name__}; must be a list of paths"
            )
        artifact_paths: list[str] = []
        for a in artifacts_raw:
            a_path = Path(str(a))
            _check_artifact_under_allowed_roots(
                a_path, card.writes.allowed_roots, out_dir, card.tool_id
            )
            artifact_paths.append(str(a_path))

        # 6. Hashes.
        output_hash = _sha256(raw_output)

        trace.append(
            {
                "event": "end",
                "tool_id": card.tool_id,
                "status": status,
                "duration_ms": duration_ms,
                "output_hash": output_hash,
                "artifacts": artifact_paths,
            }
        )

        result = ToolResult(
            tool_id=card.tool_id,
            status=status,
            result=raw_output,
            artifacts=tuple(artifact_paths),
            input_hash=input_hash,
            output_hash=output_hash,
            duration_ms=duration_ms,
            trace_path=str((out_dir / "trace.jsonl").resolve()),
        )

        if self.write_result_json:
            (out_dir / "result.json").write_text(
                json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

        return result
