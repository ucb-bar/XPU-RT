"""Typed errors for the tool-promotion pipeline.

Each error class names a single failure mode so callers can branch on
type rather than message. Every error must be raised with a concrete
``tool_id`` (or ``source`` path for cards being constructed) so the
audit log can attribute the failure.

See :mod:`compgen.tools.tool_card` for the contract these errors
defend, and :mod:`compgen.audit.tool_promotion` for the rollup that
treats these as gate violations.
"""

from __future__ import annotations


class ToolCardError(ValueError):
    """A ToolCard YAML body violated the schema or a closed-enum field."""


class ToolMaturityError(ValueError):
    """A tool was declared at a maturity level whose evidence is missing.

    Raised by :mod:`compgen.audit.tool_promotion`, never by the runner.
    """


class ToolInputSchemaError(ValueError):
    """A tool was invoked with a request that violates its input_schema."""


class ToolOutputSchemaError(ValueError):
    """A tool returned an output that violates its output_schema.

    Surfacing this is a hard rule — the runner refuses to write
    ``result.json`` if the tool's output does not validate. A T2+
    tool that hits this in its positive fixture has a real bug.
    """


class ToolEntrypointError(ImportError):
    """A ToolCard's ``entrypoints.python`` could not be resolved.

    Raised when the ``module:attr`` string fails to import or the
    attribute is not callable.
    """


class ToolRunError(RuntimeError):
    """A tool's Python entrypoint raised during execution.

    The original exception is chained via ``__cause__``; the runner
    records the typed reason in ``trace.jsonl`` so a downstream audit
    can distinguish "tool refused (blocked)" from "tool crashed
    (error)".
    """
