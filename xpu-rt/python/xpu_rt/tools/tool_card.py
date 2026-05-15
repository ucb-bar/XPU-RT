"""ToolCard schema.

A ToolCard is a frozen, deterministic declaration of a single
agent-callable tool: what it does, how it is invoked, what it writes,
what is forbidden, and what evidence backs each rung of its maturity
ladder.

ToolCards are loaded from YAML files under
``python/compgen/tools/cards/`` (see
:mod:`compgen.tools.tool_registry`). They are *declarations*, not
runtime objects — the corresponding Python entrypoint is resolved
lazily by :class:`compgen.tools.tool_runner.ToolRunner`.

The schema is intentionally narrow. Closed enums (``MATURITY_LEVELS``,
``TOOL_PHASES``, ``FORBIDDEN_ACTIONS``, ``TOOL_STATUSES``,
``PROMOTION_REQUIREMENT_KEYS``) are checked at construction; unknown
values raise :class:`compgen.tools.errors.ToolCardError` so a typo in
YAML cannot quietly produce an under-audited tool.

Mirrors :class:`compgen.providers.provider_types.ProviderCard` so
future card families can lift the same loader pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from compgen.tools.errors import ToolCardError

SCHEMA_VERSION: Final[str] = "compgen_tool_card_v1"

# Eight-rung maturity ladder. Audit enforces evidence at each
# rung; the runner treats T0+ as runnable but only validates
# schemas — it does not enforce maturity.
MATURITY_LEVELS: Final[tuple[str, ...]] = (
    "T0",  # Python function only
    "T1",  # CLI wrapper
    "T2",  # tested (positive + negative controls)
    "T3",  # artifact-emitting (writes structured outputs)
    "T4",  # skill-backed (SKILL.md exists and lints)
    "T5",  # MCP wrapper
    "T6",  # fresh-agent verified (harness has graded a real run)
    "T7",  # default workflow tool (appears in phase menu)
)

# Six-phase tool taxonomy aligning with the agent loop's named
# decision points. Every tool declares exactly one phase so the
# phase-scoped MCP menu can group them.
TOOL_PHASES: Final[tuple[str, ...]] = (
    "env_probe",
    "graph_analysis",
    "recipe_authoring",
    "kernel_codegen",
    "extension_authoring",
    "evidence",
)

# Status enum returned by every tool. The runner enforces that the
# tool's output_schema constrains a top-level ``status`` field to
# this set (or a subset). Refer to the docstring of
# :class:`compgen.tools.tool_runner.ToolResult` for the contract.
TOOL_STATUSES: Final[tuple[str, ...]] = (
    "ok",
    "blocked",
    "error",
)

# Closed enum of forbidden actions. A tool that performs any of these
# is by definition not promotable. The audit checks both the
# declaration (this list) and the implementation (AST + path probes).
FORBIDDEN_ACTIONS: Final[tuple[str, ...]] = (
    "mutate_payload_ir",
    "mutate_recipe_ir",
    "bypass_verifier",
    "write_outside_artifact_dir",
    "invent_certificate",
    "silent_failure",
    "skip_negative_control",
    "import_optional_provider_at_module_top",
)

# Promotion requirement booleans the card declares; the audit checks
# each one against repo state. A card may legally declare a
# requirement as ``false`` (e.g., a T2 tool not yet skill-backed);
# the audit then refuses to promote it past the matching rung.
PROMOTION_REQUIREMENT_KEYS: Final[tuple[str, ...]] = (
    "unit_tests",
    "negative_controls",
    "cli_wrapper",
    "artifact_outputs",
    "skill_doc",
    "mcp_wrapper",
    "fresh_agent_harness",
    "phase_menu_listing",
)


@dataclass(frozen=True)
class ToolEntrypoints:
    """The three callable surfaces a tool may expose.

    * ``python`` — ``module:attr`` string resolving to a callable
      ``(request: dict, *, out_dir: Path) -> dict``.
    * ``cli`` — a concrete shell command (without arguments) that the
      audit checks resolves on PATH; required at T1+.
    * ``mcp`` — the MCP tool name registered by the bridge;
      required at T5+.

    All three default to ``""`` (absent) so a card can declare only
    the surfaces it has reached.
    """

    python: str
    cli: str = ""
    mcp: str = ""


@dataclass(frozen=True)
class ToolWrites:
    """Allowed write roots for the tool.

    Every path the tool writes must be under at least one of these
    roots. The runner enforces this when the tool's Python entrypoint
    declares its outputs via the ``allowed_roots`` argument; audits
     check the declared list against the implementation.

    Paths may contain ``${run_dir}`` as a literal placeholder — the
    runner substitutes the caller-provided ``out_dir`` before
    checking.
    """

    allowed_roots: tuple[str, ...]


@dataclass(frozen=True)
class ToolTests:
    """Declared test pointers for the tool.

    Format: ``module.path::test_name``. Audit imports each
    pointer and asserts the test exists; negative controls must be
    parametrized fault-injection tests that *fail* when the tool's
    invariant is broken.
    """

    positive: tuple[str, ...] = ()
    negative_controls: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolPromotionRequirements:
    """Per-rung evidence checklist.

    Each key maps to a boolean. A ``true`` declares that the tool's
    author has provided this evidence; the audit verifies it.
    A ``false`` is legal but caps the achievable maturity.
    """

    flags: dict[str, bool] = field(default_factory=dict)

    def get(self, key: str) -> bool:
        if key not in PROMOTION_REQUIREMENT_KEYS:
            raise ToolCardError(
                f"unknown promotion requirement key {key!r}; "
                f"must be one of {PROMOTION_REQUIREMENT_KEYS}"
            )
        return bool(self.flags.get(key, False))

    def to_dict(self) -> dict[str, bool]:
        return {key: bool(self.flags.get(key, False)) for key in PROMOTION_REQUIREMENT_KEYS}


@dataclass(frozen=True)
class ToolCard:
    """Frozen declaration of an agent-callable tool.

    Cards live under ``python/compgen/tools/cards/*.yaml`` and are
    loaded by :func:`compgen.tools.tool_registry.iter_tool_cards`.
    A card is not evidence that the tool *works*; it is the
    contract :class:`compgen.tools.tool_runner.ToolRunner` validates
    against and the surface :mod:`compgen.audit.tool_promotion`
    audits against.
    """

    schema_version: str
    tool_id: str
    maturity: str
    phase: str
    description: str
    entrypoints: ToolEntrypoints
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    writes: ToolWrites
    forbidden: tuple[str, ...]
    promotion_requirements: ToolPromotionRequirements
    tests: ToolTests = field(default_factory=ToolTests)
    skill_path: str = ""
    fresh_agent_task_id: str = ""
    owner: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ToolCardError(
                f"tool {self.tool_id!r} schema_version="
                f"{self.schema_version!r} must be {SCHEMA_VERSION!r}"
            )
        if not self.tool_id or not isinstance(self.tool_id, str):
            raise ToolCardError(
                f"tool_id must be a non-empty string; got {self.tool_id!r}"
            )
        if self.maturity not in MATURITY_LEVELS:
            raise ToolCardError(
                f"tool {self.tool_id!r} maturity={self.maturity!r} "
                f"must be one of {MATURITY_LEVELS}"
            )
        if self.phase not in TOOL_PHASES:
            raise ToolCardError(
                f"tool {self.tool_id!r} phase={self.phase!r} "
                f"must be one of {TOOL_PHASES}"
            )
        if not self.entrypoints.python:
            raise ToolCardError(
                f"tool {self.tool_id!r} entrypoints.python is required "
                f"(format: 'module.path:attribute')"
            )
        for action in self.forbidden:
            if action not in FORBIDDEN_ACTIONS:
                raise ToolCardError(
                    f"tool {self.tool_id!r} forbidden action {action!r} "
                    f"must be one of {FORBIDDEN_ACTIONS}"
                )
        # input/output schemas must be JSON-schema-shaped dicts (check
        # shape first so a malformed output_schema produces a clear
        # error before the status-enum probe runs).
        for label, schema in (("input_schema", self.input_schema), ("output_schema", self.output_schema)):
            if not isinstance(schema, dict):
                raise ToolCardError(
                    f"tool {self.tool_id!r} {label} must be a JSON-schema "
                    f"dict; got {type(schema).__name__}"
                )
            if schema.get("type") != "object":
                raise ToolCardError(
                    f"tool {self.tool_id!r} {label}.type must be "
                    f"'object'; got {schema.get('type')!r}"
                )
        # Output schema must constrain status to a subset of TOOL_STATUSES.
        self._validate_output_status_enum()
        # Maturity rung consistency with promotion_requirements.
        self._validate_maturity_evidence()

    def _validate_output_status_enum(self) -> None:
        props = self.output_schema.get("properties") or {}
        status_schema = props.get("status")
        if status_schema is None:
            raise ToolCardError(
                f"tool {self.tool_id!r} output_schema must declare a "
                f"top-level 'status' property with an enum subset of "
                f"{TOOL_STATUSES}"
            )
        enum = status_schema.get("enum")
        if not isinstance(enum, list) or not enum:
            raise ToolCardError(
                f"tool {self.tool_id!r} output_schema.properties.status.enum "
                f"must be a non-empty list (subset of {TOOL_STATUSES})"
            )
        for value in enum:
            if value not in TOOL_STATUSES:
                raise ToolCardError(
                    f"tool {self.tool_id!r} output_schema status enum "
                    f"member {value!r} must be one of {TOOL_STATUSES}"
                )

    def _validate_maturity_evidence(self) -> None:
        """A card cannot *declare* a maturity above the level supported
        by its declared evidence flags. The audit re-verifies that each
        flag corresponds to real repo state; this check only
        catches obviously self-contradictory cards (e.g., maturity=T5
        but ``mcp_wrapper=false``).
        """
        idx = MATURITY_LEVELS.index(self.maturity)
        rules: tuple[tuple[int, str], ...] = (
            (1, "cli_wrapper"),         # T1 requires CLI
            (2, "unit_tests"),           # T2 requires positive tests
            (2, "negative_controls"),    # T2 requires negative controls
            (3, "artifact_outputs"),     # T3 requires artifact emission
            (4, "skill_doc"),            # T4 requires skill
            (5, "mcp_wrapper"),          # T5 requires MCP
            (6, "fresh_agent_harness"),  # T6 requires harness
            (7, "phase_menu_listing"),   # T7 requires phase menu
        )
        for required_idx, flag in rules:
            if idx >= required_idx and not self.promotion_requirements.get(flag):
                raise ToolCardError(
                    f"tool {self.tool_id!r} declared maturity="
                    f"{self.maturity!r} but promotion_requirements."
                    f"{flag}=false; lower maturity or supply the evidence"
                )

    @classmethod
    def from_dict(
        cls, body: dict[str, Any], *, source: Path | None = None
    ) -> ToolCard:
        try:
            schema_version = str(body["schema_version"])
            tool_id = str(body["tool_id"])
            maturity = str(body["maturity"])
            phase = str(body["phase"])
            entrypoints_body = body["entrypoints"]
            input_schema = body["input_schema"]
            output_schema = body["output_schema"]
            writes_body = body["writes"]
        except KeyError as exc:
            raise ToolCardError(
                f"tool card missing required field {exc.args[0]!r} "
                f"(source={source})"
            ) from exc

        if not isinstance(entrypoints_body, dict):
            raise ToolCardError(
                f"tool card entrypoints must be a mapping (source={source})"
            )
        entrypoints = ToolEntrypoints(
            python=str(entrypoints_body.get("python", "")),
            cli=str(entrypoints_body.get("cli", "")),
            mcp=str(entrypoints_body.get("mcp", "")),
        )

        if not isinstance(writes_body, dict):
            raise ToolCardError(
                f"tool {tool_id!r} writes must be a mapping (source={source})"
            )
        allowed_roots = writes_body.get("allowed_roots", ())
        if not isinstance(allowed_roots, (list, tuple)):
            raise ToolCardError(
                f"tool {tool_id!r} writes.allowed_roots must be a list"
            )
        writes = ToolWrites(allowed_roots=tuple(str(p) for p in allowed_roots))

        forbidden = tuple(str(a) for a in body.get("forbidden", ()))

        promotion_body = body.get("promotion_requirements", {}) or {}
        if not isinstance(promotion_body, dict):
            raise ToolCardError(
                f"tool {tool_id!r} promotion_requirements must be a mapping"
            )
        for key in promotion_body:
            if key not in PROMOTION_REQUIREMENT_KEYS:
                raise ToolCardError(
                    f"tool {tool_id!r} unknown promotion_requirements key "
                    f"{key!r}; must be one of {PROMOTION_REQUIREMENT_KEYS}"
                )
        # Normalize: every known key is present (default False) so
        # ``to_dict`` → ``from_dict`` round-trips bit-identically.
        promotion_requirements = ToolPromotionRequirements(
            flags={
                key: bool(promotion_body.get(key, False))
                for key in PROMOTION_REQUIREMENT_KEYS
            }
        )

        tests_body = body.get("tests", {}) or {}
        if not isinstance(tests_body, dict):
            raise ToolCardError(
                f"tool {tool_id!r} tests must be a mapping"
            )
        tests = ToolTests(
            positive=tuple(str(t) for t in tests_body.get("positive", ())),
            negative_controls=tuple(
                str(t) for t in tests_body.get("negative_controls", ())
            ),
        )

        return cls(
            schema_version=schema_version,
            tool_id=tool_id,
            maturity=maturity,
            phase=phase,
            description=str(body.get("description", "")),
            entrypoints=entrypoints,
            input_schema=dict(input_schema) if isinstance(input_schema, dict) else {},
            output_schema=dict(output_schema) if isinstance(output_schema, dict) else {},
            writes=writes,
            forbidden=forbidden,
            promotion_requirements=promotion_requirements,
            tests=tests,
            skill_path=str(body.get("skill_path", "")),
            fresh_agent_task_id=str(body.get("fresh_agent_task_id", "")),
            owner=str(body.get("owner", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool_id": self.tool_id,
            "maturity": self.maturity,
            "phase": self.phase,
            "description": self.description,
            "entrypoints": {
                "python": self.entrypoints.python,
                "cli": self.entrypoints.cli,
                "mcp": self.entrypoints.mcp,
            },
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "writes": {"allowed_roots": list(self.writes.allowed_roots)},
            "forbidden": list(self.forbidden),
            "promotion_requirements": self.promotion_requirements.to_dict(),
            "tests": {
                "positive": list(self.tests.positive),
                "negative_controls": list(self.tests.negative_controls),
            },
            "skill_path": self.skill_path,
            "fresh_agent_task_id": self.fresh_agent_task_id,
            "owner": self.owner,
        }

    @property
    def maturity_index(self) -> int:
        return MATURITY_LEVELS.index(self.maturity)
