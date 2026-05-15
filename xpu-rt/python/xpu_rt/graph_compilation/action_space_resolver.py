"""Action Space Resolver (Milestone 04.5).

Resolves a ``selected_candidate_id`` against the *canonical* IR
artifact ``02_graph_analysis/action_space.mlir`` rather than against
the JSON projections. This is the architectural firewall that keeps
JSON from quietly becoming the compiler's source of truth.

The resolver:

1. Reads ``action_space.mlir`` and recomputes its sha256.
2. Loads the three JSON projections (``decision_sites.json``,
   ``candidate_actions.json``, ``llm_action_space.json``) and verifies
   that **every** projection's ``source.action_space_ir_sha256`` matches
   the recomputed digest.
3. Looks up the candidate in ``candidate_actions.json``.
4. Looks up the candidate in the IR text and parses its
   ``compgen.candidate @<id> attributes { ... } { ... }`` block.
5. Cross-verifies that the IR's recipe ops + attributes match the
   JSON-projected ``recipe_delta``.
6. Applies the legality gate (``--allow-illegal`` opens it).
7. Emits ``02_graph_analysis/action_space_resolver_report.json`` plus
   optional ``03_recipe_planning/selected_recipe_delta.mlir`` /
   ``03_recipe_planning/candidate_selection.json`` writeouts.

Read-only against compiler core. The resolver is a verifier, not a
compiler pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ResolverError(RuntimeError):
    """Base class for resolver failures (used as a typed exit signal)."""


class CandidateNotFoundError(ResolverError):
    """Raised when ``candidate_id`` is not present in the canonical IR."""


class HashMismatchError(ResolverError):
    """Raised when JSON projections disagree with the canonical IR."""


class IllegalCandidateError(ResolverError):
    """Raised when the candidate is marked illegal and ``allow_illegal`` is False."""


class RecipeDeltaMismatchError(ResolverError):
    """Raised when JSON-projected ``recipe_delta`` disagrees with the IR."""


# --------------------------------------------------------------------------- #
# Public dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResolvedCandidate:
    candidate_id: str
    site_id: str
    region_id: str
    kind: str
    label: str
    legality_ok: bool
    legality_reason: str
    recipe_delta: list[dict[str, Any]]
    cost_preview: dict[str, Any]
    evidence: dict[str, Any]
    source: dict[str, str]  # action_space_ir, action_space_ir_sha256
    ir_block_text: str       # the verbatim ``compgen.candidate ... { ... }`` block


@dataclass(frozen=True)
class ResolverReport:
    overall: str  # "pass" | "fail"
    candidate_id: str
    checks: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# IR block extractor
# --------------------------------------------------------------------------- #


_CANDIDATE_HEADER_RE = re.compile(
    r"^\s*compgen\.candidate\s+@(?P<id>[A-Za-z0-9_]+)\s+attributes\s*\{(?P<attrs>[^}]*)\}\s*\{\s*$"
)
_RECIPE_OP_RE = re.compile(
    r"^\s*recipe\.(?P<op>[A-Za-z_][A-Za-z0-9_]*)\s+attributes\s*\{\s*(?P<body>.*?)\s*\}\s*$"
)
_OP_LINE_TERMINATOR_RE = re.compile(r"^\s*\}\s*$")


def _extract_candidate_block(mlir_text: str, candidate_id: str) -> tuple[str, list[str]]:
    """Return ``(verbatim_block, recipe_lines)``.

    ``verbatim_block`` includes the header line, the recipe op lines,
    and the closing ``}``. ``recipe_lines`` is the inner ``recipe.<op>
    attributes { ... }`` strings, in order.

    Raises :class:`CandidateNotFoundError` when the candidate is not in IR.
    """
    lines = mlir_text.splitlines()
    block_start = None
    for i, line in enumerate(lines):
        m = _CANDIDATE_HEADER_RE.match(line)
        if m and m.group("id") == candidate_id:
            block_start = i
            break
    if block_start is None:
        raise CandidateNotFoundError(
            f"candidate_id {candidate_id!r} not present in action_space.mlir"
        )
    # Walk forward to the matching ``}`` at the same indentation.
    block_lines = [lines[block_start]]
    recipe_lines: list[str] = []
    for j in range(block_start + 1, len(lines)):
        line = lines[j]
        block_lines.append(line)
        if _OP_LINE_TERMINATOR_RE.match(line):
            return "\n".join(block_lines), recipe_lines
        if "recipe." in line:
            recipe_lines.append(line)
    raise CandidateNotFoundError(
        f"candidate_id {candidate_id!r} block was not closed; truncated IR?"
    )


# --------------------------------------------------------------------------- #
# Tiny MLIR-attribute parser (just enough to verify recipe_delta)
# --------------------------------------------------------------------------- #


_SYMBOL_RE = re.compile(r"@[A-Za-z_][A-Za-z0-9_]*")


def _parse_attrs_value(text: str, pos: int) -> tuple[Any, int]:
    """Parse one attribute *value* starting at ``pos``. Returns
    ``(value, next_pos)``."""
    pos = _skip_ws(text, pos)
    if pos >= len(text):
        raise ValueError("unexpected end of attribute body")
    ch = text[pos]
    if ch == '"':
        return _parse_string(text, pos)
    if ch == "{":
        return _parse_dict(text, pos)
    if ch == "[":
        return _parse_list(text, pos)
    if ch == "@":
        # MLIR symbol reference (unquoted, e.g. ``@obl_recipe_0000``).
        m = _SYMBOL_RE.match(text, pos)
        if m:
            return m.group(0), m.end()
        raise ValueError(f"malformed symbol at pos {pos}: {text[pos:pos+30]!r}")
    if text.startswith("true", pos):
        return True, pos + 4
    if text.startswith("false", pos):
        return False, pos + 5
    return _parse_number(text, pos)


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_TYPE_RE = re.compile(r"\s*:\s*(?:i\d+|f\d+|index|i1)")


def _parse_number(text: str, pos: int) -> tuple[Any, int]:
    m = _NUMBER_RE.match(text, pos)
    if not m:
        raise ValueError(f"expected number at pos {pos}: {text[pos:pos+30]!r}")
    raw = m.group(0)
    end = m.end()
    type_match = _TYPE_RE.match(text, end)
    if type_match:
        type_token = type_match.group(0).split(":")[1].strip()
        end = type_match.end()
        if type_token.startswith("i") or type_token == "index":
            return int(raw), end
        return float(raw), end
    if "." in raw or "e" in raw or "E" in raw:
        return float(raw), end
    return int(raw), end


def _parse_string(text: str, pos: int) -> tuple[str, int]:
    assert text[pos] == '"'
    i = pos + 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            else:
                out.append(nxt)
            i += 2
            continue
        if ch == '"':
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    raise ValueError("unterminated string")


def _parse_list(text: str, pos: int) -> tuple[list[Any], int]:
    assert text[pos] == "["
    pos += 1
    items: list[Any] = []
    pos = _skip_ws(text, pos)
    if pos < len(text) and text[pos] == "]":
        return items, pos + 1
    while pos < len(text):
        v, pos = _parse_attrs_value(text, pos)
        items.append(v)
        pos = _skip_ws(text, pos)
        if pos < len(text) and text[pos] == ",":
            pos += 1
            continue
        if pos < len(text) and text[pos] == "]":
            return items, pos + 1
        raise ValueError(f"unexpected token in list at pos {pos}: {text[pos:pos+30]!r}")
    raise ValueError("unterminated list")


def _parse_dict(text: str, pos: int) -> tuple[dict[str, Any], int]:
    assert text[pos] == "{"
    pos += 1
    out: dict[str, Any] = {}
    pos = _skip_ws(text, pos)
    if pos < len(text) and text[pos] == "}":
        return out, pos + 1
    while pos < len(text):
        pos = _skip_ws(text, pos)
        # Parse key (identifier).
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[pos:])
        if not m:
            raise ValueError(f"expected key at pos {pos}: {text[pos:pos+30]!r}")
        key = m.group(0)
        pos += m.end()
        pos = _skip_ws(text, pos)
        if pos >= len(text) or text[pos] != "=":
            raise ValueError(f"expected '=' after key {key!r} at pos {pos}")
        pos += 1
        v, pos = _parse_attrs_value(text, pos)
        out[key] = v
        pos = _skip_ws(text, pos)
        if pos < len(text) and text[pos] == ",":
            pos += 1
            continue
        if pos < len(text) and text[pos] == "}":
            return out, pos + 1
        raise ValueError(f"unexpected token in dict at pos {pos}: {text[pos:pos+30]!r}")
    raise ValueError("unterminated dict")


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t\n\r":
        pos += 1
    return pos


def _parse_attrs_body(body: str) -> dict[str, Any]:
    """Parse the body of an MLIR ``attributes { ... }`` clause."""
    text = "{" + body + "}"
    out, end = _parse_dict(text, 0)
    if _skip_ws(text, end) != len(text):
        raise ValueError("trailing content after attribute block")
    return out


# --------------------------------------------------------------------------- #
# Recipe-delta cross-check
# --------------------------------------------------------------------------- #


_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")


def _camel_to_snake(s: str) -> str:
    return _CAMEL_BOUNDARY_RE.sub(r"\1_\2", s).lower()


def _normalize_value(v: Any) -> Any:
    """Normalize for cross-format comparison.

    - JSON ``null`` ↔ IR string ``"null"``: treat as equal (we serialize
      Python ``None`` as ``"null"`` in IR).
    - Numbers compared loosely (1 == 1.0).
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize_value(val) for k, val in sorted(v.items())}
    return v


def _values_match(a: Any, b: Any) -> bool:
    return bool(_normalize_value(a) == _normalize_value(b))


def _verify_recipe_delta_against_ir(
    recipe_delta_json: list[dict[str, Any]],
    recipe_ir_lines: list[str],
) -> tuple[bool, str]:
    """Verify that the JSON-projected recipe delta is faithful to the IR.

    Each JSON op's snake-case op name must match the corresponding IR
    op (in order), and every (key, value) in the JSON op (except
    ``op``) must match the IR-parsed attribute body.
    """
    if len(recipe_delta_json) != len(recipe_ir_lines):
        return (
            False,
            f"op count mismatch: JSON has {len(recipe_delta_json)}, "
            f"IR has {len(recipe_ir_lines)}",
        )
    for idx, (json_op, ir_line) in enumerate(zip(recipe_delta_json, recipe_ir_lines, strict=True)):
        m = _RECIPE_OP_RE.match(ir_line)
        if not m:
            return False, f"op #{idx}: cannot parse IR line {ir_line!r}"
        ir_op_snake = m.group("op")
        json_op_camel = json_op.get("op", "")
        json_op_snake = _camel_to_snake(json_op_camel)
        if ir_op_snake != json_op_snake:
            return (
                False,
                f"op #{idx} name mismatch: JSON {json_op_camel!r} -> "
                f"{json_op_snake!r}; IR {ir_op_snake!r}",
            )
        try:
            ir_attrs = _parse_attrs_body(m.group("body"))
        except (ValueError, AssertionError) as exc:
            return False, f"op #{idx}: failed to parse IR attrs: {exc}"
        json_body = {k: v for k, v in json_op.items() if k != "op"}
        for k, jval in json_body.items():
            if k not in ir_attrs:
                return False, f"op #{idx}: key {k!r} missing in IR attrs"
            if not _values_match(jval, ir_attrs[k]):
                return (
                    False,
                    f"op #{idx}: key {k!r} differs (JSON={jval!r}, IR={ir_attrs[k]!r})",
                )
    return True, "ok"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def resolve_candidate(
    run_dir: Path,
    candidate_id: str,
    *,
    allow_illegal: bool = False,
    selection_mode: str = "explicit",
    rationale: dict[str, Any] | None = None,
    write_outputs: bool = False,
) -> tuple[ResolvedCandidate, ResolverReport]:
    """Resolve ``candidate_id`` against ``02_graph_analysis/action_space.mlir``.

    When ``write_outputs`` is True, also emits:

    - ``02_graph_analysis/action_space_resolver_report.json``
    - ``03_recipe_planning/candidate_selection.json``
    - ``03_recipe_planning/selected_recipe_delta.mlir``

    Returns ``(ResolvedCandidate, ResolverReport)``. Raises a typed
    :class:`ResolverError` subclass on any failure.
    """
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    if not ga.is_dir():
        raise FileNotFoundError(
            f"02_graph_analysis/ missing under {run_dir}; run graph-analysis first"
        )

    mlir_path = ga / "action_space.mlir"
    decision_sites_path = ga / "decision_sites.json"
    candidate_actions_path = ga / "candidate_actions.json"
    llm_action_space_path = ga / "llm_action_space.json"

    for required in (mlir_path, decision_sites_path, candidate_actions_path, llm_action_space_path):
        if not required.exists():
            raise FileNotFoundError(f"required artifact missing: {required}")

    mlir_text = mlir_path.read_text(encoding="utf-8")
    actual_sha = "sha256:" + hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()

    decision_sites = _read_json(decision_sites_path)
    candidate_actions = _read_json(candidate_actions_path)
    llm_action_space = _read_json(llm_action_space_path)

    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})

    # 1. action_space_ir_sha256 chain across all three projections
    sha_links = {
        "decision_sites": decision_sites["source"]["action_space_ir_sha256"],
        "candidate_actions": candidate_actions["source"]["action_space_ir_sha256"],
        "llm_action_space": llm_action_space["source"]["action_space_ir_sha256"],
    }
    if any(s != actual_sha for s in sha_links.values()):
        _add(
            "json_projection_sha256_matches_ir",
            False,
            f"recomputed={actual_sha}; recorded={sha_links}",
        )
        if write_outputs:
            _emit_resolver_report(
                ga, candidate_id, "fail", checks
            )
        raise HashMismatchError(
            f"action_space_ir_sha256 mismatch — JSON projections do not match IR"
            f" (actual={actual_sha}, recorded={sha_links})"
        )
    _add("json_projection_sha256_matches_ir", True, f"sha256={actual_sha[:24]}...")

    # 2. candidate_id present in candidate_actions.json
    cand_json = next(
        (c for c in candidate_actions.get("candidates", []) if c["candidate_id"] == candidate_id),
        None,
    )
    if cand_json is None:
        _add("candidate_in_json_projection", False, candidate_id)
        if write_outputs:
            _emit_resolver_report(ga, candidate_id, "fail", checks)
        raise CandidateNotFoundError(
            f"candidate_id {candidate_id!r} not in candidate_actions.json"
        )
    _add("candidate_in_json_projection", True, "")

    # 3. candidate_id present in canonical IR (extract its block)
    try:
        ir_block_text, recipe_ir_lines = _extract_candidate_block(mlir_text, candidate_id)
    except CandidateNotFoundError:
        _add("candidate_in_canonical_ir", False, candidate_id)
        if write_outputs:
            _emit_resolver_report(ga, candidate_id, "fail", checks)
        raise
    _add("candidate_in_canonical_ir", True, "")

    # 4. recipe_delta in JSON matches IR
    delta_ok, delta_detail = _verify_recipe_delta_against_ir(
        cand_json["recipe_delta"], recipe_ir_lines
    )
    _add("recipe_delta_matches_ir", delta_ok, delta_detail)
    if not delta_ok:
        if write_outputs:
            _emit_resolver_report(ga, candidate_id, "fail", checks)
        raise RecipeDeltaMismatchError(
            f"recipe_delta cross-check failed for {candidate_id}: {delta_detail}"
        )

    # 5. legality gate
    legal = bool(cand_json["legality"]["ok"])
    if (not legal) and (not allow_illegal):
        _add(
            "legality_gate",
            False,
            cand_json["legality"].get("reason", "candidate is illegal and allow_illegal=False"),
        )
        if write_outputs:
            _emit_resolver_report(ga, candidate_id, "fail", checks)
        raise IllegalCandidateError(
            f"candidate {candidate_id!r} is illegal: "
            f"{cand_json['legality'].get('reason', '(no reason)')}"
        )
    _add("legality_gate", True, "ok" if legal else "allow_illegal=True")

    resolved = ResolvedCandidate(
        candidate_id=candidate_id,
        site_id=cand_json["site_id"],
        region_id=cand_json["region_id"],
        kind=cand_json["kind"],
        label=cand_json["label"],
        legality_ok=legal,
        legality_reason=cand_json["legality"].get("reason", ""),
        recipe_delta=list(cand_json["recipe_delta"]),
        cost_preview=dict(cand_json["cost_preview"]),
        evidence=dict(cand_json["evidence"]),
        source={
            "action_space_ir": str(mlir_path.relative_to(run_dir)),
            "action_space_ir_sha256": actual_sha,
            "candidate_actions": str(candidate_actions_path.relative_to(run_dir)),
            "llm_action_space": str(llm_action_space_path.relative_to(run_dir)),
        },
        ir_block_text=ir_block_text,
    )
    report = ResolverReport(overall="pass", candidate_id=candidate_id, checks=checks)

    if write_outputs:
        _emit_resolver_report(ga, candidate_id, "pass", checks)
        _write_candidate_selection(
            run_dir, resolved, selection_mode=selection_mode, rationale=rationale or {},
        )
        _write_selected_recipe_delta_mlir(run_dir, resolved)
    return resolved, report


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit_resolver_report(
    ga: Path, candidate_id: str, overall: str, checks: list[dict[str, Any]]
) -> Path:
    obj = {
        "schema_version": "action_space_resolver_report_v1",
        "candidate_id": candidate_id,
        "overall": overall,
        "generated_at_utc": _utcnow(),
        "checks": checks,
    }
    path = ga / "action_space_resolver_report.json"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_candidate_selection(
    run_dir: Path,
    resolved: ResolvedCandidate,
    *,
    selection_mode: str,
    rationale: dict[str, Any],
) -> Path:
    out_dir = run_dir / "03_recipe_planning"
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema_version": "candidate_selection_v1",
        "selected_candidate_id": resolved.candidate_id,
        "site_id": resolved.site_id,
        "region_id": resolved.region_id,
        "candidate_kind": resolved.kind,
        "label": resolved.label,
        "selection_mode": selection_mode,
        "selected_at_utc": _utcnow(),
        "source": dict(resolved.source),
        "legality": {"ok": resolved.legality_ok, "reason": resolved.legality_reason},
        "rationale": rationale or {
            "primary_reason": "",
            "evidence": [],
        },
        "recipe_delta": list(resolved.recipe_delta),
        "cost_preview": dict(resolved.cost_preview),
        "evidence": dict(resolved.evidence),
    }
    path = out_dir / "candidate_selection.json"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_selected_recipe_delta_mlir(
    run_dir: Path, resolved: ResolvedCandidate
) -> Path:
    out_dir = run_dir / "03_recipe_planning"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = (
        f"// Selected from {resolved.source['action_space_ir']}\n"
        f"// candidate_id = {resolved.candidate_id}\n"
        f"// action_space_ir_sha256 = {resolved.source['action_space_ir_sha256']}\n"
        f"{resolved.ir_block_text}\n"
    )
    path = out_dir / "selected_recipe_delta.mlir"
    path.write_text(text, encoding="utf-8")
    return path
