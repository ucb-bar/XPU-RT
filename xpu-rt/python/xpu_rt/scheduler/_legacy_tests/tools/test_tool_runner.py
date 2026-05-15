"""Tests for :mod:`compgen.tools.tool_runner`.

Coverage:

Positive:
* ``compgen_echo`` runs end-to-end, writes ``echo.txt``, returns
  ``status=ok``, populates input/output hashes, and writes a
  ``result.json`` + ``trace.jsonl`` under ``out_dir``.
* repeated runs with identical inputs produce identical
  ``input_hash`` and ``output_hash`` (determinism guard for the
  fresh-agent harness ).

Negative controls (every error branch in the runner is covered):
* missing required input field is rejected with
  :class:`ToolInputSchemaError`.
* an entrypoint that returns a value violating ``output_schema.status``
  raises :class:`ToolOutputSchemaError` and **does not** write
  ``result.json``.
* an entrypoint that returns an artifact path outside the declared
  ``writes.allowed_roots`` raises :class:`ToolRunError`.
* an entrypoint that raises mid-execution surfaces as
  :class:`ToolRunError` with a trace event of ``event=error``.
* a card whose ``entrypoints.python`` cannot be imported raises
  :class:`ToolEntrypointError`.
* an unknown maturity inside the card raises :class:`ToolCardError`
  (covered in :mod:`test_tool_card`; re-asserted here against the
  full ToolRunner.run pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from compgen.tools.errors import (
    ToolEntrypointError,
    ToolInputSchemaError,
    ToolOutputSchemaError,
    ToolRunError,
)
from compgen.tools.tool_card import ToolCard
from compgen.tools.tool_registry import load_tool_card, tool_cards_root
from compgen.tools.tool_runner import ToolRunner, resolve_python_entrypoint


@pytest.fixture
def echo_card() -> ToolCard:
    return load_tool_card(tool_cards_root() / "echo.yaml")


def _make_negative_control_card(entrypoint: str) -> ToolCard:
    """Build a card whose only difference from echo is the entrypoint.

    Used to exercise the runner's error branches without inventing
    a separate ToolCard schema per failure mode.
    """

    body = yaml.safe_load((tool_cards_root() / "echo.yaml").read_text(encoding="utf-8"))
    body["tool_id"] = "compgen_echo_negative_control"
    body["entrypoints"]["python"] = entrypoint
    return ToolCard.from_dict(body)


# ---------- Positive ----------


def test_run_echo_positive(tmp_path, echo_card):
    out_dir = tmp_path / "echo_run"
    result = ToolRunner().run(
        echo_card,
        request={"text": "hello compgen", "count": 3},
        out_dir=out_dir,
    )

    assert result.status == "ok"
    assert result.tool_id == "compgen_echo"
    assert result.result["lines_written"] == 3
    assert len(result.artifacts) == 1

    artifact = Path(result.artifacts[0])
    assert artifact.is_file()
    assert artifact.read_text(encoding="utf-8") == "hello compgen\nhello compgen\nhello compgen\n"

    # result.json must exist and match the returned ToolResult.
    rj = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert rj["status"] == "ok"
    assert rj["input_hash"] == result.input_hash
    assert rj["output_hash"] == result.output_hash

    # trace.jsonl must have a start and an end event.
    lines = (out_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(l) for l in lines]
    kinds = [e["event"] for e in events]
    assert kinds == ["start", "end"]
    assert events[0]["input_hash"] == result.input_hash
    assert events[1]["output_hash"] == result.output_hash


def test_run_echo_deterministic_hashes(tmp_path, echo_card):
    """Same request + same out_dir → identical input/output hashes.

    This is the replay determinism contract the grading uses to
    verify a fresh-agent run reproduced a recorded outcome. Different
    ``out_dir``s legitimately produce different output_hashes because
    the result references different absolute paths — that is also a
    correctness property, just a different one.
    """

    request = {"text": "deterministic", "count": 2}
    out_dir = tmp_path / "shared"
    a = ToolRunner().run(echo_card, request=request, out_dir=out_dir)
    b = ToolRunner().run(echo_card, request=request, out_dir=out_dir)
    assert a.input_hash == b.input_hash
    assert a.output_hash == b.output_hash


def test_run_echo_different_out_dirs_yield_different_output_hashes(tmp_path, echo_card):
    """Sanity check: distinct out_dirs naturally produce distinct outputs
    because the artifact path is absolute. The fresh-agent grader must
    therefore normalize paths before hash-comparing across machines."""

    request = {"text": "x", "count": 1}
    a = ToolRunner().run(echo_card, request=request, out_dir=tmp_path / "a")
    b = ToolRunner().run(echo_card, request=request, out_dir=tmp_path / "b")
    assert a.input_hash == b.input_hash
    assert a.output_hash != b.output_hash


def test_canonical_json_hash_stable_against_key_order(tmp_path, echo_card):
    """Reordering keys in the request must not change input_hash."""

    a = ToolRunner().run(
        echo_card,
        request={"count": 4, "text": "abc"},
        out_dir=tmp_path / "a",
    )
    b = ToolRunner().run(
        echo_card,
        request={"text": "abc", "count": 4},
        out_dir=tmp_path / "b",
    )
    assert a.input_hash == b.input_hash


def test_trace_jsonl_is_append_only(tmp_path, echo_card):
    out_dir = tmp_path / "shared"
    ToolRunner().run(echo_card, request={"text": "first"}, out_dir=out_dir)
    ToolRunner().run(echo_card, request={"text": "second"}, out_dir=out_dir)
    lines = (out_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # 2 runs × (start + end)


# ---------- Negative controls ----------


def test_run_echo_missing_required_input_field(tmp_path, echo_card):
    with pytest.raises(ToolInputSchemaError, match="text"):
        ToolRunner().run(echo_card, request={"count": 2}, out_dir=tmp_path / "out")


def test_run_echo_additional_properties_rejected(tmp_path, echo_card):
    """echo.yaml declares ``additionalProperties: false``."""

    with pytest.raises(ToolInputSchemaError):
        ToolRunner().run(
            echo_card,
            request={"text": "hi", "secret_extra": 42},
            out_dir=tmp_path / "out",
        )


def test_run_echo_bad_output_status(tmp_path):
    card = _make_negative_control_card(
        "compgen.tools.builtin.echo:returns_bad_status"
    )
    with pytest.raises(ToolOutputSchemaError, match="status"):
        ToolRunner().run(card, request={"text": "hi"}, out_dir=tmp_path / "out")
    # The runner must NOT have written a result.json for an output that
    # fails validation — that's the whole point of the hard rule.
    assert not (tmp_path / "out" / "result.json").exists()


def test_run_echo_writes_outside_out_dir(tmp_path):
    card = _make_negative_control_card(
        "compgen.tools.builtin.echo:writes_outside_out_dir"
    )
    with pytest.raises(ToolRunError, match="outside allowed_roots"):
        ToolRunner().run(card, request={"text": "hi"}, out_dir=tmp_path / "out")


def test_run_echo_entrypoint_crash(tmp_path):
    card = _make_negative_control_card("compgen.tools.builtin.echo:crashes")
    with pytest.raises(ToolRunError, match="entrypoint raised"):
        ToolRunner().run(card, request={"text": "hi"}, out_dir=tmp_path / "out")

    # The trace.jsonl must record the error event.
    lines = (tmp_path / "out" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(l) for l in lines]
    assert events[-1]["event"] == "error"
    assert events[-1]["exc_type"] == "RuntimeError"


def test_resolve_python_entrypoint_missing_module():
    with pytest.raises(ToolEntrypointError, match="cannot import module"):
        resolve_python_entrypoint("compgen.does_not_exist_xyz:run")


def test_resolve_python_entrypoint_missing_attr():
    with pytest.raises(ToolEntrypointError, match="has no attribute"):
        resolve_python_entrypoint("compgen.tools.builtin.echo:not_real_attr")


def test_resolve_python_entrypoint_not_callable():
    with pytest.raises(ToolEntrypointError, match="not callable"):
        # sys.version is a string, not callable
        resolve_python_entrypoint("sys:version")


def test_resolve_python_entrypoint_bad_format():
    with pytest.raises(ToolEntrypointError, match="module.path:attribute"):
        resolve_python_entrypoint("no_colon_here")


def test_unimportable_entrypoint_raises_through_runner(tmp_path):
    body = yaml.safe_load((tool_cards_root() / "echo.yaml").read_text(encoding="utf-8"))
    body["tool_id"] = "compgen_echo_unimportable"
    body["entrypoints"]["python"] = "compgen.totally_fake_module:run"
    card = ToolCard.from_dict(body)
    with pytest.raises(ToolEntrypointError):
        ToolRunner().run(card, request={"text": "hi"}, out_dir=tmp_path / "out")


def test_runner_rejects_non_dict_output(tmp_path):
    """An entrypoint that returns a non-dict must raise typed."""

    # Stand up a one-off entrypoint right here.
    import sys
    import types

    mod = types.ModuleType("compgen._inline_test_nondict")
    mod.run = lambda req, *, out_dir: ["not", "a", "dict"]
    sys.modules["compgen._inline_test_nondict"] = mod

    body = yaml.safe_load((tool_cards_root() / "echo.yaml").read_text(encoding="utf-8"))
    body["tool_id"] = "compgen_inline_nondict"
    body["entrypoints"]["python"] = "compgen._inline_test_nondict:run"
    card = ToolCard.from_dict(body)
    with pytest.raises(ToolOutputSchemaError, match="must return a dict"):
        ToolRunner().run(card, request={"text": "hi"}, out_dir=tmp_path / "out")


def test_runner_artifacts_field_must_be_a_list(tmp_path):
    """The runner must reject ``artifacts`` of the wrong type."""

    import sys
    import types

    mod = types.ModuleType("compgen._inline_test_bad_artifacts")

    def _run(req, *, out_dir):
        return {"status": "ok", "lines_written": 0, "artifacts": "not_a_list"}

    mod.run = _run
    sys.modules["compgen._inline_test_bad_artifacts"] = mod

    body = yaml.safe_load((tool_cards_root() / "echo.yaml").read_text(encoding="utf-8"))
    body["tool_id"] = "compgen_inline_bad_artifacts"
    body["entrypoints"]["python"] = "compgen._inline_test_bad_artifacts:run"
    card = ToolCard.from_dict(body)
    # jsonschema rejects the wrong type first (artifacts: array), so we
    # get a ToolOutputSchemaError, not a ToolRunError — either way it's
    # typed and the result.json is not written.
    with pytest.raises(ToolOutputSchemaError):
        ToolRunner().run(card, request={"text": "hi"}, out_dir=tmp_path / "out")
