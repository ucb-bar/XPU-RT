"""Acceptance tests for the prompt-engineering surface CompGen ships.

Verifies:

- ``agent_decision_request.json`` carries an ``agent_guidance`` block
  with the cost-column priority, disagreement-handling rules,
  rationale-field examples, response shape, and forbidden phrases.
- The block is byte-stable across reruns (deterministic; no
  measurement, no system state).
- ``cost_column_priority`` ranks the four post-M-12 cost columns
  in a defensible order.
- The candidate-selection skill content covers the post-M-12 evidence
  paths (M-21 m21_analytical_cost overlay, M-22 compiled_evidence
  overlay, kernel_calibration_status, bottleneck_classification_agreement).
- The llm-live ``build_prompt`` emits a system prompt that includes
  the same hard rules + cost-matrix interpretation rules.
- The forbidden_phrase_patterns block in the request matches the
  hard-coded forbidden patterns the M-14A validator regex rejects.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run(model: str, out_dir: Path) -> None:
    env = os.environ.copy()
    env.pop("COMPGEN_RUN_KERNELS", None)
    env.pop("COMPGEN_CALIBRATE_PROFILER", None)
    env.pop("COMPGEN_CALIBRATE_CANDIDATES", None)
    subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "agent-decision-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("pe") / "run"
    _run("merlin_mlp_wide", out)
    return out


def _request(run_dir: Path) -> dict:
    candidates = [
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json",
        run_dir / "agent_decision_request.json",
    ]
    for p in candidates:
        if p.exists():
            return _read(p)
    pytest.fail(
        f"agent_decision_request.json not found in {run_dir}"
    )
    raise AssertionError("unreachable")  # for type checkers


# --------------------------------------------------------------------------- #
# agent_guidance block exists and has the right shape
# --------------------------------------------------------------------------- #


def test_agent_guidance_present_in_request(run_dir: Path) -> None:
    req = _request(run_dir)
    assert "agent_guidance" in req, (
        "agent_decision_request must carry an agent_guidance block"
    )
    g = req["agent_guidance"]
    for field in (
        "guidance_version", "preamble",
        "cost_column_priority", "disagreement_handling",
        "rationale_field_examples", "forbidden_phrase_patterns",
        "preferred_neutral_phrases", "response_shape",
        "selection_modes_supported", "honest_non_claims",
    ):
        assert field in g, f"agent_guidance missing field {field}"


def test_cost_column_priority_ranks_four_post_m12_columns(
    run_dir: Path,
) -> None:
    g = _request(run_dir)["agent_guidance"]
    cols = g["cost_column_priority"]
    assert len(cols) >= 4, (
        "cost_column_priority must rank at least 4 columns"
    )
    columns = [c["column"] for c in cols]
    # Top-priority: real compiled measurement.
    assert columns[0] == "compiled_evidence"
    # Required columns must appear somewhere in the priority list.
    for required in (
        "compiled_evidence",
        "calibration_delta",
        "m21_analytical_cost",
        "calibration",
        "static_relative_cost",
    ):
        assert required in columns, (
            f"cost_column_priority missing required column {required}"
        )
    # Ranks are sequential from 1.
    ranks = [c["rank"] for c in cols]
    assert ranks == sorted(ranks), "cost_column_priority not rank-sorted"
    assert ranks[0] == 1


def test_disagreement_handling_covers_canonical_signals(
    run_dir: Path,
) -> None:
    g = _request(run_dir)["agent_guidance"]
    signals = [d["signal"] for d in g["disagreement_handling"]]
    joined = " | ".join(signals)
    # Required disagreement signals.
    assert "bottleneck_classification_agreement" in joined
    assert "predicted_vs_gpu_ratio" in joined
    assert "partial_kernel_calibration" in joined
    assert "not_kernel_calibrated" in joined


def test_rationale_field_examples_include_post_m21_m22_paths(
    run_dir: Path,
) -> None:
    g = _request(run_dir)["agent_guidance"]
    examples = g["rationale_field_examples"]
    # Must include at least one M-21 path and one M-22 path.
    has_m21 = any(
        "m21_analytical_cost" in p
        for p in examples
    )
    has_m22 = any(
        "compiled_evidence" in p
        for p in examples
    )
    assert has_m21, "rationale_field_examples missing m21_analytical_cost path"
    assert has_m22, "rationale_field_examples missing compiled_evidence path"


def test_forbidden_phrase_patterns_match_validator(run_dir: Path) -> None:
    """The forbidden phrases shown to the agent in agent_guidance must
    match the regex patterns the M-14A validator actually rejects."""
    g = _request(run_dir)["agent_guidance"]
    declared = set(g["forbidden_phrase_patterns"])
    # These are the patterns the validator's regex actually rejects;
    # they must all be visible to the agent up-front.
    must_include = {
        "verified correct",
        "guaranteed correct",
        "bit equivalent to eager",
        "measured fastest",
        "benchmarked",
        "profiled",
        "executed faster",
    }
    missing = must_include - declared
    assert not missing, (
        f"agent_guidance.forbidden_phrase_patterns missing: {sorted(missing)}"
    )


def test_response_shape_documents_required_fields(run_dir: Path) -> None:
    g = _request(run_dir)["agent_guidance"]
    shape = g["response_shape"]
    assert shape["schema_version"] == "agent_decision_response_v1"
    assert "selected_candidate_id" in shape
    assert "rationale" in shape
    rat = shape["rationale"]
    assert "summary" in rat
    assert "evidence" in rat


# --------------------------------------------------------------------------- #
# Determinism / byte stability
# --------------------------------------------------------------------------- #


def test_agent_guidance_is_byte_stable_across_reruns(
    run_dir: Path,
) -> None:
    """Building a fresh agent_decision_request from the existing
    on-disk artifacts must produce a byte-identical agent_guidance
    block (it's a deterministic constant function)."""
    from compgen.graph_compilation.agent_decision import (
        _build_agent_guidance,
    )
    g1 = _build_agent_guidance()
    g2 = _build_agent_guidance()
    assert json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True)


# --------------------------------------------------------------------------- #
# Sources block now carries optional cost-evidence references
# --------------------------------------------------------------------------- #


def test_sources_block_lists_optional_evidence_artifacts(
    run_dir: Path,
) -> None:
    req = _request(run_dir)
    sources = req["sources"]
    for required_key in (
        "analytical_cost_report",      # M-21 (always-on)
        "readiness_matrix",            # M-17.1
        "hardware_resource_report",    # M-17.1
        "compiled_bottleneck_report",  # M-22 (may be null when kernels off)
        "region_compiled_differential_report",  # M-20
        "calibration_report",          # M-18 (may be null)
        "candidate_calibration_report",  # M-18.3 (may be null)
    ):
        assert required_key in sources, (
            f"agent_decision_request.sources missing key {required_key}"
        )
    # M-21 analytical_cost is always-on, so it should be non-null.
    assert sources["analytical_cost_report"] is not None, (
        "M-21 analytical_cost_report should be non-null (always-on)"
    )


# --------------------------------------------------------------------------- #
# llm-live build_prompt includes the cost-matrix rules
# --------------------------------------------------------------------------- #


def test_llm_live_build_prompt_includes_cost_matrix_rules() -> None:
    from compgen.graph_compilation.llm_live_provider import build_prompt

    fake_request = {
        "schema_version": "agent_decision_request_v1",
        "candidate_ids_allowed": ["cand_x"],
        "agent_guidance": {"guidance_version": 1},
    }
    fake_view = {"regions": []}
    prompt = build_prompt(request=fake_request, llm_graph_view=fake_view)

    # Hard rules surface explicitly.
    assert "EXACTLY one candidate_id" in prompt
    assert "Do NOT invent candidate IDs" in prompt
    assert "candidate_ids_allowed" in prompt

    # Cost matrix rules surface explicitly.
    assert "compiled_evidence" in prompt
    assert "calibration_delta" in prompt
    assert "m21_analytical_cost" in prompt
    assert "static_relative_cost" in prompt

    # Disagreement-handling rules surface.
    assert "bottleneck_classification_agreement" in prompt
    assert "predicted_vs_gpu_ratio" in prompt
    assert "kernel_calibration_status" in prompt

    # Forbidden phrases listed.
    for forbidden in (
        "verified correct", "measured fastest", "benchmarked",
    ):
        assert forbidden in prompt


# --------------------------------------------------------------------------- #
# Skill content covers post-M-12 evidence
# --------------------------------------------------------------------------- #


_SKILL_PATH = (
    REPO_ROOT / ".claude" / "skills" / "compgen-candidate-selection"
    / "SKILL.md"
)


def test_candidate_selection_skill_mentions_m21_m22_evidence() -> None:
    src = _SKILL_PATH.read_text(encoding="utf-8")

    # Cost-matrix section must exist.
    assert "How to read the cost matrix" in src

    # Each post-M-12 cost column is named.
    for col in (
        "compiled_evidence",
        "m21_analytical_cost",
        "calibration_delta",
        "static_relative_cost",
        "kernel_calibration_status",
        "bottleneck_classification_agreement",
    ):
        assert col in src, f"skill missing reference to {col}"

    # The skill names the new optional artifacts.
    for art in (
        "analytical_cost",
        "compiled_bottleneck",
        "kernel_execution",
        "candidate_calibration",
    ):
        assert art in src, f"skill missing artifact reference {art}"


def test_candidate_selection_skill_keeps_forbidden_phrases() -> None:
    """The skill must still explicitly forbid correctness/perf claims —
    even after the cost-matrix update, this section is load-bearing."""
    src = _SKILL_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "verified correct",
        "measured fastest",
        "benchmarked",
    ):
        assert forbidden in src, (
            f"skill no longer mentions forbidden phrase {forbidden!r}"
        )
