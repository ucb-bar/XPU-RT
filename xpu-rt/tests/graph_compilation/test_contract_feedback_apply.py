"""Two-tier contract_feedback re-entry tests.

Coverage:

- ``TestInferKind`` — heuristic ``kind`` derivation from ``field``.
- ``TestClassifyFeedback`` — allowlisted / non-allowlisted split,
  including provider-supplied ``kind`` precedence over heuristics.
- ``TestRecipeIRProposal`` — five allowlisted kinds each translate
  to the right Recipe-IR op + args.
- ``TestAuctionIntegration`` — end-to-end on merlin_mlp_wide:
  contract_feedback.json lands with empty buckets when
  CReferenceProvider emits no feedback; run-wide aggregate file
  is created.
- ``TestAuctionWithSyntheticFeedback`` — inject a stub provider
  whose search returns layout_swap + dtype_widen + a non-allowlisted
  entry; auction routes them into the right buckets and emits two
  proposals.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Unit — _infer_kind
# --------------------------------------------------------------------------- #


class TestInferKind:
    def test_provider_supplied_kind_wins(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import _infer_kind
        from compgen.kernels.provider import ContractFeedback

        fb = ContractFeedback(field="dtype", kind="custom_kind")
        assert _infer_kind(fb) == "custom_kind"

    def test_layout_field_infers_layout_swap(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import _infer_kind
        from compgen.kernels.provider import ContractFeedback

        assert _infer_kind(ContractFeedback(field="layout")) == "layout_swap"
        assert _infer_kind(ContractFeedback(field="io.input_layout")) == "layout_swap"

    def test_dtype_field_infers_dtype_widen(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import _infer_kind
        from compgen.kernels.provider import ContractFeedback

        assert _infer_kind(ContractFeedback(field="dtype")) == "dtype_widen"

    def test_accumulator_takes_precedence_over_dtype(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import _infer_kind
        from compgen.kernels.provider import ContractFeedback

        fb = ContractFeedback(field="numerics.accumulator_dtype")
        assert _infer_kind(fb) == "accumulator_widen"

    def test_align_field_infers_alignment_request(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import _infer_kind
        from compgen.kernels.provider import ContractFeedback

        assert _infer_kind(ContractFeedback(field="alignment_bytes")) == "alignment_request"

    def test_unknown_field_returns_empty(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import _infer_kind
        from compgen.kernels.provider import ContractFeedback

        assert _infer_kind(ContractFeedback(field="random_thing")) == ""
        assert _infer_kind(ContractFeedback()) == ""


# --------------------------------------------------------------------------- #
# Unit — classify_feedback
# --------------------------------------------------------------------------- #


class TestClassifyFeedback:
    def test_allowlisted_and_non_allowlisted_split(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import classify_feedback
        from compgen.kernels.provider import ContractFeedback

        feedbacks = [
            ContractFeedback(field="layout", suggested_value="row_major"),
            ContractFeedback(field="something_else"),
            ContractFeedback(kind="dtype_widen", suggested_value="f32"),
            ContractFeedback(kind="invent_new_pattern"),
        ]
        a, na = classify_feedback(provider_name="p", feedbacks=feedbacks)
        assert {x.kind for x in a} == {"layout_swap", "dtype_widen"}
        assert {x.kind for x in na} == {"", "invent_new_pattern"}

    def test_provider_name_carried(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import classify_feedback
        from compgen.kernels.provider import ContractFeedback

        a, _ = classify_feedback(
            provider_name="my_provider",
            feedbacks=[ContractFeedback(kind="layout_swap")],
        )
        assert a[0].provider_name == "my_provider"


# --------------------------------------------------------------------------- #
# Unit — to_recipe_ir_proposal
# --------------------------------------------------------------------------- #


class TestRecipeIRProposal:
    def test_layout_swap_yields_set_layout(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import (
            ClassifiedFeedback,
            to_recipe_ir_proposal,
        )
        from compgen.kernels.provider import ContractFeedback

        entry = ClassifiedFeedback(
            provider_name="p", kind="layout_swap", is_allowlisted=True,
            feedback=ContractFeedback(
                field="io.inputs[1].layout",
                suggested_value="row_major",
                reason="row-major B is faster for K>=64",
                applies_when="K >= 64",
                measured_gain=0.4,
                kind="layout_swap",
            ),
        )
        proposal = to_recipe_ir_proposal(entry)
        assert proposal.op == "SetLayout"
        assert proposal.args == {
            "target_field": "io.inputs[1].layout",
            "new_layout": "row_major",
        }
        assert proposal.applies_when == "K >= 64"
        assert proposal.measured_gain == 0.4

    def test_alignment_request_parses_int(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import (
            ClassifiedFeedback,
            to_recipe_ir_proposal,
        )
        from compgen.kernels.provider import ContractFeedback

        entry = ClassifiedFeedback(
            provider_name="p", kind="alignment_request", is_allowlisted=True,
            feedback=ContractFeedback(
                field="io.inputs[0].alignment_bytes",
                suggested_value="128",
                kind="alignment_request",
            ),
        )
        p = to_recipe_ir_proposal(entry)
        assert p.op == "SetAlignment"
        assert p.args["new_alignment_bytes"] == 128

    def test_alignment_request_invalid_falls_to_zero(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import (
            ClassifiedFeedback,
            to_recipe_ir_proposal,
        )
        from compgen.kernels.provider import ContractFeedback

        entry = ClassifiedFeedback(
            provider_name="p", kind="alignment_request", is_allowlisted=True,
            feedback=ContractFeedback(
                field="x", suggested_value="not_an_int",
                kind="alignment_request",
            ),
        )
        p = to_recipe_ir_proposal(entry)
        assert p.args["new_alignment_bytes"] == 0

    def test_fast_math_opt_in_yields_enable_true(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import (
            ClassifiedFeedback,
            to_recipe_ir_proposal,
        )
        from compgen.kernels.provider import ContractFeedback

        entry = ClassifiedFeedback(
            provider_name="p", kind="fast_math_opt_in", is_allowlisted=True,
            feedback=ContractFeedback(kind="fast_math_opt_in"),
        )
        p = to_recipe_ir_proposal(entry)
        assert p.op == "EnableFastMath"
        assert p.args == {"enable": True}

    def test_non_allowlisted_raises(self) -> None:
        from compgen.graph_compilation.contract_feedback_apply import (
            ClassifiedFeedback,
            to_recipe_ir_proposal,
        )
        from compgen.kernels.provider import ContractFeedback

        entry = ClassifiedFeedback(
            provider_name="p", kind="random_kind", is_allowlisted=False,
            feedback=ContractFeedback(kind="random_kind"),
        )
        with pytest.raises(ValueError, match="non-allowlisted"):
            to_recipe_ir_proposal(entry)


# --------------------------------------------------------------------------- #
# Auction integration
# --------------------------------------------------------------------------- #


def _invoke_pipeline(*, model: str, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "kernel-auction",
            "--selection-mode", "greedy",
            "--auction-mode", "multi-bidder",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


class TestAuctionIntegration:
    def test_empty_feedback_still_writes_artifacts(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(
            model="merlin_mlp_wide", out_dir=tmp_path / "run",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        # Auction-local file.
        auction_dirs = list((run_dir / "04_kernel_codegen" / "auction").iterdir())
        assert len(auction_dirs) == 1
        feedback_path = auction_dirs[0] / "contract_feedback.json"
        assert feedback_path.exists()
        body = json.loads(feedback_path.read_text())
        assert body["schema_version"] == "auction_contract_feedback_v1"
        # CReferenceProvider doesn't emit feedback today.
        assert body["counts"]["total"] == 0
        assert body["allowlisted"] == []
        assert body["non_allowlisted"] == []
        assert body["proposals"] == []

        # Run-wide aggregate file.
        agg_path = run_dir / "04_kernel_codegen" / "contract_feedback_proposals.json"
        assert agg_path.exists()
        agg = json.loads(agg_path.read_text())
        assert agg["schema_version"] == "contract_feedback_proposals_v1"
        assert len(agg["entries"]) == 1
        assert agg["entries"][0]["proposals"] == []


# --------------------------------------------------------------------------- #
# Auction with synthetic provider that emits real feedback
# --------------------------------------------------------------------------- #


@dataclass
class _FeedbackEmittingProvider:
    """Stub that emits a mix of allowlisted + non-allowlisted feedback."""

    name_str: str = "feedback_stub"
    applicable_targets: tuple[str, ...] = ("host_cpu",)
    applicable_archetypes: tuple[str, ...] = ("compute_tiled",)
    priority: int = 10
    _compgen_source: str = "in_tree"

    @property
    def name(self) -> str:
        return self.name_str

    def accepts_contract(self, contract):  # noqa: ANN001
        return True

    def search(self, contract, budget):  # noqa: ANN001
        from compgen.kernels.provider import ContractFeedback, ProviderResult

        return ProviderResult(
            found=True,
            kernel_code="// stub kernel\nvoid noop(void) {}\n",
            language="c",
            contract_feedback=[
                ContractFeedback(
                    kind="layout_swap",
                    field="io.inputs[1].layout",
                    current_value="row_major",
                    suggested_value="column_major",
                    reason="col-major B is 1.4x faster",
                    applies_when="K >= 64",
                    measured_gain=0.4,
                ),
                ContractFeedback(
                    kind="dtype_widen",
                    field="io.inputs[0].dtype",
                    current_value="f16",
                    suggested_value="bf16",
                    reason="bf16 hits Tensor Cores",
                    measured_gain=0.6,
                ),
                ContractFeedback(
                    kind="some_unknown_recommendation",
                    field="z",
                    suggested_value="??",
                    reason="provider muses about something",
                ),
            ],
        )

    def export_knowledge(self):  # noqa: D401
        return []

    def bid(self, contract_v3):  # noqa: ANN001
        from compgen.kernels.provider import BidPreview

        return BidPreview(
            provider_name=self.name_str,
            confidence=0.9,
            perf_estimate_us=1.0,
            time_to_generate_s_estimate=0.1,
            rationale="feedback stub",
        )


class TestAuctionWithSyntheticFeedback:
    def _bootstrap_run_dir(self, tmp_path: Path) -> Path:
        run_dir = tmp_path / "run"
        result = subprocess.run(
            [
                sys.executable, "-m", "compgen.graph_compilation", "run",
                "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
                "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
                "--out", str(run_dir),
                "--stop-after", "kernel-codegen-request",
                "--selection-mode", "greedy",
                "--auction-mode", "disabled",
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        return run_dir

    def test_provider_feedback_routed_into_buckets(self, tmp_path: Path) -> None:
        from compgen.graph_compilation.kernel_auction import run_kernel_auction
        from compgen.kernels.registry import ProviderRegistry

        run_dir = self._bootstrap_run_dir(tmp_path)
        reg = ProviderRegistry()
        reg.register(_FeedbackEmittingProvider())
        run_kernel_auction(run_dir=run_dir, mode="multi-bidder", registry=reg)

        # contract_feedback.json carries the buckets.
        auction_dirs = list((run_dir / "04_kernel_codegen" / "auction").iterdir())
        assert len(auction_dirs) == 1
        body = json.loads((auction_dirs[0] / "contract_feedback.json").read_text())
        assert body["counts"]["allowlisted"] == 2
        assert body["counts"]["non_allowlisted"] == 1
        assert body["counts"]["proposals"] == 2

        kinds = {e["kind"] for e in body["allowlisted"]}
        assert kinds == {"layout_swap", "dtype_widen"}

        proposals = body["proposals"]
        ops = {p["op"] for p in proposals}
        assert ops == {"SetLayout", "WidenDtype"}

        # Aggregate file mirrors the per-task entry.
        agg = json.loads(
            (run_dir / "04_kernel_codegen" / "contract_feedback_proposals.json").read_text()
        )
        assert len(agg["entries"]) == 1
        assert len(agg["entries"][0]["proposals"]) == 2
        assert len(agg["entries"][0]["non_allowlisted_advisory"]) == 1
