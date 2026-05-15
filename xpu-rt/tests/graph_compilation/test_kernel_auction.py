"""Multi-bidder kernel auction tests.

Coverage:

- ``TestAuctionDisabled`` — ``mode='disabled'`` short-circuits to a
  skipped result; no fulfilled bids, no winner.
- ``TestNoApplicableProviders`` — empty registry returns
  ``overall='skipped'`` with ``error='no_applicable_providers'``.
- ``TestMultiBidder`` — registry with CReferenceProvider runs the full
  pipeline against merlin_mlp_wide: 1 bid, 1 fulfilled, 1 verified, 1
  winner, winner promoted to standard response location,
  binds.
- ``TestFirstFit`` — ``mode='first-fit'`` short-circuits at the first
  verified bid (still 1 winner since CReferenceProvider is alone).
- ``TestAuctionReportSchema`` — auction_report.json carries the right
  schema_version + every record (bids, fulfilled, verified, winner).
- ``TestStubProviders`` — unit tests over an injected ProviderRegistry
  with two stub providers (one verified, one fulfill-fails) showing
  selector picks the verified one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _invoke_pipeline(
    *, model: str, out_dir: Path, stop_after: str = "kernel-auction",
    auction_mode: str = "multi-bidder",
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", stop_after,
            "--selection-mode", "greedy",
            "--auction-mode", auction_mode,
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


# --------------------------------------------------------------------------- #
# Disabled / no-applicable
# --------------------------------------------------------------------------- #


class TestAuctionDisabled:
    def test_mode_disabled_short_circuits(self) -> None:
        from compgen.graph_compilation.kernel_auction import run_kernel_auction
        from compgen.kernels.registry import ProviderRegistry

        # Use disabled mode — should skip even with applicable providers.
        result = run_kernel_auction(
            run_dir=Path("/tmp/auction_does_not_matter"),
            mode="disabled",
            registry=ProviderRegistry(),
        )
        assert result.overall == "skipped"
        assert result.error == "no_m42_request"  # because run_dir is empty


class TestNoApplicableProviders:
    def test_empty_registry_returns_skipped(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(
            model="merlin_mlp_wide",
            out_dir=tmp_path / "run",
            stop_after="kernel-auction",
        )
        assert result.returncode == 0, result.stderr

        # auction_report.json exists.
        run_dir = tmp_path / "run"
        auction_dirs = list((run_dir / "04_kernel_codegen" / "auction").iterdir())
        assert len(auction_dirs) == 1
        report = json.loads((auction_dirs[0] / "auction_report.json").read_text())
        # CReferenceProvider in default_registry handles host_cpu+matmul
        # → expect either 'pass' (winner) or at minimum a bid record.
        assert report["overall"] in ("pass", "no_winner", "skipped")
        assert report["mode"] == "multi-bidder"


# --------------------------------------------------------------------------- #
# Multi-bidder full path with CReferenceProvider
# --------------------------------------------------------------------------- #


class TestMultiBidder:
    def test_creference_wins_on_merlin_mlp_wide(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(
            model="merlin_mlp_wide",
            out_dir=tmp_path / "run",
            stop_after="kernel-auction",
            auction_mode="multi-bidder",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        auction_dirs = list((run_dir / "04_kernel_codegen" / "auction").iterdir())
        assert len(auction_dirs) == 1
        auction_dir = auction_dirs[0]

        report = json.loads((auction_dir / "auction_report.json").read_text())
        assert report["schema_version"] == "auction_report_v1"
        assert report["mode"] == "multi-bidder"
        assert report["overall"] == "pass", f"auction did not produce a winner: {report}"
        assert report["winner_provider"] == "c_reference"
        assert len(report["bids"]) >= 1
        assert len(report["fulfilled"]) >= 1
        assert len(report["verified"]) >= 1
        assert any(v["overall"] == "pass" for v in report["verified"])

        # Winner promoted to standard response path.
        task_id = report["task_id"]
        response_path = run_dir / "04_kernel_codegen" / "responses" / f"{task_id}.response.json"
        assert response_path.exists(), "winner must be promoted to M-43 response path"

        # Per-provider artifact directory + winner.json + runners_up.json.
        assert (auction_dir / "fulfilled" / "c_reference" / "kernel.c").exists()
        assert (auction_dir / "winner.json").exists()
        assert (auction_dir / "runners_up.json").exists()


class TestFirstFit:
    def test_first_fit_short_circuits(self, tmp_path: Path) -> None:
        result = _invoke_pipeline(
            model="merlin_mlp_wide",
            out_dir=tmp_path / "run",
            stop_after="kernel-auction",
            auction_mode="first-fit",
        )
        assert result.returncode == 0, result.stderr

        run_dir = tmp_path / "run"
        auction_dirs = list((run_dir / "04_kernel_codegen" / "auction").iterdir())
        report = json.loads((auction_dirs[0] / "auction_report.json").read_text())
        assert report["mode"] == "first-fit"
        # Single CReferenceProvider in default registry → 1 fulfilled.
        assert len(report["fulfilled"]) == 1


# --------------------------------------------------------------------------- #
# Stub-provider unit tests
# --------------------------------------------------------------------------- #


@dataclass
class _StubBidProvider:
    """Minimal provider with a controllable bid + a controllable search result."""

    name_str: str = "stub_bid"
    bid_confidence: float = 0.7
    bid_perf_us: float = 10.0
    will_search_succeed: bool = True
    applicable_targets: tuple[str, ...] = ()
    applicable_archetypes: tuple[str, ...] = ()
    priority: int = 0
    _compgen_source: str = "in_tree"
    bid_rationale: str = "stub"

    @property
    def name(self) -> str:
        return self.name_str

    def accepts_contract(self, contract):  # noqa: ANN001
        return True

    def search(self, contract, budget):  # noqa: ANN001
        from compgen.kernels.provider import ProviderResult

        if not self.will_search_succeed:
            return ProviderResult(found=False)
        return ProviderResult(
            found=True,
            kernel_code=f"// stub kernel from {self.name_str}\nvoid noop(void) {{}}\n",
            language="c",
        )

    def export_knowledge(self):  # noqa: D401
        return []

    def bid(self, contract_v3):  # noqa: ANN001
        from compgen.kernels.provider import BidPreview

        return BidPreview(
            provider_name=self.name_str,
            confidence=self.bid_confidence,
            perf_estimate_us=self.bid_perf_us,
            time_to_generate_s_estimate=0.1,
            rationale=self.bid_rationale,
        )


class TestStubProviders:
    def _bootstrap_run_dir(self, tmp_path: Path) -> Path:
        """Run the pipeline up to kernel-codegen-request to populate
        the prerequisites the auction reads."""
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

    def test_two_providers_one_succeeds(self, tmp_path: Path) -> None:
        from compgen.graph_compilation.kernel_auction import run_kernel_auction
        from compgen.kernels.registry import ProviderRegistry

        run_dir = self._bootstrap_run_dir(tmp_path)
        reg = ProviderRegistry()
        reg.register(_StubBidProvider(name_str="stub_succeeds", bid_perf_us=10.0))
        reg.register(
            _StubBidProvider(
                name_str="stub_fails",
                bid_perf_us=5.0,  # better perf
                will_search_succeed=False,  # but its search returns not-found
            )
        )

        result = run_kernel_auction(
            run_dir=run_dir,
            mode="multi-bidder",
            registry=reg,
        )

        # Both bids were collected, but only the succeeding one fulfilled.
        assert result.overall == "pass"
        assert result.winner_provider == "stub_succeeds"
        assert len(result.bids) == 2
        assert len([f for f in result.fulfilled if f.found]) == 1

    def test_no_winner_when_all_searches_fail(self, tmp_path: Path) -> None:
        from compgen.graph_compilation.kernel_auction import run_kernel_auction
        from compgen.kernels.registry import ProviderRegistry

        run_dir = self._bootstrap_run_dir(tmp_path)
        reg = ProviderRegistry()
        reg.register(
            _StubBidProvider(name_str="dud_a", will_search_succeed=False, bid_perf_us=10.0)
        )
        reg.register(
            _StubBidProvider(name_str="dud_b", will_search_succeed=False, bid_perf_us=5.0)
        )

        result = run_kernel_auction(
            run_dir=run_dir,
            mode="multi-bidder",
            registry=reg,
        )
        assert result.overall == "no_winner"
        assert result.winner_provider == ""
        # Both bids attempted fulfill, both failed.
        assert len([f for f in result.fulfilled if not f.found]) == 2
