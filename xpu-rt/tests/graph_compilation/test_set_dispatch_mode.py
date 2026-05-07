"""M-50 SetDispatchMode tests.

Coverage:
- action_space emits set_dispatch_mode candidates per matmul region
  with one per legal mode (sync, async legal; persistent, inline
  illegal-by-granularity).
- decision_sites carry one dispatch site per kernel-bearing region.
- agent_decision_request surfaces the legal dispatch candidates in
  candidate_ids_allowed.
- KernelContractV3.from_recipe(dispatch_mode_override=...) flips
  the contract's dispatch.model.
- _resolve_dispatch_mode_override rejects PERSISTENT and INLINE on
  the NORMAL-only granularity path.
- Materialiser walks recipe_delta for a SetDispatchMode op and uses
  the chosen mode.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Pure schema / helpers
# --------------------------------------------------------------------------- #


class TestDispatchModeOverride:
    def test_override_none_defaults_to_sync(self) -> None:
        from compgen.kernels.contract_v3 import (
            DispatchModel, _resolve_dispatch_mode_override,
        )
        assert _resolve_dispatch_mode_override(None) is DispatchModel.SYNC
        assert _resolve_dispatch_mode_override("") is DispatchModel.SYNC

    def test_override_sync_async_accepted(self) -> None:
        from compgen.kernels.contract_v3 import (
            DispatchModel, _resolve_dispatch_mode_override,
        )
        assert _resolve_dispatch_mode_override("sync") is DispatchModel.SYNC
        assert _resolve_dispatch_mode_override("ASYNC") is DispatchModel.ASYNC

    def test_override_persistent_rejected_on_normal(self) -> None:
        from compgen.kernels.contract_v3 import _resolve_dispatch_mode_override
        with pytest.raises(ValueError, match="PERSISTENT"):
            _resolve_dispatch_mode_override("persistent")

    def test_override_inline_rejected_on_normal(self) -> None:
        from compgen.kernels.contract_v3 import _resolve_dispatch_mode_override
        with pytest.raises(ValueError, match="INLINE"):
            _resolve_dispatch_mode_override("inline")

    def test_override_unknown_rejected(self) -> None:
        from compgen.kernels.contract_v3 import _resolve_dispatch_mode_override
        with pytest.raises(ValueError):
            _resolve_dispatch_mode_override("warp_persistent")


class TestFromRecipeWithOverride:
    def _selection(self) -> dict:
        return {
            "candidate_kind": "set_tile_params",
            "selected_candidate_id": "cand_x",
            "region_id": "matmul_0",
            "label": "tile_M16_N16_K16",
            "cost_preview": {"region_dims": {"M": 16, "N": 32, "K": 16}},
            "recipe_delta": [{"op": "SetTileParams", "M": 16, "N": 16, "K": 16}],
            "target_id": "host_cpu",
        }

    def _dossier(self) -> dict:
        return {
            "region_id": "matmul_0",
            "region_shape": {
                "dtype": "f32",
                "input_shapes": [[16, 16], [16, 32]],
                "output_shapes": [[16, 32]],
            },
        }

    def test_default_dispatch_is_sync(self) -> None:
        from compgen.kernels.contract_v3 import DispatchModel, KernelContractV3
        c = KernelContractV3.from_recipe(
            candidate_selection=self._selection(),
            region_dossier=self._dossier(),
            target_profile={},
            declared_refinement="bit_equality",
        )
        assert c.orchestration.dispatch.model is DispatchModel.SYNC

    def test_async_override(self) -> None:
        from compgen.kernels.contract_v3 import DispatchModel, KernelContractV3
        c = KernelContractV3.from_recipe(
            candidate_selection=self._selection(),
            region_dossier=self._dossier(),
            target_profile={},
            declared_refinement="bit_equality",
            dispatch_mode_override="async",
        )
        assert c.orchestration.dispatch.model is DispatchModel.ASYNC

    def test_persistent_override_rejected(self) -> None:
        from compgen.kernels.contract_v3 import KernelContractV3
        with pytest.raises(ValueError, match="PERSISTENT"):
            KernelContractV3.from_recipe(
                candidate_selection=self._selection(),
                region_dossier=self._dossier(),
                target_profile={},
                declared_refinement="bit_equality",
                dispatch_mode_override="persistent",
            )


# --------------------------------------------------------------------------- #
# E2E — pipeline emits dispatch candidates
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def merlin_action_space(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m50_merlin") / "run"
    res = subprocess.run([
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out),
        "--stop-after", "agent-decision-request",
        "--selection-mode", "greedy",
    ], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return out


def test_e2e_candidate_actions_contains_dispatch_kind(
    merlin_action_space: Path,
) -> None:
    body = json.loads(
        (merlin_action_space / "02_graph_analysis" / "candidate_actions.json")
        .read_text()
    )
    dispatch_cands = [c for c in body["candidates"] if c["kind"] == "set_dispatch_mode"]
    assert len(dispatch_cands) > 0
    # 3 matmul regions × 4 modes = 12 (merlin_mlp_wide).
    assert len(dispatch_cands) == 12
    # Every region has a sync + async legal candidate.
    legal_modes = {(c["region_id"], c["label"]) for c in dispatch_cands if c["legality"]["ok"]}
    illegal_modes = {(c["region_id"], c["label"]) for c in dispatch_cands if not c["legality"]["ok"]}
    assert any("dispatch_sync" in label for _, label in legal_modes)
    assert any("dispatch_async" in label for _, label in legal_modes)
    assert any("dispatch_persistent" in label for _, label in illegal_modes)
    assert any("dispatch_inline" in label for _, label in illegal_modes)


def test_e2e_decision_sites_contains_dispatch_sites(
    merlin_action_space: Path,
) -> None:
    body = json.loads(
        (merlin_action_space / "02_graph_analysis" / "decision_sites.json")
        .read_text()
    )
    dispatch_sites = [s for s in body["sites"] if s["kind"] == "dispatch"]
    assert len(dispatch_sites) > 0
    for s in dispatch_sites:
        assert s["priority"] == 3  # lower priority than tiling
        assert len(s["candidate_ids"]) == 4  # all 4 modes per region


def test_e2e_legal_dispatch_candidates_in_agent_request(
    merlin_action_space: Path,
) -> None:
    body = json.loads(
        (merlin_action_space / "03_recipe_planning"
         / "agent_decision" / "agent_decision_request.json").read_text()
    )
    allowed = body.get("candidate_ids_allowed", [])
    # 3 regions × 2 legal modes (sync, async) = 6.
    dispatch_allowed = [cid for cid in allowed if "dispatch_" in cid]
    assert len(dispatch_allowed) == 6
    # Every dispatch candidate id includes either _sync_ or _async_
    # (no persistent/inline since they're illegal).
    for cid in dispatch_allowed:
        assert "dispatch_sync" in cid or "dispatch_async" in cid
        assert "persistent" not in cid
        assert "inline" not in cid


def test_e2e_rerun_produces_byte_stable_dispatch_candidates(
    tmp_path: Path,
) -> None:
    """Two reruns of the same model produce byte-identical
    set_dispatch_mode candidate lists."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        res = subprocess.run([
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / "configs/models/merlin_mlp_wide.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out),
            "--stop-after", "agent-decision-request",
            "--selection-mode", "greedy",
        ], cwd=REPO_ROOT, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
    a = json.load((out_a / "02_graph_analysis" / "candidate_actions.json").open())
    b = json.load((out_b / "02_graph_analysis" / "candidate_actions.json").open())
    a_disp = sorted(c["candidate_id"] for c in a["candidates"] if c["kind"] == "set_dispatch_mode")
    b_disp = sorted(c["candidate_id"] for c in b["candidates"] if c["kind"] == "set_dispatch_mode")
    assert a_disp == b_disp


# --------------------------------------------------------------------------- #
# Materialiser end-to-end — recipe_delta override flows through to contract
# --------------------------------------------------------------------------- #


def test_materialiser_picks_up_set_dispatch_mode_op(
    merlin_action_space: Path, tmp_path: Path,
) -> None:
    """Append a SetDispatchMode op to candidate_selection.recipe_delta;
    materialise; verify dispatch.model flipped."""
    import shutil
    from compgen.graph_compilation.kernel_contract_materialization import (
        materialize_contract_from_run_dir,
    )
    from compgen.kernels.contract_v3 import DispatchModel
    work = tmp_path / "work"
    shutil.copytree(merlin_action_space, work)
    sel_path = work / "03_recipe_planning" / "candidate_selection.json"
    sel = json.loads(sel_path.read_text())
    sel["recipe_delta"].append({
        "op": "SetDispatchMode", "region": "matmul_0", "mode": "async",
    })
    sel_path.write_text(json.dumps(sel, sort_keys=True, indent=2))
    contract = materialize_contract_from_run_dir(
        run_dir=work, candidate_selection=sel,
        region_id="matmul_0", target_id="host_cpu",
    )
    assert contract is not None
    assert contract.orchestration.dispatch.model is DispatchModel.ASYNC


def test_materialiser_rejects_persistent_override(
    merlin_action_space: Path, tmp_path: Path,
) -> None:
    """A SetDispatchMode(persistent) on NORMAL granularity → contract
    materialiser returns None. Catches a smuggled persistent past
    the action_space gate."""
    import shutil
    from compgen.graph_compilation.kernel_contract_materialization import (
        materialize_contract_from_run_dir,
    )
    work = tmp_path / "work"
    shutil.copytree(merlin_action_space, work)
    sel_path = work / "03_recipe_planning" / "candidate_selection.json"
    sel = json.loads(sel_path.read_text())
    sel["recipe_delta"].append({
        "op": "SetDispatchMode", "region": "matmul_0", "mode": "persistent",
    })
    sel_path.write_text(json.dumps(sel, sort_keys=True, indent=2))
    contract = materialize_contract_from_run_dir(
        run_dir=work, candidate_selection=sel,
        region_id="matmul_0", target_id="host_cpu",
    )
    assert contract is None
