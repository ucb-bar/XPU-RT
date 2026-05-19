"""Regression test for the fusion candidate generator's arity gate.

`_gen_fusion` (xpu_rt.graph_compilation.action_space) must not emit
`fuse_producer_consumer` candidates whose consumer region has more
than one input tensor — the MVP differential evaluator at
`real_fusion.py` refuses binary consumers and the downstream gate
rejection would mask the candidate as a guaranteed failure.

The pointwise chain `x + bias → x1 * scale → relu(x2)` (the
holdout_pointwise_chain_renamed model) is the regression workload:

- `add_0  → mul_0`: mul has TWO inputs (the transient from add and
  the bias/scale parameter), so this fusion candidate must be
  filtered out.
- `mul_0 → relu_0`: relu has ONE input, so this fusion candidate
  must still surface.

Lifting the gate requires extending the differential evaluator;
this test guards against re-emitting candidates the evaluator
can't verify.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xpu_rt.graph_compilation.action_space import _gen_fusion
from xpu_rt.graph_compilation.region_dossier import TargetProfile


def _profile() -> TargetProfile:
    return TargetProfile(
        target_id="host_cpu",
        device_kind="cpu",
        peak_compute_gflops=500.0,
        peak_bandwidth_gb_s=50.0,
        scratchpad_bytes=131_072,
        l2_bytes=1_048_576,
        l3_bytes=33_554_432,
        system_bytes=16 * 1024 * 1024 * 1024,
        supported_dtypes=("f32",),
    )


def _dossier(bytes_total: int) -> dict[str, Any]:
    return {"cost": {"bytes": bytes_total, "flops": bytes_total // 4}}


def _pointwise_chain_use_def() -> dict[str, Any]:
    """Matches the holdout_pointwise_chain_renamed shape:

    inputs flow into add_0 (transient::3) → mul_0 (transient::4) → relu_0.
    `mul_0` has 2 input tensors (binary consumer); `relu_0` has 1.
    """
    return {
        "tensors": [
            # add_0 inputs (2 input edges to add_0)
            {"tensor_id": "in::a", "producer_region": "input",
             "consumer_regions": ["add_0"], "consumer_count": 1,
             "producer_lifetime_class": "input", "reuse_horizon": 6},
            {"tensor_id": "in::b", "producer_region": "input",
             "consumer_regions": ["add_0"], "consumer_count": 1,
             "producer_lifetime_class": "input", "reuse_horizon": 6},
            # add_0 → mul_0 (binary consumer)
            {"tensor_id": "t::add_out", "producer_region": "add_0",
             "consumer_regions": ["mul_0"], "consumer_count": 1,
             "producer_lifetime_class": "transient", "reuse_horizon": 1},
            # mul_0's second input (the param)
            {"tensor_id": "in::scale", "producer_region": "input",
             "consumer_regions": ["mul_0"], "consumer_count": 1,
             "producer_lifetime_class": "input", "reuse_horizon": 7},
            # mul_0 → relu_0 (unary consumer)
            {"tensor_id": "t::mul_out", "producer_region": "mul_0",
             "consumer_regions": ["relu_0"], "consumer_count": 1,
             "producer_lifetime_class": "transient", "reuse_horizon": 1},
            # relu_0 → output
            {"tensor_id": "t::relu_out", "producer_region": "relu_0",
             "consumer_regions": ["output"], "consumer_count": 1,
             "producer_lifetime_class": "output", "reuse_horizon": -1},
        ],
    }


def _regions() -> list[dict[str, Any]]:
    return [
        {"region_id": "add_0", "kind": "elementwise_add"},
        {"region_id": "mul_0", "kind": "elementwise_mul"},
        {"region_id": "relu_0", "kind": "elementwise_relu"},
    ]


def test_binary_consumer_rejected_unary_consumer_kept() -> None:
    use_def = _pointwise_chain_use_def()
    regions = _regions()
    dossier_by_id = {r["region_id"]: _dossier(8704) for r in regions}
    region_ref = {r["region_id"]: f"02_graph_analysis/{r['region_id']}.json"
                  for r in regions}

    sites, cands = _gen_fusion(
        use_def, regions, dossier_by_id, _profile(), region_ref,
    )

    fuse_cands = [c for c in cands if c.kind == "fuse_producer_consumer"]
    producers = {c.region_id for c in fuse_cands}

    # The mul_0 consumer is binary → add_0 → mul_0 candidate must be dropped.
    assert "add_0" not in producers, (
        f"add_0 → mul_0 must be filtered (mul has 2 inputs); got {producers}"
    )
    # The relu_0 consumer is unary → mul_0 → relu_0 candidate must survive.
    assert "mul_0" in producers, (
        f"mul_0 → relu_0 must survive (relu has 1 input); got {producers}"
    )


def test_fusion_enabled_by_default_emits_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Without any mask override (and without env var opt-ins),
    `build_action_space` must surface `fuse_producer_consumer`
    candidates on the pointwise chain — the use-def + arity gate
    accepts mul_0 → relu_0 as a legal unary-consumer pair.

    Asserts the default-on contract. After the prove-or-kill audit
    + re-analysis on commit 4eab92c4112a, fusion is on by default;
    a regression that flips it back to off would break this test.
    Killing fusion would require both flipping the default AND
    documenting why this test should be removed.
    """
    monkeypatch.delenv("XPU_RT_ENABLE_FUSION", raising=False)
    monkeypatch.delenv("XPU_RT_SUBSYSTEM_MASK", raising=False)

    from xpu_rt.graph_compilation.run import run_graph_compilation
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    model_cfg = repo_root / "configs" / "models" / "holdout_pointwise_chain_renamed.yaml"
    target_cfg = repo_root / "configs" / "targets" / "host_cpu.yaml"
    run_dir = tmp_path / "run"
    run_graph_compilation(
        model_config_path=model_cfg,
        target_config_path=target_cfg,
        out_dir=run_dir,
        stop_after="graph-analysis",
    )
    cas = json.loads(
        (run_dir / "02_graph_analysis" / "candidate_actions.json").read_text()
    )
    fuse_cands = [
        c for c in cas.get("candidates", [])
        if c.get("kind") == "fuse_producer_consumer"
    ]
    # Pointwise chain x→add→mul→relu emits exactly one legal fusion
    # (mul → relu, unary consumer). add → mul is filtered by the
    # arity gate because mul is binary.
    assert len(fuse_cands) >= 1, (
        "fusion is on by default; expected at least one "
        "fuse_producer_consumer candidate on the pointwise chain, "
        "got zero. If you intentionally re-killed fusion, remove "
        "this test and update docs/realness/agent_decisions_fusion.yaml."
    )


def test_chain_of_unary_consumers_all_emit() -> None:
    """Three-stage unary chain: every consecutive pair should emit."""
    use_def = {
        "tensors": [
            {"tensor_id": "in::0", "producer_region": "input",
             "consumer_regions": ["a"], "consumer_count": 1,
             "producer_lifetime_class": "input", "reuse_horizon": 8},
            {"tensor_id": "t::a", "producer_region": "a",
             "consumer_regions": ["b"], "consumer_count": 1,
             "producer_lifetime_class": "transient", "reuse_horizon": 1},
            {"tensor_id": "t::b", "producer_region": "b",
             "consumer_regions": ["c"], "consumer_count": 1,
             "producer_lifetime_class": "transient", "reuse_horizon": 1},
            {"tensor_id": "t::c", "producer_region": "c",
             "consumer_regions": ["output"], "consumer_count": 1,
             "producer_lifetime_class": "output", "reuse_horizon": -1},
        ],
    }
    regions = [
        {"region_id": "a", "kind": "elementwise_relu"},
        {"region_id": "b", "kind": "elementwise_relu"},
        {"region_id": "c", "kind": "elementwise_relu"},
    ]
    dossier_by_id = {r["region_id"]: _dossier(1024) for r in regions}
    region_ref = {r["region_id"]: f"02_graph_analysis/{r['region_id']}.json"
                  for r in regions}
    _, cands = _gen_fusion(
        use_def, regions, dossier_by_id, _profile(), region_ref,
    )
    producers = {c.region_id for c in cands if c.kind == "fuse_producer_consumer"}
    assert producers == {"a", "b"}, producers
