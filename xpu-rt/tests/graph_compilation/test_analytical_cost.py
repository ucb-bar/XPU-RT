"""Acceptance tests for M-21 Per-Candidate Deterministic Analytical Cost.

Verifies:

- The per-candidate cost is **deterministic** — same inputs produce
  byte-identical JSON output across reruns.
- Cost is **rooted in the target YAML** — `peak_compute_gflops`,
  `peak_bandwidth_gb_s`, `memory_tiers`, `tier_bw_multiplier` all
  surface in `model_inputs_used`. Changing target peak_compute scales
  predicted_us inversely.
- Cost is **rooted in the graph dossier** — every modeled candidate
  has a `matmul_shape` resolved from cost_preview_v2 / region dossier
  and a `tile` from candidate_actions.recipe_delta.
- Working-set tier is correctly determined from the spec's
  `memory_tiers` (scratchpad/l2/l3/system).
- Roofline math sanity: compute_time_us = flops / peak_GFLOPS;
  memory_time_us = bytes / effective_bw; predicted_us = max(compute,
  memory) + overhead.
- Cross-reference: when M-19/M-20 measurements are on disk, the
  per-candidate entry includes ``calibration_delta``.
- Always-on: emits without env vars (it's pure analytical, no I/O cost).
- Layered onto ``cost_preview_v2`` and ``llm_graph_view`` via an
  additive ``m21_analytical_cost`` block (same pattern as M-18.3's
  ``calibration`` overlay). Region map / candidate_actions are NOT
  mutated. The overlay is byte-stable across reruns.
- No compiler-core imports.
- Best-effort: missing inputs → typed `not_run`; never raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run(model: str, out_dir: Path, *, run_kernels: bool) -> None:
    env = os.environ.copy()
    if run_kernels:
        env["COMPGEN_RUN_KERNELS"] = "1"
    else:
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


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """merlin_mlp_wide with kernels on — gives us calibration_delta data."""
    out = tmp_path_factory.mktemp("m21_kernels") / "run"
    _run("merlin_mlp_wide", out, run_kernels=True)
    return out


@pytest.fixture(scope="module")
def no_kernels_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """merlin_mlp_wide without kernels — analytical cost still emits."""
    out = tmp_path_factory.mktemp("m21_no_kernels") / "run"
    _run("merlin_mlp_wide", out, run_kernels=False)
    return out


# --------------------------------------------------------------------------- #
# Always-on emission
# --------------------------------------------------------------------------- #


def test_analytical_cost_emits_without_any_opt_in(no_kernels_run: Path) -> None:
    """M-21 is always-on; no env var needed."""
    base = no_kernels_run / "02_graph_analysis" / "analytical_cost"
    assert base.is_dir()
    assert (base / "per_candidate_analytical_cost.json").exists()
    assert (base / "analytical_cost_summary.md").exists()


def test_artifact_schema_version(no_kernels_run: Path) -> None:
    r = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    assert r["schema_version"] == "per_candidate_analytical_cost_v1"
    assert r["model_kind"] == "blocked_matmul_roofline_v1"
    assert r["deterministic"] is True


# --------------------------------------------------------------------------- #
# Determinism: byte-identical reruns
# --------------------------------------------------------------------------- #


def test_byte_identical_reruns(no_kernels_run: Path) -> None:
    """Calling run_analytical_cost twice on the same run dir must
    produce byte-identical JSON output (modulo the generated_at_utc
    timestamp). We compare the persistent bytes EXCEPT the timestamp."""
    from compgen.graph_compilation.analytical_cost import run_analytical_cost

    p = (
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    d1 = _read(p)
    run_analytical_cost(no_kernels_run)
    d2 = _read(p)
    # Strip the only non-deterministic field.
    d1.pop("generated_at_utc", None)
    d2.pop("generated_at_utc", None)
    assert d1 == d2, "M-21 output drifted on rerun"


def test_pure_function_predict_candidate_cost_is_deterministic() -> None:
    from compgen.graph_compilation.analytical_cost import predict_candidate_cost

    args = dict(
        matmul_shape=(16, 32, 16),
        tile=(16, 16, 16),
        dtype_bytes=4,
        peak_compute_gflops=100.0,
        peak_bandwidth_gb_s=30.0,
        memory_tiers={"scratchpad_bytes": 32768, "l2_bytes": 524288,
                      "l3_bytes": 16777216, "system_bytes": 17179869184},
    )
    a = predict_candidate_cost(**args)
    b = predict_candidate_cost(**args)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# Rooted in HW spec: changing peak_compute_gflops scales predicted_us
# --------------------------------------------------------------------------- #


def test_predicted_us_rooted_in_target_peak_compute() -> None:
    """Doubling peak_compute_gflops should approximately halve the
    compute_time_us (when compute-bound). This is the deterministic
    rooting in the target HW spec."""
    from compgen.graph_compilation.analytical_cost import predict_candidate_cost

    base = predict_candidate_cost(
        matmul_shape=(64, 64, 64),
        tile=(16, 16, 16),
        dtype_bytes=4,
        peak_compute_gflops=100.0,
        peak_bandwidth_gb_s=30.0,
        memory_tiers={"scratchpad_bytes": 32768, "l2_bytes": 524288,
                      "l3_bytes": 16777216, "system_bytes": 17179869184},
    )
    twice = predict_candidate_cost(
        matmul_shape=(64, 64, 64),
        tile=(16, 16, 16),
        dtype_bytes=4,
        peak_compute_gflops=200.0,                 # 2x
        peak_bandwidth_gb_s=30.0,
        memory_tiers={"scratchpad_bytes": 32768, "l2_bytes": 524288,
                      "l3_bytes": 16777216, "system_bytes": 17179869184},
    )
    # If compute-bound, doubling peak_compute halves compute_time_us.
    if base["bottleneck_resource"] == "compute":
        assert twice["compute"]["compute_time_us"] == pytest.approx(
            base["compute"]["compute_time_us"] / 2.0,
        )


def test_predicted_us_rooted_in_target_peak_bandwidth() -> None:
    from compgen.graph_compilation.analytical_cost import predict_candidate_cost

    base = predict_candidate_cost(
        matmul_shape=(16, 16, 256),       # high arithmetic intensity → memory-bound likely
        tile=(16, 16, 16),
        dtype_bytes=4,
        peak_compute_gflops=10000.0,
        peak_bandwidth_gb_s=10.0,
        memory_tiers={"scratchpad_bytes": 32768, "l2_bytes": 524288,
                      "l3_bytes": 16777216, "system_bytes": 17179869184},
    )
    twice_bw = predict_candidate_cost(
        matmul_shape=(16, 16, 256),
        tile=(16, 16, 16),
        dtype_bytes=4,
        peak_compute_gflops=10000.0,
        peak_bandwidth_gb_s=20.0,         # 2x
        memory_tiers={"scratchpad_bytes": 32768, "l2_bytes": 524288,
                      "l3_bytes": 16777216, "system_bytes": 17179869184},
    )
    if base["bottleneck_resource"] == "memory":
        assert twice_bw["memory"]["memory_time_us"] == pytest.approx(
            base["memory"]["memory_time_us"] / 2.0,
        )


# --------------------------------------------------------------------------- #
# Rooted in graph dossier: every candidate has matmul_shape + tile + iters
# --------------------------------------------------------------------------- #


def test_every_modeled_candidate_has_shape_and_tile(no_kernels_run: Path) -> None:
    r = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    ok = [c for c in r["candidates"] if c.get("model_status") == "ok"]
    assert len(ok) >= 1
    for c in ok:
        assert "matmul_shape" in c and all(
            c["matmul_shape"][k] > 0 for k in ("M", "N", "K")
        )
        assert "tile" in c and all(c["tile"][k] > 0 for k in ("M", "N", "K"))
        assert "iters" in c
        # M = tile.M * iters.M (etc.) — invariant.
        sh = c["matmul_shape"]; t = c["tile"]; i = c["iters"]
        assert sh["M"] == t["M"] * i["M"], f"{c['candidate_id']}: M shape vs iters mismatch"
        assert sh["N"] == t["N"] * i["N"]
        assert sh["K"] == t["K"] * i["K"]


def test_modeled_candidates_match_legal_set_tile_in_candidate_actions(
    no_kernels_run: Path,
) -> None:
    r = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    cas = _read(
        no_kernels_run / "02_graph_analysis" / "candidate_actions.json"
    )
    legal_set_tile = {
        c["candidate_id"] for c in cas["candidates"]
        if c.get("kind") == "set_tile_params"
        and (c.get("legality") or {}).get("ok")
    }
    modeled_ids = {
        c["candidate_id"] for c in r["candidates"]
        if c.get("model_status") == "ok"
    }
    assert modeled_ids == legal_set_tile, (
        f"M-21 modeled set drift: missing={legal_set_tile - modeled_ids} "
        f"extra={modeled_ids - legal_set_tile}"
    )


# --------------------------------------------------------------------------- #
# Working-set tier classification
# --------------------------------------------------------------------------- #


def test_tier_classification_from_memory_tiers() -> None:
    """A tile with working_set < scratchpad_bytes → 'scratchpad'.
    A tile with working_set > scratchpad but < l2 → 'l2'. Etc."""
    from compgen.graph_compilation.analytical_cost import predict_candidate_cost

    tiers = {
        "scratchpad_bytes": 32768,
        "l2_bytes": 524288,
        "l3_bytes": 16777216,
        "system_bytes": 17179869184,
    }
    # Tiny tile (16,16,16) f32 → working_set = (256+256+256)*4 = 3072
    small = predict_candidate_cost(
        matmul_shape=(64, 64, 64), tile=(16, 16, 16), dtype_bytes=4,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
        memory_tiers=tiers,
    )
    assert small["working_set"]["memory_tier"] == "scratchpad"

    # Large tile (256, 256, 256) f32 → working_set = 3*65536*4 = 786432, > l2
    large = predict_candidate_cost(
        matmul_shape=(256, 256, 256), tile=(256, 256, 256), dtype_bytes=4,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
        memory_tiers=tiers,
    )
    assert large["working_set"]["memory_tier"] == "l3"


# --------------------------------------------------------------------------- #
# Roofline math sanity
# --------------------------------------------------------------------------- #


def test_compute_time_matches_flops_over_peak() -> None:
    from compgen.graph_compilation.analytical_cost import predict_candidate_cost

    p = predict_candidate_cost(
        matmul_shape=(16, 32, 16), tile=(16, 16, 16), dtype_bytes=4,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
        memory_tiers={"scratchpad_bytes": 32768, "l2_bytes": 524288,
                      "l3_bytes": 16777216, "system_bytes": 17179869184},
    )
    flops = 2 * 16 * 32 * 16
    expected = (flops / (100.0 * 1e9)) * 1e6
    assert p["compute"]["compute_time_us"] == pytest.approx(expected)


def test_predicted_us_equals_max_bottleneck_plus_overhead() -> None:
    from compgen.graph_compilation.analytical_cost import predict_candidate_cost

    p = predict_candidate_cost(
        matmul_shape=(16, 32, 16), tile=(16, 16, 16), dtype_bytes=4,
        peak_compute_gflops=100.0, peak_bandwidth_gb_s=30.0,
        memory_tiers={"scratchpad_bytes": 32768, "l2_bytes": 524288,
                      "l3_bytes": 16777216, "system_bytes": 17179869184},
    )
    expected = (
        max(p["compute"]["compute_time_us"], p["memory"]["memory_time_us"])
        + p["overhead"]["total_overhead_us"]
    )
    assert p["predicted_us"] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Cross-reference with M-19 / M-20 measurements
# --------------------------------------------------------------------------- #


def test_calibration_delta_present_when_kernels_ran(kernels_run: Path) -> None:
    r = _read(
        kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    # At least one candidate (the greedy-selected one or an M-20
    # per-region pick) should have a calibration_delta.
    with_cal = [
        c for c in r["candidates"]
        if c.get("model_status") == "ok" and c.get("calibration_delta")
    ]
    assert len(with_cal) >= 1, "no candidate received calibration_delta"
    for c in with_cal:
        cal = c["calibration_delta"]
        # At least one of GPU/CPU measured present + a non-None ratio.
        assert "measured_gpu_us" in cal or "measured_cpu_us" in cal
        if cal.get("measured_gpu_us") and cal.get("predicted_vs_gpu_ratio") is not None:
            assert cal["predicted_vs_gpu_ratio"] > 0


def test_calibration_absent_when_kernels_off(no_kernels_run: Path) -> None:
    r = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    # Without M-19/M-20 measurements, calibration_delta should be
    # missing for all candidates.
    with_cal = [
        c for c in r["candidates"]
        if c.get("model_status") == "ok" and c.get("calibration_delta")
    ]
    assert with_cal == [], "calibration_delta leaked when no measurements"


# --------------------------------------------------------------------------- #
# Layered overlay onto cost_preview_v2 + llm_graph_view (M-21.2)
# --------------------------------------------------------------------------- #


def test_cost_preview_v2_has_m21_overlay(no_kernels_run: Path) -> None:
    """Every modeled SetTileParams candidate must have an
    ``m21_analytical_cost`` block in its cost_preview_v2 entry."""
    cp_doc = _read(
        no_kernels_run / "02_graph_analysis" / "cost_preview_v2.json"
    )
    ac = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    modeled_ids = {
        c["candidate_id"] for c in ac["candidates"]
        if c.get("model_status") == "ok"
    }
    assert modeled_ids, "no modeled candidates — fixture mis-set"

    cp_by_id = {p["candidate_id"]: p for p in cp_doc.get("cost_previews", [])}
    for cid in modeled_ids:
        assert cid in cp_by_id, f"{cid} missing from cost_preview_v2"
        block = cp_by_id[cid].get("m21_analytical_cost")
        assert block is not None, (
            f"{cid} missing m21_analytical_cost overlay block"
        )
        # Headline fields the agent reads.
        for field in (
            "model_kind", "deterministic", "predicted_us",
            "bottleneck_resource", "bottleneck_tier",
            "compute_time_us", "memory_time_us", "overhead_us",
            "matmul_shape", "tile", "iters_total",
            "effective_bandwidth_gb_s", "tile_working_set_bytes",
        ):
            assert field in block, (
                f"m21_analytical_cost overlay for {cid} missing field {field}"
            )
        assert block["model_kind"] == "blocked_matmul_roofline_v1"
        assert block["deterministic"] is True


def test_llm_graph_view_has_m21_overlay(no_kernels_run: Path) -> None:
    """Every modeled SetTileParams candidate must have an
    ``m21_analytical_cost`` block in its llm_graph_view legal_candidate
    entry."""
    lv_path = no_kernels_run / "02_graph_analysis" / "llm_graph_view.json"
    if not lv_path.exists():
        pytest.skip("llm_graph_view.json absent on this fixture")
    lv_doc = _read(lv_path)
    ac = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    modeled_ids = {
        c["candidate_id"] for c in ac["candidates"]
        if c.get("model_status") == "ok"
    }
    seen: set[str] = set()
    for region in lv_doc.get("regions", []) or []:
        for lc in region.get("legal_candidates", []) or []:
            cid = lc.get("candidate_id")
            if cid not in modeled_ids:
                continue
            seen.add(cid)
            block = lc.get("m21_analytical_cost")
            assert block is not None, (
                f"{cid} missing m21_analytical_cost overlay in llm_graph_view"
            )
            assert block.get("predicted_us") is not None
            assert block.get("model_kind") == "blocked_matmul_roofline_v1"
    assert seen == modeled_ids, (
        f"missing overlays for {sorted(modeled_ids - seen)}"
    )


def test_m21_overlay_byte_stable_across_reruns(no_kernels_run: Path) -> None:
    """Calling run_analytical_cost twice must produce a byte-identical
    cost_preview_v2.json (overlay is deterministic, like the standalone
    report). llm_graph_view.json must also be byte-identical."""
    from compgen.graph_compilation.analytical_cost import run_analytical_cost

    cp_path = no_kernels_run / "02_graph_analysis" / "cost_preview_v2.json"
    lv_path = no_kernels_run / "02_graph_analysis" / "llm_graph_view.json"

    def _sha(p: Path) -> str:
        return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()

    before_cp = _sha(cp_path)
    before_lv = _sha(lv_path) if lv_path.exists() else None
    run_analytical_cost(no_kernels_run)
    after_cp = _sha(cp_path)
    after_lv = _sha(lv_path) if lv_path.exists() else None
    assert before_cp == after_cp, "cost_preview_v2 overlay drifted on rerun"
    assert before_lv == after_lv, "llm_graph_view overlay drifted on rerun"


def test_m21_does_not_mutate_region_map_or_candidate_actions(
    no_kernels_run: Path,
) -> None:
    """region_map.json and candidate_actions.json are immutable through
    M-21 (by design — they're the canonical regions/candidates frozen
    at M-13/M-14)."""
    rm_path = no_kernels_run / "02_graph_analysis" / "region_map.json"
    ca_path = no_kernels_run / "02_graph_analysis" / "candidate_actions.json"

    def _sha(p: Path) -> str:
        return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()

    before = {"region_map": _sha(rm_path), "candidate_actions": _sha(ca_path)}
    from compgen.graph_compilation.analytical_cost import run_analytical_cost
    run_analytical_cost(no_kernels_run)
    after = {"region_map": _sha(rm_path), "candidate_actions": _sha(ca_path)}
    assert before == after, (
        f"M-21 mutated immutable artifact(s): "
        f"{[k for k in before if before[k] != after[k]]}"
    )


def test_m21_overlay_does_not_clobber_m183_calibration(
    no_kernels_run: Path,
) -> None:
    """If an M-18.3 ``calibration`` block exists on a cost_preview_v2
    entry, M-21's overlay must not clobber it. (This fixture has no
    M-18.3 calibration on; we install a synthetic one and verify the
    M-21 rerun preserves it.)"""
    from compgen.graph_compilation.analytical_cost import run_analytical_cost

    cp_path = no_kernels_run / "02_graph_analysis" / "cost_preview_v2.json"
    doc = _read(cp_path)
    ac = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    modeled_ids = {
        c["candidate_id"] for c in ac["candidates"]
        if c.get("model_status") == "ok"
    }
    if not modeled_ids:
        pytest.skip("no modeled candidates")
    target_cid = next(iter(modeled_ids))
    sentinel = {
        "measured_baseline_us": 1.23, "measured_tiled_us": 4.56,
        "measured_speedup": 0.27, "rel_error": 0.05,
        "calibration_status": "calibrated",
    }
    for cp in doc.get("cost_previews", []):
        if cp.get("candidate_id") == target_cid:
            cp["calibration"] = dict(sentinel)
    cp_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
    )

    run_analytical_cost(no_kernels_run)

    after = _read(cp_path)
    matched = next(
        cp for cp in after.get("cost_previews", [])
        if cp.get("candidate_id") == target_cid
    )
    assert matched.get("calibration") == sentinel, (
        "M-21 overlay clobbered M-18.3's calibration block"
    )
    assert matched.get("m21_analytical_cost") is not None, (
        "M-21 overlay missing after rerun"
    )


# --------------------------------------------------------------------------- #
# Aggregate sanity
# --------------------------------------------------------------------------- #


def test_summary_aggregates_match_per_candidate(no_kernels_run: Path) -> None:
    r = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    s = r["summary"]
    ok = [c for c in r["candidates"] if c.get("model_status") == "ok"]
    assert s["candidates_modeled"] == len(ok)
    if ok:
        compute_count = sum(1 for c in ok if c["bottleneck_resource"] == "compute")
        memory_count = sum(1 for c in ok if c["bottleneck_resource"] == "memory")
        assert s["compute_bound_count"] == compute_count
        assert s["memory_bound_count"] == memory_count
        predicted = [c["predicted_us"] for c in ok]
        assert s["min_predicted_us"] == pytest.approx(min(predicted))
        assert s["max_predicted_us"] == pytest.approx(max(predicted))
        assert s["mean_predicted_us"] == pytest.approx(
            sum(predicted) / len(predicted)
        )


# --------------------------------------------------------------------------- #
# Best-effort + integrity
# --------------------------------------------------------------------------- #


def test_handles_missing_candidate_actions(tmp_path: Path) -> None:
    fake = tmp_path / "fake_run"
    (fake / "02_graph_analysis").mkdir(parents=True)
    (fake / "00_graph_capture").mkdir(parents=True)
    from compgen.graph_compilation.analytical_cost import run_analytical_cost

    res = run_analytical_cost(fake)
    assert res.overall == "not_run"


def test_no_compiler_core_imports() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "analytical_cost.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir.payload",
        "import compgen.ir.payload",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
        "import compgen.pipeline",
    )
    for pat in forbidden:
        assert pat not in src, f"M-21 imports forbidden: {pat}"


def test_model_inputs_used_match_target_yaml(no_kernels_run: Path) -> None:
    """The report records the EXACT target spec values used so the
    agent can reproduce the prediction deterministically."""
    r = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    inputs = r["model_inputs_used"]
    # host_cpu.yaml has these specific values.
    assert inputs["peak_compute_gflops"] == 100.0
    assert inputs["peak_bandwidth_gb_s"] == 30.0
    assert "memory_tiers" in inputs and inputs["memory_tiers"]
    assert "tier_bw_multiplier" in inputs


def test_known_limitations_recorded(no_kernels_run: Path) -> None:
    r = _read(
        no_kernels_run / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    lims = r["known_limitations"]
    # Must explicitly state the model is roofline-only and not measured.
    assert any("roofline" in s.lower() for s in lims)
    assert any("multiplier" in s.lower() for s in lims)
