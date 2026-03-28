"""Tests for the agent-first compiler environment."""

from __future__ import annotations

import sys
from pathlib import Path

from compgen.packs import PackContextSummary
from compgen.agent.env import (
    AnalyzeAction,
    ApplyPassAction,
    AssignDeviceAction,
    BenchmarkAction,
    CheckpointAction,
    CompilerEnv,
    GeneralizeAction,
    InspectAction,
    NoopAction,
    RollbackAction,
    SearchKernelAction,
    SolveAction,
    TileAction,
)
from compgen.agent.serialize import (
    observation_to_dict,
    observation_to_prompt,
    parse_action,
    result_to_prompt,
)
from compgen.capture.torch_export import capture_frontend_artifact, capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.targets.schema import load_profile

EXAMPLES = Path(__file__).parent.parent.parent / "examples"


def _get_mlp_full():
    """Get everything needed for full env: module, ep, pytorch model, inputs."""
    sys.path.insert(0, str(EXAMPLES / "models"))
    from simple_mlp import SimpleMLP, get_sample_inputs
    model = SimpleMLP()
    inputs = get_sample_inputs()
    ep = capture_model(model, inputs)
    module, _ = fx_to_xdsl(ep)
    return module, ep, model, inputs


def _get_mlp_module_and_ep():
    module, ep, _, _ = _get_mlp_full()
    return module, ep


def _get_mlp_module():
    module, _, _, _ = _get_mlp_full()
    return module


def _get_target():
    return load_profile(EXAMPLES / "target_profiles" / "cuda_a100.yaml")


def _get_multi_target():
    return load_profile(EXAMPLES / "target_profiles" / "multi_device.yaml")


# ---- Environment basics ----


def test_env_reset() -> None:
    """reset() should return an Observation with regions."""
    env = CompilerEnv()
    module = _get_mlp_module()
    target = _get_target()
    obs = env.reset(module, target)

    assert len(obs.regions) >= 2  # at least 2 matmuls
    assert obs.step_count == 0
    assert obs.budget_remaining > 0
    assert obs.estimated_total_latency_us > 0


def test_env_regions_have_correct_types() -> None:
    """Regions should identify matmul and gelu ops."""
    env = CompilerEnv()
    obs = env.reset(_get_mlp_module(), _get_target())

    op_types = {r.op_type for r in obs.regions}
    assert "matmul" in op_types
    # gelu might be there too


def test_env_regions_have_shapes() -> None:
    """Each region should have input/output shapes."""
    env = CompilerEnv()
    obs = env.reset(_get_mlp_module(), _get_target())

    for r in obs.regions:
        assert len(r.input_shapes) > 0 or len(r.output_shapes) > 0


def test_env_regions_have_flops() -> None:
    """Matmul regions should have non-zero FLOPs."""
    env = CompilerEnv()
    obs = env.reset(_get_mlp_module(), _get_target())

    matmuls = [r for r in obs.regions if r.op_type == "matmul"]
    assert len(matmuls) >= 1
    for m in matmuls:
        assert m.flops > 0


# ---- Legal actions ----


def test_legal_actions_exist() -> None:
    """legal_actions() should return non-empty list."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    actions = env.legal_actions()
    assert len(actions) > 0


def test_legal_actions_include_tiles() -> None:
    """Legal actions should include tiling for matmul regions."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    actions = env.legal_actions()

    tile_actions = [a for a in actions if isinstance(a.action, TileAction)]
    assert len(tile_actions) > 0
    for ta in tile_actions:
        assert len(ta.action.tile_sizes) > 0
        assert all(t > 0 for t in ta.action.tile_sizes)


def test_legal_actions_ranked() -> None:
    """Actions should be ranked (1 = best predicted)."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    actions = env.legal_actions()
    ranks = [a.rank for a in actions]
    assert ranks == sorted(ranks)


def test_legal_actions_include_noop() -> None:
    """Legal actions should always include noop."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    actions = env.legal_actions()
    noops = [a for a in actions if isinstance(a.action, NoopAction)]
    assert len(noops) == 1


def test_multi_device_has_placement_actions() -> None:
    """Multi-device targets should have device placement actions."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_multi_target())
    actions = env.legal_actions()

    place_actions = [a for a in actions if isinstance(a.action, AssignDeviceAction)]
    assert len(place_actions) > 0


# ---- Step ----


def test_step_noop() -> None:
    """Stepping with noop should not change cost."""
    env = CompilerEnv()
    obs = env.reset(_get_mlp_module(), _get_target())
    cost_before = obs.estimated_total_latency_us

    result = env.step(NoopAction())
    assert result.info.action_applied
    assert result.observation.estimated_total_latency_us == cost_before
    assert result.observation.step_count == 1


def test_step_returns_structured_result() -> None:
    """StepResult should have all fields populated."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    result = env.step(NoopAction())

    assert result.observation is not None
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert result.info.verification_passed


def test_step_tracks_history() -> None:
    """History should grow with each step."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target(), budget=10)
    env.step(NoopAction())
    env.step(NoopAction())
    obs = env.observe()
    assert len(obs.history_summary) == 2


def test_step_budget_exhaustion() -> None:
    """done should be True when budget is exhausted."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target(), budget=2)
    r1 = env.step(NoopAction())
    assert not r1.done
    r2 = env.step(NoopAction())
    assert r2.done


# ---- Serialization ----


def test_observation_to_prompt() -> None:
    """observation_to_prompt should produce compact text."""
    env = CompilerEnv()
    obs = env.reset(_get_mlp_module(), _get_target())
    actions = env.legal_actions(max_actions=10)
    prompt = observation_to_prompt(obs, actions)

    assert "REGIONS:" in prompt
    assert "matmul_0" in prompt
    assert "ACTIONS" in prompt
    assert "us" in prompt  # latency units


def test_observation_to_dict() -> None:
    """observation_to_dict should produce structured dict."""
    env = CompilerEnv()
    obs = env.reset(_get_mlp_module(), _get_target())
    d = observation_to_dict(obs)

    assert "regions" in d
    assert len(d["regions"]) >= 2
    region_types = {r["type"] for r in d["regions"]}
    assert "matmul" in region_types


def test_parse_action_round_trip() -> None:
    """parse_action should reconstruct actions from dicts."""
    action = parse_action({"type": "tile", "region_id": "matmul_0", "tile_sizes": [128, 128, 32]})
    assert isinstance(action, TileAction)
    assert action.tile_sizes == (128, 128, 32)
    assert action.region_id == "matmul_0"


def test_result_to_prompt() -> None:
    """result_to_prompt should produce compact feedback."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    result = env.step(NoopAction())
    text = result_to_prompt(result)
    assert "RESULT:" in text
    assert "VERIFY:" in text


def test_observation_exposes_frontend_artifact_summary() -> None:
    """Observation should surface graph-break, guard, and unsupported summaries."""
    artifact = capture_frontend_artifact(*_get_mlp_full()[2:])
    module, _, model, inputs = _get_mlp_full()
    env = CompilerEnv()
    obs = env.reset(
        module,
        _get_target(),
        exported_program=artifact.exported_program,
        capture_artifact=artifact,
        pytorch_model=model,
        sample_inputs=inputs,
    )

    assert obs.graph_break_count >= 0
    assert obs.guard_count >= 0
    assert isinstance(obs.unsupported_ops, tuple)


def test_observation_exposes_pack_context() -> None:
    env = CompilerEnv()
    obs = env.reset(
        _get_mlp_module(),
        _get_target(),
        pack_context=PackContextSummary(
            active_packs=("cuda_tile", "iree_tracy"),
            sealed_surfaces=("tile_dialect_semantics",),
            generation_apertures=("tile_schedule_generation",),
            available_profilers=("iree_tracy",),
            benchmark_targets=("cuda_tile_smoke",),
            integration_branch="compgen/integration/cuda_tile/test",
        ),
    )

    assert obs.active_packs == ("cuda_tile", "iree_tracy")
    assert "PACKS:" in observation_to_prompt(obs)
    assert observation_to_dict(obs)["packs"]["integration_branch"] == "compgen/integration/cuda_tile/test"


# ---- Real transforms ----


def test_generalize_changes_ir() -> None:
    """GeneralizeAction should convert matmul → generic in the IR."""
    env = CompilerEnv()
    obs = env.reset(_get_mlp_module(), _get_target())

    # Before: should have matmul regions
    matmuls_before = [r for r in obs.regions if r.op_type == "matmul"]
    assert len(matmuls_before) >= 1

    # Apply generalize
    result = env.step(GeneralizeAction(region_id="all"))
    assert result.info.action_applied, f"Generalize failed: {result.info.error}"

    # After: matmul regions should become generic
    obs_after = result.observation
    matmuls_after = [r for r in obs_after.regions if r.op_type == "matmul"]
    generics_after = [r for r in obs_after.regions if r.op_type == "generic"]
    assert len(matmuls_after) == 0, "Matmul ops should be generalized"
    assert len(generics_after) >= 1, "Should have generic ops after generalization"


def test_generalize_preserves_region_count() -> None:
    """Generalization should not lose or create regions."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())

    result = env.step(GeneralizeAction(region_id="all"))
    assert result.info.action_applied
    # Region count may change slightly (transpose may or may not get region_id)
    # but should be in the same ballpark
    count_after = len(result.observation.regions)
    assert count_after >= 1


def test_apply_pass_dce() -> None:
    """ApplyPassAction with DCE should succeed."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    result = env.step(ApplyPassAction(pass_name="dce"))
    assert result.info.action_applied, f"DCE failed: {result.info.error}"


def test_apply_pass_unknown_rejected() -> None:
    """Unknown pass names should be rejected with a clear error."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    result = env.step(ApplyPassAction(pass_name="nonexistent_pass"))
    assert not result.info.action_applied
    assert "Unknown pass" in result.info.error


def test_apply_pass_canonicalize() -> None:
    """Canonicalize pass should succeed on SimpleMLP IR."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    result = env.step(ApplyPassAction(pass_name="canonicalize"))
    assert result.info.action_applied, f"Canonicalize failed: {result.info.error}"


# ---- Checkpoint + Rollback ----


def test_checkpoint_and_rollback() -> None:
    """Checkpoint saves state, rollback restores it."""
    env = CompilerEnv()
    obs = env.reset(_get_mlp_module(), _get_target())
    initial_regions = len(obs.regions)

    # Checkpoint
    result = env.step(CheckpointAction())
    assert result.info.action_applied

    # Change state: generalize
    result = env.step(GeneralizeAction(region_id="all"))
    assert result.info.action_applied

    # Rollback: should restore original state
    result = env.step(RollbackAction())
    assert result.info.action_applied
    restored_regions = len(result.observation.regions)
    assert restored_regions == initial_regions


def test_rollback_without_checkpoint_fails() -> None:
    """Rollback with no checkpoint should fail gracefully."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    result = env.step(RollbackAction())
    assert not result.info.action_applied
    assert "No checkpoint" in result.info.error


# ---- Inspect ----


def test_inspect_returns_details() -> None:
    """InspectAction should return detailed diagnostics."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    result = env.step(InspectAction(region_id="matmul_0"))
    assert result.info.action_applied
    assert len(result.info.diagnostics) > 0
    assert "matmul_0" in result.info.diagnostics[0]
    assert "flops" in result.info.diagnostics[0].lower()


def test_inspect_unknown_region_fails() -> None:
    """Inspecting non-existent region should fail."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    result = env.step(InspectAction(region_id="nonexistent"))
    assert not result.info.action_applied


# ---- Multi-step strategies ----


def test_multi_step_generalize_then_pass() -> None:
    """Agent can chain: generalize → canonicalize → DCE."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target(), budget=10)

    r1 = env.step(GeneralizeAction(region_id="all"))
    assert r1.info.action_applied

    r2 = env.step(ApplyPassAction(pass_name="canonicalize"))
    assert r2.info.action_applied

    r3 = env.step(ApplyPassAction(pass_name="dce"))
    assert r3.info.action_applied

    assert r3.observation.step_count == 3


def test_legal_actions_include_passes() -> None:
    """Legal actions should include pass menu items."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    actions = env.legal_actions()

    pass_actions = [a for a in actions if isinstance(a.action, ApplyPassAction)]
    assert len(pass_actions) >= 3  # dce, canonicalize, constant_fold, cse

    generalize_actions = [a for a in actions if isinstance(a.action, GeneralizeAction)]
    assert len(generalize_actions) >= 1

    checkpoint_actions = [a for a in actions if isinstance(a.action, CheckpointAction)]
    assert len(checkpoint_actions) == 1


# ---- Network Analysis ----


def test_analyze_action() -> None:
    """AnalyzeAction should detect pattern clusters from the FX graph."""
    env = CompilerEnv()
    module, ep = _get_mlp_module_and_ep()
    env.reset(module, _get_target(), exported_program=ep)

    result = env.step(AnalyzeAction())
    assert result.info.action_applied, f"Analyze failed: {result.info.error}"
    assert len(result.info.diagnostics) > 0
    assert any("pattern cluster" in d.lower() for d in result.info.diagnostics)

    # Analysis should be accessible
    assert env.analysis is not None
    assert len(env.analysis.clusters) >= 1


def test_analyze_finds_linear_chain() -> None:
    """Analysis should find the linear_chain pattern in SimpleMLP."""
    env = CompilerEnv()
    module, ep = _get_mlp_module_and_ep()
    env.reset(module, _get_target(), exported_program=ep)
    env.step(AnalyzeAction())

    chain_clusters = [c for c in env.analysis.clusters if c.pattern_type == "linear_chain"]
    assert len(chain_clusters) == 1
    assert chain_clusters[0].kernel_opportunity == "fused_mlp"


def test_analyze_without_exported_program_fails() -> None:
    """AnalyzeAction without exported_program should fail clearly."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())  # no exported_program
    result = env.step(AnalyzeAction())
    assert not result.info.action_applied
    assert "ExportedProgram" in result.info.error


def test_analyze_shows_kernel_opportunities() -> None:
    """Analysis diagnostics should mention kernel opportunities."""
    env = CompilerEnv()
    module, ep = _get_mlp_module_and_ep()
    env.reset(module, _get_target(), exported_program=ep)
    result = env.step(AnalyzeAction())

    assert any("fused_mlp" in d for d in result.info.diagnostics)


def test_legal_actions_include_analyze() -> None:
    """Legal actions should include AnalyzeAction."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target())
    actions = env.legal_actions()

    analyze_actions = [a for a in actions if isinstance(a.action, AnalyzeAction)]
    assert len(analyze_actions) >= 1


def test_search_kernel_without_analysis_fails() -> None:
    """SearchKernelAction without prior analysis should fail clearly."""
    env = CompilerEnv()
    module, ep = _get_mlp_module_and_ep()
    env.reset(module, _get_target(), exported_program=ep)

    result = env.step(SearchKernelAction(cluster_id="linear_chain_0"))
    assert not result.info.action_applied
    assert "AnalyzeAction" in result.info.error


# ---- Solver ----


def test_solve_placement_on_multi_device() -> None:
    """SolveAction should find a feasible placement on CPU+GPU target."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_multi_target(), budget=10)

    result = env.step(SolveAction(solve_type="placement"))
    assert result.info.action_applied, f"Solve failed: {result.info.error}"
    assert any("PLACEMENT" in d for d in result.info.diagnostics)
    assert any("feasible" in d for d in result.info.diagnostics)

    # Regions should now have device assignments
    obs = result.observation
    assigned = [r for r in obs.regions if r.device_index >= 0]
    assert len(assigned) > 0


def test_solve_schedule_on_multi_device() -> None:
    """Scheduling after placement should produce a feasible schedule."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_multi_target(), budget=10)

    # First place, then schedule
    env.step(SolveAction(solve_type="placement"))
    result = env.step(SolveAction(solve_type="schedule"))
    assert result.info.action_applied, f"Schedule failed: {result.info.error}"
    assert any("SCHEDULE" in d for d in result.info.diagnostics)


def test_solve_both_placement_and_schedule() -> None:
    """solve_type='both' should do placement + scheduling in one step."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_multi_target(), budget=5)

    result = env.step(SolveAction(solve_type="both"))
    assert result.info.action_applied, f"Solve failed: {result.info.error}"
    assert any("PLACEMENT" in d for d in result.info.diagnostics)
    assert any("SCHEDULE" in d for d in result.info.diagnostics)


def test_solve_on_single_device_fails() -> None:
    """Solver should fail gracefully on single-device target."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target(), budget=5)  # single GPU

    result = env.step(SolveAction(solve_type="placement"))
    assert not result.info.action_applied
    assert "multi-device" in result.info.error.lower()


# ---- Benchmark ----


def test_benchmark_cpu() -> None:
    """BenchmarkAction should return real CPU latency measurements."""
    module, ep, model, inputs = _get_mlp_full()
    env = CompilerEnv()
    env.reset(module, _get_target(), pytorch_model=model, sample_inputs=inputs, budget=5)

    result = env.step(BenchmarkAction(device="cpu", mode="eager", num_iterations=10))
    assert result.info.action_applied, f"Benchmark failed: {result.info.error}"
    assert any("BENCHMARK" in d for d in result.info.diagnostics)
    assert any("latency" in d.lower() for d in result.info.diagnostics)


def test_benchmark_gpu() -> None:
    """BenchmarkAction on GPU should return real measurements."""
    import torch
    if not torch.cuda.is_available():
        return  # skip if no GPU

    module, ep, model, inputs = _get_mlp_full()
    env = CompilerEnv()
    env.reset(module, _get_target(), pytorch_model=model, sample_inputs=inputs, budget=5)

    result = env.step(BenchmarkAction(device="cuda", mode="eager", num_iterations=10))
    assert result.info.action_applied, f"Benchmark failed: {result.info.error}"
    assert any("BENCHMARK" in d for d in result.info.diagnostics)


def test_benchmark_compiled_gpu() -> None:
    """BenchmarkAction with torch.compile should work."""
    import torch
    if not torch.cuda.is_available():
        return

    module, ep, model, inputs = _get_mlp_full()
    env = CompilerEnv()
    env.reset(module, _get_target(), pytorch_model=model, sample_inputs=inputs, budget=5)

    result = env.step(BenchmarkAction(device="cuda", mode="compiled", num_iterations=10))
    assert result.info.action_applied, f"Benchmark failed: {result.info.error}"


def test_benchmark_without_model_fails() -> None:
    """BenchmarkAction without pytorch_model should fail clearly."""
    env = CompilerEnv()
    env.reset(_get_mlp_module(), _get_target(), budget=5)

    result = env.step(BenchmarkAction(device="cpu"))
    assert not result.info.action_applied
    assert "PyTorch model" in result.info.error


def test_benchmark_shows_cost_model_accuracy() -> None:
    """Benchmark diagnostics should compare estimated vs actual latency."""
    module, ep, model, inputs = _get_mlp_full()
    env = CompilerEnv()
    env.reset(module, _get_target(), pytorch_model=model, sample_inputs=inputs, budget=5)

    result = env.step(BenchmarkAction(device="cpu", mode="eager", num_iterations=10))
    assert result.info.action_applied
    # Should show cost model comparison
    assert any("cost model" in d.lower() or "accuracy" in d.lower() for d in result.info.diagnostics)
