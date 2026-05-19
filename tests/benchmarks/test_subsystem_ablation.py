"""Tests for the subsystem-ablation harness.

The foundation PR ships the mask + the ablation runner without
wiring any per-subsystem off-paths. So real on/off runs of the
pipeline would currently raise ``SubsystemMaskUnwiredError`` from
each subsystem entry point. These tests therefore exercise:

- ``SubsystemMask`` data-model round trip + disable-list parsing
- ``below_noise_floor`` kill-rule check
- ``run_cell`` env-var lifecycle (set + restore) using a stub
  ``run_one_cell`` injected via monkeypatch — proves the mask
  propagation contract without touching the real pipeline
- Pack serialization shape

A separate phase-1 test will exercise real pipeline cells once
``kernels.codegen_fallback`` (or any other phase-1 flag) ships its
off-path.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from xpu_rt.benchmarks import subsystem_ablation as sa
from xpu_rt.benchmarks.pass_pool_ablation import AblationResult
from xpu_rt.benchmarks.subsystem_mask import (
    SubsystemMask,
    SubsystemMaskUnknownFlagError,
    SubsystemMaskUnwiredError,
    _ACTIVE_MASK_ENV,
    active_mask_from_env,
)


# --------------------------------------------------------------------------- #
# Mask data model
# --------------------------------------------------------------------------- #


def test_all_on_has_no_disabled_flags() -> None:
    m = SubsystemMask.all_on()
    assert m.disabled_flags() == ()


def test_all_off_disables_every_flag() -> None:
    m = SubsystemMask.all_off()
    assert sorted(m.disabled_flags()) == sorted(SubsystemMask.flag_names())


def test_from_disable_list_leaf() -> None:
    m = SubsystemMask.from_disable_list(["kernels.codegen_fallback"])
    assert m.kernels__codegen_fallback is False
    assert m.kernels__contract_v3 is True
    assert m.disabled_flags() == ("kernels.codegen_fallback",)


def test_from_disable_list_subsystem_prefix() -> None:
    m = SubsystemMask.from_disable_list(["agent.decisions"])
    assert sorted(m.disabled_flags()) == [
        "agent.decisions.codegen_backend",
        "agent.decisions.encoding",
        "agent.decisions.fusion",
        "agent.decisions.tiling",
    ]


def test_from_disable_list_rejects_unknown() -> None:
    with pytest.raises(SubsystemMaskUnknownFlagError):
        SubsystemMask.from_disable_list(["nope.no_such_flag"])


def test_to_dict_round_trip() -> None:
    m = SubsystemMask.from_disable_list(["memory.embeddings", "eqsat.rules.fusion"])
    assert SubsystemMask.from_dict(m.to_dict()) == m


def test_get_unknown_flag_raises() -> None:
    m = SubsystemMask.all_on()
    with pytest.raises(SubsystemMaskUnknownFlagError):
        m.get("not.a.real.flag")


def test_check_wired_raises_for_unwired_flag() -> None:
    # Any flag not in _WIRED_FLAGS must raise when flipped off.
    unwired = next(
        n for n in SubsystemMask.flag_names()
        if n not in SubsystemMask._WIRED_FLAGS
    )
    m = SubsystemMask.from_disable_list([unwired])
    with pytest.raises(SubsystemMaskUnwiredError) as exc:
        m.check_wired(unwired)
    msg = str(exc.value)
    assert unwired in msg
    assert "_WIRED_FLAGS" in msg


def test_check_wired_no_op_for_wired_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # A flag listed in _WIRED_FLAGS must NOT raise when off — it's
    # the contract that says "this off-path is implemented."
    monkeypatch.setattr(
        SubsystemMask, "_WIRED_FLAGS", frozenset({"kernels.codegen_fallback"}),
    )
    m = SubsystemMask.from_disable_list(["kernels.codegen_fallback"])
    m.check_wired("kernels.codegen_fallback")  # must not raise


def test_check_wired_no_op_when_on() -> None:
    SubsystemMask.all_on().check_wired("kernels.codegen_fallback")


# --------------------------------------------------------------------------- #
# Env-var round trip
# --------------------------------------------------------------------------- #


def test_active_mask_from_env_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ACTIVE_MASK_ENV, raising=False)
    assert active_mask_from_env() is None


def test_active_mask_from_env_parses_disable_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ACTIVE_MASK_ENV, "kernels.codegen_fallback, eqsat.pipeline")
    m = active_mask_from_env()
    assert m is not None
    assert sorted(m.disabled_flags()) == ["eqsat.pipeline", "kernels.codegen_fallback"]


# --------------------------------------------------------------------------- #
# Noise-floor check
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("delta,stddev,expected", [
    (0.005, 0.005, True),   # |delta|=0.005 < max(0.01, 0.00025) — below 2σ floor
    (0.5, 0.05, False),     # |delta|=0.5 >= max(0.1, 0.025)
    (-0.5, 0.05, False),    # sign-agnostic
    (0.0, 0.0, True),       # zero delta is "no signal"
])
def test_below_noise_floor(delta: float, stddev: float, expected: bool) -> None:
    assert sa.below_noise_floor(
        delta_seconds=delta, noise_stddev=stddev,
    ) is expected


# --------------------------------------------------------------------------- #
# run_cell env-var lifecycle (stubbed run_one_cell)
# --------------------------------------------------------------------------- #


def _stub_result(model_yaml: Path, target_yaml: Path, mode: str) -> AblationResult:
    return AblationResult(
        model_id=model_yaml.stem,
        target_id=target_yaml.stem,
        mode=mode,
        selected_candidate_id="cand_test",
        candidate_kind="set_tile_params",
        pass_id="set_tile_params",
        validation_overall="pass",
        validation_failures=(),
        decision_seconds=0.123,
        typed_outcome="verified",
    )


def test_run_cell_sets_and_restores_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    seen_during_call: list[str | None] = []

    def fake_run_one_cell(**kwargs: object) -> AblationResult:
        seen_during_call.append(os.environ.get(_ACTIVE_MASK_ENV))
        # Touch the run dir so _write_sidecar has somewhere to write.
        out_dir = Path(kwargs["out_dir"])  # type: ignore[arg-type]
        out_dir.mkdir(parents=True, exist_ok=True)
        return _stub_result(
            Path(kwargs["model_yaml"]),  # type: ignore[arg-type]
            Path(kwargs["target_yaml"]),  # type: ignore[arg-type]
            str(kwargs["mode"]),
        )

    monkeypatch.setattr(sa, "run_one_cell", fake_run_one_cell)
    monkeypatch.setenv(_ACTIVE_MASK_ENV, "preexisting.value.does.not.parse")

    model_yaml = tmp_path / "merlin_mlp_wide.yaml"
    target_yaml = tmp_path / "host_cpu.yaml"
    model_yaml.write_text("noop\n")
    target_yaml.write_text("noop\n")
    out_dir = tmp_path / "cell"

    mask = SubsystemMask.from_disable_list(["kernels.codegen_fallback"])
    cell = sa.run_cell(
        model_yaml=model_yaml, target_yaml=target_yaml,
        out_dir=out_dir, mask=mask, mask_label="treatment",
    )

    # During the call the env var carried the mask's disable list.
    assert seen_during_call == ["kernels.codegen_fallback"]
    # After the call the original env-var value is restored.
    assert os.environ.get(_ACTIVE_MASK_ENV) == "preexisting.value.does.not.parse"

    # The cell wrote the sidecar.
    sidecar = json.loads((out_dir / "subsystem_mask.json").read_text())
    assert sidecar["mask_label"] == "treatment"
    assert sidecar["disabled_flags"] == ["kernels.codegen_fallback"]

    # The wrapper carries the mask + result.
    assert cell.mask == mask
    assert cell.result.model_id == "merlin_mlp_wide"


def test_run_cell_pops_env_when_unset_prior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv(_ACTIVE_MASK_ENV, raising=False)

    def fake_run_one_cell(**kwargs: object) -> AblationResult:
        return _stub_result(
            Path(kwargs["model_yaml"]),  # type: ignore[arg-type]
            Path(kwargs["target_yaml"]),  # type: ignore[arg-type]
            str(kwargs["mode"]),
        )

    monkeypatch.setattr(sa, "run_one_cell", fake_run_one_cell)
    model_yaml = tmp_path / "m.yaml"
    target_yaml = tmp_path / "t.yaml"
    model_yaml.write_text("noop\n")
    target_yaml.write_text("noop\n")
    sa.run_cell(
        model_yaml=model_yaml, target_yaml=target_yaml,
        out_dir=tmp_path / "c", mask=SubsystemMask.all_on(), mask_label="control",
    )
    assert _ACTIVE_MASK_ENV not in os.environ


# --------------------------------------------------------------------------- #
# Pack shape
# --------------------------------------------------------------------------- #


def test_pack_summary_and_to_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def fake_run_one_cell(**kwargs: object) -> AblationResult:
        out_dir = Path(kwargs["out_dir"])  # type: ignore[arg-type]
        out_dir.mkdir(parents=True, exist_ok=True)
        # Treatment cell returns a different selected_candidate_id so
        # the row is flagged as candidate_changed.
        out_dir_name = out_dir.name
        return replace(
            _stub_result(
                Path(kwargs["model_yaml"]),  # type: ignore[arg-type]
                Path(kwargs["target_yaml"]),  # type: ignore[arg-type]
                str(kwargs["mode"]),
            ),
            selected_candidate_id=(
                "cand_control" if "control" in out_dir_name else "cand_treatment"
            ),
        )

    monkeypatch.setattr(sa, "run_one_cell", fake_run_one_cell)
    model_yaml = tmp_path / "merlin_mlp_wide.yaml"
    target_yaml = tmp_path / "host_cpu.yaml"
    model_yaml.write_text("noop\n")
    target_yaml.write_text("noop\n")

    pack = sa.run_subsystem_ablation(
        [sa.SubsystemAblationSpec(
            model_yaml=model_yaml,
            target_yaml=target_yaml,
            subsystem_flag="kernels.codegen_fallback",
        )],
        out_root=tmp_path / "out",
        commit="testcommit",
        subsystem_flag="kernels.codegen_fallback",
    )

    summary = pack.summary()
    assert summary["row_count"] == 1
    assert summary["candidate_changed_count"] == 1
    assert summary["outcome_changed_count"] == 0
    raw = pack.to_dict()
    assert raw["schema_version"] == "subsystem_ablation_pack_v1"
    assert raw["subsystem_flag"] == "kernels.codegen_fallback"
    assert len(raw["rows"]) == 1
    assert raw["rows"][0]["deltas"]["candidate_changed"] is True
