"""Tests for the cross-iteration bandit memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpu_rt.scheduling.loop_memory import (
    BanditArm,
    MemoryEntry,
    append_entry,
    canonical_workload_set_key,
    default_candidate_arms,
    load_entries,
    recommend_initial_arm,
    summarize_memory,
)


def _entry(
    *,
    target: str = "qrb5165",
    wkey: str = "dronet*1+yolov8n*1",
    run: str = "2026-01-01T00:00:00+00:00",
    it: int = 0,
    mco: int = 16,
    fgt: float = 0.3,
    converged: bool = True,
    err: float | None = 5.0,
) -> MemoryEntry:
    return MemoryEntry(
        target_id=target,
        workload_set_key=wkey,
        run_id=run,
        iteration=it,
        max_chunk_ops=mco,
        fusion_gain_threshold=fgt,
        solver_choice="cpsat",
        n_partitions=12,
        predicted_makespan_us=1000.0,
        measured_makespan_us=1000.0,
        abs_pct_error=err,
        was_converged=converged,
    )


def test_canonical_workload_set_key_dedupes_and_sorts() -> None:
    assert (
        canonical_workload_set_key(("dronet", "yolov8n", "dronet"))
        == "dronet*2+yolov8n*1"
    )
    assert canonical_workload_set_key(("yolov8n",)) == "yolov8n*1"


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    entries = [_entry(it=i, mco=8 + i) for i in range(3)]
    for e in entries:
        append_entry(e, tmp_path)
    got = load_entries(entries[0].target_id, entries[0].workload_set_key, tmp_path)
    assert len(got) == 3
    assert [g.max_chunk_ops for g in got] == [8, 9, 10]
    assert all(g.solver_choice == "cpsat" for g in got)


def test_recommend_initial_arm_picks_default_with_empty_memory(tmp_path: Path) -> None:
    arms = default_candidate_arms()
    chosen = recommend_initial_arm(
        target_id="qrb5165",
        workload_set_key="empty*1",
        candidate_arms=arms,
        memory_dir=tmp_path,
        rng_seed=0,
    )
    sorted_arms = sorted(arms, key=lambda a: a.max_chunk_ops)
    assert chosen == sorted_arms[len(sorted_arms) // 2]


def test_recommend_initial_arm_prefers_better_arm_after_memory(tmp_path: Path) -> None:
    target, wkey = "qrb5165", "favoured*1"
    # Better arm: max_chunk_ops=8, fgt=0.3 — converges with 3% error.
    for i in range(10):
        append_entry(
            _entry(target=target, wkey=wkey, it=i, mco=8, fgt=0.3, converged=True, err=3.0),
            tmp_path,
        )
    # Other arms: not converged with 25% error — pile failures onto them.
    for i, mco in enumerate((4, 16, 32, 64)):
        for j in range(10):
            append_entry(
                _entry(
                    target=target,
                    wkey=wkey,
                    it=j,
                    mco=mco,
                    fgt=0.3,
                    converged=False,
                    err=25.0,
                ),
                tmp_path,
            )
    arms = default_candidate_arms()
    picks = [
        recommend_initial_arm(
            target_id=target,
            workload_set_key=wkey,
            candidate_arms=arms,
            memory_dir=tmp_path,
            rng_seed=seed,
        )
        for seed in range(10)
    ]
    favoured = BanditArm(max_chunk_ops=8, fusion_gain_threshold=0.3)
    n_fav = sum(1 for p in picks if p == favoured)
    assert n_fav >= 7, f"expected >= 7/10 picks of favoured arm, got {n_fav}"


def test_thompson_sampling_explores(tmp_path: Path) -> None:
    target, wkey = "qrb5165", "thin*1"
    # Only 3 entries on the favoured arm — exploration should still pick
    # other arms on at least one of 10 seeds.
    for i in range(3):
        append_entry(
            _entry(target=target, wkey=wkey, it=i, mco=8, fgt=0.3, converged=True, err=3.0),
            tmp_path,
        )
    arms = default_candidate_arms()
    picks = {
        recommend_initial_arm(
            target_id=target,
            workload_set_key=wkey,
            candidate_arms=arms,
            memory_dir=tmp_path,
            rng_seed=seed,
        )
        for seed in range(10)
    }
    favoured = BanditArm(max_chunk_ops=8, fusion_gain_threshold=0.3)
    assert favoured in picks
    assert len(picks) > 1, f"expected exploration, all picks collapsed to {picks}"


def test_summarize_memory_shape(tmp_path: Path) -> None:
    target, wkey = "qrb5165", "sumcheck*1"
    for mco, err in ((8, 4.0), (8, 6.0), (16, 20.0)):
        append_entry(
            _entry(
                target=target,
                wkey=wkey,
                mco=mco,
                fgt=0.3,
                converged=(err < 10.0),
                err=err,
            ),
            tmp_path,
        )
    summary = summarize_memory(target, wkey, tmp_path)
    assert summary["n_entries"] == 3
    assert summary["n_converged"] == 2
    assert summary["best_arm"] == {"max_chunk_ops": 8, "fusion_gain_threshold": 0.3}
    assert summary["best_arm_mean_error_pct"] == pytest.approx(5.0)
