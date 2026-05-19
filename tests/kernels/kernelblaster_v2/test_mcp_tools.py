"""MCP-handler tests for the target-knowledge + kernel-blast tools.

Drives each handler with a synthetic session manager (None — the
handlers don't use it) and asserts the response shape + on-disk side
effects. Hermetic; no Gemini calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpu_rt.kernels.kernelblaster_v2.strategy_db import StrategyDB
from xpu_rt.memory import target_knowledge as tk
from xpu_rt.mcp.tools import target_knowledge as t_tools
from xpu_rt.mcp.tools import kernel_blast as kb_tools


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XPU_RT_KNOWLEDGE_DIR", str(tmp_path / "targets"))
    monkeypatch.setenv("XPU_RT_INGEST_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("XPU_RT_GEMINI_USAGE_DIR", str(tmp_path / "usage"))
    monkeypatch.setenv("XPU_RT_REPO_ROOT", str(tmp_path))
    return tmp_path


def _seed_card(target_id: str = "demo_target") -> tk.TargetKnowledgeCard:
    return tk.save(
        tk.TargetKnowledgeCard(
            target_id=target_id,
            target_profile_ref=f"configs/targets/{target_id}.yaml",
            hardware_spec=tk.HardwareSpec(isa_family="rocc-systolic"),
        )
    )


# ---------------------------------------------------------------------------
# target_knowledge: read-only
# ---------------------------------------------------------------------------


def test_target_list_returns_saved_targets(isolated: Path) -> None:
    out = t_tools.xpu_rt_target_list(None)
    assert out["ok"] and out["targets"] == []
    _seed_card("a")
    _seed_card("b")
    out = t_tools.xpu_rt_target_list(None)
    assert out["targets"] == ["a", "b"]
    assert out["count"] == 2


def test_target_show_missing_returns_error(isolated: Path) -> None:
    out = t_tools.xpu_rt_target_show(None, target_id="nope")
    assert out["ok"] is False
    assert "no card" in out["error"]


def test_target_show_returns_full_card(isolated: Path) -> None:
    _seed_card()
    out = t_tools.xpu_rt_target_show(None, target_id="demo_target")
    assert out["ok"]
    assert out["card"]["target_id"] == "demo_target"
    assert out["lesson_count"] == 0


def test_target_lessons_filters(isolated: Path) -> None:
    card = _seed_card()
    tk.append_lesson(
        card,
        tk.Lesson(
            timestamp="2026-05-15T00:00:00+00:00",
            archetype="COMPUTE_TILED",
            dtype_class="i8",
            layout_kind="row_major",
            op_family="matmul",
            action="tile-K=64",
            measured_gain=1.4,
        ),
    )
    tk.append_lesson(
        card,
        tk.Lesson(
            timestamp="2026-05-15T00:00:01+00:00",
            archetype="POINTWISE",
            dtype_class="fp32",
            layout_kind="row_major",
            op_family="relu",
            action="fuse-x",
            measured_gain=1.1,
        ),
    )
    out = t_tools.xpu_rt_target_lessons(
        None, target_id="demo_target", op_family="matmul"
    )
    assert out["ok"]
    assert len(out["lessons"]) == 1
    assert out["lessons"][0]["action"] == "tile-K=64"


def test_target_known_seeds_lists_both(isolated: Path) -> None:
    out = t_tools.xpu_rt_target_known_seeds(None)
    assert out["ok"]
    by_seed = {entry.get("seed"): entry for entry in out["seeds"]}
    assert {"gemmini", "saturn"}.issubset(by_seed.keys())
    # Each entry should have at least target_id + source_count.
    for entry in by_seed.values():
        if "error" in entry:
            continue
        assert "target_id" in entry
        assert "source_count" in entry


# ---------------------------------------------------------------------------
# target_knowledge: prepare + apply
# ---------------------------------------------------------------------------


def test_prepare_ingest_chunks_inline_manifest(isolated: Path) -> None:
    md = isolated / "src.md"
    md.write_text("hello there\n\nsecond paragraph", encoding="utf-8")
    code = isolated / "kernel.c"
    code.write_text("// exemplar\n", encoding="utf-8")
    out = t_tools.xpu_rt_target_prepare_ingest(
        None,
        manifest_inline={
            "target_id": "demo_target",
            "target_profile_ref": "configs/targets/demo_target.yaml",
            "isa_family": "rocc-systolic",
            "sources": [
                {"locator": str(md), "kind": "path", "role": "auto"},
                {"locator": str(code), "kind": "path", "role": "examples", "tags": ["m"]},
            ],
        },
    )
    assert out["ok"]
    assert out["chunk_count"] >= 1
    # The .c file is in exemplars_to_copy, not chunks.
    assert any(e["locator"].endswith("kernel.c") for e in out["exemplars_to_copy"])
    assert all(not c["source_locator"].endswith("kernel.c") for c in out["chunks"])
    # Schema and prompt are surfaced.
    assert "bucket" in out["router_response_schema"]["properties"]
    assert "router_system_prompt" in out and out["router_system_prompt"]


def test_apply_routing_folds_records_into_card(isolated: Path) -> None:
    code = isolated / "matmul.c"
    code.write_text("// matmul exemplar\n", encoding="utf-8")
    out = t_tools.xpu_rt_target_apply_routing(
        None,
        target_id="demo_target",
        target_profile_ref="configs/targets/demo_target.yaml",
        isa_family="rocc-systolic",
        routed_results=[
            {
                "bucket": "isa",
                "summary_md": "### mvin\nLoad scratchpad",
                "instructions": [
                    {"mnemonic": "mvin", "signature": "rs1, rs2", "funct_code": 2}
                ],
            },
            {
                "bucket": "intrinsics",
                "summary_md": "C macros",
                "intrinsics": [
                    {
                        "name": "gemmini_mvin",
                        "c_signature": "#define gemmini_mvin(d, s)",
                        "summary": "DMA",
                    }
                ],
            },
            {
                "bucket": "constraints",
                "summary_md": "must be DIM-aligned",
                "constraints": ["scratchpad addr must be DIM-aligned"],
            },
        ],
        exemplars_to_copy=[
            {"locator": str(code), "tags": ["matmul"], "role": "examples"}
        ],
    )
    assert out["ok"]
    assert out["instructions_added"] == 1
    assert out["intrinsics_added"] == 1
    assert out["constraints_added"] == 1
    assert out["exemplars_added"] == 1
    assert set(out["buckets_touched"]) >= {"isa", "intrinsics", "constraints"}

    card = tk.load("demo_target")
    mnemonics = {i.mnemonic for i in card.hardware_spec.instructions}
    assert "mvin" in mnemonics
    assert (card.exemplars_dir / "matmul.c").exists()
    # Bucket markdown materialized.
    assert card.bucket_path("isa").exists()
    assert "mvin" in card.bucket_path("isa").read_text()


# ---------------------------------------------------------------------------
# kernel_blast: read-only
# ---------------------------------------------------------------------------


def test_blast_lessons_for_region_returns_matching(isolated: Path) -> None:
    card = _seed_card()
    tk.append_lesson(
        card,
        tk.Lesson(
            timestamp="2026-05-15T00:00:00+00:00",
            archetype="COMPUTE_TILED",
            dtype_class="mixed",
            layout_kind="row_major",
            op_family="matmul",
            action="tile-K=64",
            measured_gain=1.4,
        ),
    )
    out = kb_tools.xpu_rt_blast_lessons_for_region(
        None,
        contract={
            "op_family": "matmul",
            "input_shapes": [[64, 64], [64, 64]],
            "output_shapes": [[64, 64]],
            "dtypes": ["i8", "i32"],
            "layout": "row_major",
            "target_name": "demo_target",
        },
    )
    assert out["ok"]
    assert len(out["lessons"]) == 1
    assert out["lessons"][0]["action"] == "tile-K=64"


def test_blast_strategies_for_target(isolated: Path) -> None:
    card = _seed_card()
    db = StrategyDB.for_card(card)
    from xpu_rt.kernels.kernelblaster_v2.contract_state import derive_state
    from xpu_rt.kernels.provider import KernelContract

    contract = KernelContract(
        op_family="matmul",
        dtypes=("i8", "i32"),
        layout="row_major",
        target_name="demo_target",
    )
    db.record(
        state=derive_state(contract, target_id="demo_target"),
        action="tile-K=64",
        accepted=True,
        speedup=1.4,
    )
    out = kb_tools.xpu_rt_blast_strategies_for_target(
        None, target_id="demo_target"
    )
    assert out["ok"]
    assert out["rows"]
    assert out["rows"][0]["action"] == "tile-K=64"


# ---------------------------------------------------------------------------
# kernel_blast: prepare + apply
# ---------------------------------------------------------------------------


def test_prepare_propose_returns_prompt_bundle(isolated: Path) -> None:
    _seed_card()
    out = kb_tools.xpu_rt_blast_prepare_propose(
        None,
        contract={
            "op_family": "matmul",
            "input_shapes": [[128, 256], [256, 64]],
            "output_shapes": [[128, 64]],
            "dtypes": ["i8", "i32"],
            "layout": "row_major",
            "target_name": "demo_target",
        },
    )
    assert out["ok"]
    assert "system" in out and "user" in out and "schema" in out
    assert "matmul" in out["user"]
    assert out["state"]["target_id"] == "demo_target"


def test_apply_response_records_strategy_and_lesson(isolated: Path) -> None:
    _seed_card()
    contract_payload = {
        "op_family": "matmul",
        "input_shapes": [[64, 64], [64, 64]],
        "output_shapes": [[64, 64]],
        "dtypes": ["i8", "i32"],
        "layout": "row_major",
        "target_name": "demo_target",
    }
    out = kb_tools.xpu_rt_blast_apply_response(
        None,
        contract=contract_payload,
        response={"action": "tile-K=64", "kernel_code": "// k", "language": "c"},
        evaluation={"correct": True, "score": 1.5, "diff_summary": ""},
    )
    assert out["ok"]
    assert out["lesson_written"] is True
    assert out["strategy_row"]["action"] == "tile-K=64"

    card = tk.load("demo_target")
    actions = {l.action for l in tk.iter_lessons(card)}
    assert "tile-K=64" in actions


def test_apply_response_no_lesson_on_reject(isolated: Path) -> None:
    _seed_card()
    out = kb_tools.xpu_rt_blast_apply_response(
        None,
        contract={
            "op_family": "matmul",
            "dtypes": ["i8"],
            "layout": "row_major",
            "target_name": "demo_target",
        },
        response={"action": "v1", "kernel_code": "", "language": ""},
        evaluation={"correct": False, "score": 0.0, "diff_summary": "wrong"},
    )
    assert out["ok"]
    assert out["lesson_written"] is False
