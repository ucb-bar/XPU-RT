"""Round-trip + on-disk-layout tests for :mod:`xpu_rt.memory.target_knowledge`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpu_rt.memory import target_knowledge as tk


@pytest.fixture
def isolated_knowledge_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XPU_RT_KNOWLEDGE_DIR", str(tmp_path))
    return tmp_path


def _sample_card(target_id: str = "test_target") -> tk.TargetKnowledgeCard:
    return tk.TargetKnowledgeCard(
        target_id=target_id,
        target_profile_ref=f"configs/targets/{target_id}.yaml",
        hardware_spec=tk.HardwareSpec(
            isa_family="rocc-systolic",
            parameters=(
                tk.ParameterRange(name="meshRows", min_value=4, max_value=32, default=16, unit="PEs"),
                tk.ParameterRange(
                    name="dataflow",
                    values=("weight_stationary", "output_stationary"),
                    default="weight_stationary",
                ),
            ),
            memory_tiers=(
                tk.MemoryTierSpec(name="scratchpad", kind="scratchpad", size_bytes=262144),
                tk.MemoryTierSpec(name="accumulator", kind="accumulator", size_bytes=65536),
            ),
            instructions=(
                tk.ISAInstruction(mnemonic="mvin", signature="rs1, rs2", funct_code=2),
                tk.ISAInstruction(mnemonic="mvout", signature="rs1, rs2", funct_code=3),
            ),
            intrinsics=(
                tk.IntrinsicSignature(
                    name="gemmini_mvin",
                    c_signature="void gemmini_mvin(void *src, uint32_t spad_addr)",
                    summary="DMA from DRAM into scratchpad",
                ),
            ),
            dataflow_modes=("weight_stationary", "output_stationary"),
            constraints=("scratchpad addr must be 16B aligned",),
        ),
        exemplars=(
            tk.KernelExemplar(
                name="gemm_i8_int32",
                op_family="gemm",
                path="gemm_i8_int32.c",
                language="c",
                tags=("int8", "matmul"),
            ),
        ),
        docs=(
            tk.DocSource(
                locator="https://github.com/example/repo/README.md",
                kind="url",
                sha256="deadbeef",
                fetched_at="2026-05-15T12:00:00+00:00",
                bucket="architecture",
                bytes=12345,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_save_then_load_round_trip(isolated_knowledge_dir: Path) -> None:
    saved = tk.save(_sample_card())
    assert saved.created_at
    assert saved.updated_at
    loaded = tk.load("test_target")
    assert loaded.target_id == "test_target"
    assert loaded.target_profile_ref == "configs/targets/test_target.yaml"
    assert loaded.hardware_spec.isa_family == "rocc-systolic"
    assert len(loaded.hardware_spec.parameters) == 2
    assert loaded.hardware_spec.parameters[0].name == "meshRows"
    assert loaded.hardware_spec.instructions[0].mnemonic == "mvin"
    assert loaded.hardware_spec.intrinsics[0].name == "gemmini_mvin"
    assert loaded.exemplars[0].op_family == "gemm"
    assert loaded.docs[0].sha256 == "deadbeef"


def test_save_stamps_timestamps_and_preserves_created_at(
    isolated_knowledge_dir: Path,
) -> None:
    first = tk.save(_sample_card())
    created = first.created_at
    second = tk.save(tk.load("test_target"))
    assert second.created_at == created
    assert second.updated_at >= first.updated_at


def test_load_unknown_id_raises(isolated_knowledge_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tk.load("does-not-exist")


def test_schema_version_mismatch_rejected(isolated_knowledge_dir: Path) -> None:
    tk.save(_sample_card())
    raw = json.loads((isolated_knowledge_dir / "test_target" / "target_card.json").read_text())
    raw["schema_version"] = "xpu_rt_target_knowledge_v999"
    (isolated_knowledge_dir / "test_target" / "target_card.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unsupported target knowledge schema"):
        tk.load("test_target")


# ---------------------------------------------------------------------------
# On-disk layout
# ---------------------------------------------------------------------------


def test_paths_point_under_target_dir(isolated_knowledge_dir: Path) -> None:
    card = tk.save(_sample_card())
    assert card.root == isolated_knowledge_dir / "test_target"
    assert card.card_path.exists()
    assert card.lessons_path == card.root / "lessons.jsonl"
    assert card.strategies_path == card.root / "strategies.json"
    assert card.exemplars_dir.exists()
    assert card.docs_dir.exists()
    assert card.bucket_path("isa") == card.root / "isa.md"


def test_bucket_path_rejects_unknown_bucket(isolated_knowledge_dir: Path) -> None:
    card = tk.save(_sample_card())
    with pytest.raises(ValueError, match="unknown bucket"):
        card.bucket_path("not_a_bucket")


def test_invalid_target_id_rejected(isolated_knowledge_dir: Path) -> None:
    for bad in ("", "../escape", "with/slash"):
        with pytest.raises(ValueError):
            tk.target_dir(bad)


def test_list_targets_finds_saved_cards(isolated_knowledge_dir: Path) -> None:
    tk.save(_sample_card("alpha"))
    tk.save(_sample_card("beta"))
    # An orphan dir without target_card.json must be ignored.
    (isolated_knowledge_dir / "orphan").mkdir()
    assert tk.list_targets() == ["alpha", "beta"]


def test_exists_predicate(isolated_knowledge_dir: Path) -> None:
    assert not tk.exists("nope")
    tk.save(_sample_card("yep"))
    assert tk.exists("yep")


# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------


def test_lesson_append_and_iter(isolated_knowledge_dir: Path) -> None:
    card = tk.save(_sample_card())
    tk.append_lesson(
        card,
        tk.Lesson(
            timestamp="2026-05-15T12:00:00+00:00",
            archetype="COMPUTE_TILED",
            dtype_class="int8",
            layout_kind="BLOCKED",
            op_family="gemm",
            action="tile-K=64",
            measured_gain=1.4,
        ),
    )
    tk.append_lesson(
        card,
        tk.Lesson(
            timestamp="2026-05-15T12:01:00+00:00",
            archetype="COMPUTE_TILED",
            dtype_class="int8",
            layout_kind="BLOCKED",
            op_family="gemm",
            action="use-mvin-stride",
            measured_gain=1.1,
        ),
    )
    rows = list(tk.iter_lessons(card))
    assert len(rows) == 2
    assert rows[0].action == "tile-K=64"
    assert rows[1].measured_gain == pytest.approx(1.1)


def test_iter_lessons_skips_malformed_lines(
    isolated_knowledge_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    card = tk.save(_sample_card())
    card.lessons_path.write_text(
        '{"timestamp":"t","archetype":"X","dtype_class":"i8","layout_kind":"L","op_family":"o","action":"a","measured_gain":1.0,"sample_count":1,"notes":""}\n'
        "{not json}\n"
        '{"timestamp":"t2","archetype":"X","dtype_class":"i8","layout_kind":"L","op_family":"o","action":"b","measured_gain":1.0,"sample_count":1,"notes":""}\n'
    )
    rows = list(tk.iter_lessons(card))
    assert [r.action for r in rows] == ["a", "b"]


# ---------------------------------------------------------------------------
# DerivationRule round-trip + prompt-builder integration
# ---------------------------------------------------------------------------


def test_derivation_rule_round_trip(isolated_knowledge_dir: Path) -> None:
    rule = tk.DerivationRule(
        name="spad_tile_budget",
        symbolic="(tile_I + tile_J) * tile_K * DIM ≤ sp_capacity / 2",
        concrete_value=8192.0,
        unit="rows",
        derivation="256 KB / 16 B-per-row / 2 (double-buffered)",
        applies_to="tile budgeting",
        how_to_apply="(tile_I + tile_J) * tile_K ≤ 512",
    )
    card = _sample_card("rules_target")
    card_with_rules = tk.TargetKnowledgeCard(
        target_id=card.target_id,
        target_profile_ref=card.target_profile_ref,
        hardware_spec=tk.HardwareSpec(
            isa_family=card.hardware_spec.isa_family,
            parameters=card.hardware_spec.parameters,
            memory_tiers=card.hardware_spec.memory_tiers,
            instructions=card.hardware_spec.instructions,
            intrinsics=card.hardware_spec.intrinsics,
            dataflow_modes=card.hardware_spec.dataflow_modes,
            constraints=card.hardware_spec.constraints,
            derivation_rules=(rule,),
        ),
        exemplars=card.exemplars,
        docs=card.docs,
    )
    tk.save(card_with_rules)
    reloaded = tk.load("rules_target")
    assert len(reloaded.hardware_spec.derivation_rules) == 1
    r2 = reloaded.hardware_spec.derivation_rules[0]
    assert r2.name == "spad_tile_budget"
    assert r2.concrete_value == 8192.0
    assert r2.how_to_apply == "(tile_I + tile_J) * tile_K ≤ 512"


def test_prompt_builder_renders_derivation_rules(
    isolated_knowledge_dir: Path,
) -> None:
    """The prompt MUST surface concrete bounds in their own dedicated section."""
    from xpu_rt.kernels.kernelblaster_v2.contract_state import derive_state
    from xpu_rt.kernels.kernelblaster_v2.prompt_builder import PromptBuilder
    from xpu_rt.kernels.provider import KernelContract

    base = _sample_card()
    rule = tk.DerivationRule(
        name="spad_tile_budget",
        symbolic="(tile_I + tile_J) * tile_K * DIM ≤ sp_capacity / 2",
        concrete_value=8192.0,
        unit="rows",
        derivation="256 KB / 16 B-per-row / 2 (double-buffered)",
        applies_to="tile budgeting",
        how_to_apply="(tile_I + tile_J) * tile_K ≤ 512",
    )
    card = tk.TargetKnowledgeCard(
        target_id=base.target_id,
        target_profile_ref=base.target_profile_ref,
        hardware_spec=tk.HardwareSpec(
            isa_family=base.hardware_spec.isa_family,
            parameters=base.hardware_spec.parameters,
            memory_tiers=base.hardware_spec.memory_tiers,
            instructions=base.hardware_spec.instructions,
            intrinsics=base.hardware_spec.intrinsics,
            dataflow_modes=base.hardware_spec.dataflow_modes,
            constraints=base.hardware_spec.constraints,
            derivation_rules=(rule,),
        ),
        exemplars=base.exemplars,
        docs=base.docs,
    )
    saved = tk.save(card)
    contract = KernelContract(
        op_family="matmul",
        input_shapes=((64, 720), (720, 320)),
        output_shapes=((64, 320),),
        dtypes=("i8", "i8", "i32"),
        layout="row_major",
        target_name=saved.target_id,
    )
    state = derive_state(contract, target_id=saved.target_id)
    bundle = PromptBuilder(card=saved).build(contract=contract, state=state)
    # The concrete bound must appear verbatim in the user prompt.
    assert "8192" in bundle.user
    assert "(tile_I + tile_J) * tile_K ≤ 512" in bundle.user
    assert "Sizing constraints (worked out" in bundle.user
