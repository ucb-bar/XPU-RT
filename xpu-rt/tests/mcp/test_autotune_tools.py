"""Tests for ``compgen.mcp.tools.autotune`` (W8.3)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from compgen.mcp.session import SessionManager
from compgen.mcp.tools.autotune import (
    AUTOTUNE_TOOLS,
    list_pending_autotune_trials,
    lookup_autotune_pick,
    register_autotune_pick,
    request_autotune_trial,
)


@pytest.fixture
def isolated_autotune_dir(tmp_path: Path, monkeypatch):
    """Redirect the on-disk autotune cache so the test doesn't write
    to the real ``~/.compgen/autotune/``."""
    monkeypatch.setenv("COMPGEN_AUTOTUNE_CACHE", str(tmp_path / "autotune"))
    yield tmp_path / "autotune"


@pytest.fixture
def sm(tmp_path: Path) -> SessionManager:
    s = SessionManager(scratch_root=tmp_path / "compgen_mcp")
    s.open(session_id="sess1")
    return s


def test_autotune_tools_registered_with_expected_names() -> None:
    names = {t["name"] for t in AUTOTUNE_TOOLS}
    assert names == {
        "request_autotune_trial", "register_autotune_pick",
        "lookup_autotune_pick", "list_pending_autotune_trials",
    }


def test_autotune_tools_in_all_tools_bundle() -> None:
    from compgen.mcp.tools import ALL_TOOLS
    names = {t["name"] for t in ALL_TOOLS}
    for n in ("request_autotune_trial", "register_autotune_pick",
              "lookup_autotune_pick", "list_pending_autotune_trials"):
        assert n in names


def test_request_then_register_then_lookup(sm, isolated_autotune_dir) -> None:
    out = request_autotune_trial(
        sm, session_id="sess1",
        kernel_qualname="matmul_kernel",
        key_repr="(512,512,512)",
        candidate_configs=[{"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}],
        perf_target_us=100.0,
    )
    assert out["ok"] and not out["found_in_cache"]
    rid = out["request_id"]
    assert "BLOCK_M" in out["prompt"]

    pending = list_pending_autotune_trials(sm, session_id="sess1")
    assert pending["pending_count"] == 1

    reg = register_autotune_pick(
        sm, session_id="sess1", request_id=rid,
        kwargs={"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=8, num_stages=3,
        perf_us=42.0, notes="bigger M block won",
    )
    assert reg["ok"] and reg["cached_picks"] == 1

    lk = lookup_autotune_pick(
        sm, session_id="sess1",
        kernel_qualname="matmul_kernel", key_repr="(512,512,512)",
    )
    assert lk["found"]
    assert lk["pick"]["num_warps"] == 8
    assert lk["pick"]["kwargs"]["BLOCK_M"] == 128


def test_request_short_circuits_on_cached_pick(sm, isolated_autotune_dir) -> None:
    out = request_autotune_trial(
        sm, session_id="sess1",
        kernel_qualname="k1", key_repr="(64,)",
    )
    register_autotune_pick(
        sm, session_id="sess1", request_id=out["request_id"],
        kwargs={"BLOCK": 64}, num_warps=4,
    )
    out2 = request_autotune_trial(
        sm, session_id="sess1",
        kernel_qualname="k1", key_repr="(64,)",
    )
    assert out2["found_in_cache"] is True
    assert out2["pick"]["num_warps"] == 4


def test_register_persists_pick_to_disk(sm, isolated_autotune_dir) -> None:
    out = request_autotune_trial(
        sm, session_id="sess1",
        kernel_qualname="ondisk_kernel", key_repr="(128,128,128)",
    )
    register_autotune_pick(
        sm, session_id="sess1", request_id=out["request_id"],
        kwargs={"BLOCK_M": 64}, num_warps=2, num_stages=4,
        perf_us=15.0, notes="agent pick",
    )
    disk_file = isolated_autotune_dir / "ondisk_kernel.json"
    assert disk_file.exists()
    data = json.loads(disk_file.read_text())
    assert "(128,128,128)" in data
    assert data["(128,128,128)"]["num_warps"] == 2


def test_request_rehydrates_pick_from_disk(sm, isolated_autotune_dir) -> None:
    """Pre-seed the disk cache; first request must surface the pick."""
    isolated_autotune_dir.mkdir(parents=True, exist_ok=True)
    (isolated_autotune_dir / "seeded_kernel.json").write_text(json.dumps({
        "(32,)": {
            "kwargs": {"BLOCK": 32}, "num_warps": 8, "num_stages": 2,
            "num_ctas": 1, "maxnreg": None, "perf_us": 5.0,
            "notes": "seeded", "timestamp": 0.0,
        }
    }))
    out = request_autotune_trial(
        sm, session_id="sess1",
        kernel_qualname="seeded_kernel", key_repr="(32,)",
    )
    assert out["found_in_cache"] is True
    assert out["pick"]["num_warps"] == 8


def test_register_unknown_request_id_errors(sm, isolated_autotune_dir) -> None:
    res = register_autotune_pick(
        sm, session_id="sess1", request_id="nope",
        kwargs={"BLOCK": 1},
    )
    assert res["ok"] is False
    assert "unknown" in res["error"]


def test_lookup_miss_returns_found_false(sm, isolated_autotune_dir) -> None:
    res = lookup_autotune_pick(
        sm, session_id="sess1",
        kernel_qualname="never_tried", key_repr="(0,)",
    )
    assert res["found"] is False
