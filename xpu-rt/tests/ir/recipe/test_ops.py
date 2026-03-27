"""Tests for Recipe IR operations."""

from __future__ import annotations

import pytest
from compgen.ir.recipe.ops import (
    AssignDevice,
    MatchRegion,
    RequestKernelSearch,
    RequireCheck,
    SetObjective,
    SetTileParams,
)
from compgen.llm.base import Objective


def test_match_region() -> None:
    op = MatchRegion(region_id="matmul_0", op_filter="linalg.matmul")
    assert op.region_id == "matmul_0"
    assert op.op_filter == "linalg.matmul"


def test_set_objective() -> None:
    op = SetObjective(objective=Objective.LATENCY)
    assert op.objective == Objective.LATENCY


def test_set_tile_params() -> None:
    op = SetTileParams(region_id="r0", tile_sizes=(128, 128, 32))
    assert op.tile_sizes == (128, 128, 32)


def test_assign_device() -> None:
    op = AssignDevice(region_id="r0", device_index=1, reason="GPU has tensor cores")
    assert op.device_index == 1


def test_request_kernel_search() -> None:
    op = RequestKernelSearch(region_id="r0", backend="triton", search_budget=20)
    assert op.search_budget == 20


def test_require_check() -> None:
    op = RequireCheck(region_id="r0", check_type="translation_validation")
    assert op.check_type == "translation_validation"


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_recipe_ops_are_frozen() -> None:
    """Recipe IR ops should be immutable."""
