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


def test_recipe_ops_are_frozen() -> None:
    """Recipe IR ops should be immutable."""
    op = MatchRegion(region_id="matmul_0", op_filter="linalg.matmul")
    with pytest.raises(AttributeError):
        op.region_id = "changed"  # type: ignore[misc]

    tile_op = SetTileParams(region_id="r0", tile_sizes=(128, 128, 32))
    with pytest.raises(AttributeError):
        tile_op.tile_sizes = (64, 64)  # type: ignore[misc]

    device_op = AssignDevice(region_id="r0", device_index=1, reason="GPU")
    with pytest.raises(AttributeError):
        device_op.device_index = 2  # type: ignore[misc]

    kernel_op = RequestKernelSearch(region_id="r0", backend="triton", search_budget=20)
    with pytest.raises(AttributeError):
        kernel_op.search_budget = 100  # type: ignore[misc]

    check_op = RequireCheck(region_id="r0", check_type="translation_validation")
    with pytest.raises(AttributeError):
        check_op.check_type = "differential"  # type: ignore[misc]
