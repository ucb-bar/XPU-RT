"""Tests for accelerator dialect verification."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_valid_accel_ops_pass() -> None:
    """Well-formed accel ops should pass verification."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_mismatched_event_fails() -> None:
    """DMAWait on non-existent event should fail."""
