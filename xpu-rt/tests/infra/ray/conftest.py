"""Shared fixtures for Ray integration tests.

All Ray tests use ``pytest.importorskip("ray")`` so they are
automatically skipped when Ray is not installed.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def ray_cluster():
    """Start a local Ray cluster for testing."""
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(num_cpus=4, namespace="compgen_test")
    yield ray
    ray.shutdown()
