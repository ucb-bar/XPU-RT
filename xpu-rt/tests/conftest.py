"""Shared pytest fixtures for CompGen tests.

All fixtures return lightweight objects suitable for unit testing.
No GPU or network access required unless marked with appropriate pytest markers.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Path to the repository root."""
    return Path(__file__).parent.parent


@pytest.fixture
def examples_dir(repo_root: Path) -> Path:
    """Path to the examples directory."""
    return repo_root / "examples"


@pytest.fixture
def sample_target_profile_path(examples_dir: Path) -> Path:
    """Path to the CUDA A100 example target profile."""
    return examples_dir / "target_profiles" / "cuda_a100.yaml"


@pytest.fixture
def sample_multi_device_profile_path(examples_dir: Path) -> Path:
    """Path to the multi-device example target profile."""
    return examples_dir / "target_profiles" / "multi_device.yaml"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for test artifacts."""
    output = tmp_path / "compgen_output"
    output.mkdir()
    return output


@pytest.fixture
def tmp_bundle_dir(tmp_output_dir: Path) -> Path:
    """Temporary bundle directory."""
    bundle = tmp_output_dir / "bundle"
    bundle.mkdir()
    return bundle
