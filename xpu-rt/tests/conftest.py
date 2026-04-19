"""Shared pytest fixtures for CompGen tests.

All fixtures return lightweight objects suitable for unit testing.
No GPU or network access required unless marked with appropriate pytest markers.

**Home-directory isolation.** The LLM driver and MCP server default
their transcript/extensions directories to ``~/.compgen/...``. Test
runs must never write there. The ``_compgen_home_isolation``
autouse fixture redirects both defaults to a per-session tmp path and
disables cross-session graduation + local extension loading so the
process-wide registry stays pristine.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _compgen_home_isolation():
    """Stop tests from polluting ``~/.compgen``.

    Sets env vars the LLM driver + MCP transcript recorder consult,
    and disables local extension loading + cross-session graduation.
    """
    tmp_home = Path(tempfile.mkdtemp(prefix="compgen-test-home-"))
    (tmp_home / "transcripts").mkdir(parents=True, exist_ok=True)
    (tmp_home / "extensions").mkdir(parents=True, exist_ok=True)

    saved = {}
    for key, value in (
        ("COMPGEN_SESSION_DIR", str(tmp_home / "transcripts")),
        ("COMPGEN_EXTENSIONS_DIR", str(tmp_home / "extensions")),
        ("COMPGEN_DISABLE_LOCAL_EXTENSIONS", "1"),
        ("COMPGEN_DISABLE_CROSS_SESSION_GRADUATION", "1"),
        ("COMPGEN_DISABLE_AUTHORED_GRADUATION", "1"),
    ):
        saved[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield tmp_home
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


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
