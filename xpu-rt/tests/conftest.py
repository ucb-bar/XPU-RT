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

import importlib
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Marker-driven auto-skip: `requires_gpu`, `requires_mlir`, `requires_ray`.
# ---------------------------------------------------------------------------
# These markers are declared in pyproject.toml and used throughout the
# suite. Rather than making every test author manage skip-logic manually,
# we check capability once per session and have the collection hook skip
# any marked test whose capability isn't present. That means:
# - On a CPU-only laptop, ``pytest tests/`` green-passes because
#   ``requires_gpu`` tests auto-skip.
# - On a GPU CI runner, ``pytest -m requires_gpu tests/`` runs them.
# Capability probes are cheap (single import or attribute check) and
# cached in module-level dicts so they fire once per session, not per test.


def _probe_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _probe_mlir() -> bool:
    try:
        importlib.import_module("xdsl")
        return True
    except Exception:
        return False


def _probe_ray() -> bool:
    try:
        importlib.import_module("ray")
        return True
    except Exception:
        return False


_MARKER_PROBES = {
    "requires_gpu": _probe_gpu,
    "requires_mlir": _probe_mlir,
    "requires_ray": _probe_ray,
}


_CAPABILITY_CACHE: dict[str, bool] = {}


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    """Auto-skip marker-guarded tests when the capability isn't present."""
    for marker_name, probe in _MARKER_PROBES.items():
        if marker_name not in _CAPABILITY_CACHE:
            _CAPABILITY_CACHE[marker_name] = probe()
    for item in items:
        for marker_name, present in _CAPABILITY_CACHE.items():
            if present:
                continue
            if item.get_closest_marker(marker_name) is not None:
                item.add_marker(pytest.mark.skip(reason=f"{marker_name}: capability not available on this host"))


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
