"""Smoke test: verify package imports and version string."""

from __future__ import annotations


def test_import_compgen() -> None:
    """compgen package should be importable."""
    import compgen

    assert hasattr(compgen, "__version__")


def test_version_string() -> None:
    """Version should be a valid semver-like string."""
    from compgen import __version__

    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_version_value() -> None:
    """Version pinned to the value in ``pyproject.toml``."""
    from compgen import __version__

    assert __version__ == "0.2.0"
