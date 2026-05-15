"""KernelBlaster GPU sidecar lifecycle tests.

The light tests below run by default — they cover the typed-error
paths that don't need KB itself. The heavy ``test_real_start`` test
boots an actual GPU sidecar on localhost and is gated on the KB repo
+ the ``kernelblaster-sidecar`` extras being importable. The CI lane
that does not have KB installed will skip it cleanly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from compgen.kernels.kernelblaster_sidecar import (
    DEFAULT_PORT,
    KernelBlasterSidecar,
    SidecarUnavailable,
    _check_imports,
    _port_in_use,
    _resolve_repo_root,
)


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


def test_check_imports_all_present():
    ok, missing = _check_imports(["json", "os", "sys"])
    assert ok and missing == ""


def test_check_imports_first_missing_wins():
    ok, missing = _check_imports(["json", "definitely_not_a_module_xyz", "os"])
    assert not ok
    assert missing == "definitely_not_a_module_xyz"


def test_resolve_repo_root_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPGEN_KERNELBLASTER_ROOT", str(tmp_path / "does-not-exist"))
    monkeypatch.chdir(tmp_path)
    assert _resolve_repo_root(None) is None


def test_resolve_repo_root_honours_explicit(tmp_path):
    assert _resolve_repo_root(tmp_path).resolve() == tmp_path.resolve()


def test_resolve_repo_root_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPGEN_KERNELBLASTER_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # so the conventional path doesn't shadow
    assert _resolve_repo_root(None).resolve() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Typed-error paths (no subprocess)
# ---------------------------------------------------------------------------


def test_start_repo_not_found_is_typed(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPGEN_KERNELBLASTER_ROOT", str(tmp_path / "nope"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SidecarUnavailable) as excinfo:
        KernelBlasterSidecar.start()
    assert excinfo.value.reason == "repo_not_found"


def test_start_port_in_use_is_typed(tmp_path, monkeypatch):
    kb = tmp_path / "kb"
    kb.mkdir()
    monkeypatch.setenv("COMPGEN_KERNELBLASTER_ROOT", str(kb))
    # Take port 0 (let OS pick free) then re-use that port for the
    # start() call so it conflicts.
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.listen(1)
        with pytest.raises(SidecarUnavailable) as excinfo:
            KernelBlasterSidecar.start(port=port)
        # Either port_in_use or missing_dep (if fastapi isn't here yet)
        # is acceptable — both are typed.
        assert excinfo.value.reason.startswith(("port_in_use:", "missing_dep:"))


def test_port_in_use_helper():
    # Port 0 is always free for binding; can't be in use.
    # Use a high port unlikely to clash.
    assert _port_in_use(0) in (False,)


# ---------------------------------------------------------------------------
# Real subprocess boot — only when KB repo + sidecar deps are present
# ---------------------------------------------------------------------------


def _kb_available() -> bool:
    repo = Path("third_party/kernelblaster")
    if not repo.exists():
        return False
    for name in ("fastapi", "uvicorn", "pydantic", "loguru", "dotenv"):
        if importlib.util.find_spec(name) is None:
            return False
    return True


@pytest.mark.skipif(
    not _kb_available(),
    reason="KB repo + `uv sync --extra kernelblaster-sidecar` required",
)
def test_real_start_and_health():
    """End-to-end: spawn sidecar, hit /health, capture receipt, tear down."""
    # Use a non-default port to avoid clashing with any persistent sidecar.
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    # Port now closed; KB can grab it.

    with KernelBlasterSidecar.start(port=port, health_timeout_s=30.0) as sidecar:
        assert sidecar.url == f"http://127.0.0.1:{port}"
        assert sidecar.health_response.get("status") == "healthy"
        assert sidecar.health_response.get("service") == "gpu-server"

        # Receipt is well-shaped + serialisable.
        receipt = sidecar.receipt()
        d = receipt.to_json()
        assert d["schema_version"] == "kernelblaster_sidecar_v1"
        assert d["port"] == port
        assert d["pid"] > 0
        assert d["health_probe_ms"] > 0.0
        assert d["health_response"]["status"] == "healthy"

    # Process must be reaped on context exit.
    assert sidecar.process.poll() is not None
