"""Tests for the Python-C native runtime bridge."""

from __future__ import annotations

import ctypes
from unittest.mock import MagicMock, patch

import pytest
from compgen.runtime.native_bridge import (
    _NOT_COMPILED_MSG,
    NativeBuffer,
    NativeDevice,
    NativeEngine,
    _reset_library_cache,
    _try_load_library,
)


@pytest.fixture(autouse=True)
def _clean_lib_cache() -> None:
    """Reset the cached library probe between tests."""
    _reset_library_cache()
    yield  # type: ignore[misc]
    _reset_library_cache()


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------


class TestLibraryLoading:
    """Tests for the library auto-detection logic."""

    def test_try_load_library_returns_none_when_missing(self) -> None:
        """When no native .so exists, _try_load_library returns None."""
        lib = _try_load_library()
        # In CI/test environments the C library will not be compiled.
        assert lib is None

    @patch("compgen.runtime.native_bridge.ctypes.CDLL")
    @patch("compgen.runtime.native_bridge.ctypes.util.find_library")
    def test_try_load_library_uses_find_library(
        self,
        mock_find: MagicMock,
        mock_cdll: MagicMock,
    ) -> None:
        """When ctypes.util.find_library succeeds, the lib is loaded."""
        mock_find.return_value = "/usr/lib/libcompgen_runtime.so"
        sentinel = MagicMock()
        mock_cdll.return_value = sentinel
        lib = _try_load_library()
        assert lib is sentinel
        mock_cdll.assert_called_with("/usr/lib/libcompgen_runtime.so")


# ---------------------------------------------------------------------------
# NativeEngine -- fallback path (no C library)
# ---------------------------------------------------------------------------


class TestNativeEngineFallback:
    """Tests for NativeEngine when the C library is absent."""

    def test_instantiation_without_lib(self) -> None:
        """NativeEngine should instantiate even when the lib is missing."""
        engine = NativeEngine()
        assert not engine.available

    def test_available_is_false(self) -> None:
        """available should be False when the C library is missing."""
        engine = NativeEngine()
        assert engine.available is False

    def test_submit_raises(self) -> None:
        """submit() should raise RuntimeError without the C lib."""
        engine = NativeEngine()
        with pytest.raises(RuntimeError, match="C runtime not compiled"):
            engine.submit(0)

    def test_wait_idle_raises(self) -> None:
        """wait_idle() should raise RuntimeError without the C lib."""
        engine = NativeEngine()
        with pytest.raises(RuntimeError, match="C runtime not compiled"):
            engine.wait_idle()

    def test_shutdown_is_safe(self) -> None:
        """shutdown() should be a no-op when the lib is missing."""
        engine = NativeEngine()
        engine.shutdown()  # should not raise

    def test_repr(self) -> None:
        """repr should include status."""
        engine = NativeEngine()
        assert "unavailable" in repr(engine)


# ---------------------------------------------------------------------------
# NativeDevice -- fallback path
# ---------------------------------------------------------------------------


class TestNativeDeviceFallback:
    """Tests for NativeDevice when the C library is absent."""

    def test_instantiation_without_lib(self) -> None:
        """NativeDevice should instantiate even when the lib is missing."""
        device = NativeDevice(device_type="cpu", device_index=0)
        assert not device.available

    def test_available_is_false(self) -> None:
        """available should be False when the C library is missing."""
        device = NativeDevice()
        assert device.available is False

    def test_alloc_raises(self) -> None:
        """alloc() should raise RuntimeError without the C lib."""
        device = NativeDevice()
        with pytest.raises(RuntimeError, match="C runtime not compiled"):
            device.alloc(1024)

    def test_dispatch_raises(self) -> None:
        """dispatch() should raise RuntimeError without the C lib."""
        device = NativeDevice()
        with pytest.raises(RuntimeError, match="C runtime not compiled"):
            device.dispatch(0)

    def test_sync_raises(self) -> None:
        """sync() should raise RuntimeError without the C lib."""
        device = NativeDevice()
        with pytest.raises(RuntimeError, match="C runtime not compiled"):
            device.sync()

    def test_close_is_safe(self) -> None:
        """close() should be a no-op when the lib is missing."""
        device = NativeDevice()
        device.close()  # should not raise

    def test_repr(self) -> None:
        """repr should include device type and status."""
        device = NativeDevice(device_type="cuda", device_index=1)
        r = repr(device)
        assert "cuda" in r
        assert "unavailable" in r

    def test_default_args(self) -> None:
        """Default device_type is 'cpu' and device_index is 0."""
        device = NativeDevice()
        assert device.device_type == "cpu"
        assert device.device_index == 0


# ---------------------------------------------------------------------------
# NativeBuffer -- fallback path
# ---------------------------------------------------------------------------


class TestNativeBufferFallback:
    """Tests for NativeBuffer when constructed without a native backend."""

    def test_instantiation(self) -> None:
        """Buffer should instantiate with None handle and lib."""
        buf = NativeBuffer(size=256, handle=None, lib=None)
        assert buf.size == 256
        assert buf.handle is None

    def test_write_raises(self) -> None:
        """write() should raise RuntimeError without the C lib."""
        buf = NativeBuffer(size=256, handle=None, lib=None)
        with pytest.raises(RuntimeError, match="C runtime not compiled"):
            buf.write(b"\x00" * 10)

    def test_read_raises(self) -> None:
        """read() should raise RuntimeError without the C lib."""
        buf = NativeBuffer(size=256, handle=None, lib=None)
        with pytest.raises(RuntimeError, match="C runtime not compiled"):
            buf.read()

    def test_free_is_safe(self) -> None:
        """free() should be a no-op on a fallback buffer."""
        buf = NativeBuffer(size=256, handle=None, lib=None)
        buf.free()  # should not raise
        assert buf._freed is True

    def test_double_free_is_safe(self) -> None:
        """Calling free() twice should not raise."""
        buf = NativeBuffer(size=256, handle=None, lib=None)
        buf.free()
        buf.free()  # should not raise

    def test_repr(self) -> None:
        """repr should include size and state."""
        buf = NativeBuffer(size=512, handle=None, lib=None)
        r = repr(buf)
        assert "512" in r
        assert "live" in r
        buf.free()
        r2 = repr(buf)
        assert "freed" in r2


# ---------------------------------------------------------------------------
# NativeBuffer -- freed-buffer guard
# ---------------------------------------------------------------------------


class TestNativeBufferFreedGuard:
    """After free(), operations on a buffer with a real lib should fail."""

    def test_write_after_free_raises(self) -> None:
        """write() on a freed buffer should raise, even with a lib set."""
        mock_lib = MagicMock(spec=ctypes.CDLL)
        buf = NativeBuffer(size=64, handle=ctypes.c_void_p(0x1), lib=mock_lib)
        buf._freed = True
        buf.handle = None
        with pytest.raises(RuntimeError, match="buffer has been freed"):
            buf.write(b"\x00")

    def test_read_after_free_raises(self) -> None:
        """read() on a freed buffer should raise, even with a lib set."""
        mock_lib = MagicMock(spec=ctypes.CDLL)
        buf = NativeBuffer(size=64, handle=ctypes.c_void_p(0x1), lib=mock_lib)
        buf._freed = True
        buf.handle = None
        with pytest.raises(RuntimeError, match="buffer has been freed"):
            buf.read()


# ---------------------------------------------------------------------------
# NativeBuffer -- write validation
# ---------------------------------------------------------------------------


class TestNativeBufferValidation:
    """Validation tests for NativeBuffer with a mocked library."""

    def test_write_too_large_raises_value_error(self) -> None:
        """Writing more data than the buffer size should raise ValueError."""
        mock_lib = MagicMock(spec=ctypes.CDLL)
        buf = NativeBuffer(size=8, handle=ctypes.c_void_p(0x1), lib=mock_lib)
        with pytest.raises(ValueError, match="exceeds buffer size"):
            buf.write(b"\x00" * 16)
        # Prevent __del__ from calling into mock (no compgen_buffer_free)
        buf._lib = None
        buf._freed = True


# ---------------------------------------------------------------------------
# Interface contract -- expected public attributes & methods
# ---------------------------------------------------------------------------


class TestInterfaceContract:
    """Verify that the classes expose the expected public interface."""

    def test_engine_has_expected_methods(self) -> None:
        """NativeEngine should have submit, wait_idle, shutdown, available."""
        engine = NativeEngine()
        assert hasattr(engine, "submit")
        assert hasattr(engine, "wait_idle")
        assert hasattr(engine, "shutdown")
        assert hasattr(engine, "available")
        assert callable(engine.submit)
        assert callable(engine.wait_idle)
        assert callable(engine.shutdown)

    def test_device_has_expected_methods(self) -> None:
        """NativeDevice should have alloc, dispatch, sync, close, available."""
        device = NativeDevice()
        assert hasattr(device, "alloc")
        assert hasattr(device, "dispatch")
        assert hasattr(device, "sync")
        assert hasattr(device, "close")
        assert hasattr(device, "available")
        assert callable(device.alloc)
        assert callable(device.dispatch)
        assert callable(device.sync)
        assert callable(device.close)

    def test_buffer_has_expected_methods(self) -> None:
        """NativeBuffer should have write, read, free."""
        buf = NativeBuffer(size=0, handle=None, lib=None)
        assert hasattr(buf, "write")
        assert hasattr(buf, "read")
        assert hasattr(buf, "free")
        assert hasattr(buf, "size")
        assert hasattr(buf, "handle")
        assert callable(buf.write)
        assert callable(buf.read)
        assert callable(buf.free)

    def test_not_compiled_message_is_descriptive(self) -> None:
        """The error message should clearly say the lib is not compiled."""
        assert "not compiled" in _NOT_COMPILED_MSG.lower() or "not found" in _NOT_COMPILED_MSG.lower()
