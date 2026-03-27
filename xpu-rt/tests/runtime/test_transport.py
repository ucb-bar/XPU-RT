"""Tests for runtime/transport.py -- transport abstraction."""

from __future__ import annotations

import pytest

from compgen.runtime.transport import (
    LocalTransport,
    SharedMemTransport,
    StubNetworkTransport,
    Transport,
    TransportMessage,
    ZephyrIPCConfig,
    ZephyrIPCTransport,
    create_transport,
    register_transport,
)


# ---- TransportMessage ----


class TestTransportMessage:
    def test_defaults(self) -> None:
        m = TransportMessage()
        assert m.tag == 0
        assert m.payload == b""
        assert m.metadata == {}

    def test_with_data(self) -> None:
        m = TransportMessage(tag=42, payload=b"hello", metadata={"shape": [3, 4]})
        assert m.tag == 42
        assert m.payload == b"hello"
        assert m.metadata["shape"] == [3, 4]


# ---- LocalTransport ----


class TestLocalTransport:
    def test_protocol(self) -> None:
        t = LocalTransport()
        assert isinstance(t, Transport)

    def test_lifecycle(self) -> None:
        t = LocalTransport()
        assert t.name == "local"
        assert t.is_open is False
        t.open()
        assert t.is_open is True
        t.close()
        assert t.is_open is False

    def test_send_recv(self) -> None:
        t = LocalTransport()
        t.open()
        msg = TransportMessage(tag=1, payload=b"data")
        assert t.send(msg) is True
        received = t.recv(timeout_us=1000)
        assert received is not None
        assert received.tag == 1
        assert received.payload == b"data"
        t.close()

    def test_send_when_closed(self) -> None:
        t = LocalTransport()
        assert t.send(TransportMessage()) is False

    def test_recv_timeout(self) -> None:
        t = LocalTransport()
        t.open()
        result = t.recv(timeout_us=1000)  # 1ms timeout
        assert result is None
        t.close()

    def test_send_multiple(self) -> None:
        t = LocalTransport()
        t.open()
        for i in range(5):
            t.send(TransportMessage(tag=i))
        for i in range(5):
            msg = t.recv(timeout_us=1000)
            assert msg is not None
            assert msg.tag == i
        t.close()

    def test_max_depth(self) -> None:
        t = LocalTransport()
        t.open(max_depth=2)
        assert t.send(TransportMessage(tag=1)) is True
        assert t.send(TransportMessage(tag=2)) is True
        assert t.send(TransportMessage(tag=3)) is False  # queue full
        t.close()

    def test_pending_count(self) -> None:
        t = LocalTransport()
        t.open()
        assert t.pending_count == 0
        t.send(TransportMessage())
        assert t.pending_count == 1
        t.recv(timeout_us=1000)
        assert t.pending_count == 0
        t.close()

    def test_barrier_noop(self) -> None:
        t = LocalTransport()
        t.open()
        t.barrier()  # should not raise
        t.close()


# ---- SharedMemTransport ----


class TestSharedMemTransport:
    def test_protocol(self) -> None:
        t = SharedMemTransport()
        assert isinstance(t, Transport)
        assert t.name == "shared_memory"

    def test_send_recv(self) -> None:
        t = SharedMemTransport()
        t.open(shm_name="test_shm", buffer_size=4096)
        assert t.is_open is True
        t.send(TransportMessage(tag=10, payload=b"shm_data"))
        msg = t.recv(timeout_us=1000)
        assert msg is not None
        assert msg.payload == b"shm_data"
        t.close()


# ---- ZephyrIPCTransport ----


class TestZephyrIPCTransport:
    def test_protocol(self) -> None:
        t = ZephyrIPCTransport()
        assert isinstance(t, Transport)
        assert t.name == "zephyr_ipc"

    def test_config_defaults(self) -> None:
        t = ZephyrIPCTransport()
        assert t.config.mechanism == "k_msgq"
        assert t.config.msg_size == 64
        assert t.config.queue_depth == 16

    def test_custom_config(self) -> None:
        cfg = ZephyrIPCConfig(
            mechanism="ipc_service",
            msg_size=256,
            queue_depth=32,
            endpoint_name="ml_ep",
        )
        t = ZephyrIPCTransport(config=cfg)
        assert t.config.mechanism == "ipc_service"
        assert t.config.endpoint_name == "ml_ep"

    def test_send_recv(self) -> None:
        t = ZephyrIPCTransport()
        t.open()
        msg = TransportMessage(tag=5, payload=b"cmd")
        assert t.send(msg) is True
        received = t.recv(timeout_us=1000)
        assert received is not None
        assert received.tag == 5
        t.close()

    def test_queue_full(self) -> None:
        cfg = ZephyrIPCConfig(queue_depth=2)
        t = ZephyrIPCTransport(config=cfg)
        t.open()
        assert t.send(TransportMessage(tag=1)) is True
        assert t.send(TransportMessage(tag=2)) is True
        assert t.send(TransportMessage(tag=3)) is False  # full
        t.close()

    def test_codegen_context(self) -> None:
        cfg = ZephyrIPCConfig(
            mechanism="ipc_service",
            msg_size=128,
            queue_depth=8,
            pipe_size=8192,
            endpoint_name="accel_ep",
            thread_priority=3,
            stack_size=8192,
        )
        t = ZephyrIPCTransport(config=cfg)
        ctx = t.codegen_context()
        assert ctx["mechanism"] == "ipc_service"
        assert ctx["use_ipc_service"] is True
        assert ctx["use_pipe"] is False
        assert ctx["msg_size"] == 128
        assert ctx["thread_priority"] == 3

    def test_codegen_context_pipe(self) -> None:
        cfg = ZephyrIPCConfig(mechanism="k_pipe")
        t = ZephyrIPCTransport(config=cfg)
        ctx = t.codegen_context()
        assert ctx["use_pipe"] is True
        assert ctx["use_ipc_service"] is False


# ---- StubNetworkTransport ----


class TestStubNetworkTransport:
    def test_protocol(self) -> None:
        t = StubNetworkTransport()
        assert isinstance(t, Transport)
        assert t.name == "network"

    def test_open_close(self) -> None:
        t = StubNetworkTransport()
        t.open(address="10.0.0.1", port=50051)
        assert t.is_open is True
        t.close()
        assert t.is_open is False

    def test_send_recv(self) -> None:
        t = StubNetworkTransport()
        t.open()
        t.send(TransportMessage(tag=99, payload=b"net"))
        msg = t.recv(timeout_us=1000)
        assert msg is not None
        assert msg.tag == 99
        t.close()


# ---- Factory ----


class TestTransportFactory:
    def test_create_local(self) -> None:
        t = create_transport("local")
        assert t.name == "local"

    def test_create_shared_memory(self) -> None:
        t = create_transport("shared_memory")
        assert t.name == "shared_memory"

    def test_create_zephyr_ipc(self) -> None:
        t = create_transport("zephyr_ipc")
        assert t.name == "zephyr_ipc"

    def test_create_network(self) -> None:
        t = create_transport("network")
        assert t.name == "network"

    def test_create_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown transport"):
            create_transport("bluetooth")

    def test_register_custom(self) -> None:
        class CustomTransport:
            @property
            def name(self) -> str:
                return "custom"

            @property
            def is_open(self) -> bool:
                return False

            def open(self, **kwargs: ...) -> None:
                pass

            def close(self) -> None:
                pass

            def send(self, msg: TransportMessage, timeout_us: float | None = None) -> bool:
                return False

            def recv(self, timeout_us: float | None = None) -> TransportMessage | None:
                return None

            def barrier(self) -> None:
                pass

        register_transport("custom_test", CustomTransport)
        t = create_transport("custom_test")
        assert t.name == "custom"
