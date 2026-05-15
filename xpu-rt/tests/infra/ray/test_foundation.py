"""Tests for Ray foundation — _require.py, RayTransport, placement templates.

These tests verify that the optional dependency pattern works correctly.
"""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")


class TestRequire:
    def test_require_ray(self) -> None:
        from infra.ray._require import require_ray

        r = require_ray()
        assert hasattr(r, "init")

    def test_require_serve(self) -> None:
        from infra.ray._require import require_serve

        serve = require_serve()
        assert hasattr(serve, "deployment")

    def test_require_tune(self) -> None:
        from infra.ray._require import require_tune

        tune = require_tune()
        assert hasattr(tune, "Tuner")

    def test_ensure_initialized(self, ray_cluster) -> None:
        from infra.ray._require import ensure_ray_initialized

        ensure_ray_initialized()
        assert ray.is_initialized()


class TestRayTransport:
    def test_name(self) -> None:
        from compgen.runtime._ray_transport import RayTransport

        t = RayTransport()
        assert t.name == "ray"

    def test_lifecycle(self, ray_cluster) -> None:
        from compgen.runtime._ray_transport import RayTransport

        t = RayTransport()
        assert t.is_open is False
        t.open()
        assert t.is_open is True
        t.close()
        assert t.is_open is False

    def test_send_recv(self, ray_cluster) -> None:
        from compgen.runtime._ray_transport import RayTransport
        from compgen.runtime.transport import TransportMessage

        t = RayTransport()
        t.open()
        msg = TransportMessage(tag=42, payload=b"hello_ray")
        assert t.send(msg) is True
        received = t.recv(timeout_us=5_000_000)
        assert received is not None
        assert received.tag == 42
        assert received.payload == b"hello_ray"
        t.close()

    def test_send_when_closed(self) -> None:
        from compgen.runtime._ray_transport import RayTransport
        from compgen.runtime.transport import TransportMessage

        t = RayTransport()
        assert t.send(TransportMessage()) is False

    def test_registered_in_transport_factory(self, ray_cluster) -> None:
        from compgen.runtime.transport import create_transport

        t = create_transport("ray")
        assert t.name == "ray"


class TestPlacementTemplates:
    def test_compile_placement(self) -> None:
        from infra.ray.resources.placement_templates import compile_placement

        pg = compile_placement(num_workers=4)
        assert pg["strategy"] == "SPREAD"
        assert len(pg["bundles"]) == 4

    def test_benchmark_gpu(self) -> None:
        from infra.ray.resources.placement_templates import benchmark_gpu_placement

        pg = benchmark_gpu_placement()
        assert pg["strategy"] == "STRICT_PACK"
        assert pg["bundles"][0]["GPU"] == 1

    def test_hardware_test(self) -> None:
        from infra.ray.resources.placement_templates import hardware_test_placement

        pg = hardware_test_placement("fpga_xilinx")
        assert "fpga_xilinx" in pg["bundles"][0]

    def test_distributed_compile(self) -> None:
        from infra.ray.resources.placement_templates import distributed_compile_placement

        pg = distributed_compile_placement(num_cpu_workers=2, num_gpu_workers=1)
        assert len(pg["bundles"]) == 3
