"""Tests for compgen.llm.tools.observability."""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.llm import get_registry

# Importing triggers auto-registration into the global registry.
from compgen.llm.tools import observability  # noqa: F401


def test_observability_tools_auto_registered() -> None:
    r = get_registry()
    names = {t.name for t in r.list_tools(phase=2)}
    for expected in ("read_target_features", "read_analyzer_dossier", "read_region_shapes"):
        assert expected in names, f"{expected!r} not auto-registered into phase 2"


def test_read_target_features_serializes() -> None:
    @dataclass
    class FakeProfile:
        name: str = "fake"
        devices: list = field(default_factory=list)

    @dataclass
    class FakeDevice:
        profile: FakeProfile = field(default_factory=FakeProfile)

    out = observability._read_target_features_impl(device=FakeDevice())
    assert out["status"] == "ok"
    assert out["target"]["name"] == "fake"


def test_read_target_features_slice_keys_filters() -> None:
    @dataclass
    class FakeProfile:
        name: str = "fake"
        devices: list = field(default_factory=list)
        costs: dict = field(default_factory=dict)

    @dataclass
    class FakeDevice:
        profile: FakeProfile = field(default_factory=FakeProfile)

    out = observability._read_target_features_impl(device=FakeDevice(), slice_keys=("name",))
    assert out["status"] == "ok"
    assert set(out["target"].keys()) == {"name"}


def test_read_analyzer_dossier_handles_none() -> None:
    class FakeAnalysis:
        dossier = None

    out = observability._read_analyzer_dossier_impl(analysis=FakeAnalysis())
    assert out["status"] == "error"


def test_read_analyzer_dossier_reports_totals() -> None:
    @dataclass
    class FakeDossier:
        regions: list = field(default_factory=list)
        dynamic_shape_regions: list = field(default_factory=list)

    @dataclass
    class FakeAnalysis:
        dossier: FakeDossier = field(default_factory=FakeDossier)
        model_name: str = "m"
        total_params: int = 100
        total_flops: int = 200
        total_bytes: int = 300
        clusters: list = field(default_factory=list)
        bottleneck_clusters: list = field(default_factory=list)
        optimization_opportunities: list = field(default_factory=list)

    out = observability._read_analyzer_dossier_impl(analysis=FakeAnalysis())
    assert out["status"] == "ok"
    assert out["total_params"] == 100
    assert out["total_flops"] == 200
    assert out["total_bytes"] == 300


def test_read_region_shapes_not_found() -> None:
    class FakeDossier:
        regions = ()

    class FakeAnalysis:
        dossier = FakeDossier()

    out = observability._read_region_shapes_impl(analysis=FakeAnalysis(), region_id="missing")
    assert out["status"] == "not_found"


def test_read_region_shapes_found() -> None:
    @dataclass
    class FakeRegion:
        region_id: str
        kind: str = "matmul"
        flops: int = 1_000_000
        bytes: int = 100_000
        arithmetic_intensity: float = 10.0
        dynamic_shapes: bool = False
        repeated_count: int = 2
        producers: list = field(default_factory=list)
        consumers: list = field(default_factory=list)

    @dataclass
    class FakeDossier:
        regions: tuple = ()

    @dataclass
    class FakeAnalysis:
        dossier: FakeDossier = field(default_factory=FakeDossier)

    r = FakeRegion(region_id="r0")
    a = FakeAnalysis()
    a.dossier = FakeDossier(regions=(r,))
    out = observability._read_region_shapes_impl(analysis=a, region_id="r0")
    assert out["status"] == "ok"
    assert out["kind"] == "matmul"
    assert out["flops"] == 1_000_000


def test_register_is_idempotent() -> None:
    first = observability.register()
    second = observability.register()
    # Second call sees they are already registered; returns empty list.
    assert second == []
