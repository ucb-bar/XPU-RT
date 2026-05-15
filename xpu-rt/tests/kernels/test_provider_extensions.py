"""Tests for the Phase-E ProviderResult / KernelProvider extensions.

Covers:
- REQ-014 — ``index.json`` carries ``region_id`` + ``dispatch_id``;
  filenames include ``region_id`` to avoid collisions.
- REQ-015 — ``ProviderResult.emit_mode`` round-trips into ``index.json``.
- REQ-016 — ``ProviderResult.expected_inputs`` materialises a
  ``<provider>/<region>_<op>.data.h`` companion file.
- REQ-017 — Provider ``priority`` selects the highest-priority winner
  even when multiple providers accept the same contract.
- REQ-018 — ``ProviderResult.kernel_files`` writes a multi-file
  per-region directory; ``additional_files`` listed in ``index.json``.
- REQ-019 — ``ProviderResult.dispatch_geometry`` round-trips into
  ``index.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.kernels.codegen_fallback import run_provider_fallback
from compgen.kernels.provider import (
    DispatchGeometry,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from compgen.kernels.provider import (
    KernelContract as ProviderContract,
)
from compgen.runtime.bundle_emit import emit_extended_artefacts
from compgen.targets.schema import load_profile

_TARGET = "examples/target_profiles/cuda_a100.yaml"


def _module_for(model: nn.Module, *args: torch.Tensor):
    ep = capture_model(model, args)
    module, _ = fx_to_xdsl(ep)
    return module


# ---------------------------------------------------------------------------
# REQ-014 — region_id + dispatch_id in index.json + collision-safe filenames
# ---------------------------------------------------------------------------


class _ConstProvider:
    name: str = "const_provider"

    def __init__(
        self,
        emit_mode: str = "compute_callback",
        geometry: DispatchGeometry | None = None,
        kernel_files: dict[str, str] | None = None,
        expected_inputs: dict[str, dict] | None = None,
        priority: int = 0,
        accepted: frozenset[str] = frozenset({"add", "mul", "sub", "relu"}),
    ) -> None:
        self.emit_mode = emit_mode
        self.geometry = geometry
        self.kernel_files = kernel_files
        self.expected_inputs = expected_inputs
        self.priority = priority
        self._accepted = accepted

    def accepts_contract(self, c: ProviderContract) -> bool:
        return c.op_family in self._accepted

    def search(self, c: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
        return ProviderResult(
            found=True,
            kernel_code=f"// op={c.op_family}\n",
            language="cpp",
            emit_mode=self.emit_mode,
            dispatch_geometry=self.geometry,
            kernel_files=self.kernel_files,
            expected_inputs=self.expected_inputs,
            correct=True,
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        return []


def _emit_to_bundle(out: list[dict], tmp_path: Path, sample_inputs: tuple = (torch.randn(2, 2),)) -> Path:
    """Run bundle_emit with ``out`` in pipeline_artifacts, return bundle dir."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}))

    class _FakeCapture:
        exported_program = None
        diagnostics = None

    emit_extended_artefacts(
        bundle_dir,
        capture_artifact=_FakeCapture(),
        sample_inputs=sample_inputs,
        pipeline_artifacts={"generated_kernels": out},
    )
    return bundle_dir


def test_index_json_carries_region_id_and_dispatch_id(tmp_path: Path) -> None:
    class ReluAdd(nn.Module):
        def forward(self, a, b):
            return torch.relu(a + b)

    module = _module_for(ReluAdd(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_ConstProvider()],
    )
    bundle = _emit_to_bundle(out, tmp_path)

    index = json.loads((bundle / "generated_kernels" / "index.json").read_text())
    assert len(index) == 2
    for entry in index:
        assert "region_id" in entry, entry
        assert "dispatch_id" in entry, entry
        assert entry["region_id"].startswith("region_") or entry["region_id"]
        # dispatch_id either echoes the IR annotation or the synthesised d_<i>
        assert entry["dispatch_id"], entry


def test_two_regions_with_same_op_name_do_not_clobber_each_other(tmp_path: Path) -> None:
    """``(a+b)+(c+d)`` → two ``aten_add`` regions; both files must exist."""

    class TwoAdds(nn.Module):
        def forward(self, a, b, c, d):
            return (a + b) + (c + d)

    module = _module_for(TwoAdds(), torch.randn(4), torch.randn(4), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4), torch.randn(4), torch.randn(4)),
        extra_providers=[_ConstProvider()],
    )
    # Three add regions: (a+b), (c+d), and ((a+b)+(c+d)).
    assert len(out) >= 2

    bundle = _emit_to_bundle(out, tmp_path)
    files = sorted((bundle / "generated_kernels" / "const_provider").glob("*.cpp"))
    # Each file must exist on disk — no clobbering.
    assert len(files) == len(out), [p.name for p in files]
    # Filenames must be unique. Either the IR's explicit
    # ``compgen.region_id`` (e.g. ``add_0_aten_add.cpp``) or the
    # synthesised ``region_<i>`` prefix is acceptable; what matters is
    # that no two files collapse to the same name.
    assert len({f.name for f in files}) == len(files), [f.name for f in files]
    # And every file's name carries the region prefix (not just the
    # bare op_name), so the entries are disambiguated regardless of
    # which annotation source the IR used.
    for f in files:
        assert not f.name.startswith("aten_add."), f.name


# ---------------------------------------------------------------------------
# REQ-015 — emit_mode round-trips
# ---------------------------------------------------------------------------


def test_emit_mode_self_contained_round_trips(tmp_path: Path) -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(Add(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_ConstProvider(emit_mode="self_contained")],
    )
    bundle = _emit_to_bundle(out, tmp_path)
    index = json.loads((bundle / "generated_kernels" / "index.json").read_text())
    assert all(e["emit_mode"] == "self_contained" for e in index)


def test_emit_mode_default_is_compute_callback() -> None:
    """Backward-compat: providers that don't set emit_mode get the default."""
    r = ProviderResult(found=True, kernel_code="// x\n", language="cpp")
    assert r.emit_mode == "compute_callback"


# ---------------------------------------------------------------------------
# REQ-017 — priority signaling
# ---------------------------------------------------------------------------


def test_higher_priority_provider_wins() -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(Add(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)

    low = _ConstProvider(priority=0)
    low.name = "low"  # type: ignore[misc]
    high = _ConstProvider(priority=10)
    high.name = "high"  # type: ignore[misc]

    # Pass them in ENTRY-POINT order with low first; priority should
    # override and pick high.
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[low, high],
    )
    assert out[0]["provider"] == "high", out


def test_equal_priority_falls_back_to_registration_order() -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(Add(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    a = _ConstProvider(priority=5)
    a.name = "a"  # type: ignore[misc]
    b = _ConstProvider(priority=5)
    b.name = "b"  # type: ignore[misc]
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[a, b],
    )
    assert out[0]["provider"] == "a"


# ---------------------------------------------------------------------------
# REQ-019 — dispatch_geometry round-trips
# ---------------------------------------------------------------------------


def test_dispatch_geometry_round_trips(tmp_path: Path) -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(Add(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    geo = DispatchGeometry(num_warps=4, threadblock_shape=(32, 32), grid_shape=(8,))
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_ConstProvider(geometry=geo)],
    )
    assert out[0]["dispatch_geometry"]["num_warps"] == 4
    assert out[0]["dispatch_geometry"]["threadblock_shape"] == [32, 32]
    assert out[0]["dispatch_geometry"]["grid_shape"] == [8]

    bundle = _emit_to_bundle(out, tmp_path)
    index = json.loads((bundle / "generated_kernels" / "index.json").read_text())
    assert index[0]["dispatch_geometry"]["num_warps"] == 4


def test_dispatch_geometry_absent_when_not_set(tmp_path: Path) -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(Add(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_ConstProvider()],
    )
    assert "dispatch_geometry" not in out[0]


# ---------------------------------------------------------------------------
# REQ-018 — multi-file kernel bundles
# ---------------------------------------------------------------------------


def test_kernel_files_emitted_into_per_region_dir(tmp_path: Path) -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(Add(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)

    files = {
        "aten_add.cpp": '// kernel\n#include "helpers.h"\n',
        "helpers.h": "/* helper macros */\n",
        "lookup.inc": "/* table */\n",
    }
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_ConstProvider(kernel_files=files)],
    )
    bundle = _emit_to_bundle(out, tmp_path)

    index = json.loads((bundle / "generated_kernels" / "index.json").read_text())
    entry = index[0]
    assert "additional_files" in entry
    assert len(entry["additional_files"]) == 2
    # Primary path points at the .cpp matching op stem.
    assert entry["path"].endswith("aten_add.cpp")

    # All three files present on disk under <provider>/<region>_<op>/.
    region_dir = bundle / "generated_kernels" / "const_provider" / Path(entry["path"]).parent.name
    assert (region_dir / "aten_add.cpp").is_file()
    assert (region_dir / "helpers.h").is_file()
    assert (region_dir / "lookup.inc").is_file()
    # And the helper sibling actually has the right contents.
    assert "helper macros" in (region_dir / "helpers.h").read_text()


def test_single_file_path_unchanged_when_no_kernel_files(tmp_path: Path) -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(Add(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_ConstProvider()],
    )
    bundle = _emit_to_bundle(out, tmp_path)
    index = json.loads((bundle / "generated_kernels" / "index.json").read_text())
    # Path points at a flat file in the provider dir, no extra subdirectory.
    rel_parts = Path(index[0]["path"]).parts
    assert len(rel_parts) == 2, rel_parts  # provider_dir / file


# ---------------------------------------------------------------------------
# REQ-016 — expected_inputs materialises <op>.data.h with golden bits
# ---------------------------------------------------------------------------


def test_expected_inputs_materialises_data_header(tmp_path: Path) -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([0.5, 1.5, -0.5, 0.0])
    module = _module_for(Add(), a, b)
    target = load_profile(_TARGET)

    expected = {
        "A_raw": {"size": 4, "dtype": "uint32", "init": "from_golden:0"},
        "B_raw": {"size": 4, "dtype": "uint32", "init": "from_golden:1"},
        "C_raw": {"size": 4, "dtype": "uint32", "init": "zeros"},
        "n": {"size": 1, "dtype": "uint32", "init": "literal:4"},
    }
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(a, b),
        extra_providers=[_ConstProvider(expected_inputs=expected)],
    )
    bundle = _emit_to_bundle(out, tmp_path, sample_inputs=(a, b))

    index = json.loads((bundle / "generated_kernels" / "index.json").read_text())
    entry = index[0]
    assert "data_header" in entry
    data_h = bundle / "generated_kernels" / Path(entry["data_header"]).relative_to(Path(entry["data_header"]).parts[0])
    # Reconstruct the absolute path:
    data_h_abs = bundle / "generated_kernels" / entry["data_header"]
    assert data_h_abs.is_file(), entry
    body = data_h_abs.read_text()
    assert "static const uint32_t A_raw[4]" in body
    assert "static const uint32_t B_raw[4]" in body
    assert "static const uint32_t C_raw[4]" in body
    # ``n`` → literal:4
    assert "static const uint32_t n[1] = { 4 };" in body
    # A_raw was baked from the f32 tensor [1.0, 2.0, 3.0, 4.0]:
    # float bits of 1.0 = 0x3f800000.
    assert "0x3f800000" in body


def test_expected_inputs_absent_means_no_data_header(tmp_path: Path) -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(Add(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_ConstProvider()],
    )
    bundle = _emit_to_bundle(out, tmp_path)
    index = json.loads((bundle / "generated_kernels" / "index.json").read_text())
    assert "data_header" not in index[0]
    assert not list((bundle / "generated_kernels").rglob("*.data.h"))


# ---------------------------------------------------------------------------
# Sanity: existing 5-tuple ProviderResult fields all coexist with the new ones
# ---------------------------------------------------------------------------


def test_provider_result_constructs_with_all_new_fields() -> None:
    r = ProviderResult(
        found=True,
        kernel_code="// k\n",
        language="cpp",
        emit_mode="self_contained",
        dispatch_geometry=DispatchGeometry(num_warps=4),
        kernel_files={"k.cpp": "// k\n"},
        expected_inputs={"x": {"size": 4, "dtype": "uint32", "init": "zeros"}},
    )
    assert r.emit_mode == "self_contained"
    assert r.dispatch_geometry.num_warps == 4
    assert "k.cpp" in r.kernel_files
    assert "x" in r.expected_inputs
