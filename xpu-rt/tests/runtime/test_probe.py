"""Phase-6 device-probe tests.

CPU-only. Validates the probe API surface + the
:meth:`DeviceTraits.with_probe` merge contract. The actual probe
output on real hardware is exercised on the remote Blackwell box
via the conformance harness's ``--probe-device-only`` flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestProbeApi:
    def test_probe_module_imports(self) -> None:
        from compgen.runtime.probe import (
            probe_cuda_device,
            probe_via_native_hal,
            probe_via_torch,
        )

        assert callable(probe_cuda_device)
        assert callable(probe_via_torch)
        assert callable(probe_via_native_hal)

    def test_native_hal_raises_when_cuda_lib_absent(self) -> None:
        """Phase 4 wired ``probe_via_native_hal`` to the C primitive,
        but on a CPU-only host (no ``libcompgen_rt-cuda.so`` from
        ``make build-cuda-rt``) it must still raise
        :class:`_NativeHalUnavailable` so :func:`probe_cuda_device`
        falls through to the torch path."""
        from compgen.runtime.probe import _NativeHalUnavailable, probe_via_native_hal

        with pytest.raises(_NativeHalUnavailable):
            probe_via_native_hal(0)

    def test_probe_returns_dict_with_provenance(self) -> None:
        from compgen.runtime.probe import probe_cuda_device

        out = probe_cuda_device(0)
        assert isinstance(out, dict)
        # Either the torch path ran (CUDA present) or fallback ran (CPU host).
        assert out.get("probe_source") in {"torch", "fallback", "native_hal"}

    def test_probe_on_cpu_host_emits_clean_fallback(self) -> None:
        """When torch.cuda.is_available() is False the probe must
        return a clear fallback record, not raise."""
        import torch
        from compgen.runtime.probe import probe_via_torch

        if torch.cuda.is_available():
            pytest.skip("This test exercises the CPU-fallback branch")
        out = probe_via_torch(0)
        assert out["probe_source"] == "fallback"
        assert out.get("probe_error")


class TestDeviceTraitsWithProbe:
    """The probe + with_probe contract: probe values override
    profile-derived ones, supports_event_tensors is re-derived after."""

    def test_with_probe_merges_metadata(self) -> None:
        from compgen.runtime.traits import DeviceTraits
        from compgen.targets.schema import (
            ComputeUnit,
            DeviceSpec,
            MemoryLevel,
            TargetProfile,
        )

        profile = TargetProfile(
            name="blackwell-rtx-pro-6000",
            devices=[
                DeviceSpec(
                    device_type="gpu",
                    name="RTX PRO 6000",
                    vendor="nvidia",
                    compute_units=[ComputeUnit(name="tc", count=132, peak_tflops=2500.0)],
                    memory_hierarchy=[MemoryLevel(name="hbm", size_bytes=192 * 1024**3, bandwidth_gbps=8000.0)],
                    features=["tma", "fp8_tensor_cores", "cooperative_launch", "persistent_kernels"],
                )
            ],
            metadata={
                "compute_capability_major": 12,
                "compute_capability_minor": 0,
                "sm_count": 132,  # placeholder — probe will correct to 188
            },
        )
        traits = DeviceTraits.from_target_profile(profile)
        # YAML had sm_count=132; baseline metadata reflects that.
        assert traits.metadata["sm_count"] == 132
        # cc-derived booleans (not yet overridden by probe).
        assert traits.metadata.get("supports_tma") is True
        assert traits.metadata.get("supports_clusters") is True
        # Live probe overrides:
        probe = {
            "sm_count": 188,
            "compute_capability_major": 12,
            "compute_capability_minor": 0,
            "supports_clusters": False,  # hypothetical: sm_120 lacks cluster launch
            "peak_flops_per_s": 3.2e15,
            "peak_bandwidth_bps": 1.79e12,
            "interconnect_topology": "pcie",
        }
        merged = traits.with_probe(probe)
        # Probe wins.
        assert merged.metadata["sm_count"] == 188
        assert merged.metadata["supports_clusters"] is False
        assert merged.metadata["peak_flops_per_s"] == 3.2e15
        # Top-level dataclass fields untouched (probe didn't override them).
        assert merged.supports_persistent_kernels is True
        # Probe didn't say anything about features set; still True.
        assert merged.supports_event_tensors is True

    def test_with_probe_re_derives_event_tensor_support(self) -> None:
        """If the probe declares atomics or persistent kernels are
        unavailable, supports_event_tensors must flip False even
        though the profile said it was True."""
        from compgen.runtime.traits import DeviceTraits
        from compgen.targets.schema import DeviceSpec, TargetProfile

        profile = TargetProfile(
            name="hypothetical",
            devices=[DeviceSpec(device_type="gpu", name="card", vendor="nvidia")],
        )
        traits = DeviceTraits.from_target_profile(profile)
        assert traits.supports_event_tensors is True

        merged = traits.with_probe({"supports_persistent_kernels": False})
        assert merged.supports_persistent_kernels is False
        # Composite re-derived.
        assert merged.supports_event_tensors is False

    def test_to_dict_carries_metadata(self) -> None:
        from compgen.runtime.traits import DeviceTraits
        from compgen.targets.schema import DeviceSpec, TargetProfile

        profile = TargetProfile(
            name="t",
            devices=[DeviceSpec(device_type="gpu", name="card", vendor="nvidia")],
            metadata={"sm_count": 188, "compute_capability_major": 12},
        )
        traits = DeviceTraits.from_target_profile(profile)
        d = traits.to_dict()
        assert "metadata" in d
        assert d["metadata"]["sm_count"] == 188

    def test_metadata_keys_are_whitelisted(self) -> None:
        """Random profile-metadata keys don't leak into traits.metadata —
        only the _FORWARDED_METADATA_KEYS set."""
        from compgen.runtime.traits import DeviceTraits
        from compgen.targets.schema import DeviceSpec, TargetProfile

        profile = TargetProfile(
            name="t",
            devices=[DeviceSpec(device_type="gpu", name="card", vendor="nvidia")],
            metadata={
                "sm_count": 188,
                "totally_random_key": "value",
                "supports_tma": False,  # explicit override
                "compute_capability_major": 12,
            },
        )
        traits = DeviceTraits.from_target_profile(profile)
        assert "totally_random_key" not in traits.metadata
        assert traits.metadata["sm_count"] == 188
        # User's explicit `supports_tma=False` survives the cc-derived setdefault.
        assert traits.metadata["supports_tma"] is False


class TestProfileYamls:
    """The two Blackwell profile YAMLs parse + produce sane traits."""

    @pytest.fixture
    def profiles_dir(self) -> Path:
        from compgen import __file__ as compgen_init

        return Path(compgen_init).parent / "targets" / "profiles"

    def test_b200_profile_loads(self, profiles_dir: Path) -> None:

        path = profiles_dir / "blackwell_b200.yaml"
        assert path.is_file()
        import yaml

        data = yaml.safe_load(path.read_text())
        # Spot-check critical fields that downstream phases consume.
        assert data["name"] == "blackwell-b200"
        assert data["metadata"]["compute_capability_major"] == 10
        assert data["metadata"]["sm_count"] == 132
        assert data["metadata"]["supports_clusters"] is True
        assert data["metadata"]["supports_fp8"] is True
        assert data["metadata"]["supports_fp4"] is True
        assert data["metadata"]["interconnect_topology"] == "nvlink"

    def test_rtx_pro_6000_profile_loads(self, profiles_dir: Path) -> None:
        path = profiles_dir / "blackwell_rtx_pro_6000.yaml"
        assert path.is_file()
        import yaml

        data = yaml.safe_load(path.read_text())
        assert data["name"] == "blackwell-rtx-pro-6000"
        assert data["metadata"]["compute_capability_major"] == 12
        assert data["metadata"]["sm_count"] == 188
        # Probe-confirmed on bwrc-bwell 2026-04-26 (REMOTE #011):
        # cluster launch + DSMEM ARE available on sm_120. The YAML
        # was previously a deferred-to-probe field; now baked in.
        assert data["metadata"]["supports_clusters"] is True
        assert data["metadata"]["supports_fp4"] is True
        assert data["metadata"]["supports_fp8"] is True
        assert data["metadata"]["interconnect_topology"] == "pcie"
        assert data["metadata"]["conformance_max_gpus"] == 2
        # Probe-derived numerics — these MUST match the bwell probe
        # so the cost model + emitter agree with the silicon.
        assert data["metadata"]["l2_cache_bytes"] == 134217728
        assert data["metadata"]["max_shared_memory_per_block_optin_bytes"] == 101376

    def test_traits_from_b200_profile(self, profiles_dir: Path) -> None:
        """End-to-end: YAML → TargetProfile → DeviceTraits → metadata
        carries every key the cost model + emitter need."""
        import yaml
        from compgen.runtime.traits import DeviceTraits
        from compgen.targets.schema import (
            ComputeUnit,
            DeviceSpec,
            Interconnect,
            MemoryLevel,
            TargetProfile,
        )

        data = yaml.safe_load((profiles_dir / "blackwell_b200.yaml").read_text())
        # Build TargetProfile from the YAML directly. (Real loader
        # may fold this into TargetProfile.from_yaml; for now we
        # do the minimum to test trait derivation.)
        devices = [
            DeviceSpec(
                device_type=d["device_type"],
                name=d["name"],
                vendor=d.get("vendor", ""),
                compute_units=[
                    ComputeUnit(
                        name=cu["name"],
                        count=int(cu["count"]),
                        peak_tflops=cu.get("peak_tflops"),
                        supported_dtypes=set(cu.get("supported_dtypes", [])),
                    )
                    for cu in d.get("compute_units", [])
                ],
                memory_hierarchy=[
                    MemoryLevel(
                        name=ml["name"],
                        size_bytes=int(ml["size_bytes"]),
                        bandwidth_gbps=ml.get("bandwidth_gbps"),
                    )
                    for ml in d.get("memory_hierarchy", [])
                ],
                features=list(d.get("features", [])),
            )
            for d in data["devices"]
        ]
        interconnects = [
            Interconnect(
                topology=i["topology"],
                bandwidth_gbps=float(i["bandwidth_gbps"]),
                devices=tuple(i.get("devices", (0, 1))),
            )
            for i in data.get("interconnects", [])
        ]
        profile = TargetProfile(
            name=data["name"],
            devices=devices,
            interconnects=interconnects,
            metadata=data.get("metadata", {}),
        )
        traits = DeviceTraits.from_target_profile(profile)
        # Every probe key the v1 honest-state doc names should be reachable.
        assert traits.metadata["sm_count"] == 132
        assert traits.metadata["compute_capability_major"] == 10
        assert traits.metadata["supports_clusters"] is True
        assert traits.metadata["supports_fp8"] is True
        assert traits.metadata["supports_fp4"] is True
        assert traits.metadata["peak_flops_per_s"] == 2.5e15
        assert traits.metadata["interconnect_topology"] == "nvlink"
        # Top-level dataclass fields derived correctly.
        assert traits.supports_event_tensors is True
        assert traits.supports_task_grid is True


class TestProbeDeviceOnlyCli:
    def test_cli_writes_device_probe_json(self, tmp_path: Path) -> None:
        """``compgen-run-conformance --probe-device-only`` writes
        device_probe.json + prints headline values. Exits 0 even on
        a CPU host (the probe falls back cleanly)."""
        import sys

        from compgen.testing.etc_conformance import _cli

        old_argv = sys.argv
        try:
            sys.argv = [
                "compgen-run-conformance",
                "--probe-device-only",
                "--output-dir",
                str(tmp_path),
            ]
            rc = _cli()
        finally:
            sys.argv = old_argv
        assert rc == 0
        path = tmp_path / "device_probe.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        # Probe always carries provenance, even on the CPU fallback.
        assert "probe_source" in loaded
