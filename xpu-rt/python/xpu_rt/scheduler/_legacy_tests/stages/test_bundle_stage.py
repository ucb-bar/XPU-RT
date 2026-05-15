"""Bundle-stage hard guarantees: no /tmp fallback, no silent failures.

These tests pin down the Phase-1 decisions: the bundle must never land
in an ephemeral tempdir, and constructing a ``BundleStage`` without
``output_dir`` must fail fast. They complement the broader integration
coverage in :mod:`tests.runtime.test_bundle_emit` — those exercise the
round-trip; this file exercises the contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.stages.bundle import BundleStage


class TestBundleStageOutputDirContract:
    def test_none_output_dir_rejected(self) -> None:
        """Ephemeral tempdirs hide bundles from callers — refuse them."""
        with pytest.raises(ValueError, match="output_dir"):
            BundleStage(output_dir=None)  # type: ignore[arg-type]

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        stage = BundleStage(output_dir=tmp_path / "bundle")
        # Accessing the private attribute is fine in a contract test —
        # the goal is to pin down that no tempdir is substituted.
        assert stage._output_dir == tmp_path / "bundle"

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        stage = BundleStage(output_dir=str(tmp_path / "bundle"))
        assert stage._output_dir == Path(str(tmp_path / "bundle"))

    def test_never_creates_its_own_tempdir(self, tmp_path: Path) -> None:
        """The stage must not synthesize a tempdir — it must use exactly
        what the caller passed. (pytest's own tmp_path lives under /tmp
        on Linux, so we can't blanket-ban /tmp in the stored path; we
        can only verify identity with the input.)"""
        given = tmp_path / "bundle"
        stage = BundleStage(output_dir=given)
        assert stage._output_dir == given

    def test_no_tempfile_mkdtemp_import(self) -> None:
        """Guard against a regression re-introducing the fallback:
        ``tempfile`` must not even be imported by ``stages.bundle.stage``.
        """
        import compgen.stages.bundle.stage as bundle_stage

        assert not hasattr(bundle_stage, "tempfile"), (
            "stages.bundle.stage must not import tempfile — a tempdir fallback defeats the artifact contract"
        )


class TestFactoryOutputDirContract:
    """Each stack factory must reject None output_dir."""

    def test_cuda_factory(self) -> None:
        from compgen.stages.targets.cuda_gpu import create_cuda_gpu_stack

        with pytest.raises(ValueError, match="output_dir"):
            create_cuda_gpu_stack(output_dir=None)  # type: ignore[arg-type]

    def test_rocm_factory(self) -> None:
        from compgen.stages.targets.rocm_gpu import create_rocm_gpu_stack

        with pytest.raises(ValueError, match="output_dir"):
            create_rocm_gpu_stack(output_dir=None)  # type: ignore[arg-type]
