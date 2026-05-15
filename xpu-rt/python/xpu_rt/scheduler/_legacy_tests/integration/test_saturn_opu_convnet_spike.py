"""End-to-end integration test for the Saturn OPU Zephyr/Spike bring-up.

Exercises the full chain ``saturn_opu_compile`` → host-archive build
→ ``saturn_opu_zephyr_overlay`` → ``saturn_opu_spike_run``. The Spike
run itself is gated behind ``requires_spike`` because it needs the
Zephyr SDK + Spike binaries that CI workstations generally do not
have. Without those, the test still verifies the upstream Python
pipeline produces every artifact Zephyr expects.

Run against a real Spike (when installed)::

    ZEPHYR_CHIPYARD_SW=/scratch2/agustin/zephyr-chipyard-sw \\
        uv run pytest tests/integration/test_saturn_opu_convnet_spike.py -m requires_spike
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fixtures

from compgen.mcp.session import SessionManager
from compgen.mcp.tools.embedded import (
    compile_embedded,
    simulator_run,
    zephyr_overlay,
)


def _host_cc() -> str | None:
    return shutil.which("cc")


def _build_archive(bundle_dir: Path) -> None:
    """Compile the emitted sources into ``libcompgen_model.a`` with host cc."""
    objs: list[str] = []
    sources = [
        bundle_dir / "compgen_model.c",
        bundle_dir / "model_blob.c",
        *(bundle_dir / "kernels").glob("*.c"),
    ]
    for src in sources:
        obj = bundle_dir / (src.stem + ".o")
        subprocess.run(
            ["cc", "-std=c17", "-c", str(src), "-o", str(obj), f"-I{bundle_dir}"],
            check=True,
        )
        objs.append(str(obj))
    subprocess.run(
        ["ar", "rcs", str(bundle_dir / "libcompgen_model.a"), *objs],
        check=True,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
SATURN_SPEC = str(REPO_ROOT / "examples" / "hardware_specs" / "saturn_opu.yaml")


def _rvv_only_spec(tmp_path: Path) -> str:
    """Feature-stripped HardwareSpec for A/B comparability (no ``+xopu``)."""
    source = Path(SATURN_SPEC).read_text()
    stripped = source.replace(
        '    - name: Xopu\n      version: "1.0"\n      description: "Outer-product unit (VOPACC). 8x8 MACC array, 4 matrix regs."\n',
        "",
    ).replace(
        '    - name: XopuMmt4d\n      version: "1.0"\n      description: "Mmt4d s8s8s32 16x16x128 fast-path via encoding-swapped layouts."\n',
        "",
    )
    out = tmp_path / "saturn_rvv_only.yaml"
    out.write_text(stripped)
    return str(out)


@pytest.mark.skipif(_host_cc() is None, reason="no host cc available")
@pytest.mark.parametrize("flavour", ["opu", "rvv_only"])
def test_pipeline_produces_complete_overlay(tmp_path: Path, flavour: str) -> None:
    """Verify both capability-driven paths (OPU / RVV-only) emit a complete overlay.

    This is the CompGen-side half of the kill-test — it asserts the
    Python pipeline produces every artifact ``west build`` needs,
    *without* requiring the Zephyr SDK or Spike. Spec selection drives
    which ukernel lane is emitted; no ``use_opu`` toggle.
    """
    sm = SessionManager()
    bundle_dir = tmp_path / f"bundle_{flavour}"
    spec = SATURN_SPEC if flavour == "opu" else _rvv_only_spec(tmp_path)
    compile_result = compile_embedded(
        sm,
        output_dir=str(bundle_dir),
        model_module="tests.fixtures.saturn_opu_convnet.model",
        spec_path=spec,
    )
    assert compile_result["ok"], compile_result
    if flavour == "opu":
        assert any("xopu" in n for n in compile_result["ukernels"])
    else:
        assert not any("xopu" in n for n in compile_result["ukernels"])
        assert any("rvv" in n for n in compile_result["ukernels"])

    _build_archive(bundle_dir)

    zephyr_root = tmp_path / "zephyr-chipyard-sw"
    (zephyr_root / "samples").mkdir(parents=True)
    overlay_result = zephyr_overlay(
        sm,
        zephyr_root=str(zephyr_root),
        session_id=compile_result["session_id"],
        sample_name=f"compgen_convnet_{flavour}",
    )
    assert overlay_result["ok"], overlay_result
    sample_root = Path(overlay_result["overlay_dir"])
    for required in [
        "CMakeLists.txt",
        "prj.conf",
        "custom-sections.ld",
        "libcompgen_model.a",
        "model_blob.c",
        "compgen_model.h",
        "src/main.c",
    ]:
        assert (sample_root / required).exists(), required


def _zephyr_root() -> Path | None:
    env = os.environ.get("ZEPHYR_CHIPYARD_SW")
    if env:
        return Path(env)
    default = Path("/scratch2/agustin/zephyr-chipyard-sw")
    return default if default.exists() else None


@pytest.mark.requires_spike
@pytest.mark.skipif(
    shutil.which("west") is None or shutil.which("spike") is None,
    reason="Zephyr SDK + spike required",
)
def test_spike_runs_compgen_convnet() -> None:
    """End-to-end ``west build`` + ``spike`` run against a real Spike.

    Assumes the Zephyr SDK + Spike are on ``PATH`` and
    ``ZEPHYR_CHIPYARD_SW`` points at a prepared clone. This test does
    not check numerical correctness — it asserts that the build
    succeeds and the sample's ``compgen: invoke ok`` banner appears in
    Spike's UART log. Numerical correctness is covered by the
    ctypes-based level-2 tests (``test_exo_riscv_opu.py``) which run
    without hardware.
    """
    zephyr = _zephyr_root()
    if zephyr is None:
        pytest.skip("ZEPHYR_CHIPYARD_SW not set and /scratch2/agustin/zephyr-chipyard-sw absent")

    sm = SessionManager()
    bundle_dir = zephyr / "_compgen_bundle"
    # Spike's stock build lacks +xopu — use the RVV-only spec so the
    # capability-driven pipeline routes mmt4d to the pure RVV fallback.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        spec = _rvv_only_spec(Path(td))
        compile_result = compile_embedded(
            sm,
            output_dir=str(bundle_dir),
            model_module="tests.fixtures.saturn_opu_convnet.model",
            spec_path=spec,
        )
    assert compile_result["ok"], compile_result

    # Cross-build the archive. Developers with the Zephyr SDK already
    # have riscv64-unknown-elf-gcc on PATH; the Makefile's default CC
    # drops straight through.
    subprocess.run(["make"], cwd=bundle_dir, check=True)

    overlay = zephyr_overlay(
        sm,
        zephyr_root=str(zephyr),
        session_id=compile_result["session_id"],
        sample_name="compgen_convnet_rvv",
    )
    assert overlay["ok"], overlay

    run = simulator_run(
        sm,
        spec_path=compile_result.get("spec_path"),
        zephyr_root=str(zephyr),
        sample_name="compgen_convnet_rvv",
        simulator_override="spike --isa=rv64gcv",
        execute=True,
        timeout_s=600,
    )
    # Build must succeed; simulator run may fail if an OPU-specific op
    # slipped through (shouldn't, but the kill criterion is that the
    # banner is present in the log).
    assert run.get("build_returncode") == 0, run.get("build_tail", "")
    assert "compgen: invoke ok" in run.get("simulator_tail", ""), run.get("simulator_tail", "")
