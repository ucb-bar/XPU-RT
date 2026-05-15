"""Tests for the backend-agnostic embedded MCP tool surface.

These tools — ``compile_embedded``, ``zephyr_overlay``, ``simulator_run``,
``firesim_workload`` — are not tied to Saturn OPU. They take a
HardwareSpec + a model module and let the target's advertised
capabilities drive everything downstream (ukernel lane, simulator
command, etc.). The tests exercise the Saturn OPU ConvNet as one
concrete case, plus a second spec with ``+xopu`` stripped to prove
the same tools route to the RVV fallback automatically.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # fixtures import

from compgen.mcp.session import SessionManager
from compgen.mcp.tools import ALL_TOOLS
from compgen.mcp.tools.embedded import (
    EMBEDDED_TOOLS,
    compile_embedded,
    firesim_workload,
    simulator_run,
    zephyr_overlay,
)


def _sm() -> SessionManager:
    return SessionManager()


def _fake_zephyr_root(tmp_path: Path) -> Path:
    root = tmp_path / "zephyr-chipyard-sw"
    (root / "samples").mkdir(parents=True)
    return root


REPO_ROOT = Path(__file__).resolve().parents[2]
SATURN_SPEC = str(REPO_ROOT / "examples" / "hardware_specs" / "saturn_opu.yaml")
CONVNET_FIXTURE = "tests.fixtures.saturn_opu_convnet.model"


def _rvv_only_spec(tmp_path: Path) -> str:
    """Derive a feature-stripped spec (no ``+xopu``) for A/B testing."""
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


def test_all_tools_contains_embedded_verbs() -> None:
    names = {t["name"] for t in ALL_TOOLS}
    assert {
        "compile_embedded",
        "zephyr_overlay",
        "simulator_run",
        "firesim_workload",
    }.issubset(names)
    for tool in EMBEDDED_TOOLS:
        assert callable(tool["handler"])
        assert "input_schema" in tool
        # Name must not contain a backend identifier — we want generic verbs.
        assert "saturn" not in tool["name"] and "opu" not in tool["name"]


def test_compile_embedded_picks_opu_lane_from_saturn_spec(tmp_path: Path) -> None:
    result = compile_embedded(
        _sm(),
        output_dir=str(tmp_path / "bundle"),
        model_module=CONVNET_FIXTURE,
        spec_path=SATURN_SPEC,
    )
    assert result["ok"], result
    assert any("xopu" in n for n in result["ukernels"])
    assert "xopu" in result["selected_lanes"]
    # xopu extension is present in target_features after lowercasing.
    assert "xopu" in result["target_features"]
    # Header declares the VOPACC ukernel symbol.
    header = Path(result["header"]).read_text()
    assert "compgen_mmt4d_s8s8s32_16x16x128_xopu" in header
    # Input / output sizes match the ConvNet fixture.
    assert result["model_input_bytes"] == 49152
    assert result["model_output_bytes"] == 64


def test_compile_embedded_routes_to_rvv_when_spec_lacks_xopu(tmp_path: Path) -> None:
    rvv_spec = _rvv_only_spec(tmp_path)
    result = compile_embedded(
        _sm(),
        output_dir=str(tmp_path / "bundle_rvv"),
        model_module=CONVNET_FIXTURE,
        spec_path=rvv_spec,
    )
    assert result["ok"], result
    names = result["ukernels"]
    assert any("rvv" in n and "mmt4d" in n for n in names)
    assert not any("xopu" in n for n in names)
    rvv_src = (Path(result["output_dir"]) / "kernels" / "mmt4d_s8s8s32_16x16x128_rvv.c").read_text()
    assert ".insn r 0x57" not in rvv_src
    assert "rvv" in result["selected_lanes"]
    assert "xopu" not in result["target_features"]


def test_zephyr_overlay_from_session_metadata(tmp_path: Path) -> None:
    sm = _sm()
    compile_result = compile_embedded(
        sm,
        output_dir=str(tmp_path / "bundle"),
        model_module=CONVNET_FIXTURE,
        spec_path=SATURN_SPEC,
    )
    session_id = compile_result["session_id"]

    bundle_dir = Path(compile_result["output_dir"])
    objs = []
    for src in [
        bundle_dir / "compgen_model.c",
        bundle_dir / "model_blob.c",
        *(bundle_dir / "kernels").glob("*.c"),
    ]:
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

    zephyr_root = _fake_zephyr_root(tmp_path)
    overlay = zephyr_overlay(
        sm,
        zephyr_root=str(zephyr_root),
        session_id=session_id,
    )
    assert overlay["ok"], overlay
    assert overlay["overlay_dir"].endswith("samples/compgen_app")
    for required in [
        "CMakeLists.txt",
        "prj.conf",
        "custom-sections.ld",
        "libcompgen_model.a",
        "model_blob.c",
        "compgen_model.h",
        "src/main.c",
    ]:
        assert required in overlay["files"], required


def test_zephyr_overlay_reports_missing_bundle(tmp_path: Path) -> None:
    zephyr_root = _fake_zephyr_root(tmp_path)
    result = zephyr_overlay(
        _sm(),
        zephyr_root=str(zephyr_root),
        bundle_dir=str(tmp_path / "does-not-exist"),
    )
    assert not result["ok"]
    assert "missing required artifact" in result["error"]


def test_simulator_run_dry_run_uses_spec_command(tmp_path: Path) -> None:
    result = simulator_run(
        _sm(),
        spec_path=SATURN_SPEC,
        zephyr_root=str(_fake_zephyr_root(tmp_path)),
        sample_name="compgen_app",
        execute=False,
    )
    assert result["ok"]
    assert result["executed"] is False
    # HardwareSpec advertises ``spike --isa=rv64gcv`` as the simulator
    # command; the tool picks it up without the caller hard-coding it.
    assert "spike" in result["simulator_command"]
    assert result["simulator_name"] == "spike"


def test_simulator_run_override_wins(tmp_path: Path) -> None:
    result = simulator_run(
        _sm(),
        spec_path=SATURN_SPEC,
        zephyr_root=str(_fake_zephyr_root(tmp_path)),
        simulator_override="qemu-system-riscv64 -bios none -kernel zephyr.elf",
        execute=False,
    )
    assert result["ok"]
    assert result["simulator_command"].startswith("qemu-system-riscv64")
    assert result["simulator_name"] == "override"


def test_simulator_run_execute_reports_missing_toolchain(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    result = simulator_run(
        _sm(),
        spec_path=SATURN_SPEC,
        zephyr_root=str(_fake_zephyr_root(tmp_path)),
        execute=True,
    )
    assert not result["ok"]
    assert "missing tools" in result["error"]


def test_simulator_run_execute_skips_build_when_explicit_empty(tmp_path: Path) -> None:
    """``build_command=""`` runs only the simulator — no Zephyr west required.

    Mirrors the chipyard / RTL-sim flow where the simulator_command itself
    builds + runs (``make -C sims/vcs run-binary BINARY=…``).
    """
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "name: test-target\n"
        "schema_version: '2.0'\n"
        "verification_surface:\n"
        "  has_simulator: true\n"
        '  simulator_command: "true"\n'
    )
    elf = tmp_path / "x.elf"
    elf.touch()

    result = simulator_run(
        _sm(),
        spec_path=str(spec),
        elf_path=str(elf),
        execute=True,
        build_command="",
        timeout_s=10,
    )

    assert result.get("ok") is True, result
    assert result.get("simulator_returncode") == 0, result
    # Build step must not have run.
    assert "build_returncode" not in result, result
    assert result.get("build_command") == ""


def test_simulator_command_substitutes_sim_backend_from_spec(tmp_path: Path) -> None:
    """``verification_surface.sim_backend`` fills the ``{sim_backend}``
    placeholder when the caller doesn't override (REQ-006)."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "name: t\n"
        "schema_version: '2.0'\n"
        "verification_surface:\n"
        "  has_simulator: true\n"
        "  sim_backend: vcs\n"
        '  simulator_command: "make -C sims/{sim_backend} run-binary BINARY={elf}"\n'
    )
    elf = tmp_path / "x.elf"
    elf.touch()

    res = simulator_run(_sm(), spec_path=str(spec_path), elf_path=str(elf), execute=False)
    assert res["ok"]
    assert "sims/vcs" in res["simulator_command"], res["simulator_command"]


def test_simulator_command_unknown_placeholder_surfaces_verbatim(tmp_path: Path) -> None:
    """Strict policy: a typo'd placeholder leaves ``{key}`` in the
    rendered command rather than silently disappearing.

    Without the strict default, a misspelled ``{cypath_root}`` (typo
    of ``{chipyard_root}``) would evaporate to empty string and the
    rendered ``make -C /sims/...`` would look syntactically OK but
    point at the wrong directory. Strict mode keeps the typo visible
    so the caller fixes it instead of debugging a silent miss."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "name: t\n"
        "schema_version: '2.0'\n"
        "verification_surface:\n"
        "  has_simulator: true\n"
        '  simulator_command: "make -C {cypath_root}/sims/vcs BINARY={elf}"\n'
    )
    elf = tmp_path / "x.elf"
    elf.touch()

    res = simulator_run(_sm(), spec_path=str(spec_path), elf_path=str(elf), execute=False)
    assert res["ok"]
    # The typo is visible in the rendered command — caller can fix it.
    assert "{cypath_root}" in res["simulator_command"], res["simulator_command"]


def test_simulator_command_no_trailing_newline_on_substitution_path(tmp_path: Path) -> None:
    """The substitution path no longer appends a stray ``\\n`` to the
    rendered command. Pack-side workaround for the loose REQ-001
    assertion is gone now that the assertion was tightened."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "name: t\n"
        "schema_version: '2.0'\n"
        "verification_surface:\n"
        "  has_simulator: true\n"
        '  simulator_command: "make -C /cy/sims/vcs BINARY={elf}"\n'
    )
    elf = tmp_path / "x.elf"
    elf.touch()
    res = simulator_run(_sm(), spec_path=str(spec_path), elf_path=str(elf), execute=False)
    cmd = res["simulator_command"]
    assert not cmd.endswith("\n"), repr(cmd)
    # Single trailing newline isn't part of the canonical command anymore;
    # consumers can append their own if their shell needs it.


def test_simulator_command_caller_substitution_overrides_spec_sim_backend(tmp_path: Path) -> None:
    """Caller-supplied ``substitutions['sim_backend']`` overrides the spec."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "name: t\n"
        "schema_version: '2.0'\n"
        "verification_surface:\n"
        "  has_simulator: true\n"
        "  sim_backend: verilator\n"
        '  simulator_command: "make -C sims/{sim_backend} run-binary BINARY={elf}"\n'
    )
    elf = tmp_path / "x.elf"
    elf.touch()

    res = simulator_run(
        _sm(),
        spec_path=str(spec_path),
        elf_path=str(elf),
        execute=False,
        substitutions={"sim_backend": "firesim"},
    )
    assert res["ok"]
    assert "sims/firesim" in res["simulator_command"]
    assert "sims/verilator" not in res["simulator_command"]


def test_simulator_run_execute_uses_spec_build_command(tmp_path: Path) -> None:
    """``verification_surface.build_command`` flows into execution."""
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "name: test-target\n"
        "schema_version: '2.0'\n"
        "verification_surface:\n"
        "  has_simulator: true\n"
        '  simulator_command: "true"\n'
        '  build_command: "true"\n'
    )
    elf = tmp_path / "x.elf"
    elf.touch()

    result = simulator_run(
        _sm(),
        spec_path=str(spec),
        elf_path=str(elf),
        execute=True,
        timeout_s=10,
    )

    assert result.get("ok") is True, result
    assert result.get("build_returncode") == 0, result
    assert result.get("simulator_returncode") == 0, result
    assert result.get("build_command") == "true"


def test_firesim_workload_emits_json(tmp_path: Path) -> None:
    elf = tmp_path / "zephyr.elf"
    elf.write_bytes(b"ELFPLACEHOLDER")
    workload_dir = tmp_path / "workloads"
    result = firesim_workload(
        _sm(),
        boot_binary=str(elf),
        workload_dir=str(workload_dir),
        workload_name="compgen-app",
        chipyard_config="OPUV128D64ShuttleConfig",
    )
    assert result["ok"]
    payload = json.loads(Path(result["workload_json"]).read_text())
    assert payload["benchmark_name"] == "compgen-app"
    assert payload["common_bootbinary"] == str(elf.resolve())
    assert payload["common_rootfs"] is None
    assert payload["common_simulation_outputs"] == ["uartlog"]
    assert payload["metadata"]["chipyard_config"] == "OPUV128D64ShuttleConfig"


def test_firesim_workload_rejects_missing_elf(tmp_path: Path) -> None:
    result = firesim_workload(
        _sm(),
        boot_binary=str(tmp_path / "nope.elf"),
        workload_dir=str(tmp_path / "workloads"),
        workload_name="x",
    )
    assert not result["ok"]
    assert "not found" in result["error"]
