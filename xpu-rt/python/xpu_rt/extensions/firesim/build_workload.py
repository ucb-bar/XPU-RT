"""Stage a CompGen embedded bundle as a FireSim bare-metal workload.

Produces, for a given lowered model:

1. A bare-metal boot ELF linked via Chipyard's ``htif_nano.specs`` + ``htif.ld``
   with picolibc + libgloss_htif providing printf-over-HTIF and a clean
   ``tohost = 1`` poweroff.
2. A workload directory under ``<chipyard>/sims/firesim/deploy/workloads/``
   containing the boot ELF named identically to ``benchmark_name``.
3. A workload JSON (matches Merlin's ``merlin-bench-*.json`` shape exactly).
4. (Optional) ``config_runtime.yaml`` update pointing ``workload_name``
   at the new JSON.

The actual ``firesim kill / infrasetup / runworkload / kill`` sequence
lives in the shell wrapper ``scripts/compgen_firesim_run.sh`` — this
Python module's job is to produce every file the wrapper consumes.

Cross-compile is done with :func:`subprocess.run` against
``riscv64-unknown-elf-gcc`` from Chipyard's ``.conda-env/riscv-tools``;
the caller must have activated the chipyard conda env (usually via
``source chipyard/env.sh``) or explicitly point at the toolchain.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from compgen.runtime.embedded import (
    EmbeddedOptions,
    LoweredModel,
    emit_embedded,
    lower_cnn_to_c,
)

_TEMPLATE_DIR = Path(__file__).resolve().parents[3].parent / "runtime" / "templates" / "firesim"


@dataclass
class FiresimWorkload:
    """Paths the workload build produced."""

    name: str
    bundle_dir: Path
    elf_path: Path
    workload_dir: Path
    workload_json: Path


def _render_main_c(lowered: LoweredModel, out_path: Path) -> None:
    tmpl = (_TEMPLATE_DIR / "main.c.tmpl").read_text()
    rendered = (
        tmpl.replace("@ARENA_BYTES@", str(max(lowered.arena_bytes, 1 << 20)))
        .replace("@INPUT_BYTES@", "COMPGEN_MODEL_INPUT_BYTES")
        .replace("@OUTPUT_BYTES@", "COMPGEN_MODEL_OUTPUT_BYTES")
    )
    out_path.write_text(rendered)


def _emit_input_blob(out_path: Path, raw: bytes) -> None:
    rows = []
    for i in range(0, len(raw), 32):
        rows.append("    " + ", ".join(f"0x{b:02x}" for b in raw[i : i + 32]))
    body = ",\n".join(rows)
    out_path.write_text(
        "/* Auto-generated golden input for FireSim harness. */\n"
        "#include <stdint.h>\n"
        f"/* {len(raw)} bytes = float32[...] little-endian */\n"
        "__attribute__((aligned(16)))\n"
        f"const uint8_t compgen_input_bytes[] = {{\n{body}\n}};\n"
    )


def build_firesim_workload(
    *,
    model_fixture_module: str,
    chipyard_root: Path,
    workload_name: str = "compgen-convnet",
    golden_input_path: Path | None = None,
    bundle_dir: Path | None = None,
    riscv_gcc: str = "riscv64-unknown-elf-gcc",
    riscv_ar: str = "riscv64-unknown-elf-ar",
    arch: str = "rv64imafd",
    abi: str = "lp64d",
    update_config_runtime: bool = True,
) -> FiresimWorkload:
    """Produce a FireSim-ready workload for a compiled CompGen model.

    Args:
        model_fixture_module: Importable Python module exposing
            ``build_model()`` and ``default_inputs()``. Lowered through
            :func:`compgen.runtime.embedded.lower_cnn_to_c`.
        chipyard_root: Path to the Chipyard checkout whose
            ``sims/firesim/deploy/workloads/`` will receive the staged
            boot binary + JSON.
        workload_name: Directory + JSON name under ``workloads/``.
            Must be a simple identifier (used for file names).
        golden_input_path: Optional path to a saved input tensor. When
            present, those bytes are baked into the ELF. Otherwise the
            fixture's ``default_inputs()`` is used.
        bundle_dir: Where to write the embedded bundle sources. Defaults
            to ``/tmp/compgen_firesim_<name>``.
        riscv_gcc: Cross compiler (bare-metal RISC-V). Must be in
            PATH (typically after ``source chipyard/env.sh``).
        riscv_ar: Cross archiver, same conda env as ``riscv_gcc``.
        arch: ``-march`` flag for the bare-metal build.
        abi: ``-mabi`` flag.
        update_config_runtime: If True, point
            ``<chipyard>/sims/firesim/deploy/config_runtime.yaml``'s
            ``workload.workload_name`` at the new JSON.

    Returns:
        :class:`FiresimWorkload` with paths to every artifact.

    Raises:
        subprocess.CalledProcessError: If the cross toolchain fails.
        FileNotFoundError: If ``chipyard_root/sims/firesim/deploy`` is
            missing (chipyard not set up).
    """
    import importlib
    import json

    mod = importlib.import_module(model_fixture_module)
    model = mod.build_model()
    if golden_input_path is not None:
        loaded = torch.load(golden_input_path)
        inp = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    else:
        inp = mod.default_inputs()[0]

    sample_input_shape = tuple(inp.shape[1:])

    # --- Lower + emit bundle --------------------------------------
    lowered = lower_cnn_to_c(
        model,
        sample_input_shape=sample_input_shape,
        model_name=workload_name,
    )
    bundle_dir = bundle_dir or Path(f"/tmp/compgen_firesim_{workload_name}")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    emit_embedded(
        bundle_dir,
        lowered_model=lowered,
        options=EmbeddedOptions(
            model_name=workload_name,
            version="firesim",
            cross_compiler=riscv_gcc,
            archiver=riscv_ar,
            march=arch,
            mabi=abi,
        ),
    )

    # --- Emit bare-metal main + input blob ------------------------
    _render_main_c(lowered, bundle_dir / "firesim_main.c")
    raw_input = inp.detach().cpu().numpy().astype("<f4").tobytes()
    assert len(raw_input) == lowered.input_bytes
    _emit_input_blob(bundle_dir / "input_blob.c", raw_input)

    # --- Cross-compile bare-metal ELF -----------------------------
    deploy_dir = chipyard_root / "sims" / "firesim" / "deploy"
    if not deploy_dir.is_dir():
        raise FileNotFoundError(f"FireSim deploy dir missing: {deploy_dir}")
    workload_dir = deploy_dir / "workloads" / workload_name
    workload_dir.mkdir(parents=True, exist_ok=True)
    elf_path = workload_dir / workload_name
    if elf_path.exists():
        elf_path.unlink()

    # Mirror chipyard/tests CMakeLists.txt flags verbatim:
    #   -march=rv64imafd -mabi=lp64d -mcmodel=medany
    #   -specs=htif_nano.specs + -T htif.ld (both provided by the specs)
    #   -static -fno-common -fno-builtin-printf
    c_sources = [
        bundle_dir / "arena.c",
        bundle_dir / "ops.c",
        bundle_dir / "compgen_model_forward.c",
        bundle_dir / "compgen_model.c",
        bundle_dir / "model_blob.c",
        bundle_dir / "firesim_main.c",
        bundle_dir / "input_blob.c",
    ]
    cmd = [
        riscv_gcc,
        f"-march={arch}",
        f"-mabi={abi}",
        "-mcmodel=medany",
        "-O2",
        "-std=gnu99",
        "-Wall",
        "-fno-common",
        "-fno-builtin-printf",
        "-specs=htif_nano.specs",
        f"-I{bundle_dir}",
        *map(str, c_sources),
        "-static",
        "-o",
        str(elf_path),
    ]
    subprocess.run(cmd, check=True)

    # --- Emit workload JSON ---------------------------------------
    workload_json = deploy_dir / "workloads" / f"{workload_name}.json"
    workload_json.write_text(
        json.dumps(
            {
                "benchmark_name": workload_name,
                "common_bootbinary": workload_name,
                "common_rootfs": None,
                "common_outputs": [],
                "common_simulation_outputs": ["uartlog"],
            },
            indent=2,
        )
        + "\n"
    )

    # --- Update config_runtime.yaml -------------------------------
    if update_config_runtime:
        import yaml

        cfg = deploy_dir / "config_runtime.yaml"
        data = yaml.safe_load(cfg.read_text())
        data.setdefault("workload", {})
        data["workload"]["workload_name"] = f"{workload_name}.json"
        cfg.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))

    return FiresimWorkload(
        name=workload_name,
        bundle_dir=bundle_dir,
        elf_path=elf_path,
        workload_dir=workload_dir,
        workload_json=workload_json,
    )


__all__ = ["FiresimWorkload", "build_firesim_workload"]
