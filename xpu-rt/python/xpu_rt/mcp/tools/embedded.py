"""Backend- and model-agnostic MCP tools for embedded deployment.

Four verbs that work for *any* target / model pair whose HardwareSpec
declares ``deployment_model: zephyr_rtos`` (or ``bare_metal``). The
Saturn OPU → Zephyr → Chipyard/FireSim bring-up uses these without
any backend-specific code in the tool surface; backend specifics live
in the kernel providers (``compgen.kernels.providers.*``) and overlay
generators (``compgen.extensions.*``) that the tools delegate to.

Verbs:

* ``compile_embedded`` — load a PyTorch module, consult the target's
  HardwareSpec capabilities, emit a portable C ABI bundle
  (``compgen_model.{h,c}`` + ``model_blob.c`` + ``kernels/*.c`` +
  Makefile). Ukernel lane is chosen by the spec's features, not by
  the caller.
* ``zephyr_overlay`` — drop a bundle into a Zephyr sample tree for
  any target whose spec deploys under Zephyr. The overlay is
  independent of whether the target is Saturn OPU, a pure-RVV
  Shuttle, or a vendor NPU.
* ``simulator_run`` — return (and optionally execute) a simulator
  invocation. The default simulator command comes from the spec's
  ``verification_surface.simulator_command`` field, so Spike, RTL
  sims, or vendor sims all plug in through the same verb.
* ``firesim_workload`` — emit a FireMarshal workload JSON for any
  bootable ELF. FireSim is a specific simulator, not a backend, so
  this tool is orthogonal to which target produced the ELF.

Every handler is a pure-Python callable returning a JSON-serialisable
dict.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from compgen.mcp.session import SessionManager


def _default_zephyr_root() -> str | None:
    return os.environ.get("ZEPHYR_CHIPYARD_SW") or None


def _load_fixture_from_path(fixture_path: str) -> tuple[Any, tuple[Any, ...]]:
    """Import a ``.py`` file by filesystem path without touching ``sys.path``."""
    path = Path(fixture_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model_path does not exist: {path}")
    module_name = f"_compgen_user_model_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model(), mod.default_inputs()


def _load_fixture_model(fixture_module: str) -> tuple[Any, tuple[Any, ...]]:
    """Import a dotted module exposing ``build_model()`` and ``default_inputs()``."""
    mod = importlib.import_module(fixture_module)
    return mod.build_model(), mod.default_inputs()


def _flat_byte_size(tensor: Any) -> int:
    try:
        return int(tensor.numel() * tensor.element_size())
    except AttributeError:
        return 0


def compile_embedded(
    sm: SessionManager,
    *,
    output_dir: str,
    demo: str | None = None,
    model_module: str | None = None,
    model_path: str | None = None,
    spec_path: str | None = None,
    spec_demo: str | None = None,
    model_name: str = "compgen_model",
    version: str = "0.0.1",
    include_ops: tuple[str, ...] = ("matmul", "im2col"),
    session_id: str | None = None,
) -> dict[str, Any]:
    """Compile any PyTorch model for any embedded target.

    Ukernel selection is entirely capability-driven: the handler loads
    the HardwareSpec, extracts its ``features`` and supported ops,
    then asks each registered kernel provider which ukernels it can
    emit for the resulting ``KernelContract``. No backend-specific
    flags on this surface. A target advertising ``+xopu`` picks VOPACC;
    a target advertising only ``+v`` picks the RVV fallback; a target
    advertising neither picks scalar. The user does not choose.

    Exactly one model source and one spec source must be supplied.

    Model source (choose one):
        * ``demo="saturn_opu_convnet"`` — packaged demo shipped inside
          ``compgen.examples.*``. Works after ``pip install compgen``
          with no source-tree access; see :func:`compgen.examples.list_demos`.
        * ``model_module="pkg.mod"`` — an importable dotted module
          exposing ``build_model()`` and ``default_inputs()``. Caller
          ensures the module resolves on ``sys.path``.
        * ``model_path="/abs/path/to/model.py"`` — a ``.py`` file on
          disk. Loaded via ``importlib.util.spec_from_file_location``,
          never touching ``sys.path``.

    Spec source (choose one):
        * ``spec_demo="saturn_opu"`` — packaged HardwareSpec YAML shipped
          under ``compgen.examples.hardware_specs``. See
          :func:`compgen.examples.list_specs`.
        * ``spec_path="/abs/path/to/target.yaml"`` — a HardwareSpec
          YAML (``v2.0``) on disk.

    Args:
        output_dir: Destination for the emitted bundle.
        model_name: Used in the generated C identifiers.
        version: Version string stamped into ``compgen_model.h``.
        include_ops: Op families the caller wants ukernels for. The
            provider may skip ops outside its domain. Default covers
            the mmt4d + im2col kernels the ConvNet bring-up needs.
        session_id: Optional MCP session to attach results to.

    Returns:
        ``{"ok": True, "output_dir", "header", "ukernels",
        "selected_lanes", "target_name", "target_features", ...}``.
    """

    from compgen import examples
    from compgen.kernels.provider import KernelContract
    from compgen.kernels.providers.exo_riscv_opu import emit_kernels
    from compgen.runtime.embedded import EmbeddedOptions, emit_embedded
    from compgen.targetgen.load import load_hardware_spec

    session = sm.open(session_id) if session_id else sm.open()

    model_sources = [("demo", demo), ("model_module", model_module), ("model_path", model_path)]
    supplied_model = [(k, v) for k, v in model_sources if v]
    if len(supplied_model) != 1:
        return {
            "ok": False,
            "error": (
                "exactly one of demo/model_module/model_path must be set; "
                f"got {[k for k, _ in supplied_model] or 'none'}"
            ),
            "session_id": session.session_id,
        }
    spec_sources = [("spec_demo", spec_demo), ("spec_path", spec_path)]
    supplied_spec = [(k, v) for k, v in spec_sources if v]
    if len(supplied_spec) != 1:
        return {
            "ok": False,
            "error": (
                "exactly one of spec_demo/spec_path must be set; "
                f"got {[k for k, _ in supplied_spec] or 'none'}"
            ),
            "session_id": session.session_id,
        }

    model_source_kind, model_source_value = supplied_model[0]
    try:
        if model_source_kind == "demo":
            resolved_module = examples.resolve_demo_module(model_source_value)
            model, sample_inputs = _load_fixture_model(resolved_module)
            model_source_display = f"demo:{model_source_value}"
        elif model_source_kind == "model_module":
            model, sample_inputs = _load_fixture_model(model_source_value)
            model_source_display = model_source_value
        else:  # model_path
            model, sample_inputs = _load_fixture_from_path(model_source_value)
            model_source_display = model_source_value
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"failed to load model ({model_source_kind}={model_source_value!r}): {exc}",
            "session_id": session.session_id,
        }

    spec_source_kind, spec_source_value = supplied_spec[0]
    if spec_source_kind == "spec_demo":
        try:
            resolved_spec_path = str(examples.resolve_spec_path(spec_source_value))
        except ValueError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "session_id": session.session_id,
            }
    else:
        resolved_spec_path = spec_source_value

    input_bytes = sum(_flat_byte_size(t) for t in sample_inputs)
    import torch  # local import — optional dep at tool-discovery time

    with torch.no_grad():
        out = model(*sample_inputs)
    output_bytes = _flat_byte_size(out)

    try:
        spec = load_hardware_spec(resolved_spec_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"failed to load spec '{resolved_spec_path}': {exc}",
            "session_id": session.session_id,
        }
    target_name = spec.name
    target_features = tuple(
        sorted({ext.name.lower() for ext in spec.isa.extensions})
    )
    constraints_common = {"features": list(target_features)}

    ukernels: list[Any] = []
    op_set = {op.lower() for op in include_ops}
    if "matmul" in op_set or "mmt4d" in op_set:
        ukernels.extend(
            emit_kernels(
                KernelContract(
                    op_family="matmul",
                    dtypes=("int8", "int8", "int32"),
                    target_name=target_name,
                    hardware_key=target_name,
                    constraints={**constraints_common, "inner_tile": [16, 16, 128]},
                )
            )
        )
    if "im2col" in op_set or "conv2d" in op_set:
        ukernels.extend(
            emit_kernels(
                KernelContract(
                    op_family="im2col",
                    dtypes=("int8",),
                    target_name=target_name,
                    hardware_key=target_name,
                    constraints=constraints_common,
                )
            )
        )
    selected_lanes = sorted({
        k.name.rsplit("_", 1)[-1] for k in ukernels
    })  # e.g. {"xopu", "rvv"}

    options = EmbeddedOptions(
        model_name=model_name,
        version=version,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )
    artifacts = emit_embedded(
        output_dir,
        options=options,
        model_blob=b"",
        ukernels=ukernels,
    )

    session.metadata["embedded_bundle"] = str(artifacts.output_dir)
    session.metadata["embedded_spec"] = resolved_spec_path
    session.metadata["embedded_input_bytes"] = input_bytes
    session.metadata["embedded_output_bytes"] = output_bytes
    session.metadata["embedded_target_name"] = target_name

    return {
        "ok": True,
        "session_id": session.session_id,
        "output_dir": str(artifacts.output_dir),
        "header": str(artifacts.header),
        "runtime_source": str(artifacts.runtime_source),
        "blob_source": str(artifacts.blob_source),
        "makefile": str(artifacts.makefile),
        "ukernels": [p.name for p in artifacts.ukernel_sources],
        "kernel_contracts": (
            str(artifacts.kernel_contracts)
            if artifacts.kernel_contracts is not None
            else None
        ),
        "model_input_bytes": input_bytes,
        "model_output_bytes": output_bytes,
        "target_name": target_name,
        "target_features": list(target_features),
        "selected_lanes": selected_lanes,
        "model_source": model_source_display,
        "spec_path": resolved_spec_path,
    }


def zephyr_overlay(
    sm: SessionManager,
    *,
    bundle_dir: str | None = None,
    zephyr_root: str | None = None,
    sample_name: str = "compgen_app",
    board: str = "spike_riscv64",
    arena_bytes: int = 8 * 1024 * 1024,
    smp: bool = False,
    mp_max_num_cpus: int = 1,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Drop an embedded bundle into a Zephyr sample tree.

    Works for any bundle emitted by :func:`compile_embedded` — the
    CompGen C ABI is target-agnostic. If ``bundle_dir`` is omitted,
    reuses the most recent bundle from the current session.
    """
    from compgen.extensions.zephyr import ZephyrOverlayOptions, emit_overlay

    session = sm.open(session_id) if session_id else sm.open()

    if bundle_dir is None:
        bundle_dir = session.metadata.get("embedded_bundle")
    if bundle_dir is None:
        return {
            "ok": False,
            "error": "no bundle_dir supplied and session has no prior compile",
            "session_id": session.session_id,
        }

    zephyr_root = zephyr_root or _default_zephyr_root()
    zephyr_path = Path(zephyr_root).expanduser()
    if not zephyr_path.exists():
        return {
            "ok": False,
            "error": f"zephyr_root not found: {zephyr_path}",
            "session_id": session.session_id,
        }

    options = ZephyrOverlayOptions(
        sample_name=sample_name,
        project_name=sample_name,
        board=board,
        arena_bytes=arena_bytes,
        smp=smp,
        mp_max_num_cpus=mp_max_num_cpus,
        model_input_bytes=session.metadata.get("embedded_input_bytes", 0),
        model_output_bytes=session.metadata.get("embedded_output_bytes", 0),
    )
    try:
        result = emit_overlay(Path(bundle_dir), zephyr_path, options)
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "session_id": session.session_id,
        }

    session.metadata["embedded_overlay"] = str(result.paths.root)

    return {
        "ok": True,
        "session_id": session.session_id,
        "overlay_dir": str(result.paths.root),
        "build_command": result.build_command,
        "run_commands": dict(result.run_commands),
        "files": sorted(
            str(p.relative_to(result.paths.root))
            for p in result.paths.root.rglob("*")
            if p.is_file()
        ),
    }


def _simulator_command(
    spec: Any,
    elf_path: Path,
    override: str | None,
) -> tuple[str, str]:
    """Return ``(simulator_name, command_line)`` for a spec + ELF.

    Reads ``verification_surface.simulator_command`` from the spec
    and appends the ELF path. Caller may override the whole command
    line via ``override`` — useful for one-off re-runs.
    """
    if override:
        return ("override", override)
    cmd = getattr(spec.verification_surface, "simulator_command", "") or ""
    if not cmd:
        # Reasonable default for Zephyr/Chipyard RISC-V targets.
        cmd = "spike --isa=rv64gcv"
    name = cmd.split()[0]
    return name, f"{cmd} {elf_path}"


def simulator_run(
    sm: SessionManager,
    *,
    spec_path: str | None = None,
    zephyr_root: str | None = None,
    sample_name: str | None = None,
    board: str = "spike_riscv64",
    elf_path: str | None = None,
    simulator_override: str | None = None,
    execute: bool = False,
    timeout_s: int = 120,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Produce (or run) the ``west build`` + simulator command pair.

    The simulator invocation is read from the HardwareSpec's
    ``verification_surface.simulator_command`` — so Spike, vendor
    sims, or RTL sims all go through the same verb. Pass
    ``simulator_override`` to pin a one-off command.
    """
    from compgen.targetgen.load import load_hardware_spec

    session = sm.open(session_id) if session_id else sm.open()

    spec_path = spec_path or session.metadata.get("embedded_spec")
    if spec_path is None:
        return {
            "ok": False,
            "error": "no spec_path supplied and session has no prior compile",
            "session_id": session.session_id,
        }
    try:
        spec = load_hardware_spec(spec_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"failed to load spec: {exc}",
            "session_id": session.session_id,
        }

    zephyr_root = zephyr_root or _default_zephyr_root()
    zephyr_path = Path(zephyr_root).expanduser()
    if sample_name is None:
        overlay = session.metadata.get("embedded_overlay", "")
        sample_name = Path(overlay).name if overlay else "compgen_app"
    sample_path = zephyr_path / "samples" / sample_name
    elf = Path(elf_path) if elf_path else zephyr_path / "build" / "zephyr" / "zephyr.elf"

    build_cmd = f"west build -p -b {board} samples/{sample_name}/"
    simulator_name, simulator_cmd = _simulator_command(spec, elf, simulator_override)

    if not execute:
        return {
            "ok": True,
            "session_id": session.session_id,
            "build_command": build_cmd,
            "simulator_command": simulator_cmd,
            "simulator_name": simulator_name,
            "sample_dir": str(sample_path),
            "executed": False,
        }

    tool0 = simulator_cmd.split()[0]
    missing = [tool for tool in ("west", tool0) if shutil.which(tool) is None]
    if missing or not sample_path.exists():
        return {
            "ok": False,
            "error": (
                f"cannot execute: missing tools={missing}, "
                f"sample_exists={sample_path.exists()}"
            ),
            "session_id": session.session_id,
            "build_command": build_cmd,
            "simulator_command": simulator_cmd,
        }

    results: dict[str, Any] = {
        "build_command": build_cmd,
        "simulator_command": simulator_cmd,
        "simulator_name": simulator_name,
    }
    for label, cmd in (("build", build_cmd), ("simulator", simulator_cmd)):
        try:
            proc = subprocess.run(
                shlex.split(cmd),
                cwd=zephyr_path,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            results[f"{label}_timeout"] = True
            break
        results[f"{label}_returncode"] = proc.returncode
        tail = (proc.stdout or "") + (proc.stderr or "")
        results[f"{label}_tail"] = tail[-8192:]
        if proc.returncode != 0:
            break

    results["ok"] = all(
        results.get(f"{k}_returncode") == 0 for k in ("build", "simulator")
    )
    results["session_id"] = session.session_id
    results["executed"] = True
    return results


def firesim_workload(
    sm: SessionManager,
    *,
    boot_binary: str,
    workload_dir: str,
    workload_name: str,
    chipyard_config: str = "",
    simulation_outputs: tuple[str, ...] = ("uartlog",),
    session_id: str | None = None,
) -> dict[str, Any]:
    """Emit a FireMarshal workload JSON for any bootable ELF.

    FireSim is a simulator, not a backend — this verb works for Saturn
    OPU, pure-RVV Shuttle, vendor RoCC accelerators, or any other ELF
    that boots on a Chipyard design. ``chipyard_config`` is optional
    metadata only; FireSim picks the actual SoC config from its own
    config tree, not from this JSON.
    """
    session = sm.open(session_id) if session_id else sm.open()

    elf_path = Path(boot_binary).expanduser().resolve()
    if not elf_path.exists():
        return {
            "ok": False,
            "error": f"boot_binary not found: {elf_path}",
            "session_id": session.session_id,
        }

    workload_path = Path(workload_dir).expanduser()
    workload_path.mkdir(parents=True, exist_ok=True)
    workload_json = workload_path / f"{workload_name}.json"

    payload = {
        "benchmark_name": workload_name,
        "common_bootbinary": str(elf_path),
        "common_rootfs": None,
        "common_simulation_outputs": list(simulation_outputs),
        "common_simulation_inputs": [],
        "common_outputs": [],
        "metadata": {
            "chipyard_config": chipyard_config,
            "source": "compgen.mcp.firesim_workload",
        },
    }
    workload_json.write_text(json.dumps(payload, indent=2) + "\n")

    firesim_cmd = f"firesim runworkload -c deploy/workloads/{workload_name}.json"

    session.metadata["firesim_workload"] = str(workload_json)

    return {
        "ok": True,
        "session_id": session.session_id,
        "workload_json": str(workload_json),
        "firesim_command": firesim_cmd,
        "chipyard_config": chipyard_config,
    }


def list_packaged_examples(sm: SessionManager, **_: Any) -> dict[str, Any]:
    """Enumerate demos and specs shipped inside the installed ``compgen`` wheel.

    The returned names are the exact strings accepted by
    ``compile_embedded(demo=..., spec_demo=...)``. No CompGen source
    tree access required — everything resolves via ``importlib.resources``
    on the installed package.
    """
    from compgen import examples

    return {
        "ok": True,
        "demos": examples.list_demos(),
        "hardware_specs": examples.list_specs(),
        "target_profiles": examples.list_target_profiles(),
    }


def find_zephyr(sm: SessionManager, **_: Any) -> dict[str, Any]:
    """Report which Zephyr / Spike toolchain pieces are reachable from here.

    Checks the standard env vars and ``$PATH`` for the binaries the
    embedded / Spike flow needs. Does no filesystem snooping beyond
    what is explicitly configured — users bring their own Zephyr
    tree. The return value lists what was found and what is still
    missing so the caller can prompt the user to fix their env.
    """
    env_keys = (
        "ZEPHYR_CHIPYARD_SW",
        "ZEPHYR_BASE",
        "ZEPHYR_SDK_INSTALL_DIR",
        "ZEPHYR_TOOLCHAIN_VARIANT",
    )
    env_report = {name: os.environ.get(name) for name in env_keys}

    binary_keys = ("spike", "west", "riscv64-zephyr-elf-gcc", "riscv64-zephyr-elf-ar")
    binary_report = {name: shutil.which(name) for name in binary_keys}

    missing_env = [name for name, value in env_report.items() if not value]
    missing_bin = [name for name, path in binary_report.items() if path is None]

    zephyr_root = env_report.get("ZEPHYR_CHIPYARD_SW") or env_report.get("ZEPHYR_BASE")
    zephyr_root_exists = bool(zephyr_root) and Path(zephyr_root).expanduser().exists()

    remediation: list[str] = []
    if not zephyr_root:
        remediation.append(
            "set $ZEPHYR_CHIPYARD_SW (preferred) or $ZEPHYR_BASE to your Zephyr checkout"
        )
    elif not zephyr_root_exists:
        remediation.append(f"$ZEPHYR_CHIPYARD_SW/$ZEPHYR_BASE points to a missing path: {zephyr_root}")
    if missing_bin:
        remediation.append(
            "install / PATH-expose: " + ", ".join(missing_bin)
        )
    if not env_report.get("ZEPHYR_SDK_INSTALL_DIR"):
        remediation.append("set $ZEPHYR_SDK_INSTALL_DIR for west build")

    return {
        "ok": zephyr_root_exists and not missing_bin,
        "env": env_report,
        "binaries": binary_report,
        "zephyr_root": zephyr_root,
        "zephyr_root_exists": zephyr_root_exists,
        "missing_env": missing_env,
        "missing_binaries": missing_bin,
        "remediation": remediation,
    }


EMBEDDED_TOOLS: list[dict[str, Any]] = [
    {
        "name": "compile_embedded",
        "description": (
            "Compile a PyTorch model for any embedded target described by a "
            "HardwareSpec; ukernel lane is chosen by target capabilities. "
            "Supply exactly one model source (demo | model_module | model_path) "
            "and exactly one spec source (spec_demo | spec_path)."
        ),
        "phase": "lifecycle",
        "handler": compile_embedded,
        "input_schema": {
            "type": "object",
            "properties": {
                "output_dir": {"type": "string"},
                "demo": {
                    "type": "string",
                    "description": (
                        "Packaged demo name under compgen.examples (e.g. "
                        "'saturn_opu_convnet'). See list_demos()."
                    ),
                },
                "model_module": {
                    "type": "string",
                    "description": "Importable dotted module exposing build_model/default_inputs.",
                },
                "model_path": {
                    "type": "string",
                    "description": "Filesystem path to a .py file exposing build_model/default_inputs.",
                },
                "spec_demo": {
                    "type": "string",
                    "description": (
                        "Packaged HardwareSpec name under "
                        "compgen.examples.hardware_specs (e.g. 'saturn_opu')."
                    ),
                },
                "spec_path": {
                    "type": "string",
                    "description": "Filesystem path to a HardwareSpec YAML.",
                },
                "model_name": {"type": "string"},
                "version": {"type": "string"},
                "include_ops": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Op families to request ukernels for.",
                },
                "session_id": {"type": "string"},
            },
            "required": ["output_dir"],
        },
    },
    {
        "name": "zephyr_overlay",
        "description": (
            "Drop an embedded bundle into a Zephyr sample tree for any target "
            "whose HardwareSpec deploys under Zephyr."
        ),
        "phase": "lifecycle",
        "handler": zephyr_overlay,
        "input_schema": {
            "type": "object",
            "properties": {
                "bundle_dir": {"type": "string"},
                "zephyr_root": {"type": "string"},
                "sample_name": {"type": "string"},
                "board": {"type": "string"},
                "arena_bytes": {"type": "integer"},
                "smp": {"type": "boolean"},
                "mp_max_num_cpus": {"type": "integer"},
                "session_id": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "simulator_run",
        "description": (
            "Return (and optionally execute) the west-build + simulator "
            "command pair for a compiled overlay; simulator read from the spec."
        ),
        "phase": "job",
        "handler": simulator_run,
        "input_schema": {
            "type": "object",
            "properties": {
                "spec_path": {"type": "string"},
                "zephyr_root": {"type": "string"},
                "sample_name": {"type": "string"},
                "board": {"type": "string"},
                "elf_path": {"type": "string"},
                "simulator_override": {"type": "string"},
                "execute": {"type": "boolean", "default": False},
                "timeout_s": {"type": "integer"},
                "session_id": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "firesim_workload",
        "description": (
            "Emit a FireMarshal workload JSON for any bootable ELF, regardless "
            "of which target produced it."
        ),
        "phase": "job",
        "handler": firesim_workload,
        "input_schema": {
            "type": "object",
            "properties": {
                "boot_binary": {"type": "string"},
                "workload_dir": {"type": "string"},
                "workload_name": {"type": "string"},
                "chipyard_config": {"type": "string"},
                "simulation_outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "session_id": {"type": "string"},
            },
            "required": ["boot_binary", "workload_dir", "workload_name"],
        },
    },
    {
        "name": "list_packaged_examples",
        "description": (
            "List demos, hardware specs, and target profiles shipped inside "
            "the installed compgen wheel. Names returned here are directly "
            "accepted by compile_embedded(demo=..., spec_demo=...) and by "
            "the declarative target-profile API."
        ),
        "phase": "inspect",
        "handler": list_packaged_examples,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "find_zephyr",
        "description": (
            "Report whether the Zephyr / RISC-V toolchain pieces the embedded "
            "flow needs are reachable from the current environment. Returns a "
            "remediation checklist when pieces are missing; does not modify "
            "the environment or perform filesystem search."
        ),
        "phase": "inspect",
        "handler": find_zephyr,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


__all__ = [
    "EMBEDDED_TOOLS",
    "compile_embedded",
    "find_zephyr",
    "firesim_workload",
    "list_packaged_examples",
    "simulator_run",
    "zephyr_overlay",
]
