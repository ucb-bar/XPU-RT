"""Locate and import modelblaster's reusable pipeline modules.

modelblaster is a nested submodule and is *not* pip-installed here: its
`src/modelblaster/__init__.py` is a namespace shim that extends
`__path__` back to the repo root, so putting `<mb>/src` on sys.path is
enough to import `modelblaster.pipeline.*`.  The three modules Flow C
reuses (`core_registry`, `ingest_xpurt_schedule`, `profile_writer`) are
stdlib-only, so this costs nothing — no torch, no ultralytics.

`extract_graph` is the exception: it needs torch, so it runs
out-of-process in modelblaster's own conda env (see `mb_python()`).
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys


def repo_root() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # flow_c/
    return os.path.abspath(os.path.join(here, "..", ".."))               # XPU-RT/


def _submodule_path(root: str, name: str, parent: str | None = None) -> str | None:
    base = parent or root
    try:
        out = subprocess.check_output(
            ["git", "-C", base, "config", "-f", ".gitmodules",
             "--get", f"submodule.{name}.path"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return os.path.join(base, out) if out else None


@functools.lru_cache(maxsize=1)
def modelblaster_root() -> str:
    """Resolve the modelblaster checkout, honouring $MODELBLASTER_ROOT."""
    env = os.environ.get("MODELBLASTER_ROOT")
    if env:
        return os.path.abspath(env)
    root = repo_root()
    zcs = _submodule_path(root, "zephyr-chipyard-sw") or os.path.join(root, "zephyr-chipyard-sw")
    mb = _submodule_path(root, "modelblaster", parent=zcs) or os.path.join(zcs, "modelblaster")
    if not os.path.isdir(mb):
        raise FileNotFoundError(
            f"modelblaster checkout not found at {mb}; set MODELBLASTER_ROOT or run:\n"
            f"  git -C {zcs} submodule update --init modelblaster")
    return mb


@functools.lru_cache(maxsize=1)
def _install_path() -> None:
    src = os.path.join(modelblaster_root(), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    xpurt = os.path.join(repo_root(), "xpu-rt")
    if xpurt not in sys.path:
        sys.path.insert(0, xpurt)


def core_registry():
    _install_path()
    from modelblaster.pipeline import core_registry as m
    return m


def ingest_xpurt_schedule():
    _install_path()
    from modelblaster.pipeline import ingest_xpurt_schedule as m
    return m


def profile_writer():
    _install_path()
    from modelblaster.pipeline import profile_writer as m
    return m


def emit_dispatch_graph():
    _install_path()
    from modelblaster.pipeline import emit_dispatch_graph as m
    return m


def onnx_python() -> str:
    """Interpreter for the ONNX export — needs torch *and* onnx.

    modelblaster's own env has torch but not onnx (its flow never leaves
    PyTorch), so this falls back to any conda env that has both rather
    than installing into someone else's environment. Override with
    $FLOWC_ONNX_PYTHON.
    """
    env = os.environ.get("FLOWC_ONNX_PYTHON")
    if env:
        return env
    cands = [mb_python(), sys.executable]
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        envs = os.path.join(os.path.dirname(os.path.dirname(conda)), "envs")
        if os.path.isdir(envs):
            cands += [os.path.join(envs, n, "bin", "python") for n in sorted(os.listdir(envs))]
    for py in cands:
        if not (py and os.path.exists(py)):
            continue
        probe = subprocess.run([py, "-c", "import torch, onnx"], capture_output=True)
        if probe.returncode == 0:
            return py
    raise RuntimeError(
        "no interpreter with both torch and onnx found for the ONNX export; "
        "set FLOWC_ONNX_PYTHON=/path/to/python")


def mb_python() -> str:
    """Interpreter that can run modelblaster's torch-dependent stages.

    Order: $FLOWC_MB_PYTHON, this interpreter if it already has torch,
    then a conda env named by $FLOWC_MB_ENV (default "zephyr" — the env
    modelblaster's own install_conda.sh creates).
    """
    env = os.environ.get("FLOWC_MB_PYTHON")
    if env:
        return env
    try:
        import torch  # noqa: F401
        return sys.executable
    except ImportError:
        pass
    name = os.environ.get("FLOWC_MB_ENV", "zephyr")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        base = os.path.dirname(os.path.dirname(conda))
        cand = os.path.join(base, "envs", name, "bin", "python")
        if os.path.exists(cand):
            return cand
    raise RuntimeError(
        "no interpreter with torch found for modelblaster's extract stage; "
        "set FLOWC_MB_PYTHON=/path/to/python (modelblaster's conda env)")


# --------------------------------------------------------------------------
# The one upstream patch Flow C needs.
#
# ingest_xpurt_schedule._resolve_target() understands exactly two machine
# slots, CPU_P and CPU_E.  A three-lane board (HTA + DSP + CPU) schedules
# onto CPU_X as well.  Upstream this is a six-line change — replace the
# if/elif with a dict lookup — and until it lands we install the same
# behaviour here rather than forking the module.
# --------------------------------------------------------------------------
def install_slot_map(slot_to_kind: dict[str, str]) -> None:
    ing = ingest_xpurt_schedule()
    original = getattr(ing, "_flowc_original_resolve_target", None) or ing._resolve_target
    ing._flowc_original_resolve_target = original

    def _resolve(core_label, cpu_p_kind, cpu_e_kind, reg):
        label, _, idx_str = core_label.partition("#")
        kind = slot_to_kind.get(label.upper())
        if kind is None:
            return original(core_label, cpu_p_kind, cpu_e_kind, reg)
        cores = reg.by_kind.get(kind, ())
        if not cores:
            raise ValueError(
                f"registry {reg.system!r} has no cores of kind {kind!r} "
                f"(needed by machine slot {core_label})")
        idx = int(idx_str) if idx_str else 0
        if idx >= len(cores):
            raise ValueError(
                f"{core_label} indexes core #{idx} but kind {kind!r} has "
                f"{len(cores)} core(s)")
        c = cores[idx]
        return c.name, c.kind, (c.harts[0] if c.harts else -1)

    ing._resolve_target = _resolve
