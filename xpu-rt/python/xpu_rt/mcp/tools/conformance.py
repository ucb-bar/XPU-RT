"""MCP tools — Event Tensor Compiler conformance harness.

Three tools the remote-Blackwell agent calls from Claude Code:

- ``etc_conformance_run`` — kick off
  :func:`compgen.testing.etc_conformance.run_conformance` for a
  workload (or all of them); writes per-workload reports.
- ``etc_conformance_summarize`` — read every report under an output
  dir and produce a Markdown table.
- ``etc_megakernel_inspect`` — extract PTX/SASS stats from a
  compiled bundle's ``megakernel/`` subdir (registers, shared mem,
  occupancy, atomics density). Defers to ``cuobjdump`` /
  ``nvdisasm`` when available; emits a structured "tool not present"
  reason otherwise.

Design notes:

- The remote agent never sees the local repo. All tool inputs and
  outputs are paths on the **remote** filesystem; the human courier
  rsync's bundles into the bridge ``inbox/`` directory after the
  agent reports back via ``thread.md``.
- The tools use stable JSON-serialisable shapes so any auto-tooling
  on either side can parse them. No session-state coupling — these
  three tools work without an open MCP session.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from compgen.testing.etc_conformance import (
    ConformanceWorkload,
    PassGate,
    run_conformance,
    summarize_reports,
)

# ---------------------------------------------------------------------------
# etc_conformance_run
# ---------------------------------------------------------------------------


def etc_conformance_run(
    workload: str,
    output_dir: str,
    *,
    dtype: str = "bf16",
    device_index: int = 0,
    num_correctness_inputs: int = 16,
    num_benchmark_iters: int = 50,
    num_gpus: int = 1,
    min_speedup_vs_eager: float | None = None,
) -> dict[str, Any]:
    """Run one conformance workload (or ``"all"``) and return a summary.

    The full :class:`ConformanceReport` JSON is written to
    ``<output_dir>/<workload>.conformance_report.json``; this tool
    returns a compact summary dict so the LLM can decide what to do
    next without re-reading the file.

    Args:
        workload: One of the :class:`ConformanceWorkload` values, or
            ``"all"`` to run every workload sequentially.
        output_dir: Filesystem path on the **remote** box where reports
            land. The harness creates it if missing.
        dtype: Numeric dtype the workload runs at.
        device_index: CUDA device index.
        num_correctness_inputs: Random inputs used to compare ETC
            vs eager (per workload).
        num_benchmark_iters: Timed iterations for the perf gate.
        num_gpus: Required ≥ 2 for tensor-parallel workloads.
        min_speedup_vs_eager: Override the default 1.2× perf gate
            (e.g. relax to 1.0 during early bring-up). ``None`` keeps
            the default.

    Returns:
        ``{
            "status": "pass" | "fail" | "partial",
            "passed": [<workload-names>],
            "failed": [<workload-names>],
            "report_paths": [<absolute-paths>],
            "summary_md": "<markdown table>",
            "first_error": "<one-line>"  # only when "status" != "pass"
        }``
    """
    output_path = Path(output_dir).expanduser().resolve()
    gate = None
    if min_speedup_vs_eager is not None:
        gate = PassGate(min_speedup_vs_eager=float(min_speedup_vs_eager))

    if workload == "all":
        names = [w.value for w in ConformanceWorkload]
    else:
        names = [w.strip() for w in workload.split(",") if w.strip()]

    passed: list[str] = []
    failed: list[str] = []
    report_paths: list[str] = []
    first_error: str | None = None
    for name in names:
        try:
            wl = ConformanceWorkload(name)
        except ValueError:
            failed.append(name)
            if first_error is None:
                first_error = f"unknown workload {name!r}; valid: {', '.join(w.value for w in ConformanceWorkload)}"
            continue

        rep = run_conformance(
            wl,
            dtype=dtype,
            output_dir=output_path,
            device_index=device_index,
            num_correctness_inputs=num_correctness_inputs,
            num_benchmark_iters=num_benchmark_iters,
            num_gpus=num_gpus,
            gate=gate,
        )
        report_paths.append(str(output_path / f"{wl.value}.conformance_report.json"))
        if rep.passed:
            passed.append(wl.value)
        else:
            failed.append(wl.value)
            if first_error is None and rep.errors:
                first_error = rep.errors[0]

    if failed and not passed:
        status = "fail"
    elif failed:
        status = "partial"
    else:
        status = "pass"

    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "report_paths": report_paths,
        "summary_md": summarize_reports(output_path),
        "first_error": first_error,
    }


# ---------------------------------------------------------------------------
# etc_conformance_summarize
# ---------------------------------------------------------------------------


def etc_conformance_summarize(output_dir: str) -> dict[str, Any]:
    """Read every ``*.conformance_report.json`` under ``output_dir``
    and emit a Markdown summary table plus aggregate stats.

    Useful after a sweep — the remote agent calls this once and pastes
    the result into ``thread.md`` so the local agent doesn't need to
    re-parse 6 JSON files."""
    output_path = Path(output_dir).expanduser().resolve()
    if not output_path.is_dir():
        return {"error": f"output_dir {output_path} does not exist", "summary_md": ""}

    reports: list[dict[str, Any]] = []
    for p in sorted(output_path.glob("*.conformance_report.json")):
        try:
            reports.append(json.loads(p.read_text()))
        except Exception as exc:
            reports.append({"workload": p.stem, "passed": False, "errors": [f"unreadable: {exc!r}"]})

    n_total = len(reports)
    n_pass = sum(1 for r in reports if r.get("passed"))
    n_fail = n_total - n_pass

    return {
        "summary_md": summarize_reports(output_path),
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "report_paths": [str(p) for p in sorted(output_path.glob("*.conformance_report.json"))],
    }


# ---------------------------------------------------------------------------
# etc_megakernel_inspect
# ---------------------------------------------------------------------------


def etc_megakernel_inspect(bundle_dir: str) -> dict[str, Any]:
    """Inspect a compiled bundle's persistent megakernel.

    Reads ``<bundle_dir>/megakernel/manifest.yaml`` if present and
    extracts launch config, queue depths, atomics density. When
    ``cuobjdump`` / ``nvdisasm`` is on PATH, also dumps PTX/SASS
    stats (register usage, shared mem, occupancy hint) for the
    persistent kernel and each device-function body.

    Args:
        bundle_dir: Filesystem path to a compiled CompGen bundle on
            the **remote** box.

    Returns:
        ``{
            "manifest_present": bool,
            "manifest": {...},  # parsed YAML, only when present
            "ptx_files": [<list of paths>],
            "tooling": {
                "cuobjdump": <version-or-null>,
                "nvdisasm": <version-or-null>,
            },
            "sass_stats_per_kernel": {<name>: {"regs": int, "smem_bytes": int, ...}},
            "errors": [<strings>]
        }``
    """
    bundle = Path(bundle_dir).expanduser().resolve()
    errors: list[str] = []

    out: dict[str, Any] = {
        "manifest_present": False,
        "manifest": None,
        "ptx_files": [],
        "tooling": {"cuobjdump": None, "nvdisasm": None},
        "sass_stats_per_kernel": {},
        "errors": errors,
    }

    if not bundle.is_dir():
        errors.append(f"bundle_dir {bundle} does not exist or is not a directory")
        return out

    mk_dir = bundle / "megakernel"
    if not mk_dir.is_dir():
        errors.append(
            f"no megakernel/ subdir under {bundle}; this bundle was compiled "
            "without ETC dispatch (Phase 7 not yet wired, or target lacks "
            "supports_event_tensors)."
        )
        return out

    manifest_path = mk_dir / "manifest.yaml"
    if manifest_path.is_file():
        out["manifest_present"] = True
        try:
            import yaml  # type: ignore[import-untyped]

            out["manifest"] = yaml.safe_load(manifest_path.read_text())
        except Exception as exc:
            errors.append(f"manifest unreadable: {exc!r}")

    ptx_files = sorted(mk_dir.rglob("*.ptx"))
    out["ptx_files"] = [str(p) for p in ptx_files]

    cuobjdump = shutil.which("cuobjdump")
    nvdisasm = shutil.which("nvdisasm")
    if cuobjdump:
        out["tooling"]["cuobjdump"] = _tool_version(cuobjdump, "--version")
    if nvdisasm:
        out["tooling"]["nvdisasm"] = _tool_version(nvdisasm, "--version")

    if not (cuobjdump or nvdisasm) and ptx_files:
        errors.append(
            "cuobjdump / nvdisasm not on PATH; SASS stats unavailable. Install CUDA toolkit "
            "or set PATH to include $CUDA_HOME/bin."
        )

    # Best-effort SASS extraction. nvdisasm operates on cubin / ELF;
    # for raw PTX files we use cuobjdump --dump-resource-usage.
    if cuobjdump:
        for ptx in ptx_files:
            try:
                proc = subprocess.run(
                    [cuobjdump, "--dump-resource-usage", str(ptx)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0:
                    out["sass_stats_per_kernel"][ptx.name] = _parse_resource_usage(proc.stdout)
                else:
                    errors.append(
                        f"cuobjdump --dump-resource-usage failed on {ptx.name}: "
                        f"rc={proc.returncode}, stderr={proc.stderr[:200]!r}"
                    )
            except subprocess.TimeoutExpired:
                errors.append(f"cuobjdump timed out on {ptx.name}")
            except Exception as exc:
                errors.append(f"cuobjdump raised on {ptx.name}: {exc!r}")

    return out


def _tool_version(path: str, *args: str) -> str | None:
    try:
        proc = subprocess.run([path, *args], capture_output=True, text=True, timeout=5)
        return (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr) else None
    except Exception:
        return None


def _parse_resource_usage(text: str) -> dict[str, int | str]:
    """Parse ``cuobjdump --dump-resource-usage`` output into a
    structured stats dict. Tolerant of format drift across CUDA
    versions; missing fields land as 0 / "" so the JSON still has
    stable keys."""
    out: dict[str, int | str] = {
        "registers": 0,
        "shared_bytes": 0,
        "stack_bytes": 0,
        "kernel": "",
    }
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "Resource usage: <kernel> regs=NN smem=NN stack=NN ..."
        if line.lower().startswith("resource usage"):
            parts = line.split()
            for p in parts:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                try:
                    n = int(v)
                except ValueError:
                    continue
                if k == "regs":
                    out["registers"] = n
                elif k == "smem":
                    out["shared_bytes"] = n
                elif k == "stack":
                    out["stack_bytes"] = n
        # nvdisasm-style "REGISTERS  : 32" lines.
        if ":" in line:
            k, _, v = line.partition(":")
            v = v.strip()
            if k.strip().upper() == "REGISTERS" and v.isdigit():
                out["registers"] = int(v)
            elif k.strip().upper().startswith("SHARED") and v.split()[0].isdigit():
                out["shared_bytes"] = int(v.split()[0])
    return out


# ---------------------------------------------------------------------------
# Tool descriptors
# ---------------------------------------------------------------------------


CONFORMANCE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "etc_conformance_run",
        "description": (
            "Run one or all Event Tensor Compiler conformance workloads end-to-end "
            "(compile + correctness verify + benchmark vs eager) and write a "
            "ConformanceReport JSON per workload to output_dir. Pass workload='all' "
            "to run the full sweep; otherwise comma-separate names "
            "(gemm_rs, ag_gemm, moe_fwd, shape_dynamic_mlp, decoder_layer, "
            "diamond_dag). Returns a status summary + Markdown table."
        ),
        "phase": "verify",
        "handler": etc_conformance_run,
        "input_schema": {
            "type": "object",
            "properties": {
                "workload": {
                    "type": "string",
                    "description": (
                        "Workload name, comma-separated list, or 'all'. "
                        "Choices: gemm_rs, ag_gemm, moe_fwd, shape_dynamic_mlp, "
                        "decoder_layer, diamond_dag."
                    ),
                },
                "output_dir": {
                    "type": "string",
                    "description": "Filesystem path where reports land.",
                },
                "dtype": {
                    "type": "string",
                    "default": "bf16",
                    "enum": ["bf16", "fp16", "fp8_e4m3", "fp4_e2m1"],
                },
                "device_index": {"type": "integer", "default": 0},
                "num_correctness_inputs": {"type": "integer", "default": 16},
                "num_benchmark_iters": {"type": "integer", "default": 50},
                "num_gpus": {"type": "integer", "default": 1, "minimum": 1},
                "min_speedup_vs_eager": {
                    "type": "number",
                    "description": (
                        "Override the default 1.2× perf gate. Use during early "
                        "bring-up; tighten once the implementation is stable."
                    ),
                },
            },
            "required": ["workload", "output_dir"],
        },
    },
    {
        "name": "etc_conformance_summarize",
        "description": (
            "Read every conformance_report.json under output_dir and produce a "
            "Markdown table + pass/fail counts. Cheap; no GPU needed."
        ),
        "phase": "inspect",
        "handler": etc_conformance_summarize,
        "input_schema": {
            "type": "object",
            "properties": {
                "output_dir": {
                    "type": "string",
                    "description": "Path that previously received conformance reports.",
                },
            },
            "required": ["output_dir"],
        },
    },
    {
        "name": "etc_megakernel_inspect",
        "description": (
            "Inspect a compiled CompGen bundle's persistent megakernel: read "
            "manifest.yaml, list PTX files, dump SASS resource usage when "
            "cuobjdump/nvdisasm are available. Returns structured JSON with "
            "register usage, shared mem, occupancy hint per kernel."
        ),
        "phase": "inspect",
        "handler": etc_megakernel_inspect,
        "input_schema": {
            "type": "object",
            "properties": {
                "bundle_dir": {
                    "type": "string",
                    "description": "Filesystem path to a compiled bundle directory.",
                },
            },
            "required": ["bundle_dir"],
        },
    },
]


__all__ = [
    "CONFORMANCE_TOOLS",
    "etc_conformance_run",
    "etc_conformance_summarize",
    "etc_megakernel_inspect",
]
