"""Emit a board-runnable plan.json from a converged feedback-loop state.

Companion to ``scripts/board/run_loop_plan_on_qrb5165.sh``. Reads a
persisted :class:`LoopState` JSON (produced by
``xpu_rt.scheduling.feedback_loop.save_loop_state`` — the default location
is ``build/loops/<workload>__<target>.json``) and writes a
``loop_plan_board_v1`` JSON the bash runner consumes.

The emitted plan keeps one entry per chunk in the loop's
``current_chunks``, mapping each chunk's ``preferred_backend`` to a
concrete DLC path. DLCs are looked up via a small backend → variant
table that mirrors :data:`xpu_rt.targets.backends.qnn.on_board_runner.BACKEND_DLC_VARIANT`;
override with ``--dlc-overrides chunk_id=path,...``.

Important caveats — this is the host-side half of the round-trip:

* The script does not ssh anywhere. Copying the output to the board is
  the operator's job (see ``scripts/board/README_board_loop.md``).
* The script does not pre-build DLCs; it only assembles paths the board
  can resolve.
* The script does not feed measurements back into the loop. Use
  :func:`xpu_rt.mcp.tools.feedback_loop_tools.xpu_rt_apply_measurement`
  on the host once you have the board's measurement.json.

Usage:
    uv run python scripts/board/emit_loop_plan.py \\
        --loop-state build/loops/yolov8n__qrb5165.json \\
        --output     build/loops/yolov8n__qrb5165__plan.json \\
        [--model-dir /root/models/yolov8n] \\
        [--input-list /root/models/yolov8n/input_list.txt] \\
        [--iters 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "loop_plan_board_v1"

# Mirrors xpu_rt.targets.backends.qnn.on_board_runner.BACKEND_DLC_VARIANT.
BACKEND_DLC_VARIANT: dict[str, str] = {
    "CPU": "",
    "GPU": "_fp16",
    "DSP": "_quantized",
    "HTP": "_quantized",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit a board-runnable plan.json from a converged LoopState.",
    )
    p.add_argument(
        "--loop-state",
        required=True,
        type=Path,
        help="Path to a persisted LoopState JSON "
        "(e.g. build/loops/yolov8n__qrb5165.json).",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the board plan JSON (loop_plan_board_v1).",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Remote directory on the board holding the DLCs. If omitted, "
        "defaults to /root/models/<workload_id>.",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Base model name (without _fp16/_quantized variant). Defaults "
        "to the LoopState's workload_id.",
    )
    p.add_argument(
        "--input-list",
        type=str,
        default=None,
        help="Default input_list.txt path on the board. Optional — the "
        "bash runner has its own --input-list flag too.",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Iterations per partition (default: 10).",
    )
    p.add_argument(
        "--dlc-overrides",
        type=str,
        default=None,
        help="Comma-separated chunk_id=dlc_path overrides. Useful when a "
        "chunk needs a custom DLC (e.g. partial graph).",
    )
    return p.parse_args()


def parse_dlc_overrides(spec: str | None) -> dict[str, str]:
    """Parse ``chunk_id=path,chunk_id2=path2`` into a mapping."""
    if not spec:
        return {}
    out: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise SystemExit(f"--dlc-overrides entry missing '=': {entry!r}")
        key, _, val = entry.partition("=")
        out[key.strip()] = val.strip()
    return out


def resolve_dlc_path(
    model_dir: str,
    model_name: str,
    backend: str,
) -> str:
    """Return the conventional DLC path for ``(model_name, backend)``."""
    variant = BACKEND_DLC_VARIANT.get(backend, "")
    return f"{model_dir.rstrip('/')}/{model_name}{variant}.dlc"


def build_plan(
    loop_state: dict[str, Any],
    *,
    model_dir: str | None,
    model_name: str | None,
    input_list: str | None,
    iters: int,
    dlc_overrides: dict[str, str],
) -> dict[str, Any]:
    """Assemble the board plan from a LoopState dict."""
    workload_id = str(loop_state.get("workload_id", ""))
    target_id = str(loop_state.get("target_id", "qrb5165"))
    if not workload_id:
        raise SystemExit("LoopState is missing workload_id")

    effective_model_name = model_name or workload_id
    effective_model_dir = model_dir or f"/root/models/{workload_id}"

    chunks = loop_state.get("current_chunks", [])
    if not chunks:
        raise SystemExit(
            "LoopState.current_chunks is empty — has the loop been stepped "
            "at least once?"
        )

    partitions = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        backend = str(chunk.get("preferred_backend", "DSP"))
        if backend == "UNKNOWN":
            # Fall back to the cheapest backend in durations_us_by_backend.
            durations = chunk.get("durations_us_by_backend", {}) or {}
            if durations:
                backend = min(durations.items(), key=lambda kv: kv[1])[0]
            else:
                backend = "DSP"
        dlc_path = dlc_overrides.get(chunk_id) or resolve_dlc_path(
            effective_model_dir, effective_model_name, backend
        )
        partitions.append(
            {
                "partition_id": chunk_id,
                "backend": backend,
                "dlc_path": dlc_path,
                "iters": iters,
                "n_ops": len(chunk.get("op_ids", [])),
                "input_list": input_list or "",
            }
        )

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "workload_id": workload_id,
        "target_id": target_id,
        "iters": iters,
        "loop_iteration": int(loop_state.get("iteration", 0)),
        "loop_status": str(loop_state.get("status", "init")),
        "loop_predicted_makespan_us": loop_state.get("current_predicted_makespan_us"),
        "partitions": partitions,
    }


def main() -> int:
    args = parse_args()
    if not args.loop_state.is_file():
        raise SystemExit(f"loop state file not found: {args.loop_state}")

    loop_state = json.loads(args.loop_state.read_text(encoding="utf-8"))
    overrides = parse_dlc_overrides(args.dlc_overrides)
    plan = build_plan(
        loop_state,
        model_dir=args.model_dir,
        model_name=args.model_name,
        input_list=args.input_list,
        iters=args.iters,
        dlc_overrides=overrides,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"[emit_loop_plan] wrote {args.output}")
    print(
        f"[emit_loop_plan] {len(plan['partitions'])} partitions, "
        f"workload={plan['workload_id']}, target={plan['target_id']}, "
        f"iters={plan['iters']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
