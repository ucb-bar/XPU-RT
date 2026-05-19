#!/usr/bin/env python3
"""Run a heterogeneous schedule on the QRB5165 board with real data flow.

Phase F orchestrator. Given a schedule.json (from
heterogeneous_loop.py) and a profiled_manifest.json (from
third_party/merlin/tools/profile_dispatch_matrix.py), this script:

  1. Pushes each per-(canonical, target) VMFB to the board.
  2. Emits a flat plan file (TAB-separated) consumed by
     merlin-dispatch-flow-runner.
  3. Runs the flow runner via SSH with the requested LD_LIBRARY_PATH for
     QNN backends.
  4. Optionally pulls back the trace.csv and the final output bytes.

Usage:
  python3 XPU-RT/scripts/run_on_board_flow.py \\
      --schedule /tmp/het_loop_yolov8/schedule.json \\
      --manifest /tmp/het_loop_yolov8/round_0/profiled_manifest.json \\
      --out-dir /tmp/het_loop_yolov8/onboard \\
      [--input-from <bytes>] [--output-to <bytes>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys

import os

DEFAULT_BOARD_BIN = "/root/merlin-dispatch-flow-runner"
DEFAULT_BOARD_VMFB_DIR = "/root/dispatch_flow"
DEFAULT_QNN_LIB_DIR = "/root/qairt/lib/target"
# Historical default: the DIMA_SLICE private key two levels above
# ``scripts/``. Kept as the *last-resort* fallback only; callers should
# pass ``--ssh-identity`` or set ``XPURT_SSH_KEY`` instead.
LEGACY_SSH_IDENTITY = pathlib.Path(__file__).resolve().parents[2] / "DIMA_SLICE"
DEFAULT_DOTENV = pathlib.Path("/scratch2/agustin/XPU-RT/.env")

_TARGET_FOR_MACHINE = {"CPU": "cpu", "GPU": "qnn_gpu", "HTA": "qnn_hta"}


def _parse_dotenv(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _resolve_ssh_host(cli_arg: str | None) -> str:
    """Pick an SSH host in priority order: CLI > env > dotenv > fallback."""
    if cli_arg:
        return cli_arg if "@" in cli_arg else f"root@{cli_arg}"
    for key in ("XPURT_BOARD_HOST", "XPURT_SSH_HOST"):
        val = os.environ.get(key)
        if val:
            return val if "@" in val else f"root@{val}"
    dotenv = _parse_dotenv(DEFAULT_DOTENV)
    for key in ("Q_DEV_BOAR", "Q_DEV_BOARD"):
        val = dotenv.get(key) or os.environ.get(key)
        if val:
            return val if "@" in val else f"root@{val}"
    return "root@10.44.120.201"


def _resolve_ssh_identity(cli_arg: pathlib.Path | None) -> pathlib.Path | None:
    """Pick an SSH key path in priority order: CLI > env > legacy."""
    if cli_arg is not None and pathlib.Path(cli_arg).is_file():
        return pathlib.Path(cli_arg)
    for key in ("XPURT_SSH_KEY", "Q_DEV_BOAR_KEY", "Q_DEV_BOARD_KEY"):
        val = os.environ.get(key)
        if val and pathlib.Path(val).is_file():
            return pathlib.Path(val)
    if LEGACY_SSH_IDENTITY.is_file():
        return LEGACY_SSH_IDENTITY
    return None


def _ssh_base(ssh_host: str, ssh_identity: pathlib.Path | None) -> list[str]:
    cmd = ["ssh"]
    if ssh_identity:
        cmd.extend(["-i", str(ssh_identity)])
    cmd.append(ssh_host)
    return cmd


def _scp_base(ssh_identity: pathlib.Path | None) -> list[str]:
    cmd = ["scp", "-q"]
    if ssh_identity:
        cmd.extend(["-i", str(ssh_identity)])
    return cmd


def _scp(ssh_host: str, ssh_identity: pathlib.Path | None,
         local: pathlib.Path, remote: str) -> None:
    subprocess.run([*_scp_base(ssh_identity), str(local), f"{ssh_host}:{remote}"],
                   check=True)


def _scp_back(ssh_host: str, ssh_identity: pathlib.Path | None,
              remote: str, local: pathlib.Path, recursive: bool = False) -> None:
    cmd = _scp_base(ssh_identity)
    if recursive:
        cmd.append("-r")
    cmd.extend([f"{ssh_host}:{remote}", str(local)])
    subprocess.run(cmd, check=True)


def _schedule_ops(schedule: dict) -> list[dict]:
    if "ops" in schedule:
        return sorted(schedule["ops"], key=lambda o: o["start_us"])
    dispatches = []
    for name, row in schedule.get("dispatches", {}).items():
        machine = row.get("machine") or row.get("hardware_target")
        dispatches.append({
            "name": name,
            "machine": machine,
            "start_us": row.get("start_us", 0.0),
        })
    return sorted(dispatches, key=lambda o: o["start_us"])


def _default_binding_sources(
    op: dict,
    sizes: list[int],
    preds_list: list[str],
    remote_constant_arena: str,
) -> list[str]:
    if not sizes:
        return []
    sources: list[str] = []
    for i, _ in enumerate(sizes):
        if i == 0:
            sources.append(f"pred:{preds_list[0]}:0" if preds_list else "input")
        elif i == len(sizes) - 1:
            sources.append("zero")
        elif remote_constant_arena:
            sources.append(f"file:{remote_constant_arena}")
        else:
            sources.append("auto")
    return sources


def emit_plan(
    schedule: dict,
    manifest: dict,
    vmfb_dir_remote: str,
    remote_constant_arena: str = "",
    binding_sources: dict | None = None,
) -> str:
    """Build the TAB-separated flat plan consumed by the on-board runner.

    Format per line:
      <op_name>\\t<machine>\\t<func>\\t<vmfb_remote>\\t<sizes_csv>\\t<preds_csv>\\t<sources>
    """
    ops = _schedule_ops(schedule)
    dispatch_graph = manifest.get("dispatch_graph", {})
    lines = ["# op_name\tmachine\tfunc\tvmfb\tsizes\tpreds\tsources"]
    prev_name: str | None = None
    for op in ops:
        machine = op["machine"]
        target = _TARGET_FOR_MACHINE[machine]
        cell = manifest["dispatches"][op["name"]][target]
        func = "module." + cell["func"]
        raw_sizes = [int(s) for s in cell["binding_byte_sizes"]]
        sizes = ",".join(str(s) for s in raw_sizes)
        vmfb = f"{vmfb_dir_remote}/{target}__{op['name']}.vmfb"
        preds_list = list(dispatch_graph.get(op["name"], {}).get("dependencies", []))
        if not preds_list and prev_name:
            # Legacy fallback for manifests that predate explicit dependency
            # export. Keep the older chain behavior, but only as a fallback.
            preds_list = [prev_name]
        preds = ",".join(preds_list)
        sources_list = None
        if binding_sources:
            sources_list = binding_sources.get(op["name"])
        if sources_list is None:
            sources_list = _default_binding_sources(
                op, raw_sizes, preds_list, remote_constant_arena)
        sources = ";".join(str(s) for s in sources_list)
        lines.append("\t".join(
            [op["name"], machine, func, vmfb, sizes, preds, sources]))
        prev_name = op["name"]
    return "\n".join(lines) + "\n"


def push_vmfbs(ssh_host: str, ssh_identity: pathlib.Path | None,
               schedule: dict, manifest: dict,
               vmfb_dir_remote: str) -> None:
    subprocess.run([*_ssh_base(ssh_host, ssh_identity),
                    f"mkdir -p {shlex.quote(vmfb_dir_remote)}"],
                   check=True)
    # The vmfb_remote field already references a board-side path produced
    # by tools/profile_dispatch_matrix.py. We assume those files are still
    # on the board (the profiler pushed them); if the user specified a
    # different vmfb_dir_remote, we copy the on-board VMFBs to the target
    # location with the canonical name <target>__<canonical>.vmfb.
    for op in _schedule_ops(schedule):
        target = _TARGET_FOR_MACHINE[op["machine"]]
        cell = manifest["dispatches"][op["name"]][target]
        src_remote = cell.get("vmfb_remote", "")
        dst_remote = f"{vmfb_dir_remote}/{target}__{op['name']}.vmfb"
        local_vmfb = cell.get("vmfb", "")
        if local_vmfb:
            _scp(ssh_host, ssh_identity, pathlib.Path(local_vmfb), dst_remote)
            continue
        if not src_remote:
            continue
        if src_remote == dst_remote:
            continue
        # Copy on-board (avoids the host->board scp roundtrip when the
        # profiler already pushed under a different path).
        subprocess.run([*_ssh_base(ssh_host, ssh_identity),
                        f"cp {shlex.quote(src_remote)} {shlex.quote(dst_remote)}"],
                       check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule", type=pathlib.Path, required=True)
    p.add_argument("--manifest", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--ssh-host", default=None,
                   help="user@host of the board. Default: --board-ip / env "
                        "XPURT_BOARD_HOST / dotenv Q_DEV_BOAR / "
                        "root@10.44.120.201.")
    p.add_argument("--board-ip", default=None,
                   help="Convenience override: bare IP or user@host. "
                        "Equivalent to --ssh-host but ``root@`` is prepended "
                        "for bare-IP inputs.")
    p.add_argument("--ssh-identity", type=pathlib.Path, default=None,
                   help="Private key path. Default: env XPURT_SSH_KEY / "
                        "legacy DIMA_SLICE alongside the repo.")
    p.add_argument("--board-bin", default=DEFAULT_BOARD_BIN)
    p.add_argument("--board-vmfb-dir", default=DEFAULT_BOARD_VMFB_DIR)
    p.add_argument("--qnn-lib-dir", default=DEFAULT_QNN_LIB_DIR)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--input-from", type=pathlib.Path, default=None,
                   help="Local file with raw bytes for the chain's first input")
    p.add_argument("--constant-arena", type=pathlib.Path, default=None,
                   help="Local constant arena file to source read-only "
                        "constant bindings from")
    p.add_argument("--binding-sources-json", type=pathlib.Path, default=None,
                   help="Optional JSON mapping dispatch name to per-binding "
                        "source specs. Specs: input, zero, file:<remote>, "
                        "pred:<dispatch>:<output-index>, auto")
    p.add_argument("--output-to", type=pathlib.Path, default=None,
                   help="Local file to receive the final output bytes")
    p.add_argument("--capture-dispatch-io-dir", type=pathlib.Path, default=None,
                   help="Local dir to receive per-dispatch input/output capture "
                        "from the board runner")
    p.add_argument("--capture-dispatches", default="",
                   help="Comma-separated dispatch names to capture when "
                        "--capture-dispatch-io-dir is set. Empty captures all.")
    p.add_argument("--strict-binding-sources", action="store_true",
                   help="Fail on any binding without an explicit source "
                        "instead of falling back to zero-filled buffers")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    args.ssh_host = _resolve_ssh_host(args.ssh_host or args.board_ip)
    args.ssh_identity = _resolve_ssh_identity(args.ssh_identity)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    schedule = json.loads(args.schedule.read_text())
    manifest = json.loads(args.manifest.read_text())

    # 1. Ensure VMFBs are on board (Phase C profiler typically already did this).
    push_vmfbs(args.ssh_host, args.ssh_identity, schedule, manifest,
               args.board_vmfb_dir)

    # 2. Push source side inputs that can be referenced by the flat plan.
    remote_constant_arena = ""
    if args.constant_arena:
        remote_constant_arena = "/root/dispatch_flow_constant_arena.bin"
        _scp(args.ssh_host, args.ssh_identity, args.constant_arena,
             remote_constant_arena)
    binding_sources = None
    if args.binding_sources_json:
        binding_sources = json.loads(args.binding_sources_json.read_text())

    # 3. Emit and push the flat plan file.
    plan = emit_plan(
        schedule,
        manifest,
        args.board_vmfb_dir,
        remote_constant_arena=remote_constant_arena,
        binding_sources=binding_sources,
    )
    local_plan = args.out_dir / "plan.tsv"
    local_plan.write_text(plan)
    remote_plan = "/root/dispatch_flow_plan.tsv"
    _scp(args.ssh_host, args.ssh_identity, local_plan, remote_plan)

    # 4. Push input bytes if provided.
    remote_input = ""
    if args.input_from:
        remote_input = "/root/dispatch_flow_input.bin"
        _scp(args.ssh_host, args.ssh_identity, args.input_from, remote_input)

    remote_output = "/root/dispatch_flow_output.bin"
    remote_trace = "/root/dispatch_flow_trace.csv"
    remote_capture = "/root/dispatch_flow_capture"
    if args.capture_dispatch_io_dir:
        subprocess.run([*_ssh_base(args.ssh_host, args.ssh_identity),
                        f"rm -rf {shlex.quote(remote_capture)} && "
                        f"mkdir -p {shlex.quote(remote_capture)}"],
                       check=True)

    # 5. Run on board.
    cmd_parts = [
        f"LD_LIBRARY_PATH={args.qnn_lib_dir}",
        args.board_bin,
        f"--plan={remote_plan}",
        f"--iterations={args.iterations}",
        f"--warmup={args.warmup}",
        f"--trace-csv={remote_trace}",
    ]
    if remote_input:
        cmd_parts.append(f"--input-from={remote_input}")
    if args.output_to:
        cmd_parts.append(f"--output-to={remote_output}")
    if args.capture_dispatch_io_dir:
        cmd_parts.append(f"--capture-dispatch-io-dir={remote_capture}")
    if args.capture_dispatches:
        cmd_parts.append(f"--capture-dispatches={args.capture_dispatches}")
    if args.strict_binding_sources:
        cmd_parts.append("--strict-binding-sources")
    if args.verbose:
        cmd_parts.append("--verbose")

    remote_cmd = " ".join(cmd_parts)
    print(f"[on-board] {remote_cmd}")
    rc = subprocess.run([*_ssh_base(args.ssh_host, args.ssh_identity),
                         remote_cmd]).returncode
    if rc != 0:
        print(f"[on-board] runner returned rc={rc}", file=sys.stderr)

    # 6. Pull back trace + output.
    try:
        _scp_back(args.ssh_host, args.ssh_identity,
                  remote_trace, args.out_dir / "trace.csv")
        print(f"trace -> {args.out_dir / 'trace.csv'}")
    except subprocess.CalledProcessError:
        print("[on-board] trace.csv not produced")
    if args.output_to:
        try:
            _scp_back(args.ssh_host, args.ssh_identity,
                      remote_output, args.output_to)
            print(f"output bytes -> {args.output_to}")
        except subprocess.CalledProcessError:
            print(f"[on-board] failed to fetch output -> {args.output_to}")
    if args.capture_dispatch_io_dir:
        try:
            args.capture_dispatch_io_dir.mkdir(parents=True, exist_ok=True)
            _scp_back(args.ssh_host, args.ssh_identity, remote_capture + "/.",
                      args.capture_dispatch_io_dir, recursive=True)
            print(f"dispatch capture -> {args.capture_dispatch_io_dir}")
        except subprocess.CalledProcessError:
            print("[on-board] failed to fetch dispatch capture")
    return rc


if __name__ == "__main__":
    sys.exit(main())
