#!/usr/bin/env python3
"""data/ros_arms.json: every ROS 2 arm, as the board script launches it and as its runs recorded it.

An arm name means whatever `scripts/ros_traced_matrix.sh` does with it: which traced binary, which
`taskset`, which node flags per process. That knowledge lives in a bash `case` statement and a set
of `if` branches, which a person can read and a program cannot. This writes it down as data, from
two sources and nothing else:

  * the script itself, run with `ssh` and `sleep` replaced by stubs that record what would have been
    sent to the board. Every process line in the table below is therefore the literal command the
    script would launch -- not a re-typing of it -- with the camera rate, the replicate and the
    kernels sha left as `{hz}`, `{rep}` and `{kernels_sha}` placeholders. No host is contacted:
    the host is set to an unresolvable name as well, so a stub that failed to shadow `ssh` fails
    to connect instead of reaching a board.
  * the run directories under `results/codesign_feedback/ros_traced/`: which configurations were
    measured, at which camera rates, how many replicates, and the per-process `manifest.json` each
    run wrote. The knobs a configuration was run with (QOS, CTRL_HZ, HOGS, BINSUF, NAVPOOL, SUFFIX)
    are recovered from the tag and the manifest by `build_implementation_bundles.ros_command`, the
    same derivation the generated `artifact/implementations/ros_deployments/` pages use.

Each configuration's expanded processes are then compared field by field with the manifest of its
most recent run (nodes, executor, QoS depth, control mode and timer, YOLO and nav pools, cameras,
frame alternation, and the affinity mask the process reported). Disagreements are recorded in the
entry's `manifest_mismatch` list rather than hidden: an old run can predate a script change.

    scripts/gen_ros_arms.py            # write data/ros_arms.json
    scripts/gen_ros_arms.py --check    # exit 1 if the file differs from a fresh derivation

`scripts/ros_baseline.py` reads this file; `tests/test_ros_baseline.py` runs `--check`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import build_implementation_bundles as BIB   # noqa: E402

OUT = os.path.join(REPO, "data", "ros_arms.json")
MATRIX = "scripts/ros_traced_matrix.sh"
NODE_SRC = "board/k1_ros_mb/ros_mb_chain_traced.cpp"
TRACED = os.path.join(REPO, "results", "codesign_feedback", "ros_traced")

HZ_SENTINEL = "424242"          # substituted for the camera rate, then replaced by {hz}
REP_SENTINEL = "__REP__"        # the replicate argument, replaced by {rep}
NO_VALUE_FLAGS = {"--alternate"}


def binary_defaults() -> dict:
    """The node binary's own defaults, read from its declarations in ros_mb_chain_traced.cpp.

    A flag the launcher does not pass takes this value, so the effective configuration of a process
    is these defaults overlaid with its flags. Read from the source so that a changed default in the
    program changes the table.
    """
    src = open(os.path.join(REPO, NODE_SRC)).read()
    body = src[src.index("int main("):src.index("for (int i = 1; i < argc; i++)")]
    want = {"rate": "rate", "ctrl_hz": "ctrl_hz", "secs": "seconds", "exec": "executor",
            "nodes": "nodes", "pool_n": "yolo_pool", "pool_harts": "pool_harts", "pin_main": "pin_main",
            "nav_pool_n": "nav_pool", "ctl_pool_n": "ctrl_pool", "nav_pool_harts": "nav_harts",
            "ctl_pool_harts": "ctrl_harts", "ctrl_mode": "ctrl_mode", "extra": "extra",
            "qos_depth": "qos_depth", "cameras": "cameras", "alternate": "alternate"}
    out = {}
    for var, key in want.items():
        m = re.search(rf'\b{var}\s*=\s*("[^"]*"|[^,;]+)', body)
        if m:
            v = m.group(1).strip()
            if v.startswith('"'):
                out[key] = v.strip('"')
            elif v in ("true", "false"):
                out[key] = v == "true"
            else:
                try:
                    out[key] = int(v) if re.fullmatch(r"-?\d+", v) else float(v)
                except ValueError:
                    out[key] = v
        else:
            out[key] = ""   # declared without an initialiser: an empty std::string
    return out


def _env_of(command: str) -> tuple[dict, str]:
    """(env assignments, base arm) from a `KEY=VAL ... scripts/ros_traced_matrix.sh <arm>` line."""
    toks = shlex.split(command)
    env = dict(t.split("=", 1) for t in toks if "=" in t and not t.startswith("-"))
    env.pop("RATES", None)
    return env, toks[-1]


def dry_run(arm: str, env: dict) -> str:
    """The run step ros_traced_matrix.sh would send to the board for one rate, captured, not sent."""
    with tempfile.TemporaryDirectory() as td:
        stub = os.path.join(td, "bin")
        os.makedirs(stub)
        log = os.path.join(td, "sent.log")
        with open(os.path.join(stub, "ssh"), "w") as f:
            f.write('#!/bin/sh\nprintf "%s\\0" "$2" >> "$DRYRUN_LOG"\n')
        with open(os.path.join(stub, "sleep"), "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(os.path.join(stub, "ssh"), 0o755)
        os.chmod(os.path.join(stub, "sleep"), 0o755)
        e = {k: v for k, v in os.environ.items() if k not in ("QOS", "CTRL_HZ", "HOGS", "SUFFIX", "BINSUF",
                                                             "NAVPOOL", "NAVHARTS", "SECS")}
        e.update(env)
        e.update({"PATH": stub + os.pathsep + os.environ.get("PATH", ""), "DRYRUN_LOG": log,
                  "RATES": HZ_SENTINEL, "MODELBLASTER_K1_HOST": "dryrun.invalid", "TMPDIR": td})
        subprocess.run(["bash", os.path.join(REPO, MATRIX), arm, REP_SENTINEL], env=e, cwd=REPO,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        sent = open(log).read().split("\0") if os.path.exists(log) else []
    run = [s for s in sent if "--tag" in s]
    if not run:
        raise SystemExit(f"{MATRIX} {arm}: no run step captured (unknown arm?)")
    return run[0]


def _placeholders(s: str) -> str:
    s = re.sub(r"--kernels-sha '[^']*'", "--kernels-sha '{kernels_sha}'", s)
    return s.replace(HZ_SENTINEL, "{hz}").replace(REP_SENTINEL, "{rep}")


def parse_processes(run_step: str) -> list[dict]:
    """One dict per launched process: its name (the --out-dir leaf), taskset, binary and flags."""
    procs = []
    for line in run_step.replace(";", "\n").split("\n"):
        for m in re.finditer(r"(?:taskset -c (\S+)\s+)?\./(ros_mb_chain_traced\w*)\s+(.*?)(?=\s+>\s*out/|\s+2>&1|$)",
                             line):
            toks = shlex.split(m.group(3))
            flags, i = {}, 0
            while i < len(toks):
                t = toks[i]
                if t in NO_VALUE_FLAGS:
                    flags[t] = True; i += 1
                elif t.startswith("--") and i + 1 < len(toks):
                    flags[t] = toks[i + 1]; i += 2
                else:
                    i += 1
            od = flags.get("--out-dir", "")
            leaf = od.rstrip("/").split("/")[-1]
            name = "main" if re.fullmatch(r"\d+_.+_r.+", leaf) else leaf
            procs.append({"name": name, "taskset": m.group(1), "binary": m.group(2),
                          "flags": {k: v for k, v in flags.items()
                                    if k not in ("--rate", "--tag", "--out-dir", "--t0", "--kernels-sha",
                                                 "--seconds")},
                          "command": _placeholders(m.group(0).strip())})
    return procs


def _cpus_to_mask(spec: str | None) -> str:
    if not spec:
        return "0xff"
    bits = 0
    for part in spec.split(","):
        a, _, b = part.partition("-")
        for c in range(int(a), int(b or a) + 1):
            bits |= 1 << c
    return hex(bits)


FLAG_KEY = {"--nodes": "nodes", "--executor": "executor", "--ctrl-mode": "ctrl_mode", "--ctrl-hz": "ctrl_hz",
            "--qos-depth": "qos_depth", "--yolo-pool": "yolo_pool", "--pool-harts": "pool_harts",
            "--nav-pool": "nav_pool", "--nav-harts": "nav_harts", "--ctrl-pool": "ctrl_pool",
            "--ctrl-harts": "ctrl_harts", "--pin-main": "pin_main", "--extra": "extra",
            "--cameras": "cameras", "--alternate": "alternate"}


def effective(proc: dict, defaults: dict) -> dict:
    """The process's configuration: the binary's defaults overlaid with its flags, plus placement."""
    eff = {k: defaults.get(k) for k in FLAG_KEY.values()}
    for f, v in proc["flags"].items():
        k = FLAG_KEY.get(f)
        if k is None:
            continue
        d = defaults.get(k)
        if isinstance(d, bool):
            eff[k] = bool(v)
        elif isinstance(d, int) and not isinstance(d, bool):
            eff[k] = int(float(v))
        elif isinstance(d, float):
            eff[k] = float(v)
        else:
            eff[k] = v
    eff["taskset"] = proc["taskset"]
    eff["binary"] = proc["binary"]
    pm = eff.get("pin_main")
    eff["expected_affinity_mask"] = hex(1 << pm) if isinstance(pm, int) and pm >= 0 else _cpus_to_mask(proc["taskset"])
    return eff


def latest_manifest(cfg: str, rates) -> tuple[str | None, dict]:
    """The most recent run of `cfg` and its per-process manifest ({name: fields})."""
    best = (None, -1, {})
    for hz, n in rates.items():
        for rep in range(1, 20):
            tag = f"{hz}_{cfg}_r{rep}"
            p = os.path.join(TRACED, tag, "manifest.json")
            if not os.path.exists(p):
                continue
            try:
                j = json.load(open(p))
            except Exception:
                continue
            procs = j.get("processes") if isinstance(j.get("processes"), dict) else {"main": j}
            t = max((q.get("wall_start_epoch_ms") or 0) for q in procs.values())
            if t > best[1]:
                best = (tag, t, procs)
    return best[0], best[2]


def _same(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


# Fields a later version of the node added to its manifest. A run whose manifest lacks one cannot
# say whether it ran with today's non-default value, so the entry records that it is unconfirmed.
NEWER_FIELDS = ("nav_pool", "nav_harts", "alternate", "cameras")
DEFAULTS_CACHE: dict = {}


def compare(eff_by_proc: dict, manifest: dict) -> list[str]:
    """Where the script's expansion and the recorded manifest disagree, one line per field."""
    out = []
    for name, eff in eff_by_proc.items():
        m = manifest.get(name)
        if m is None:
            out.append(f"{name}: no manifest for this process in the latest run")
            continue
        for k in ("nodes", "executor", "qos_depth", "ctrl_hz", "yolo_pool", "pool_harts", "nav_pool",
                  "nav_harts", "cameras", "alternate", "pin_main", "extra"):
            if k not in m:
                if k in eff and not _same(eff[k], DEFAULTS_CACHE.get(k)) and k in NEWER_FIELDS:
                    out.append(f"{name}.{k}: script {eff[k]!r}; the run's manifest predates the field, so the "
                               f"run neither confirms nor contradicts it")
                continue
            a, b = eff.get(k), m.get(k)
            if k == "alternate":
                a, b = bool(a), bool(b)
            if not _same(a, b):
                out.append(f"{name}.{k}: script {a!r}, manifest {b!r}")
        if "control" in str(eff.get("nodes", "")).split(",") and m.get("ctrl_mode") \
                and m["ctrl_mode"] != eff.get("ctrl_mode"):
            out.append(f"{name}.ctrl_mode: script {eff.get('ctrl_mode')!r}, manifest {m['ctrl_mode']!r}")
        if m.get("affinity_mask") and m["affinity_mask"] != eff["expected_affinity_mask"]:
            out.append(f"{name}.affinity_mask: script {eff['expected_affinity_mask']}, "
                       f"manifest {m['affinity_mask']}")
    for name in manifest:
        if name not in eff_by_proc:
            out.append(f"{name}: in the manifest, not launched by the script")
    return out


def build() -> dict:
    defaults = binary_defaults()
    DEFAULTS_CACHE.update(defaults)
    arms = BIB.script_arms()
    runs = BIB.ros_runs(arms)
    out = {}
    names = sorted(set(runs) | set(arms))
    for cfg in names:
        g = runs.get(cfg)
        rates = {str(h): n for h, n in sorted((g or {}).get("rates", {}).items())}
        if g is not None:
            cmd = BIB.ros_command(cfg, g, arms)
            base, suffix = g["arm"], g["suffix"]
        else:
            base, suffix = cfg, ""
            cmd = f'RATES="{{hz}}" {MATRIX} {cfg}'
        e = {"base_arm": base, "suffix": suffix, "measured": bool(rates), "rates": rates,
             "script_note": arms.get(base, {}).get("note", "")}
        if cmd is None:
            e.update({"env": None, "matrix_command": None, "processes": None,
                      "why_no_command": "its runs recorded no goal and not every process wrote a manifest, "
                                        "so the knobs it ran with are not recoverable from the run"})
            tag, man = latest_manifest(cfg, (g or {}).get("rates", {}))
            e["latest_run"] = tag
            e["manifest_processes"] = sorted(man)
            out[cfg] = e
            continue
        env, _ = _env_of(cmd)
        step = dry_run(base, env)
        procs = parse_processes(step)
        effs = {p["name"]: effective(p, defaults) for p in procs}
        tag, man = latest_manifest(cfg, (g or {}).get("rates", {}))
        e.update({"env": env,
                  "matrix_command": re.sub(r'^RATES="[^"]*"', 'RATES="{hz}"', cmd),
                  "processes": [dict(p, effective=effs[p["name"]]) for p in procs],
                  "latest_run": tag,
                  "manifest_mismatch": (compare(effs, man) if man
                                        else (["measured, but no manifest on disk"] if rates else []))})
        out[cfg] = e
    return {"generated_by": "scripts/gen_ros_arms.py",
            "sources": [MATRIX, NODE_SRC, "results/codesign_feedback/ros_traced/*/manifest.json",
                        "scripts/build_implementation_bundles.py (script_arms, ros_runs, ros_command)"],
            "placeholders": {"{hz}": "camera rate (RATES)", "{rep}": "replicate (2nd argument)",
                             "{kernels_sha}": "git short sha + staged YOLO IR hash, set by the script"},
            "binary_defaults": defaults, "arms": out}


def load() -> dict:
    """The committed table; ros_baseline.py's entry point to it."""
    return json.load(open(OUT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if data/ros_arms.json is stale")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    fresh = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    if a.check:
        cur = open(a.out).read() if os.path.exists(a.out) else ""
        if cur != fresh:
            old = json.loads(cur) if cur else {"arms": {}}
            new = json.loads(fresh)
            diff = sorted(k for k in set(old.get("arms", {})) | set(new["arms"])
                          if old.get("arms", {}).get(k) != new["arms"].get(k))
            print(f"STALE {os.path.relpath(a.out, REPO)}: differs from {MATRIX} / the run manifests "
                  f"for {len(diff)} arm(s): {', '.join(diff[:12]) or '(header)'}; "
                  f"re-run scripts/gen_ros_arms.py")
            return 1
        print(f"OK {os.path.relpath(a.out, REPO)} matches {MATRIX} and the run manifests "
              f"({len(json.loads(fresh)['arms'])} arms)")
        return 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(fresh)
    n = json.loads(fresh)["arms"]
    mm = {k: v["manifest_mismatch"] for k, v in n.items() if v.get("manifest_mismatch")}
    print(f"wrote {os.path.relpath(a.out, REPO)}: {len(n)} arms, {len(mm)} with a script/manifest "
          f"disagreement recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
