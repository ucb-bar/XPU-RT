"""Flight rows that are not measurements, and the one place every reader drops them.

WHY. When the simulator loses the GPU mid-batch it logs "PhysX Internal CUDA error. Simulation cannot
continue!" and keeps going: the physics is frozen, so every remaining episode is written to the campaign
CSV as a 1800-step timeout with no gates. Those rows look like a policy that never moves. They are not
flights, and they sit against whichever arm happened to be running.

results/codesign_feedback/flight_quarantine.csv lists each such batch by the fields that identify it in its
CSV, with the log and line that show the fault. A batch goes in only with that evidence; an arm that simply
times out a lot is behaviour, not a fault. `expected_n` is the number of rows the batch wrote: if a key ever
matches a different count (a re-flown batch appended under the same key), readers stop rather than guess
which rows to drop.

    from flight_quarantine import flight_rows
    for r in flight_rows(path): ...

    scripts/flight_quarantine.py            # list entries, confirm each still matches, report unlisted faulted logs
"""
import csv
import json
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "results", "codesign_feedback")
REGISTRY = os.path.join(RES, "flight_quarantine.csv")
FIELDS = ("ctrl_trace", "percep_latency_ms", "percep_hold_ms", "course", "prop_density", "person_h",
          "walk_speed", "cruise_speed", "moment_scale")
FAULT = re.compile(r"PhysX Internal CUDA error|Simulation cannot continue|GPU is out of memory|CUDA error, code")


def _norm(field, v):
    v = (v or "").strip()
    if field == "ctrl_trace":
        return os.path.basename(v)
    try:
        return repr(round(float(v), 6))
    except ValueError:
        return v


def _key(row):
    return tuple(_norm(f, row.get(f)) for f in FIELDS)


def _rel(path):
    return os.path.relpath(os.path.abspath(path), RES)


_ENTRIES = None


def entries():
    global _ENTRIES
    if _ENTRIES is None:
        _ENTRIES = {}
        if os.path.exists(REGISTRY):
            for e in csv.DictReader(open(REGISTRY)):
                _ENTRIES.setdefault(e["csv"], []).append(e)
    return _ENTRIES


def flight_rows(path):
    """csv.DictReader(open(path)) as a list, minus quarantined batches. Unlisted files pass through unchanged."""
    rows = list(csv.DictReader(open(path)))
    listed = entries().get(_rel(path))
    if not listed:
        return rows
    drop = {}
    for e in listed:
        drop[_key(e)] = e
    seen = {}
    for r in rows:
        k = _key(r)
        if k in drop:
            seen[k] = seen.get(k, 0) + 1
    for k, e in drop.items():
        if seen.get(k, 0) != int(e["expected_n"]):
            raise SystemExit(f"flight_quarantine: {e['csv']} {e['ctrl_trace']} cruise {e['cruise_speed']}: the quarantine "
                             f"key matches {seen.get(k, 0)} rows, the faulted batch wrote {e['expected_n']}. A batch was "
                             f"re-flown under the same key; decide which rows are the fault before reading this CSV.")
    return [r for r in rows if _key(r) not in drop]


EVID = os.path.join(RES, "quarantine_evidence")


def _evidence_line(rel, line):
    """The cited line, from the log itself or from the excerpt tracked beside it.

    The three logs these entries point at total 2.36 GB, one of them 2.3 GB, and they are untracked
    and in no archive -- so this check passed here and failed in a clean clone, which is where it
    matters. Storing 2.36 GB to preserve three lines is the wrong trade; the excerpt carries the
    line, its number, and the sha256 of the log it came from, so the evidence travels and the full
    log stays optional. When the log IS present it is read, and the excerpt is confirmed against it.
    """
    full = os.path.join(RES, rel)
    exc = os.path.join(EVID, rel.replace("/", "__") + ".json")
    recorded = None
    if os.path.exists(exc):
        try:
            recorded = (json.load(open(exc)).get("lines") or {}).get(str(line))
        except Exception:
            recorded = None
    if os.path.exists(full):
        try:
            live = open(full, errors="ignore").read().splitlines()[line - 1]
        except Exception:
            return recorded or ""
        if recorded is not None and recorded != live:
            print(f"     note: {rel}:{line} no longer matches the tracked excerpt")
        return live
    return recorded or ""


def main():
    ok = True
    for rel, es in sorted(entries().items()):
        p = os.path.join(RES, rel)
        rows = list(csv.DictReader(open(p)))
        for e in es:
            m = [r for r in rows if _key(r) == _key(e)]
            frozen = sum(r["outcome"] == "timeout" and str(r.get("steps")) == "1800" for r in m)
            log = os.path.join(RES, e["evidence_log"]); line = int(e["evidence_line"])
            ev = _evidence_line(e["evidence_log"], line)
            good = len(m) == int(e["expected_n"]) and bool(FAULT.search(ev))
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} {rel}  {e['ctrl_trace']} cruise {e['cruise_speed']}: {len(m)} rows "
                  f"({frozen} frozen 1800-step timeouts), fault at {e['evidence_log']}:{line}")
    listed_logs = {e["evidence_log"] for es in entries().values() for e in es}
    for f in sorted(glob.glob(os.path.join(RES, "campaign*", "*.log"))):
        rel = _rel(f)
        if rel in listed_logs or os.path.basename(f).startswith(("driver", "campaign")):
            continue
        try:
            txt = open(f, errors="ignore").read()
        except OSError:
            continue
        mm = FAULT.search(txt)
        if mm and "[ep00]" in txt[mm.start():]:
            ok = False
            print(f"UNLISTED faulted batch that went on to log episodes: {rel}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
