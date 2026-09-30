#!/usr/bin/env python3
"""The ledger of executed schedule tables: which file each K1 run manifest names, the file's hash at run time,
and the content hash of its dispatches.

A schedule JSON carries the dispatch table and a metadata block. The board runs record the file's sha256 prefix
(`schedule_sha256` in xpurt_long/manifest_*.json). The metadata block may be annotated after a run (the executed
tables were given `solver`, `solver_status` and `solved_from` fields in commit a75e9c21), which changes the file's
hash but not the table the board executed. The ledger therefore records, per manifest:

  file                    the schedule path relative to the repo
  schedule_sha256_at_run  the manifest's 16-hex prefix
  dispatch_sha256         sha256 of the canonical JSON of the `dispatches` block of the file the board ran
  annotation              the commit whose metadata edit changed the file's hash, and the fields it added (or null)

`check()` recomputes the dispatch hash of the file on disk and compares it with the ledger; the verifier calls it
for every Gantt row. `--write` rebuilds the ledger from the manifests and git history.

    scripts/executed_tables.py --write        # rebuild results/codesign_feedback/executed_tables.json
    scripts/executed_tables.py                # check every entry against the files on disk
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
LEDGER = os.path.join(RES, "executed_tables.json")


def dispatch_sha256(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc.get("dispatches", {}), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def relpath(p: str) -> str:
    """repo-relative form of a path a manifest recorded, which is absolute on the machine that drove the board run.

    A path is only taken as given when it resolves INSIDE this repo: another checkout of the same repo may sit on the
    same filesystem, and resolving to it would read that tree's files instead of this one's."""
    q = p if os.path.isabs(p) else os.path.join(REPO, p)
    rel = os.path.relpath(os.path.realpath(q), os.path.realpath(REPO))
    if not rel.startswith(os.pardir):
        return rel
    i = p.find("schedules/")   # recorded from another checkout: keep the repo-relative tail
    return p[i:] if i >= 0 else p


def _git_versions(rel: str):
    out = subprocess.run(["git", "log", "--format=%H", "--", rel], cwd=REPO, capture_output=True, text=True).stdout.split()
    for h in out:
        blob = subprocess.run(["git", "show", f"{h}:{rel}"], cwd=REPO, capture_output=True)
        if blob.returncode == 0:
            yield h, blob.stdout


def _shipped_tables() -> set:
    """schedule files a shipped figure reads: the executed table each Gantt sidecar's manifest names."""
    out = set()
    for side in glob.glob(os.path.join(REPO, "schedules", "measured_gantt*_metrics.json")):
        try:
            man = json.load(open(side)).get("manifest") or ""
            man = man if os.path.isabs(man) else os.path.join(REPO, man)
            if os.path.exists(man):
                sched = json.load(open(man)).get("schedule") or ""
                if sched:
                    out.add(relpath(sched))
        except (OSError, ValueError):
            continue
    return out


def build() -> dict:
    entries = {}
    for manp in sorted(glob.glob(os.path.join(RES, "xpurt_long", "manifest_*_other_run*.json"))):
        man = json.load(open(manp)); sched = man.get("schedule") or ""
        if not sched:
            continue
        rel = relpath(sched); path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            continue
        rec = man.get("schedule_sha256") or ""
        cur = file_sha256(path)
        e = {"file": rel, "schedule_sha256_at_run": rec, "dispatch_sha256": None, "annotation": None,
             "file_sha256_now": cur, "manifests": []}
        key = os.path.basename(manp)[len("manifest_"):-len(".json")]
        if rec and cur.startswith(rec):
            e["dispatch_sha256"] = dispatch_sha256(json.load(open(path)))
        elif rec:
            for h, blob in _git_versions(rel):   # find the version the board ran and what changed since
                if hashlib.sha256(blob).hexdigest().startswith(rec):
                    ran = json.loads(blob); now = json.load(open(path))
                    e["dispatch_sha256"] = dispatch_sha256(ran)
                    same = ran.get("dispatches") == now.get("dispatches")
                    added = sorted(set(now.get("metadata", {})) - set(ran.get("metadata", {})))
                    changed = sorted(k for k in ran.get("metadata", {}) if k in now.get("metadata", {}) and ran["metadata"][k] != now["metadata"][k])
                    e["annotation"] = {"dispatches_identical": same, "metadata_added": added, "metadata_changed": changed, "version_at_run": h}
                    break
        e["shipped"] = rel in _shipped_tables()
        if rel in entries:
            entries[rel]["manifests"].append(key)
            if entries[rel]["schedule_sha256_at_run"] != rec:
                entries[rel].setdefault("other_hashes_at_run", []).append({key: rec})
        else:
            e["manifests"].append(key); entries[rel] = e
    return {"note": __doc__.strip().split("\n")[0], "entries": entries}


def check(rel: str, ledger: dict | None = None):
    """returns (ok, message) for one schedule file against the ledger and the file on disk."""
    ledger = ledger or json.load(open(LEDGER))
    e = ledger["entries"].get(relpath(rel))
    if not e:
        return None, f"{rel}: not in the executed-tables ledger"
    path = os.path.join(REPO, e["file"])
    if not os.path.exists(path):
        return False, f"{e['file']}: missing on disk"
    now = dispatch_sha256(json.load(open(path)))
    if e["dispatch_sha256"] is None:
        return False, f"{e['file']}: the version the board ran ({e['schedule_sha256_at_run']}) is not in git history"
    if now != e["dispatch_sha256"]:
        return False, f"{e['file']}: dispatch table differs from the one the board executed"
    ann = e.get("annotation")
    note = "file hash equals the manifest's" if not ann else f"metadata annotated after the run ({', '.join(ann['metadata_added'])}), dispatches identical"
    return True, f"{e['file']}: dispatches equal the executed table ({now[:16]}); {note}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    if a.write:
        led = build(); json.dump(led, open(LEDGER, "w"), indent=1); print(f"wrote {LEDGER}: {len(led['entries'])} tables")
        for rel, e in led["entries"].items():
            print(f"  {rel}: at run {e['schedule_sha256_at_run']} now {e['file_sha256_now'][:16]} dispatch {str(e['dispatch_sha256'])[:16]} annotation={'none' if not e['annotation'] else e['annotation']['metadata_added']} runs={len(e['manifests'])}")
        return 0
    led = json.load(open(LEDGER)); bad = 0
    for rel, e in led["entries"].items():
        ok, msg = check(rel, led)
        if ok:
            print("PASS " + msg)
        elif e.get("shipped"):
            print("FAIL " + msg); bad += 1
        else:   # a board run from the study's history whose table was regenerated afterwards; no figure reads it
            print(f"NOTE {msg}; no shipped figure reads this table ({len(e.get('manifests', []))} run(s): {', '.join(e.get('manifests', [])[:3])})")
    return bad


if __name__ == "__main__":
    sys.exit(main())
