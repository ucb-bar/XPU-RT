#!/usr/bin/env python3
"""Print the frozen cost model as a markdown table, for SETUP.md / ANALYSIS.md.

Kept as a script rather than pasted numbers so the tables in the write-ups are
derived from the committed cost model and cannot drift from it.

    python3 cell_table.py [--compose] [--audit]
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
LANES = ["dsp", "cpu", "gpu", "hta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compose", action="store_true",
                    help="also print the compose-failure table")
    ap.add_argument("--audit", action="store_true",
                    help="also print the undeclared cpu@int8 cells")
    a = ap.parse_args()
    cm = json.load(open(os.path.join(SWEEP, "cost_model.json")))
    cells, prov = cm["cells"], cm.get("cell_provenance", {})
    print(f"| tile | {' | '.join(l + ' (ms)' for l in LANES)} | statistic |")
    print("|---|" + "---|" * (len(LANES) + 1))
    for k in sorted(cells):
        v = cells[k]
        row = []
        for l in LANES:
            row.append(f"{v[l]/1000:.3f}" if v.get(l) is not None else "—")
        st = {prov.get(f"{k}@{l}", {}).get("statistic") for l in LANES
              if v.get(l) is not None}
        print(f"| `{k}` | {' | '.join(row)} | {'/'.join(sorted(x for x in st if x))} |")
    if a.audit:
        print("\nUndeclared int8-on-CPU cells (measured for audit, not solved against):\n")
        print("| tile | cpu fp32 (ms) | cpu int8 (ms) | int8/fp32 |")
        print("|---|---|---|---|")
        for k in sorted(cells):
            v = cells[k]
            if v.get("cpu@int8") is None or v.get("cpu") is None:
                continue
            print(f"| `{k}` | {v['cpu']/1000:.3f} | {v['cpu@int8']/1000:.3f} | "
                  f"{v['cpu@int8']/v['cpu']:.1f}x |")
    if a.compose:
        print("\nCompose failures:\n")
        print("| tile | backend | reason |")
        print("|---|---|---|")
        seen = set()
        for f in cm.get("compose_failures", []):
            key = (f.get("cell"), f.get("backend"))
            if key in seen:
                continue
            seen.add(key)
            r = " ".join(str(f.get("reason", "")).split())[:150]
            print(f"| `{f.get('cell')}` | {f.get('backend')} | {r} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
