#!/usr/bin/env python3
"""Parse Vivado hierarchical utilization.txt reports for the three Saturn
V128D128 prototype builds (vanilla / no-FP64 / no-FP32 FP16-only) and emit
a comparison plot showing per-design LUT breakdown by major component."""

import argparse
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Node:
    name: str          # leaf instance name (no path)
    depth: int         # indentation depth in the report
    module: str
    total_luts: int
    children: List["Node"] = field(default_factory=list)
    parent: Optional["Node"] = None

    def descendants(self):
        for c in self.children:
            yield c
            yield from c.descendants()


_PCT_RE = re.compile(r"\((\d+(?:\.\d+)?)%\)")


def _int(cell: str) -> int:
    """'788198(45.61%)' -> 788198"""
    cell = cell.strip()
    if not cell:
        return 0
    m = re.match(r"\s*([0-9]+)", cell)
    return int(m.group(1)) if m else 0


def parse_report(path: Path) -> Node:
    """Parse Vivado hierarchical utilization report into a tree of Node.

    Vivado emits one row per instance:
      |  <leading spaces><instance>  | <module> | <total_luts>(<pct>%) | ... |
    The number of leading spaces in the instance column (before the
    instance name) encodes the hierarchy depth (2 spaces per level)."""
    root: Optional[Node] = None
    stack: List[Node] = []           # depth-indexed open path

    in_table = False
    for raw in path.read_text().splitlines():
        if "Utilization by Hierarchy" in raw and not in_table:
            in_table = True
            continue
        if not in_table:
            continue
        # Table delimiters are '+---...' rows
        if not raw.startswith("|"):
            continue
        # Skip the column header row (contains "Instance")
        if "Instance" in raw and "Module" in raw and "Total LUTs" in raw:
            continue

        # Split on '|' — first and last cells are empty (leading/trailing pipes)
        cells = [c for c in raw.split("|")][1:-1]
        if len(cells) < 3:
            continue

        # cells[0] is "  <indent><instance>  " (always trimmed of pipes only).
        # Compute the indent from BEFORE the trim.
        col0 = cells[0]
        # Strip the leading single space Vivado pads pipes with, then count
        # remaining leading spaces as 2-per-depth-level.
        stripped = col0.lstrip(" ")
        # Vivado uses 2 spaces per nesting level after the pipe-padding "+ 1 space".
        indent_chars = len(col0) - len(stripped) - 1
        if indent_chars < 0:
            indent_chars = 0
        depth = indent_chars // 2

        inst = stripped.strip()
        module = cells[1].strip()
        # Skip the synthetic "(instance)" self rows that just attribute LUTs
        # to the instance itself (parens-wrapped).  Their depth equals the
        # parent, and their LUTs are already a subset.
        if inst.startswith("(") and inst.endswith(")"):
            continue
        luts = _int(cells[2])

        node = Node(name=inst, depth=depth, module=module, total_luts=luts)

        # Pop stack to depth-1 (so the top of stack is the parent).
        while len(stack) > depth:
            stack.pop()
        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        else:
            root = node
        stack.append(node)
    if root is None:
        raise RuntimeError(f"No hierarchy table found in {path}")
    return root


def find_by_substr(root: Node, *substrs) -> Optional[Node]:
    """Find first descendant whose instance OR module matches any substr."""
    for n in root.descendants():
        for s in substrs:
            if s in n.name or s in n.module:
                return n
    return None


def sum_subtree(node: Optional[Node]) -> int:
    return node.total_luts if node else 0


# Component buckets to surface in the plot.  Each entry is a list of
# matchers — a matcher is either:
#   - {"inst": exact_str}  — instance name equality
#   - {"mod":  exact_str}  — module name equality
#   - {"inst_contains": str} — substring on instance name
#   - {"mod_contains":  str} — substring on module name
# A matcher hits if either applies.  Buckets are filled in order; once a
# node (and its subtree) is claimed by an earlier bucket, later buckets
# cannot re-claim it.  Tuned for chipyard fpga/vcu118 hierarchy:
#   VCU118FPGATestHarness
#     mig (XilinxVCU118MIG)              <- DDR controller
#     chiptop0/system (DigitalTop)
#       coh_wrapper (CoherenceManagerWrapper)  <- L2
#       cbus/sbus/pbus/tlDM/bootrom_domain  <- system fabric
#       tile_prci_domain/element_reset_domain_rockettile (RocketTile)
#         core (Rocket)                   <- integer core
#         dcache (DCache)                 <- L1 D
#         frontend (Frontend)             <- L1 I + branch
#         fpuOpt (FPU)                    <- SCALAR FPU
#         vector_unit (SaturnRocketUnit)  <- SATURN VECTOR UNIT
COMPONENT_BUCKETS: "OrderedDict[str, List[dict]]" = OrderedDict([
    ("DDR controller (MIG)",  [{"mod": "XilinxVCU118MIG"}]),
    ("L2 cache",              [{"inst": "coh_wrapper"},
                               {"mod": "CoherenceManagerWrapper"},
                               {"mod": "InclusiveCache"}]),
    # Saturn first so it's claimed inside RocketTile before "core" matches
    # something deeper inside Saturn.
    ("Saturn vector unit",    [{"inst": "vector_unit"},
                               {"mod": "SaturnRocketUnit"}]),
    ("Scalar FPU",            [{"inst": "fpuOpt"}]),
    ("Rocket core (integer)", [{"inst": "core"}, {"mod": "Rocket"}]),
    ("L1 D-cache",            [{"inst": "dcache"}]),
    ("L1 I-cache + frontend", [{"inst": "frontend"}]),
    ("System bus & periph",   [{"inst": "cbus"}, {"inst": "sbus"},
                               {"inst": "pbus"}, {"inst": "bootrom_domain"},
                               {"inst": "tlDM"}, {"mod": "TLDebugModule"}]),
])


def _node_matches(node: Node, matchers: List[dict]) -> bool:
    for m in matchers:
        if "inst" in m and node.name == m["inst"]:
            return True
        if "mod" in m and node.module == m["mod"]:
            return True
        if "inst_contains" in m and m["inst_contains"] in node.name:
            return True
        if "mod_contains" in m and m["mod_contains"] in node.module:
            return True
    return False


def pick_components(root: Node) -> "OrderedDict[str, int]":
    """For a parsed report, return labelled LUT counts for the component
    buckets we care about, plus an 'Other' bucket for unaccounted LUTs."""
    out: "OrderedDict[str, int]" = OrderedDict()
    claimed_ids = set()  # id(node) — node + all descendants claimed

    for label, matchers in COMPONENT_BUCKETS.items():
        # Collect topmost matches that aren't already claimed.  Walk the
        # tree depth-first; once we accept a node, prune its subtree.
        matches: List[Node] = []
        def walk(n: Node):
            if id(n) in claimed_ids:
                return
            if _node_matches(n, matchers):
                matches.append(n)
                return  # do not recurse — claim only the topmost
            for c in n.children:
                walk(c)
        walk(root)
        bucket_total = sum(n.total_luts for n in matches)
        for n in matches:
            claimed_ids.add(id(n))
            for d in n.descendants():
                claimed_ids.add(id(d))
        out[label] = bucket_total

    out["Other"] = max(0, root.total_luts - sum(out.values()))
    out["__TOTAL__"] = root.total_luts
    return out


def emit_plot(designs: "OrderedDict[str, OrderedDict[str, int]]", outpath: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax_total, ax_stack) = plt.subplots(1, 2, figsize=(14, 6),
                                              gridspec_kw=dict(width_ratios=[1, 2]))

    labels = list(designs.keys())
    totals = [designs[d]["__TOTAL__"] for d in labels]

    # Plot A: total LUT count per design (simple bars).
    bars = ax_total.bar(labels, totals,
                        color=["#4477aa", "#ee6677", "#228833"])
    ax_total.set_ylabel("Total LUTs (synth, hierarchical)")
    ax_total.set_title("Top-level LUT count per design")
    for b, t in zip(bars, totals):
        ax_total.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.005,
                      f"{t:,}", ha="center", va="bottom", fontsize=9)
    ax_total.tick_params(axis="x", rotation=15)
    ax_total.set_ylim(0, max(totals) * 1.15)

    # Plot B: stacked-bar by component.
    component_order = [k for k in next(iter(designs.values())).keys()
                       if k != "__TOTAL__"]
    bottoms = np.zeros(len(labels))
    palette = ["#332288", "#88CCEE", "#44AA99", "#117733",
               "#999933", "#DDCC77", "#CC6677", "#AA4499"]
    for i, comp in enumerate(component_order):
        vals = np.array([designs[d].get(comp, 0) for d in labels])
        ax_stack.bar(labels, vals, bottom=bottoms,
                     label=comp, color=palette[i % len(palette)])
        # Annotate value inside each segment if it's >5% of total.
        for x, (v, t) in enumerate(zip(vals, totals)):
            if t > 0 and v / t > 0.05:
                ax_stack.text(x, bottoms[x] + v / 2, f"{int(v):,}",
                              ha="center", va="center", fontsize=8,
                              color="white" if v / t > 0.1 else "black")
        bottoms += vals
    ax_stack.set_ylabel("LUTs")
    ax_stack.set_title("Hierarchical LUT breakdown by component")
    ax_stack.legend(loc="upper right", fontsize=9)
    ax_stack.tick_params(axis="x", rotation=15)

    fig.suptitle("Saturn V128D128 prototype: FPU/vector area comparison\n"
                 "(VCU118, Vivado 2023.1 post-synth, -hierarchical)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    print(f"Wrote plot: {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", required=True, nargs=2,
                    metavar=("LABEL", "PATH"),
                    help="design label + path to utilization.txt (repeat per design)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--print-only", action="store_true",
                    help="just dump the per-design tables to stdout")
    args = ap.parse_args()

    designs: "OrderedDict[str, OrderedDict[str, int]]" = OrderedDict()
    for label, path in args.report:
        # Allow shell-friendly literal \n in labels.
        label = label.replace("\\n", "\n")
        root = parse_report(Path(path))
        comp = pick_components(root)
        designs[label] = comp
        print(f"\n=== {label} ({path}) ===")
        for k, v in comp.items():
            print(f"  {k:30s} {v:>10,}")

    if args.print_only:
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    emit_plot(designs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
