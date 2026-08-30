#!/usr/bin/env python3
"""Per-tensor int8 comparison of two qnn-net-run --debug dumps.

    python3 tools/compare_dumps.py <dir_a> <dir_b> [--label-a GPU --label-b CPU]
                                   [--order <converter _net.json>]

Both directories are Result_0/ dumps taken with
`--debug --use_native_input_files --use_native_output_files`, so every tensor is
raw int8 and the two runs saw byte-identical inputs.  With --order the tensors
are listed in graph execution order, which is what makes error growth with
depth visible (see README §5).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir_a')
    ap.add_argument('dir_b')
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    ap.add_argument('--order', help='converter <model>_net.json, for execution order')
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.dir_a) if f.endswith('_native.raw'))
    if args.order:
        graph = json.load(open(args.order))['graph']
        ordered = []
        for node in graph['nodes'].values():
            for t in node['output_names']:
                f = t.replace('/', '_').replace(':', '_') + '_native.raw'
                if f in names and f not in ordered:
                    ordered.append(f)
        names = ordered + [n for n in names if n not in ordered]

    print(f"{'tensor':44s} {'n':>8s}  {args.label_a}=={args.label_b}   max   mean")
    for f in names:
        pa = os.path.join(args.dir_a, f)
        pb = os.path.join(args.dir_b, f)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        a = np.fromfile(pa, dtype=np.uint8)
        b = np.fromfile(pb, dtype=np.uint8)
        if a.size != b.size:
            print(f'{f[:44]:44s} size mismatch {a.size} vs {b.size}')
            continue
        d = np.abs(a.astype(int) - b.astype(int))
        print(f'{f[:-11][:44]:44s} {a.size:8d}  {100 * np.mean(a == b):9.2f}%  {d.max():4d}  '
              f'{d.mean():6.3f}')


if __name__ == '__main__':
    main()
