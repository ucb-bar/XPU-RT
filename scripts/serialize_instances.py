#!/usr/bin/env python3
"""Serialise concurrent instances of the SAME network in a scheduled JSON.

Why this is needed (not a workaround -- a missing constraint):

  modelblaster generates ONE set of intermediate buffers per MODEL
  (examples/<m>/<q>/generated/<backend>/buffers.c, e.g. 884 arrays / 30.9 MB
  for vint). Its header states the contract: "This file must be linked
  EXACTLY ONCE per model into any binary." Every instance of a network
  therefore shares that scratch storage.

  That is safe for periodic models, whose instances are separated by their
  period, and safe across BACKENDS within one instance (the documented case).
  It breaks when two instances of one network run CONCURRENTLY -- they
  clobber each other's scratch and a kernel then reads through a corrupted
  pointer:

      mcause: 5, Load access fault    mtval: 0x3ffc478b   (below DRAM base)
      t0/t1 = 0x7f7f7f7f...           (saturated int8 tensor data)

  Observed on fused_vint seeds 0/1/3 (two vint instances overlapping from
  0.11 ms and 0.43 ms); seed2, with a single vint instance, passed.

  Unbounded non-periodic tasks make this reachable: with no release window
  the scheduler is free to start vint0 and vint1 microseconds apart on the
  two cores.

The scheduler does not model this resource, so it emits infeasible schedules.
This pass makes them feasible by shifting whole instances (a constant offset
per instance -- instances are independent jobs with no cross-instance
dependencies, so relative timing inside an instance is preserved) until no
two instances of one network overlap. The makespan grows accordingly; that
growth is real and is what the hardware would have to do anyway.

Usage: serialize_instances.py IN.json OUT.json

(This was runs/sweeps/fpga_20260829-195805/drivers/serialize_instances.py,
moved here so a reproduction does not depend on one past run's output
directory. scripts/repro_fpga_sweep.sh --serialize inlines the same pass.)
"""
import json, re, sys, collections

KEY = re.compile(r"^([a-z0-9_]+?)(\d*)_dispatch_")

def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = json.load(open(src))
    D = d["dispatches"]

    inst = collections.defaultdict(list)          # (net, idx) -> [keys]
    for k in D:
        m = KEY.match(k)
        if not m:
            continue
        inst[(m.group(1), int(m.group(2) or 0))].append(k)

    span = {}
    for key, keys in inst.items():
        s = min(float(D[k]["start_time"]) for k in keys)
        e = max(float(D[k]["start_time"]) + float(D[k]["duration"]) for k in keys)
        span[key] = (s, e)

    bynet = collections.defaultdict(list)
    for (net, idx) in inst:
        bynet[net].append(idx)

    shifted = 0
    for net, idxs in bynet.items():
        if len(idxs) < 2:
            continue
        order = sorted(idxs, key=lambda i: span[(net, i)][0])
        free = None
        for i in order:
            s, e = span[(net, i)]
            if free is not None and s < free:
                off = free - s
                for k in inst[(net, i)]:
                    D[k]["start_time"] = float(D[k]["start_time"]) + off
                s, e = s + off, e + off
                span[(net, i)] = (s, e)
                shifted += 1
            free = e

    mk = max(float(v["start_time"]) + float(v["duration"]) for v in D.values())
    old = d.get("metadata", {}).get("makespan")
    d.setdefault("metadata", {})["makespan"] = mk
    d["metadata"]["serialized_same_network_instances"] = True
    json.dump(d, open(dst, "w"), indent=1)
    print(f"shifted {shifted} instance(s); makespan {old:.2f} -> {mk:.2f} ms"
          if old else f"shifted {shifted}; makespan {mk:.2f} ms")

if __name__ == "__main__":
    main()
