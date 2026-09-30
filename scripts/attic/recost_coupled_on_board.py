#!/usr/bin/env python3
"""Dependency-honoring board re-cost for a COUPLED (data-dependency-chain) schedule.

Identical calibration/multiplier logic to recost_schedule_on_board.py (imported, not
re-implemented), but re-times the schedule in TOPOLOGICAL order honoring the dispatch
`dependencies` DAG (dict-key strings), not just per-hart occupancy. The stock recost keys
new_end by the per-network integer id while deps are dict-key strings, so its cross-op
dependency term silently never resolves -- harmless for the INDEPENDENT warehouse nets
(edges=[]), wrong for a real camera->yolo->nav->control chain. Here we resolve deps by the
dict key so nav truly waits for yolo and control for nav, under board-inflated costs.

Also computes the end-to-end chain latency: release(camera/yolo instance) -> completion of
the dependent control instance, and checks it against chain.chain_end_to_end_deadline_ms.
"""
import argparse, json, os, sys, heapq, collections

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))
import recost_schedule_on_board as R  # reuse mult(), _net_inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--spec", required=True, help="spec with REAL per-net periods/windows for deadline reporting")
    ap.add_argument("--calibration", default=os.path.join(_REPO, "results/codesign_feedback/k1_board_calibration.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sched = json.load(open(a.schedule))
    disp = sched["dispatches"]                       # dict: key -> dispatch
    cal = json.load(open(a.calibration))
    spec = json.load(open(a.spec))
    nets = spec["networks"]
    known = set(nets)
    period = {n: float(v.get("period", 0) or 0) for n, v in nets.items()}
    window = {n: float(v.get("window_duration", 0) or 0) for n, v in nets.items()}
    chain = spec.get("chain", {})
    e2e_deadline = float(chain.get("chain_end_to_end_deadline_ms", 0) or 0)

    # ---- board durations + fixed metadata (net, instance, release) per dispatch key ----
    stats = collections.defaultdict(int); applied = []
    bdur = {}; net_of = {}; inst_of = {}; harts_of = {}; release_of = {}
    for k, d in disp.items():
        net, inst = R._net_inst(d["job_name"], known)
        m = R.mult(cal, net, d.get("module_name", ""), stats); applied.append(m)
        bdur[k] = float(d["duration"]) * m
        net_of[k] = net; inst_of[k] = inst
        harts_of[k] = d["hardware_target"].split("+")
        release_of[k] = inst * period.get(net, 0.0)

    # ---- topological order (Kahn), tie-broken by original start_time to preserve
    #      the solver's per-hart sequencing where the DAG leaves it free ----
    succ = collections.defaultdict(list); indeg = collections.defaultdict(int)
    for k, d in disp.items():
        deps = [dp for dp in d.get("dependencies", []) if dp in disp]
        indeg[k] += len(deps)
        for dp in deps:
            succ[dp].append(k)
    start0 = {k: float(d.get("start_time", 0.0)) for k, d in disp.items()}
    pq = [(start0[k], k) for k in disp if indeg[k] == 0]
    heapq.heapify(pq)
    topo = []
    while pq:
        _, k = heapq.heappop(pq)
        topo.append(k)
        for s in succ[k]:
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(pq, (start0[s], s))
    assert len(topo) == len(disp), f"cycle? {len(topo)} != {len(disp)}"

    # ---- ASAP re-time under board costs, honoring deps AND per-hart serialization ----
    hart_free = collections.defaultdict(float)
    new_end = {}
    inst_finish = collections.defaultdict(float)   # (net, inst) -> board end
    for k in topo:
        d = disp[k]
        dep_end = max((new_end[dp] for dp in d.get("dependencies", []) if dp in new_end), default=0.0)
        start = max(release_of[k], dep_end, max((hart_free[h] for h in harts_of[k]), default=0.0))
        end = start + bdur[k]
        for h in harts_of[k]:
            hart_free[h] = end
        new_end[k] = end
        inst_finish[(net_of[k], inst_of[k])] = max(inst_finish[(net_of[k], inst_of[k])], end)
        d["start_time"] = start; d["duration"] = bdur[k]
        dl = release_of[k] + window.get(net_of[k], 0.0)
        late = window.get(net_of[k], 0.0) > 0 and end > dl + 1e-6
        d["deadline_miss"] = bool(late)
        d["deadline_overrun_us"] = max(0.0, end - dl) * 1000.0

    makespan = max(new_end.values(), default=0.0)
    miss = sum(1 for d in disp.values() if d.get("deadline_miss"))
    per_net_miss = collections.Counter(net_of[k] for k, d in disp.items() if d.get("deadline_miss"))
    total_lateness = sum(float(d["deadline_overrun_us"]) / 1000.0 for d in disp.values() if d.get("deadline_miss"))

    # ---- end-to-end chain latency: earliest release in chain -> latest completion of tail net ----
    chain_nodes = chain.get("nodes", list(nets))
    tail = chain_nodes[-1]
    head = chain_nodes[0]
    chain_release = min((release_of[k] for k in disp if net_of[k] == head), default=0.0)
    tail_finish = max((new_end[k] for k in disp if net_of[k] == tail), default=0.0)
    e2e = tail_finish - chain_release
    yolo_service = max((inst_finish[(n, i)] - i * period[n]
                        for (n, i) in inst_finish if "yolo" in n.lower()), default=0.0)

    md = sched.setdefault("metadata", {})
    md.update({
        "makespan": makespan, "board_recost": True, "dependency_honoring": True,
        "deadline_miss_count": miss, "per_net_deadline_miss": dict(per_net_miss),
        "total_lateness_ms": total_lateness,
        "chain_nodes": chain_nodes,
        "end_to_end_latency_ms": e2e,
        "end_to_end_deadline_ms": e2e_deadline,
        "end_to_end_deadline_miss": bool(e2e_deadline > 0 and e2e > e2e_deadline + 1e-6),
        "yolo_frame_latency_ms": yolo_service,
        "yolo_frame_latency_statistic": "maximum release-to-output response",
    })
    json.dump(sched, open(a.out, "w"), indent=1)
    metrics = {
        "makespan_ms": makespan,
        "end_to_end_latency_ms": e2e,
        "end_to_end_deadline_ms": e2e_deadline,
        "end_to_end_deadline_miss": bool(e2e_deadline > 0 and e2e > e2e_deadline + 1e-6),
        "yolo_frame_latency_ms": yolo_service,
        "deadline_miss_count": miss,
        "per_net_deadline_miss": dict(per_net_miss),
        "total_lateness_ms": total_lateness,
        "board_recost": True, "dependency_honoring": True,
        "multiplier_source_counts": dict(stats),
        "multiplier_min": min(applied), "multiplier_max": max(applied),
        "multiplier_mean": sum(applied) / len(applied),
    }
    json.dump(metrics, open(a.out.replace(".json", "_metrics.json"), "w"), indent=1)
    print(f"recost {os.path.basename(a.schedule)} -> makespan {makespan:.2f} ms | "
          f"E2E {e2e:.2f} ms (deadline {e2e_deadline:.1f} ms, "
          f"{'MISS' if metrics['end_to_end_deadline_miss'] else 'MET'}) | "
          f"yolo_service {yolo_service:.2f} ms | per-net miss {dict(per_net_miss)} "
          f"| mult mean {metrics['multiplier_mean']:.2f}")


if __name__ == "__main__":
    main()
