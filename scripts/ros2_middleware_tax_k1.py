#!/usr/bin/env python3
"""Measure the ROS 2 middleware tax on the K1, so our static-pin model can be bounded.

WHY THIS EXISTS. The analytical ROS arms (Tier A in docs/Baselines/ros_baseline_tiers.md) are a POLICY
MODEL: scripts/ros_pinning_*.py re-lay XPU-RT's own measured per-dispatch durations under a "one
node per net, pinned to one hart, graph runs sequentially, periodic timer" policy. (The measured
arms, Tier C, run ROS 2 C++ nodes on the board; this script bounds the model, not those.) It pays the same per-op compute cost as
XPU-RT, so it isolates placement policy -- but it charges NOTHING for the middleware: no DDS
serialization, no message copies, no executor wake-up, no callback jitter. On the coupled
chain the static-pin model gives 35.58 ms end-to-end against a 33.3 ms deadline, while real
ROS 2 C++ nodes running these same kernels measure 53.22 ms.

THE ASYMMETRY THAT MATTERS, AND WHY BOTH EXECUTORS ARE MEASURED. The omission cuts both ways
and only one way is safe for us:

  * Middleware cost is strictly POSITIVE, so adding it makes the modelled arm slower. If that
    is the whole story, the model is a LOWER BOUND on real ROS and every claim of the form
    "static pinning misses the deadline" only gets stronger.
  * But our model also assumes SINGLE-THREADED, one-node-per-hart execution. A real deployment
    using a MultiThreadedExecutor, callback groups or composed nodes can overlap callbacks that
    our model serializes, and would be FASTER than the model. That is the direction that cuts
    against us, and it is not bounded by measuring the tax alone.

So this measures both executors. The single-threaded number bounds the model from one side and
the multi-threaded number from the other; reporting both keeps the comparison from pairing our
best metric against their worst.

WHAT IT MEASURES. A HOPS-long chain of ROS 2 nodes, matching the camera -> YOLO -> nav ->
control topology. Each node receives a message, optionally busy-waits its stage's measured
board compute, and republishes. The tax is the end-to-end latency MINUS the compute that was
deliberately burned:

    tax = e2e_latency - sum(compute_ms)

Run with --compute-ms 0 first: that is the pure middleware cost with nothing else in the way.
Then run with the real per-stage durations to check the tax does not shrink under load
(it usually grows -- a busy executor wakes up later).

HOW TO RUN. Needs no cross-compilation: it is rclpy, so it is copied to the board and run
there directly.

    scp scripts/ros2_middleware_tax_k1.py k1:/root/
    ssh k1 'source /opt/ros/*/setup.bash && python3 /root/ros2_middleware_tax_k1.py \
              --hops 3 --rate 45 --seconds 20 --executor single --out /root/tax_single.csv'
    ssh k1 'source /opt/ros/*/setup.bash && python3 /root/ros2_middleware_tax_k1.py \
              --hops 3 --rate 45 --seconds 20 --executor multi  --out /root/tax_multi.csv'

DO NOT run this while the board is in use: it spins CPU by design and would corrupt anyone
else's timing measurement, including ours.

READING THE RESULT. The number to quote is the MEDIAN tax with the p95 alongside; a real-time
claim lives on the tail, not the mean. Then:

    real ROS end-to-end, measured directly on the board : 53.65 ms, sustaining <=20 Hz

and if the multi-threaded tax is much lower, say so -- that is the honest caveat, and it is
better to publish it than to have it found.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
    from rclpy.callback_groups import ReentrantCallbackGroup
    from std_msgs.msg import String
except ImportError as e:                                    # noqa: BLE001
    sys.exit(f"rclpy not importable ({e}). Source the ROS 2 setup first:\n"
             "  source /opt/ros/*/setup.bash")


def _busy_ms(ms: float) -> None:
    """Burn `ms` of CPU. A BUSY WAIT, not a sleep, deliberately.

    time.sleep() yields the core, which lets the executor overlap the 'compute' with message
    handling and understates the tax. A real inference occupies the core; so does this.
    """
    if ms <= 0:
        return
    end = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < end:
        pass


class Stage(Node):
    """One hop: receive, burn this stage's measured compute, republish with the t0 intact."""

    def __init__(self, idx: int, n_hops: int, compute_ms: float, group):
        super().__init__(f"stage_{idx}")
        self._pub = self.create_publisher(String, f"chain_{idx + 1}", 10)
        self._compute = compute_ms
        self.create_subscription(String, f"chain_{idx}", self._cb, 10, callback_group=group)

    def _cb(self, msg: String) -> None:
        _busy_ms(self._compute)
        self._pub.publish(msg)                              # t0 travels unchanged down the chain


class Source(Node):
    def __init__(self, rate_hz: float, group):
        super().__init__("source")
        self._pub = self.create_publisher(String, "chain_0", 10)
        self.create_timer(1.0 / rate_hz, self._tick, callback_group=group)

    def _tick(self) -> None:
        m = String()
        m.data = repr(time.perf_counter())                  # t0, one clock, one process
        self._pub.publish(m)


class Sink(Node):
    def __init__(self, n_hops: int, group):
        super().__init__("sink")
        self.samples: list[float] = []
        self.create_subscription(String, f"chain_{n_hops}", self._cb, 10, callback_group=group)

    def _cb(self, msg: String) -> None:
        self.samples.append((time.perf_counter() - float(msg.data)) * 1000.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=3,
                    help="intermediate stages; 3 matches camera->YOLO->nav->control")
    ap.add_argument("--rate", type=float, default=45.0, help="source publish rate, Hz")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--compute-ms", default="0",
                    help="per-stage busy-wait, comma-separated (one per hop) or a single value "
                         "applied to all. Use the MEASURED board durations for the loaded case")
    ap.add_argument("--executor", choices=["single", "multi"], default="single",
                    help="single models our per-node-pinned baseline; multi is the "
                         "configuration that could BEAT the model -- measure both")
    ap.add_argument("--warmup", type=float, default=3.0, help="seconds discarded up front")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cs = [float(x) for x in str(a.compute_ms).split(",")]
    if len(cs) == 1:
        cs *= a.hops
    if len(cs) != a.hops:
        sys.exit(f"--compute-ms needs 1 or {a.hops} values, got {len(cs)}")

    rclpy.init()
    group = ReentrantCallbackGroup() if a.executor == "multi" else None
    src = Source(a.rate, group)
    stages = [Stage(i, a.hops, cs[i], group) for i in range(a.hops)]
    sink = Sink(a.hops, group)
    ex = MultiThreadedExecutor() if a.executor == "multi" else SingleThreadedExecutor()
    for n in (src, *stages, sink):
        ex.add_node(n)

    t_end = time.perf_counter() + a.seconds
    try:
        while time.perf_counter() < t_end:
            ex.spin_once(timeout_sec=0.05)
    except KeyboardInterrupt:
        pass

    keep = int(len(sink.samples) * (a.warmup / a.seconds))
    s = sorted(sink.samples[keep:])
    if not s:
        sys.exit("no messages completed the chain -- check the ROS 2 environment and DDS")

    burned = sum(cs)
    med = statistics.median(s)
    p95 = s[int(0.95 * (len(s) - 1))]
    print(f"\nROS 2 middleware tax on this board  [executor={a.executor}, hops={a.hops}, "
          f"rate={a.rate} Hz, n={len(s)}]")
    print(f"  per-stage compute burned : {cs}  (sum {burned:.3f} ms)")
    print(f"  end-to-end latency       : median {med:8.3f} ms   p95 {p95:8.3f} ms   "
          f"min {s[0]:.3f}  max {s[-1]:.3f}")
    print(f"  MIDDLEWARE TAX           : median {med - burned:8.3f} ms   p95 {p95 - burned:8.3f} ms")
    print(f"\n  our static-pin model says 35.58 ms end-to-end against a 33.3 ms deadline.")
    print(f"  with this tax added      : median {35.58 + med - burned:.2f} ms   "
          f"p95 {35.58 + p95 - burned:.2f} ms")
    print("  (single-threaded only bounds the direction that FAVOURS us -- report multi too)")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["executor", "hops", "rate_hz", "compute_ms_sum", "n",
                        "e2e_median_ms", "e2e_p95_ms", "tax_median_ms", "tax_p95_ms"])
            w.writerow([a.executor, a.hops, a.rate, burned, len(s),
                        f"{med:.4f}", f"{p95:.4f}", f"{med - burned:.4f}", f"{p95 - burned:.4f}"])
        print(f"\n  wrote {a.out}")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
