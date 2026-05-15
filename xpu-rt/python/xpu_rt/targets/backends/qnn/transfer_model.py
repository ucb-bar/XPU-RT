"""Per-edge transfer cost — volume × dtype × machine-pair × qparam-delta.

Replaces the N×N machine matrix used in the prior demo. Every edge in
the island DAG has a tensor flowing across it; the transfer cost depends
on that tensor's volume and on the dtype/qp delta between producer and
consumer islands.

Components, all measurable on board (see scripts/profile_transfers.py):
  1. memcpy(volume_bytes, src_machine, dst_machine)
       — host buffer copy. CPU↔CPU is 0; cross-machine adds a fixed
         FastRPC/ION setup + bandwidth-driven term.
  2. dequant_quant(n_elem, src_dtype, dst_dtype)
       — pays only when dtypes differ (e.g. uint8↔fp16).
  3. rescale(n_elem, dtype)
       — pays when dtypes match but qparams differ (per-tensor scale or
         offset shift). Common in QDQ-imported IR where every layer has a
         distinct (scale, offset).

A transition with same dtype + same qp + same machine is free. Same
machine + same dtype + different qp is a rescale. Different dtype or
different machine triggers the corresponding additional term.

The CostTable holds the per-component coefficients; this module
combines them per-edge.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from .cost_table import CostTable
from .island_dag import IslandCandidate, QParams, TensorSpec


@dataclasses.dataclass
class EdgeSpec:
    """One DAG edge: producer's output tensor → consumer's input tensor."""
    producer_id: str
    consumer_id: str
    producer_out: TensorSpec
    consumer_in: TensorSpec
    producer_machine: str
    consumer_machine: str


def _qp_match(a: QParams, b: QParams) -> bool:
    if a.per_channel is not None or b.per_channel is not None:
        return a.per_channel == b.per_channel
    return abs(a.scale - b.scale) < 1e-6 and a.zero_point == b.zero_point


@dataclasses.dataclass
class TransferModel:
    """Stateless evaluator that prices a single DAG edge."""
    table: CostTable

    def cost_us(self, edge: EdgeSpec) -> float:
        out = edge.producer_out
        inp = edge.consumer_in
        # Producer's output volume drives bandwidth-dominated terms.
        vol_bytes = out.volume_bytes
        n_elem = vol_bytes // max(1, _bytes_per_elem(out.dtype))

        cost = 0.0
        # 1. Cross-machine memcpy (zero on same machine).
        cost += self.table.memcpy_us(vol_bytes, edge.producer_machine,
                                     edge.consumer_machine)

        # 2. Dtype change: pay dequant+quant.
        if out.dtype != inp.dtype:
            cost += self.table.dequant_quant_us(n_elem, out.dtype, inp.dtype)
        # 3. Same dtype but qparams differ: pay rescale.
        elif not _qp_match(out.qp, inp.qp):
            cost += self.table.rescale_us(n_elem, out.dtype)
        # 4. Same dtype + same qp + same machine: free (already covered by
        #    memcpy_us returning 0 above).
        return cost


def _bytes_per_elem(d: str) -> int:
    return {"uint8": 1, "int8": 1, "fp16": 2, "fp32": 4,
            "int32": 4, "sfixed_32": 4}[d]
