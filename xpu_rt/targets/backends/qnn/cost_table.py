"""On-board-measured cost table — the data the scheduler consults.

The table replaces every hardcoded latency in the codebase. Every entry
is the result of an actual board measurement (or an explicit
extrapolation flagged as such). The scheduler will refuse to use entries
flagged extrapolated unless the caller passes `allow_extrapolation=True`.

Schema (JSON on disk):
{
  "schema_version": 1,
  "device": "qrb5165",
  "qairt_sdk": "2.45.0.260326",
  "measured_at": "2026-05-07T03:34:11Z",
  "execute": {
      "<op_key>::<backend>::<dtype>::<fused?>": {
          "mean_us": float,
          "p50_us": float,
          "p99_us": float,
          "iters": int,
          "extrapolated": bool,
          "source": "qnn-net-run|merlin-dispatch-bench|estimate"
      }, ...
  },
  "init": {
      "<backend>": { "mean_us": float, "iters": int }, ...
  },
  "memcpy": {
      "<src_machine>__<dst_machine>": {
          "bytes_per_us_mean": float,
          "fixed_overhead_us": float
      }, ...
  },
  "rescale": {
      "<dtype>": { "us_per_elem": float, "fixed_overhead_us": float }, ...
  },
  "dequant_quant": {
      "<src_dtype>__<dst_dtype>": { "us_per_elem": float,
                                     "fixed_overhead_us": float }, ...
  }
}

`op_key` is a stable string built by canonical_op_key() so different code
paths (recognizer, profiler, scheduler) all hit the same row.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class OpKey:
    op_kind: str           # "Conv2d", "DepthwiseConv2d", "ElementWiseNeuron", "Concat", ...
    shape_signature: str   # canonical shape string ("1x320x320x3->1x160x160x16,k3s2g1")
    dtype: str             # "uint8", "fp16", ...

    def canonical(self) -> str:
        return f"{self.op_kind}@{self.shape_signature}@{self.dtype}"


@dataclasses.dataclass(frozen=True)
class BackendKey:
    backend: str   # "HTA" | "GPU" | "CPU"
    fused: bool


def canonical_op_key(op_kind: str,
                     in_shape: tuple[int, ...],
                     out_shape: tuple[int, ...],
                     dtype: str,
                     extra: dict | None = None) -> OpKey:
    """Build a stable string key. extra carries op-specific shape detail
    (kernel size, stride, group, axis...). Order is fixed for stability."""
    parts: list[str] = []
    parts.append("x".join(str(d) for d in in_shape))
    parts.append("->")
    parts.append("x".join(str(d) for d in out_shape))
    if extra:
        for k in sorted(extra):
            parts.append(f",{k}{extra[k]}")
    return OpKey(op_kind=op_kind,
                 shape_signature="".join(parts),
                 dtype=dtype)


@dataclasses.dataclass
class CostTable:
    schema_version: int = 1
    device: str = "qrb5165"
    qairt_sdk: str = ""
    measured_at: str = ""
    execute: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    init: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    memcpy: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    rescale: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    dequant_quant: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, path: pathlib.Path) -> CostTable:
        d = json.loads(path.read_text())
        return cls(
            schema_version=d.get("schema_version", 1),
            device=d.get("device", ""),
            qairt_sdk=d.get("qairt_sdk", ""),
            measured_at=d.get("measured_at", ""),
            execute=d.get("execute", {}),
            init=d.get("init", {}),
            memcpy=d.get("memcpy", {}),
            rescale=d.get("rescale", {}),
            dequant_quant=d.get("dequant_quant", {}),
        )

    def save(self, path: pathlib.Path) -> None:
        if not self.measured_at:
            self.measured_at = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))

    # ----- lookup API --------------------------------------------------
    def execute_us(self, op: OpKey, backend: BackendKey,
                   *, allow_extrapolation: bool = False) -> float:
        key = f"{op.canonical()}::{backend.backend}::{int(backend.fused)}"
        row = self.execute.get(key)
        if row is None:
            raise KeyError(f"no execute cost for {key}")
        if row.get("extrapolated") and not allow_extrapolation:
            raise ValueError(f"{key} is extrapolated; pass allow_extrapolation=True")
        return float(row["mean_us"])

    def init_us(self, backend: str) -> float:
        row = self.init.get(backend)
        return float(row["mean_us"]) if row else 0.0

    def memcpy_us(self, volume_bytes: int, src: str, dst: str) -> float:
        if src == dst:
            return 0.0
        key = f"{src}__{dst}"
        row = self.memcpy.get(key)
        if row is None:
            # Fallback: assume 8 GB/s LPDDR5 + 50µs setup per cross-machine hop.
            return 50.0 + volume_bytes / 8000.0
        return float(row["fixed_overhead_us"]) + volume_bytes / float(row["bytes_per_us_mean"])

    def rescale_us(self, n_elem: int, dtype: str) -> float:
        row = self.rescale.get(dtype)
        if row is None:
            return 5.0 + n_elem * 0.05  # 50ns/elem on A77, fallback
        return float(row["fixed_overhead_us"]) + n_elem * float(row["us_per_elem"])

    def dequant_quant_us(self, n_elem: int, src_dtype: str, dst_dtype: str) -> float:
        if src_dtype == dst_dtype:
            return 0.0
        key = f"{src_dtype}__{dst_dtype}"
        row = self.dequant_quant.get(key)
        if row is None:
            # Fallback: dequant ~50 ns/elem + quant ~50 ns/elem on A77.
            return 5.0 + n_elem * 0.10
        return float(row["fixed_overhead_us"]) + n_elem * float(row["us_per_elem"])
