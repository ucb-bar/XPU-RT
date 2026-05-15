#!/usr/bin/env python3
"""V2 demo: per-edge transfer + QDQ-aware bridges + variant groups.

Builds the same 7-island YOLOv8-stem-shaped DAG as v1 but using the
qnn_scheduler package: every cost (execute, init, memcpy, rescale,
dequant/quant) comes from `qrb5165_costs.json` — no constants in this
file. The transitions encode a real QDQ scenario where:

  * HTA stem A outputs uint8 with q-params (s=0.10, zp=128).
  * GPU stem B outputs fp16.
  * CPU concat bridge takes (uint8 from HTA) + (fp16 from GPU) and
    requantizes to a *different* uint8 q-params (s=0.05, zp=64) — so even
    HTA→CPU here pays a rescale, not just memcpy.
  * HTA trunk consumes that new uint8 q-params natively.
  * Heads run in parallel on HTA / GPU / CPU; the GPU head input is fp16
    so the HTA→GPU transition pays a dequant+quant.
  * CPU decode reads fp16 from the GPU head and uint8 from HTA/CPU heads
    and returns fp32 boxes.

Run from XPU-RT root:
    cd /scratch2/agustin/XPU-RT
    conda run -n merlin-dev uv run python scripts/qnn_island_demo_v2.py
"""

from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))

from qnn_scheduler.cost_table import CostTable  # noqa: E402
from qnn_scheduler.island_dag import (  # noqa: E402
    IslandCandidate,
    IslandVariantGroup,
    QParams,
    TensorSpec,
)
from qnn_scheduler.scheduler import schedule_groups  # noqa: E402
from qnn_scheduler.seed_table_qrb5165 import seed  # noqa: E402


def _ts(name: str, shape: tuple[int, ...], dtype: str,
        scale: float = 1.0, zp: int = 0) -> TensorSpec:
    return TensorSpec(name=name, shape=shape, dtype=dtype,
                      qp=QParams(scale=scale, zero_point=zp))


def build_groups() -> list[IslandVariantGroup]:
    # -- group 0: input split (CPU passthrough) --
    in_t  = _ts("input", (1, 320, 320, 3), "uint8", 0.035, 128)
    g_split = IslandVariantGroup(
        group_id="input_split",
        upstream_group_ids=(),
        alternatives=[
            IslandCandidate(
                candidate_id="input_split_cpu",
                group_id="input_split",
                backend="CPU",
                op_key="Split@1x320x320x3@uint8",
                inputs=(in_t,),
                outputs=(in_t,),
            ),
        ],
    )

    # -- group 1: HTA stem (uint8 conv, output qp distinct from input) --
    hta_stem_out = _ts("hta_stem_out", (1, 160, 160, 16), "uint8", 0.10, 128)
    g_hta_stem = IslandVariantGroup(
        group_id="hta_stem",
        upstream_group_ids=("input_split",),
        alternatives=[
            IslandCandidate(
                candidate_id="hta_stem_uint8",
                group_id="hta_stem",
                backend="HTA",
                op_key="Conv2d@1x320x320x3->1x160x160x16,g1,k3,s2@uint8",
                inputs=(in_t,),
                outputs=(hta_stem_out,),
                static_setup_us=20731.0,
            ),
        ],
    )

    # -- group 2: GPU stem (fp16 conv) --
    gpu_stem_out = _ts("gpu_stem_out", (1, 160, 160, 16), "fp16")
    g_gpu_stem = IslandVariantGroup(
        group_id="gpu_stem",
        upstream_group_ids=("input_split",),
        alternatives=[
            IslandCandidate(
                candidate_id="gpu_stem_fp16",
                group_id="gpu_stem",
                backend="GPU",
                op_key="Conv2d@1x320x320x3->1x160x160x16,g1,k3,s2@fp16",
                inputs=(in_t,),    # GPU island's internal Dequantize handles dtype
                outputs=(gpu_stem_out,),
                static_setup_us=6983.0,
            ),
        ],
    )

    # -- group 3: CPU concat + requantize bridge (qp shifts!) --
    bridge_out = _ts("bridge_out", (1, 160, 160, 32), "uint8", 0.05, 64)
    bridge_in_a = _ts("bridge_in_a", (1, 160, 160, 16), "uint8", 0.10, 128)  # from HTA stem
    g_bridge = IslandVariantGroup(
        group_id="cpu_bridge",
        upstream_group_ids=("hta_stem", "gpu_stem"),
        alternatives=[
            IslandCandidate(
                candidate_id="cpu_bridge",
                group_id="cpu_bridge",
                backend="CPU",
                op_key="Concat@1x160x160x32@uint8",
                inputs=(bridge_in_a, gpu_stem_out),  # uint8 + fp16 → uint8
                outputs=(bridge_out,),
            ),
        ],
    )

    # -- group 4: HTA trunk conv --
    trunk_out = _ts("trunk_out", (1, 80, 80, 32), "uint8", 0.04, 128)
    g_trunk = IslandVariantGroup(
        group_id="hta_trunk",
        upstream_group_ids=("cpu_bridge",),
        alternatives=[
            IslandCandidate(
                candidate_id="hta_trunk_uint8",
                group_id="hta_trunk",
                backend="HTA",
                op_key="Conv2d@1x160x160x16->1x80x80x32,g1,k3,s2@uint8",
                inputs=(bridge_out,),
                outputs=(trunk_out,),
            ),
        ],
    )

    # -- groups 5-7: three parallel detection heads --
    head_out_u8  = _ts("head_out_u8",  (1, 80, 80, 16), "uint8", 0.03, 128)
    head_out_fp16 = _ts("head_out_fp16", (1, 80, 80, 16), "fp16")
    g_head1 = IslandVariantGroup(
        group_id="head1",
        upstream_group_ids=("hta_trunk",),
        alternatives=[
            IslandCandidate(
                candidate_id="head1_hta",
                group_id="head1",
                backend="HTA",
                op_key="Conv2d@1x80x80x32->1x80x80x16,g1,k1,s1@uint8",
                inputs=(trunk_out,),
                outputs=(head_out_u8,),
            ),
        ],
    )
    g_head2 = IslandVariantGroup(
        group_id="head2",
        upstream_group_ids=("hta_trunk",),
        alternatives=[
            IslandCandidate(
                candidate_id="head2_gpu",
                group_id="head2",
                backend="GPU",
                op_key="Conv2d@1x80x80x32->1x80x80x16,g1,k1,s1@fp16",
                inputs=(trunk_out,),  # uint8→fp16 transition charged here
                outputs=(head_out_fp16,),
            ),
        ],
    )
    g_head3 = IslandVariantGroup(
        group_id="head3",
        upstream_group_ids=("hta_trunk",),
        alternatives=[
            IslandCandidate(
                candidate_id="head3_cpu",
                group_id="head3",
                backend="CPU",
                op_key="Conv2d@1x80x80x32->1x80x80x16,g1,k1,s1@uint8",
                inputs=(trunk_out,),
                outputs=(head_out_u8,),
            ),
        ],
    )

    # -- group 8: CPU decode (fp32 box/score) --
    decode_out = _ts("decode_out", (1, 100, 5), "fp32")
    g_decode = IslandVariantGroup(
        group_id="cpu_decode",
        upstream_group_ids=("head1", "head2", "head3"),
        alternatives=[
            IslandCandidate(
                candidate_id="cpu_decode",
                group_id="cpu_decode",
                backend="CPU",
                op_key="Decode@1x80x80x16@fp32",
                inputs=(head_out_u8,),
                outputs=(decode_out,),
            ),
        ],
    )

    return [g_split, g_hta_stem, g_gpu_stem, g_bridge, g_trunk,
            g_head1, g_head2, g_head3, g_decode]


def main() -> int:
    table = seed()
    groups = build_groups()
    res = schedule_groups(groups, table, pick_strategy="first",
                          iterations_amortized=100.0)
    print("\n=== Heterogeneous schedule (v2: per-edge, qp-aware) ===")
    print(f"{'island':<28} {'machine':<6} {'start (us)':>12} {'finish (us)':>12} {'dur (us)':>10}")
    for cand_id, m in res.machine.items():
        s = res.start_us[cand_id]; f = res.finish_us[cand_id]
        print(f"{cand_id:<28} {m:<6} {s:>12.0f} {f:>12.0f} {f-s:>10.0f}")
    print(f"\nmakespan: {res.makespan_us:.0f} us  ({res.makespan_us/1000:.2f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
