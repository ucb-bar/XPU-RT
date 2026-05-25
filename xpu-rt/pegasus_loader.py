"""
M20 — Pegasus literature DAG loader.

We don't need the actual Pegasus DAX XML format. The published Pegasus
workflow shapes (Montage / CyberShake / Epigenomics) are well-known
topologically. We construct them programmatically here with realistic
per-stage op_kind affinities so our existing scheduler registry can run
them, then compare HEFT's makespan to the critical-path lower bound — the
standard heterogeneous-scheduling literature metric.

Each workflow returns a Workload using the 3-machine (CPU/GPU/NPU) SoC
that the rest of the benchmark suite uses.

Job-type → per-machine cost affinities (from typical heterogeneous
scheduling papers and from QRB5165 measurements where applicable):

  compute_heavy   (mProject, SeismogramSynthesis, fastQC): NPU best
  cpu_serial      (mConcatFit, ZipPSA): CPU only
  io_bound        (mBackground, ExtractSGT): CPU best
  reduction       (mImgTbl, peakValCalc): CPU or GPU
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Workload


# 3-machine SoC config (must match scenarios.MACHINES / COMBOS / TRANSFER)
MACHINES = ["CPU", "GPU", "NPU"]
COMBOS = [["CPU"], ["GPU"], ["NPU"]]
TRANSFER = np.array([
    [0.0, 5.0, 30.0],
    [5.0, 0.0, 30.0],
    [30.0, 30.0, 0.0],
])

# Per-(stage, machine) cost in microseconds. Derived from the
# qrb5165_costs.json mean_us where applicable (Conv2d ≈ 5910 us on HTA;
# elementwise ≈ 260 us on CPU) and scaled to keep workflows in a
# representative microsecond range.
_KIND_COSTS = {
    "compute_heavy":   [320.0, 180.0, 60.0],   # NPU best (e.g. matmul, conv)
    "cpu_serial":      [120.0, 200.0, 1e9],    # CPU only (np.inf on NPU)
    "io_bound":        [80.0, 100.0, 250.0],   # CPU/GPU close, NPU expensive
    "reduction":       [60.0, 50.0, 150.0],    # GPU mildly best
    "small_glue":      [20.0, 30.0, 100.0],    # Tiny CPU op
}


def _op(name: str, kind: str, *, preds=None) -> Operation:
    costs = list(_KIND_COSTS[kind])
    inf = set()
    for k, c in enumerate(costs):
        if c >= 1e8:
            inf.add(k)
    op = Operation(
        processing_times=[1e9 if c >= 1e8 else c for c in costs],
        predecessors=list(preds or []),
        operation_name=name,
        infeasible_combinations=inf,
    )
    return op


# ----------------------------------------------------------------------------
# Workflow builders (programmatic Pegasus topologies)
# ----------------------------------------------------------------------------


def montage(n_images: int = 5) -> Workload:
    """Montage astronomy mosaic. 25 ops typical (n_images=5).

    Topology:
      n_images mProject (compute_heavy, parallel)
      → for each pair of adjacent images: mDiffFit (reduction)
      → 1 mConcatFit (cpu_serial)
      → 1 mBgModel (cpu_serial)
      → n_images mBackground (io_bound, parallel)
      → 1 mImgTbl (cpu_serial)
      → 1 mAdd (compute_heavy)
      → 1 mShrink (small_glue)
      → 1 mJPEG (cpu_serial)
    """
    ops: List[Operation] = []
    projects = [_op(f"mProject_{i}", "compute_heavy") for i in range(n_images)]
    ops.extend(projects)
    difffits = []
    for i in range(n_images - 1):
        d = _op(f"mDiffFit_{i}", "reduction", preds=[projects[i], projects[i + 1]])
        difffits.append(d); ops.append(d)
    concat = _op("mConcatFit", "cpu_serial", preds=difffits)
    bgmodel = _op("mBgModel", "cpu_serial", preds=[concat])
    ops.extend([concat, bgmodel])
    backgrounds = [_op(f"mBackground_{i}", "io_bound", preds=[bgmodel, projects[i]])
                   for i in range(n_images)]
    ops.extend(backgrounds)
    imgtbl = _op("mImgTbl", "cpu_serial", preds=backgrounds)
    madd = _op("mAdd", "compute_heavy", preds=[imgtbl])
    mshrink = _op("mShrink", "small_glue", preds=[madd])
    mjpeg = _op("mJPEG", "cpu_serial", preds=[mshrink])
    ops.extend([imgtbl, madd, mshrink, mjpeg])
    return Workload(ops, MACHINES, TRANSFER, machine_combinations=COMBOS,
                    job_names=["montage"])


def cybershake(n_sgts: int = 10) -> Workload:
    """CyberShake earthquake simulation. ~30 ops typical (n_sgts=10).

    Topology:
      1 ZipPSA (cpu_serial)
      → n_sgts ExtractSGT (io_bound, fanout)
      → n_sgts SeismogramSynthesis (compute_heavy, one per ExtractSGT)
      → n_sgts/2 PeakValCalcOkaya (reduction, fanin pairs)
      → 1 ZipSeis (cpu_serial)
    """
    ops: List[Operation] = []
    zip_psa = _op("ZipPSA", "cpu_serial"); ops.append(zip_psa)
    extracts = [_op(f"ExtractSGT_{i}", "io_bound", preds=[zip_psa])
                for i in range(n_sgts)]
    ops.extend(extracts)
    seismos = [_op(f"SeismogramSynth_{i}", "compute_heavy", preds=[extracts[i]])
               for i in range(n_sgts)]
    ops.extend(seismos)
    peaks = []
    for i in range(0, n_sgts, 2):
        pair = [seismos[i]] + ([seismos[i + 1]] if i + 1 < n_sgts else [])
        p = _op(f"PeakValCalc_{i//2}", "reduction", preds=pair)
        peaks.append(p); ops.append(p)
    zipseis = _op("ZipSeis", "cpu_serial", preds=peaks); ops.append(zipseis)
    return Workload(ops, MACHINES, TRANSFER, machine_combinations=COMBOS,
                    job_names=["cybershake"])


def epigenomics(n_lanes: int = 6) -> Workload:
    """Epigenomics genomics pipeline. ~46 ops typical (n_lanes=6).

    Topology (multi-stage parallel):
      n_lanes fastQC (io_bound, parallel)
      → n_lanes filterContams (cpu_serial, one per lane)
      → n_lanes sol2sanger (cpu_serial, one per lane)
      → n_lanes fastq2bfq (cpu_serial, one per lane)
      → n_lanes mapMerge (compute_heavy, joins pairs)
      → 1 maqIndex (cpu_serial)
      → 1 pileup (compute_heavy)
      → 1 chr21 (reduction)
    """
    ops: List[Operation] = []
    fastqc = [_op(f"fastQC_{i}", "io_bound") for i in range(n_lanes)]
    ops.extend(fastqc)
    filterc = [_op(f"filterContams_{i}", "cpu_serial", preds=[fastqc[i]])
               for i in range(n_lanes)]
    sol2sanger = [_op(f"sol2sanger_{i}", "cpu_serial", preds=[filterc[i]])
                  for i in range(n_lanes)]
    fastq2bfq = [_op(f"fastq2bfq_{i}", "cpu_serial", preds=[sol2sanger[i]])
                 for i in range(n_lanes)]
    ops.extend(filterc); ops.extend(sol2sanger); ops.extend(fastq2bfq)
    mapmerges = []
    for i in range(0, n_lanes, 2):
        pair = [fastq2bfq[i]] + ([fastq2bfq[i + 1]] if i + 1 < n_lanes else [])
        m = _op(f"mapMerge_{i//2}", "compute_heavy", preds=pair)
        mapmerges.append(m); ops.append(m)
    maqindex = _op("maqIndex", "cpu_serial", preds=mapmerges)
    pileup = _op("pileup", "compute_heavy", preds=[maqindex])
    chr21 = _op("chr21", "reduction", preds=[pileup])
    ops.extend([maqindex, pileup, chr21])
    return Workload(ops, MACHINES, TRANSFER, machine_combinations=COMBOS,
                    job_names=["epigenomics"])


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


PEGASUS_DAGS = {
    "montage_25":     lambda: montage(n_images=5),
    "cybershake_30":  lambda: cybershake(n_sgts=10),
    "epigenomics_46": lambda: epigenomics(n_lanes=6),
}
