"""FireSim extension: stage a compiled CompGen bundle as a FireSim workload.

Mirrors Merlin's ``build_tools/hardware/scripts/run_baremetal_benchmarks.sh``
pattern — bare-metal RISC-V ELF linked with picolibc + libgloss_htif +
``htif.ld`` — so the output lands in ``results-workload/<name>/<name>0/uartlog``.

Deliberately separate from the Zephyr overlay (:mod:`compgen.extensions.zephyr`):
FireSim's canonical path on Chipyard is bare-metal HTIF, not Zephyr, so
we match what the vendor tooling exercises and expects.
"""

from __future__ import annotations

from compgen.extensions.firesim.build_workload import (
    FiresimWorkload,
    build_firesim_workload,
)

__all__ = ["FiresimWorkload", "build_firesim_workload"]
