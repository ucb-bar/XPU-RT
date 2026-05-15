"""Baremetal C/C++ code emission for compiled models.

Generates a complete cross-compilable C project from a ``ModelProgram``:
main.c, npu_driver.h/c, memory_map.h, weights, linker script, and Makefile.

Supports two deployment models:
- ``bare_metal``: Raw C with polling loop (chipyard tests style)
- ``zephyr``: Zephyr RTOS application with threads and semaphores
"""

from compgen.runtime.baremetal.emitter import BaremetalEmitter

__all__ = ["BaremetalEmitter"]
