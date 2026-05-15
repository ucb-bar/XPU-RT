# Runtime Model: Host-Device Execution

This document describes CompGen's runtime model for executing compiled models on heterogeneous systems (CPU host + accelerator device).

## Architecture

```
PyTorch Model
    ↓  CompGen Pipeline
┌─────────────────────────┐
│  ModelProgram            │
│  ├── host_kernels (CPU)  │
│  ├── device_kernels (NPU)│
│  ├── execution_order     │
│  ├── data_transfers      │
│  └── memory_layout       │
└────────┬────────────────┘
         ↓  BaremetalEmitter
┌─────────────────────────┐
│  Generated C Project    │
│  ├── main.c             │  ← Dispatch loop
│  ├── npu_driver.h/c     │  ← Device API (scaffold + target impl)
│  ├── memory_map.h       │  ← Buffer addresses
│  ├── weights.h/c        │  ← Weight data
│  ├── kernels/*.S         │  ← NPU kernel code
│  ├── linker.ld          │  ← Memory layout (chipyard-compatible)
│  └── Makefile           │  ← Cross-compilation
└────────┬────────────────┘
         ↓  riscv64-unknown-elf-gcc
┌─────────────────────────┐
│  Binary (.elf / .bin)   │
│  Runs on:               │
│  ├── Chipyard Verilator │
│  ├── FireSim FPGA       │
│  └── NPU Simulator      │
└─────────────────────────┘
```

## Multi-Device Execution

### Host-Device Model

The CPU host orchestrates the execution:

1. **Initialize** — Set up NPU hardware, DMA channels
2. **Load weights** — DMA weight tensors from host DRAM to device DRAM
3. **Execute model** — For each layer:
   - If CPU op (embedding, control flow): execute on host
   - If NPU op (matmul, attention): dispatch to NPU, wait
   - If device switch: DMA transfer intermediate results
4. **Collect output** — DMA results back to host

### Memory Hierarchy

```
Host CPU                          NPU Device
┌──────────┐                     ┌──────────────┐
│ Host RAM │ ←── DMA ──────────→ │ DRAM (16 GiB)│
│          │                     │     ↕ DMA     │
└──────────┘                     │ VMEM (1 MiB)  │
                                 │     ↕ VLoad   │
                                 │ Tensor Regs   │
                                 │ (64 × 1KB)    │
                                 │     ↕         │
                                 │ MXU Weights   │
                                 │ MXU Accumulators│
                                 └──────────────┘
```

### Data Transfer

Transfers between CPU and NPU go through DMA:
- **CPU → NPU**: Weight loading, input data
- **NPU → CPU**: Output results, intermediate results for CPU ops
- **NPU internal**: DRAM ↔ VMEM (8 DMA channels, async)

## Deployment Options

### 1. Bare Metal (chipyard)

Raw C program with polling loop. Suitable for simulation and simple benchmarks.

- Cross-compiler: `riscv64-unknown-elf-gcc`
- Linker: HTIF-compatible (tohost/fromhost for test pass/fail)
- Base address: `0x80000000`

### 2. Zephyr RTOS (chipyard)

Zephyr application with threads and semaphores. Better for:
- Multi-device coordination (CPU thread + NPU dispatch thread)
- Logging and profiling (Zephyr logging subsystem)
- Production deployment

Templates in `targetgen/runtime_templates/zephyr/`.

### 3. NPU Simulator

Direct execution on `third_party/npu_model/`. Produces Python `Program` objects.

## Extension Points

| Component | Location | Purpose |
|-----------|----------|---------|
| `ProgramBuilder` | `runtime/program_builder.py` | Scaffold: assembles kernels into programs |
| `MemoryPlanner` | `runtime/memory_layout.py` | Scaffold: plans buffer allocations |
| `BaremetalEmitter` | `runtime/baremetal/emitter.py` | Scaffold: generates C code |
| `chipyard.py` | `runtime/baremetal/chipyard.py` | Scaffold: chipyard-specific helpers |
| `NpuProgramEmitter` | `targets/backends/npu/` | Extension: NPU-specific emission |
| `npu_driver.c` | Generated | Extension: target fills in real hardware impl |
