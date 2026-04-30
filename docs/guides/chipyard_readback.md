# Chipyard host ↔ GPU readback

Two canonical patterns for getting result data out of a Chipyard-driven
guest (SoC, GPU, accelerator) and into Python on the host side.

## The problem

The fused-SoC convention used by `radiance-kernels`'s
`soc/fuse_rv32_into_rv64.sh` (and similar Chipyard flows) places kernel
data at **`0x110000000+`** (rv64 view) / **`0x10000000+`** (rv32 view).
The rv64 host CPU **cannot freely read** from `0x110000000+` — the only
TileLink slave at that range is the TSI harness
(`testchipip/src/main/scala/tsi/TSIHarness.scala`), and TSI **only
accepts `Put`** for LOADMEM-style writes — `Get` is unsupported. Any
host-side `volatile uint32_t *p; uint32_t v = *p;` against that region
fires:

```
TLMonitor 'A' channel carries Get type which slave claims it can't support
```

You have to either route the data out a different way (HTIF), or place
your shared region somewhere the host can `Get` (system DRAM at
`0x80000000+`).

## Pattern 1 — Stream results out via HTIF (easy, low bandwidth)

The HTIF (Host-Target Interface) ``__tohost`` register is already
plumbed for the rocket-chip pass/fail convention. The same channel can
carry a stream of result words:

- LSB=0 → data word; payload is `(val >> 1)`.
- LSB=1 → exit; code is `(val >> 1)`.

### Guest side (your kernel)

`compgen.runtime.baremetal.chipyard.htif_data_stream_c()` returns C
helpers ready to drop into your guest source:

```c
// inserted by htif_data_stream_c()
extern volatile uint64_t __tohost;
static inline void htif_emit_u32(uint32_t word);
static inline void htif_emit_bytes(const void *src, unsigned n_bytes);
```

Use them after your kernel produces results:

```c
float result[N];
compute_kernel(args, result);
htif_emit_bytes(result, sizeof(result));   // stream out
htif_exit(0);                              // terminate
```

### Host side (your verifier)

`compgen.runtime.baremetal.chipyard.parse_htif_data_stream(log)`
walks the sim log and concatenates every LSB=0 ``tohost`` payload
(little-endian) until the first LSB=1 (exit):

```python
from compgen.runtime.baremetal.chipyard import (
    parse_htif_data_stream,
    parse_htif_exit,
)

log = open("vcs.log").read()
data = parse_htif_data_stream(log)              # bytes
saw_exit, code = parse_htif_exit(log)           # (True, 0) on success

import numpy as np
result = np.frombuffer(data, dtype=np.float32)
```

**When to use:** small payloads (kilobytes). Each `htif_emit_u32`
costs one TileLink transaction; large arrays add minutes of sim time.

## Pattern 2 — Shared system DRAM region (high bandwidth)

For larger payloads, place a labelled region in system DRAM
(`0x80000000+`) — outside the upper-DRAM TSI region — so the host can
read it with a normal TileLink `Get` after the sim terminates.

### Guest side — linker fragment

`compgen.runtime.baremetal.chipyard.shared_dram_section(symbol, size)`
returns a linker-script fragment to drop inside your `SECTIONS { … }`
block:

```python
from compgen.runtime.baremetal.chipyard import shared_dram_section
print(shared_dram_section(symbol="compgen_results", size_bytes=0x10000))
```

```ld
.compgen_shared ALIGN(64) : {
    PROVIDE(compgen_results = .);
    . = . + 0x10000;
} > REGION_DRAM
```

In your guest C, reference the symbol:

```c
extern volatile float compgen_results[];

// after compute …
for (int i = 0; i < N; ++i) compgen_results[i] = result[i];
htif_exit(0);   // sim terminates; data persists in DRAM
```

### Host side

After the sim exits, dump the DRAM region from VCS / FireSim's
DRAM backing file (the `+permissive +loadmem=…` flow has a paired
dump option), or read it back via the same TSI channel using `Put`-
compatible read protocols depending on your sim harness.

For VCS specifically:

```bash
# Configure sim with a DRAM backing file (one-time):
make … run-binary CONFIG=… BINARY=… +mm_writeFile=dram.bin

# After sim exits, read the symbol's offset into dram.bin from
# the linker map and slice it out:
python -c "
import numpy as np
data = np.fromfile('dram.bin', dtype=np.uint8)
# offset of compgen_results from the linker map:
result = data[OFFSET:OFFSET+SIZE].view(np.float32)
print(result)
"
```

**When to use:** larger payloads where the per-word HTIF overhead is
prohibitive, or when you need the result available across multiple
sim runs without re-streaming.

## Picking between the two

| | HTIF stream-out | Shared DRAM |
|---|---|---|
| Setup | Drop in `htif_data_stream_c()` + parse log | Linker fragment + post-sim DRAM dump |
| Best size | < ~64 KB | > ~64 KB |
| Sim cost | ~1 cycle per 4 bytes (slow on big payloads) | ~free |
| Host read | Just parse log | Needs DRAM dump tool |
| Determinism | Streamed in payload order | Layout-dependent |

For most agentic loops (correctness checks on small tensors), Pattern
1 is the right default. Drop down to Pattern 2 only when streaming is
bottlenecking sim time.

## Worked example

Saturn OPU's `chipyard.py` shows the pattern end-to-end:

```python
from compgen.runtime.baremetal.chipyard import (
    htif_c_section,           # __tohost / __fromhost section
    htif_data_stream_c,       # emit helpers
    htif_pass_fail_c,         # TEST_PASS / TEST_FAIL macros
    parse_chipyard_finish,    # post-sim diagnostic
    parse_htif_data_stream,
    parse_htif_exit,
)

# Build-time: compose the guest C
guest_c = (
    htif_pass_fail_c()
    + htif_data_stream_c()
    + my_kernel_source
)

# Post-sim: parse + verify
log = (Path(bundle) / "sim.log").read_text()
finish = parse_chipyard_finish(log)
assert finish["finish_reason"] == "GPUResetAggregator", finish

saw_exit, code = parse_htif_exit(log)
assert saw_exit and code == 0

result_bytes = parse_htif_data_stream(log)
result = np.frombuffer(result_bytes, dtype=np.float32)
np.testing.assert_array_equal(result, expected)
```

This same pattern works for any Chipyard-derived target: Saturn OPU,
Gemmini, Muon GPU, vendor RoCC, etc.
