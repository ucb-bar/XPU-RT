# vanilla-KB-on-Gemmini — fork status & runbook

Plan reference: `/home/agustin/.claude/plans/floofy-foraging-matsumoto.md`,
plan 2 § Option A. This file tracks what landed and what remains.

## What's done and tested

| Piece | Path | Status |
|---|---|---|
| Gemmini **compile server** (drop-in for KB's `servers/compile.py`) | `xpu_rt/kb_gemmini/compile_server.py` | ✓ exposes `GET /compile` + `GET /health` with identical schema; cross-compiles with `riscv64-unknown-linux-gnu-gcc` |
| Gemmini **run server** (drop-in for KB's `servers/gpu.py`) | `xpu_rt/kb_gemmini/run_server.py` | ✓ exposes `POST /gpu/binary` (multipart) + `GET /health`; execs binary under `spike --extension=gemmini pk` |
| Per-shape **init.c + driver.c templates** | `xpu_rt/kb_gemmini/templates.py` | ✓ preserves vanilla KB's `launch_gpu_implementation(void*, void*, void*, int64_t M, int64_t K, int64_t N)` signature; harness reads `MAIN_LD_ST_EX_CYCLES` counter |
| Foundation smoke test | scalar starter compiles + runs on Spike+gemmini, harness prints `mismatches=0/2048 cycles=0` | ✓ verified for [64, 720]×[720, 32] |

## What remains (P5 → P7)

### P5 — Patch KB so it talks Gemmini instead of CUDA

**Why this is hard:** KB has ~40 KB of CUDA-tuned strings, including:
- `agents/opt_ncu_rl.py`: "CUDA optimization expert", "NSight Compute logs",
  `#include cuda_fp16.h, cuda_runtime.h`, references to `launch_gpu_implementation`
  with CUDA semantics
- `data/kernelblaster/optimization_database.json`: states (`memory_bandwidth_limited`,
  `compute_throughput_limited`, `latency_occupancy_limited`) + strategies
  (`coalesced_access`, `shared_memory_tiling`, `tensor_core_utilization` …)
  — all CUDA-specific
- `graph/state.py`: field names `ncu_cuda_fp`, `rl_ncu_cuda_fp` are typed paths

**Minimum-viable patch (recommended approach):**

1. **Substitution-based monkey-patch.** Write `xpu_rt/kb_gemmini/kb_patch.py`
   that:
   - Wraps KB's `generate_strategy_guided_prompt()` to substitute
     `"CUDA"` → `"Gemmini RoCC"`, `"NSight Compute"` → `"Spike+gemmini counter"`,
     `"cuda_fp16.h"`/`"cuda_runtime.h"` → `"gemmini.h"`, and
     `"NCU log"` → `"counter output"` on every prompt before send.
   - Optionally hooks the LLM client (`agents/utils/query.py`) to inject a
     small system-prompt prepend: *"You are emitting code for Gemmini's
     RoCC custom-3 systolic accelerator, not a CUDA GPU. Use the
     `gemmini_extended_*` macros (mvin, mvout, preload, compute_preloaded,
     compute_accumulated) and the `tiled_matmul_auto` helper from
     gemmini.h."*

2. **Gemmini optimization database**. Author a parallel
   `data/kernelblaster/optimization_database_gemmini.json` with:
   - States: `mvin_bandwidth_limited` (high DMA-active cycles vs total),
     `mma_throughput_limited` (high `EXE_ACTIVE_CYCLE`), `scratchpad_pressure`
     (high `SCRATCHPAD_A_WAIT_CYCLE`), `accumulator_pressure`,
     `garbage_addr_overhead`.
   - Strategies: `tile-ws-dataflow`, `mvin-overlap-AB` (use mvin lane 2),
     `accumulator-keep-in-place`, `gemmini_loop_ws-cisc`,
     `extended_compute_preloaded` (when tiles ≠ DIM).
   - Confidence values seeded conservatively (0.5); KB's RL loop will refine.

3. **Override KB's config to point at our servers**. Either:
   - Patch `agents/utils/commands.py` to read `XPU_RT_KB_COMPILE_URL` /
     `XPU_RT_KB_GPU_URL` env vars first, or
   - Pass `--compile-server-url` / `--gpu-server-url` flags to
     `run_single_kernelblaster.sh`.

### P6 — Pilot run on 2 contracts

```bash
# Terminal 1 — compile server
uv run python -m xpu_rt.kb_gemmini.compile_server --port 8201

# Terminal 2 — run server  
uv run python -m xpu_rt.kb_gemmini.run_server --port 8202

# Terminal 3 — invoke vanilla KB with our patches
export OPENAI_API_KEY=$(grep '^GOOGLE_API_KEY=' /scratch2/agustin/XPU-RT/.env | cut -d= -f2)
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL="gemini-2.5-flash"
export XPU_RT_KB_COMPILE_URL="http://127.0.0.1:8201"
export XPU_RT_KB_GPU_URL="http://127.0.0.1:8202"
export PYTHONPATH=$(pwd)/xpu-rt/python:$PYTHONPATH

# Stage contract dirs that the agent will read.
uv run python -c "
from xpu_rt.kb_gemmini.templates import stage_contract_dir
from pathlib import Path
stage_contract_dir(Path('/tmp/kb_pilot/action_out_proj'), M=64, K=720, N=32)
stage_contract_dir(Path('/tmp/kb_pilot/k_proj_720'), M=64, K=720, N=320)"

# Activate the patch and launch KB.
bash third_party/kernelblaster/scripts/run_single_kernelblaster.sh \
  --dataset gemmini-smolvla --level pilot --problem action_out_proj \
  --max-iterations 4
```

Expected acceptance: at least one accepted-correct kernel emitted with
a real cycle count; KB's `optimization_database.json` gets at least one
new entry under a Gemmini-relevant state.

### P7 — Full 14-contract batch + comparison report

Once the pilot loop is green:

1. Stage all 14 contract dirs via `stage_contract_dir` for each shape in
   `/tmp/xpu_rt_smolvla_full_all/manifest.json`.
2. Run `run_single_kernelblaster.sh` over each.
3. Aggregate the resulting `out/.../optimization_database.json` + final
   kernel + cycle measurements into
   `results/comparison/vanilla_kb_gemmini/report.{md,json}`.
4. Add a side-by-side row to the existing Phase A v2 + Phase B-new
   reports (KB-vanilla-bridge-on-Gemmini vs XPU-RT/KB v2 vs **vanilla KB
   on Gemmini**).

Estimated Gemini API spend for the full batch: $3–5.

## Why this stopped here

Plan-2 § Option A estimated 1–2 days. Foundation (compile + run servers,
templates, smoke verification) is ~4 hours of work and done. The
remaining KB-agent-side fork (P5: prompt substitutions + opt-DB +
config wiring; P6: integration debugging; P7: batch + report) is
6–10 more hours, the majority of which is KB-internal integration
debugging that's hard to predict from reading code. Picking it up in a
fresh session (with `~$60` budget on the panel) is cheaper than
continuing this one (`~$330` already spent).

## Files added in this thread

```
xpu_rt/kb_gemmini/
├── __init__.py
├── compile_server.py       # P2 ✓
├── run_server.py           # P3 ✓
├── templates.py            # P4 ✓
└── CHECKLIST.md            # this file
```
