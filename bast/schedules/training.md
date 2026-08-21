# XPU-RT Scheduling Heuristics

Authoritative, hand-maintained rules the LLM scheduler should follow.
`training_data.md` is an append-only log of prior outputs + self-analyses;
this file is the distilled knowledge.

## Hardware model

- Target SoC: SpacemiT x60 (RISC-V). Cores split by type:
  - `CPU_P#0..#3` — performance cores (RVV), fast on quantized matmul/conv.
  - `CPU_E#0..#4` — efficient cores (scalar), slower but plentiful.
- Shared memory → transfer times between cores are zero unless stated.
- A dispatch may be assigned to a single core, a pair of same-type cores,
  or the full set of a type (parallel execution).

## Objective

Minimise the makespan of each periodic window while honouring:
1. Dispatch dependency order.
2. Core exclusivity (one dispatch per core at a time).
3. Per-network deadlines (`window_duration` from the config).

## Heuristics (apply in order)

1. **Critical path first.** Dispatches on the longest dependency chain go
   first and prefer the fastest available core (usually `CPU_P`).
2. **Pack the P cores.** Keep `CPU_P#0..#3` fully loaded with compute-heavy
   dispatches before assigning work to E cores.
3. **Offload light tails to E cores.** Short, dependency-free dispatches
   (`duration < 0.3 ms` on P) go to `CPU_E` to free P cores.
4. **Use pairs only when beneficial.** A pair of same-type cores helps only
   when the dispatch's profiled pair time is < 0.6× its single-core time.
   Otherwise a pair wastes a core.
5. **Respect periodicity.** For a network with `period` and `window_duration`,
   all its dispatches must finish within `window_duration` of the window start.
6. **Break ties by fan-out.** When two ready dispatches compete, schedule the
   one with more descendants first.

## Output format (STRICT JSON, no prose)

```json
{
  "dispatches": {
    "<dispatch_name>": {
      "id": <int>,
      "hardware_target": "CPU_P#0" | "CPU_E#2" | "CPU_P#0+CPU_P#1",
      "start_time": <float ms>,
      "duration": <float ms>,
      "dependencies": ["<dispatch_name>", ...],
      "job_name": "<network_name>"
    }
  },
  "metadata": {
    "makespan": <float ms>,
    "machines": ["CPU_P", "CPU_E"]
  }
}
```

Rules:
- Every dispatch from the input must appear exactly once.
- `start_time >= max(dep.start_time + dep.duration for dep in dependencies)`.
- No two dispatches on the same `hardware_target` overlap.

## Self-analysis checklist (for each generated schedule)

After producing a schedule, write a short entry to `training_data.md`:
- What makespan was achieved vs. the naive sum of durations?
- Which heuristic paid off most?
- Any heuristic that seemed wrong on this workload (propose amendment)?
- One sentence on what to try differently next time.
