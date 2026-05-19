# Autonomous QNN scheduling proof — final report

_Generated 2026-05-15T20:14:01.319786+00:00_

**Target:** schedule **12× DroNet** within YOLOv8n's measured makespan (305.5 ms) while every DroNet instance meets its 40 ms latency budget.

**Final verdict:** ✅ **PASS**

## Optimization arc

| round | granularity | action | predicted (ms) | measured (ms) | feasibility | deadlines |
|---:|---|---|---:|---:|---|---|
| 1 | whole_net | solve+execute+feedback | 354.9 | 254.8 | pass | 13/13 |
| 2 | whole_net | solve+execute+feedback | 304.8 | 350.9 | pass | 13/13 |
| 3 | whole_net | solve+execute+feedback | 356.7 | 255.6 | pass | 13/13 |
| 4 | whole_net | solve+execute+feedback | 305.5 | 257.3 | pass | 13/13 |

## Contention convergence

| round | CPU | DSP | converged? |
|---:|---:|---:|:--:|
| 1 | 1.259 | 0.859 | … |
| 2 | 1.225 | 1.005 | … |
| 3 | 1.234 | 0.861 | … |
| 4 | 1.228 | 0.852 | ✅ |

## Agent decision log

### Round 1 · solve+execute+feedback (whole_net)

pred 355ms/frame; per-lane wall (10 iters): CPU=1343ms, DSP=2548ms; per-iter CPU=134.3ms, DSP=254.8ms

### Round 2 · solve+execute+feedback (whole_net)

pred 305ms/frame; per-lane wall (10 iters): CPU=1325ms, DSP=3509ms; per-iter CPU=132.5ms, DSP=350.9ms

### Round 3 · solve+execute+feedback (whole_net)

pred 357ms/frame; per-lane wall (10 iters): CPU=1346ms, DSP=2556ms; per-iter CPU=134.6ms, DSP=255.6ms

### Round 4 · solve+execute+feedback (whole_net)

pred 305ms/frame; per-lane wall (10 iters): CPU=1335ms, DSP=2573ms; per-iter CPU=133.5ms, DSP=257.3ms

## MCP tool invocations

| tool | calls |
|---|---:|
| `closed_loop.steady_state` | 4 |
