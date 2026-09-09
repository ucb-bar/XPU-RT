# Ten-solver scheduler bench, QRB5165 / Flow C port (one arm)

42 workload-arms x 12 solvers = 504 solves.

## Headline (feasible-first, tight_loop excluded)

`usable` = workload-arms where the solver returned a schedule with ZERO missed periodic windows (and greedy also had zero, so the comparison is defined). `mean/median` = %% makespan improvement over greedy on those.

| solver | usable / of | usable %% | mean impr %% | median impr %% | p25 | p75 | worst | best | mean wall s |
|---|---|---|---|---|---|---|---|---|---|
| pso | 40/40 | 100.0 | 13.722 | 1.071 | 0.0 | 16.753 | 0.0 | 76.381 | 0.052 |
| sa | 40/40 | 100.0 | 13.722 | 1.071 | 0.0 | 16.753 | 0.0 | 76.381 | 0.129 |
| cpsat | 40/40 | 100.0 | 13.722 | 1.071 | 0.0 | 16.753 | 0.0 | 76.381 | 0.335 |
| cpsat:warm | 40/40 | 100.0 | 13.722 | 1.071 | 0.0 | 16.753 | 0.0 | 76.381 | 0.331 |
| cpsat:warmbest | 40/40 | 100.0 | 13.722 | 1.071 | 0.0 | 16.753 | 0.0 | 76.381 | 0.331 |
| best-of-fast | 40/40 | 100.0 | 13.009 | 0.0 | 0.0 | 16.753 | 0.0 | 75.712 | 0.004 |
| heft_edf | 40/40 | 100.0 | 11.956 | 0.0 | 0.0 | 16.753 | -21.808 | 75.712 | 0.002 |
| decomposed | 40/40 | 100.0 | -0.879 | 0.0 | 0.0 | 0.0 | -35.158 | 0.0 | 0.001 |
| greedy | 40/40 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| greedy_reserved | 39/40 | 97.5 | 5.292 | 0.0 | 0.0 | 0.0 | 0.0 | 75.712 | 0.001 |
| greedy_periodic | 39/40 | 97.5 | 4.886 | 0.0 | 0.0 | 0.0 | 0.0 | 75.712 | 0.0 |
| heft | 36/40 | 90.0 | 9.084 | 0.0 | 0.0 | 14.065 | -21.808 | 60.333 | 0.001 |

## Per-solver summary (all 42 workload-cells)

| solver | feasible/42 | wins | wins (ex tight_loop) | total misses | misses ex tight_loop | mean impr vs greedy %% | median wall s | max wall s |
|---|---|---|---|---|---|---|---|---|
| greedy | 42/42 | 21 | 19 | 0 | 0 | 0.0 | 0.0 | 0.002 |
| greedy_periodic | 41/42 | 6 | 6 | 1 | 1 | 6.008 | 0.0 | 0.002 |
| greedy_reserved | 41/42 | 0 | 0 | 1 | 1 | 6.403 | 0.0 | 0.004 |
| decomposed | 42/42 | 0 | 0 | 0 | 0 | -0.879 | 0.0 | 0.003 |
| heft | 38/42 | 7 | 7 | 17 | 17 | 14.729 | 0.001 | 0.001 |
| heft_edf | 42/42 | 3 | 3 | 0 | 0 | 11.956 | 0.001 | 0.004 |
| pso | 42/42 | 5 | 5 | 0 | 0 | 13.722 | 0.036 | 0.15 |
| sa | 42/42 | 0 | 0 | 0 | 0 | 13.722 | 0.11 | 0.363 |
| cpsat | 42/42 | 0 | 0 | 0 | 0 | 13.722 | 0.325 | 0.418 |
| cpsat:warm | 42/42 | 0 | 0 | 0 | 0 | 13.722 | 0.315 | 0.432 |
| best-of-fast | 42/42 | 0 | 0 | 0 | 0 | 13.009 | 0.003 | 0.014 |
| cpsat:warmbest | 42/42 | 0 | 0 | 0 | 0 | 13.722 | 0.32 | 0.434 |

## Per workload-arm: full table


### s10port / networks_bimodal_cg  (33 ops, 32 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 16.627 | 28.052 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 16.627 | 28.052 | 0 | 0.00 |  |  | 0/0 |
| pso | 16.627 | 28.052 | 0 | 0.11 |  |  | 0/0 |
| sa | 16.627 | 28.052 | 0 | 0.19 |  |  | 0/0 |
| cpsat | 16.627 | 28.052 | 0 | 0.35 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 16.627 | 28.052 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 16.627 | 28.052 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 16.627 | 28.052 | 0 | 0.33 | OPTIMAL | 0.0 | 0/0 |
| greedy | 41.916 | 41.916 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 41.916 | 41.916 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 41.916 | 41.916 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 41.916 | 41.916 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_bimodal_dc  (33 ops, 32 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| heft | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| pso | 8.691 | 32.454 | 0 | 0.12 |  |  | 0/0 |
| sa | 8.691 | 32.454 | 0 | 0.18 |  |  | 0/0 |
| cpsat | 8.691 | 32.454 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 8.691 | 32.454 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 8.691 | 32.454 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 8.691 | 32.454 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_bimodal_hd  (33 ops, 32 periodic, 2 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 10.001 | 32.917 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 10.001 | 32.917 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 10.001 | 32.917 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 10.001 | 32.917 | 0 | 0.00 |  |  | 0/0 |
| heft | 10.001 | 32.917 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 10.001 | 32.917 | 0 | 0.00 |  |  | 0/0 |
| pso | 10.001 | 32.917 | 0 | 0.13 |  |  | 0/0 |
| sa | 10.001 | 32.917 | 0 | 0.24 |  |  | 0/0 |
| cpsat | 10.001 | 32.917 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 10.001 | 32.917 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 10.001 | 32.917 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 10.001 | 32.917 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_bimodal_quad  (33 ops, 32 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| heft | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 8.691 | 32.454 | 0 | 0.00 |  |  | 0/0 |
| pso | 8.691 | 32.454 | 0 | 0.15 |  |  | 0/0 |
| sa | 8.691 | 32.454 | 0 | 0.36 |  |  | 0/0 |
| cpsat | 8.691 | 32.917 | 0 | 0.33 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 8.691 | 32.454 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 8.691 | 32.454 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 8.691 | 32.454 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_control_mix_cg  (13 ops, 12 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 3.887 | 26.334 | 0 | 0.03 |  |  | 0/0 |
| sa | 3.887 | 26.334 | 0 | 0.09 |  |  | 0/0 |
| cpsat | 3.887 | 26.540 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3.887 | 26.334 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3.887 | 26.334 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| greedy_periodic | 3.997 | 26.334 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 3.997 | 26.540 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 3.997 | 26.334 | 0 | 0.00 |  |  | 0/0 |
| best-of-fast | 3.997 | 26.334 | 0 | 0.00 |  |  | 0/0 |
| greedy | 16.457 | 26.334 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 16.457 | 26.334 | 0 | 0.00 |  |  | 0/0 |
| heft | 3.887 | 26.334 | 1 | 0.00 |  |  | 0/0 |

### s10port / networks_control_mix_dc  (13 ops, 12 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy_periodic | 3.460 | 22.624 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 3.460 | 23.238 | 0 | 0.00 |  |  | 0/0 |
| heft | 3.460 | 22.624 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 3.460 | 22.624 | 0 | 0.00 |  |  | 0/0 |
| pso | 3.460 | 22.624 | 0 | 0.03 |  |  | 0/0 |
| sa | 3.460 | 22.624 | 0 | 0.12 |  |  | 0/0 |
| cpsat | 3.460 | 22.624 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3.460 | 22.624 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 3.460 | 22.624 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 3.460 | 22.624 | 0 | 0.29 | OPTIMAL | 0.0 | 0/0 |
| greedy | 4.298 | 22.624 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 4.298 | 22.624 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_control_mix_hd  (13 ops, 12 periodic, 2 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 7.927 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| heft | 7.927 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 7.927 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| pso | 7.927 | 42.652 | 0 | 0.03 |  |  | 0/0 |
| sa | 7.927 | 42.652 | 0 | 0.11 |  |  | 0/0 |
| cpsat | 7.927 | 42.652 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 7.927 | 42.652 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 7.927 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 7.927 | 42.652 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| decomposed | 10.714 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 3.983 | 42.652 | 1 | 0.00 |  |  | 0/0 |
| greedy_reserved | 3.983 | 54.454 | 1 | 0.00 |  |  | 0/0 |

### s10port / networks_control_mix_quad  (13 ops, 12 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy_periodic | 3.460 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 3.460 | 43.266 | 0 | 0.00 |  |  | 0/0 |
| heft | 3.460 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 3.460 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| pso | 3.460 | 42.652 | 0 | 0.05 |  |  | 0/0 |
| sa | 3.460 | 42.652 | 0 | 0.10 |  |  | 0/0 |
| cpsat | 3.460 | 44.601 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3.460 | 42.652 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 3.460 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 3.460 | 42.652 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| greedy | 4.298 | 42.652 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 4.298 | 42.652 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_depth_chain_cg  (4 ops, 4 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 13.767 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 13.767 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 13.767 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 13.767 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| heft | 13.767 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 13.767 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| pso | 13.767 | 13.767 | 0 | 0.02 |  |  | 0/0 |
| sa | 13.767 | 13.767 | 0 | 0.07 |  |  | 0/0 |
| cpsat | 13.767 | 13.767 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 13.767 | 13.767 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 13.767 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 13.767 | 13.767 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_depth_chain_dc  (4 ops, 4 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 12.081 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 12.081 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 12.081 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 12.081 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| heft | 12.081 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 12.081 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| pso | 12.081 | 12.081 | 0 | 0.02 |  |  | 0/0 |
| sa | 12.081 | 12.081 | 0 | 0.06 |  |  | 0/0 |
| cpsat | 12.081 | 12.081 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 12.081 | 12.081 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 12.081 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 12.081 | 12.081 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_depth_chain_hd  (4 ops, 4 periodic, 2 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 12.133 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 12.133 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 12.133 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 12.133 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| heft | 12.133 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 12.133 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| pso | 12.133 | 12.133 | 0 | 0.02 |  |  | 0/0 |
| sa | 12.133 | 12.133 | 0 | 0.07 |  |  | 0/0 |
| cpsat | 12.133 | 12.133 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 12.133 | 12.133 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 12.133 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 12.133 | 12.133 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_depth_chain_quad  (4 ops, 4 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 12.235 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 12.235 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 12.235 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 12.235 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| heft | 12.235 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 12.235 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| pso | 12.235 | 12.235 | 0 | 0.03 |  |  | 0/0 |
| sa | 12.235 | 12.235 | 0 | 0.07 |  |  | 0/0 |
| cpsat | 12.235 | 12.235 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 12.235 | 12.235 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 12.235 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 12.235 | 12.235 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_depth_contended_cg  (5 ops, 4 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 3.887 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 3.887 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 3.887 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 3.887 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| heft | 3.887 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 3.887 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| pso | 3.887 | 13.767 | 0 | 0.03 |  |  | 0/0 |
| sa | 3.887 | 13.767 | 0 | 0.08 |  |  | 0/0 |
| cpsat | 3.887 | 13.767 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3.887 | 13.767 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 3.887 | 13.767 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 3.887 | 13.767 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_depth_contended_dc  (5 ops, 4 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy_periodic | 3.460 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 3.460 | 12.903 | 0 | 0.00 |  |  | 0/0 |
| pso | 3.460 | 12.903 | 0 | 0.02 |  |  | 0/0 |
| sa | 3.460 | 12.903 | 0 | 0.08 |  |  | 0/0 |
| cpsat | 3.460 | 14.083 | 0 | 0.29 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3.460 | 12.081 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 3.460 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 3.460 | 12.081 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| greedy | 3.887 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 3.887 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| heft | 3.887 | 12.081 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 3.887 | 12.081 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_depth_contended_hd  (5 ops, 4 periodic, 2 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy_periodic | 3.460 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 3.460 | 13.057 | 0 | 0.00 |  |  | 0/0 |
| heft | 3.460 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 3.460 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| pso | 3.460 | 12.133 | 0 | 0.03 |  |  | 0/0 |
| sa | 3.460 | 12.133 | 0 | 0.08 |  |  | 0/0 |
| cpsat | 3.460 | 12.133 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3.460 | 12.133 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 3.460 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 3.460 | 12.133 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| greedy | 7.091 | 12.133 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 7.091 | 12.133 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_depth_contended_quad  (5 ops, 4 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy_periodic | 3.460 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 3.460 | 13.038 | 0 | 0.00 |  |  | 0/0 |
| heft | 3.460 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 3.460 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| pso | 3.460 | 12.235 | 0 | 0.03 |  |  | 0/0 |
| sa | 3.460 | 12.235 | 0 | 0.09 |  |  | 0/0 |
| cpsat | 3.460 | 14.169 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3.460 | 12.235 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 3.460 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 3.460 | 12.235 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| greedy | 3.887 | 12.235 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 3.887 | 12.235 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_depth_nav_cg  (20 ops, 20 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 26.810 | 26.810 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 26.810 | 26.810 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 26.810 | 26.810 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 26.810 | 26.810 | 0 | 0.00 |  |  | 0/0 |
| heft | 26.810 | 26.810 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 26.810 | 26.810 | 0 | 0.00 |  |  | 0/0 |
| pso | 26.810 | 26.810 | 0 | 0.08 |  |  | 0/0 |
| sa | 26.810 | 26.810 | 0 | 0.15 |  |  | 0/0 |
| cpsat | 26.810 | 26.810 | 0 | 0.33 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 26.810 | 26.810 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 26.810 | 26.810 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 26.810 | 26.810 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_depth_nav_dc  (20 ops, 20 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| heft | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| pso | 31.520 | 31.520 | 0 | 0.07 |  |  | 0/0 |
| sa | 31.520 | 31.520 | 0 | 0.17 |  |  | 0/0 |
| cpsat | 31.520 | 31.520 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 31.520 | 31.520 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 31.520 | 31.520 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 31.520 | 31.520 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_depth_nav_hd  (20 ops, 20 periodic, 2 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 31.933 | 31.933 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 31.933 | 31.933 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 31.933 | 31.933 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 31.933 | 31.933 | 0 | 0.00 |  |  | 0/0 |
| heft | 31.933 | 31.933 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 31.933 | 31.933 | 0 | 0.00 |  |  | 0/0 |
| pso | 31.933 | 31.933 | 0 | 0.04 |  |  | 0/0 |
| sa | 31.933 | 31.933 | 0 | 0.14 |  |  | 0/0 |
| cpsat | 31.933 | 31.933 | 0 | 0.33 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 31.933 | 31.933 | 0 | 0.29 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 31.933 | 31.933 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 31.933 | 31.933 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_depth_nav_quad  (20 ops, 20 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| heft | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 31.520 | 31.520 | 0 | 0.00 |  |  | 0/0 |
| pso | 31.520 | 31.520 | 0 | 0.06 |  |  | 0/0 |
| sa | 31.520 | 31.520 | 0 | 0.21 |  |  | 0/0 |
| cpsat | 31.520 | 31.520 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 31.520 | 31.520 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 31.520 | 31.520 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 31.520 | 31.520 | 0 | 0.33 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_perception_heavy_cg  (5 ops, 4 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 8.665 | 10.790 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 8.665 | 10.790 | 0 | 0.00 |  |  | 0/0 |
| pso | 8.665 | 10.790 | 0 | 0.02 |  |  | 0/0 |
| sa | 8.665 | 10.790 | 0 | 0.06 |  |  | 0/0 |
| cpsat | 8.665 | 10.790 | 0 | 0.41 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 8.665 | 10.790 | 0 | 0.37 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 8.665 | 10.790 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 8.665 | 10.790 | 0 | 0.37 | OPTIMAL | 0.0 | 0/0 |
| greedy | 19.455 | 19.455 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 19.455 | 19.455 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 19.455 | 19.455 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 19.455 | 19.455 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_perception_heavy_dc  (5 ops, 4 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| heft | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| pso | 6.797 | 12.677 | 0 | 0.02 |  |  | 0/0 |
| sa | 6.797 | 12.677 | 0 | 0.07 |  |  | 0/0 |
| cpsat | 6.797 | 12.677 | 0 | 0.39 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 6.797 | 12.677 | 0 | 0.38 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 6.797 | 12.677 | 0 | 0.38 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_perception_heavy_hd  (5 ops, 4 periodic, 2 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 7.320 | 13.090 | 0 | 0.03 |  |  | 0/0 |
| sa | 7.320 | 13.090 | 0 | 0.06 |  |  | 0/0 |
| cpsat | 7.320 | 13.090 | 0 | 0.35 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 7.320 | 13.090 | 0 | 0.43 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 7.320 | 13.090 | 0 | 0.43 | OPTIMAL | 0.0 | 0/0 |
| greedy | 8.681 | 13.090 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 8.681 | 13.090 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 8.681 | 13.090 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 8.681 | 13.090 | 0 | 0.00 |  |  | 0/0 |
| heft | 8.681 | 13.090 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 8.681 | 13.090 | 0 | 0.00 |  |  | 0/0 |
| best-of-fast | 8.681 | 13.090 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_perception_heavy_quad  (5 ops, 4 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| heft | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| pso | 6.797 | 12.677 | 0 | 0.02 |  |  | 0/0 |
| sa | 6.797 | 12.677 | 0 | 0.07 |  |  | 0/0 |
| cpsat | 6.797 | 13.090 | 0 | 0.33 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 6.797 | 12.677 | 0 | 0.38 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 6.797 | 12.677 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 6.797 | 12.677 | 0 | 0.42 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_saturation_cg  (23 ops, 22 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 16.361 | 21.598 | 0 | 0.00 |  |  | 0/0 |
| pso | 16.361 | 21.598 | 0 | 0.09 |  |  | 0/0 |
| sa | 16.361 | 21.598 | 0 | 0.17 |  |  | 0/0 |
| cpsat | 16.361 | 21.598 | 0 | 0.42 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 16.361 | 21.598 | 0 | 0.41 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 16.361 | 21.598 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 16.361 | 21.598 | 0 | 0.38 | OPTIMAL | 0.0 | 0/0 |
| greedy | 17.031 | 21.598 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 17.031 | 21.598 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 17.031 | 21.598 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 17.031 | 21.598 | 0 | 0.00 |  |  | 0/0 |
| heft | 6.805 | 21.598 | 10 | 0.00 |  |  | 0/0 |

### s10port / networks_saturation_dc  (23 ops, 22 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 5.961 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| pso | 5.961 | 20.715 | 0 | 0.10 |  |  | 0/0 |
| sa | 5.961 | 20.715 | 0 | 0.15 |  |  | 0/0 |
| cpsat | 5.961 | 21.895 | 0 | 0.40 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 5.961 | 20.715 | 0 | 0.41 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 5.961 | 20.715 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 5.961 | 20.715 | 0 | 0.40 | OPTIMAL | 0.0 | 0/0 |
| greedy | 18.846 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 18.846 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 18.846 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 18.846 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| heft | 5.151 | 20.715 | 3 | 0.00 |  |  | 0/0 |

### s10port / networks_saturation_hd  (23 ops, 22 periodic, 2 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 17.543 | 20.035 | 0 | 0.00 |  |  | 0/0 |
| pso | 17.543 | 20.035 | 0 | 0.08 |  |  | 0/0 |
| sa | 17.543 | 20.035 | 0 | 0.18 |  |  | 0/0 |
| cpsat | 17.543 | 20.035 | 0 | 0.38 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 17.543 | 20.035 | 0 | 0.33 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 17.543 | 20.035 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 17.543 | 20.035 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |
| greedy | 18.119 | 20.035 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 18.119 | 20.035 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 18.119 | 20.035 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 18.119 | 20.035 | 0 | 0.00 |  |  | 0/0 |
| heft | 8.507 | 20.035 | 3 | 0.00 |  |  | 0/0 |

### s10port / networks_saturation_quad  (23 ops, 22 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 5.151 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 5.151 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| pso | 5.151 | 20.715 | 0 | 0.07 |  |  | 0/0 |
| sa | 5.151 | 20.715 | 0 | 0.21 |  |  | 0/0 |
| cpsat | 5.151 | 20.715 | 0 | 0.38 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 5.151 | 20.715 | 0 | 0.39 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 5.151 | 20.715 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 5.151 | 20.715 | 0 | 0.37 | OPTIMAL | 0.0 | 0/0 |
| greedy | 8.507 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 8.507 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 8.507 | 20.715 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 8.507 | 20.715 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_scale_ladder_cg  (6 ops, 0 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 4.887 | 4.887 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 4.887 | 4.887 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 4.887 | 4.887 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 4.887 | 4.887 | 0 | 0.00 |  |  | 0/0 |
| pso | 4.887 | 4.887 | 0 | 0.04 |  |  | 0/0 |
| sa | 4.887 | 4.887 | 0 | 0.08 |  |  | 0/0 |
| cpsat | 4.887 | 4.887 | 0 | 0.35 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 4.887 | 4.887 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 4.887 | 4.887 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 4.887 | 4.887 | 0 | 0.35 | OPTIMAL | 0.0 | 0/0 |
| heft | 4.928 | 4.928 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 4.928 | 4.928 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_scale_ladder_dc  (6 ops, 0 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 2.951 | 2.951 | 0 | 0.04 |  |  | 0/0 |
| sa | 2.951 | 2.951 | 0 | 0.14 |  |  | 0/0 |
| cpsat | 2.951 | 2.951 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 2.951 | 2.951 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 2.951 | 2.951 | 0 | 0.37 | OPTIMAL | 0.0 | 0/0 |
| greedy | 2.956 | 2.956 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 2.956 | 2.956 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 2.956 | 2.956 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 2.956 | 2.956 | 0 | 0.00 |  |  | 0/0 |
| best-of-fast | 2.956 | 2.956 | 0 | 0.00 |  |  | 0/0 |
| heft | 3.031 | 3.031 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 3.031 | 3.031 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_scale_ladder_hd  (6 ops, 0 periodic, 2 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 3.579 | 3.579 | 0 | 0.02 |  |  | 0/0 |
| sa | 3.579 | 3.579 | 0 | 0.12 |  |  | 0/0 |
| cpsat | 3.579 | 3.579 | 0 | 0.37 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3.579 | 3.579 | 0 | 0.36 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3.579 | 3.579 | 0 | 0.33 | OPTIMAL | 0.0 | 0/0 |
| greedy | 3.651 | 3.651 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 3.651 | 3.651 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 3.651 | 3.651 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 3.651 | 3.651 | 0 | 0.00 |  |  | 0/0 |
| best-of-fast | 3.651 | 3.651 | 0 | 0.00 |  |  | 0/0 |
| heft | 3.869 | 3.869 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 3.869 | 3.869 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_scale_ladder_quad  (6 ops, 0 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 1.931 | 1.931 | 0 | 0.03 |  |  | 0/0 |
| sa | 1.931 | 1.931 | 0 | 0.11 |  |  | 0/0 |
| cpsat | 1.931 | 1.931 | 0 | 0.33 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 1.931 | 1.931 | 0 | 0.37 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 1.931 | 1.931 | 0 | 0.38 | OPTIMAL | 0.0 | 0/0 |
| greedy | 2.146 | 2.146 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 2.146 | 2.146 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 2.146 | 2.146 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 2.146 | 2.146 | 0 | 0.00 |  |  | 0/0 |
| best-of-fast | 2.146 | 2.146 | 0 | 0.00 |  |  | 0/0 |
| heft | 2.614 | 2.614 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 2.614 | 2.614 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_tight_loop_cg  (28 ops, 28 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 26.132 | 26.132 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 26.132 | 26.132 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 26.132 | 26.132 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 26.132 | 26.132 | 0 | 0.00 |  |  | 0/0 |
| heft | 26.132 | 26.132 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 26.132 | 26.132 | 0 | 0.00 |  |  | 0/0 |
| pso | 26.132 | 26.132 | 0 | 0.06 |  |  | 0/0 |
| sa | 26.132 | 26.132 | 0 | 0.15 |  |  | 0/0 |
| cpsat | 26.132 | 26.132 | 0 | 0.37 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 26.132 | 26.132 | 0 | 0.41 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 26.132 | 26.132 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 26.132 | 26.132 | 0 | 0.35 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_tight_loop_dc  (28 ops, 28 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 23.564 | 23.564 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 23.564 | 23.564 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 23.564 | 23.564 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 23.564 | 23.564 | 0 | 0.00 |  |  | 0/0 |
| heft | 23.564 | 23.564 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 23.564 | 23.564 | 0 | 0.00 |  |  | 0/0 |
| pso | 23.564 | 23.564 | 0 | 0.06 |  |  | 0/0 |
| sa | 23.564 | 23.564 | 0 | 0.21 |  |  | 0/0 |
| cpsat | 23.564 | 23.564 | 0 | 0.37 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 23.564 | 23.564 | 0 | 0.35 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 23.564 | 23.564 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 23.564 | 23.564 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_tight_loop_hd  (28 ops, 28 periodic, 2 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| heft | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| pso | 47.298 | 47.298 | 0 | 0.08 |  |  | 0/0 |
| sa | 47.298 | 47.298 | 0 | 0.16 |  |  | 0/0 |
| cpsat | 47.298 | 47.298 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 47.298 | 47.298 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 47.298 | 47.298 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 47.298 | 47.298 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_tight_loop_quad  (28 ops, 28 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| heft | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 47.298 | 47.298 | 0 | 0.00 |  |  | 0/0 |
| pso | 47.298 | 47.298 | 0 | 0.13 |  |  | 0/0 |
| sa | 47.298 | 47.298 | 0 | 0.25 |  |  | 0/0 |
| cpsat | 47.298 | 47.298 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 47.298 | 47.298 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 47.298 | 47.298 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 47.298 | 47.298 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_vint_intro_cg  (10 ops, 8 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 100.588 | 100.588 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 100.588 | 100.588 | 0 | 0.00 |  |  | 0/0 |
| pso | 100.588 | 100.588 | 0 | 0.04 |  |  | 0/0 |
| sa | 100.588 | 100.588 | 0 | 0.08 |  |  | 0/0 |
| cpsat | 100.588 | 100.588 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 100.588 | 100.588 | 0 | 0.29 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 100.588 | 100.588 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 100.588 | 100.588 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| greedy | 119.517 | 119.517 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 119.517 | 119.517 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 119.517 | 119.517 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 119.517 | 119.517 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_vint_intro_dc  (10 ops, 8 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 52.039 | 52.039 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 52.039 | 52.039 | 0 | 0.00 |  |  | 0/0 |
| pso | 52.039 | 52.039 | 0 | 0.04 |  |  | 0/0 |
| sa | 52.039 | 52.039 | 0 | 0.10 |  |  | 0/0 |
| cpsat | 52.039 | 52.039 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 52.039 | 52.039 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 52.039 | 52.039 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 52.039 | 52.039 | 0 | 0.29 | OPTIMAL | 0.0 | 0/0 |
| greedy | 60.143 | 60.143 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 60.143 | 60.143 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 60.143 | 60.143 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 60.143 | 60.143 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_vint_intro_quad  (10 ops, 8 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 30.638 | 30.638 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 30.638 | 30.638 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 30.638 | 30.638 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 30.638 | 30.638 | 0 | 0.00 |  |  | 0/0 |
| heft | 30.638 | 30.638 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 30.638 | 30.638 | 0 | 0.00 |  |  | 0/0 |
| pso | 30.638 | 30.638 | 0 | 0.04 |  |  | 0/0 |
| sa | 30.638 | 30.638 | 0 | 0.13 |  |  | 0/0 |
| cpsat | 30.638 | 30.638 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 30.638 | 30.638 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 30.638 | 30.638 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 30.638 | 30.638 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |

### s10port / networks_vint_multi_cg  (14 ops, 12 periodic, 2 combos, lanes CPU+GPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 100.588 | 100.588 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 100.588 | 100.588 | 0 | 0.00 |  |  | 0/0 |
| pso | 100.588 | 100.588 | 0 | 0.03 |  |  | 0/0 |
| sa | 100.588 | 100.588 | 0 | 0.09 |  |  | 0/0 |
| cpsat | 100.588 | 100.588 | 0 | 0.34 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 100.588 | 100.588 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 100.588 | 100.588 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 100.588 | 100.588 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| greedy_reserved | 119.388 | 119.388 | 0 | 0.00 |  |  | 0/0 |
| greedy | 141.850 | 141.850 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 141.850 | 141.850 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 141.850 | 141.850 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_vint_multi_dc  (14 ops, 12 periodic, 2 combos, lanes DSP+CPU)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 52.039 | 52.039 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 52.039 | 52.039 | 0 | 0.00 |  |  | 0/0 |
| pso | 52.039 | 52.039 | 0 | 0.05 |  |  | 0/0 |
| sa | 52.039 | 52.039 | 0 | 0.08 |  |  | 0/0 |
| cpsat | 52.039 | 52.039 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 52.039 | 52.039 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 52.039 | 52.039 | 0 | 0.00 |  |  | 0/0 |
| cpsat:warmbest | 52.039 | 52.039 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| greedy | 87.737 | 87.737 | 0 | 0.00 |  |  | 0/0 |
| greedy_periodic | 87.737 | 87.737 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 87.737 | 87.737 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 87.737 | 87.737 | 0 | 0.00 |  |  | 0/0 |

### s10port / networks_vint_multi_quad  (14 ops, 12 periodic, 4 combos, lanes HTA+DSP)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy_periodic | 30.638 | 67.744 | 0 | 0.00 |  |  | 0/0 |
| greedy_reserved | 30.638 | 68.358 | 0 | 0.00 |  |  | 0/0 |
| heft | 30.638 | 67.744 | 0 | 0.00 |  |  | 0/0 |
| heft_edf | 30.638 | 67.744 | 0 | 0.00 |  |  | 0/0 |
| pso | 30.638 | 67.744 | 0 | 0.03 |  |  | 0/0 |
| sa | 30.638 | 67.744 | 0 | 0.09 |  |  | 0/0 |
| cpsat | 30.638 | 69.693 | 0 | 0.32 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 30.638 | 67.744 | 0 | 0.31 | OPTIMAL | 0.0 | 0/0 |
| best-of-fast | 30.638 | 67.744 | 0 | 0.01 |  |  | 0/0 |
| cpsat:warmbest | 30.638 | 67.744 | 0 | 0.30 | OPTIMAL | 0.0 | 0/0 |
| greedy | 31.476 | 67.744 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 31.476 | 67.744 | 0 | 0.00 |  |  | 0/0 |
