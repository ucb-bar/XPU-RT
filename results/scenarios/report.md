# M6 — Diagnostic scenarios

**Summary: 8 / 12 expectation checks passed.**

- schedulers: ['heft', 'critical_path', 'edf', 'fastest_device', 'fifo', 'round_robin', 'peft', 'min_min', 'max_min', 'simulated_annealing', 'mosek', 'cpsat']
- scenarios: ['vision_pipeline', 'sensor_fusion_diamond', 'multirate_periodic', 'memory_pressured_residual', 'tiny_op_quantized_chain', 'heterogeneous_parallel']

## Scoreboard (across all scenarios x metrics)

| scheduler | wins | losses |
|---|---:|---:|
| heft | 4 | 0 |
| critical_path | 6 | 3 |
| edf | 6 | 3 |
| fastest_device | 5 | 3 |
| fifo | 5 | 4 |
| round_robin | 6 | 10 |
| peft | 4 | 0 |
| min_min | 5 | 2 |
| max_min | 6 | 0 |
| simulated_annealing | 6 | 0 |
| mosek | 2 | 5 |
| cpsat | 12 | 2 |

## vision_pipeline

![composite](side_by_side/vision_pipeline.png)

| scheduler | makespan_us | misses | total_lateness | cross_device | n_ops |
|---|---:|---:|---:|---:|---:|
| heft | 490.0 | 0 | 0.0 | 9 | 14 |
| critical_path | 1190.0 | 1 | 290.0 | 5 | 14 |
| edf | 1190.0 | 1 | 290.0 | 5 | 14 |
| fastest_device | 515.0 | 0 | 0.0 | 7 | 14 |
| fifo | 1190.0 | 1 | 290.0 | 5 | 14 |
| round_robin | 1035.0 | 1 | 135.0 | 14 | 14 |
| peft | 490.0 | 0 | 0.0 | 9 | 14 |
| min_min | 515.0 | 0 | 0.0 | 7 | 14 |
| max_min | 540.0 | 0 | 0.0 | 7 | 14 |
| simulated_annealing | 490.0 | 0 | 0.0 | 9 | 14 |
| mosek | 695.0 | 0 | 0.0 | 7 | 14 |
| cpsat | 490.0 | 0 | 0.0 | 7 | 14 |

Expectation checks:
- PASS  metric `makespan_us` expected best in `['heft', 'cpsat', 'mosek', 'critical_path']` observed best `['heft', 'peft', 'simulated_annealing', 'cpsat']`
- PASS  metric `deadline_miss_count` expected best in `['edf', 'cpsat', 'mosek', 'heft']` observed best `['heft', 'fastest_device', 'peft', 'min_min', 'max_min', 'simulated_annealing', 'mosek', 'cpsat']`

## sensor_fusion_diamond

![composite](side_by_side/sensor_fusion_diamond.png)

| scheduler | makespan_us | misses | total_lateness | cross_device | n_ops |
|---|---:|---:|---:|---:|---:|
| heft | 480.0 | 0 | 0.0 | 3 | 16 |
| critical_path | 510.0 | 0 | 0.0 | 1 | 16 |
| edf | 510.0 | 0 | 0.0 | 1 | 16 |
| fastest_device | 510.0 | 0 | 0.0 | 3 | 16 |
| fifo | 510.0 | 0 | 0.0 | 1 | 16 |
| round_robin | 790.0 | 0 | 0.0 | 15 | 16 |
| peft | 480.0 | 0 | 0.0 | 3 | 16 |
| min_min | 510.0 | 0 | 0.0 | 3 | 16 |
| max_min | 510.0 | 0 | 0.0 | 3 | 16 |
| simulated_annealing | 480.0 | 0 | 0.0 | 3 | 16 |
| mosek | 745.0 | 0 | 0.0 | 3 | 16 |
| cpsat | 475.0 | 0 | 0.0 | 2 | 16 |

Expectation checks:
- PASS  metric `makespan_us` expected best in `['heft', 'cpsat', 'mosek']` observed best `['cpsat']`
- PASS  metric `deadline_miss_count` expected best in `['edf', 'cpsat', 'mosek', 'heft']` observed best `['heft', 'critical_path', 'edf', 'fastest_device', 'fifo', 'round_robin', 'peft', 'min_min', 'max_min', 'simulated_annealing', 'mosek', 'cpsat']`
- WARN  metric `deadline_miss_count (negative)` expected best in `NOT ['fastest_device', 'fifo']` observed best `['heft', 'critical_path', 'edf', 'fastest_device', 'fifo', 'round_robin', 'peft', 'min_min', 'max_min', 'simulated_annealing', 'mosek', 'cpsat']`

## multirate_periodic

![composite](side_by_side/multirate_periodic.png)

| scheduler | makespan_us | misses | total_lateness | cross_device | n_ops |
|---|---:|---:|---:|---:|---:|
| heft | 1570.0 | 3 | 1605.0 | 6 | 22 |
| critical_path | 1570.0 | 3 | 1605.0 | 6 | 22 |
| edf | 1570.0 | 1 | 255.0 | 6 | 22 |
| fastest_device | 1840.0 | 4 | 2960.0 | 3 | 22 |
| fifo | 1570.0 | 3 | 1575.0 | 5 | 22 |
| round_robin | 1795.0 | 4 | 2785.0 | 14 | 22 |
| peft | 1570.0 | 3 | 1365.0 | 6 | 22 |
| min_min | 1840.0 | 4 | 2620.0 | 3 | 22 |
| max_min | 1570.0 | 2 | 1250.0 | 3 | 22 |
| simulated_annealing | 1570.0 | 3 | 1605.0 | 6 | 22 |
| mosek | 1630.0 | 2 | 0.0 | 6 | 22 |
| cpsat | 1570.0 | 0 | 0.0 | 0 | 22 |

Expectation checks:
- PASS  metric `deadline_miss_count` expected best in `['edf', 'cpsat', 'mosek']` observed best `['cpsat']`

## memory_pressured_residual

![composite](side_by_side/memory_pressured_residual.png)

| scheduler | makespan_us | misses | total_lateness | cross_device | n_ops |
|---|---:|---:|---:|---:|---:|
| heft | 845.0 | 0 | 0.0 | 8 | 21 |
| critical_path | 1095.0 | 0 | 0.0 | 7 | 21 |
| edf | 1095.0 | 0 | 0.0 | 7 | 21 |
| fastest_device | 805.0 | 0 | 0.0 | 12 | 21 |
| fifo | 1095.0 | 0 | 0.0 | 7 | 21 |
| round_robin | 1190.0 | 0 | 0.0 | 21 | 21 |
| peft | 845.0 | 0 | 0.0 | 8 | 21 |
| min_min | 805.0 | 0 | 0.0 | 12 | 21 |
| max_min | 805.0 | 0 | 0.0 | 12 | 21 |
| simulated_annealing | 820.0 | 0 | 0.0 | 13 | 21 |
| mosek | 1045.0 | 0 | 0.0 | 12 | 21 |
| cpsat | 855.0 | 0 | 0.0 | 8 | 21 |

Expectation checks:
- FAIL  metric `makespan_us` expected best in `['heft', 'cpsat', 'mosek']` observed best `['fastest_device', 'min_min', 'max_min']`
- skip  metric `peak_memory_bytes` expected best in `['cpsat_memory']` observed best `None`

## tiny_op_quantized_chain

![composite](side_by_side/tiny_op_quantized_chain.png)

| scheduler | makespan_us | misses | total_lateness | cross_device | n_ops |
|---|---:|---:|---:|---:|---:|
| heft | 224.0 | 0 | 0.0 | 1 | 25 |
| critical_path | 350.0 | 0 | 0.0 | 0 | 25 |
| edf | 350.0 | 0 | 0.0 | 0 | 25 |
| fastest_device | 390.0 | 0 | 0.0 | 10 | 25 |
| fifo | 350.0 | 0 | 0.0 | 0 | 25 |
| round_robin | 809.0 | 0 | 0.0 | 24 | 25 |
| peft | 224.0 | 0 | 0.0 | 1 | 25 |
| min_min | 390.0 | 0 | 0.0 | 10 | 25 |
| max_min | 390.0 | 0 | 0.0 | 10 | 25 |
| simulated_annealing | 224.0 | 0 | 0.0 | 1 | 25 |
| mosek | 810.0 | 0 | 0.0 | 10 | 25 |
| cpsat | 220.0 | 0 | 0.0 | 0 | 25 |

Expectation checks:
- PASS  metric `makespan_us` expected best in `['cpsat', 'heft', 'critical_path', 'edf', 'fifo']` observed best `['cpsat']`
- PASS  metric `cross_device_transitions` expected best in `['cpsat', 'heft', 'critical_path', 'edf', 'fifo']` observed best `['critical_path', 'edf', 'fifo', 'cpsat']`

## heterogeneous_parallel

![composite](side_by_side/heterogeneous_parallel.png)

| scheduler | makespan_us | misses | total_lateness | cross_device | n_ops |
|---|---:|---:|---:|---:|---:|
| heft | 400.0 | 0 | 0.0 | 5 | 18 |
| critical_path | 975.0 | 0 | 0.0 | 4 | 18 |
| edf | 975.0 | 0 | 0.0 | 4 | 18 |
| fastest_device | 380.0 | 0 | 0.0 | 4 | 18 |
| fifo | 1510.0 | 0 | 0.0 | 5 | 18 |
| round_robin | 1255.0 | 0 | 0.0 | 10 | 18 |
| peft | 400.0 | 0 | 0.0 | 5 | 18 |
| min_min | 380.0 | 0 | 0.0 | 4 | 18 |
| max_min | 380.0 | 0 | 0.0 | 4 | 18 |
| simulated_annealing | 380.0 | 0 | 0.0 | 4 | 18 |
| mosek | 590.0 | 0 | 0.0 | 8 | 18 |
| cpsat | 380.0 | 0 | 0.0 | 4 | 18 |

Expectation checks:
- PASS  metric `makespan_us` expected best in `['heft', 'cpsat', 'mosek', 'critical_path']` observed best `['fastest_device', 'min_min', 'max_min', 'simulated_annealing', 'cpsat']`
- WARN  metric `makespan_us (negative)` expected best in `NOT ['fastest_device', 'fifo']` observed best `['fastest_device', 'min_min', 'max_min', 'simulated_annealing', 'cpsat']`
