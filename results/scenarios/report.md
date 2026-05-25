# M6 — Diagnostic scenarios

**Summary: 8 / 12 expectation checks passed.**

- schedulers: ['heft', 'critical_path', 'edf', 'fastest_device', 'fifo', 'mosek', 'cpsat']
- scenarios: ['vision_pipeline', 'sensor_fusion_diamond', 'multirate_periodic', 'memory_pressured_residual', 'tiny_op_quantized_chain', 'heterogeneous_parallel']

## Scoreboard (across all scenarios x metrics)

| scheduler | wins | losses |
|---|---:|---:|
| heft | 4 | 3 |
| critical_path | 6 | 5 |
| edf | 6 | 5 |
| fastest_device | 7 | 6 |
| fifo | 9 | 5 |
| mosek | 2 | 11 |
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
| mosek | 695.0 | 0 | 0.0 | 7 | 14 |
| cpsat | 490.0 | 0 | 0.0 | 7 | 14 |

Expectation checks:
- PASS  metric `makespan_us` expected best in `['heft', 'cpsat', 'mosek', 'critical_path']` observed best `['heft', 'cpsat']`
- PASS  metric `deadline_miss_count` expected best in `['edf', 'cpsat', 'mosek', 'heft']` observed best `['heft', 'fastest_device', 'mosek', 'cpsat']`

## sensor_fusion_diamond

![composite](side_by_side/sensor_fusion_diamond.png)

| scheduler | makespan_us | misses | total_lateness | cross_device | n_ops |
|---|---:|---:|---:|---:|---:|
| heft | 480.0 | 0 | 0.0 | 3 | 16 |
| critical_path | 510.0 | 0 | 0.0 | 1 | 16 |
| edf | 510.0 | 0 | 0.0 | 1 | 16 |
| fastest_device | 510.0 | 0 | 0.0 | 3 | 16 |
| fifo | 510.0 | 0 | 0.0 | 1 | 16 |
| mosek | 745.0 | 0 | 0.0 | 3 | 16 |
| cpsat | 475.0 | 0 | 0.0 | 2 | 16 |

Expectation checks:
- PASS  metric `makespan_us` expected best in `['heft', 'cpsat', 'mosek']` observed best `['cpsat']`
- PASS  metric `deadline_miss_count` expected best in `['edf', 'cpsat', 'mosek', 'heft']` observed best `['heft', 'critical_path', 'edf', 'fastest_device', 'fifo', 'mosek', 'cpsat']`
- WARN  metric `deadline_miss_count (negative)` expected best in `NOT ['fastest_device', 'fifo']` observed best `['heft', 'critical_path', 'edf', 'fastest_device', 'fifo', 'mosek', 'cpsat']`

## multirate_periodic

![composite](side_by_side/multirate_periodic.png)

| scheduler | makespan_us | misses | total_lateness | cross_device | n_ops |
|---|---:|---:|---:|---:|---:|
| heft | 1570.0 | 3 | 1605.0 | 6 | 22 |
| critical_path | 1570.0 | 3 | 1605.0 | 6 | 22 |
| edf | 1570.0 | 1 | 255.0 | 6 | 22 |
| fastest_device | 1840.0 | 4 | 2960.0 | 3 | 22 |
| fifo | 1570.0 | 3 | 1575.0 | 5 | 22 |
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
| mosek | 1045.0 | 0 | 0.0 | 12 | 21 |
| cpsat | 855.0 | 0 | 0.0 | 8 | 21 |

Expectation checks:
- FAIL  metric `makespan_us` expected best in `['heft', 'cpsat', 'mosek']` observed best `['fastest_device']`
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
| mosek | 590.0 | 0 | 0.0 | 8 | 18 |
| cpsat | 380.0 | 0 | 0.0 | 4 | 18 |

Expectation checks:
- PASS  metric `makespan_us` expected best in `['heft', 'cpsat', 'mosek', 'critical_path']` observed best `['fastest_device', 'cpsat']`
- WARN  metric `makespan_us (negative)` expected best in `NOT ['fastest_device', 'fifo']` observed best `['fastest_device', 'cpsat']`
