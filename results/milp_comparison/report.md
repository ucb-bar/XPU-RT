# MILP solver comparison

Same MILP formulation (xpu-rt/scheduler.py), different CVXPY backends.
Used to disambiguate 'is the win formulation-driven or solver-driven?'

Backends compared: ['MOSEK', 'HIGHS']

| workload | n_ops | MOSEK ms / s | HIGHS ms / s |
|---|---:|---:|---:|
| montage_25 | 20 | 990 / 0.33 | 990 / 0.30 |
| cybershake_30 | 27 | 550 / 2.13 | 1270 / 2.14 |
| epigenomics_46 | 30 | 940 / 2.41 | 940 / 2.30 |
| dronet_chipyard | 15 | 1502288 / 0.03 | 1502288 / 0.03 |
| mlp_wide_chipyard | 3 | 7210 / 0.01 | 7210 / 0.01 |
| vision_pipeline | 14 | skip | skip |
| sensor_fusion_diamond | 16 | 745 / 0.15 | 745 / 0.19 |
| tiny_op_quantized_chain | 25 | 810 / 0.05 | 810 / 0.04 |