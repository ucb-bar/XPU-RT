# M20 — Pegasus literature DAGs

Programmatically constructed Pegasus-shaped workflows 
(no DAX XML required). Compares HEFT-on-Pegasus to the 
published 1.3-1.8x makespan-vs-LB range; lower is better.

| DAG | scheduler | ms (us) | LB (us) | ms/LB | solver_s |
|---|---|---:|---:|---:|---:|
| montage_25 | heft | 1185 | 750 | 1.58 | 0.005 |
| montage_25 | critical_path | 1510 | 750 | 2.01 | 0.000 |
| montage_25 | edf | 1510 | 750 | 2.01 | 0.000 |
| montage_25 | fastest_device | 1405 | 750 | 1.87 | 0.000 |
| montage_25 | fifo | 1510 | 750 | 2.01 | 0.000 |
| montage_25 | peft | 1185 | 750 | 1.58 | 0.001 |
| montage_25 | simulated_annealing | 1185 | 750 | 1.58 | 0.030 |
| montage_25 | mosek | 990 | 750 | 1.32 | 0.895 |
| montage_25 | cpsat | 1300 | 750 | 1.73 | 30.256 |
| montage_25 | gnn_placement | 1185 | 750 | 1.58 | 2.575 |
| cybershake_30 | heft | 1085 | 430 | 2.52 | 0.001 |
| cybershake_30 | critical_path | 1400 | 430 | 3.26 | 0.001 |
| cybershake_30 | edf | 1400 | 430 | 3.26 | 0.001 |
| cybershake_30 | fastest_device | 1215 | 430 | 2.83 | 0.001 |
| cybershake_30 | fifo | 1400 | 430 | 3.26 | 0.000 |
| cybershake_30 | peft | 1085 | 430 | 2.52 | 0.001 |
| cybershake_30 | simulated_annealing | 1085 | 430 | 2.52 | 0.041 |
| cybershake_30 | mosek | 550 | 430 | 1.28 | 2.335 |
| cybershake_30 | cpsat | 1140 | 430 | 2.65 | 30.017 |
| cybershake_30 | gnn_placement | 1085 | 430 | 2.52 | 0.068 |
| epigenomics_46 | heft | 2010 | 730 | 2.75 | 0.001 |
| epigenomics_46 | critical_path | 2210 | 730 | 3.03 | 0.001 |
| epigenomics_46 | edf | 2210 | 730 | 3.03 | 0.001 |
| epigenomics_46 | fastest_device | 3050 | 730 | 4.18 | 0.001 |
| epigenomics_46 | fifo | 2210 | 730 | 3.03 | 0.001 |
| epigenomics_46 | peft | 2010 | 730 | 2.75 | 0.001 |
| epigenomics_46 | simulated_annealing | 1970 | 730 | 2.70 | 0.050 |
| epigenomics_46 | mosek | 940 | 730 | 1.29 | 2.281 |
| epigenomics_46 | cpsat | 1955 | 730 | 2.68 | 30.017 |
| epigenomics_46 | gnn_placement | 2010 | 730 | 2.75 | 0.014 |