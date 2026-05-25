# M17+M18 — Real-frequency QRB5165 packing

- SoC: QRB5165 (real silicon, NO time scaling)
- envelope: 200000.0 us
- mixes:
  - camera_only: [('camera', 30)]
  - camera_imu: [('camera', 30), ('imu', 200)]
  - camera_imu_control: [('camera', 30), ('imu', 200), ('control', 100)]
  - full_stack: [('camera', 30), ('imu', 200), ('control', 100), ('planning', 10), ('monitor', 1)]
- schedulers: ['heft', 'critical_path', 'edf', 'fastest_device', 'fifo', 'peft', 'simulated_annealing', 'cpsat', 'gnn_placement']

## camera_only

| scheduler | n_ops | makespan_us | misses | total_lateness_us | feasible |
|---|---:|---:|---:|---:|---|
| heft | 90 | 291062 | 6 | 992176 | True |
| critical_path | 90 | 284803 | 6 | 969505 | True |
| edf | 90 | 284663 | 6 | 968806 | True |
| fastest_device | 90 | 516565 | 6 | 2280849 | True |
| fifo | 90 | 284803 | 6 | 969505 | True |
| peft | 90 | 277582 | 6 | 915110 | True |
| simulated_annealing | 90 | 290259 | 6 | 997692 | True |
| cpsat | 90 | 229629 | 6 | 191568 | True |
| gnn_placement | 90 | 291062 | 6 | 992176 | True |

## camera_imu

| scheduler | n_ops | makespan_us | misses | total_lateness_us | feasible |
|---|---:|---:|---:|---:|---|
| heft | 210 | 344831 | 46 | 10260907 | True |
| critical_path | 210 | 341949 | 46 | 10111874 | True |
| edf | 210 | 342969 | 46 | 9746986 | True |
| fastest_device | 210 | 516565 | 46 | 9123505 | True |
| fifo | 210 | 363098 | 46 | 6068509 | True |
| peft | 210 | 332351 | 46 | 9680660 | True |
| simulated_annealing | 210 | 344831 | 46 | 10260907 | True |
| cpsat | 210 | 230502 | 6 | 293754 | True |
| gnn_placement | 210 | 344831 | 46 | 10260907 | True |

## camera_imu_control

| scheduler | n_ops | makespan_us | misses | total_lateness_us | feasible |
|---|---:|---:|---:|---:|---|
| heft | 270 | 353739 | 66 | 15204270 | True |
| critical_path | 270 | 350321 | 66 | 14989245 | True |
| edf | 270 | 351261 | 66 | 14612705 | True |
| fastest_device | 270 | 550465 | 66 | 12458626 | True |
| fifo | 270 | 393087 | 66 | 10521978 | True |
| peft | 270 | 340744 | 66 | 14364429 | True |
| simulated_annealing | 270 | 353739 | 66 | 15204270 | True |
| cpsat | 270 | 253847 | 6 | 352600 | True |
| gnn_placement | 270 | 353739 | 66 | 15204270 | True |

## full_stack

| scheduler | n_ops | makespan_us | misses | total_lateness_us | feasible |
|---|---:|---:|---:|---:|---|
| heft | 302 | 386320 | 68 | 17713933 | True |
| critical_path | 302 | 389317 | 68 | 17894149 | True |
| edf | 302 | 390000 | 68 | 17487648 | True |
| fastest_device | 302 | 569660 | 68 | 13852596 | True |
| fifo | 302 | 432971 | 68 | 11617643 | True |
| peft | 302 | 373660 | 68 | 16806927 | True |
| simulated_annealing | 302 | 386320 | 68 | 17713933 | True |
| cpsat | 302 | 265136 | 6 | 524493 | True |
| gnn_placement | 302 | 386320 | 68 | 17713933 | True |
