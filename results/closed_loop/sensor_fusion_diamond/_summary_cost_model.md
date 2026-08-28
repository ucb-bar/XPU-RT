# closed_loop — scenario:sensor_fusion_diamond via fastest_device (cost_model)

- baseline makespan: **510.0 us** (16 dispatches)
- final makespan: **510.0 us** (16 dispatches)
- improvement: **0.0 us** (0.0%)
- candidates evaluated: 10
- candidates accepted: 0

## Trace

| iter | candidate | type | before | after | delta | accepted |
|---:|---|---|---:|---:|---:|---:|
| 0 | fuse_chain_fuse_pre_ctrl_out | fuse_linear_chain | 510.0 | 510.0 | 0.0 | no |
| 1 | fuse_fuse_post__ctrl_in | fuse_producer_consumer | 510.0 | 510.0 | 0.0 | no |
| 2 | fuse_ctrl_in__ctrl_calc | fuse_producer_consumer | 510.0 | 510.0 | 0.0 | no |
| 3 | fuse_ctrl_calc__ctrl_out | fuse_producer_consumer | 510.0 | 510.0 | 0.0 | no |
| 4 | fuse_cam_in__cam_stage_0 | fuse_producer_consumer | 510.0 | 510.0 | 0.0 | no |
| 5 | split_ctrl_calc | split_heavy_dispatch | 510.0 | 510.0 | 0.0 | no |
| 6 | split_ctrl_in | split_heavy_dispatch | 510.0 | 510.0 | 0.0 | no |
| 7 | split_ctrl_out | split_heavy_dispatch | 510.0 | 510.0 | 0.0 | no |
| 8 | split_fuse_post | split_heavy_dispatch | 510.0 | 510.0 | 0.0 | no |
| 9 | fuse_lid_in__lid_stage_0 | fuse_producer_consumer | 510.0 | 510.0 | 0.0 | no |
