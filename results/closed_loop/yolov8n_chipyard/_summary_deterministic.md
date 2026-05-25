# closed_loop — model:yolov8n@chipyard via round_robin (deterministic)

- baseline makespan: **252159723.9 us** (48 dispatches)
- final makespan: **159365568.1 us** (31 dispatches)
- improvement: **92794155.8 us** (36.8%)
- candidates evaluated: 4
- candidates accepted: 4

## Trace

| iter | candidate | type | before | after | delta | accepted |
|---:|---|---|---:|---:|---:|---:|
| 0 | fuse_chain_yolov8n_d30_yolov8n_d35 | fuse_linear_chain | 252159723.9 | 213040519.34601825 | -39119204.54745421 | YES |
| 1 | fuse_chain_yolov8n_d30+yolov8n_d31+yolov8n_d32+yolov8n_d33+yolov8n_d34+yolov8n_d35_yolov8n_d40 | fuse_linear_chain | 213040519.3 | 184435434.72661045 | -28605084.619407803 | YES |
| 2 | fuse_chain_yolov8n_d30+yolov8n_d31+yolov8n_d32+yolov8n_d33+yolov8n_d34+yolov8n_d35+yolov8n_d36+yolov8n_d37+yolov8n_d38+yolov8n_d39+yolov8n_d40_yolov8n_d45 | fuse_linear_chain | 184435434.7 | 167926215.55006284 | -16509219.176547617 | YES |
| 3 | fuse_chain_yolov8n_d30+yolov8n_d31+yolov8n_d32+yolov8n_d33+yolov8n_d34+yolov8n_d35+yolov8n_d36+yolov8n_d37+yolov8n_d38+yolov8n_d39+yolov8n_d40+yolov8n_d41+yolov8n_d42+yolov8n_d43+yolov8n_d44+yolov8n_d45_yolov8n_d47 | fuse_linear_chain | 167926215.6 | 159365568.0561472 | -8560647.493915647 | YES |
