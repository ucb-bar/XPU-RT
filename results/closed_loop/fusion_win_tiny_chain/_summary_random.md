# closed_loop — scenario:fusion_win_tiny_chain via fastest_device (random)

- baseline makespan: **171.0 us** (6 dispatches)
- final makespan: **42.0 us** (3 dispatches)
- improvement: **129.0 us** (75.4%)
- candidates evaluated: 6
- candidates accepted: 3

## Trace

| iter | candidate | type | before | after | delta | accepted |
|---:|---|---|---:|---:|---:|---:|
| 0 | fuse_tiny_2__tiny_3 | fuse_producer_consumer | 171.0 | 118.0 | -53.0 | YES |
| 1 | fuse_tiny_0__tiny_1 | fuse_producer_consumer | 118.0 | 95.0 | -23.0 | YES |
| 2 | fuse_tiny_4__tiny_5 | fuse_producer_consumer | 95.0 | 42.0 | -53.0 | YES |
| 3 | fuse_chain_tiny_0+tiny_1_tiny_4+tiny_5 | fuse_linear_chain | 42.0 | 42.0 | 0.0 | no |
| 4 | fuse_tiny_2+tiny_3__tiny_4+tiny_5 | fuse_producer_consumer | 42.0 | 42.0 | 0.0 | no |
| 5 | fuse_tiny_0+tiny_1__tiny_2+tiny_3 | fuse_producer_consumer | 42.0 | 42.0 | 0.0 | no |
