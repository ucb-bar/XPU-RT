# closed_loop — scenario:tiny_op_quantized_chain via fastest_device (cost_model)

- baseline makespan: **390.0 us** (25 dispatches)
- final makespan: **312.0 us** (13 dispatches)
- improvement: **78.0 us** (20.0%)
- candidates evaluated: 10
- candidates accepted: 4

## Trace

| iter | candidate | type | before | after | delta | accepted |
|---:|---|---|---:|---:|---:|---:|
| 0 | fuse_L0_quant__L0_conv | fuse_producer_consumer | 390.0 | 367.0 | -23.0 | YES |
| 1 | fuse_chain_L1_bias_L2_bias | fuse_linear_chain | 367.0 | 339.0 | -28.0 | YES |
| 2 | fuse_chain_L3_relu_L4_relu | fuse_linear_chain | 339.0 | 316.0 | -23.0 | YES |
| 3 | fuse_L0_bias__L0_relu | fuse_producer_consumer | 316.0 | 316.0 | 0.0 | no |
| 4 | fuse_L0_relu__L0_dequant | fuse_producer_consumer | 316.0 | 316.0 | 0.0 | no |
| 5 | fuse_L2_relu__L2_dequant | fuse_producer_consumer | 316.0 | 316.0 | 0.0 | no |
| 6 | fuse_L0_dequant__L1_quant | fuse_producer_consumer | 316.0 | 316.0 | 0.0 | no |
| 7 | fuse_L2_dequant__L3_quant | fuse_producer_consumer | 316.0 | 316.0 | 0.0 | no |
| 8 | fuse_L1_bias+L1_relu+L1_dequant+L2_quant+L2_conv+L2_bias__L2_relu | fuse_producer_consumer | 316.0 | 317.0 | 1.0 | no |
| 9 | fuse_L3_bias__L3_relu+L3_dequant+L4_quant+L4_conv+L4_bias+L4_relu | fuse_producer_consumer | 316.0 | 312.0 | -4.0 | YES |
