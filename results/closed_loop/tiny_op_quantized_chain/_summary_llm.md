# closed_loop — scenario:tiny_op_quantized_chain via fastest_device (llm)

- baseline makespan: **390.0 us** (25 dispatches)
- final makespan: **220.0 us** (1 dispatches)
- improvement: **170.0 us** (43.6%)
- candidates evaluated: 5
- candidates accepted: 5

## Trace

| iter | candidate | type | before | after | delta | accepted |
|---:|---|---|---:|---:|---:|---:|
| 0 | fuse_chain_L3_relu_L4_relu | fuse_linear_chain | 390.0 | 367.0 | -23.0 | YES |
| 1 | fuse_chain_L2_bias_L3_bias | fuse_linear_chain | 367.0 | 329.0 | -38.0 | YES |
| 2 | fuse_chain_L1_conv_L2_conv | fuse_linear_chain | 329.0 | 295.0 | -34.0 | YES |
| 3 | fuse_chain_L0_quant_L1_quant | fuse_linear_chain | 295.0 | 262.0 | -33.0 | YES |
| 4 | fuse_chain_L0_quant+L0_conv+L0_bias+L0_relu+L0_dequant+L1_quant_L4_dequant | fuse_linear_chain | 262.0 | 220.0 | -42.0 | YES |
