# closed_loop — scenario:tiny_op_quantized_chain via fastest_device (deterministic)

- baseline makespan: **390.0 us** (25 dispatches)
- final makespan: **220.0 us** (1 dispatches)
- improvement: **170.0 us** (43.6%)
- candidates evaluated: 5
- candidates accepted: 5

## Trace

| iter | candidate | type | before | after | delta | accepted |
|---:|---|---|---:|---:|---:|---:|
| 0 | fuse_chain_L1_conv_L2_conv | fuse_linear_chain | 390.0 | 356.0 | -34.0 | YES |
| 1 | fuse_chain_L1_conv+L1_bias+L1_relu+L1_dequant+L2_quant+L2_conv_L3_conv | fuse_linear_chain | 356.0 | 322.0 | -34.0 | YES |
| 2 | fuse_chain_L1_conv+L1_bias+L1_relu+L1_dequant+L2_quant+L2_conv+L2_bias+L2_relu+L2_dequant+L3_quant+L3_conv_L4_conv | fuse_linear_chain | 322.0 | 288.0 | -34.0 | YES |
| 3 | fuse_chain_L0_quant_L1_quant | fuse_linear_chain | 288.0 | 255.0 | -33.0 | YES |
| 4 | fuse_chain_L0_quant+L0_conv+L0_bias+L0_relu+L0_dequant+L1_quant_L4_dequant | fuse_linear_chain | 255.0 | 220.0 | -35.0 | YES |
