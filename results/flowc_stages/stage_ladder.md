# QRB5165 feedback-stage ladder — full experiment log

Each stage adds exactly one degree of freedom, so the delta between two rows is attributable to that knob alone. Costs are pooled medians from `qnn_models/slicing_study/experiments.jsonl` (the measured sweep), not a re-run.

| stage | knob | freedom added |
|---|---|---|
| S0 | `monolith` | the fallback, no choice at all |
| S1 | `+backend` | see below |
| S2 | `+precision` | see below |
| S3 | `+slice` | see below |
| S4 | `+branch` | see below |


## vint

| stage | knob | ms | vs S0 | step | tiles | assignment |
|---|---|---:|---:|---:|---:|---|
| S0 | `monolith (cpu int8)` | 59.775 | 1.00x | — | 1 | `cpu@int8` |
| S1 | `+backend` | 59.775 | 1.00x | 1.00x | 1 | `cpu@int8` |
| S2 | `+precision` | 59.775 | 1.00x | 1.00x | 1 | `cpu@int8` |
| S3 | `+slice (contiguous)` | 29.379 | 2.03x | 2.03x | 2 | `t0:dsp@int8 t1:cpu@fp32` |
| S4 | `+branch (1 lane/kind)` | 23.201 | 2.58x | 1.27x | 3 | `t0:cpu@int8 t1:dsp@int8 t2:cpu@fp32` |

Concurrency is worth **+5.558 ms** (28.758 serial → 23.201 overlapped).


### vint S3 evidence — `vint_k2_prod`

cut: `{'boundary_tensors': ['/compress_obs_enc/Gemm_output_0'], 'n_tiles': 2, 'op_ranges': [[0, 1069], [1070, 1930]], 'src_node_count': 1931}`  ·  sweeps: 3

| tile | measured cells (ms) | rejected |
|---|---|---|
| t0 | `cpu@fp32`=84.314, `cpu@int8`=21.785, `dsp@int8`=15.486 | `hta@int8`: unsupported op StridedSlice |
| t1 | `cpu@fp32`=13.893, `cpu@int8`=41.02 | `dsp@int8`: Param[0] has incorrect Value 1., `hta@int8`: unsupported op Transpose |

### vint S4 evidence — `vint_par_enc`

cut: `{'boundary_tensors': [], 'tile_outputs': [['/compress_goal_enc/Gemm_output_0'], ['/compress_obs_enc/Gemm_output_0']], 'mode': 'branch', 'n_tiles': 3, 'op_ranges': [[6, 536], [540, 1069], [538, 1930]], 'op_range_sets': [[[6, 536]], [[540, 1069]], [[538, 538], [1071, 1930]]], 'independent_pairs': [[0, 1]], 'static_nodes_shared': 864, 'src_node_count': 1931}`  ·  sweeps: 3

| tile | measured cells (ms) | rejected |
|---|---|---|
| t0 | `cpu@fp32`=13.595, `cpu@int8`=6.615, `dsp@int8`=5.558 | `hta@int8`: unsupported op StridedSlice |
| t1 | `cpu@fp32`=69.773, `cpu@int8`=15.549, `dsp@int8`=9.288 | `hta@int8`: unsupported op Split |
| t2 | `cpu@fp32`=13.913, `cpu@int8`=37.801 | `dsp@int8`: Param[0] has incorrect Value 1., `hta@int8`: QnnHta [ ERROR ] QnnHtaHTA op Reshape supports only equal Input and Ou |

## yolov8n

| stage | knob | ms | vs S0 | step | tiles | assignment |
|---|---|---:|---:|---:|---:|---|
| S0 | `monolith (cpu int8)` | 72.722 | 1.00x | — | 1 | `cpu@int8` |
| S1 | `+backend` | 25.232 | 2.88x | 2.88x | 1 | `dsp@int8` |
| S2 | `+precision` | 25.232 | 2.88x | 1.00x | 1 | `dsp@int8` |
| S3 | `+slice (contiguous)` | 28.462 | 2.56x | 0.89x | 2 | `t0:dsp@int8 t1:dsp@int8` |

### yolov8n S3 evidence — `yolo_k2_prod`

cut: `{'boundary_tensors': ['/model.9/cv2/act/Mul_output_0'], 'n_tiles': 2, 'op_ranges': [[0, 98], [99, 232]], 'src_node_count': 233}`  ·  sweeps: 1

| tile | measured cells (ms) | rejected |
|---|---|---|
| t0 | `cpu@fp32`=104.634, `cpu@int8`=33.691, `dsp@int8`=12.396 | `hta@int8`: unsupported op Split |
| t1 | `cpu@fp32`=116.942, `cpu@int8`=38.427, `dsp@int8`=16.066 | `hta@int8`: unsupported op Split |

## fused_full

| stage | knob | ms | vs S0 | step | tiles | assignment |
|---|---|---:|---:|---:|---:|---|
| S0 | `monolith (cpu int8)` | n/a | — | — | 1 | `no cpu int8 cell` |
| S1 | `+backend` | 3.434 | — | — | 1 | `dsp@int8` |
| S2 | `+precision` | 0.896 | — | 3.83x | 1 | `cpu@fp32` |
| S3 | `+slice (contiguous)` | 0.452 | — | 1.98x | 3 | `t0:cpu@int8 t1:cpu@int8 t2:cpu@fp32` |
| S4 | `+branch (1 lane/kind)` | 0.503 | — | 0.90x | 3 | `t0:cpu@int8 t1:cpu@fp32 t2:cpu@fp32` |

Concurrency is worth **+0.000 ms** (0.503 serial → 0.503 overlapped).


### fused_full S3 evidence — `fused_k3_convs`

cut: `{'boundary_tensors': ['/vision_cnn/vision_cnn.7/Relu_output_0', '/depth_conv/depth_conv.3/Relu_output_0'], 'n_tiles': 3, 'op_ranges': [[0, 7], [8, 13], [14, 90]], 'src_node_count': 91}`  ·  sweeps: 2

| tile | measured cells (ms) | rejected |
|---|---|---|
| t0 | `cpu@fp32`=0.425, `cpu@int8`=0.164, `dsp@int8`=0.452, `hta@int8`=0.641 | — |
| t1 | `cpu@fp32`=0.236, `cpu@int8`=0.092, `dsp@int8`=0.75 | `hta@int8`: unsupported op Transpose |
| t2 | `cpu@fp32`=0.197, `dsp@int8`=3.277 | `cpu@int8`: validation failed for Reshape, `hta@int8`: unsupported op Transpose |

### fused_full S4 evidence — `fused_par_fc`

cut: `{'boundary_tensors': [], 'tile_outputs': [['/vision_fc/Gemm_output_0'], ['/depth_fc/Gemm_output_0']], 'mode': 'branch', 'n_tiles': 3, 'op_ranges': [[0, 9], [10, 15], [16, 90]], 'op_range_sets': [[[0, 9]], [[10, 15]], [[16, 90]]], 'independent_pairs': [[0, 1]], 'static_nodes_shared': 35, 'src_node_count': 91}`  ·  sweeps: 2

| tile | measured cells (ms) | rejected |
|---|---|---|
| t0 | `cpu@fp32`=0.567, `cpu@int8`=0.265, `dsp@int8`=0.868 | `hta@int8`: unsupported op Transpose |
| t1 | `cpu@fp32`=0.126, `cpu@int8`=0.13, `dsp@int8`=0.395 | `hta@int8`: unsupported op Transpose |
| t2 | `cpu@fp32`=0.111, `dsp@int8`=3.055 | `cpu@int8`: validation failed for Reshape, `hta@int8`: unsupported op Convert |

## dronet

| stage | knob | ms | vs S0 | step | tiles | assignment |
|---|---|---:|---:|---:|---:|---|
| S0 | `monolith (cpu int8)` | 7.466 | 1.00x | — | 1 | `cpu@int8` |
| S1 | `+backend` | 0.659 | 11.34x | 11.34x | 1 | `dsp@int8` |
| S2 | `+precision` | 0.659 | 11.34x | 1.00x | 1 | `dsp@int8` |
| S3 | `+slice (contiguous)` | 0.651 | 11.47x | 1.01x | 2 | `t0:dsp@int8 t1:cpu@int8` |

### dronet S3 evidence — `dronet_k2_head`

cut: `{'boundary_tensors': ['backbone_relu_4d'], 'n_tiles': 2, 'op_ranges': [[0, 26], [27, 28]], 'src_node_count': 29}`  ·  sweeps: 1

| tile | measured cells (ms) | rejected |
|---|---|---|
| t0 | `cpu@fp32`=1.385, `cpu@int8`=6.772, `dsp@int8`=0.639, `hta@int8`=1.394 | — |
| t1 | `cpu@fp32`=0.049, `cpu@int8`=0.012, `dsp@int8`=0.399, `hta@int8`=1.104 | — |

## mlp_control

| stage | knob | ms | vs S0 | step | tiles | assignment |
|---|---|---:|---:|---:|---:|---|
| S0 | `monolith (cpu int8)` | 0.030 | 1.00x | — | 1 | `cpu@int8` |
| S1 | `+backend` | 0.030 | 1.00x | 1.00x | 1 | `cpu@int8` |
| S2 | `+precision` | 0.030 | 1.00x | 1.00x | 1 | `cpu@int8` |
| S3 | `+slice (contiguous)` | 0.026 | 1.16x | 1.16x | 4 | `t0:cpu@fp32 t1:cpu@int8 t2:cpu@fp32 t3:cpu@int8` |

### mlp_control S3 evidence — `mlp_k4`

cut: `{'boundary_tensors': ['/mlp/mlp.1/Elu_output_0', '/mlp/mlp.3/Elu_output_0', '/mlp/mlp.5/Elu_output_0'], 'n_tiles': 4, 'op_ranges': [[0, 1], [2, 3], [4, 5], [6, 6]], 'src_node_count': 7}`  ·  sweeps: 1

| tile | measured cells (ms) | rejected |
|---|---|---|
| t0 | `cpu@fp32`=0.005, `cpu@int8`=0.026, `dsp@int8`=0.372 | `hta@int8`: unsupported elementwise neuson op 0 |
| t1 | `cpu@fp32`=0.049, `cpu@int8`=0.012, `dsp@int8`=0.357 | `hta@int8`: unsupported elementwise neuson op 0 |
| t2 | `cpu@fp32`=0.006, `cpu@int8`=0.006, `dsp@int8`=0.364 | `hta@int8`: unsupported elementwise neuson op 0 |
| t3 | `cpu@fp32`=0.002, `cpu@int8`=0.002, `dsp@int8`=0.358, `hta@int8`=0.541 | — |
