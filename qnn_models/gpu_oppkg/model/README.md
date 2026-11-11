# Converter output used by the measurements

`qnn-onnx-converter -o <name>_ref.cpp` output (C++ model + weight blob +
`_net.json` with every tensor's encoding), one per graph. The `_ref` variants
carry the stock `qti.aisw` package name; the `_flowc` twins used on the GPU are
derived from them with `../tools/make_flowc_model.sh` and are not stored here,
because the only difference is one string per node and the weight blob is
byte-identical.

| artifact | source ONNX | calibration |
|---|---|---|
| `mlp_ref` | `flow_c/gen/onnx/mlp_control.onnx` | `runtime/gen/mlp_control_full/calib_list.txt` |
| `mlpf32_ref`, `mlpf16_ref` | same, unquantized / `--float_bitwidth 16` | — |
| `mlp_cut2_ref`, `mlp_cut3_ref` | `mlp_ref` truncated after node 1 / node 2 by `../tools/make_cut_model.py` | — |
| `dronet_ref` | `qnn_models/dronet_simplified.onnx` (`--input_layout input NCHW`) | `list_dronet.txt` |
| `visconv_ref` | `flow_c/gen/convert/tiles/fused_vision_conv.onnx` | `list_vision_conv.txt` |
| `fused_ref` | `flow_c/gen/onnx/fused_full.onnx` | `list_fused.txt` |

`bin_extract/` is `mlp_ref.bin` untarred: the raw int8 weight/bias blobs the
fp64 emulation in `../tools/emulate_mlp_int8.py` reads.
