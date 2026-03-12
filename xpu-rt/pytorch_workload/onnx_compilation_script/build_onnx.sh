iree-import-onnx -o --opset-version=17  ../../mlir_pipelining_script/workload_mlir/fastdepth.mlir ../fastdepth/MobileNetSkipAdd.onnx

python3 ../dronet/dronet.py
iree-import-onnx -o --opset-version=17   ../../mlir_pipelining_script/workload_mlir/dronet.mlir ../dronet/dronet.onnx

python3 ../mlp/simple_mlp.py
iree-import-onnx -o --opset-version=17   ../../mlir_pipelining_script/workload_mlir/mlp.mlir ../mlp/simple_mlp.onnx


