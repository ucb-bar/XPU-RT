#!/bin/bash
# Rewrite a converter-generated QNN model .cpp so its nodes are served by the
# flowc.gpu op package instead of the backend's own qti.aisw package, and copy
# the weight blob alongside.  The GPU backend dispatches on
# Qnn_OpConfig_t::packageName, so this one string is the whole difference
# between "stock kernels" and "our kernels" for an otherwise identical graph.
#
#   tools/make_flowc_model.sh model/dronet_ref.cpp   ->  model/dronet_flowc.{cpp,bin}
set -eu
src=$1
base=${src%_ref.cpp}
sed 's|"qti.aisw", // Package Name|"flowc.gpu", // Package Name|' "$src" > "${base}_flowc.cpp"
cp "${base}_ref.bin" "${base}_flowc.bin"
echo "wrote ${base}_flowc.cpp (+ .bin): $(grep -c 'flowc.gpu' ${base}_flowc.cpp) nodes repackaged"
