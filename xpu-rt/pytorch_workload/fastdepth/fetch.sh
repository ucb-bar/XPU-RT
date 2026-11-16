#!/usr/bin/env bash
# Download the FastDepth ONNX release assets next to this script and verify them.
set -euo pipefail
: "${XPURT_RELEASE_URL:?set XPURT_RELEASE_URL to the release download URL (directory containing the .onnx files)}"
cd "$(dirname "$0")"
for f in MobileNetSkipAdd.onnx MobileNetSkipAdd_fp16.onnx; do
  [ -f "$f" ] || curl -fL -o "$f" "$XPURT_RELEASE_URL/$f"
done
cat > .SHA256SUMS <<'SUMS'
827f7e199d66ca6a95b91b3264d9b10019d5c2a43e346298ce6a4ba431775183  MobileNetSkipAdd.onnx
b4f58a2a7c0ddf94e97b4c3dabd959704a6e022601738d4becf1c275672ed460  MobileNetSkipAdd_fp16.onnx
SUMS
shasum -a 256 -c .SHA256SUMS
