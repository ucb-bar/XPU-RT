# FastDepth ONNX models (release asset)

The two FastDepth ONNX files are too large for the source tree and are published
as a release asset instead.

| file | size | SHA-256 |
|---|---|---|
| `MobileNetSkipAdd.onnx` | 15.8 MB | `827f7e199d66ca6a95b91b3264d9b10019d5c2a43e346298ce6a4ba431775183` |
| `MobileNetSkipAdd_fp16.onnx` | 7.9 MB | `b4f58a2a7c0ddf94e97b4c3dabd959704a6e022601738d4becf1c275672ed460` |

Fetch and verify with `bash fetch.sh` (set `XPURT_RELEASE_URL` to the release
download URL once it is published). The files are git-ignored (`*.onnx`).
