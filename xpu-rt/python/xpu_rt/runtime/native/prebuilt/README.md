Prebuilt native libraries land here. CI's build_cuda step + Makefile target build-cuda-rt populate it before `python -m build` runs.

This dir is intentionally checked into the package layout (with this README only) so hatch's force-include works on a clean source tree.
