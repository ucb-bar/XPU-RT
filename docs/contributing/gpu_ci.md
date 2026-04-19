# GPU CI

The `gpu` workflow (`.github/workflows/gpu.yml`) exercises tests gated by
`torch.cuda.is_available()` — the ones CI runners without a GPU silently
skip. It is `workflow_dispatch` only; there is no schedule and no
push trigger.

## Tests in scope

- `tests/runtime/test_gpu_executor.py` — GPU-side Triton launch + zero-diff
  vs `torch.matmul`.
- `tests/kernels/megakernel/` — every megakernel regression test emits a
  real Triton kernel and runs it on the GPU.
- `tests/stages/test_cuda_gpu.py` — CUDA backend integration.

## Setting up a self-hosted runner

The workflow targets `runs-on: [self-hosted, gpu]`. To attach a machine:

1. In the GitHub repo, **Settings → Actions → Runners → New self-hosted
   runner**. Pick Linux x64.
2. Follow the shown `./config.sh` + `./run.sh` instructions on the GPU
   box. When prompted for **labels**, add `gpu` on top of the default
   `self-hosted`, `linux`, `x64`.
3. The runner needs: recent NVIDIA driver, CUDA-capable PyTorch (picked
   up from `uv sync`), and Triton. No extra setup if `uv` can install
   `torch` + `triton` wheels with the default CUDA variant for your
   driver.

## Running the workflow

- GitHub UI: **Actions → gpu → Run workflow**.
- `gh` CLI: `gh workflow run gpu.yml`.
- Optional `test_filter` input takes a pytest `-k` expression, e.g.
  `gh workflow run gpu.yml -f test_filter=static_schedule`.

## What it *doesn't* cover

- Multi-GPU: needs a host with ≥2 devices. Not wired in the workflow.
- Non-NVIDIA accelerators (AMD, NPU, etc.): out of scope.
- Long-running benchmarks: kept in the nightly + local benchmark
  harness (`python/compgen/bench/`), not here.

If no self-hosted runner is available the job queues indefinitely; it
will never fail CI by itself because the workflow never auto-triggers.
