# Model setup: Octo (diffusion VLA, single denoising step)

## Overview

Diffusion-based generalist policy. Admission probes a single denoising step, not full sampling.

## Source verification

- model_ref: TO_BE_VERIFIED_ONLINE
- repo_url: TO_BE_VERIFIED_ONLINE
- docs_url: TO_BE_VERIFIED_ONLINE
- verified_at: null
- verified_by: null
- access requirements: TO_BE_VERIFIED_ONLINE
- source_verified: false

> The fields above must remain ``TO_BE_VERIFIED_ONLINE`` until a human
> confirms the upstream identifiers against the live repository or model
> hub. Do not flip ``source_verified`` to true based on training-data
> recall.

## Installation

```bash
uv add jax jaxlib torch
# Or the official octo Python package once verified.
```

The admission probe **never downloads weights**; supply them out of
band into the local HuggingFace cache (``$HF_HOME/hub`` or
``~/.cache/huggingface/hub``).

## Minimal smoke command

```bash
uv run python -m compgen.model_admission torch-compile \
    --model configs/models/octo_step.yaml \
    --slice configs/slices/octo_single_step.yaml \
    --out results/model_admission/octo_step/octo_single_step
```

## Expected artifacts

- ``admission_report.json``
- ``torch_compile_report.json``
- ``dynamo_report.json``
- ``eager_report.json``
- ``environment.json``
- ``input_summary.json``
- ``error.txt`` (only if a failure produced an error)

## Known failure modes

- JAX / XLA toolchain mismatch with torch.
- Diffusion-step API drift across releases.

## Support status

Blocking. Slice loaded as a one-step policy probe only.
