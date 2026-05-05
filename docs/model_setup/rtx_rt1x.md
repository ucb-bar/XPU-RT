# Model setup: RT-1-X / RT-X (one-step policy)

## Overview

Open robot-policy baseline. Non-blocking.

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
uv add transformers tensorflow
# RT-1-X originally TF; admission probes the torch port.
```

The admission probe **never downloads weights**; supply them out of
band into the local HuggingFace cache (``$HF_HOME/hub`` or
``~/.cache/huggingface/hub``).

## Minimal smoke command

```bash
uv run python -m compgen.model_admission torch-compile \
    --model configs/models/rtx_rt1x_step.yaml \
    --slice configs/slices/rtx_rt1x_single_step.yaml \
    --out results/model_admission/rtx_rt1x_step/rtx_rt1x_single_step
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

- TF/PT bridge missing.
- Discretized-action head mismatch.

## Support status

Non-blocking; best-effort.
