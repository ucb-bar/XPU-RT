# Model setup: AGIBOT (Go1 / Go2)

## Overview

AGIBOT VLA family. Stretch / admission_only -- non-blocking until upstream stabilises.

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
# Upstream repo TBD; admission flag is admission_only, not full_or_slice_smoke.
```

The admission probe **never downloads weights**; supply them out of
band into the local HuggingFace cache (``$HF_HOME/hub`` or
``~/.cache/huggingface/hub``).

## Minimal smoke command

```bash
uv run python -m compgen.model_admission torch-compile \
    --model configs/models/agibot_go1_step.yaml \
    --slice configs/slices/agibot_go1_single_step.yaml \
    --out results/model_admission/agibot_go1_step/agibot_go1_single_step
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

- Repo gating.
- Loader not yet wired to a stable upstream release.

## Support status

Non-blocking. Default outcome: `unavailable_missing_dependency` until loader is implemented.
