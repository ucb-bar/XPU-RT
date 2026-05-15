# Model setup: SmolVLA (one-step policy)

## Overview

Small VLA policy used for one-step admission. Loaded via the existing compgen.models catalog entry ``smolvla_one_step``.

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
uv add transformers
# Workspace config / external repo path resolution per compgen.models.
```

The admission probe **never downloads weights**; supply them out of
band into the local HuggingFace cache (``$HF_HOME/hub`` or
``~/.cache/huggingface/hub``).

## Minimal smoke command

```bash
uv run python -m compgen.model_admission torch-compile \
    --model configs/models/smolvla_step.yaml \
    --slice configs/slices/smolvla_single_step.yaml \
    --out results/model_admission/smolvla_step/smolvla_single_step
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

- External SmolVLA repo not provisioned in workspace config.
- Missing tokenizer / processor cache.
- Image preprocessing mismatch.

## Support status

Blocking. Bridge loader (`compgen_model_spec` -> `smolvla_one_step`).
