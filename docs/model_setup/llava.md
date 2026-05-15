# Model setup: LLaVA

## Overview

Reference open-source LLaVA VLM. Admission-blocking on the 7B baseline.

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
uv add transformers accelerate sentencepiece pillow
```

The admission probe **never downloads weights**; supply them out of
band into the local HuggingFace cache (``$HF_HOME/hub`` or
``~/.cache/huggingface/hub``).

## Minimal smoke command

```bash
uv run python -m compgen.model_admission torch-compile \
    --model configs/models/llava.yaml \
    --slice configs/slices/llava_single_image_qa.yaml \
    --out results/model_admission/llava/llava_single_image_qa
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

- Missing CLIP vision tower weights.
- Tokenizer mismatch (LLaMA-1 vs LLaMA-2).
- Graph break in `prepare_inputs_for_generation`.

## Support status

Blocking. Must reach `available` or `unavailable_*`.
