# Model setup: Qwen2.5-VL

## Overview

Mid-size vision-language model. Admission-blocking.

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
uv add transformers accelerate qwen-vl-utils
```

The admission probe **never downloads weights**; supply them out of
band into the local HuggingFace cache (``$HF_HOME/hub`` or
``~/.cache/huggingface/hub``).

## Minimal smoke command

```bash
uv run python -m compgen.model_admission torch-compile \
    --model configs/models/qwen2_5_vl.yaml \
    --slice configs/slices/qwen2_5_vl_single_image_qa.yaml \
    --out results/model_admission/qwen2_5_vl/qwen2_5_vl_single_image_qa
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

- Gated repo / missing HF token.
- Older transformers without Qwen2.5-VL processor.
- CUDA OOM on full 32K-context probes.

## Support status

Blocking. Must reach `available` or `unavailable_*`.
