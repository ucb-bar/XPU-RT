# Model setup: DeepSeek-V4-Flash (text MoE / hybrid attention)

## Overview

Huge hybrid-attention MoE model. Slice-only; admission probes hybrid attention and MoE block separately.

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
uv add transformers accelerate
```

The admission probe **never downloads weights**; supply them out of
band into the local HuggingFace cache (``$HF_HOME/hub`` or
``~/.cache/huggingface/hub``).

## Minimal smoke command

```bash
uv run python -m compgen.model_admission torch-compile \
    --model configs/models/deepseek_v4_flash_text.yaml \
    --slice configs/slices/deepseek_v4_flash_hybrid_attention.yaml \
    --out results/model_admission/deepseek_v4_flash_text/deepseek_v4_flash_hybrid_attention
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

- Full weights too large for local admission.
- Custom hybrid-attention kernel missing on CPU.

## Support status

Blocking on slices, never on the full model.
