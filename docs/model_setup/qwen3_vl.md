# Model setup: Qwen3-VL family (8B / 235B-A22B-Instruct)

## Overview

Vision-language family covering both medium (8B) and huge (235B-A22B Mixture-of-Experts) variants. The 8B is admission-blocking; the 235B is slice-only and non-blocking by default.

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
# Stage weights into HF cache out of band; the admission probe never downloads.
```

The admission probe **never downloads weights**; supply them out of
band into the local HuggingFace cache (``$HF_HOME/hub`` or
``~/.cache/huggingface/hub``).

## Minimal smoke command

```bash
uv run python -m compgen.model_admission torch-compile \
    --model configs/models/qwen3_vl_8b.yaml \
    --slice configs/slices/qwen3_vl_8b_single_image_qa.yaml \
    --out results/model_admission/qwen3_vl_8b/qwen3_vl_8b_single_image_qa
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

- Gated weights (HF token / repo agreement not accepted).
- Missing AutoProcessor (transformers version mismatch).
- CUDA OOM on the 235B variant -- expected; use slice configs instead.
- Remote-code dependency missing (`trust_remote_code=true` required).
- Graph break on dynamic image-token concat in some transformers versions.

## Support status

- 8B: blocking. Must reach `available` or an explicit `unavailable_*` status.
- 235B-A22B: slice-only, non-blocking. Slices may be `available` or `unavailable_too_large`.
