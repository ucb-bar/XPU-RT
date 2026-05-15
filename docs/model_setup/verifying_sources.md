# Verifying model sources (one-time workflow)

The model admission registry ships with every real-model YAML marked
``source_verified: false`` and ``model_ref: TO_BE_VERIFIED_ONLINE``.
This is intentional: model identifiers must be confirmed against the
live HuggingFace hub once, by a human, before the admission probe will
trust them.

After one run of ``verify-sources``, the resolved canonical ref + the
exact revision SHA are written back into each ``configs/models/<id>.yaml``
and committed to git. The probe trusts those values; verification does
not run again unless you explicitly pass ``--refresh``.

## TL;DR — running it for the first time

```bash
# 1. (optional) supply a HF token if you want gated repos resolved
export HF_TOKEN=...

# 2. review the candidate list and edit any wrong refs
$EDITOR configs/model_admission/source_candidates.yaml

# 3. run the verifier (one HTTP GET per candidate; never downloads weights)
uv run python -m compgen.model_admission verify-sources \
    --candidates configs/model_admission/source_candidates.yaml

# 4. inspect the diff and commit
git diff configs/models/
git add configs/models/ configs/model_admission/source_candidates.yaml
git commit -m "feat(model_admission): verify upstream HF sources"
```

That's it. Every subsequent ``run-suite`` invocation skips verification
and trusts the YAMLs.

## How it works

The verifier issues exactly one read-only API call per candidate:

```python
HfApi().model_info(candidate_ref, token=os.environ.get("HF_TOKEN"))
```

This hits ``https://huggingface.co/api/models/<ref>`` and returns
metadata (canonical id, revision SHA, gated/private flags). **No
weights are downloaded.**

The result is classified into one of:

| Status               | Effect on the YAML                                              |
| -------------------- | --------------------------------------------------------------- |
| ``passed``           | ``source_verified: true``, canonical ref + revision SHA pinned. |
| ``gated``            | ``source_verified: false``, note appended (acquire HF access).  |
| ``not_found``        | ``source_verified: false``, note flags candidate is wrong.      |
| ``auth_required``    | ``source_verified: false``, set ``HF_TOKEN`` and re-run.        |
| ``network_error``    | YAML untouched; transient network failure.                      |
| ``skipped``          | Candidate is empty / ``TO_BE_VERIFIED_ONLINE``; YAML untouched. |

## Re-running

By default, models that already have ``source_verified: true`` and a
non-empty ``revision`` are skipped — no API call is made. To force a
re-check (for example, to bump to the latest revision):

```bash
uv run python -m compgen.model_admission verify-sources --refresh
```

Use ``--only model_id1 model_id2`` to scope to a subset.

Use ``--dry-run`` to print the table without writing the YAMLs.

## Updating one entry

If you discover the candidate ref is wrong (e.g. ``DeepSeek-V4-Flash``
vs ``DeepSeek-V3``), edit the candidate YAML and re-run scoped:

```bash
$EDITOR configs/model_admission/source_candidates.yaml
uv run python -m compgen.model_admission verify-sources \
    --refresh --only deepseek_v4_flash_text
```

## Why we pin a revision SHA

HuggingFace's ``main`` branch can move forward; pinning a revision means
your local cache + verified YAML stay coherent over time. If you later
download a different snapshot, re-run ``verify-sources --refresh
--only <id>`` to pin the new SHA.
