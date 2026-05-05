# Fresh-agent task: compile a holdout model

This is the canonical task a fresh Claude Code session is given to
prove the public repo surface is sufficient to drive a CompGen compile.

## Task

Compile `holdout_mlp_odd_shapes` for `host_cpu` using only the MCP
tools and documented skills bundled in this task pack.

## Acceptance

A run is *acceptable* iff exactly one of:

1. **Verified**: `verification_report.json` exists in the run dir and
   reports `pass` for every gate.
2. **Typed-blocked**: a typed exception from `compgen.runtime.errors`
   is raised, OR the M-15B downstream-gate rejection fires with a
   typed retry surface.

A run is *unacceptable* iff:

- A pipeline stage silently partial-passes without a verification gate
  reporting it.
- The agent invents a `candidate_id`, `pass_id`, or tile size that is
  not in the request's `candidate_ids_allowed` / `passes_allowed`.
- Source code is edited.

## Recording the outcome

After the session ends:

```bash
uv run python -m compgen.audit.fresh_agent_modes \
    record-manual-session-result \
    --ledger results/audit/<commit>/caveat_ledger.json \
    --mode fresh_claude \
    --success true \
    --evidence-paths /tmp/run/run_manifest.json \
    --evidence-paths /tmp/run/import_provenance.json \
    --notes "fresh session reached typed_blocked on holdout_mlp_odd_shapes"
```

## Why the holdout

The model uses K=63, N=129, M=257 — none divisible by common tile
sizes. A pipeline that hardcodes clean-divide assumptions will fail
silently here; the audit catches that.
