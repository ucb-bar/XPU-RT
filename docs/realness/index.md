# Realness Contracts

Every paper-claimable feature in XPU-RT has a **realness contract** —
a YAML record declaring the feature's realness level (the strength of
the claim it backs), the forbidden constructs, and the evidence required
to keep the claim valid.

| Level | Meaning | Paper-claimable? |
|---|---|---|
| `schema_only` | Type definitions only; no runtime behaviour. | ❌ |
| `write_only` | Emits artefacts but no reader consumes them yet. | ❌ |
| `read_only` | Reads artefacts but doesn't affect decisions. | ❌ |
| `decision_affecting` | Output affects compiler/scheduler decisions. | ✅ |
| `production_path` | Exercised in CI on real workloads. | ✅ |
| `hardware_backed` | Closed-loop measurement against silicon. | ✅ |

Each contract lives at `docs/realness/<feature_id>.yaml` and is enforced
by `xpu_rt.audit.contracts` + the trust report gates.

## Contracts in this directory

The files in this directory are the source of truth. Each YAML lists:

- `feature_id`: stable name (e.g. `m26_promotion_bridge`).
- `realness_level`: one of the six levels above.
- `forbidden_constructs`: regex patterns that, if matched in the code,
  invalidate the claim (e.g. `# for now`, `TODO`, `NotImplementedError`).
- `required_evidence`: artefacts that must exist + their typed content
  (e.g. a JSON file with a specific schema).

See `xpu_rt.audit.contracts.load_all_contracts()` for the loader and
`xpu_rt.audit.realness_scan` for the enforcement scan.

## Related

- `xpu_rt.audit.caveat_ledger` — known limitations that block specific
  claims; each entry names the contract it affects.
- `xpu_rt.audit.trust_report` — single-page aggregator that runs every
  contract gate.
- `architecture/agentic_e2e_audit.md` — the broader audit framework.
