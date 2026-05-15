# Realness policy

The trust audit enforces a checkable definition of *fully
implemented*. Every shipped feature must satisfy this policy before its
claims are paper-eligible.

## The eight rules

A feature is *done* iff:

1. **Clean checkout rebuild.** Its artifacts produce from source on a
   fresh clone, not from checked-in outputs.
2. **No stubs / mocks / placeholders on production paths.**
   `compgen.audit.realness_scan` finds zero unallowlisted matches in
   `python/compgen`, `scripts/`, `docs/`, and `.claude/`.
3. **Production-import provenance clean.** A real run on the milestone's
   declared workload writes
   `<run_dir>/import_provenance.json` with
   `forbidden_modules_imported = []` and `evidence_mode = "real"`.
4. **At least one negative control fires.** Inject a specific fault
   that the gate is supposed to catch; the typed error must be raised.
5. **Fresh-agent reproducibility.** When Claude Code is expected to
   use the milestone, a fresh task-pack run (greedy baseline plus an
   operator-recorded fresh-Claude session) reaches an honest outcome.
6. **Caveat ledger up to date.** Every limitation that affects a
   paper-claimable row is in
   `results/audit/<commit>/caveat_ledger.json`, schema-valid, and
   freshly verified within the staleness window.
7. **Hash-replayable agent decisions.** Every agent decision the
   milestone makes is replayable from the saved
   `agent_decision_trace_<n>.json` alone — same inputs ⇒ same
   `decision_id` and same output hashes.
8. **Holdout / perturbation evidence.** The milestone is exercised on
   at least one holdout model (`holdout: true` in YAML) or runs through
   one perturbation (renamed regions, varied tile divisibility,
   corrupted promotion library) without a silent partial pass.

## How the audit checks each rule

- Rule 1 — `COMPGEN_FORCE_REBUILD=1` refuses to overwrite a non-empty
  `out_dir`. Audit-mode runs are operator-launched into clean dirs.
- Rule 2 — `uv run python scripts/dev/audit_realness.py` exits 0 only
  when every match is in `realness_allowlist.yaml` with a reason.
- Rule 3 — `uv run python scripts/dev/audit_production_imports.py
  <run_dir>` reads `import_provenance.json` and fails on any forbidden
  module.
- Rule 4 — `tests/audit/test_negative_controls.py` parametrizes over
  every gate; each row injects a specific fault and asserts the typed
  error fires. Aggregated by `compgen.audit.negative_controls.run_all_negative_controls`.
- Rule 5 — `tests/audit/test_greedy_baseline_reproducibility.py`
  builds a task pack and runs the greedy resolver against it. Operator
  records the fresh-Claude outcome via
  `compgen.audit.fresh_agent_modes.record_manual_session_result`.
- Rule 6 — `compgen.audit.caveat_ledger.CaveatLedger.validate(...)`
  rejects malformed rows; `tests/audit/test_caveat_ledger.py`
  enforces schema + staleness.
- Rule 7 — `compgen.audit.trace_replay.replay(...)` re-derives every
  hash; `scripts/dev/replay_agent_decision.py` is the operator CLI.
- Rule 8 — `tests/audit/test_holdout_models.py` runs every
  `holdout: true` YAML through capture+lowering and asserts honest
  outcome; `compgen.audit.perturbations` provides the perturbation
  utilities.

## The trust report

`uv run python scripts/dev/build_trust_report.py [--run-dir <run>]`
runs every gate and emits:

- `results/audit/<commit>/trust_report.json` — machine-readable.
- `results/audit/<commit>/trust_report.md` — single-page summary.

Overall pass requires zero failed gates. Skipped gates are tolerated
honestly (e.g., `import_provenance` is skipped when no run-dir is
supplied).

## Realness contracts

Each feature ships a `docs/realness/<feature_id>.yaml` declaring its
claim, realness level, forbidden constructs, and required evidence.
The file is the canonical claim; the audit proves or rejects it.

Realness levels (ascending strength):

| Level | Meaning | Paper-claimable? |
| --- | --- | --- |
| `schema_only` | Format exists, not consumed | No |
| `write_only` | Artifact emitted, not used downstream | No |
| `read_only` | Artifact consumed but does not affect behavior | Limited |
| `decision_affecting` | Artifact changes candidate / pass choice | Yes |
| `production_path` | Affects real end-to-end run | Yes |
| `hardware_backed` | Exercised with real kernel / profile / runtime evidence | Strongest |

The promotion-pipeline contracts are seeded under `docs/realness/`.
The audit layer's own contract is `docs/realness/m31a_audit_layer.yaml`.

## Honest residuals

Some controls cannot run in CI without defeating their purpose:

- **Fresh-Claude session** — spawning a Claude Code session from CI
  defeats the no-private-context goal. The greedy baseline is the
  contractual reproducibility floor; the fresh-Claude run is operator-
  driven on a quarterly cadence and recorded in the caveat ledger as
  `manual_fresh_claude_<timestamp>`.
- **Adversarial review** — by construction operator-driven. One
  fresh Claude session is given the red-team prompt
  ("Find stubs, mocks, hardcoded IDs, stale artifacts that aren't
  declared. You are not allowed to fix code.") and every high-severity
  finding becomes either a fix, a caveat-ledger row, or a rejected
  finding with evidence.
- **`portable` gate level** — requires real cuda hardware on the
  audit host. Tracked as `portable_gate_single_target` in the seed
  caveat ledger.

These residuals are *declared*, not hidden. The caveat ledger names
them; the trust report lists them; the policy doc explains why they
are operator-driven rather than CI-driven.
