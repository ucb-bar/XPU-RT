# Extension / Provider / Dialect / Pass-Tool Architecture

> Companion to [`extension-points.md`](extension-points.md) (user-facing how-to).
> This document is the *enforceable contract* every kernel provider, MLIR
> dialect lowering, agent-callable pass, and user extension must obey.

## Core rule

> **Agents may propose tools, passes, dialect lowerings, and kernel providers.
> CompGen only executes certified artifacts.**

Cards are not evidence of support. Provider results are not certificates.
Pass-tool outputs are Recipe-IR deltas, not direct IR mutations. Every
optional backend must produce a typed status — there is no silent
disappearance, no silent fallback, and no fake success.

## Three planes

```
┌──────────────────────────────────────────────────────────────┐
│ Agent plane                                                  │
│  - Claude Code / Codex via MCP                               │
│  - PyPI compgen package                                      │
│  - .rcg-artifacts/extensions/<id>/  (user-writable, sandboxed)│
└──────────────────────────────────────────────────────────────┘
            │ tool calls / extension files
            ▼
┌──────────────────────────────────────────────────────────────┐
│ Control plane                                                │
│  - Multi-level analysis snapshots                            │
│  - Provider / Target / Dialect / Pass-tool registries        │
│  - Probes (typed blocked statuses)                           │
│  - Extension manifest + sandbox enforcement                  │
│  - Contract verifier → certificate                           │
└──────────────────────────────────────────────────────────────┘
            │ certified artifacts only
            ▼
┌──────────────────────────────────────────────────────────────┐
│ Execution plane                                              │
│  - Generated kernels (cffi-C, Triton, dialect lowerings)     │
│  - Execution plan + emitted glue                             │
│  - Runtime / profile feedback                                │
└──────────────────────────────────────────────────────────────┘
```

## The ten hard rules

These are enforced by `scripts/dev/audit_extension_architecture.py` and
`tests/architecture/test_extension_architecture_guard.py`.
Violation of any rule fails CI.

1. **Providers do not certify themselves.** A `ProviderResult` is never a
   `Certificate`. The verifier owns the certificate path. Code that
   treats a `ProviderResult.status == "generated"` as proof of correctness
   is a hard error.
2. **Providers may only write under their assigned `artifact_dir`.** No
   writes to `python/compgen/`, `configs/`, `payload.mlir`, contract
   files, recipe IR, run manifests, or any parent via `..`.
3. **User extensions may not mutate Payload IR, Recipe IR, contracts,
   manifests, or run ledgers directly.** They emit proposals; the core
   accepts or rejects them through the verifier.
4. **Pass tools produce Recipe-IR deltas, never direct IR mutations.**
   `pass_tool_result_v1.recipe_delta` is the *only* legal channel.
5. **Missing SDKs, hardware, licenses, or packages produce typed
   `blocked` statuses** with a typed `blocked_reason` — never a crash and
   never a silent disappearance.
6. **Provider cards are not evidence of support.** Only verified /
   certified artifacts are evidence. Cards with `integration_level:
   card_only` or `probe` are never `paper_claimable: true`.
7. **Core code depends on provider interfaces, not implementations.**
   `from cuda_tile import ...` outside `providers/adapters/*` is a hard
   error.
8. **Optional providers must not be imported at core module import
   time.** Imports are deferred to `probe()` / `propose()`.
9. **Every provider decision is auditable via the chain**
   `ProviderProbeResult → BidPreview → ProviderResult → VerifierReport →
   Certificate`. Skipping any link breaks the audit trail.
10. **No generated artifact is paper-claimable unless the evidence pack
    contains a certificate or typed execution report.** "Implemented"
    in a claim matrix requires a concrete evidence-artifact path.

## Five-level integration ladder

Every `ProviderCard`, `TargetCard`, `DialectProviderCard`, and
`PassToolCard` declares one of:

| Level         | Meaning                                                     | `paper_claimable` |
|---------------|-------------------------------------------------------------|-------------------|
| `card_only`   | CompGen knows this provider exists.                         | false             |
| `probe`       | CompGen can detect installed-or-blocked + emit typed status.| false             |
| `generate`    | Provider emits source / IR artifacts.                       | false             |
| `verify`      | Provider compiles, runs, differentials against a reference. | true (with caveats) |
| `promote`     | Provider produces reusable certified artifacts.             | true              |

Claiming `verify` or `promote` without the corresponding evidence
artifact is a hard error caught by the architecture audit.

## Typed enums

### `ProviderProbeStatus`

```
available
blocked
unsupported
probe_error
not_installed
```

### `BlockedReason`

```
env_missing
python_package_missing
command_missing
hardware_unavailable
sdk_missing
license_missing
version_mismatch
unsupported_platform
unsupported_contract_kind
probe_exception
```

Any status or reason outside these sets is rejected by the manifest
validator.

## Unsupported-op / extension-task flow

```
1. Pipeline reaches graph analysis.
2. unsupported_ops.json records op, shape, dtype, source location,
   closest supported patterns.
3. Agent sees unsupported-op gap in llm_graph_view.
4. Agent calls compgen_emit_extension_task (MCP).
5. Task asks for one of:
     - new kernel provider
     - new pass tool
     - new dialect lowering
     - new kernel template
     - new contract rule
6. Claude Code / user writes files under
   .rcg-artifacts/extensions/<task_id>/.
7. CompGen probes the extension.
8. CompGen validates manifest + sandbox paths.
9. CompGen runs unit smoke tests.
10. CompGen runs contract-derived verification.
11. If accepted, extension is registered for this run.
12. Pipeline resumes.
```

The task artifact (`extension_task_v1`) carries the
`kernel_facing_contract`, the `region_dossier`, the
`payload_ir_summary`, the `allowed_outputs`, the `forbidden` actions,
and the `verification_required` list. The extension response is
committed via `compgen_commit_extension_response` and only registered
after probe + sandbox + verifier all return green.

## Object types

| Type | Purpose |
|---|---|
| `ProviderCard` | Static declaration of a kernel provider: id, integration_level, target_families, contract_kinds, emits, probe spec, paper_claimable. |
| `TargetCard` | Static declaration of a target: family, vendor, dispatch_modes, memory_tiers. |
| `DialectProviderCard` | Static declaration of a dialect provider: dialect_name, consumes, emits, entrypoint. |
| `PassToolCard` | Static declaration of an agent-callable pass: phase, reads, writes, allowed_recipe_ops, refinement, verifier. |
| `ExtensionManifest` | Bundle of cards declared by a user-installed extension; includes `security.sandbox_required`, `security.allowed_write_root`. |
| `ProviderProbeResult` | Typed result of `provider.probe()`. Carries availability, version, supports, blocked_reason. |
| `BidPreview` | Typed result of `provider.can_bid(contract, target)` — used by routing. |
| `ProviderResult` | Typed result of `provider.propose(request)` — artifacts + claims + contract_feedback. **Not a certificate.** |
| `VerificationCertificate` | Output of the verifier — the only legal proof of correctness. |
| `ExtensionTask` | `extension_task_v1` — unsupported-op or provider-gap task emitted to `.rcg-artifacts/tasks/<task_id>/`. |

## Related documents

- [`extension-points.md`](extension-points.md) — user-facing extension
  authoring how-to.
- [`add-a-new-target.md`](add-a-new-target.md) — target-package walkthrough
  that this architecture generalizes.
- [`promotion-and-memory.md`](promotion-and-memory.md) — promoted-recipe
  cache; certified artifacts feed into here.

## Realness contracts

Each extension-substrate component has a realness contract under
`docs/realness/`:

```
docs/realness/m79_extension_substrate.yaml
docs/realness/m80_extension_cards.yaml
docs/realness/m81_extension_probe.yaml
docs/realness/m82_provider_normalization.yaml
docs/realness/m83_extension_sandbox.yaml
docs/realness/m84_unsupported_op_task.yaml
docs/realness/m85_pass_tool_registry.yaml
docs/realness/m86_multi_level_analysis_snapshots.yaml
docs/realness/m87_extension_evidence_pack.yaml
```

Contracts start at `realness_level: schema_only` and are upgraded to
`production_path` (or `hardware_backed`) once the implementation lands
and evidence exists.
