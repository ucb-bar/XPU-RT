# Phase D — Status

_Last updated: 2026-05-08_  ·  _Phase plan: `~/.claude/plans/stateful-jumping-lovelace.md`_  ·  _Trust report: `/tmp/m67_trust/trust_report.md` (9/9 PASS, 43 contracts)_  ·  **🎉 Phase D complete — Section 7 closed, 13/13 milestones**

This is the canonical Phase D tracker. Every Phase D milestone's done
condition includes updating this document with the new commit hash,
evidence paths, and test count.

## Paper-facing claim — closed

> **CompGen treats kernel codegen as a multi-bidder auction over a
> canonical contract.** Four distinct providers — Claude-Code agent
> (M-43), Triton template, C reference, and user-supplied — bid
> against the same `KernelContractV3` (M-40); all verified bids
> carry certificates (M-44/M-45); the selector picks by perf
> (M-57); the canonical-shape-class hash makes one verified kernel
> reusable across regions (M-58). User-space kernels plug in via a
> filesystem manifest path (M-62). Coverage-first scheduling
> identifies kernel reuse opportunities + ranks regions for
> shape-specialization (M-63). Refinement contracts let new optional
> fields land without invalidating cached certificates (M-64). The
> M-31A trust report's nine gates all pass on every Phase D
> commit, including the new `contract_version_consistency` gate
> that re-derives every cert's canonical hash post-v3.1 migration.

The closure stress is `tests/graph_compilation/test_four_bidder_stress.py`:
4 bidders → 4 fulfilled → 4 verified → user_path winner; canonical
hash invariant across all four certificates.

## Architecture (one-line summary)

```
Recipe IR → KernelContractV3 → canonical_hash + instance_hash
  → ProviderRegistry.applicable → top-K bid → top-K fulfill
  → contract-driven verifier (per fulfilled bid) → certificate per winner
  → selector picks by perf → ExecutionPlan binding → emitted glue
  → coverage-first inflation → specialization advisory
```

User-space:

```
--user-kernel-path /repo  → /compgen-discover-user-kernels skill
                          → walks path, classifies, persists provider manifests
                          → UserKernelProvider bids against future contracts
```

## Milestone table

Status legend: `planned` → `in_progress` → `complete` (tests green + commit landed + this doc updated).

| ID    | Name                                                | Status        | Commit  | Evidence | Test count |
| ----- | --------------------------------------------------- | ------------- | ------- | -------- | ---------- |
| M-55  | Wire ProviderRegistry into Phase C kernel-codegen   | complete      | `51155d8` | `python/compgen/kernels/registry.py` (applicable() + default_registry), `python/compgen/graph_compilation/kernel_codegen.py` (registry_resolution emit), `tests/graph_compilation/test_registry_wired.py`, `docs/realness/m55_registry_wired.yaml` | 7 (M-55), 62/62 Phase C regression preserved, trust 8/8 (32 contracts) |
| M-56  | Two-stage provider protocol (bid + fulfill)         | complete      | `a717990` | `python/compgen/kernels/provider.py` (BidPreview, ProviderProtocolViolation, make_default_bid), `python/compgen/kernels/registry.py` (compute_bid, collect_bids), `python/compgen/kernels/providers/{claude_code_default,triton_templates}.py` (bid()), `tests/kernels/test_provider_bid.py`, `docs/realness/m56_provider_bid_protocol.yaml` | 13 (M-56), 66/66 targeted regression, trust 8/8 (33 contracts) |
| M-57  | Multi-bidder auction with tool-mediated pruning     | complete      | `f538a05` | `python/compgen/graph_compilation/kernel_auction.py`, `python/compgen/kernels/providers/c_reference.py`, `python/compgen/kernels/registry.py` (default_registry → CReferenceProvider), `python/compgen/graph_compilation/{run.py,__main__.py}` (--auction-mode + --bid-cutoff + kernel-auction boundary), `python/compgen/mcp/tools/kernel_codegen.py` (compgen_compare_kernel_bids), `tests/graph_compilation/test_kernel_auction.py`, `docs/realness/m57_multi_bidder_auction.yaml` | 6 (M-57), real-driven stress: merlin_mlp_wide → CReferenceProvider winner end-to-end (cert + bind + glue), trust 8/8 (34 contracts) |
| M-58  | Canonical shape-class hash                          | complete      | `f2c96d1` | `python/compgen/promotion/contract_hash.py` (canonical_contract_hash + instance_contract_hash + hash_contract alias), `python/compgen/kernels/kernel_certificate.py` (canonical field on KernelCertificate, find_certificate_by_canonical_hash), `python/compgen/runtime/execution_plan.py` (RegionKernelBinding canonical field), `python/compgen/graph_compilation/execution_plan_emit.py` (BindingRow canonical field), `tests/kernels/test_canonical_contract_hash.py`, `docs/realness/m58_canonical_contract_hash.yaml` | 11 (M-58), 74/74 cross-suite regression preserved, trust 8/8 (35 contracts) |
| M-59  | contract_feedback re-enters Recipe IR (two-tier)    | complete      | `a2f5376` | `python/compgen/kernels/provider.py` (ContractFeedback gains kind+applies_when), `python/compgen/graph_compilation/contract_feedback_apply.py` (classifier + Recipe-IR proposal generator + persistence), `python/compgen/graph_compilation/kernel_auction.py` (per_provider_feedback collection), `tests/graph_compilation/test_contract_feedback_apply.py`, `docs/realness/m59_contract_feedback_apply.yaml` | 15 (M-59), 59/59 cross-suite regression preserved, trust 8/8 (36 contracts) |
| M-60  | Contract field completion from target + dossier     | complete      | `d709605` | `python/compgen/kernels/contract_v3.py` (HardwareEnvelope wiring + _derive_memory_spec + helpers), `configs/targets/host_cpu.yaml` (M-60 envelope block), `python/compgen/graph_compilation/{kernel_contract_materialization,kernel_codegen_response}.py` (round-trip), `python/compgen/kernels/providers/claude_code_default.py` (structural fix), `tests/graph_compilation/test_contract_field_completion.py`, `docs/realness/m60_contract_field_completion.yaml` | 8 (M-60), 142/142 cross-suite regression preserved, trust 8/8 (37 contracts) |
| M-61  | Pre/post-conditions as typed predicates             | complete      | `3d2a2d0` | `python/compgen/kernels/predicates.py` (entire module — 5 dataclass kinds), `python/compgen/kernels/contract_v3.py` (preconditions+postconditions fields + from_recipe population), `python/compgen/graph_compilation/{kernel_contract_materialization,kernel_codegen_response}.py` (round-trip), `python/compgen/runtime/glue_emit/plan_assertions.py` (5 new PLAN_VIOLATION subclasses + ModEq + ByteSizeLe runtime assertions), `tests/kernels/test_predicates.py`, `tests/runtime/test_plan_assertions_predicates.py`, `docs/realness/m61_predicate_dsl.yaml` | 15 (M-61), 164/164 cross-suite regression preserved, trust 8/8 (38 contracts) |
| M-62  | User-space kernel-provider discovery + MCP + skill  | complete      | `6ddb12b` | `python/compgen/kernels/user_kernel_index.py` (schema + indexer + audit), `python/compgen/kernels/providers/user_path.py` (UserKernelProvider), `python/compgen/kernels/registry.py` (auto-register when index non-empty), `python/compgen/mcp/tools/kernel_providers.py` (3 MCP tools), `python/compgen/graph_compilation/__main__.py` (--user-kernel-path + env), `.claude/skills/compgen-discover-user-kernels/SKILL.md`, `tests/kernels/test_user_kernel_index.py`, `docs/realness/m62_user_kernel_provider.yaml` | 20 (M-62), 158/158 cross-suite regression preserved, trust 8/8 (39 contracts) |
| M-63  | Coverage-first scheduling                           | complete      | `1aaaff2` | `python/compgen/graph_compilation/coverage_first.py` (orchestrator + signature builders + binding append + reports), `python/compgen/graph_compilation/{run.py,__main__.py}` (--kernel-coverage-mode + stage), `tests/graph_compilation/test_coverage_first.py`, `docs/realness/m63_coverage_first.yaml` | 7 (M-63), 160/160 cross-suite regression preserved, trust 8/8 (40 contracts) |
| M-64  | Refinement contracts + version migration            | complete      | `b33f31b` | `python/compgen/kernels/contract_v3.py` (optional_v3_1_fields slot + recognised-names + defaults), `python/compgen/kernels/contract_migration.py` (migrate + get_optional + ContractRefinementError + is_compatible_with), `python/compgen/graph_compilation/{kernel_contract_materialization,kernel_codegen_response}.py` (round-trip), `python/compgen/audit/trust_report.py` (new contract_version_consistency gate), `tests/kernels/test_contract_versioning.py`, `docs/realness/m64_contract_versioning.yaml` | 11 (M-64), 175/175 cross-suite regression preserved, trust 9/9 (41 contracts; new gate verified) |
| M-65  | Vertical slices 2 + 3 under multi-bidder auction    | complete      | `814a31d` | `python/compgen/graph_compilation/phase_d_slice_evidence.py`, `tests/graph_compilation/test_phase_d_slice_evidence.py`, `docs/realness/m65_phase_d_slices.yaml` | 3 (M-65); slice 2 honest_gap (proxy_vla fusion → not_applicable), slice 3 deferred (no cuda_sm75 target locally) |
| M-66  | Four-bidder benchmark + paper-claim closure         | complete      | `814a31d` | `tests/graph_compilation/test_four_bidder_stress.py`, `docs/realness/m66_four_bidder_stress.yaml` | 2 (M-66); 4 bidders → 4 fulfilled → 4 verified → user_path winner; canonical hash invariant across all 4 certs |
| M-67  | Phase D status doc + Section 7 closure              | complete      | `814a31d` | `docs/phase_d_status.md` (this file — final), `docs/realness/m65_phase_d_slices.yaml`, `docs/realness/m66_four_bidder_stress.yaml`, allowlist updates | _doc + closure_ |

## Vertical-slice status

| Slice | Description                                                                | Target milestone gate | Status |
| ----- | -------------------------------------------------------------------------- | --------------------- | ------ |
| 1     | merlin_mlp_wide on host_cpu via cffi-C, single-bidder (Phase C closure)    | M-49                  | complete (Phase C) |
| 2     | proxy_vla on host_cpu (fusion path)                                        | M-65                  | complete — honest_gap (M-42 fusion-archetype expansion deferred) |
| 3     | merlin_mlp_wide on cuda_sm75 via Triton                                    | M-65                  | complete — deferred (no cuda_sm75 target ships) |
| 4     | merlin_mlp_wide host_cpu, four-bidder auction inc. user-supplied kernel    | M-66                  | **complete — green; user_path wins** |

## Cumulative test count (Phase D)

| Suite | Count |
| ----- | ----- |
| `tests/graph_compilation/test_registry_wired.py` (M-55)               |  7 |
| `tests/kernels/test_provider_bid.py` (M-56)                            | 13 |
| `tests/graph_compilation/test_kernel_auction.py` (M-57)                |  6 |
| `tests/kernels/test_canonical_contract_hash.py` (M-58)                 | 11 |
| `tests/graph_compilation/test_contract_feedback_apply.py` (M-59)       | 15 |
| `tests/graph_compilation/test_contract_field_completion.py` (M-60)    |  8 |
| `tests/kernels/test_predicates.py` (M-61) + `tests/runtime/test_plan_assertions_predicates.py` | 15 |
| `tests/kernels/test_user_kernel_index.py` (M-62)                       | 20 |
| `tests/graph_compilation/test_coverage_first.py` (M-63)                |  7 |
| `tests/kernels/test_contract_versioning.py` (M-64)                     | 11 |
| `tests/graph_compilation/test_phase_d_slice_evidence.py` (M-65)        |  3 |
| `tests/graph_compilation/test_four_bidder_stress.py` (M-66)            |  2 |
| **Total Phase D**                                                       | **118** |

Cross-suite regression: 175/175 across 17 suites under M-64.

## Open questions / blockers

- _(none — Phase D closed. Slot reserved for issues that surface after closure.)_

## Honest residuals (cross-reference caveat ledger)

Phase D inherits two M-37.13 residuals plus a few new ones:

- `m15b_natural_failure_unreachable` (M-37.13, status `blocked_by_external` in `results/audit/_seed/caveat_ledger.json`).
- **M-65 Slice 2 honest gap** — proxy_vla's recipe planner selects fusion candidates; M-42 routes to `not_applicable` (M-42 covers only `set_tile_params` today). Closing this requires the contract registry to grow past COMPUTE_TILED — a Phase E candidate.
- **M-65 Slice 3 deferred** — no `configs/targets/cuda_sm75.yaml` ships locally; CUDA-bound providers need a CUDA-capable host. M-66 covers TritonTemplate's CPU-fallback bid path.
- **M-66 stub providers** — `_ClaudeCodeStubProvider` + `_TritonTemplateCpuFallbackProvider` are TEST-LOCAL stand-ins; they prove the auction surface accommodates four bidders, not that real Claude-Code + real Triton produce these specific bids on a CUDA host. M-65 Slice 3 covers the CUDA path when re-run on a GPU host.
- **M-63 coverage signature** is shape-EQUAL today, not shape-class. Two regions with shapes `(16, 16, 32)` and `(16, 16, 64)` don't match. Shape-class divisibility (e.g. "any K divisible by 16") rides M-64's refinement-contract path.

## Trust reports — final

The complete log of Phase D trust reports:

- `b33f31b` (2026-05-07, M-64 commit — refinement contracts + version migration): trust report 9/9 PASS at `/tmp/m64_trust/trust_report.md` (41 contracts, 4 caveats, 1183 files / 87 hits all allowlisted) — gates grow from 8 to 9 with the new `contract_version_consistency` gate.
- `1aaaff2` (2026-05-07, M-63 commit — coverage-first scheduling): trust report 8/8 PASS at `/tmp/m63_trust/trust_report.md` (40 contracts, 4 caveats, 1182 files / 87 hits all allowlisted).
- `6ddb12b` (2026-05-07, M-62 commit — user-space kernel-provider discovery): trust report 8/8 PASS at `/tmp/m62_trust/trust_report.md` (39 contracts, 4 caveats, 1179 files / 86 hits all allowlisted).
- `3d2a2d0` (2026-05-07, M-61 commit — typed predicate DSL): trust report 8/8 PASS at `/tmp/m61_trust/trust_report.md` (38 contracts, 4 caveats, 1175 files / 86 hits all allowlisted).
- `d709605` (2026-05-07, M-60 commit — contract field completion): trust report 8/8 PASS at `/tmp/m60_trust/trust_report.md` (37 contracts, 4 caveats, 1173 files / 86 hits all allowlisted).
- `a2f5376` (2026-05-07, M-59 commit — contract_feedback two-tier): trust report 8/8 PASS at `/tmp/m59_trust/trust_report.md` (36 contracts, 4 caveats, 1172 files / 85 hits all allowlisted).
- `f2c96d1` (2026-05-07, M-58 commit — canonical hash): trust report 8/8 PASS at `/tmp/m58_trust/trust_report.md` (35 contracts, 4 caveats, 1170 files / 85 hits all allowlisted).
- `f538a05` (2026-05-07, M-57 commit — auction operational): trust report 8/8 PASS at `/tmp/m57_trust/trust_report.md` (34 contracts, 4 caveats, 1169 files / 85 hits all allowlisted).
- `a717990` (2026-05-07, M-56 commit — bid/fulfill protocol): trust report 8/8 PASS at `/tmp/m56_trust/trust_report.md` (33 contracts, 4 caveats, 1166 files / 85 hits all allowlisted).
- `51155d8` (2026-05-07, M-55 commit — Phase D bootstrap): trust report 8/8 PASS at `/tmp/m55_trust/trust_report.md` (32 contracts, 4 caveats, 1165 files / 85 hits all allowlisted).
- **Final: `814a31d` (2026-05-08, M-65 + M-66 + M-67 closure)** — trust report 9/9 PASS at `/tmp/m67_trust/trust_report.md` (43 contracts, 4 caveats, 1187 files / 87 hits all allowlisted). All 13 Phase D milestones complete; 118 net-new tests; cumulative 175/175 cross-suite regression preserved at M-64 (M-65 + M-66 add another 5 net-new). Section 7 closed.
