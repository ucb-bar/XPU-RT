# Phase C — Status

_Last updated: 2026-05-07_  ·  _Current head: `bdce554`_  ·  _Trust report: `/tmp/m52_trust/trust_report.md` (8/8 PASS, 30 contracts)_   ·   **🎉 Phase C complete (M-40 → M-52) — vertical slice 1 paper claim real, dispatch widened SYNC → ASYNC → CUDA**

This is the canonical Phase C tracker. Every Phase C milestone's done
condition includes updating this document with the new commit hash,
evidence paths, and test count. If the doc is not updated, the milestone
is not done — the M-31A `phase_c_status_consistency` audit gate fails.

## Paper-facing claim being built toward

> CompGen uses kernel contracts as the calling convention between Recipe IR,
> kernel providers, verifiers, and emitted glue. For a selected recipe, it
> materializes a canonical KernelContractV3, generates a shape-specialized
> kernel artifact via a sub-spawned Claude Code agent treated as one
> provider among several, verifies it against contract-derived obligations,
> binds it into an execution plan, emits a per-workload executor, and passes
> end-to-end differential testing.

The intermediate paper-claimable milestone is **M-49** (glue differential):
once green, the system can claim *"CompGen emits per-workload glue from a
validated execution plan; the generated executor calls verified
shape-specialized kernels, checks plan invariants at launch, and passes
end-to-end differential testing against the original model."*

Full plan: `~/.claude/plans/stateful-jumping-lovelace.md` (saved post-`/plan`).

## Architecture (one-line summary)

```
Recipe IR decision → KernelContractV3 → contract_hash → kernel-codegen task
  → provider response → contract-derived verifier checklist → certificate
  → ExecutionPlan.region_kernel_bindings → emitted plan executor
  → runtime plan assertions → glue differential
```

Dispatch widens at M-50..M-52 (SYNC → ASYNC → CUDA graph capture).

## Milestone table

Status legend: `planned` → `in_progress` → `complete` (tests green + commit landed + this doc updated).

| ID    | Name                                          | Status      | Commit  | Evidence                                                                              | Test count |
| ----- | --------------------------------------------- | ----------- | ------- | ------------------------------------------------------------------------------------- | ---------- |
| M-39  | Kernel-specialization request emitter (data-only prereq) | complete    | `c3b8a30` | `python/compgen/graph_compilation/kernel_specialization.py`, `tests/graph_compilation/test_kernel_specialization_request.py`, `docs/realness/m39_kernel_specialization_request.yaml` | 8 (tests/graph_compilation), 1347/7 (full graph_compilation suite) |
| M-40  | Contract materialization from Recipe op       | complete    | `b28b9de` | `python/compgen/kernels/contract_v3.py:from_recipe`, `python/compgen/graph_compilation/kernel_contract_materialization.py`, `tests/graph_compilation/test_kernel_contract_materialization.py`, `docs/realness/m40_contract_materialization.yaml` | 11 (M-40), 1138/83-hits-allowlisted, trust report 8/8 PASS |
| M-41  | Contract hash discipline                      | complete    | `dc756d9` | `python/compgen/graph_compilation/kernel_contract_materialization.py:hash_contract_from_run_dir`, `promotion_bridge.py` (legacy retired), `agent_decision.py`, `kernel_specialization.py`, `docs/realness/m41_contract_hash_discipline.yaml` | 35 affected tests pass, 0 derive_contract_hash production callers, 5-phase real-driven stress green (6/6 warm-cache hits preserved) |
| M-42  | Kernel-codegen task emitter (supersedes M-39) | complete    | `cc428fe` | `python/compgen/graph_compilation/kernel_codegen.py`, `kernel_specialization.py` (deprecated shim), `run.py` + `__main__.py` (boundary + flag), `tests/graph_compilation/test_kernel_codegen_request.py`, `docs/realness/m42_kernel_codegen_task.yaml` | 23 (M-40+M-42), 6/6 models clean migration (no legacy dir leak), 5-phase real-driven stress green |
| M-43  | Provider response schema + commit tool + 4 MCP tools | complete | `0df601b` | `python/compgen/graph_compilation/kernel_codegen_response.py`, `python/compgen/mcp/tools/kernel_codegen.py`, `tests/graph_compilation/test_kernel_codegen_response.py`, `docs/realness/m43_provider_response.yaml` | 14 (M-43), real-driven 4-phase end-to-end provider loop on merlin_mlp_wide (1 bug surfaced + fixed) |
| M-44  | Contract-driven verifier checklist (LOAD-BEARING) | complete | `ea69e04` | `python/compgen/kernels/contract_verifier.py`, `graph_compilation/kernel_codegen_response.py` (M-44 hook + reconstruct_contract), `tests/kernels/test_contract_verifier.py`, `docs/realness/m44_contract_driven_verifier.yaml` | 12 (M-44), 46 across M-40..M-44, real-driven stress: 5/5 models verified end-to-end, tampered shape correctly typed |
| M-45  | Kernel certificate                            | complete    | `62a2bc6` | `python/compgen/kernels/kernel_certificate.py`, `graph_compilation/kernel_codegen_response.py` (M-45 hook on verified path), `mcp/tools/kernel_codegen.py` (inspect surfaces cert + validation), `tests/kernels/test_kernel_certificate.py`, `docs/realness/m45_kernel_certificate.yaml` | 12 (M-45) + 46 from M-40..M-44 (58 total kernel-track tests), 5/5 models emit cert + validate; tamper → typed artifact_hash_drift |
| M-46  | Plan ↔ certified-kernels link                 | complete    | `0c80c28` | `python/compgen/runtime/execution_plan.py` (RegionKernelBinding + validate_with_run_dir), `python/compgen/graph_compilation/execution_plan_emit.py`, `run.py` + `__main__.py` (boundary + flag), `tests/runtime/test_region_kernel_binding.py`, `docs/realness/m46_plan_kernel_binding.yaml` | 10 (M-46) + 12 (M-45), 5/5 models flip unbound→bound on cert emit, tamper → typed artifact_hash_drift |
| M-47  | Python SYNC plan executor                     | complete    | `543edd4` | `python/compgen/runtime/glue_emit/{__init__,python_sync}.py`, `python/compgen/graph_compilation/run.py` (--stop-after glue-emit boundary), `tests/runtime/test_glue_emit_python_sync.py`, `docs/realness/m47_python_sync_executor.yaml` | 9 (M-47), real-driven stress: imported emitted module + ran compgen_run end-to-end, output=(16,32) Tensor |
| M-48  | Runtime plan assertions                       | complete    | `99ac9ff` | `python/compgen/runtime/glue_emit/plan_assertions.py`, `python/compgen/runtime/glue_emit/python_sync.py` (M-48 wiring), `tests/runtime/test_plan_assertions.py`, `docs/realness/m48_plan_assertions.yaml` | 7 (M-48), 9 typed PLAN_VIOLATION_<KIND> classes, real-driven stress: 5 fault-injection cases all fire correctly |
| M-49  | Glue differential — paper-facing              | complete    | `6b20c78` | `python/compgen/graph_compilation/glue_differential.py`, `python/compgen/graph_compilation/downstream_retry.py` (glue_differential row), `tests/graph_compilation/test_glue_differential.py`, `docs/realness/m49_glue_differential.yaml` | 6 (M-49), real-driven: merlin discharged_bit_equality 8/8, tiny_mlp discharged_tolerance_eps 8/8, tamper → fail+M-15B fires |
| M-50  | SetDispatchMode as Recipe IR decision         | complete    | `69b7121` | `python/compgen/graph_compilation/action_space.py` (Family 6 _gen_dispatch_modes), `python/compgen/kernels/contract_v3.py` (_resolve_dispatch_mode_override + from_recipe override kwarg), `python/compgen/graph_compilation/kernel_contract_materialization.py` (_dispatch_mode_override_for + materializer wiring), `tests/graph_compilation/test_set_dispatch_mode.py`, `docs/realness/m50_set_dispatch_mode.yaml` | 14 (M-50), real-driven stress: merlin_mlp_wide → 12 set_dispatch_mode candidates emitted (3 regions × 4 modes), 6 legal in agent_decision_request, persistent/inline emitted illegal-by-granularity, byte-stable across reruns, recipe_delta override flips contract dispatch.model SYNC→ASYNC |
| M-51  | CPU ASYNC + EventTensor executor              | complete    | `f453f4a` | `python/compgen/runtime/glue_emit/python_async.py`, `python/compgen/runtime/glue_emit/__init__.py` (re-export), `python/compgen/graph_compilation/run.py` (M-51 wired after M-47), `tests/runtime/test_glue_emit_python_async.py`, `docs/realness/m51_cpu_async_executor.yaml` | 10 (M-51), real-driven 3-phase stress (handshake / timeout / exception) green; M-50→M-51 chain: SetDispatchMode(async) recipe_delta materialises ASYNC contract; sync/async terminal-output parity verified |
| M-52  | CUDA ASYNC + graph capture executor           | complete    | `bdce554` | `python/compgen/runtime/glue_emit/python_cuda.py`, `python/compgen/runtime/glue_emit/__init__.py` (re-export), `python/compgen/graph_compilation/run.py` (M-52 wired after M-51 in glue-emit), `tests/runtime/test_glue_emit_python_cuda.py`, `docs/realness/m52_cuda_async_executor.yaml` | 12 (M-52), real-driven 5-phase stress: sync 2-region dispatch (3 sync calls), async 2-region handshake (order [matmul_0, bias_add_0]), capture multi-region rejected with typed ValueError, capture available (replay called), capture unavailable honest fallback (capture_status=unavailable_no_cuda). 666 runtime tests pass; trust report 8/8 PASS at 30 contracts. |

## Vertical-slice status

| Slice | Description                                           | Target milestone gate | Status   |
| ----- | ----------------------------------------------------- | --------------------- | -------- |
| 1     | merlin_mlp_wide on host_cpu via cffi-C, SYNC          | M-49                  | **complete (M-49 b28b9de paper-facing claim now real)** |
| 2     | proxy_vla on host_cpu (fusion path), SYNC             | M-49                  | planned  |
| 3     | merlin_mlp_wide on cuda_sm75 via Triton, SYNC + ASYNC | M-52                  | planned  |
| 4     | proxy_vla on host_cpu, ASYNC + EventTensor            | M-51                  | planned  |

## Open questions / blockers

- _(none — all design questions resolved during planning. Slot reserved for issues that surface during implementation.)_

## Honest residuals (cross-reference caveat ledger)

- `m15b_natural_failure_unreachable` (M-37.13, status `blocked_by_external` in `results/audit/_seed/caveat_ledger.json`) — Phase C may resolve this when real backend codegen produces genuine numerical disagreement past Higham's bound. M-44's contract-driven verifier closes the gap structurally even before that signal lands.
- _(future Phase C residuals appended here.)_

## Last 3 trust reports

Append-only log of full Phase C audit runs (commit + verdict + run path).

- `bdce554` (2026-05-07, **M-52 commit — Phase C complete**): trust report 8/8 PASS at `/tmp/m52_trust/trust_report.md` (30 contracts, 4 caveats, 9 negative controls). Real-driven 5-phase stress on a 2-region cuda_sm75 plan: Phase A (mode="sync") dispatch_calls=2, synchronize_calls=3, terminal=bias_out. Phase B (mode="async") observed order ['matmul_0','bias_add_0'] via EventTensor handshake. Phase C (capture multi-region) raised ValueError honestly. Phase D (capture available) capture_calls=1, replay_calls=1, capture_status=captured. Phase E (capture unavailable) honest fallback to single dispatch with capture_status=unavailable_no_cuda. 12 tests for M-52, 36 across M-50+M-51+M-52. **Phase C closed** — dispatch widened SYNC→ASYNC→CUDA-graph-capture; M-40..M-52 (13 milestones, 30 realness contracts) all complete.
- `f453f4a` (2026-05-07, M-51 commit): trust report 8/8 PASS at `/tmp/m51_trust/trust_report.md` (29 contracts, 4 caveats, 9 negative controls). Real-driven 3-phase stress: Phase A — 2-region linear plan with both bindings async; EventTensor handshake serialised matmul_0→bias_add_0 in observed order ['matmul_0','bias_add_0'], dispatch_calls=2, sync_called=True. Phase B — kernel that hangs; compgen_run_async(timeout_s=0.5) raised TimeoutError naming the worker thread; no deadlock. Phase C — kernel that raises; RuntimeError propagated to main caller; siblings released via EventTensor._cancel. M-50→M-51 chain: SetDispatchMode(async) op in merlin_mlp_wide candidate_selection.recipe_delta materialised contract.orchestration.dispatch.model = DispatchModel.ASYNC; canonical hash_contract = 30e8bed909330277. Sync/async terminal-output parity verified on 2-region linear plan. 10 tests for M-51, 24 across M-49+M-50+M-51.
- `69b7121` (2026-05-07, M-50 commit): trust report 8/8 PASS at `/tmp/m50_trust/trust_report.md` (28 contracts, 4 caveats, 9 negative controls). Real-driven stress on merlin_mlp_wide: 12 set_dispatch_mode candidates emitted (3 matmul regions × 4 DispatchModel values), 6 legal in agent_decision_request.candidate_ids_allowed (sync+async per region), PERSISTENT/INLINE correctly emitted illegal-by-granularity with typed reasons, byte-stable across reruns. Materialiser stress: SetDispatchMode(async) op in recipe_delta flips contract.orchestration.dispatch.model SYNC→ASYNC; PERSISTENT-on-NORMAL correctly returns None (defence-in-depth). 14 tests for M-50.
- `6b20c78` (2026-05-07, **M-49 paper-facing milestone**): trust report 8/8 PASS at `/tmp/m49_trust/trust_report.md` (27 contracts, 4 caveats, 9 negative controls). Real-driven stress: merlin_mlp_wide → discharged_bit_equality 8/8 cases bit-exact; tiny_mlp → discharged_tolerance_eps 8/8 within Higham bound; tamper → fail+M-15B detection. 6 tests for M-49, 21 across M-47..M-49. **Vertical slice 1 (merlin_mlp_wide on host_cpu via cffi-C, SYNC) is now complete**.
- `99ac9ff` (2026-05-07, M-48 commit): trust report 8/8 PASS at `/tmp/m48_trust/trust_report.md` (26 contracts, 4 caveats, 9 negative controls). Real-driven stress: 5 fault-injection cases (IO_TYPE / INPUT_COUNT / INPUT_SHAPE / INPUT_DTYPE / INPUT_BYTES) all fire typed PLAN_VIOLATION subclasses; well-formed io passes assertions + dispatches normally. 15 tests across M-47+M-48.
- `543edd4` (2026-05-07, M-47 commit): trust report 8/8 PASS at `/tmp/m47_trust/trust_report.md` (25 contracts, 4 caveats, 9 negative controls). Real-driven stress: pipeline → plan → glue-emit → import + RUN compgen_run on merlin_mlp_wide; dispatch_count=1, synchronize_called=True, output=Tensor(16,32). 9 tests for M-47, 18 across M-46+M-47.
- `0c80c28` (2026-05-07, M-46 commit): trust report 8/8 PASS at `/tmp/m46_trust/trust_report.md` (24 contracts, 4 caveats, 9 negative controls). Real-driven stress: phase 1 (no provider) all 6 models unbound; phase 2 (with provider) 5/5 set_tile_params flip to bound; phase 3 (tamper) artifact_hash_drift fires. 22 tests across M-45+M-46.
- `62a2bc6` (2026-05-07, M-45 commit): trust report 8/8 PASS at `/tmp/m45_trust/trust_report.md` (23 contracts, 4 caveats, 9 negative controls). Real-driven stress: 5/5 models emit certificate; paper_claimable=True (no fallback); tamper test (edit kernel.c post-cert) → typed `artifact_hash_drift`. 58 tests across M-40..M-45 pass.
- `ea69e04` (2026-05-07, M-44 commit): trust report 8/8 PASS at `/tmp/m44_trust/trust_report.md` (22 contracts, 4 caveats, 9 negative controls). Real-driven stress: 5/5 models verified end-to-end with 16 typed obligations each; tamper test (wrong shape) → typed shape_mismatch + retry. 46 tests across M-40..M-44 pass.
- `0df601b` (2026-05-07, M-43 commit): trust report 8/8 PASS at `/tmp/m43_trust/trust_report.md` (21 contracts, 4 caveats, 9 negative controls). Real-driven stress: 4-phase end-to-end provider loop on merlin_mlp_wide (invalid JSON → schema_invalid+retry; contract_hash mismatch → fatal; sandbox escape → fatal; well-formed → accepted+verifier_pending). 1 bug surfaced + fixed.
- `cc428fe` (2026-05-07, M-42 commit): trust report 8/8 PASS at `/tmp/m42_trust/trust_report.md` (20 contracts, 4 caveats, 9 negative controls). Real-driven stress: 5 phases (clean migration, request schema fidelity, sandbox readiness, kernel_facing leakage on disk, alias compat). 6/6 models clean migration (no legacy 04_kernel_specialization/ leak).
- `dc756d9` (2026-05-07, M-41 commit): trust report 8/8 PASS at `/tmp/m41_trust/trust_report.md` (18 contracts, 4 caveats, 9 negative controls). Real-driven stress: 5 phases (read/write parity, byte-stability, cold→warm, graceful degradation, full inspection harness). 6/6 warm-cache hits preserved.
- `b28b9de` (2026-05-07, M-40 commit): trust report 8/8 PASS at `/tmp/m40_trust/trust_report.md` (18 contracts, 4 caveats, 9 negative controls). Realness scan 1138 files, 83 hits all allowlisted. M-40 tests 11/11 pass.
- `d02ce8a` (2026-05-07, Phase C bootstrap): trust report 8/8 PASS at `/tmp/phase_c_bootstrap_trust/trust_report.md` (17 contracts, 4 caveats, 9 negative controls). Realness scan 1136 files, 83 hits all allowlisted.

## Subagent behavior contract (the load-bearing rules)

When invoked via `compgen_run_kernel_codegen_task` (M-43+), the spawned Claude Code agent receives:

- `04_kernel_codegen/requests/<task_id>.request.json` — the bounded task
- `04_kernel_codegen/views/<region_id>.kernel_facing.json` — the only contract projection it may read
- a sandboxed write directory: `04_kernel_codegen/artifacts/<task_id>/`
- a fixed list of allowed backends and required outputs

The subagent **must**: write `kernel_source` (kernel.py or kernel.c), `kernel_metadata.json` (symbol, args, shape, dtype, layout), `launch_config.json` (grid, block, smem), `provider_claims.json` (estimated_registers, expected_numerics).

The subagent **must not**: edit any source file under `python/compgen/`, mutate the contract or contract files, change tolerance, invent shape classes, claim success, or write outside the sandboxed directory.

If the subagent is uncertain whether a kernel is correct, it **must** still write the artifact and let the parent verifier decide. The subagent never marks success; it only proposes.

## How to update this document

1. When starting a milestone, change its row's status from `planned` to `in_progress`.
2. When completing a milestone, change status to `complete` and fill in: commit hash, evidence paths (sources + tests + realness contract), test count.
3. Update the header `Last updated` date and `Current head` commit.
4. After every Phase C audit run (`scripts/dev/build_trust_report.py`), append the result to "Last 3 trust reports" (keep the 3 most recent).
5. New honest residuals → both this doc AND `results/audit/_seed/caveat_ledger.json` (machine-readable).
6. New open questions / blockers → append to "Open questions / blockers"; remove when resolved.

The `phase_c_status_consistency` audit gate (added with M-43+) parses this file and verifies:
- every commit hash resolves
- every evidence path exists on disk
- every milestone marked `complete` has a matching `docs/realness/m<N>_*.yaml`
- the doc was updated in the same commit that flipped a milestone to `complete` (git log check)
