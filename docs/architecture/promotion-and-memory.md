# Section 19 — Recipe promotion and optimization memory

This document describes the seam that turns a single Phase B run from a
disposable artifact pile into a unit of reusable optimization knowledge.
After M-26..M-30, every successful CompGen run lands a promoted recipe
in `.compgen_cache/recipes/`, and a *future* run on a different model
with the same region pattern surfaces that recipe as a preferred
candidate before asking an agent to choose anew.

## Falsifiable claim

> Cold-run vs warm-run on the same model suite shows
> `fresh_emit_count_warm < fresh_emit_count_cold` and
> `gemini_token_delta < 0`, while every correctness gate in
> `verification_report.json` still passes.

The measurement script
(`scripts/dev/measure_promotion_efficiency.py`) is the test harness.
The aggregator (`compgen.graph_compilation.efficiency_report`) is the
pure-function implementation.

## State of play (after M-30)

| concern | answer |
|---|---|
| Where does a promoted recipe live? | `.compgen_cache/recipes/<key>/manifest.json` plus `promoted_recipe.json` sidecar (M-26). |
| What is the cache key? | Two-tier: model-level `RecipeKey(target_hash, model_hash, objective_hash, version)` for directory naming; pattern-level `(contract_hash, region_signature)` for cross-model reuse. The pattern tier rides in the sidecar + SQLite `memory.promotions` index. |
| What evidence ships with a recipe? | The `PromotedRecipe` body: `recipe_id`, `recipe_signature`, `recipe_ir_path`, `evidence_summary`, `applies_when` (fact predicates), `fallback_chain`, `certificates`, `validity` (target_class + dtype + layout), `gate_level`. |
| Who decides the gate level? | M-29 `evaluate_gate(run_dir, ...)` in `compgen.promotion.gates`, six levels: `observed → verified_fx → verified_kernel → characterized → promoted → portable`. |
| Who emits a promoted recipe? | The M-26 bridge in `compgen.graph_compilation.promotion_bridge.emit()`, called from `run.py` after Phase B passes M-15B. |
| Who reads promoted recipes back? | The M-28 retrieval path: `compgen.graph_compilation.promotion_retrieval.retrieve_for_region()`, called by `agent_decision.build_agent_decision_request()` per region. |
| Where does the Recipe IR fit? | M-27 extends `recipe.promote` with `recipe_signature`, `applies_when`, `evidence_summary`, `fallback_chain`, `target_class` so the IR carries the full pattern. The on-disk sidecar is a JSON projection of those attrs. |

## The two-tier cache key

A single `RecipeKey` directory name is **not** rich enough to support
cross-model reuse: `model_hash` is part of it, so two runs on
different models never share keys. The fix lives *outside* the
directory name:

- `RecipeKey.target_hash`, `RecipeKey.model_hash`,
  `RecipeKey.objective_hash`, `RecipeKey.version` — directory
  naming, unchanged from before M-26 to keep the existing
  `RecipeCache.list_recipes` parser working.
- `RecipeKey.contract_hash` — exact-kernel identity. Promotion
  retrieval uses this for "have we already compiled this exact
  kernel?" lookups.
- `RecipeKey.region_signature` — pattern identity. Computed by
  `compgen.promotion.region_signature.hash_region_signature` over
  `(op_family, dtype, layout, abstracted_shape, target_class)`. The
  abstracted shape supports `int`, `None` (dynamic),
  `{"mod": k}` (any size divisible by `k`); two regions hash the
  same iff their shape patterns match under abstraction.

Both extra dimensions ride in:

1. **The recipe directory's `promoted_recipe.json` sidecar** —
   authoritative on-disk record.
2. **`memory.promotions.region_signature` / `.contract_hash`** —
   indexed SQLite columns added in M-26's idempotent migration.
   The `idx_promotions_region` and `idx_promotions_contract` indexes
   make pattern-level lookups O(log N).

## The gate ladder

`PromotionLevel` is ordered low → high; each level requires strictly
more evidence than the one below.

| level | floor evidence |
|---|---|
| `observed` | `candidate_selection.json` records a non-null `selected_candidate_id`. |
| `verified_fx` | M-12 / M-16.2 / M-09 differential report has `status ∈ {pass, tolerance_eps, bit_equality}`. |
| `verified_kernel` | M-19 / M-20 / M-23 compiled-kernel differential pass. |
| `characterized` | M-21 analytical cost AND (M-22 OR M-22.1) measured cost both present. |
| `promoted` | M-17.1 + M-24 readiness matrices `overall=pass` AND certificates recorded under `04_promotion/verification_report.json`. **This is the default cutoff** — promotions below this level surface in retrieval, but the agent should rank them lower. |
| `portable` | ≥2 distinct `target_class` values observed in the recipe library for the same `region_signature`. The strongest claim Section 19 makes. |

Stripping evidence demotes the level monotonically — verified by
`tests/promotion/test_gates.py::test_stripping_evidence_demotes_monotonically`.

## R009 hash-chain safety

Phase B writes immutable per-stage trees rooted at
`run_dir/<NN>_<stage>/`, each pinned by its stage record's
`output_hash`. Any post-stage write into an earlier stage's tree
breaks the R009 hash chain.

The bridge sidesteps this by writing every Section 19 artifact
under a brand-new `04_promotion/` subdir not covered by any
earlier stage's `output_hash`:

- `04_promotion/verification_report.json` — synthesised gate input
  (M-26).
- `04_promotion/efficiency_pack.json` — efficiency aggregate
  (M-30).

Same pattern M-10B / M-13 / M-22.1 use.

## On-disk layout

```
.compgen_cache/recipes/<target>_<model>_<obj>_v1/
├── manifest.json              # Bundle (existing format)
├── promoted_recipe.json       # M-26 sidecar — two-tier key + recipe body
├── 01_payload_lowering/       # Copied from run dir
│   └── payload.mlir
├── 03_recipe_planning/        # Copied from run dir
│   ├── recipe.mlir
│   ├── candidate_selection.json
│   └── ...
└── 04_promotion/              # Copied from run dir
    ├── verification_report.json
    └── efficiency_pack.json    # M-30, when emit_efficiency_pack ran

.compgen_cache/recipes/audit.jsonl  # Promotion audit log (gate_level
                                     # included since M-29)

.compgen_cache/memory.db            # SQLite — promotions indexed by
                                     # region_signature + contract_hash
```

## Reuse path (warm cache, agent's view)

1. Phase B runs through M-15B as usual; produces the standard
   `agent_decision_request.json` skeleton.
2. For each visible region, `agent_decision.py` calls
   `derive_region_signature` to recompute the M-26 region pattern,
   then `promotion_retrieval.retrieve_for_region()` to find matching
   promoted recipes.
3. Each region's `promoted_candidates` block in the request lists
   matches, ordered by match strength (`exact_contract` before
   `region_pattern`). Each carries `gate_level` + `evidence_summary`
   so the agent can rank them.
4. Promoted candidates do **not** join `candidate_ids_allowed`. The
   legal-candidate gate stays narrow. The agent surfaces a promoted
   recipe by selecting the legal candidate it references —
   `promoted_candidates[*]` exists only to tell the agent "this
   legal candidate is backed by prior evidence; rank it first."

## Falsifiability check

`scripts/dev/measure_promotion_efficiency.py` runs the user-supplied
model list cold (empty library) and warm (post-cold library) and
emits:

- `promotion_efficiency_pack.json` — per-model
  `EfficiencyDelta` with `fresh_emit_delta`, `gemini_token_delta`,
  `claim_supported`.
- `promotion_efficiency_pack.md` — paper-ready table.

Exit code is 0 iff **every** model satisfies the claim. Honest
errors (a model that hits an M-15B downstream rejection unrelated
to Section 19) are recorded in `errors[]` rather than aborting.

## Limitations and open work

- `applies_when` is currently empty in the M-26 sidecar. M-27 added
  the IR slot for it, but the bridge doesn't yet derive fact
  predicates from the dossier. M-28 retrieval honours an empty
  `applies_when` as "unconditionally applicable" — likely too
  permissive for cuda kernels. Short follow-up.
- `contract_hash` is empty in M-26 sidecars. The kernel contract is
  defined and hashable
  (`compgen.promotion.contract_hash.hash_contract`), but the bridge
  doesn't yet thread the M-23 / M-24 kernel contract objects through
  to the bridge. Wiring exists for M-28 retrieval to use
  `contract_hash` once the bridge starts populating it.
- `llm_graph_view.json` overlay deferred. The agent's primary
  surface is `agent_decision_request.json`, which already carries
  `promoted_candidates` per region; the redundant overlay can land
  later without changing the contract.
- `targets/maturity.py` is unchanged; it represents *target-package*
  maturity (a separate concept from recipe promotion) and is
  referenced from `targets/package.py` and
  `targetgen/verification_ladder.py`. The original M-29 plan
  proposed deleting it; audit showed this would break unrelated
  tests, so it stays.

## Change history

| milestone | what | files |
|---|---|---|
| M-26 | Phase B → promotion bridge (write side) | `promotion/contract_hash.py`, `promotion/region_signature.py`, `graph_compilation/promotion_bridge.py`; `RecipeKey` + `Promotion` schema. |
| M-27 | Recipe IR additions for promoted recipes | `ir/recipe/ops_provenance.py` (`PromoteOp` extended), `serialize.py`, `lower.py`, `validate.py`. |
| M-28 | Retrieval path in agent decision (read side) | `graph_compilation/promotion_retrieval.py`, `agent_decision.py` per-region `promoted_candidates`. |
| M-29 | Promotion gates ladder | `promotion/gates.py`, bridge wires gate_level into sidecar + memory + audit. |
| M-30 | Falsifiable reuse-efficiency claim + this doc | `graph_compilation/efficiency_report.py`, `scripts/dev/measure_promotion_efficiency.py`. |
