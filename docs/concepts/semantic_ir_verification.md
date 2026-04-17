# Semantic IR Verification

CompGen's **Semantic IR** (Layer 3) provides formal verification for compiler
transforms. It is powered by the xDSL SMT dialect and Z3.

## Architecture

```
Payload IR (Layer 1)    Recipe IR (Layer 2)     Semantic IR (Layer 3)
─────────────────────   ──────────────────────  ────────────────────────
Canonical computation   LLM-facing decisions    Verification/trust layer
arith, func, linalg,    TileOp, FuseOp,         TranslationValidation,
memref, tile, accel      PlaceOnDeviceOp,        PeepholeVerification,
                         RequireTV Op             TransferVerification
```

## Three Verification Backends

### Translation Validation (TV)

Checks that a transformed Payload IR refines the original — for all inputs
where the original has defined behavior, the transformed version produces
the same outputs.

**How it works:**
1. Walk both `func.func` bodies and lower arith ops to Z3 bitvectors
2. Build refinement formula: `∃ inputs. output_before ≠ output_after`
3. If Z3 says UNSAT → transform is correct
4. If Z3 says SAT → counterexample found → repair loop

**Supported ops:** arith.addi, subi, muli, divui, divsi, remui, remsi, cmpi, select, constant

**Entry point:** `compgen.ir.semantic.translation_validation.validate_translation()`

### PDL Rewrite Verification

Verifies that DAG-to-DAG rewrite rules are sound across all bitwidths.

**How it works:**
1. Express pattern and replacement as Z3 expression builders
2. For each bitwidth w ∈ [1, max_bitwidth]:
   - Assert `pattern(inputs) ≠ replacement(inputs)`
   - If UNSAT → sound at this width
3. If sound at all widths → rewrite family is promotable

**Entry point:** `compgen.semantic.rewrite.verify_pdl.verify_rewrite_family()`

### Transfer Function Verification

Verifies that dataflow analyses (known-bits, tile divisibility, etc.) are
sound over-approximations of the concrete semantics.

**How it works:**
1. Define concrete op semantics and abstract transfer function
2. For all concrete inputs consistent with abstract domain:
   - Check concrete output is consistent with predicted abstract output
3. If always consistent → transfer function is sound
4. Sound facts materialize as Recipe IR fact ops

**Entry point:** `compgen.ir.semantic.dataflow_verify.verify_analysis()`

## Agent Integration

The agent interacts with verification at four points:

1. **Verification strategy** — LLM decides which regions get TV vs diff-test
2. **Counterexample repair** — when TV fails, LLM sees the counterexample and proposes a fix
3. **Semantics discovery** — LLM generates semantics for unknown ops
4. **Transfer analysis** — LLM designs verified dataflow analyses

See [Agent Verification Loop](../agent/verification_loop.md) for details.

## Recipe IR Verification Ops

| Op | Purpose |
|----|---------|
| `RequireTranslationValidationOp` | Request SMT-backed TV for a region |
| `RequireDiffTestOp` | Request differential testing |
| `RequireLayoutInvariantOp` | Check layout preservation |
| `RequireMemoryBoundOp` | Check memory usage constraint |
| `RequireProfileBudgetOp` | Check latency constraint (runtime) |
| `RequireCheckFileOp` | FileCheck-style assertions |

## Verified Fact Ops

| Op | Purpose |
|----|---------|
| `TileDivisibleOp` | Dimensions divisible by tile sizes (verified=1) |
| `ContiguousLayoutOp` | Contiguous memory layout (verified=1) |
| `BackendEligibleOp` | Region eligible for specific backend (verified=1) |
