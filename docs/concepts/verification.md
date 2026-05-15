# Verification model

CompGen is designed around one rule: trust nothing from the generator
until it is verified.

## Four-level verification ladder

1. **Structural** — schema validation, IR verifier, parser round-trip,
   CHECK assertions.
2. **Functional** — eager vs compiled outputs, randomised tensor tests,
   dynamic shapes.
3. **Performance** — compile time, warm run time, graph coverage, bytes
   moved.
4. **Formal** — translation validation, rewrite verification
   (solver-backed, optional).

Only bundles that pass every applicable level can be promoted into the
deterministic recipe library.

## What is available today

- Structural-style validation is wired into several subsystems and tests.
- The demo runs a transform verification step and emits bundle artifacts.
- Target generation emits a `verification_manifest.json` describing the
  expected verification surface for a target.
- `compgen verify` runs best-effort structural / functional / performance
  / formal checks against a bundle; see
  [CLI Reference](../reference/cli.md).

## Further reading

- [Concepts → Agent Verification Loop](agent_verification_loop.md) — how
  the LLM loop is bounded by verification gates.
- [Concepts → Semantic IR Verification](semantic_ir_verification.md) —
  the Semantic IR trust layer.
- [Architecture → Compiler Generation](../architecture/compiler-generation.md)
  — where the ladder sits in the overall pipeline.

## Why users should care

The point of CompGen is not just to generate compiler artifacts; it is to
generate them in a way that can be checked, reproduced, and promoted only
after passing validation.
