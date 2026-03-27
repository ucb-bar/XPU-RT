# Verification Model

CompGen is designed around the rule "trust nothing from the generator until it is verified."

## Verification Layers

The project documents four verification layers:

1. Structural
2. Functional
3. Performance
4. Formal

## What Is Available Today

- Structural-style validation exists in several subsystems and tests.
- The demo runs a transform verification step and emits bundle artifacts.
- Target generation emits a `verification_manifest.json` describing the expected verification surface for a target.

## What Is Still Ahead

The full CLI-driven verification ladder is documented but not yet implemented end to end. Public docs treat it as design direction, not current user workflow.

## Why Users Should Care

The point of CompGen is not just to generate compiler artifacts. It is to generate them in a way that can be checked, reproduced, and promoted only after passing validation.
