# What CompGen Is

CompGen is a compiler generator, not a monolithic compiler.

Given a model plus a hardware description, the system aims to generate:

- IR transforms
- kernel strategies and generated kernels
- placement and scheduling decisions
- runtime artifacts
- verification outputs

The LLM is intended to act as a proposal engine inside that process, not as an unchecked compiler author.

## What Matters for Users

- You are evaluating a workflow for new-target bring-up, not installing a finished production compiler.
- The repo already contains runnable pieces: capture, IR conversion, target generation, planning, bundling, and local benchmarking.
- The full CLI pipeline is still ahead of the implementation state, so the public docs prioritize runnable entrypoints first.

## What CompGen Is Not

- Not a replacement for PyTorch's frontend
- Not a wholesale rebuild of IREE
- Not a generic VM
- Not a system that ships unverified LLM output
