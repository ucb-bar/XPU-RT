# Architecture Overview

CompGen is organized around a staged pipeline plus a target-generation subsystem.

## Current Runnable Path

The demo and top-level Python API exercise this shape today:

1. Capture a PyTorch model
2. Convert the graph into Payload IR
3. Analyze kernels and choose strategies
4. Run equality-saturation optimization
5. Plan execution
6. Bundle artifacts and benchmark locally

## Target Generation Path

When you create a `CompGenDevice`, CompGen:

1. Loads a hardware spec YAML
2. Validates it
3. Extracts a target profile
4. Classifies the hardware family
5. Generates a support plan
6. Builds a target-specific dialect stack
7. Emits target-generation artifacts

## Public Surfaces

| Surface | Purpose |
|--------|---------|
| CLI | Discover the intended command surface |
| Python API | Script the current working flows |
| Demo script | Run the most complete vertical slice |
| Target profiles | Describe deployment targets for profile-centric flows |
| Hardware specs | Drive target generation and `compgen.device()` |

## Internal Detail

The deeper architecture records, ADRs, scheduling design notes, and roadmap material were intentionally moved out of `docs/` into `tmp/agentic_documentation/` so the public docs can stay focused on user workflows.
