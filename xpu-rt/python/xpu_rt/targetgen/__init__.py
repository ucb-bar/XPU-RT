"""TargetGen — hardware spec driven compiler infrastructure generator.

Given a hardware specification YAML, TargetGen:
  1. Loads and validates the spec
  2. Classifies the target into a family
  3. Generates a support plan (which stages, dialects, patches)
  4. Emits TargetDialectStack + plugins + verification manifest
  5. All generated output goes to a configurable directory (gitignored)

The repo contains only the generator; no real HW designs are stored.
"""

from __future__ import annotations
