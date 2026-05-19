# Stage: Bundle — Target-Specific Requirements

## What This Stage Does

The Bundle stage packages all compilation artifacts into a deployable
bundle matching the Artifact Contract.  This is the final stage in
every target's dialect stack.

## Input Contract

- Module is valid xDSL IR
- All prior stages have completed (encoding, dispatch, codegen, etc.)

## Output Contract

- A bundle directory exists with `manifest.json`
- `payload.mlir` is serialized
- All artifacts referenced in manifest exist

## What You Must Generate

Generate a `TargetStagePlugin` for target: `{target_name}`.

```python
class {TargetName}BundlePlugin:
    @property
    def target_name(self) -> str:
        return "{target_name}"

    @property
    def stage_name(self) -> str:
        return "bundle"

    def configure(self, target, capabilities):
        # Extract: target-specific bundle format requirements,
        # HAL driver config, firmware paths, runtime dependencies
        ...

    def transform(self, module):
        # Module passes through unchanged.
        # Target-specific artifacts are added via get_artifacts().
        return module

    def get_artifacts(self):
        # Return target-specific additions:
        #   - HAL driver configuration YAML
        #   - Firmware binary paths
        #   - Runtime library requirements
        #   - Device initialization scripts
        return {{"hal_config": ..., "runtime_deps": ...}}
```

## Tests Your Code Must Pass

1. Module must pass through unchanged
2. Artifacts dict must be serializable to JSON
3. No side effects on the IR
