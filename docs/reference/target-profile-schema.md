# Target Profile Schema

Target profiles describe deployment targets for profile-centric flows in CompGen.

Bundled examples live under `examples/target_profiles/`.

## Top-Level Fields

| Field | Type | Required | Purpose |
|------|------|----------|---------|
| `name` | string | Yes | Profile identifier |
| `schema_version` | string | Yes | Schema version |
| `devices` | list | Yes | Device specifications |
| `interconnects` | list | No | Device-to-device links |
| `constraints` | dict | No | System constraints |
| `cost_model` | dict | No | Performance hints |
| `calibration_data` | dict | No | Measured calibration data |
| `metadata` | dict | No | Extra annotations |

## Device Fields

Each device entry can describe:

- device type and name
- vendor
- compute units
- memory hierarchy
- supported operations
- hardware features

## Example Profiles

- `examples/target_profiles/cuda_a100.yaml`
- `examples/target_profiles/multi_device.yaml`
- `examples/target_profiles/riscv_soc.yaml`
- `examples/target_profiles/trainium1.yaml`

## Target Profiles vs Hardware Specs

This is the current distinction to keep in mind:

- Target profiles are the simpler schema used by lower-level modules and the demo.
- Hardware specs are the richer targetgen input used by `compgen.device()` and target generation.

If you are following the top-level Python API, use the hardware-spec example documented in [Python API](python-api.md).
