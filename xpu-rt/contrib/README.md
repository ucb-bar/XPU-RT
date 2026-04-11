# CompGen Community Contributions

This directory holds community-contributed extensions to CompGen. Each subdirectory corresponds to an extension point in the main codebase.

## Extension Points

| Directory | Extension Point | Protocol | Docs |
|-----------|----------------|----------|------|
| `providers/` | Kernel generators | `KernelProvider` | `docs/architecture/extension-points.md` |
| `quantization/` | Quantization methods | `AOBaseConfig` | `docs/architecture/extension-points.md` |
| `targets/` | Hardware backends | `TargetBackendProtocol` | `docs/architecture/target-backend-model.md` |
| `dialects/` | MLIR dialects | `DialectSpec` | `docs/architecture/extension-points.md` |

## How to Contribute

1. Pick an extension point from the table above
2. Copy the corresponding `_template.py` from the main codebase
3. Implement the protocol
4. Add your implementation here with a README
5. Submit a PR

## Structure

```
contrib/
├── providers/         # Community kernel generators
│   └── my_provider/
│       ├── README.md
│       └── my_provider.py
├── quantization/      # Community quantization methods
│   └── int4_awq/
│       ├── README.md
│       └── awq_config.py
├── targets/           # Community target backends
│   └── fpga_hls/
│       ├── README.md
│       └── hls_backend.py
└── dialects/          # Community MLIR dialects
    └── my_accel/
        ├── README.md
        └── dialect_spec.py
```

Each contribution should include:
- A `README.md` explaining what it does and how to use it
- The implementation file(s)
- Test data or examples (if applicable)
