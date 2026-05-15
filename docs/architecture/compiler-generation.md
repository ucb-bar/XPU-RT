# Compiler Generation: xDSL to C++ MLIR

This document describes how XPU-RT generates a standalone MLIR C++ compiler from its xDSL Python prototypes.

## Overview

XPU-RT's development workflow is:

1. **Prototype in xDSL** (Python) — define dialects, passes, transformations
2. **Iterate and validate** — run transforms, verify with Z3, benchmark
3. **Generate C++ compiler** — emit TableGen, C++, CMake that links against `third_party/llvm-project/`
4. **Build and deploy** — `cmake + ninja` produces `xpu-rt-opt` binary

The generated compiler is an **artifact**, not checked into the repo. It's the "compiler" that XPU-RT (the compiler generator) produces.

## Architecture

```
XPU-RT Python (xDSL dialects + passes)
    ↓  mlir_cppgen introspects live xDSL objects
┌──────────────────────────────────────────┐
│  python/xpu_rt/extensions/mlir_cppgen/  │
│  ├── introspect.py   (read xDSL Dialect) │
│  ├── tablegen_emitter.py  (.td files)    │
│  ├── cpp_emitter.py  (.h/.cpp files)     │
│  ├── pass_emitter.py (pass .td + .cpp)   │
│  ├── cmake_emitter.py (CMakeLists.txt)   │
│  ├── driver_emitter.py (xpu-rt-opt.cpp) │
│  └── templates/*.j2  (Jinja2 templates)  │
└──────────────┬───────────────────────────┘
               ↓  generates
┌──────────────────────────────────────────┐
│  artifacts/compiler/  (generated C++)    │
│  ├── CMakeLists.txt                      │
│  ├── include/XPU-RT/{Layout,Tile,Accel} │
│  ├── lib/{Layout,Tile,Accel}             │
│  ├── xpu-rt-opt/xpu-rt-opt.cpp         │
│  └── Dockerfile                          │
└──────────────┬───────────────────────────┘
               ↓  cmake + ninja
┌──────────────────────────────────────────┐
│  xpu-rt-opt binary                      │
│  Parses MLIR, runs passes, emits MLIR    │
└──────────────────────────────────────────┘
```

## Usage

### Generate the compiler

```bash
# Generate all dialects
python -m xpu_rt.extensions.mlir_cppgen \
    --dialects layout,tile,accel \
    --output artifacts/compiler/

# Generate with Dockerfile
python -m xpu_rt.extensions.mlir_cppgen \
    --dialects layout,tile,accel \
    --output artifacts/compiler/ \
    --docker
```

### Python API

```python
from xpu_rt.extensions.mlir_cppgen import generate_compiler

generate_compiler(
    dialects=["layout", "tile", "accel"],
    output_dir="artifacts/compiler",
    include_docker=True,
)
```

### Build the generated compiler

```bash
# Prerequisites: LLVM/MLIR built from third_party/llvm-project
cmake -G Ninja -S artifacts/compiler -B build \
    -DMLIR_DIR=third_party/llvm-project/build/lib/cmake/mlir
ninja -C build
```

### Run xpu-rt-opt

```bash
build/bin/xpu-rt-opt input.mlir --layout-propagate-layouts -o output.mlir
```

## What Gets Generated

### Per Dialect

For each xDSL dialect, the generator produces:

| File | Purpose |
|------|---------|
| `{Prefix}Dialect.td` | TableGen dialect definition |
| `{Prefix}Attrs.td` | Custom attribute definitions |
| `{Prefix}Ops.td` | Operation definitions with properties, traits, verifiers |
| `{Prefix}Dialect.h/cpp` | C++ dialect initialization |
| `{Prefix}Attrs.h/cpp` | C++ attribute implementations |
| `{Prefix}Ops.h/cpp` | C++ op implementations + verifier code |
| `CMakeLists.txt` | Build targets |

### Dialect Coverage

| Dialect | Ops | Attrs | Generate C++? |
|---------|-----|-------|--------------|
| Layout | 4 | 2 | Yes |
| Tile | 7 | 3 | Yes |
| Accel | 6 | 0 | Yes |
| RecipeBase | 0 | 2 | Yes (shared attrs) |
| Agent (40+) | — | — | No (Python-only) |
| Semantic (3) | — | — | No (Python-only) |

## How Introspection Works

The generator does **not** parse Python source files. It introspects live xDSL objects at runtime:

1. `Dialect.get_irdl_definition()` → `OpDef` with `properties`, `regions`, `traits`
2. `ParametrizedAttribute.get_irdl_definition()` → `ParamAttrDef` with `parameters`
3. `PropertyDef.constr.attr.__name__` → xDSL type name (e.g., `"SymbolRefAttr"`)
4. Type mapping table converts xDSL types → TableGen types

### Type Mapping

| xDSL Type | TableGen Type |
|-----------|--------------|
| `StringAttr` | `StrAttr` |
| `IntegerAttr` | `I64Attr` |
| `SymbolRefAttr` | `FlatSymbolRefAttr` |
| `ArrayAttr` | `ArrayAttr` |
| `LayoutEncodingAttr` | `Layout_LayoutEncodingAttr` |
| `ProvenanceAttr` | `RecipeBase_ProvenanceAttr` |

### Verifier Detection

The generator auto-detects three verifier patterns:

1. **Enum check**: Op has `_VALID_*` class variable → generates `llvm::StringSet<>` check
2. **Dimension check**: `verify_()` checks `len(dims)` → generates dimension validation
3. **Range check**: `verify_()` checks value membership → generates range validation

## Pass Generation

The generator translates Python layout transforms to C++ MLIR passes. All 10 layout passes are generated:

| # | Pass | C++ Class | Pattern |
|---|------|-----------|---------|
| 1 | canonicalize-transposes | CanonicalizeTransposesPass | attr_annotation |
| 2 | attach-layout-hints | AttachLayoutHintsPass | attr_annotation |
| 3 | set-virtual-encodings | SetVirtualEncodingsPass | structural |
| 4 | propagate-layouts | PropagateLayoutsPass | attr_annotation |
| 5 | hoist-layout-ops | HoistLayoutOpsPass | attr_annotation |
| 6 | fuse-layout-into-producers | FuseLayoutIntoProducersPass | attr_annotation |
| 7 | introduce-prepacking | IntroducePrepackingPass | structural |
| 8 | specialize-layouts | SpecializeLayoutsPass | attr_annotation |
| 9 | materialize-boundaries | MaterializeBoundariesPass | structural |
| 10 | cleanup-artifacts | CleanupArtifactsPass | structural |

### Translation rules (Python → C++)

| Python | C++ MLIR |
|--------|----------|
| `module.walk()` | `getOperation()->walk(...)` |
| `op.attributes.get("key")` | `op->getAttrOfType<StringAttr>("key")` |
| `op.attributes["key"] = StringAttr(val)` | `op->setAttr("key", StringAttr::get(ctx, val))` |
| `isinstance(op, FuncOp)` | `isa<func::FuncOp>(op)` |
| `op.name.startswith("arith.")` | `op->getName().getStringRef().starts_with("arith.")` |
| `id(result)` mapping | `DenseMap<Value, StringRef>` |
| `parent.insert_op_before(...)` | `OpBuilder(op).create<NewOp>(...)` |
| `parent.erase_op(op)` | `op->erase()` |

## Integration with Existing Pipeline

The generated `xpu-rt-opt` connects to the existing Python pipeline via MLIR text serialization:

```
Python (xDSL)  →  recipe_to_mlir()  →  MLIR text
                                           ↓
                                      xpu-rt-opt (C++ passes)
                                           ↓
                                       MLIR text  →  mlir_to_recipe()  →  Python (xDSL)
```

This roundtrip is already implemented in `ir/recipe/serialize.py`.

### Runner Integration

`python/xpu_rt/extensions/mlir_cppgen/runner.py` provides:

```python
from xpu_rt.extensions.mlir_cppgen.runner import run_xpu_rt_opt, run_layout_pipeline

# Single pass
output = run_xpu_rt_opt(mlir_text, ["--layout-propagate-layouts"])

# Full 10-pass pipeline
output = run_layout_pipeline(mlir_text)
```

## Testing

The generator produces MLIR FileCheck tests:
- `test/{Dialect}/roundtrip.mlir` — parse+print roundtrip for each dialect
- `test/Passes/{name}.mlir` — per-pass functional tests
- `test/lit.cfg.py` — LLVM lit test runner configuration
