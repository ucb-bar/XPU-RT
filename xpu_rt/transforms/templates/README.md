# Transform Templates

This directory contains Jinja2 and/or MLIR Transform Dialect template files
that the LLM fills in during Stage 3 (transform generation).

## Template Format

Templates use Jinja2 syntax with MLIR Transform Dialect as the target:

```mlir
// Template: {{ template_name }}
// Target: {{ target_profile.name }}
// Objective: {{ objective }}

transform.sequence failures(propagate) {
^bb0(%module: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module
    : (!transform.any_op) -> !transform.any_op
  transform.structured.tile_using_forall %matmul
    tile_sizes [{{ tile_m }}, {{ tile_n }}, {{ tile_k }}]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}
```

## Variables

Templates receive these variables from the generation context:

- `target_profile` -- the full TargetProfile dataclass
- `objective` -- the optimization objective (latency/throughput/memory/energy)
- `kernel_contracts` -- list of KernelContract for ops in the IR
- `tile_*`, `vector_*`, `unroll_*` -- LLM-chosen parameters
- `available_ops` -- list of available transform ops

## Adding Templates

1. Create a `.mlir.j2` file in this directory
2. Register it in the template registry (future)
3. Add corresponding CHECK assertions for regression testing
