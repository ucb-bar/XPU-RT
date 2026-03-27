# Stage: Dispatch — Target-Specific Requirements

## What This Stage Does

The Dispatch stage partitions the computation graph into dispatch groups
(fusion regions).  Each dispatch group becomes a single kernel launch or
hardware dispatch slot.  The key decision is **which ops to fuse together**.

## Input Contract

- All tensor-producing ops have `compgen.encoding` attributes
- Module passes xDSL verifier

## Output Contract

- Every non-structural op has a `compgen.dispatch_id` attribute
- Dispatch groups respect data dependencies (no cycles)
- Each dispatch group is a contiguous region in the data flow graph

## What You Must Generate

Generate a `TargetStagePlugin` for target: `{target_name}`.

```python
class {TargetName}DispatchPlugin:
    @property
    def target_name(self) -> str:
        return "{target_name}"

    @property
    def stage_name(self) -> str:
        return "dispatch"

    def configure(self, target, capabilities):
        # Extract: max dispatch size, fusion rules, memory limits per dispatch,
        # whether target supports multi-output kernels, pipeline depth
        ...

    def transform(self, module):
        # For each op:
        #   1. Decide if it should fuse with its producer or start a new dispatch
        #   2. Apply fusion heuristic based on target
        #   3. Set compgen.dispatch_id attribute
        # Key fusion rules:
        #   - Elementwise after matmul → same dispatch (free fusion)
        #   - Two matmuls with shared operand → separate dispatches
        #   - Reduction ops → dispatch boundary
        ...

    def get_artifacts(self):
        return {{"dispatch_groups": ..., "fusion_decisions": ...}}
```

## Examples

**For GPU (Triton):**
- Matmul + bias_add + activation → single dispatch (fused kernel)
- Adjacent independent matmuls → separate dispatches (concurrent execution)
- Reduction (layernorm, softmax) → dispatch boundary

**For NPU (fixed pipeline):**
- One dispatch per hardware execution slot
- Limited to ops the NPU pipeline can handle
- Fallback dispatch for unsupported ops (runs on CPU)

**For Heterogeneous (CPU+GPU):**
- Small ops → CPU dispatch
- Large compute-heavy ops → GPU dispatch
- Cross-device boundary = dispatch boundary

## Tests Your Code Must Pass

1. Every non-structural op must have `compgen.dispatch_id`
2. No cyclic dependencies between dispatch groups
3. Module must still pass xDSL verifier
4. Function signatures must not change
