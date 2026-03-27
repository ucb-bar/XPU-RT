# Stage: Encoding — Target-Specific Requirements

## What This Stage Does

The Encoding stage resolves data layout and dtype decisions for every tensor
in the program.  It annotates ops with `compgen.encoding` attributes that
downstream stages use for tiling, dispatch, and codegen decisions.

## Input Contract

- Module is valid xDSL IR (passes verifier)
- Ops are in linalg/arith/func/tensor dialects
- No `compgen.encoding` attributes present yet

## Output Contract

- Every tensor-producing op has a `compgen.encoding` attribute
- Encoding values are one of: `row_major`, `column_major`, `tiled_MxN`, or a target-specific string

## What You Must Generate

Generate a `TargetStagePlugin` implementation for target: `{target_name}`.

```python
class {TargetName}EncodingPlugin:
    @property
    def target_name(self) -> str:
        return "{target_name}"

    @property
    def stage_name(self) -> str:
        return "encoding"

    def configure(self, target, capabilities):
        # Extract: device type, memory hierarchy, supported MMA intrinsics,
        # DMA alignment requirements, cache line sizes
        ...

    def transform(self, module):
        # For each tensor-producing op:
        #   1. Check op type (matmul → MMA layout, elementwise → row_major, etc.)
        #   2. Check tensor shape (small → no tiling, large → tile to hardware dims)
        #   3. Set compgen.encoding attribute with target-optimal layout
        ...

    def get_artifacts(self):
        return {{"encoding_decisions": ...}}
```

## Target Information Available

```yaml
{target_profile}
```

## Examples

**For GPU (Triton/CUDA):**
- Matmul LHS: `tiled_128x64` (matches A100 tensor core tile)
- Matmul RHS: `tiled_64x128` (transposed for coalesced access)
- Elementwise: `row_major` (no special layout needed)

**For NPU with DMA:**
- All tensors: aligned to DMA burst size (e.g., 256 bytes)
- Matmul: `tiled_16x16` (systolic array dimension)

**For CPU:**
- Matmul LHS: `tiled_8x1` (cache-line width / element size)
- If narrow-N matmul: transpose to narrow-M for ukernel efficiency

## Tests Your Code Must Pass

1. Every tensor-producing op must have `compgen.encoding` after your transform
2. Module must still pass xDSL verifier
3. Function signatures must not change
4. Running the stage twice must produce the same result (idempotent)
