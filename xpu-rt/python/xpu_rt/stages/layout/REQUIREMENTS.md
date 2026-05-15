# Layout Stage — Requirements for LLM Plugin Generation

## Purpose

The Layout stage resolves virtual layout encodings into concrete data
movement operations (pack/unpack/transpose). It sits between the Encoding
stage and the Dispatch stage in the compilation pipeline.

## Input Contract

- All tensor-producing ops have `compgen.encoding` attribute (from Encoding stage).

## Output Contract

- No `layout.set_layout` or `layout.unset_layout` ops remain.
- All `layout.pack` ops have concrete `PackSpecAttr`.
- Module is marked with `compgen.layout_clean = "1"`.

## What the Target Plugin Should Do

The plugin runs two target-specific passes:

1. **fuse_layout_into_producers** (Pass 6): When a producer op can directly
   emit the encoded layout, eliminate the SetLayout/UnsetLayout boundary.

2. **specialize_layouts** (Pass 8): Convert generic encodings to target-specific
   PackSpecAttr values using a LayoutResolver.

## Available Context

- `TargetProfile`: Hardware devices, memory hierarchy, features.
- `CapabilitySpec`: Op-to-backend mapping, supported dtypes.
- Layout encoding strings from the Encoding stage (e.g., `tiled_128x64`).

## Code Template

```python
class MyTargetLayoutPlugin:
    @property
    def target_name(self) -> str:
        return "my_target"

    @property
    def stage_name(self) -> str:
        return "layout"

    def configure(self, target, capabilities):
        self._target = target
        self._caps = capabilities

    def transform(self, module):
        from compgen.transforms.layout.fuse_layout_into_producers import fuse_layout_into_producers
        from compgen.transforms.layout.specialize_layouts import specialize_layouts
        from compgen.transforms.layout.cuda_resolver import CudaLayoutResolver

        module = fuse_layout_into_producers(module)
        resolver = CudaLayoutResolver()
        module = specialize_layouts(module, resolver=resolver, capabilities=self._caps)
        return module

    def get_artifacts(self):
        return {"layout_strategy": "my_target_layout"}
```
