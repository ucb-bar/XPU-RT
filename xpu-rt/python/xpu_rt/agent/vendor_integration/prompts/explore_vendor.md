# Vendor-dialect exploration

You are classifying a third-party MLIR dialect repository so CompGen can generate
an adapter. The scanner output below is ground truth; do not invent facts.

## Scanner summary

```
{scanner_summary}
```

## README excerpt

```
{readme_excerpt}
```

## Detected TableGen ops

```
{td_ops}
```

## CLI tools

```
{cli_tools}
```

## Task

Return a JSON object with these keys. Use empty arrays / strings where
the scan does not give enough evidence; do NOT guess. No prose, just JSON.

```
{{
  "input_ir": ["linalg" | "tosa" | "stablehlo" | "triton" | "<vendor-native>"],
  "output_format": "cubin" | "hexagon_elf" | "llvm_ir" | "bytecode" | ...,
  "kernel_authoring_required": true | false,
  "lowering_mode": "direct_linalg" | "torch_mlir" | "stablehlo" | "kernel_authoring",
  "op_families": ["matmul", "softmax", ...],
  "bundle_steps": ["<cli-tool> <args>", ...],
  "runtime_entry": "<launcher or library symbol, if visible>",
  "notes": "<one-paragraph rationale citing README/tools evidence>"
}}
```

Rules:

* Set ``kernel_authoring_required`` to ``true`` only when no linalg/tosa/
  stablehlo/torch-mlir ingress exists — i.e. the only way into the vendor
  dialect is to *write* its ops directly.
* ``bundle_steps`` must reference CLI tools from the scan list only.
* Cite the README lines or tools that justify the mode in ``notes``.
