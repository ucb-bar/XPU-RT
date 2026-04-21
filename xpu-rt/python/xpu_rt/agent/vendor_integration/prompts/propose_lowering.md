# Lowering proposal

Given the frozen descriptor and the target workload list, propose a
concrete lowering plan the scaffolded adapter should implement.

## Descriptor (YAML)

```
{descriptor_yaml}
```

## Workloads

```
{workloads}
```

## Task

Return a JSON object:

```
{{
  "rules": [
    {{"op_family": "matmul",  "strategy": "template" | "llm" | "passthrough", "rationale": "..."}},
    ...
  ],
  "risks": ["..."],
  "verification_hooks": ["structural", "matmul_diff", "workload_diff"]
}}
```

Rules:

* ``strategy=template`` implies the user-space package ships a hand-crafted
  vendor-dialect kernel for the op family.
* ``strategy=llm`` delegates to ``ClaudeKernelProvider``.
* ``strategy=passthrough`` is only valid when ``lowering_mode=direct_linalg``.
* List real risks, not hypothetical ones.
