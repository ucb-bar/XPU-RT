# Benchmark Suites

This guide covers the implemented benchmark harness in `benchmarks/cli.py`.

It is separate from the top-level `compgen` CLI. Use it when you want to run:

- recognized benchmark suites: `torchbench`, `huggingface`, `timm`, `mlperf`, `sol_execbench`, `heterobench`
- pack-backed integrations: `pack_integrations`
- normalized cross-suite result exports

## Current Model

The harness keeps one canonical verbose record format, `RunRecord`, then derives a flat cross-suite JSON export from those records.

The suite runner supports two integration styles:

- PyTorch-backed suites.
  These run local eager / compiled references plus the CompGen capture-analysis path.
- External-driver suites.
  These wrap configured upstream commands and parse emitted metrics files.

MLPerf, SOL-ExecBench, HeteroBench, and pack-backed integrations depend on configured command templates. The harness does not auto-clone repos or auto-download datasets.

## Workspace Configuration

Use a workspace YAML when your suite roots, datasets, or runner commands live outside the repo.

```yaml
repo_root: /path/to/CompGen

external_roots:
  torchbench: /path/to/benchmark
  mlperf_inference: /path/to/inference
  sol_execbench: /path/to/SOL-ExecBench
  heterobench: /path/to/HeteroBench

pack_roots:
  cuda_tile: /path/to/cuda-tile-pack

integration_worktrees_root: /path/to/worktrees

suite_configs:
  torchbench:
    official_command:
      - /usr/bin/python3
      - /path/to/pytorch/benchmarks/dynamo/torchbench.py
      - --performance
      - --output
      - "{metrics_path}"
  mlperf:
    reference_command:
      - /usr/bin/python3
      - /path/to/mlperf_runner.py
      - reference
      - "{metrics_path}"
    compgen_command:
      - /usr/bin/python3
      - /path/to/mlperf_runner.py
      - compgen
      - "{metrics_path}"

pack_configs:
  cuda_tile:
    reference_command:
      - /usr/bin/python3
      - /path/to/pack_runner.py
      - reference
      - "{metrics_path}"
    compgen_command:
      - /usr/bin/python3
      - /path/to/pack_runner.py
      - compgen
      - "{metrics_path}"
```

The command templates may use these placeholders:

- `{repo_root}`
- `{suite_root}`
- `{output_dir}`
- `{metrics_path}`
- `{workload_id}`
- `{upstream_workload_id}`
- `{mode}`
- `{device}`
- `{dtype}`
- `{batch_size}`
- `{pack_name}`
- `{integration_branch}`

## Common Commands

Probe the current environment:

```bash
env PYTHONPATH=python python -m benchmarks.cli list-suites
env PYTHONPATH=python python -m benchmarks.cli probe-suite mlperf
```

Inspect workloads:

```bash
env PYTHONPATH=python python -m benchmarks.cli list-suite-workloads mlperf --blessed-only
```

Run one workload:

```bash
env PYTHONPATH=python python -m benchmarks.cli \
  --workspace-config workspace.yaml \
  run-suite-workload mlperf llama3.1-8b \
  --output-dir benchmarks/results/suites \
  --iterations 10 \
  --warmup 3
```

Run the blessed subset of a suite:

```bash
env PYTHONPATH=python python -m benchmarks.cli \
  --workspace-config workspace.yaml \
  run-suite mlperf \
  --output-dir benchmarks/results/suites
```

Include non-blessed workloads:

```bash
env PYTHONPATH=python python -m benchmarks.cli \
  --workspace-config workspace.yaml \
  run-suite mlperf \
  --all-workloads \
  --output-dir benchmarks/results/suites
```

Export normalized cross-suite JSON files:

```bash
env PYTHONPATH=python python -m benchmarks.cli \
  export-suite-results benchmarks/results/suites \
  --output-dir benchmarks/results/normalized
```

## Output Layout

Each suite run writes:

- canonical `RunRecord` JSON files under the selected results directory
- per-run normalized JSON files under a `normalized/` subdirectory
- any parsed metrics or suite artifacts referenced from `record.artifacts.artifact_paths`

The normalized export is the right format for cross-suite dashboards and paper tables. The `RunRecord` files remain the source of truth for detailed debugging and artifact inspection.
