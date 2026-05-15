"""End-to-end tests for recognized benchmark suite integration."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from benchmarks.cli import main
from benchmarks.compare import load_all_results
from benchmarks.record import RunRecord
from benchmarks.suite_runner import export_suite_results, list_suites


def _fake_external_runner(path: Path) -> None:
    path.write_text(
        """
import json
import sys
from pathlib import Path

mode = sys.argv[1]
metrics_path = Path(sys.argv[2])
payload = {
    "status": "pass",
    "compile_time_ms": 275.0 if mode == "compgen" else 10.0,
    "latency_ms_p50": 4.0 if mode == "reference" else 5.5,
    "latency_ms_p90": 4.4 if mode == "reference" else 6.0,
    "throughput": 200.0 if mode == "reference" else 150.0,
    "peak_memory_mb": 64.0,
    "verification_ok": True,
    "correctness_ok": True,
    "device_assignment": {"host": "cpu", "accel0": "gpu0"},
    "transfer_time_ms": 0.5,
    "config_time_ms": 1.5,
    "overlap_ratio": 0.25,
    "utilization": 0.8,
    "official_metrics": [{"name": "offline_qps", "value": 2000, "unit": "qps"}],
}
metrics_path.write_text(json.dumps(payload))
""".strip()
    )


def _write_fake_torchbench_root(root: Path) -> Path:
    model_code = """
from __future__ import annotations

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x))


class Model:
    def __init__(self, test: str = "eval", device: str = "cpu", batch_size: int = 1) -> None:
        self.test = test
        self.device = device
        self.batch_size = batch_size

    def get_module(self):
        return TinyModel(), (torch.randn(self.batch_size, 16),)
""".strip()
    (root / "torchbenchmark" / "models" / "resnet50").mkdir(parents=True, exist_ok=True)
    (root / "torchbenchmark" / "models" / "hf_Bert").mkdir(parents=True, exist_ok=True)
    (root / "torchbenchmark" / "models" / "timm_vision_transformer").mkdir(parents=True, exist_ok=True)
    (root / "torchbenchmark" / "__init__.py").write_text("")
    (root / "torchbenchmark" / "models" / "__init__.py").write_text("")
    for name in ("resnet50", "hf_Bert", "timm_vision_transformer"):
        (root / "torchbenchmark" / "models" / name / "__init__.py").write_text(model_code)
    return root


def _install_fake_transformers(monkeypatch) -> None:
    module = types.ModuleType("transformers")

    class BertConfig:
        def __init__(
            self,
            hidden_size: int = 128,
            intermediate_size: int = 512,
            num_attention_heads: int = 4,
            num_hidden_layers: int = 2,
            vocab_size: int = 30522,
        ) -> None:
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_attention_heads = num_attention_heads
            self.num_hidden_layers = num_hidden_layers
            self.vocab_size = vocab_size

    class GPT2Config:
        def __init__(
            self, n_embd: int = 128, n_layer: int = 2, n_head: int = 4, vocab_size: int = 50257, n_positions: int = 32
        ) -> None:
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.vocab_size = vocab_size
            self.n_positions = n_positions

    class T5Config:
        def __init__(
            self,
            d_model: int = 128,
            d_ff: int = 256,
            num_layers: int = 2,
            num_decoder_layers: int = 2,
            num_heads: int = 4,
            vocab_size: int = 32128,
        ) -> None:
            self.d_model = d_model
            self.d_ff = d_ff
            self.num_layers = num_layers
            self.num_decoder_layers = num_decoder_layers
            self.num_heads = num_heads
            self.vocab_size = vocab_size

    class _TokenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(1, 1)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            decoder_input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            x = input_ids.float().unsqueeze(-1)
            if attention_mask is not None:
                x = x * attention_mask.float().unsqueeze(-1)
            if decoder_input_ids is not None:
                x = x + decoder_input_ids.float().unsqueeze(-1)
            return self.proj(x).squeeze(-1)

    module.BertConfig = BertConfig
    module.GPT2Config = GPT2Config
    module.T5Config = T5Config
    module.BertModel = lambda cfg: _TokenModel()
    module.GPT2LMHeadModel = lambda cfg: _TokenModel()
    module.T5ForConditionalGeneration = lambda cfg: _TokenModel()

    monkeypatch.setitem(sys.modules, "transformers", module)


def _install_fake_timm(monkeypatch) -> None:
    module = types.ModuleType("timm")
    known_models = [
        "convnext_tiny",
        "vit_tiny_patch16_224",
        "swin_tiny_patch4_window7_224",
        "mobilenetv3_small_100",
        "resnet50",
    ]

    class _VisionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    def list_models(pretrained: bool = False):
        del pretrained
        return known_models

    def create_model(name: str, pretrained: bool = False):
        del name, pretrained
        return _VisionModel()

    module.list_models = list_models
    module.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", module)


def _patch_fake_pack_integration(monkeypatch, pack_root: Path, worktrees_root: Path) -> None:
    class FakeProbe:
        available = True
        source_root = pack_root
        missing_paths: list[str] = []

    class FakeBranch:
        branch_name = "compgen-cuda-tile"
        worktree_path = worktrees_root / "cuda_tile"

    class FakeManifest:
        name = "cuda_tile"
        kinds = ("gpu", "tile")
        reference_runner = "fake-runner"
        benchmark_targets = ("smoke",)
        integration_mode = "worktree"

    class FakePack:
        def probe(self, workspace) -> FakeProbe:
            del workspace
            return FakeProbe()

        def branch_plan(self, workspace, run_id: str) -> FakeBranch:
            del workspace, run_id
            return FakeBranch()

    loaded = SimpleNamespace(manifest=FakeManifest(), pack=FakePack())
    monkeypatch.setattr("benchmarks.suite_adapters.load_builtin_packs", lambda: [loaded])
    monkeypatch.setattr("benchmarks.suite_adapters.load_pack", lambda path: loaded)


def _write_workspace_yaml(path: Path, data: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _suite_workspace(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "CompGen"
    repo_root.mkdir()
    results_dir = tmp_path / "results"
    runner = tmp_path / "fake_runner.py"
    _fake_external_runner(runner)

    _install_fake_transformers(monkeypatch)
    _install_fake_timm(monkeypatch)

    torchbench_root = _write_fake_torchbench_root(tmp_path / "benchmark")
    mlperf_root = tmp_path / "mlperf_inference"
    sol_root = tmp_path / "sol-execbench"
    hetero_root = tmp_path / "HeteroBench"
    for root in (mlperf_root, sol_root, hetero_root):
        root.mkdir()
        (root / "runner.py").write_text(runner.read_text())
    (sol_root / "data" / "SOL-ExecBench" / "benchmark" / "gemm").mkdir(parents=True, exist_ok=True)
    (hetero_root / "config_json").mkdir(parents=True, exist_ok=True)
    (hetero_root / "config_json" / "image_pipeline.json").write_text("{}")

    pack_root = tmp_path / "cuda_tile"
    pack_root.mkdir()
    (pack_root / "runner.py").write_text(runner.read_text())
    worktrees_root = tmp_path / "worktrees"
    _patch_fake_pack_integration(monkeypatch, pack_root, worktrees_root)

    workspace_yaml = tmp_path / "workspace.yaml"
    _write_workspace_yaml(
        workspace_yaml,
        {
            "repo_root": str(repo_root),
            "external_roots": {
                "torchbench": str(torchbench_root),
                "mlperf_inference": str(mlperf_root),
                "sol_execbench": str(sol_root),
                "heterobench": str(hetero_root),
            },
            "pack_roots": {"cuda_tile": str(pack_root)},
            "integration_worktrees_root": str(worktrees_root),
            "suite_configs": {
                suite_id: {"official_command": [sys.executable, str(runner), "reference", "{metrics_path}"]}
                for suite_id in ("torchbench", "huggingface", "timm")
            }
            | {
                suite_id: {
                    "reference_command": [sys.executable, str(runner), "reference", "{metrics_path}"],
                    "compgen_command": [sys.executable, str(runner), "compgen", "{metrics_path}"],
                }
                for suite_id in ("mlperf", "sol_execbench", "heterobench")
            },
            "pack_configs": {
                "cuda_tile": {
                    "reference_command": [sys.executable, str(runner), "reference", "{metrics_path}"],
                    "compgen_command": [sys.executable, str(runner), "compgen", "{metrics_path}"],
                }
            },
        },
    )
    return SimpleNamespace(workspace_yaml=workspace_yaml, results_dir=results_dir)


def test_list_suites_reports_all_registered_suites(tmp_path: Path, monkeypatch) -> None:
    env = _suite_workspace(tmp_path, monkeypatch)
    exit_code = main(["--workspace-config", str(env.workspace_yaml), "list-suites"])
    assert exit_code == 0
    statuses = list_suites()
    assert "torchbench" in statuses
    assert "huggingface" in statuses
    assert "timm" in statuses
    assert "pack_integrations" in statuses
    assert "mlperf" in statuses
    assert "sol_execbench" in statuses
    assert "heterobench" in statuses


def test_run_suite_workload_e2e_for_all_registered_suites(tmp_path: Path, monkeypatch) -> None:
    env = _suite_workspace(tmp_path, monkeypatch)
    cases = [
        ("torchbench", "resnet50"),
        ("huggingface", "bert-base-uncased"),
        ("timm", "resnet50"),
        ("mlperf", "llama3.1-8b"),
        ("sol_execbench", "gemm"),
        ("heterobench", "image_pipeline"),
        ("pack_integrations", "cuda_tile"),
    ]
    for suite_id, workload_id in cases:
        exit_code = main(
            [
                "--workspace-config",
                str(env.workspace_yaml),
                "run-suite-workload",
                suite_id,
                workload_id,
                "--output-dir",
                str(env.results_dir),
                "--iterations",
                "1",
                "--warmup",
                "0",
            ]
        )
        assert exit_code == 0

    records = load_all_results(env.results_dir)
    observed = {(record.suite.suite_id, record.workload_id) for record in records}
    for suite_id, workload_id in cases:
        assert (suite_id, workload_id) in observed
    assert all("normalized_result" in record.artifacts.artifact_paths for record in records if record.suite.suite_id)


def test_run_suite_e2e_respects_blessed_subset_and_cli_export(tmp_path: Path, monkeypatch) -> None:
    env = _suite_workspace(tmp_path, monkeypatch)
    exit_code = main(
        [
            "--workspace-config",
            str(env.workspace_yaml),
            "run-suite",
            "mlperf",
            "--output-dir",
            str(env.results_dir),
            "--iterations",
            "1",
            "--warmup",
            "0",
        ]
    )
    assert exit_code == 0

    records = [record for record in load_all_results(env.results_dir) if record.suite.suite_id == "mlperf"]
    workloads = {record.workload_id for record in records}
    assert "llama3.1-8b" in workloads
    assert "whisper" in workloads
    assert "dlrm-v3" in workloads
    assert "resnet50-v1.5" in workloads
    assert "rgat" not in workloads

    normalized_dir = tmp_path / "normalized"
    export_exit_code = main(
        [
            "export-suite-results",
            str(env.results_dir),
            "--output-dir",
            str(normalized_dir),
        ]
    )
    assert export_exit_code == 0
    exported = sorted(normalized_dir.glob("*.normalized.json"))
    assert exported


def test_export_suite_results_writes_normalized_json(tmp_path: Path) -> None:
    record = RunRecord(model_name="mlperf_resnet50", system_name="mlperf_official")
    record.status = "pass"
    record.suite.suite_id = "mlperf"
    record.suite.upstream_workload_id = "resnet50-v1.5"
    record.suite.mode = "inference"
    record.suite.device = "cpu"
    record.suite.dtype = "float32"
    record.performance.latency_median_us = 5000.0
    record.performance.latency_p90_us = 5500.0
    record.performance.throughput_samples_per_sec = 250.0

    paths = export_suite_results([record], tmp_path / "normalized")
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text())
    assert payload["suite"] == "mlperf"
    assert payload["workload"] == "resnet50-v1.5"
    assert payload["latency_ms_p50"] == 5.0
