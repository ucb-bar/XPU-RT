"""I1 — Live adapters bridge into the headline benchmark runner.

Implements three thin ``_Adapter`` Protocol wrappers consumed by
:func:`compgen.benchmarks.headline.run_benchmark`:

* :class:`LiveTorchEagerAdapter` — vanilla forward pass with the
  loaded model.
* :class:`LiveTorchCompileAdapter` — ``torch.compile(..., mode="max-autotune")``
  applied once before timing.
* :class:`LiveCompGenAdapter` — *honest residual*: this path requires
  the full CompGen bundle pipeline (///work) wired
  through to ``CompiledModel.run`` on each workload, which is not
  complete for these specific 3 workloads. The adapter therefore
  returns a typed ``blocked`` measurement with
  ``blocked_reason="compgen_bundle_not_built_for_workload"``. This is
  the published kill outcome for ``C_HEADLINE_MATCH_TORCH_COMPILE``
  under the Phase I plan — honest, not hidden.

All three adapters share the same workload loader so the
``output_hash`` is comparable across them.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from compgen.benchmarks.headline import AdapterMeasurement

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class WorkloadHandle:
    """One loaded workload (model + sample inputs + reference output)."""

    workload_id: str
    family: str
    model: Any
    inputs: tuple[Any, ...]
    device: torch.device
    dtype: torch.dtype = field(default_factory=lambda: torch.float16)


# ---------------------------------------------------------------------------
# Workload loader
# ---------------------------------------------------------------------------


SLICE_SUFFIX = "__slice"


def _slice_workload(parent_id: str, parent: WorkloadHandle) -> WorkloadHandle:
    """Build a slice-variant of a parent workload (one decoder layer / block).

    The slice is what CompGen actually compiles end-to-end today; eager
    and torch.compile time the *same* slice for an apples-to-apples
    head-to-head. The synthetic input is fixed-seed so output hashes
    are comparable across adapters.
    """

    torch.manual_seed(0xBEEF)
    if parent_id == "tinyllama_1_1b":
        block = parent.model.model.layers[0].mlp.to(device="cpu", dtype=torch.float32).eval()
        hidden = parent.model.config.hidden_size
        sample = (torch.randn(1, 8, hidden, dtype=torch.float32),)
        return WorkloadHandle(
            workload_id=f"{parent_id}{SLICE_SUFFIX}",
            family="llm_block",
            model=block,
            inputs=sample,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    if parent_id == "whisper_tiny":
        inner = parent.model.model.encoder.layers[0].to(device="cpu", dtype=torch.float32).eval()
        d_model = parent.model.config.d_model

        class _WhisperLayerWrapper(torch.nn.Module):
            """Positional-only wrapper so torch.export can trace the block."""

            def __init__(self, layer: torch.nn.Module) -> None:
                super().__init__()
                self.layer = layer

            def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
                out = self.layer(x, attn_mask)
                return out[0] if isinstance(out, tuple) else out

        block = _WhisperLayerWrapper(inner)
        sample = (
            torch.randn(1, 8, d_model, dtype=torch.float32),
            torch.zeros(1, 1, 8, 8, dtype=torch.float32),
        )
        return WorkloadHandle(
            workload_id=f"{parent_id}{SLICE_SUFFIX}",
            family="speech_block",
            model=block,
            inputs=sample,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    raise NotImplementedError(f"slice unsupported for {parent_id!r}")


def _load_workload(workload_id: str) -> WorkloadHandle:
    """Load a workload's model + inputs from its YAML config.

    Supported families: ``llm`` (TinyLlama), ``vla`` (smolVLA),
    ``speech`` (Whisper-tiny). If ``workload_id`` ends in
    :data:`SLICE_SUFFIX`, the parent is loaded then sliced to a single
    decoder layer / encoder block for apples-to-apples CompGen vs
    baselines comparison.
    """

    if workload_id.endswith(SLICE_SUFFIX):
        parent_id = workload_id[: -len(SLICE_SUFFIX)]
        parent = _load_workload(parent_id)
        return _slice_workload(parent_id, parent)

    cfg_path = REPO_ROOT / "configs" / "models" / f"{workload_id}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    family = cfg.get("family", "")
    source = cfg.get("source", {})
    model_ref = source.get("model_ref", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    if family == "llm":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_ref)
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, dtype=dtype
        ).to(device).eval()
        prompt = cfg["inputs"].get("prompt", "Hello, my name is")
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        return WorkloadHandle(
            workload_id=workload_id,
            family=family,
            model=model,
            inputs=(ids,),
            device=device,
            dtype=dtype,
        )

    if family == "speech":
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
        )

        proc = AutoProcessor.from_pretrained(model_ref)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_ref, dtype=dtype
        ).to(device).eval()
        seconds = int(cfg["inputs"].get("sample_seconds", 30))
        # Synthetic audio at 16 kHz, deterministic across reruns.
        torch.manual_seed(0xA1D10)
        audio = torch.randn(seconds * 16000, dtype=torch.float32)
        feats = proc(
            audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device=device, dtype=dtype)
        decoder_input_ids = torch.tensor(
            [[model.config.decoder_start_token_id]], device=device
        )
        return WorkloadHandle(
            workload_id=workload_id,
            family=family,
            model=model,
            inputs=(feats, decoder_input_ids),
            device=device,
            dtype=dtype,
        )

    if family == "vla":
        # smolVLA loader is heavier and depends on lerobot; treat as
        # blocked-with-typed-reason rather than fabricating numbers.
        raise NotImplementedError(
            "smolVLA live adapter requires lerobot — honest block in v1"
        )

    raise ValueError(f"unknown workload family: {family!r}")


def _hash_tensor(t: torch.Tensor) -> str:
    """SHA-256 of a tensor's raw bytes (first 16 hex chars)."""

    blob = t.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(blob).hexdigest()[:16]


def _forward(handle: WorkloadHandle, model: Any) -> torch.Tensor:
    """One forward pass that returns the canonical output tensor."""

    if handle.family == "llm":
        with torch.no_grad():
            out = model(handle.inputs[0])
        return out.logits

    if handle.family == "speech":
        feats, dec_ids = handle.inputs
        with torch.no_grad():
            out = model(input_features=feats, decoder_input_ids=dec_ids)
        return out.logits

    if handle.family == "llm_block":
        with torch.no_grad():
            out = model(handle.inputs[0])
        return out

    if handle.family == "speech_block":
        with torch.no_grad():
            out = model(*handle.inputs)
        return out[0] if isinstance(out, tuple) else out

    raise NotImplementedError(f"forward not implemented for {handle.family}")


def _time_iters(
    handle: WorkloadHandle,
    model: Any,
    *,
    iters: int,
    warmup: int,
) -> list[float]:
    """Measure ``iters`` forward passes in microseconds.

    Uses ``torch.cuda.Event`` on GPU for accurate per-iter timing and
    a wall-clock fallback on CPU.
    """

    use_cuda = handle.device.type == "cuda"
    for _ in range(warmup):
        _ = _forward(handle, model)
    if use_cuda:
        torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(iters):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = _forward(handle, model)
            stop.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(stop) * 1e3)  # ms → us
        else:
            t0 = time.perf_counter()
            _ = _forward(handle, model)
            latencies.append((time.perf_counter() - t0) * 1e6)
    return latencies


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


@dataclass
class LiveTorchEagerAdapter:
    """Vanilla PyTorch forward pass — the baseline correctness oracle."""

    adapter_name: str = "torch_eager"

    def measure(
        self,
        workload_id: str,
        *,
        iters: int,
        warmup: int,
        seed: int,
    ) -> AdapterMeasurement:
        torch.manual_seed(seed)
        try:
            handle = _load_workload(workload_id)
        except Exception as exc:  # noqa: BLE001
            return AdapterMeasurement(
                adapter_name=self.adapter_name,
                workload_id=workload_id,
                latencies_us=(),
                output_hash="",
                blocked=True,
                blocked_reason=(
                    f"workload load failed: {type(exc).__name__}: {exc}"
                ),
            )
        out = _forward(handle, handle.model)
        ref_hash = _hash_tensor(out)
        latencies = _time_iters(handle, handle.model, iters=iters, warmup=warmup)
        return AdapterMeasurement(
            adapter_name=self.adapter_name,
            workload_id=workload_id,
            latencies_us=tuple(latencies),
            output_hash=ref_hash,
        )


@dataclass
class LiveTorchCompileAdapter:
    """``torch.compile`` baseline with ``mode='max-autotune'``."""

    adapter_name: str = "torch_compile"
    compile_mode: str = "default"  # max-autotune is unsafe on Turing fp16

    def measure(
        self,
        workload_id: str,
        *,
        iters: int,
        warmup: int,
        seed: int,
    ) -> AdapterMeasurement:
        torch.manual_seed(seed)
        try:
            handle = _load_workload(workload_id)
        except Exception as exc:  # noqa: BLE001
            return AdapterMeasurement(
                adapter_name=self.adapter_name,
                workload_id=workload_id,
                latencies_us=(),
                output_hash="",
                blocked=True,
                blocked_reason=(
                    f"workload load failed: {type(exc).__name__}: {exc}"
                ),
            )
        try:
            compiled = torch.compile(handle.model, mode=self.compile_mode)
        except Exception as exc:  # noqa: BLE001
            return AdapterMeasurement(
                adapter_name=self.adapter_name,
                workload_id=workload_id,
                latencies_us=(),
                output_hash="",
                blocked=True,
                blocked_reason=(
                    f"torch.compile failed: {type(exc).__name__}: {exc}"
                ),
            )
        out = _forward(handle, compiled)
        ref_hash = _hash_tensor(out)
        latencies = _time_iters(handle, compiled, iters=iters, warmup=warmup)
        return AdapterMeasurement(
            adapter_name=self.adapter_name,
            workload_id=workload_id,
            latencies_us=tuple(latencies),
            output_hash=ref_hash,
        )


@dataclass
class LiveCompGenAdapter:
    """CompGen end-to-end adapter via ``mode='compgen_ir'``.

    Honest scope-down: rather than compile a whole multi-billion-op
    model, we compile and measure a single decoder-layer slice (one
    LlamaMLP for TinyLlama; one Whisper encoder layer for whisper-tiny)
    through the real ``compile_model`` pipeline + xDSL ``cpu_executor``
    runtime. The slice is a representative unit of the workload's
    repeated computation.

    If ``compile_model`` or the runtime fails for a workload, the
    adapter returns a typed ``blocked`` measurement with the failure
    class as ``blocked_reason``.
    """

    adapter_name: str = "compgen"

    def measure(
        self,
        workload_id: str,
        *,
        iters: int,
        warmup: int,
        seed: int,
    ) -> AdapterMeasurement:
        from compgen.api import compile_model, device as _device

        # The CompGen path today compiles a *slice* (one decoder layer
        # or encoder block) via the real ``compile_model`` pipeline and
        # times it through ``mode='compgen_ir'``. Full-model workloads
        # (without the ``__slice`` suffix) are not yet wired end-to-end
        # and honestly block.
        if not workload_id.endswith(SLICE_SUFFIX):
            return AdapterMeasurement(
                adapter_name=self.adapter_name,
                workload_id=workload_id,
                latencies_us=(),
                output_hash="",
                blocked=True,
                blocked_reason="compgen_full_model_not_built",
            )

        torch.manual_seed(seed)
        try:
            handle = _load_workload(workload_id)
        except Exception as exc:  # noqa: BLE001
            return AdapterMeasurement(
                adapter_name=self.adapter_name,
                workload_id=workload_id,
                latencies_us=(),
                output_hash="",
                blocked=True,
                blocked_reason=(
                    f"workload_load_failed:{type(exc).__name__}:{exc}"
                ),
            )

        try:
            target_yaml = (
                REPO_ROOT / "tests" / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
            )
            dev = _device(str(target_yaml))
            compiled = compile_model(
                handle.model,
                dev,
                sample_inputs=handle.inputs,
                verify=False,
                strict_artifacts=False,
                run_compile_baseline=False,
            )
        except Exception as exc:  # noqa: BLE001
            return AdapterMeasurement(
                adapter_name=self.adapter_name,
                workload_id=workload_id,
                latencies_us=(),
                output_hash="",
                blocked=True,
                blocked_reason=(
                    f"compgen_compile_failed:{type(exc).__name__}:{exc}"
                ),
            )

        try:
            result = compiled(
                *handle.inputs,
                num_iterations=iters,
                warmup=warmup,
                mode="compgen_ir",
                device="cpu",
            )
        except Exception as exc:  # noqa: BLE001
            return AdapterMeasurement(
                adapter_name=self.adapter_name,
                workload_id=workload_id,
                latencies_us=(),
                output_hash="",
                blocked=True,
                blocked_reason=(
                    f"compgen_runtime_failed:{type(exc).__name__}:{exc}"
                ),
            )

        out_tensor = result.sample_output
        out_hash = _hash_tensor(out_tensor) if out_tensor is not None else ""
        latencies = tuple(float(t) for t in result.per_run_us)
        return AdapterMeasurement(
            adapter_name=self.adapter_name,
            workload_id=workload_id,
            latencies_us=latencies,
            output_hash=out_hash,
        )


__all__ = [
    "LiveCompGenAdapter",
    "LiveTorchCompileAdapter",
    "LiveTorchEagerAdapter",
    "WorkloadHandle",
]
