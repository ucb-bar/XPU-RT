"""Adapter wrapping autocomp for kernel generation.

Translates CompGen's PatternCluster into autocomp's search format and runs
the beam search. Uses autocomp's LLMClient, SearchStrategy, and EvalBackend.

The adapter:
1. Maps PatternCluster → reference Triton/CUDA code + test harness
2. Creates autocomp Prob + HardwareConfig
3. Runs SearchStrategy.optimize(iterations)
4. Returns the best verified kernel

Environment:
    GOOGLE_API_KEY must be set for Gemini-backed search.
    GPU must be available for CUDA kernel evaluation.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.agent.analyzer import PatternCluster
from compgen.observability.gemini_usage import (
    install_genai_instrumentation,
    tracking_source,
)
from compgen.targets.schema import TargetProfile


@dataclass(frozen=True)
class KernelResult:
    """Result from a kernel search.

    Attributes:
        cluster_id: Which cluster this kernel is for.
        kernel_code: The best kernel code found.
        language: Language ("triton", "cuda", "python").
        latency_us: Measured latency in microseconds.
        correct: Whether the kernel passed correctness tests.
        speedup_vs_baseline: Speedup over the reference implementation.
        iterations_used: How many search iterations were used.
        total_candidates: Total candidates evaluated.
        search_cost_tokens: LLM tokens consumed.
        plan: The optimization plan that produced this kernel.
    """

    cluster_id: str
    kernel_code: str
    language: str
    latency_us: float
    correct: bool
    speedup_vs_baseline: float
    iterations_used: int
    total_candidates: int
    search_cost_tokens: int
    plan: str


def _ensure_google_api_key() -> None:
    """Set GOOGLE_API_KEY from GEMMINI_API if not already set."""
    if "GOOGLE_API_KEY" not in os.environ:
        gemmini_key = os.environ.get("GEMMINI_API", "")
        if not gemmini_key:
            # Try loading from .env
            env_path = Path(__file__).parent.parent.parent.parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("GEMMINI_API="):
                        gemmini_key = line.split("=", 1)[1].strip()
                        break
        if gemmini_key:
            os.environ["GOOGLE_API_KEY"] = gemmini_key
    # Patch the google-genai SDK so autocomp's LLMClient calls flow into
    # the same usage tracker as our own GeminiClient. Idempotent + safe
    # when the SDK isn't importable.
    install_genai_instrumentation()


def _generate_reference_code(cluster: PatternCluster) -> str:
    """Generate a reference PyTorch implementation for a pattern cluster."""
    if cluster.pattern_type == "linear_chain":
        # MLP: linear → gelu → linear
        shapes = cluster.input_shapes
        first_shape = next(iter(shapes.values())) if shapes else (8, 768)
        out_shape = next(iter(cluster.output_shapes.values())) if cluster.output_shapes else (8, 768)
        m = first_shape[0]
        k_in = first_shape[-1]
        k_out = out_shape[-1]
        # Infer hidden dim from total FLOPs
        # FLOPs ≈ 2*M*K_in*hidden + 2*M*hidden*K_out
        # Rough: hidden ≈ sqrt(total_flops / (4*M))
        hidden = max(k_in, k_out) * 4  # rough estimate

        return f"""
import torch
import torch.nn.functional as F

def test(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
         w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    # Reference: linear -> gelu -> linear
    h = F.linear(x, w1, b1)  # [{m}, {k_in}] @ [{hidden}, {k_in}]^T -> [{m}, {hidden}]
    h = F.gelu(h)
    return F.linear(h, w2, b2)  # [{m}, {hidden}] @ [{k_out}, {hidden}]^T -> [{m}, {k_out}]
"""

    elif cluster.pattern_type == "linear":
        shapes = cluster.input_shapes
        first_shape = next(iter(shapes.values())) if shapes else (8, 768)
        return f"""
import torch
import torch.nn.functional as F

def test(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.linear(x, w, b)  # shape: {first_shape}
"""

    # Generic fallback
    return f"""
import torch

def test(*args):
    # Pattern: {cluster.pattern_type}
    # FLOPs: {cluster.total_flops}
    raise NotImplementedError("No reference code for pattern: {cluster.pattern_type}")
"""


def _generate_test_code(cluster: PatternCluster) -> str:
    """Generate a test harness that checks correctness."""
    return f"""
import torch

def get_test_inputs():
    # Generate random inputs matching cluster shapes
    torch.manual_seed(42)
    inputs = []
    for name, shape in {dict(cluster.input_shapes)}.items():
        inputs.append(torch.randn(shape, device='cuda', dtype=torch.float32))
    return inputs

def check_correctness(test_fn, ref_fn, inputs):
    ref_out = ref_fn(*inputs)
    test_out = test_fn(*inputs)
    return torch.allclose(ref_out, test_out, atol=1e-3, rtol=1e-3)
"""


@dataclass
class AutocompAdapter:
    """Adapter between CompGen and autocomp kernel search."""

    default_model: str = "gemini-2.0-flash"
    beam_size: int = 4
    max_iterations: int = 10
    num_plan_candidates: int = 4
    num_code_candidates: int = 2

    def search_kernel(
        self,
        cluster: PatternCluster,
        target: TargetProfile,
        budget: int | None = None,
        output_dir: str | Path | None = None,
    ) -> KernelResult:
        """Run autocomp beam search to find an optimized kernel.

        Args:
            cluster: Pattern cluster to generate a kernel for.
            target: Target hardware profile.
            budget: Override max_iterations.
            output_dir: Where to save search artifacts.

        Returns:
            KernelResult with the best kernel found.
        """
        _ensure_google_api_key()

        from autocomp.agents.cuda.cuda_agent import CudaLLMAgent
        from autocomp.agents.llm_ensemble import LLMEnsemble
        from autocomp.backend.kernelbench.kb_eval import KBEvalBackend
        from autocomp.hw_config import CudaHardwareConfig
        from autocomp.search.search import BeamSearchStrategy, Prob

        iterations = budget or self.max_iterations

        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="compgen_kernel_"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate reference code and test
        ref_code = _generate_reference_code(cluster)
        test_code = _generate_test_code(cluster)

        # Write to files
        sol_file = output_dir / "reference.py"
        sol_file.write_text(ref_code)
        test_file = output_dir / "test.py"
        test_file.write_text(test_code)

        # Create autocomp problem
        prob = Prob(
            prob_type="compgen",
            prob_id=0,
            sol_file=sol_file,
            test_file=test_file,
            context=f"Pattern: {cluster.pattern_type}, FLOPs: {cluster.total_flops:,}, "
            f"Kernel opportunity: {cluster.kernel_opportunity}",
        )

        # Create hardware config
        gpu_name = "NVIDIA GPU"
        for dev in target.devices:
            if dev.device_type == "gpu":
                gpu_name = dev.name
                break

        import torch

        hw_config = CudaHardwareConfig(
            gpu_name=gpu_name,
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda or "12.0",
        )

        # Create agent (LLMClient is managed internally by the agent)
        agent = CudaLLMAgent(
            hw_config=hw_config,
            models=[self.default_model],
            code_models=[self.default_model],
        )
        ensemble = LLMEnsemble(agents=[agent])

        # Create eval backend
        eval_backend = KBEvalBackend(hw_config=hw_config)

        # Create search strategy
        strategy = BeamSearchStrategy(
            output_dir=output_dir,
            eval_backend=eval_backend,
            agent=ensemble,
            orig_code=ref_code,
            prob=prob,
            metric="latency",
            simulator="",
            give_score_feedback=1.0,
            give_util_feedback=0.0,
            give_hw_feedback=0.0,
            include_ancestors=True,
            plan_icl_examples=False,
            code_icl_examples=False,
            dropout_menu_options=0.25,
            prevent_duplicate_level=1,
            translate_iters=0,
            translate_perf_threshold=1.2,
            translate_drop_original=False,
            translate_score=False,
        )

        # Run search — every Gemini call autocomp issues during this
        # block is tagged with source="autocomp" in the usage log.
        with tracking_source(
            "autocomp",
            cluster_id=cluster.cluster_id,
            pattern_type=cluster.pattern_type,
            iterations=iterations,
        ):
            strategy.optimize(iterations)

        # Extract best result
        best = self._extract_best(strategy, cluster.cluster_id, output_dir)
        return best

    def _extract_best(
        self,
        strategy: Any,
        cluster_id: str,
        output_dir: Path,
    ) -> KernelResult:
        """Extract the best kernel from search results."""
        # Read the results from the output directory
        best_code = ""
        best_latency = float("inf")
        best_plan = ""
        total_candidates = 0

        # Walk the output directory for candidate results
        for candidate_dir in sorted(output_dir.glob("candidates-iter-*")):
            for code_file in candidate_dir.glob("*.py"):
                total_candidates += 1

        # Try to find the best from the strategy's state
        # The beam search keeps track of the best candidates
        results_file = output_dir / "results.json"
        if results_file.exists():
            results = json.loads(results_file.read_text())
            if "best_score" in results:
                best_latency = results["best_score"]
            if "best_code" in results:
                best_code = results["best_code"]

        # Speedup-vs-baseline needs BOTH a measured candidate latency
        # and a measured (or rooflined) baseline on the same hardware.
        # ``best_latency`` is the candidate; the baseline timing is
        # whatever was written as ``baseline_score`` (if the search
        # strategy recorded one), else NaN. We never return a
        # placeholder ``1.0`` — an unknown speedup stays NaN so
        # downstream selectors can skip the candidate instead of
        # treating "no improvement" and "unmeasured" as identical.
        import math

        baseline_latency = math.nan
        if results_file.exists():
            results = json.loads(results_file.read_text())
            if "baseline_score" in results:
                baseline_latency = float(results["baseline_score"])

        if (
            math.isfinite(baseline_latency)
            and baseline_latency > 0
            and math.isfinite(best_latency)
            and best_latency > 0
        ):
            speedup_vs_baseline = baseline_latency / best_latency
        else:
            speedup_vs_baseline = math.nan

        return KernelResult(
            cluster_id=cluster_id,
            kernel_code=best_code,
            language="cuda",
            latency_us=best_latency,
            correct=best_latency < float("inf"),
            speedup_vs_baseline=speedup_vs_baseline,
            iterations_used=0,
            total_candidates=total_candidates,
            search_cost_tokens=0,
            plan=best_plan,
        )

    def quick_check(self, cluster: PatternCluster, target: TargetProfile) -> bool:
        """Quick check if autocomp search is viable for this cluster+target.

        Returns True if we have the necessary infrastructure (GPU, API key,
        eval backend) to run a search.
        """
        import torch

        if not torch.cuda.is_available():
            return False

        _ensure_google_api_key()
        if "GOOGLE_API_KEY" not in os.environ:
            return False

        # Check target has a GPU device
        has_gpu = any(d.device_type == "gpu" for d in target.devices)
        return has_gpu


def search_kernel(region_id: str, job: dict[str, Any], target: Any) -> dict[str, Any]:
    """Bridge between recipe executor and kernel providers.

    Extracts ``op_family`` and shape information from *job*, builds a
    :class:`~compgen.kernels.provider.KernelContract`, and tries the
    :class:`~compgen.kernels.providers.triton_templates.TritonTemplateProvider`
    first.  Falls back to an empty result if nothing matches.

    Args:
        region_id: Identifier for the IR region being lowered.
        job: Dict with at least ``op_family`` and optionally ``input_shapes``,
            ``output_shapes``, ``dtypes``, ``target_name``.
        target: Target profile (used for metadata only).

    Returns:
        Dict with keys ``region_id``, ``found``, ``kernel_code``,
        ``latency_us``, and ``error``.
    """
    from compgen.kernels.provider import KernelContract, SearchBudget
    from compgen.kernels.providers.triton_templates import TritonTemplateProvider

    op_family: str = job.get("op_family", "")
    raw_input_shapes = job.get("input_shapes", ())
    raw_output_shapes = job.get("output_shapes", ())
    dtypes = tuple(job.get("dtypes", ("float32",)))
    target_name: str = job.get("target_name", "")

    input_shapes = tuple(tuple(s) for s in raw_input_shapes)
    output_shapes = tuple(tuple(s) for s in raw_output_shapes)

    contract = KernelContract(
        region_id=region_id,
        op_family=op_family,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        dtypes=dtypes,
        target_name=target_name,
    )

    provider = TritonTemplateProvider()
    budget = SearchBudget()

    try:
        if provider.accepts_contract(contract):
            result = provider.search(contract, budget)
            return {
                "region_id": region_id,
                "found": result.found,
                "kernel_code": result.kernel_code,
                "latency_us": result.latency_us,
                "error": None,
            }
    except Exception as exc:
        return {
            "region_id": region_id,
            "found": False,
            "kernel_code": "",
            "latency_us": 0.0,
            "error": str(exc),
        }

    return {
        "region_id": region_id,
        "found": False,
        "kernel_code": "",
        "latency_us": 0.0,
        "error": f"No provider accepts op_family={op_family!r}",
    }


__all__ = ["AutocompAdapter", "KernelResult", "search_kernel"]
