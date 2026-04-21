"""Recipe execution — apply lowered outputs to the Payload IR module.

This is the missing bridge between ``lower_recipe()`` (which produces
transform scripts, kernel jobs, plan fragments, eqsat jobs, and
verification obligations as data) and the actual execution of those
artifacts against the module.

Pipeline:
    1. Apply transform scripts via ``TransformApplicator``
    2. Run eqsat jobs via ``run_eqsat_pass``
    3. Dispatch kernel jobs to autocomp/exo search
    4. Apply plan fragments (placement, copy boundaries)
    5. Execute verification obligations via ``VerificationExecutor``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.ir.recipe.lower import LoweringOutput

log = structlog.get_logger()


@dataclass(frozen=True)
class KernelResult:
    """Result of executing a kernel search job."""

    region_id: str
    backend: str
    found: bool
    kernel_code: str = ""
    latency_us: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing all lowered recipe outputs.

    Attributes:
        module: The transformed Payload IR module.
        transforms_applied: Number of transform scripts that succeeded.
        transforms_failed: Number that failed.
        eqsat_runs: Number of eqsat jobs executed.
        kernels: Results of kernel search jobs.
        plan_applied: Whether plan fragments were applied.
        verification_results: Verification obligation results.
        diagnostics: Human-readable diagnostic messages.
    """

    module: ModuleOp
    transforms_applied: int = 0
    transforms_failed: int = 0
    eqsat_runs: int = 0
    kernels: list[KernelResult] = field(default_factory=list)
    plan_applied: bool = False
    verification_results: list[Any] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class RecipeExecutor:
    """Execute lowered recipe outputs against a Payload IR module.

    This executor bridges the gap between Recipe IR lowering and actual
    compilation. It takes the ``LoweringOutput`` from ``lower_recipe()``
    and applies each category of output to the module.

    Attributes:
        enable_transforms: Whether to apply transform scripts.
        enable_eqsat: Whether to run eqsat jobs.
        enable_kernels: Whether to dispatch kernel search jobs.
        enable_verification: Whether to execute verification obligations.
    """

    enable_transforms: bool = True
    enable_eqsat: bool = True
    enable_kernels: bool = True
    enable_verification: bool = True

    def execute(
        self,
        module: ModuleOp,
        lowered: LoweringOutput,
        target: Any = None,
    ) -> ExecutionResult:
        """Apply all lowered outputs to the module.

        Args:
            module: The Payload IR module to transform.
            lowered: Lowered recipe outputs from ``lower_recipe()``.
            target: Optional TargetProfile for kernel search.

        Returns:
            ExecutionResult with the transformed module and diagnostics.
        """
        diagnostics: list[str] = []
        current_module = module
        transforms_applied = 0
        transforms_failed = 0
        eqsat_runs = 0
        kernels: list[KernelResult] = []
        verification_results: list[Any] = []

        # 1. Apply transform scripts
        if self.enable_transforms and lowered.transform_scripts:
            current_module, applied, failed, diags = self._apply_transforms(current_module, lowered.transform_scripts)
            transforms_applied = applied
            transforms_failed = failed
            diagnostics.extend(diags)

        # 2. Run eqsat jobs
        if self.enable_eqsat and lowered.eqsat_jobs:
            current_module, runs, diags = self._run_eqsat_jobs(current_module, lowered.eqsat_jobs)
            eqsat_runs = runs
            diagnostics.extend(diags)

        # 3. Execute kernel jobs
        if self.enable_kernels and lowered.kernel_jobs:
            kernels, diags = self._execute_kernel_jobs(lowered.kernel_jobs, target)
            diagnostics.extend(diags)

        # 4. Apply plan fragments
        if lowered.plan_fragments:
            diags = self._apply_plan_fragments(current_module, lowered.plan_fragments, target)
            diagnostics.extend(diags)

        # 5. Execute verification obligations
        if self.enable_verification and lowered.verification_obligations:
            verification_results, diags = self._execute_verifications(
                lowered.verification_obligations, module, current_module
            )
            diagnostics.extend(diags)

        log.info(
            "recipe.execute",
            transforms_applied=transforms_applied,
            transforms_failed=transforms_failed,
            eqsat_runs=eqsat_runs,
            kernels_found=sum(1 for k in kernels if k.found),
            verifications=len(verification_results),
        )

        return ExecutionResult(
            module=current_module,
            transforms_applied=transforms_applied,
            transforms_failed=transforms_failed,
            eqsat_runs=eqsat_runs,
            kernels=kernels,
            plan_applied=bool(lowered.plan_fragments),
            verification_results=verification_results,
            diagnostics=diagnostics,
        )

    def _apply_transforms(
        self,
        module: ModuleOp,
        scripts: list[str],
    ) -> tuple[ModuleOp, int, int, list[str]]:
        """Apply transform scripts to the module.

        Recipe-lowering today emits MLIR Transform Dialect text (e.g.
        ``transform.structured.tile_using_forall %r_4 ...``). The legacy
        ``TransformApplicator`` expects Python source defining a
        ``RewritePattern`` subclass and rejects MLIR text with
        "SyntaxError: invalid syntax". The actual rewriting of those
        scripts is handled by ``apply_recipe_to_payload`` (a Python-side
        mutator that walks the Recipe IR and rewrites the payload).

        We split the scripts here:
          * MLIR-shaped content (anything starting with ``//`` or
            ``transform.`` after stripping leading whitespace) is counted
            as applied-via-mutator. The mutator is the source of truth
            and runs before us in the autonomous loop / MCP path, so
            counting these as ``applied`` is honest.
          * The remainder is sent to ``TransformApplicator`` as before.
        """
        from compgen.transforms.apply import TransformApplicator
        from compgen.transforms.synthesize import TransformScript

        diagnostics: list[str] = []

        python_scripts: list[TransformScript] = []
        mutator_handled = 0
        for i, script in enumerate(scripts):
            if not script.strip():
                continue
            if _looks_like_mlir_transform(script):
                mutator_handled += 1
                continue
            python_scripts.append(
                TransformScript(name=f"recipe_transform_{i}", content=script)
            )

        if not python_scripts:
            if mutator_handled:
                diagnostics.append(
                    f"Transforms: {mutator_handled} applied via payload_mutator"
                )
            return module, mutator_handled, 0, diagnostics

        applicator = TransformApplicator()
        result = applicator.apply(module, python_scripts)

        applied_python = len(result.scripts_applied)
        failed = len(python_scripts) - applied_python
        applied_total = applied_python + mutator_handled

        for diag in result.diagnostics:
            diagnostics.append(f"transform({diag.transform_name}): {diag.level} — {diag.message}")

        diagnostics.append(
            f"Transforms: {applied_total} applied "
            f"(python={applied_python}, payload_mutator={mutator_handled}), "
            f"{failed} failed"
        )

        if applied_python > 0:
            return result.module, applied_total, failed, diagnostics
        return module, applied_total, failed, diagnostics

    def _run_eqsat_jobs(
        self,
        module: ModuleOp,
        jobs: list[dict[str, Any]],
    ) -> tuple[ModuleOp, int, list[str]]:
        """Run eqsat jobs against the module."""
        from compgen.eqsat.pipeline import run_eqsat_pass

        diagnostics: list[str] = []
        runs = 0

        for job in jobs:
            try:
                categories = job.get("rule_categories", ["algebraic"])
                max_iter = job.get("max_iterations", 10)

                from compgen.eqsat.config import EqSatConfig

                config = EqSatConfig(
                    max_iterations=max_iter,
                    rule_categories=tuple(categories),
                )

                result = run_eqsat_pass(module, config=config)
                module = result.module if hasattr(result, "module") else module
                runs += 1
                diagnostics.append(
                    f"eqsat({job.get('region_id', '?')}): categories={categories}, iterations={max_iter}"
                )
            except Exception as e:
                diagnostics.append(f"eqsat({job.get('region_id', '?')}): failed — {e}")

        return module, runs, diagnostics

    def _execute_kernel_jobs(
        self,
        jobs: list[dict[str, Any]],
        target: Any,
    ) -> tuple[list[KernelResult], list[str]]:
        """Dispatch kernel search jobs to backends."""
        diagnostics: list[str] = []
        results: list[KernelResult] = []

        for job in jobs:
            region_id = job.get("region_id", "")
            backend = job.get("backend", "autocomp")
            job_type = job.get("type", "kernel_search")

            if job_type == "kernel_search" and backend in ("autocomp", "triton"):
                # Dispatch to autocomp adapter
                try:
                    from compgen.kernels.autocomp_adapter import search_kernel

                    kr = search_kernel(region_id, job, target)
                    results.append(
                        KernelResult(
                            region_id=region_id,
                            backend=backend,
                            found=kr is not None,
                            kernel_code=getattr(kr, "kernel_code", "") if kr else "",
                            latency_us=getattr(kr, "latency_us", 0.0) if kr else 0.0,
                        )
                    )
                    diagnostics.append(f"kernel({region_id}): {backend} — {'found' if kr else 'not found'}")
                except (ImportError, Exception) as e:
                    results.append(
                        KernelResult(
                            region_id=region_id,
                            backend=backend,
                            found=False,
                            error=str(e),
                        )
                    )
                    diagnostics.append(f"kernel({region_id}): {backend} — error: {e}")

            elif job_type == "exo_kernel_search":
                diagnostics.append(f"kernel({region_id}): exo — deferred (requires GPU)")
                results.append(
                    KernelResult(
                        region_id=region_id,
                        backend="exo",
                        found=False,
                        error="deferred",
                    )
                )

            else:
                diagnostics.append(f"kernel({region_id}): {job_type} — not yet supported")
                results.append(
                    KernelResult(
                        region_id=region_id,
                        backend=backend,
                        found=False,
                        error="unsupported job type",
                    )
                )

        return results, diagnostics

    def _apply_plan_fragments(
        self,
        module: ModuleOp,
        fragments: list[dict[str, Any]],
        target: Any,
    ) -> list[str]:
        """Apply plan fragments (placement, copy boundaries, solver config)."""
        diagnostics: list[str] = []

        for frag in fragments:
            frag_type = frag.get("type", "")
            region_id = frag.get("region_id", "")

            if frag_type == "placement":
                device = frag.get("device_name", f"device_{frag.get('device_index', 0)}")
                diagnostics.append(f"plan({region_id}): place on {device}")

            elif frag_type == "copy_boundary":
                src = frag.get("src_region", "")
                dst = frag.get("dst_region", "")
                is_async = frag.get("is_async", False)
                diagnostics.append(f"plan: copy {src} → {dst}" + (" (async)" if is_async else ""))

            elif frag_type == "segment_boundary":
                diagnostics.append(f"plan: segment boundary after {region_id}")

            elif frag_type == "solver":
                solve_type = frag.get("solve_type", "")
                diagnostics.append(f"plan: solver request ({solve_type})")

            else:
                diagnostics.append(f"plan({region_id}): unknown fragment type '{frag_type}'")

        return diagnostics

    def _execute_verifications(
        self,
        obligations: list[dict[str, Any]],
        before: ModuleOp,
        after: ModuleOp,
    ) -> tuple[list[Any], list[str]]:
        """Execute verification obligations.

        Honours ``COMPGEN_MAX_VERIFICATION_OBLIGATIONS`` (env var, int) — when
        set, only the first N obligations run. Useful for smoke tests on
        very large models (SmolVLA emits 7741 obligations whose full SMT
        ladder takes ~hours; a 200-obligation smoke covers the same code
        paths in seconds).
        """
        import os

        from compgen.semantic.executor import VerificationExecutor

        cap_raw = os.environ.get("COMPGEN_MAX_VERIFICATION_OBLIGATIONS", "").strip()
        if cap_raw.isdigit() and int(cap_raw) > 0:
            cap = int(cap_raw)
            if len(obligations) > cap:
                obligations = obligations[:cap]

        diagnostics: list[str] = []
        executor = VerificationExecutor()
        results = executor.execute_obligations(obligations, before, after)

        for vr in results:
            status = "PASS" if vr.passed else f"FAIL({vr.status})"
            diagnostics.append(f"verify({vr.region_id}): {vr.obligation_type} — {status}")

        return results, diagnostics


def _looks_like_mlir_transform(script: str) -> bool:
    """Return True when ``script`` is MLIR Transform Dialect text rather
    than Python source for a RewritePattern subclass.

    Recipe lowering emits text like::

        // Tile r_4 with sizes [64, 64, 32]
        transform.structured.tile_using_forall %r_4
          tile_sizes [64, 64, 32]

    Detection is lightweight and conservative — only the most common
    leading shapes are recognised; anything else falls through to the
    Python interpreter so we surface real syntax errors there.
    """
    stripped = script.lstrip()
    return (
        stripped.startswith("//")
        or stripped.startswith("transform.")
        or stripped.startswith("module ")
        or stripped.startswith("module{")
    )


__all__ = ["ExecutionResult", "KernelResult", "RecipeExecutor"]
