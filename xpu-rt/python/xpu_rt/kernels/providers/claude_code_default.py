"""Claude Code as the default kernel provider.

Implements the contract from ``compgen.kernels.provider``: takes a
``KernelContract`` (or v3 contract via the bridge), formulates a prompt
from ``KernelContractV3.kernel_facing()`` if available, and asks Claude
Code (or any pluggable code-generator callable) to produce a kernel
implementation in one shot.

Three execution modes — the provider is mode-agnostic and just calls a
``CodegenCallable``:

* ``in_session`` — when the agentic compile loop is already running
  inside a Claude Code session, the callable is a thin shim that yields
  the prompt back to the calling MCP session. The kernel comes back as
  a tool-call response — **no extra Anthropic API call, no extra cost**.
  This is the "as much as we can is contained within Claude Code" path.

* ``cli_subprocess`` — for headless ``compile_with_llm`` runs (CI, batch),
  the callable spawns ``claude-code`` as a subprocess with the prompt.

* ``api`` — direct Anthropic API call. Fallback when neither in-session
  nor CLI is available.

The router (``escalating_router.py``) selects this provider first; only
on gate failure or perf miss does it escalate to autocomp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.kernels.provider import (
    BidPreview,
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)

# ---------------------------------------------------------------------------
# Codegen callable — pluggable execution mode
# ---------------------------------------------------------------------------


class CodegenCallable:
    """A callable that turns ``(prompt, contract)`` into kernel source.

    Implementations:

    * ``InSessionCallable`` — yields the prompt to the active Claude Code
      session via the configured MCP transport; awaits the tool-call
      response. Zero extra LLM cost (uses the parent session's context).
    * ``CliSubprocessCallable`` — spawns ``claude-code`` CLI with the
      prompt; reads the tool's stdout for the kernel source.
    * ``ApiCallable`` — direct Anthropic API call (one-shot).

    The provider doesn't care which mode is wired — it just calls
    ``codegen(prompt, contract) -> str``.
    """

    def __call__(self, prompt: str, contract: KernelContract) -> str:
        raise NotImplementedError


@dataclass
class StubCodegen(CodegenCallable):
    """Test/CI codegen — returns a deterministic stub from a lookup table.

    Useful for unit tests and offline smoke runs. Production paths use
    one of the three real callables above.
    """

    canned: dict[str, str] = field(default_factory=dict)

    def __call__(self, prompt: str, contract: KernelContract) -> str:
        # Match by op_family first, then by region_id, then default.
        return (
            self.canned.get(contract.op_family)
            or self.canned.get(contract.region_id)
            or self.canned.get("__default__", "// stub kernel\n")
        )


@dataclass
class InSessionCodegen(CodegenCallable):
    """Codegen that reads from the MCP session's kernel cache.

    The orchestration layer pre-populates the cache by issuing
    ``request_kernel_codegen`` MCP tool calls; the agent (Claude Code)
    fulfils them via ``register_kernel_result``. By the time the
    provider runs, the cache holds the kernel under the contract's
    fingerprint.

    Cache miss → returns empty source so the provider's
    ``ProviderResult.found`` is False, and the escalating router falls
    through to the next tier (autocomp). This is the "as much as we can
    contained within Claude Code" path: when a kernel is already in the
    session's pocket, no Anthropic API call fires.
    """

    sm: Any  # SessionManager — typed Any to avoid import cycle
    session_id: str

    def __call__(self, prompt: str, contract: KernelContract) -> str:
        from compgen.mcp.tools.kernel import _kernel_cache

        # The bridge attaches the v3 view; we serialise its salient bits
        # the same way the MCP fingerprint does so cache hits work.
        view = (contract.constraints or {}).get("kernel_facing_view")
        if view is None:
            return ""
        fp = _fingerprint_from_view(view, contract)
        cache = _kernel_cache(self.sm.get(self.session_id))
        entry = cache.entries.get(fp)
        if entry is None:
            return ""
        return entry.kernel_code


def _fingerprint_from_view(view: Any, contract: KernelContract) -> str:
    """Build the same fingerprint that ``contract_fingerprint`` uses on
    the JSON form. Keeps in-session cache lookups stable across the
    Python-object form (provider side) and the dict form (MCP side).
    """
    from compgen.mcp.tools.kernel import contract_fingerprint

    # Reconstruct the v3-shape dict from the kernel_facing view + the v1
    # contract's target name. Only the fields contract_fingerprint reads
    # need to be present.
    target = contract.target_name or ""
    io_inputs = []
    for t in view.io.inputs:
        io_inputs.append(
            {
                "name": t.name,
                "shape": {"dims": list(t.shape.dims)},
                "dtype_class": list(t.dtype_class),
                "layout": t.layout.value,
                "alignment_bytes": t.alignment_bytes,
            }
        )
    io_outputs = []
    for t in view.io.outputs:
        io_outputs.append(
            {
                "name": t.name,
                "shape": {"dims": list(t.shape.dims)},
                "dtype_class": list(t.dtype_class),
                "layout": t.layout.value,
                "alignment_bytes": t.alignment_bytes,
            }
        )
    n = view.io.numerics
    contract_dict = {
        "op_name": view.op_name,
        "archetype": view.archetype.value,
        "granularity": view.granularity.value,
        "io": {
            "inputs": io_inputs,
            "outputs": io_outputs,
            "attributes": [{"name": a.name, "value": a.value} for a in view.io.attributes],
            "numerics": {
                "accumulator_dtype": n.accumulator_dtype,
                "fast_math": n.fast_math,
                "max_relative_error": n.max_relative_error,
            },
        },
        "orchestration": {
            "execution": {"hardware": {"target_name": target}},
        },
    }
    return contract_fingerprint(contract_dict)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCodeKernelProvider:
    """Default kernel provider — Claude Code one-shot generation.

    Cost profile (per kernel attempt):
        * in-session mode  : $0   (folded into parent session context)
        * cli/api mode     : ~$0.05 (one prompt + one completion)

    Compare to autocomp's ~$13-20 per kernel for its 8-iter beam search.
    Claude Code wins for well-known patterns + precise contracts; the
    router escalates to autocomp on the long-tail (~5%).
    """

    codegen: CodegenCallable
    name_str: str = "claude_code_default"
    accepted_op_families: tuple[str, ...] = ()  # empty = accept all
    _exports: list[KnowledgeExport] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.name_str

    def accepts_contract(self, contract: KernelContract) -> bool:
        """Default-accept everything when no filter is configured.

        The router relies on this provider answering ``True`` widely —
        escalation to autocomp happens via gate-failure feedback, not
        via per-contract refusal.
        """
        if not self.accepted_op_families:
            return True
        return contract.op_family in self.accepted_op_families

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        prompt = self._build_prompt(contract)
        try:
            kernel_code = self.codegen(prompt, contract)
        except Exception as exc:  # noqa: BLE001
            # Emit a not-found result so the router escalates cleanly;
            # the failure reason rides in metadata for observability.
            return ProviderResult(
                found=False,
                kernel_code="",
                metadata={"error": f"{type(exc).__name__}: {exc}"},
            )

        if not kernel_code.strip():
            return ProviderResult(
                found=False,
                kernel_code="",
                metadata={"error": "codegen returned empty source"},
            )

        # Record what we generated so the router / memory layer can replay.
        export = KnowledgeExport(
            kind="claude_code_kernel",
            scope="contract",
            scope_key=f"{contract.op_family}:{contract.target_name}",
            content=kernel_code,
            confidence=0.6,  # one-shot — escalation may bump this
        )
        self._exports.append(export)

        return ProviderResult(
            found=True,
            kernel_code=kernel_code,
            language=self._guess_language(kernel_code),
            iterations_used=1,
            total_candidates=1,
            knowledge_exports=[export],
            metadata={
                "provider": self.name,
                "prompt_chars": len(prompt),
                "code_chars": len(kernel_code),
            },
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        return list(self._exports)

    # -- Phase D / M-56: bid() ------------------------------------------------

    def bid(self, contract_v3: Any) -> BidPreview:
        """Cheap pre-codegen estimate for a :class:`KernelContractV3`.

        Cache-aware: when the in-session cache (``InSessionCodegen``)
        already holds a kernel for this contract's fingerprint, the bid
        carries ``confidence=0.9`` and ``time_to_generate_s_estimate=1.0``
        (essentially free fulfill). On cache miss, the bid carries
        ``confidence=0.3`` and ``time_to_generate_s_estimate=900`` (the
        provider runs the codegen callable, which for a real Claude-Code
        subagent can take minutes).

        Pure ``StubCodegen`` configurations report a deterministic
        ``confidence=0.5`` / ``time_to_generate_s_estimate=0.05`` —
        that's the test-fixture path; real production picks a real
        codegen callable.
        """
        cache_hit = False
        codegen = self.codegen
        try:
            from compgen.kernels.providers.claude_code_default import (
                InSessionCodegen,
                StubCodegen,
            )

            if isinstance(codegen, InSessionCodegen):
                # Probe the cache without invoking codegen.
                from compgen.kernels.contract_v3 import KernelContractV3
                from compgen.mcp.tools.kernel import _kernel_cache, contract_fingerprint

                if isinstance(contract_v3, KernelContractV3):
                    facing = contract_v3.kernel_facing()
                    fp_dict = _kernel_facing_to_fp_dict(facing, contract_v3)
                    fp = contract_fingerprint(fp_dict)
                    cache = _kernel_cache(codegen.sm.get(codegen.session_id))
                    cache_hit = fp in cache.entries
            elif isinstance(codegen, StubCodegen):
                # Stub is deterministic + fast — treat every call as a "hit".
                cache_hit = True
        except Exception:  # noqa: BLE001
            cache_hit = False

        if cache_hit:
            return BidPreview(
                provider_name=self.name,
                perf_estimate_us=float("inf"),
                confidence=0.9,
                time_to_generate_s_estimate=1.0,
                rationale="cache_hit",
                cache_hit=True,
            )

        return BidPreview(
            provider_name=self.name,
            perf_estimate_us=float("inf"),
            confidence=0.3,
            time_to_generate_s_estimate=900.0,
            rationale="claude_code_one_shot_codegen",
            cache_hit=False,
        )


def _kernel_facing_to_fp_dict(facing: Any, contract_v3: Any) -> dict[str, Any]:
    """Build the fingerprint-shaped dict from a v3 KernelFacingView.

    Mirrors :func:`_fingerprint_from_view` but reads directly from the
    v3 object rather than going through the legacy v1 contract.
    """
    target = ""
    try:
        target = contract_v3.orchestration.execution.hardware.target_name
    except AttributeError:
        target = ""

    io_inputs = []
    for t in facing.io.inputs:
        io_inputs.append(
            {
                "name": t.name,
                "shape": {"dims": list(t.shape.dims)},
                "dtype_class": list(t.dtype_class),
                "layout": t.layout.value,
                "alignment_bytes": t.alignment_bytes,
            }
        )
    io_outputs = []
    for t in facing.io.outputs:
        io_outputs.append(
            {
                "name": t.name,
                "shape": {"dims": list(t.shape.dims)},
                "dtype_class": list(t.dtype_class),
                "layout": t.layout.value,
                "alignment_bytes": t.alignment_bytes,
            }
        )
    n = facing.io.numerics
    return {
        "op_name": facing.op_name,
        "archetype": facing.archetype.value,
        "granularity": facing.granularity.value,
        "io": {
            "inputs": io_inputs,
            "outputs": io_outputs,
            "attributes": [{"name": a.name, "value": a.value} for a in facing.io.attributes],
            "numerics": {
                "accumulator_dtype": n.accumulator_dtype,
                "fast_math": n.fast_math,
                "max_relative_error": n.max_relative_error,
            },
        },
        "orchestration": {
            "execution": {"hardware": {"target_name": target}},
        },
    }

    # ----- internals -----

    def _build_prompt(self, contract: KernelContract) -> str:
        """Render the contract as a prompt the codegen callable can read.

        When the contract carries a v3 ``kernel_facing()`` projection in
        its ``constraints['kernel_facing_view']`` slot (the bridge
        injects it), we surface that explicitly. Otherwise we fall back
        to v1's terse ``op_family / shapes / dtypes / target`` render.
        """
        facing = (contract.constraints or {}).get("kernel_facing_view")
        lines = [
            "Generate a kernel for the following contract.",
            "",
            f"op_family    : {contract.op_family}",
            f"region_id    : {contract.region_id}",
            f"target       : {contract.target_name}",
            f"hardware_key : {contract.hardware_key}",
            f"objective    : {contract.objective}",
            f"layout       : {contract.layout}",
            f"input shapes : {contract.input_shapes}",
            f"output shapes: {contract.output_shapes}",
            f"dtypes       : {contract.dtypes}",
        ]
        if contract.constraints:
            extras = {k: v for k, v in contract.constraints.items() if k != "kernel_facing_view"}
            if extras:
                lines.append(f"constraints  : {extras}")
        if facing is not None:
            lines.append("")
            lines.append("KernelFacingView (v3) — exhaustive kernel-readable spec:")
            lines.append(repr(facing))
        lines.append("")
        lines.append("Output only the kernel source. No explanation, no markdown fences.")
        return "\n".join(lines)

    def _guess_language(self, code: str) -> str:
        head = code.lstrip()[:120].lower()
        if head.startswith("@triton.jit") or "import triton" in head:
            return "triton"
        if "__global__" in head or "#include <cuda" in head:
            return "cuda"
        if "#include" in head:
            return "c"
        if "def " in head or "import " in head:
            return "python"
        return "unknown"


__all__ = [
    "ClaudeCodeKernelProvider",
    "CodegenCallable",
    "InSessionCodegen",
    "StubCodegen",
]
