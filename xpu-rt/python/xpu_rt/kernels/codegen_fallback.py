"""Provider fallback for codegen.

When the auto-generated codegen stage's native emitter (Triton / vendor)
declines all candidates, dispatch to any registered ``KernelProvider``
that ``accepts_contract`` for the IR-extracted kernel contracts. The
first provider whose :meth:`KernelProvider.search` returns
``ProviderResult.found=True`` with a non-empty ``kernel_code`` wins.

Output is shaped for :func:`compgen.runtime.bundle_emit._extract_generated_kernels`
so that the bundle stage writes the source files into
``bundle/generated_kernels/<provider>/<op_name>.<extension>`` without
any further plumbing.

This honours the documented :class:`~compgen.kernels.provider.KernelProvider`
extension point; packs ship a provider via the
``compgen.kernels.providers`` entry-point group and the codegen stage
calls them when the in-tree emitter has nothing to ship.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from compgen.kernels.contracts import build_kernel_contracts, spec_to_provider_contract
from compgen.kernels.provider import SearchBudget
from compgen.kernels.registry import ProviderRegistry
from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR

if TYPE_CHECKING:
    from xdsl.dialects.builtin import ModuleOp

    from compgen.kernels.provider import KernelProvider
    from compgen.targets.schema import TargetProfile

log = structlog.get_logger(__name__)


# Map provider-result language → bundle file extension. Providers are
# free to use any string; unknown languages fall back to ``txt``.
_LANGUAGE_EXTENSIONS: dict[str, str] = {
    "triton": "py",
    "python": "py",
    "cuda": "cu",
    "hip": "cpp",
    "cpp": "cpp",
    "c++": "cpp",
    "c": "c",
    "asm": "S",
    "ptx": "ptx",
    "metal": "metal",
}


def _ext_for(language: str) -> str:
    return _LANGUAGE_EXTENSIONS.get(language.lower(), "txt") if language else "txt"


def _discover_providers() -> list[KernelProvider]:
    """Collect kernel providers from the plugin registry.

    Scans the ``compgen.kernels.providers`` entry-point group via
    :func:`compgen.plugins.discover_all`. Each loaded entry resolves to
    either a :class:`KernelProvider` instance, a class (instantiated
    with no args), or a factory callable returning one.
    """
    try:
        from compgen.plugins import GROUP_KERNEL_PROVIDERS, discover_all, registry
    except Exception:  # noqa: BLE001
        return []

    discover_all()
    out: list[KernelProvider] = []
    for plugin in registry().get(GROUP_KERNEL_PROVIDERS):
        obj = plugin.object
        # Resolve class → instance, or factory → instance, or pass through.
        try:
            if isinstance(obj, type):
                instance = obj()
            elif callable(obj) and not all(hasattr(obj, m) for m in ("name", "accepts_contract", "search")):
                instance = obj()
            else:
                instance = obj
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "codegen_fallback.provider_instantiate_failed",
                plugin_name=plugin.name,
                error=str(exc),
            )
            continue
        out.append(instance)
    return out


def run_provider_fallback(
    module: ModuleOp,
    target_profile: TargetProfile,
    sample_inputs: Any = None,
    *,
    extra_providers: list[KernelProvider] | None = None,
    budget: SearchBudget | None = None,
    memory: Any = None,
    feedback_out: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Run kernel-provider fallback over every contract in ``module``.

    Args:
        module: The post-pipeline payload IR module.
        target_profile: Target profile (consumed by
            :func:`spec_to_provider_contract` to set hardware key/name).
        sample_inputs: Optional concrete sample inputs used to refine
            shapes during contract extraction.
        extra_providers: Optional in-process providers to consider in
            addition to those discovered via entry points. Useful for
            tests and for callers that want to register providers
            programmatically.
        budget: Search budget per provider invocation. Defaults to
            :class:`SearchBudget` defaults.
        memory: Optional :class:`compgen.memory.store.CompilerMemory`
            (or any object exposing ``ingest_provider_knowledge``).
            When provided, accumulated ``ProviderResult.knowledge_exports``
            from every dispatched provider are persisted via
            :meth:`ProviderRegistry.ingest_knowledge` after the search
            loop. Without ``memory``, knowledge is dropped (current
            backward-compatible default).
        feedback_out: Optional mutable list. When provided, accumulated
            ``ProviderResult.contract_feedback`` items are appended to
            it. Lets callers route the bidirectional contract-evolution
            signal without owning a ``CompilerMemory``.

    Returns:
        A list of dicts shaped for the bundle artifact contract::

            {"provider": str, "op_name": str, "source": str, "extension": str,
             "language": str, "region_id": str}

        The list is empty when no provider accepted any contract or
        when no contract produced a ``found=True`` result. As a side
        effect, the matching IR ops have their
        ``compgen.codegen_backend`` annotation rewritten from
        ``"fallback"`` to the winning provider's ``name``.
    """
    providers = list(extra_providers or [])
    providers.extend(_discover_providers())
    if not providers:
        log.info("codegen_fallback.no_providers")
        return []

    # Sort by ``priority`` (REQ-017) — higher wins. Stable sort
    # preserves entry-point order for equal-priority providers, which
    # keeps behaviour deterministic. Default priority is 0 for any
    # provider that doesn't expose the attribute.
    providers.sort(key=lambda p: -int(getattr(p, "priority", 0)))

    # Build a transient ProviderRegistry for ordered dispatch — providers
    # are registered in priority-descending order so the registry's
    # "first that accepts" rule yields the priority-correct winner.
    pr = ProviderRegistry()
    for p in providers:
        pr.register(p)

    specs = build_kernel_contracts(module, target_profile, sample_inputs)
    if not specs:
        return []

    out: list[dict[str, Any]] = []
    # Per-region provenance: ``spec index → winning provider name``.
    # Each spec carries its own ``region_id`` + ``dispatch_id`` in
    # ``contract.metadata`` (REQ-026), set by the contract extractor
    # when the IR op was tagged with ``compgen.region_id``. No
    # parallel walk needed.
    region_winners: dict[int, str] = {}
    for i, spec in enumerate(specs):
        ir_meta = spec.contract.metadata or {}
        region_id = ir_meta.get("region_id") or f"region_{i}"
        dispatch_id = ir_meta.get("dispatch_id") or region_id
        contract = spec_to_provider_contract(spec, region_id, target_profile)

        # ProviderRegistry.search() walks providers in registration
        # order and returns the first found result; on no match it
        # returns an empty ProviderResult (found=False).
        result = pr.search(contract, budget)
        if not result.found or not result.kernel_code:
            continue

        # Recover the winning provider name. ProviderRegistry doesn't
        # expose which provider answered, so we re-walk to identify it.
        # (Cheap — accepts_contract is meant to be O(1).)
        winner = next(
            (p for p in providers if p.accepts_contract(contract)),
            None,
        )
        provider_name = winner.name if winner is not None else "unknown"
        region_winners[i] = provider_name

        entry: dict[str, Any] = {
            "provider": provider_name,
            "op_name": spec.contract.op_name,
            "region_id": region_id,
            "dispatch_id": dispatch_id,
            "source": result.kernel_code,
            "language": result.language,
            "extension": _ext_for(result.language),
            "emit_mode": result.emit_mode,
        }
        if result.dispatch_geometry is not None:
            geo = result.dispatch_geometry
            entry["dispatch_geometry"] = {
                "num_warps": geo.num_warps,
                "threadblock_shape": list(geo.threadblock_shape),
                "grid_shape": list(geo.grid_shape),
            }
        if result.kernel_files:
            entry["kernel_files"] = dict(result.kernel_files)
        if result.expected_inputs:
            entry["expected_inputs"] = dict(result.expected_inputs)
        out.append(entry)

    if region_winners:
        # Map region_id → winning provider for the per-op rewrite.
        rid_winners: dict[str, str] = {}
        for spec_idx, name in region_winners.items():
            spec_meta = specs[spec_idx].contract.metadata or {}
            rid = spec_meta.get("region_id")
            if rid:
                rid_winners[rid] = name
        _annotate_codegen_backend_by_region_id(module, rid_winners)

    # Persist provider-emitted knowledge if a memory was supplied.
    ingested = 0
    if memory is not None:
        try:
            ingested = pr.ingest_knowledge(memory)
        except Exception as exc:  # noqa: BLE001
            log.warning("codegen_fallback.ingest_failed", error=str(exc))

    # Surface accumulated contract feedback to the caller. Always
    # collect (clears the registry's buffer); only forward if the
    # caller passed a list.
    feedback = pr.collect_feedback()
    if feedback_out is not None:
        feedback_out.extend(feedback)

    distinct = sorted({name for name in region_winners.values()})
    log.info(
        "codegen_fallback.done",
        emitted=len(out),
        providers=distinct,
        regions=len(region_winners),
        knowledge_ingested=ingested,
        feedback_items=len(feedback),
    )
    return out


def _annotate_codegen_backend_by_region_id(
    module: ModuleOp,
    rid_winners: dict[str, str],
) -> None:
    """Rewrite ``compgen.codegen_backend`` per ``compgen.region_id``.

    Walks every op carrying ``compgen.region_id`` and, if a provider
    won that region, replaces the codegen-stage's ``"fallback"``
    sentinel with the provider's name. Indexing by region_id (REQ-026)
    is more robust than the prior position-based walk: the codegen
    stage tags every op that has results (including ``tensor.empty``
    init buffers), so a positional walk drifts as soon as the spec
    list filters more selectively than the codegen tagger.

    Multi-provider-correct: when Provider A wins ``matmul_0`` and
    Provider B wins ``matmul_1``, both annotations reflect their
    actual emitters, in any module-walk order.
    """
    from xdsl.dialects.builtin import StringAttr

    for op in module.walk():
        rid_attr = op.attributes.get("compgen.region_id")
        if not isinstance(rid_attr, StringAttr):
            continue
        rid = rid_attr.data
        if rid not in rid_winners:
            continue
        backend_attr = op.attributes.get(CODEGEN_BACKEND_ATTR)
        if isinstance(backend_attr, StringAttr) and backend_attr.data == "fallback":
            op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr(rid_winners[rid])


__all__ = ["run_provider_fallback"]
