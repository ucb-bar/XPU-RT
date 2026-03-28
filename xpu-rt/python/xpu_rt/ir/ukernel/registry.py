"""Ukernel registry and selection engine.

Holds decl+match+body triples and selects the best ukernel for a given
operation context. Completely target-agnostic — RISC-V, CUDA, NPU, and
any future target all register into the same registry with different
match constraints.

Selection algorithm:
    1. Filter by ``op_family``
    2. For each candidate, evaluate ALL constraints against context
    3. Return the highest-priority passing match
    4. Return None if no match
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from compgen.ir.ukernel.constraints import ConstraintContext, evaluate_all_constraints
from compgen.ir.ukernel.ops import UkernelBodyOp, UkernelDeclOp, UkernelMatchOp

log = structlog.get_logger()


@dataclass
class RegisteredUkernel:
    """A complete ukernel: declaration + matches + bodies."""

    decl: UkernelDeclOp
    matches: list[UkernelMatchOp] = field(default_factory=list)
    bodies: list[UkernelBodyOp] = field(default_factory=list)


class UkernelRegistry:
    """Registry of ukernel decl+match+body triples with selection.

    Target-agnostic: any target registers ukernels with appropriate
    match constraints. Selection evaluates constraints against the
    current context to find the best match.
    """

    def __init__(self) -> None:
        self._ukernels: dict[str, RegisteredUkernel] = {}
        self._by_op_family: dict[str, list[str]] = {}

    def register_ukernel(
        self,
        decl: UkernelDeclOp,
        matches: list[UkernelMatchOp] | None = None,
        bodies: list[UkernelBodyOp] | None = None,
    ) -> None:
        """Register a complete ukernel (decl + matches + bodies).

        Args:
            decl: The kernel declaration.
            matches: Match constraints (one per target context variant).
            bodies: Implementation bodies (one per target/body_kind).
        """
        entry = RegisteredUkernel(
            decl=decl,
            matches=list(matches or []),
            bodies=list(bodies or []),
        )
        self._ukernels[decl.kernel_name] = entry

        # Index by op_family for fast lookup
        for match in entry.matches:
            if match.op_family:
                self._by_op_family.setdefault(match.op_family, []).append(decl.kernel_name)

        log.debug(
            "ukernel.registry.register",
            kernel=decl.kernel_name,
            matches=len(entry.matches),
            bodies=len(entry.bodies),
            transparency=decl.transparency,
        )

    def register_decl(self, decl: UkernelDeclOp) -> None:
        """Register or update a declaration."""
        if decl.kernel_name in self._ukernels:
            self._ukernels[decl.kernel_name].decl = decl
        else:
            self._ukernels[decl.kernel_name] = RegisteredUkernel(decl=decl)

    def register_match(self, match: UkernelMatchOp) -> None:
        """Register a match constraint for an existing declaration."""
        entry = self._ukernels.get(match.kernel_name)
        if entry is None:
            log.warning("ukernel.registry.orphan_match", kernel=match.kernel_name)
            return
        entry.matches.append(match)
        if match.op_family:
            self._by_op_family.setdefault(match.op_family, []).append(match.kernel_name)

    def register_body(self, body: UkernelBodyOp) -> None:
        """Register a body for an existing declaration."""
        entry = self._ukernels.get(body.kernel_name)
        if entry is None:
            log.warning("ukernel.registry.orphan_body", kernel=body.kernel_name)
            return
        entry.bodies.append(body)

    def select_ukernel(
        self,
        op_family: str,
        context: ConstraintContext,
    ) -> UkernelDeclOp | None:
        """Select the best ukernel for an op family and context.

        Evaluates match constraints against the context and returns
        the highest-priority passing match. Returns None if no match.

        Args:
            op_family: The operation pattern to match (e.g., "matmul").
            context: Current target/shape/dtype/layout context.

        Returns:
            Best matching UkernelDeclOp, or None.
        """
        # Get candidate kernel names for this op family
        candidate_names = set(self._by_op_family.get(op_family, []))

        best_decl: UkernelDeclOp | None = None
        best_priority = -1

        for name in candidate_names:
            entry = self._ukernels.get(name)
            if entry is None:
                continue

            # Check all match ops for this kernel
            for match in entry.matches:
                if match.op_family != op_family:
                    continue

                # Evaluate all constraint categories
                all_constraints = (
                    list(match.dtype_constraints)
                    + list(match.shape_constraints)
                    + list(match.target_constraints)
                    + list(match.layout_constraints)
                )

                if evaluate_all_constraints(all_constraints, context):
                    if match.priority > best_priority:
                        best_priority = match.priority
                        best_decl = entry.decl

        if best_decl:
            log.debug(
                "ukernel.registry.selected",
                kernel=best_decl.kernel_name,
                op_family=op_family,
                priority=best_priority,
            )
        return best_decl

    def select_body(
        self,
        kernel_name: str,
        target_family: str = "any",
    ) -> UkernelBodyOp | None:
        """Select the best body for a kernel and target family.

        Prefers target-specific bodies over generic ("any") bodies.

        Args:
            kernel_name: The kernel to find a body for.
            target_family: Target family to match (or "any" for generic).

        Returns:
            Best matching UkernelBodyOp, or None.
        """
        entry = self._ukernels.get(kernel_name)
        if entry is None:
            return None

        # Prefer exact target match, fall back to "any"
        specific = None
        generic = None
        for body in entry.bodies:
            if body.target_family == target_family:
                specific = body
            elif body.target_family == "any":
                generic = body

        return specific or generic

    def all_decls(self) -> list[UkernelDeclOp]:
        """Return all registered declarations."""
        return [e.decl for e in self._ukernels.values()]

    def matches_for(self, kernel_name: str) -> list[UkernelMatchOp]:
        """Return all match ops for a kernel."""
        entry = self._ukernels.get(kernel_name)
        return list(entry.matches) if entry else []

    def bodies_for(self, kernel_name: str) -> list[UkernelBodyOp]:
        """Return all bodies for a kernel."""
        entry = self._ukernels.get(kernel_name)
        return list(entry.bodies) if entry else []

    def __len__(self) -> int:
        return len(self._ukernels)


__all__ = ["RegisteredUkernel", "UkernelRegistry"]
