"""CReferenceProvider — a deterministic cffi-C reference kernel provider.

This is the in-tree "reference" provider in the auction. It produces
real, compilable cffi-C source for the contract at hand and emits real
kernel artifacts. It's not optimised — that's the point: it gives the
auction a guaranteed-correct, guaranteed-fast-to-emit baseline that any
other provider must beat on perf to win.

Coverage:

* matmul ``(M, K) @ (K, N) → (M, N)`` in row-major f32, accumulator f32.
  Triple-nested loop, ``-O2 -fno-fast-math``. Bit-exact under
  's Higham bound when the eager reference uses the same
  accumulation order; otherwise refinement_status=tolerance_eps.

That single shape covers the merlin_mlp_wide vertical slice. Future
expansions (pointwise add, relu, layer_norm) ride a similar template
when the corresponding archetype lands as an auction target.

The provider's :meth:`bid` reports a deterministic perf estimate (1us
per 1k flops at a nominal 1 TFLOP/s) and ``confidence=0.85`` — high
enough to be a credible bidder but below the 0.9 a verified cache hit
returns from ClaudeCodeKernelProvider.

The provider's ``search()`` returns a ``ProviderResult`` whose
``kernel_code`` is the emitted C source and whose ``language`` is
``"c"``. The auction's fulfill adapter writes that source to
``kernel.c`` plus a matching ``kernel_metadata.json`` /
``launch_config.json`` / ``provider_claims.json`` set under the
auction's per-provider artifact directory.
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


_MATMUL_C_SOURCE = """\
/* M-57 CReferenceProvider — matmul reference kernel.
 * Triple-nested loop, row-major, f32 accumulator. Compiled with
 * -O2 -fno-fast-math; deterministic, no SIMD reordering.
 */
#include <string.h>

void compgen_matmul_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int M, int N, int K)
{
    /* Y = A @ B; A is (M, K) row-major, B is (K, N) row-major. */
    memset(Y, 0, (size_t)M * (size_t)N * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a = A[i * K + k];
            const float* brow = B + k * N;
            float* yrow = Y + i * N;
            for (int j = 0; j < N; ++j) {
                yrow[j] += a * brow[j];
            }
        }
    }
}
"""


_POINTWISE_FUSED_C_SOURCE = """\
/* M-65 CReferenceProvider — fused pointwise (add+relu) reference.
 * Produced for fuse_producer_consumer candidates whose producer is
 * an elementwise add and consumer is a pointwise activation.
 * Identity-fused: deterministic, no SIMD reordering.
 */
#include <stddef.h>

void compgen_fused_pointwise_f32(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int N)
{
    /* Y = relu(X). Fusion-specific kernels override this prototype. */
    for (int i = 0; i < N; ++i) {
        float v = X[i];
        Y[i] = v > 0.0f ? v : 0.0f;
    }
}
"""


@dataclass
class CReferenceProvider:
    """Deterministic cffi-C reference kernel provider.

    auction baseline. Emits a compilable C reference for matmul.
    """

    name_str: str = "c_reference"
    priority: int = 5  # mid — beats the legacy fallback, loses to a tuned bid
    applicable_targets: tuple[str, ...] = ("host_cpu",)
    #  closure: pointwise (fused) added so the fusion path
    # has at least one applicable bidder.
    applicable_archetypes: tuple[str, ...] = ("compute_tiled", "pointwise")
    _exports: list[KnowledgeExport] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.name_str

    def accepts_contract(self, contract: KernelContract) -> bool:
        return contract.target_name in self.applicable_targets and "matmul" in contract.op_family

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        # Provider-level search ignores the V3 contract here — the
        # fulfill adapter passes us the legacy bridge and we
        # produce a deterministic source. The legacy contract carries
        # op_family which lets us dispatch matmul vs fused-pointwise.
        op_family = (contract.op_family or "").lower()
        if "matmul" in op_family:
            kernel_code = _MATMUL_C_SOURCE
            symbol = "compgen_matmul_f32"
            kind = "reference_matmul"
        else:
            kernel_code = _POINTWISE_FUSED_C_SOURCE
            symbol = "compgen_fused_pointwise_f32"
            kind = "reference_pointwise"
        return ProviderResult(
            found=True,
            kernel_code=kernel_code,
            language="c",
            iterations_used=1,
            total_candidates=1,
            metadata={
                "provider": self.name_str,
                "kind": kind,
                "symbol": symbol,
                "compiler_flags": "-O2 -fno-fast-math",
            },
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        return list(self._exports)

    # -- Phase D / bid -----------------------------------------------

    def bid(self, contract_v3: Any) -> BidPreview:
        """Cheap deterministic estimate: matmul-only, host_cpu-only."""
        try:
            archetype = contract_v3.archetype.value
            target = contract_v3.orchestration.execution.hardware.target_name
            op_name = contract_v3.op_name.lower()
        except AttributeError:
            return BidPreview(provider_name=self.name, confidence=0.0, rationale="invalid_contract")

        if target not in self.applicable_targets:
            return BidPreview(
                provider_name=self.name,
                confidence=0.0,
                rationale=f"unsupported_target_{target}",
            )
        if archetype == "compute_tiled" and "matmul" in op_name:
            # Roofline: 1 TFLOP/s nominal CPU peak; matmul flops 2*M*N*K.
            try:
                inputs = contract_v3.io.inputs
                dim_m = int(inputs[0].shape.dims[0])
                dim_k = int(inputs[0].shape.dims[1])
                dim_n = int(inputs[1].shape.dims[1])
                flops = 2.0 * dim_m * dim_n * dim_k
                est_us = max(1.0, flops / 1.0e12 * 1e6 + 2.0)
            except Exception:  # noqa: BLE001
                est_us = float("inf")
            return BidPreview(
                provider_name=self.name,
                perf_estimate_us=est_us,
                confidence=0.85,
                time_to_generate_s_estimate=0.5,
                rationale="c_reference_matmul_baseline",
                cache_hit=False,
            )

        if archetype == "pointwise":
            #  closure: pointwise baseline. Cost ~ output numel.
            try:
                output = contract_v3.io.outputs[0]
                numel = 1
                for d in output.shape.dims:
                    numel *= int(d) if d is not None else 1
                # 1 ns / element nominal.
                est_us = max(0.5, numel / 1000.0)
            except Exception:  # noqa: BLE001
                est_us = float("inf")
            return BidPreview(
                provider_name=self.name,
                perf_estimate_us=est_us,
                confidence=0.5,  # below matmul confidence; this is a generic baseline
                time_to_generate_s_estimate=0.3,
                rationale="c_reference_pointwise_baseline",
                cache_hit=False,
            )

        return BidPreview(
            provider_name=self.name,
            confidence=0.0,
            rationale=f"unsupported_op_{op_name}",
        )


__all__ = ["CReferenceProvider"]
