"""Tests for the Claude-Code-default + autocomp-escalation provider router.

Covers:

  * ``v3_to_v1_contract`` bridges a KernelContractV3 down to the v1 surface
    while preserving the full ``kernel_facing()`` view inside constraints
  * ``ClaudeCodeKernelProvider`` returns a ProviderResult driven by a
    pluggable ``CodegenCallable``
  * ``EscalatingProviderRouter``:
      - routes to the first provider on success
      - escalates on ``found=False``
      - escalates on gate failure (correctness)
      - escalates on perf miss (> perf_target_us × slack)
      - records the full escalation_path for observability
"""

from __future__ import annotations

from compgen.kernels.contract_v3 import Granularity, KernelArchetype
from compgen.kernels.contract_v3_references import reference_matmul_contract
from compgen.kernels.provider import (
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from compgen.kernels.providers.claude_code_default import (
    ClaudeCodeKernelProvider,
    StubCodegen,
)
from compgen.kernels.providers.contract_bridge import v3_to_v1_contract
from compgen.kernels.providers.escalating_router import (
    EscalatingProviderRouter,
    EscalationReason,
)

# ---------------------------------------------------------------------------
# Bridge: v3 → v1 keeps the v1 fields and attaches the rich kernel_facing
# ---------------------------------------------------------------------------


def test_bridge_preserves_v1_fields_and_attaches_kernel_facing_view() -> None:
    v3 = reference_matmul_contract()
    v1 = v3_to_v1_contract(v3, region_id="r_42")

    assert v1.op_family == "linalg.matmul"
    assert v1.region_id == "r_42"
    assert len(v1.input_shapes) == 2  # matmul has 2 inputs
    assert len(v1.output_shapes) == 1
    # dtype set is the UNION across all IO (bf16/f16/f32 + bf16 output)
    assert "bf16" in v1.dtypes
    assert "f32" in v1.dtypes
    # kernel_facing_view is attached so v3-aware providers see the full spec
    facing = v1.constraints["kernel_facing_view"]
    assert facing.archetype is KernelArchetype.COMPUTE_TILED
    assert facing.granularity is Granularity.NORMAL
    assert v1.constraints["archetype"] == "compute_tiled"
    assert v1.constraints["granularity"] == "normal"


# ---------------------------------------------------------------------------
# ClaudeCodeKernelProvider — drives a CodegenCallable
# ---------------------------------------------------------------------------


def test_claude_code_provider_returns_kernel_from_codegen() -> None:
    stub = StubCodegen(canned={"linalg.matmul": "@triton.jit\ndef matmul(...): ...\n"})
    provider = ClaudeCodeKernelProvider(codegen=stub)
    contract = v3_to_v1_contract(reference_matmul_contract())

    result = provider.search(contract, SearchBudget())

    assert result.found
    assert "@triton.jit" in result.kernel_code
    assert result.language == "triton"
    assert result.iterations_used == 1
    assert any(isinstance(e, KnowledgeExport) for e in result.knowledge_exports)


def test_claude_code_provider_escalates_on_empty_codegen() -> None:
    """Empty codegen output → found=False so the router escalates."""
    provider = ClaudeCodeKernelProvider(codegen=StubCodegen(canned={"__default__": ""}))
    contract = v3_to_v1_contract(reference_matmul_contract())

    result = provider.search(contract, SearchBudget())

    assert not result.found
    assert "empty source" in result.metadata.get("error", "")


def test_claude_code_provider_catches_codegen_exceptions() -> None:
    class _Boom(StubCodegen):
        def __call__(self, *_a, **_kw):
            raise RuntimeError("transient API hiccup")

    provider = ClaudeCodeKernelProvider(codegen=_Boom())
    result = provider.search(v3_to_v1_contract(reference_matmul_contract()), SearchBudget())

    assert not result.found
    assert "RuntimeError" in result.metadata.get("error", "")


# ---------------------------------------------------------------------------
# Router — escalation paths
# ---------------------------------------------------------------------------


def _claude_code_with(canned_code: str) -> ClaudeCodeKernelProvider:
    return ClaudeCodeKernelProvider(
        codegen=StubCodegen(canned={"__default__": canned_code}),
        name_str="claude_code",
    )


class _ConstantProvider:
    """Mimics autocomp's tier — returns a fixed ProviderResult."""

    def __init__(self, name: str, result: ProviderResult) -> None:
        self.name_str = name
        self.result = result

    @property
    def name(self) -> str:
        return self.name_str

    def accepts_contract(self, _contract):
        return True

    def search(self, _contract, _budget):
        return self.result

    def export_knowledge(self):
        return []


def test_router_picks_first_provider_on_success() -> None:
    router = EscalatingProviderRouter(
        providers=[
            _claude_code_with("@triton.jit\ndef k(...): ..."),
            _ConstantProvider("autocomp", ProviderResult(found=True, kernel_code="// autocomp\n")),
        ],
    )
    outcome = router.route(
        v3_to_v1_contract(reference_matmul_contract()),
        SearchBudget(),
    )
    # We named the first via name_str="claude_code" — chosen_provider matches.
    assert outcome.chosen_provider == "claude_code"
    assert outcome.escalation_path[0] == "claude_code"
    assert outcome.final_reason is EscalationReason.NONE
    assert "@triton.jit" in outcome.result.kernel_code


def test_router_escalates_on_not_found() -> None:
    router = EscalatingProviderRouter(
        providers=[
            _claude_code_with(""),  # empty → found=False
            _ConstantProvider("autocomp", ProviderResult(found=True, kernel_code="// autocomp\n")),
        ],
    )
    outcome = router.route(
        v3_to_v1_contract(reference_matmul_contract()),
        SearchBudget(),
    )
    assert outcome.chosen_provider == "autocomp"
    assert outcome.escalation_path == ("claude_code", "autocomp")
    assert outcome.final_reason is EscalationReason.NONE
    assert "autocomp" in outcome.result.kernel_code


def test_router_escalates_on_gate_failure() -> None:
    """Gate says the kernel is wrong → router falls through to next tier."""

    def gate(_c, result):
        # Reject anything that looks like the default stub
        return ("autocomp" in result.kernel_code), "wrong shape"

    router = EscalatingProviderRouter(
        providers=[
            _claude_code_with("@triton.jit\ndef bad(...): ..."),  # passes found, fails gate
            _ConstantProvider("autocomp", ProviderResult(found=True, kernel_code="// autocomp\n")),
        ],
        gate=gate,
    )
    outcome = router.route(
        v3_to_v1_contract(reference_matmul_contract()),
        SearchBudget(),
    )
    assert outcome.chosen_provider == "autocomp"
    assert outcome.escalation_path == ("claude_code", "autocomp")


def test_router_escalates_on_perf_miss() -> None:
    """Result is correct but too slow → escalate."""
    slow = ProviderResult(found=True, kernel_code="// slow but right\n", latency_us=500.0)
    fast = ProviderResult(found=True, kernel_code="// autocomp tuned\n", latency_us=80.0)

    router = EscalatingProviderRouter(
        providers=[
            _ConstantProvider("claude_code", slow),
            _ConstantProvider("autocomp", fast),
        ],
        perf_slack_factor=2.0,
    )
    outcome = router.route(
        v3_to_v1_contract(reference_matmul_contract()),
        SearchBudget(),
        perf_target_us=100.0,  # 500us > 100us × 2.0 → escalate; 80us ≤ 200us → accept
    )
    assert outcome.chosen_provider == "autocomp"
    assert outcome.final_reason is EscalationReason.NONE


def test_router_returns_last_result_when_all_tiers_escalate() -> None:
    """Every provider misses; outcome carries the last attempt + reason."""

    def gate_always_fail(_c, _r):
        return False, "always fail"

    router = EscalatingProviderRouter(
        providers=[
            _ConstantProvider("claude_code", ProviderResult(found=True, kernel_code="// 1\n")),
            _ConstantProvider("autocomp", ProviderResult(found=True, kernel_code="// 2\n")),
        ],
        gate=gate_always_fail,
    )
    outcome = router.route(v3_to_v1_contract(reference_matmul_contract()), SearchBudget())

    assert outcome.escalation_path == ("claude_code", "autocomp")
    assert outcome.final_reason is EscalationReason.CORRECTNESS
    assert outcome.chosen_provider == "autocomp"  # the last one we tried


def test_router_treats_provider_exception_as_escalate() -> None:
    class _Boom:
        name_str = "boom"

        @property
        def name(self):
            return self.name_str

        def accepts_contract(self, _c):
            return True

        def search(self, _c, _b):
            raise ValueError("provider crashed")

        def export_knowledge(self):
            return []

    router = EscalatingProviderRouter(
        providers=[
            _Boom(),
            _ConstantProvider("autocomp", ProviderResult(found=True, kernel_code="// rescued\n")),
        ],
    )
    outcome = router.route(v3_to_v1_contract(reference_matmul_contract()), SearchBudget())
    assert outcome.escalation_path == ("boom", "autocomp")
    assert outcome.chosen_provider == "autocomp"
    assert outcome.final_reason is EscalationReason.NONE


def test_router_skips_providers_that_decline_the_contract() -> None:
    class _DeclinesAll:
        name_str = "declines"

        @property
        def name(self):
            return self.name_str

        def accepts_contract(self, _c):
            return False

        def search(self, *_):
            raise AssertionError("should not be called")

        def export_knowledge(self):
            return []

    router = EscalatingProviderRouter(
        providers=[
            _DeclinesAll(),
            _ConstantProvider("autocomp", ProviderResult(found=True, kernel_code="// only one\n")),
        ],
    )
    outcome = router.route(v3_to_v1_contract(reference_matmul_contract()), SearchBudget())
    # The decliner is filtered out before search; not in path.
    assert outcome.escalation_path == ("autocomp",)
    assert outcome.chosen_provider == "autocomp"
