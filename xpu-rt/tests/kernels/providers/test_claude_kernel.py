"""ClaudeKernelProvider — retry loop + structural gate + knowledge export."""

from __future__ import annotations

from compgen.kernels.provider import KernelContract, SearchBudget
from compgen.kernels.providers.claude_kernel import ClaudeKernelProvider, PromptPack


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def chat(self, prompt: str, num_samples: int = 1) -> list[str]:
        self.calls.append(prompt)
        if not self._responses:
            return ["(exhausted)"]
        return [self._responses.pop(0)]


_GOOD_MLIR = """```mlir
module {
  func.func @matmul(%a: tensor<32x64xf16>) -> tensor<32x64xf16> { return %a : tensor<32x64xf16> }
}
```"""


def _contract() -> KernelContract:
    return KernelContract(
        region_id="r0",
        op_family="matmul",
        input_shapes=((32, 64), (64, 32)),
        dtypes=("f16", "f16"),
        layout="row_major",
        target_name="nvidia-h100",
    )


def _pack() -> PromptPack:
    return PromptPack(
        name="fake_dialect",
        system="emit MLIR",
        op_templates={"matmul": "matmul {shapes} {dtypes} {hints}"},
        code_fence_language="mlir",
        max_iterations=3,
    )


def test_accepts_contract_requires_op_family() -> None:
    prov = ClaudeKernelProvider(name="fake", prompt_pack=_pack(), target_name="nvidia-h100", llm_client=_FakeLLM([]))
    assert prov.accepts_contract(_contract())
    empty = KernelContract(target_name="nvidia-h100")
    assert not prov.accepts_contract(empty)


def test_search_succeeds_first_try() -> None:
    llm = _FakeLLM([_GOOD_MLIR])
    prov = ClaudeKernelProvider(name="fake", prompt_pack=_pack(), target_name="nvidia-h100", llm_client=llm)
    result = prov.search(_contract(), SearchBudget(max_iterations=3))
    assert result.found
    assert "func.func @matmul" in result.kernel_code
    assert result.language == "mlir"
    assert result.iterations_used == 1
    assert len(result.knowledge_exports) == 1
    assert result.knowledge_exports[0].scope_key == "fake_dialect/matmul"


def test_search_retries_when_gate_rejects() -> None:
    bad = "```mlir\nnonsense\n```"
    good = _GOOD_MLIR
    llm = _FakeLLM([bad, good])

    def gate(candidate: str, _contract):
        if "func.func" in candidate:
            return True, ""
        return False, "need a func.func"

    prov = ClaudeKernelProvider(
        name="fake",
        prompt_pack=_pack(),
        target_name="nvidia-h100",
        llm_client=llm,
        structural_gate=gate,
    )
    result = prov.search(_contract(), SearchBudget(max_iterations=3))
    assert result.found
    assert result.iterations_used == 2
    # The second prompt must reference the gate diagnostic.
    assert "need a func.func" in llm.calls[1]


def test_search_gives_up_when_exceeding_budget() -> None:
    # All responses fail the gate.
    bad_responses = ["```mlir\nbad1\n```", "```mlir\nbad2\n```"]
    llm = _FakeLLM(bad_responses)

    def gate(_candidate: str, _contract):
        return False, "always reject"

    prov = ClaudeKernelProvider(
        name="fake",
        prompt_pack=_pack(),
        target_name="nvidia-h100",
        llm_client=llm,
        structural_gate=gate,
    )
    result = prov.search(_contract(), SearchBudget(max_iterations=2))
    assert not result.found
    assert result.metadata.get("reason") == "no_accepted_candidate"
    assert "always reject" in result.metadata.get("last_diagnostic", "")


def test_export_knowledge_drains_buffer() -> None:
    llm = _FakeLLM([_GOOD_MLIR])
    prov = ClaudeKernelProvider(name="fake", prompt_pack=_pack(), target_name="nvidia-h100", llm_client=llm)
    prov.search(_contract(), SearchBudget(max_iterations=2))
    knowledge = prov.export_knowledge()
    assert len(knowledge) == 1
    assert not prov.export_knowledge(), "second call must drain the buffer"


def test_forbidden_substrings_reject_candidate() -> None:
    pack = _pack()
    pack.forbidden_substrings = ("BANNED",)
    good = _GOOD_MLIR
    banned = "```mlir\nBANNED content\n```"
    llm = _FakeLLM([banned, good])

    prov = ClaudeKernelProvider(name="fake", prompt_pack=pack, target_name="nvidia-h100", llm_client=llm)
    result = prov.search(_contract(), SearchBudget(max_iterations=2))
    assert result.found
    assert result.iterations_used == 2
