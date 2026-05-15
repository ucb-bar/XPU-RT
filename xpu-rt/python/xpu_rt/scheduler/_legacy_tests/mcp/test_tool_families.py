"""H4 — tool families with shared base schemas.

Coverage:

1. ``TOOL_FAMILIES`` is the closed enum (exactly 6 families).
2. Every family has a BaseInput + BaseOutput entry in ``FAMILY_BASES``.
3. The shared ``BaseInput`` / ``BaseOutput`` round-trip through
   ``from_dict`` / ``to_dict``.
4. Family-specialised inputs / outputs round-trip too.
5. ``family_for_tool`` returns None for unmapped tools.
6. ``family_for_tool`` returns the right family for known tools.
7. ``is_known_family`` rejects unknowns.
8. Every entry in ``DEFAULT_TOOL_FAMILY`` references a known family.
"""

from __future__ import annotations

from compgen.mcp.tool_families import (
    BaseInput,
    BaseOutput,
    BenchInput,
    BenchOutput,
    DEFAULT_TOOL_FAMILY,
    DecisionInput,
    DecisionOutput,
    DossierReadInput,
    DossierReadOutput,
    FAMILY_BASES,
    FAMILY_BENCH,
    FAMILY_DECISION,
    FAMILY_DOSSIER_READ,
    FAMILY_RECIPE_EDIT,
    FAMILY_VERIFICATION,
    KernelRequestInput,
    KernelRequestOutput,
    RecipeEditInput,
    RecipeEditOutput,
    TOOL_FAMILIES,
    VerificationInput,
    VerificationOutput,
    family_for_tool,
    is_known_family,
)


def test_tool_families_closed_enum() -> None:
    assert len(TOOL_FAMILIES) == 6
    for f in TOOL_FAMILIES:
        assert is_known_family(f)
    assert not is_known_family("unknown_family")


def test_family_bases_complete() -> None:
    for f in TOOL_FAMILIES:
        assert f in FAMILY_BASES
        inp_cls, out_cls = FAMILY_BASES[f]
        # Every base class is a concrete dataclass type.
        assert inp_cls.__name__.endswith("Input")
        assert out_cls.__name__.endswith("Output")


def test_base_input_round_trip() -> None:
    inp = BaseInput(session_id="s1")
    assert inp.to_dict() == {"session_id": "s1"}
    assert BaseInput.from_dict({"session_id": "s2"}).session_id == "s2"


def test_base_output_round_trip() -> None:
    out = BaseOutput(ok=True, status="ok", error="")
    blob = out.to_dict()
    assert blob == {"ok": True, "status": "ok", "error": ""}
    restored = BaseOutput.from_dict(blob)
    assert restored == out


def test_dossier_read_specialisation() -> None:
    inp = DossierReadInput(session_id="s", focus_region_id="r0")
    assert inp.session_id == "s"
    assert inp.focus_region_id == "r0"
    out = DossierReadOutput(region_count=5)
    assert out.region_count == 5


def test_decision_specialisation() -> None:
    inp = DecisionInput(decision_key="dispatch:cpu")
    assert inp.decision_key == "dispatch:cpu"
    out = DecisionOutput(decision_key="dispatch:cpu", resolution="sync")
    assert out.resolution == "sync"


def test_recipe_edit_specialisation() -> None:
    inp = RecipeEditInput(target="reg_0", edit_kind="fuse")
    out = RecipeEditOutput(op_id="r0", applied=True)
    assert inp.target == "reg_0"
    assert out.applied is True


def test_kernel_request_specialisation() -> None:
    inp = KernelRequestInput(op_signature="matmul:f16:64x64x64")
    out = KernelRequestOutput(request_id="req_0", kernel_id="k_0")
    assert inp.op_signature.startswith("matmul")
    assert out.kernel_id == "k_0"


def test_bench_specialisation() -> None:
    inp = BenchInput(bench_id="b0")
    out = BenchOutput(bench_id="b0", latency_us=12.5)
    assert inp.bench_id == "b0"
    assert out.latency_us == 12.5


def test_verification_specialisation() -> None:
    inp = VerificationInput(verifier="z3")
    out = VerificationOutput(verdict="pass", counterexample=None)
    assert inp.verifier == "z3"
    assert out.verdict == "pass"


# ----------------------------------------------------------------------
# Default family mapping
# ----------------------------------------------------------------------


def test_family_for_tool_known() -> None:
    assert family_for_tool("apply_recipe") == FAMILY_RECIPE_EDIT
    assert family_for_tool("propose_decision") == FAMILY_DECISION
    assert family_for_tool("verify_proposal") == FAMILY_VERIFICATION
    assert family_for_tool("get_dossier") == FAMILY_DOSSIER_READ
    assert family_for_tool("register_bench_result") == FAMILY_BENCH


def test_family_for_tool_unknown() -> None:
    assert family_for_tool("not_a_tool_at_all") is None


def test_default_mapping_only_references_known_families() -> None:
    for tool_name, family in DEFAULT_TOOL_FAMILY.items():
        assert family in TOOL_FAMILIES, f"{tool_name!r} -> {family!r}"


def test_default_mapping_nonempty_per_family() -> None:
    """Every family has at least one canonical tool in the default mapping."""

    seen = set(DEFAULT_TOOL_FAMILY.values())
    for f in TOOL_FAMILIES:
        assert f in seen, f"family {f!r} has no canonical tool"
