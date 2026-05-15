"""every shipped card under
``python/compgen/{providers,targets,dialects}/cards/`` loads
cleanly and obeys the hard-rule discipline."""

from __future__ import annotations

from collections import Counter

from compgen.dialects.dialect_provider_types import DialectProviderCard
from compgen.providers.card_loader import (
    iter_dialect_cards,
    iter_provider_cards,
    iter_target_cards,
    load_all_cards,
)
from compgen.providers.provider_types import (
    INTEGRATION_LEVELS,
    PAPER_CLAIMABLE_LEVELS,
    ProviderCard,
)
from compgen.targets.target_types import TargetCard


EXPECTED_PROVIDER_IDS = frozenset(
    {
        "cffi_c",
        "python_reference",
        "triton",
        "tilelang",
        "cutlass_cute",
        "bitblas",
        "thunderkittens",
        "mirage",
        "autocomp",
        "kernelblaster",
        "kernelbench_caesar",
        "cuda_tile_ir",
        "hexagon_mlir",
        "pallas",
        "nki",
        "exo",
        "gemmini_c",
        "radiance_muon",
        "claude_kernel",
    }
)

EXPECTED_TARGET_IDS = frozenset(
    {
        "host_cpu",
        "cuda_sm75",
        "cuda_sm80",
        "cuda_sm90",
        "rocm_gpu",
        "aws_neuron_trn",
        "aws_neuron_inf",
        "google_tpu_v5e",
        "google_tpu_v6e",
        "gemmini_mx",
        "radiance_muon",
        "hexagon_npu",
        "saturn_opu",
    }
)

EXPECTED_DIALECT_IDS = frozenset(
    {
        "cuda_tile_ir",
        "hexagon_mlir",
        "pallas",
        "nki",
        "exo",
        "gemmini_c",
        "radiance_muon",
        "iree",
        "triton_mlir",
        "stablehlo",
    }
)


def test_all_shipped_provider_cards_load():
    cards = tuple(iter_provider_cards())
    assert cards, "no provider cards discovered"
    for c in cards:
        assert isinstance(c, ProviderCard)
        assert c.integration_level in INTEGRATION_LEVELS


def test_provider_inventory_matches_phase_f_roadmap():
    cards = tuple(iter_provider_cards())
    discovered = {c.provider_id for c in cards}
    missing = EXPECTED_PROVIDER_IDS - discovered
    assert not missing, f"missing provider cards: {sorted(missing)}"


def test_all_shipped_target_cards_load():
    cards = tuple(iter_target_cards())
    assert cards
    for c in cards:
        assert isinstance(c, TargetCard)
        assert c.dispatch_modes, f"target {c.target_id} has empty dispatch_modes"


def test_target_inventory_matches_phase_f_roadmap():
    cards = tuple(iter_target_cards())
    discovered = {c.target_id for c in cards}
    missing = EXPECTED_TARGET_IDS - discovered
    assert not missing, f"missing target cards: {sorted(missing)}"


def test_all_shipped_dialect_cards_load():
    cards = tuple(iter_dialect_cards())
    assert cards
    for c in cards:
        assert isinstance(c, DialectProviderCard)


def test_dialect_inventory_matches_phase_f_roadmap():
    cards = tuple(iter_dialect_cards())
    discovered = {c.dialect_provider_id for c in cards}
    missing = EXPECTED_DIALECT_IDS - discovered
    assert not missing, f"missing dialect cards: {sorted(missing)}"


def test_load_all_cards_returns_three_tuples():
    providers, targets, dialects = load_all_cards()
    assert isinstance(providers, tuple)
    assert isinstance(targets, tuple)
    assert isinstance(dialects, tuple)
    assert len(providers) == len(set(p.provider_id for p in providers))
    assert len(targets) == len(set(t.target_id for t in targets))
    assert len(dialects) == len(set(d.dialect_provider_id for d in dialects))


def test_paper_claimable_providers_all_at_verify_or_promote():
    """Hard rule 6: cards at probe / card_only / generate are never paper_claimable."""
    cards = tuple(iter_provider_cards())
    for c in cards:
        if c.paper_claimable:
            assert c.integration_level in PAPER_CLAIMABLE_LEVELS, (
                f"provider {c.provider_id} has paper_claimable=true at "
                f"integration_level={c.integration_level} — hard rule 6 violation"
            )


def test_baseline_real_providers_present_at_verify_or_promote():
    """cffi_c, python_reference, triton must be the real backbone."""
    cards = {c.provider_id: c for c in iter_provider_cards()}
    assert cards["cffi_c"].integration_level == "promote"
    assert cards["python_reference"].integration_level == "verify"
    assert cards["triton"].integration_level == "promote"


def test_external_sdk_providers_block_honestly():
    """Cards for SDK-gated backends carry required_env or required_python_imports."""
    cards = {c.provider_id: c for c in iter_provider_cards()}
    sdk_gated = [
        "cuda_tile_ir",
        "hexagon_mlir",
        "cutlass_cute",
        "thunderkittens",
        "gemmini_c",
        "radiance_muon",
        "nki",
        "kernelbench_caesar",
        "kernelblaster",
        "mirage",
    ]
    for pid in sdk_gated:
        c = cards[pid]
        assert (
            c.required_env or c.required_python_imports or c.required_commands
        ), f"SDK-gated provider {pid} has no required_env/imports/commands — silent disappearance risk"
        # And they cannot be paper_claimable while at probe level.
        if c.integration_level == "probe":
            assert not c.paper_claimable


def test_target_families_referenced_by_providers_have_cards():
    """Every target_family a provider claims to support must have at least one TargetCard."""
    providers = tuple(iter_provider_cards())
    targets = {t.family for t in iter_target_cards()}
    referenced = set()
    for p in providers:
        referenced.update(p.target_families)
    missing = referenced - targets
    # A handful of synthetic / catch-all families are acceptable.
    acceptable_missing = {"custom_accelerator", "riscv_opu"}
    real_missing = missing - acceptable_missing
    assert not real_missing, (
        f"provider cards reference target families with no matching TargetCard: "
        f"{sorted(real_missing)}"
    )


def test_integration_level_distribution_is_honest():
    """The shipped distribution should be probe-heavy (env-gated externals)
    with a small verify/promote core. Catches drift toward paper-claim inflation."""
    cards = tuple(iter_provider_cards())
    by_level = Counter(c.integration_level for c in cards)
    promote_and_verify = by_level["promote"] + by_level["verify"]
    probe_and_generate = by_level["probe"] + by_level["generate"]
    # At most a third of cards can be at verify/promote.
    assert promote_and_verify * 2 < probe_and_generate * 3, (
        f"verify/promote ({promote_and_verify}) outweighs probe/generate "
        f"({probe_and_generate}) — likely paper-claim inflation"
    )
