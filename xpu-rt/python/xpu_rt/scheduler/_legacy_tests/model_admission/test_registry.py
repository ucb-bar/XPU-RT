"""Registry validation: every YAML config parses, cross-refs resolve, and the
top-level ``model_registry.yaml`` matches the on-disk model configs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.model_admission.registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    DEFAULT_SLICES_DIR,
    DEFAULT_SUITES_DIR,
    RegistryError,
    load_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def registry():
    return load_registry(
        registry_path=REPO_ROOT / DEFAULT_REGISTRY_PATH,
        models_dir=REPO_ROOT / DEFAULT_MODELS_DIR,
        slices_dir=REPO_ROOT / DEFAULT_SLICES_DIR,
        suites_dir=REPO_ROOT / DEFAULT_SUITES_DIR,
    )


def test_registry_loads(registry):
    # Sanity: matches the registry file declared.
    assert len(registry.entries) == 31
    # 31 real-model configs + 5 proxy configs.
    assert len(registry.models) == 36


def test_every_registry_entry_has_a_model_config(registry):
    missing = [mid for mid in registry.entries if mid not in registry.models]
    assert missing == []


def test_every_slice_resolves_to_a_model(registry):
    for slice_id, sl in registry.slices.items():
        assert sl.parent_model_id in registry.models, (
            f"slice {slice_id!r} references unknown parent_model_id={sl.parent_model_id!r}"
        )


def test_every_suite_entry_resolves(registry):
    for suite_name, suite in registry.suites.items():
        for entry in suite.all_entries():
            assert entry.model_id in registry.models, f"{suite_name}: {entry.model_id} missing"
            if entry.slice_id:
                assert entry.slice_id in registry.slices, f"{suite_name}: {entry.slice_id} missing"


def test_required_proxies_have_proxy_loader(registry):
    suite = registry.suites["always_test_models"]
    for entry in suite.required_proxy:
        cfg = registry.models[entry.model_id]
        assert cfg.loader.kind == "proxy", f"{entry.model_id}: loader.kind={cfg.loader.kind} (expected 'proxy')"
        assert cfg.loader.proxy_module.startswith("compgen.model_admission.proxies."), cfg.loader.proxy_module


def test_huge_models_are_slice_only(registry):
    huge = ["qwen3_vl_235b_a22b_instruct", "llama4_scout_text", "deepseek_v4_flash_text"]
    for mid in huge:
        cfg = registry.models[mid]
        assert cfg.loader.device_policy == "unavailable_for_full_local", mid
        # Registry entries: blocking flag must follow the user spec.
        if mid == "qwen3_vl_235b_a22b_instruct":
            assert registry.entries[mid].blocking is False
        if mid in ("llama4_scout_text", "deepseek_v4_flash_text"):
            assert registry.entries[mid].blocking is True


def test_verified_models_have_pinned_revision(registry):
    """Invariant: source_verified=true ALWAYS carries a 40-char revision SHA.

    Verification is only created by ``verify-sources``, which always pins a
    revision. A YAML with source_verified=true and an empty/short revision
    means someone hand-edited it and bypassed the verifier.
    """

    for mid, cfg in registry.models.items():
        if cfg.family == "proxy":
            continue
        if cfg.source.source_verified:
            rev = cfg.source.revision or ""
            assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), (
                f"{mid}: source_verified=true but revision is not a 40-char SHA "
                f"(got {rev!r}); did someone hand-edit configs/models/{mid}.yaml?"
            )
            assert cfg.source.model_ref and cfg.source.model_ref != "TO_BE_VERIFIED_ONLINE", mid
            assert cfg.source.repo_url.startswith("https://huggingface.co/"), mid


def test_unverified_real_models_keep_placeholder(registry):
    """Real models without verification keep model_ref=TO_BE_VERIFIED_ONLINE."""

    for mid, cfg in registry.models.items():
        if cfg.family == "proxy":
            continue
        if not cfg.source.source_verified:
            assert cfg.source.model_ref in ("TO_BE_VERIFIED_ONLINE", ""), (
                f"{mid}: source_verified=false but model_ref={cfg.source.model_ref!r}; "
                "either flip source_verified=true via verify-sources or restore the placeholder."
            )


def test_duplicate_model_id_rejected(tmp_path: Path):
    models = tmp_path / "models"
    slices = tmp_path / "slices"
    suites = tmp_path / "suites"
    models.mkdir()
    slices.mkdir()
    suites.mkdir()
    common = (
        "schema_version: model_config_v1\n"
        "model_id: dup\n"
        "family: proxy\n"
        "loader:\n  kind: proxy\n  proxy_module: x.y\n"
    )
    (models / "a.yaml").write_text(common)
    (models / "b.yaml").write_text(common)
    (suites / "model_registry.yaml").write_text(
        "schema_version: model_registry_v1\nmodels: []\n"
    )
    with pytest.raises(RegistryError):
        load_registry(
            registry_path=suites / "model_registry.yaml",
            models_dir=models,
            slices_dir=slices,
            suites_dir=suites,
        )
