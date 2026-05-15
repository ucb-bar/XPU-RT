"""Acceptance tests for Structured Payload Attribution Hardening (02.5).

Asserts that every ``decomposed_structured`` FX call_function node has
``payload_ops_len >= 1``, and that opaque/closed_by_registry/resolved
classifications carry the right attribution shape.

Builds on the canonical 6-model suite (same fixture pattern as
``test_payload_lowering.py``) so every change to FXImporter or to the
diagnostic message formats is caught here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

ATTRIBUTION_MODELS: tuple[str, ...] = (
    "tiny_mlp",
    "tiny_attention",
    "tiny_conv_block",
    "proxy_vlm",
    "proxy_vla",
    "custom_unsupported_op",
)


@pytest.fixture(scope="module")
def attribution_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Capture+lower every model once; ``{model_id: run_dir}``."""
    base = tmp_path_factory.mktemp("attribution_runs")
    out: dict[str, Path] = {}
    for model_id in ATTRIBUTION_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="payload-lowering",
            run_id=f"attr_{model_id}",
        )
        out[model_id] = run_dir
    return out


# --------------------------------------------------------------------------- #
# payload_attribution.json shape + integrity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", ATTRIBUTION_MODELS)
def test_attribution_artifact_emitted(
    model_id: str, attribution_runs: dict[str, Path]
) -> None:
    p = attribution_runs[model_id] / "01_payload_lowering" / "payload_attribution.json"
    assert p.exists(), f"{model_id}: missing payload_attribution.json"
    obj = json.loads(p.read_text())
    assert obj["schema_version"] == "payload_attribution_v1"
    assert "totals" in obj
    assert "modules" in obj


@pytest.mark.parametrize("model_id", ATTRIBUTION_MODELS)
def test_no_unattributed_or_count_mismatch(
    model_id: str, attribution_runs: dict[str, Path]
) -> None:
    """Every forward-body op must be claimed by exactly one FX node, and
    the diagnostic counts must match the actual op count per module."""
    p = attribution_runs[model_id] / "01_payload_lowering" / "payload_attribution.json"
    obj = json.loads(p.read_text())
    assert obj["totals"]["unattributed_ops"] == 0, obj["totals"]
    assert obj["totals"]["modules_with_count_mismatch"] == [], obj["totals"]
    for mod in obj["modules"]:
        assert mod["totals"]["count_mismatch"] is False, (model_id, mod["module_id"])


@pytest.mark.parametrize("model_id", ATTRIBUTION_MODELS)
def test_no_double_attribution(
    model_id: str, attribution_runs: dict[str, Path]
) -> None:
    """Each (payload_ref, line_index) pair must appear in at most one
    FX node's payload_ops list across the whole module."""
    p = attribution_runs[model_id] / "01_payload_lowering" / "payload_attribution.json"
    obj = json.loads(p.read_text())
    seen: dict[tuple[str, int], str] = {}
    for mod in obj["modules"]:
        for attribution in mod["fx_attributions"]:
            for op in attribution["payload_ops"]:
                key = (op["payload_ref"], op["line_index"])
                assert key not in seen, (
                    f"{model_id}: op {key} attributed to both "
                    f"{seen[key]!r} and {attribution['fx_node']!r}"
                )
                seen[key] = attribution["fx_node"]


# --------------------------------------------------------------------------- #
# fx_to_payload_accounting.json — pass-condition for 02.5
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", ATTRIBUTION_MODELS)
def test_no_decomposed_structured_with_empty_payload_ops(
    model_id: str, attribution_runs: dict[str, Path]
) -> None:
    """The 02.5 pass condition: every decomposed_structured FX call_function
    node has at least one Payload op attributed."""
    p = (
        attribution_runs[model_id]
        / "01_payload_lowering"
        / "fx_to_payload_accounting.json"
    )
    acc = json.loads(p.read_text())
    offenders: list[dict] = []
    for mod in acc["modules"]:
        for n in mod["nodes"]:
            if n["classification"] == "decomposed_structured" and not n["payload_ops"]:
                offenders.append({"module": mod["module_id"], "node": n})
    assert not offenders, offenders


@pytest.mark.parametrize("model_id", ATTRIBUTION_MODELS)
def test_opaque_fallback_points_to_func_call(
    model_id: str, attribution_runs: dict[str, Path]
) -> None:
    p = (
        attribution_runs[model_id]
        / "01_payload_lowering"
        / "fx_to_payload_accounting.json"
    )
    acc = json.loads(p.read_text())
    for mod in acc["modules"]:
        for n in mod["nodes"]:
            if n["classification"] != "opaque_fallback":
                continue
            assert n["payload_ops"], (model_id, n)
            assert n["payload_ops"][0]["op_name"] == "func.call", (model_id, n)


@pytest.mark.parametrize("model_id", ATTRIBUTION_MODELS)
def test_payload_ref_paths_resolve(
    model_id: str, attribution_runs: dict[str, Path]
) -> None:
    """Every payload_ref in payload_ops must resolve to a real file."""
    run = attribution_runs[model_id]
    p = run / "01_payload_lowering" / "fx_to_payload_accounting.json"
    acc = json.loads(p.read_text())
    for mod in acc["modules"]:
        for n in mod["nodes"]:
            for po in n["payload_ops"]:
                full = run / po["payload_ref"]
                assert full.exists(), (model_id, po)


@pytest.mark.parametrize("model_id", ATTRIBUTION_MODELS)
def test_decomposed_op_count_matches_dialect_coverage(
    model_id: str, attribution_runs: dict[str, Path]
) -> None:
    """The total ops attributed to decomposed_structured + opaque_fallback +
    closed_by_registry FX nodes must match the count of @forward ops
    surfaced by the attribution sidecar (each op is attributed to exactly
    one node, no double-counting)."""
    run = attribution_runs[model_id]
    attr = json.loads(
        (run / "01_payload_lowering" / "payload_attribution.json").read_text()
    )
    acc = json.loads(
        (run / "01_payload_lowering" / "fx_to_payload_accounting.json").read_text()
    )
    accounted_ops = sum(
        len(n["payload_ops"])
        for mod in acc["modules"]
        for n in mod["nodes"]
    )
    assert accounted_ops == attr["totals"]["attributed_ops"], (
        model_id,
        accounted_ops,
        attr["totals"]["attributed_ops"],
    )


# --------------------------------------------------------------------------- #
# Closed-by-registry attribution (custom_unsupported_op + registry)
# --------------------------------------------------------------------------- #


def test_closed_by_registry_carries_inlined_payload_ops(tmp_path: Path) -> None:
    """When an extension is inlined, the original FX node's payload_ops
    must contain the expanded extension body (not be empty)."""
    cfg = REPO_ROOT / "configs" / "models" / "custom_unsupported_op.yaml"
    target = HOST_CPU_TARGET
    registry = REPO_ROOT / "user_extensions" / "registry.yaml"
    if not registry.exists():
        pytest.skip("user_extensions/registry.yaml not populated")
    out = tmp_path / "closure_attr"
    run_graph_compilation(
        model_config_path=cfg,
        target_config_path=target,
        out_dir=out,
        stop_after="payload-lowering",
        run_id="attr_closure",
        extension_registry=registry,
    )
    acc = json.loads(
        (out / "01_payload_lowering" / "fx_to_payload_accounting.json").read_text()
    )
    closed_nodes = [
        n for mod in acc["modules"] for n in mod["nodes"]
        if n["classification"] == "closed_by_registry"
    ]
    assert closed_nodes, "no closed_by_registry nodes found — registry empty?"
    for n in closed_nodes:
        assert n["registry_closure"], n
        assert len(n["payload_ops"]) >= 1, (
            f"closed_by_registry FX node {n['fx_node']!r} has empty payload_ops"
        )
