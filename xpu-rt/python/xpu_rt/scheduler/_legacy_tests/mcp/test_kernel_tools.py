"""Tests for ``compgen.mcp.tools.kernel``.

Covers the in-session kernel-codegen flow end-to-end:

  1. ``request_kernel_codegen`` queues a request with a rendered prompt
  2. ``list_pending_kernel_requests`` surfaces it for the agent
  3. ``register_kernel_result`` fulfills the request and caches it
  4. ``lookup_cached_kernel`` hits on the same v3-fingerprint
  5. Re-issuing ``request_kernel_codegen`` for an already-cached
     fingerprint short-circuits with the cached kernel (zero queue churn)
  6. ``ClaudeCodeKernelProvider`` driven by ``InSessionCodegen`` returns
     the cached kernel — proving the no-extra-API-cost path works
  7. Empty ``kernel_code`` re-queues the pending request (doesn't poison
     the cache)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.kernels.contract_v3 import KernelContractV3
from compgen.kernels.contract_v3_references import reference_matmul_contract
from compgen.kernels.provider import SearchBudget
from compgen.kernels.providers.claude_code_default import (
    ClaudeCodeKernelProvider,
    InSessionCodegen,
)
from compgen.kernels.providers.contract_bridge import v3_to_v1_contract
from compgen.kernels.store import KernelStore, set_shared_store
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.kernel import (
    KERNEL_TOOLS,
    contract_fingerprint,
    list_pending_kernel_requests,
    lookup_cached_kernel,
    register_kernel_result,
    request_kernel_codegen,
)

# ---------------------------------------------------------------------------
# Helpers — serialise a v3 contract to the dict shape the tools accept
# ---------------------------------------------------------------------------


def _serialise_v3(c: KernelContractV3) -> dict:
    """Minimal serialisation — only the fields ``contract_fingerprint`` and
    ``_render_prompt`` actually read."""
    return {
        "op_name": c.op_name,
        "archetype": c.archetype.value,
        "granularity": c.granularity.value,
        "io": {
            "inputs": [
                {
                    "name": t.name,
                    "shape": {"dims": list(t.shape.dims)},
                    "dtype_class": list(t.dtype_class),
                    "layout": t.layout.value,
                    "alignment_bytes": t.alignment_bytes,
                }
                for t in c.io.inputs
            ],
            "outputs": [
                {
                    "name": t.name,
                    "shape": {"dims": list(t.shape.dims)},
                    "dtype_class": list(t.dtype_class),
                    "layout": t.layout.value,
                    "alignment_bytes": t.alignment_bytes,
                }
                for t in c.io.outputs
            ],
            "attributes": [{"name": a.name, "value": a.value} for a in c.io.attributes],
            "numerics": {
                "accumulator_dtype": c.io.numerics.accumulator_dtype,
                "fast_math": c.io.numerics.fast_math,
                "max_relative_error": c.io.numerics.max_relative_error,
            },
        },
        "orchestration": {
            "execution": {
                "hardware": {
                    "target_name": (c.orchestration.execution.hardware.target_name if c.orchestration.execution else "")
                }
            }
        },
    }


@pytest.fixture(autouse=True)
def isolated_kernel_store(tmp_path: Path):
    """Each test gets a fresh on-disk store under tmp_path so writes don't
    leak across tests (or into the user's real ~/.compgen/kernels)."""
    set_shared_store(KernelStore(root=tmp_path / "kernel_store"))
    yield
    set_shared_store(None)


@pytest.fixture
def session_manager(tmp_path: Path) -> SessionManager:
    sm = SessionManager(scratch_root=tmp_path / "compgen_mcp")
    sm.open(session_id="sess1")
    return sm


# ---------------------------------------------------------------------------
# Tools registration
# ---------------------------------------------------------------------------


def test_kernel_tools_registered_with_expected_names() -> None:
    names = {t["name"] for t in KERNEL_TOOLS}
    assert names == {
        "request_kernel_codegen",
        "register_kernel_result",
        "lookup_cached_kernel",
        "list_pending_kernel_requests",
    }


def test_kernel_tools_appear_in_all_tools_bundle() -> None:
    """The MCP server iterates ALL_TOOLS to surface tool decorators —
    the kernel tools must be in there."""
    from compgen.mcp.tools import ALL_TOOLS

    names = {t["name"] for t in ALL_TOOLS}
    for kt in (
        "request_kernel_codegen",
        "register_kernel_result",
        "lookup_cached_kernel",
        "list_pending_kernel_requests",
    ):
        assert kt in names, f"{kt} missing from ALL_TOOLS"


# ---------------------------------------------------------------------------
# Fingerprint stability
# ---------------------------------------------------------------------------


def test_fingerprint_stable_across_serialisations_of_same_contract() -> None:
    c = reference_matmul_contract()
    fp1 = contract_fingerprint(_serialise_v3(c))
    fp2 = contract_fingerprint(_serialise_v3(c))
    assert fp1 == fp2


def test_fingerprint_changes_when_target_changes() -> None:
    """Two contracts identical except for target_name must hash differently."""
    c = reference_matmul_contract()
    d1 = _serialise_v3(c)
    d2 = _serialise_v3(c)
    d2["orchestration"]["execution"]["hardware"]["target_name"] = "different_chip"
    assert contract_fingerprint(d1) != contract_fingerprint(d2)


# ---------------------------------------------------------------------------
# Round-trip: request → list pending → register → lookup
# ---------------------------------------------------------------------------


def test_request_then_register_then_lookup_round_trip(session_manager: SessionManager) -> None:
    contract_dict = _serialise_v3(reference_matmul_contract())

    # 1. request — queues, returns rendered prompt + request_id
    req = request_kernel_codegen(
        session_manager,
        session_id="sess1",
        contract_v3=contract_dict,
        perf_target_us=100.0,
    )
    assert req["ok"] and not req["found_in_cache"]
    rid = req["request_id"]
    fp = req["fingerprint"]
    assert "linalg.matmul" in req["prompt"]
    assert "compute_tiled" in req["prompt"]

    # 2. list pending — agent sees the request
    pending = list_pending_kernel_requests(session_manager, session_id="sess1")
    assert pending["pending_count"] == 1
    assert pending["requests"][0]["request_id"] == rid

    # 3. register — agent fulfils
    fulfilled = register_kernel_result(
        session_manager,
        session_id="sess1",
        request_id=rid,
        kernel_code="@triton.jit\ndef matmul(...): ...\n",
        language="triton",
    )
    assert fulfilled["ok"]
    assert fulfilled["fingerprint"] == fp
    assert fulfilled["cached_kernels"] == 1

    # Pending queue empty after fulfilment
    assert list_pending_kernel_requests(session_manager, session_id="sess1")["pending_count"] == 0

    # 4. lookup — hits
    hit = lookup_cached_kernel(session_manager, session_id="sess1", contract_v3=contract_dict)
    assert hit["found"] is True
    assert hit["language"] == "triton"
    assert "@triton.jit" in hit["kernel_code"]


def test_request_short_circuits_on_cached_fingerprint(session_manager: SessionManager) -> None:
    """A second request for the same v3 contract returns the cached
    kernel directly — no new pending entry."""
    contract_dict = _serialise_v3(reference_matmul_contract())
    req = request_kernel_codegen(session_manager, session_id="sess1", contract_v3=contract_dict)
    register_kernel_result(
        session_manager, session_id="sess1", request_id=req["request_id"], kernel_code="// cached body\n", language="c"
    )

    req2 = request_kernel_codegen(session_manager, session_id="sess1", contract_v3=contract_dict)
    assert req2["found_in_cache"] is True
    assert req2["kernel_code"] == "// cached body\n"
    # No new pending entry was queued.
    assert list_pending_kernel_requests(session_manager, session_id="sess1")["pending_count"] == 0


def test_register_with_empty_kernel_code_requeues(session_manager: SessionManager) -> None:
    """Empty body should re-queue, not poison the cache."""
    req = request_kernel_codegen(
        session_manager,
        session_id="sess1",
        contract_v3=_serialise_v3(reference_matmul_contract()),
    )
    out = register_kernel_result(
        session_manager,
        session_id="sess1",
        request_id=req["request_id"],
        kernel_code="   \n",
    )
    assert out["ok"] is False
    assert "empty" in out["error"]
    # Request still pending so the agent can retry.
    assert list_pending_kernel_requests(session_manager, session_id="sess1")["pending_count"] == 1


def test_register_with_unknown_request_id_errors(session_manager: SessionManager) -> None:
    out = register_kernel_result(
        session_manager,
        session_id="sess1",
        request_id="req_does_not_exist",
        kernel_code="// whatever\n",
    )
    assert out["ok"] is False
    assert "unknown" in out["error"]


def test_lookup_miss_returns_found_false(session_manager: SessionManager) -> None:
    out = lookup_cached_kernel(
        session_manager,
        session_id="sess1",
        contract_v3=_serialise_v3(reference_matmul_contract()),
    )
    assert out["found"] is False
    assert out["fingerprint"]


# ---------------------------------------------------------------------------
# End-to-end: ClaudeCodeKernelProvider via InSessionCodegen
# ---------------------------------------------------------------------------


def test_in_session_codegen_returns_cached_kernel_to_provider(session_manager: SessionManager) -> None:
    """The full no-extra-API-cost path:

    1. Pre-populate the cache (simulating the agent's fulfillment turn).
    2. Build the v1 contract via the bridge.
    3. ClaudeCodeKernelProvider uses InSessionCodegen and finds the
       kernel in cache — no external call, ProviderResult.found = True.
    """
    contract_v3 = reference_matmul_contract()
    contract_dict = _serialise_v3(contract_v3)

    # Step 1: pre-populate cache.
    req = request_kernel_codegen(session_manager, session_id="sess1", contract_v3=contract_dict)
    register_kernel_result(
        session_manager,
        session_id="sess1",
        request_id=req["request_id"],
        kernel_code="@triton.jit\ndef k(...): pass\n",
        language="triton",
    )

    # Step 2 + 3: bridge + provider.
    v1 = v3_to_v1_contract(contract_v3)
    provider = ClaudeCodeKernelProvider(
        codegen=InSessionCodegen(sm=session_manager, session_id="sess1"),
    )
    result = provider.search(v1, SearchBudget())

    assert result.found
    assert "@triton.jit" in result.kernel_code
    assert result.language == "triton"
    assert result.iterations_used == 1


def test_in_session_codegen_cache_miss_makes_provider_escalate(session_manager: SessionManager) -> None:
    """No prior request fulfilled → InSessionCodegen returns "" → provider
    reports found=False → escalation router would route to next tier."""
    contract_v3 = reference_matmul_contract()
    v1 = v3_to_v1_contract(contract_v3)

    provider = ClaudeCodeKernelProvider(
        codegen=InSessionCodegen(sm=session_manager, session_id="sess1"),
    )
    result = provider.search(v1, SearchBudget())

    assert result.found is False
    assert "empty source" in result.metadata.get("error", "")


# ---------------------------------------------------------------------------
# Cross-session disk persistence — kernel survives session restart
# ---------------------------------------------------------------------------


def test_kernel_persists_across_session_restart(tmp_path: Path) -> None:
    """Generate a kernel in session A, close it, open fresh session B,
    look up the same v3 contract → cache hit, no re-generation needed.

    The write-through path lives in ``register_kernel_result`` and the
    rehydrate-on-open path lives in ``_kernel_cache``.
    """
    contract_v3 = reference_matmul_contract()
    contract_dict = _serialise_v3(contract_v3)
    expected_kernel = "@triton.jit\ndef matmul_persistent(...): ...\n"

    # ----- Session A: request + register, then drop the session manager -----
    sm_a = SessionManager(scratch_root=tmp_path / "compgen_mcp_a")
    sm_a.open(session_id="sess_a")
    req = request_kernel_codegen(sm_a, session_id="sess_a", contract_v3=contract_dict)
    register_kernel_result(
        sm_a,
        session_id="sess_a",
        request_id=req["request_id"],
        kernel_code=expected_kernel,
        language="triton",
        perf_us=87.3,
        correctness_passed=True,
    )
    fingerprint = req["fingerprint"]
    sm_a.close("sess_a")

    # ----- Session B: fresh manager, fresh session — should rehydrate -----
    sm_b = SessionManager(scratch_root=tmp_path / "compgen_mcp_b")
    sm_b.open(session_id="sess_b")

    hit = lookup_cached_kernel(sm_b, session_id="sess_b", contract_v3=contract_dict)
    assert hit["found"] is True
    assert hit["fingerprint"] == fingerprint
    assert hit["kernel_code"] == expected_kernel
    assert hit["language"] == "triton"

    # And requesting the same contract short-circuits via the cache too.
    req2 = request_kernel_codegen(sm_b, session_id="sess_b", contract_v3=contract_dict)
    assert req2["found_in_cache"] is True
    assert req2["kernel_code"] == expected_kernel


def test_disk_store_writes_files_to_user_folder_layout(tmp_path: Path) -> None:
    """After register_kernel_result, the on-disk store has the kernel file
    at ``<store>/<target>/<fingerprint>.<lang>`` and a manifest entry."""
    sm = SessionManager(scratch_root=tmp_path / "compgen_mcp")
    sm.open(session_id="s1")

    # Use the MICRO ukernel reference because it carries a real
    # ExecutionEnvelope (and therefore a target_name we can assert on).
    from compgen.kernels.contract_v3_references import reference_micro_matmul_tile_contract

    contract = reference_micro_matmul_tile_contract()
    contract_dict = _serialise_v3(contract)
    target_name = contract.orchestration.execution.hardware.target_name

    req = request_kernel_codegen(sm, session_id="s1", contract_v3=contract_dict)
    register_kernel_result(
        sm,
        session_id="s1",
        request_id=req["request_id"],
        kernel_code="@triton.jit\ndef mm(...): pass\n",
        language="triton",
        perf_us=92.1,
        correctness_passed=True,
    )

    store_root = tmp_path / "kernel_store"
    manifest_path = store_root / "manifest.json"
    assert manifest_path.exists(), "manifest.json must be written"

    import json

    manifest = json.loads(manifest_path.read_text())
    fp = req["fingerprint"]
    assert fp in manifest, f"fingerprint {fp} missing from manifest"
    entry = manifest[fp]
    assert entry["language"] == "triton"
    assert entry["op_name"] == "ukernel.matmul_tile_16x16x16_fp16"
    assert entry["archetype"] == "compute_tiled"
    assert entry["granularity"] == "micro"
    assert entry["target"] == target_name
    assert entry["perf_us"] == 92.1
    assert entry["correctness_passed"] is True

    kernel_file = store_root / entry["path"]
    assert kernel_file.exists()
    assert "@triton.jit" in kernel_file.read_text()
    # File path layout: <target>/<fingerprint>.triton.py
    assert entry["path"].startswith(f"{target_name}/{fp}.")
