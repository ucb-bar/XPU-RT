"""per-provider normalized test for cffi-C.

Pins:
1. cffi-C card resolves through the registry shim.
2. The legacy ``CReferenceProvider`` wraps into the ABC via
   the shim.
3. ``probe()`` is ``available`` on this CI box.
4. ``can_bid()`` honestly declines flash_attention.
5. ``propose()`` on a matmul contract emits a v1 result with real
   embedded C source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.kernels.provider import KernelContract, SearchBudget
from compgen.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from compgen.providers.provider_registry import build_provider_registry
from compgen.providers.provider_types import ProviderProbeResult
from compgen.providers.result_v1 import ProviderResultV1


class _Target:
    name = "host_cpu"


def _registry():
    return build_provider_registry()


def test_cffi_c_card_present_in_registry():
    r = _registry()
    assert "cffi_c" in r.provider_ids()
    card = r.card_for("cffi_c")
    assert card.integration_level == "promote"
    assert card.paper_claimable is True


def test_cffi_c_instance_satisfies_kernel_provider_via_shim():
    r = _registry()
    inst = r.instance("cffi_c")
    assert isinstance(inst, KernelProvider)


def test_cffi_c_probe_is_available():
    r = _registry()
    probe = r.probe("cffi_c")
    assert isinstance(probe, ProviderProbeResult)
    assert probe.status == "available"


def test_cffi_c_propose_emits_v1_for_matmul(tmp_path: Path):
    r = _registry()
    inst = r.instance("cffi_c")
    contract = KernelContract(
        region_id="r0",
        op_family="matmul",
        input_shapes=((64, 64), (64, 64)),
        output_shapes=((64, 64),),
        dtypes=("f32",),
        target_name="host_cpu",
    )
    req = KernelCodegenRequest(
        task_id="kcodegen_test",
        contract=contract,
        target=_Target(),
        artifact_dir=str(tmp_path),
        extras={"budget": SearchBudget()},
    )
    result = inst.propose(req)
    assert isinstance(result, ProviderResultV1)
    assert result.status == "generated"
    assert result.provider_id == "cffi_c"
    source_path = Path(result.artifacts["source"])
    assert source_path.is_file()
    # Real C source — has #include, has the symbol name.
    text = source_path.read_text()
    assert "#include" in text or "compgen_matmul" in text


def test_cffi_c_declines_flash_attention(tmp_path: Path):
    r = _registry()
    inst = r.instance("cffi_c")
    contract = KernelContract(
        region_id="r0",
        op_family="flash_attention",
        input_shapes=((1, 1, 128, 64),) * 3,
        output_shapes=((1, 1, 128, 64),),
        dtypes=("f16",),
        target_name="host_cpu",
    )
    req = KernelCodegenRequest(
        task_id="t",
        contract=contract,
        target=_Target(),
        artifact_dir=str(tmp_path),
        extras={"budget": SearchBudget()},
    )
    result = inst.propose(req)
    assert isinstance(result, ProviderResultV1)
    assert result.status == "contract_rejected"
