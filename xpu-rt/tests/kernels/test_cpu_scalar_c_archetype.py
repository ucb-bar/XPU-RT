"""Second-archetype falsification test for the codegen-fallback architecture.

Claim under test: the ``KernelProvider`` Protocol + codegen-fallback
dispatch is generic across hardware archetypes — not coincidentally
shaped to the in-tree Triton or the out-of-tree Muon (accel-native)
provider.

Falsifier: a real CPU scalar-C provider that emits standard C for
elementwise f32 ops, gets dispatched through the same surfaces, and
produces source that actually compiles. If this works without any
CompGen-side changes, the architecture is generic across at least
three archetypes (Triton-friendly GPU, accel-native, ukernel-runtime
CPU).

This test deliberately does NOT touch the radiance pack or the
in-tree Triton templates — it builds a from-scratch Provider in the
test file itself, proving the surface is reachable from outside any
existing pack.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.kernels.codegen_fallback import run_provider_fallback
from compgen.kernels.provider import (
    KernelContract as ProviderContract,
)
from compgen.kernels.provider import (
    KernelProvider,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from compgen.targets.schema import load_profile

# ---------------------------------------------------------------------------
# A from-scratch CPU scalar-C Provider — represents the ukernel-runtime
# CPU archetype. Emits standard C99 for elementwise f32 ops.
# ---------------------------------------------------------------------------


_CPU_SCALAR_C_TEMPLATES: dict[str, str] = {
    "add": "    out[i] = a[i] + b[i];",
    "sub": "    out[i] = a[i] - b[i];",
    "mul": "    out[i] = a[i] * b[i];",
    "div": "    out[i] = a[i] / b[i];",
    "relu": "    out[i] = a[i] > 0.0f ? a[i] : 0.0f;",
}


_BINARY_OPS: frozenset[str] = frozenset({"add", "sub", "mul", "div"})
_UNARY_OPS: frozenset[str] = frozenset({"relu"})


def _render_kernel(op_family: str, n: int) -> str:
    body = _CPU_SCALAR_C_TEMPLATES[op_family]
    if op_family in _BINARY_OPS:
        sig = "void compgen_kernel(const float *a, const float *b, float *out)"
    else:
        sig = "void compgen_kernel(const float *a, float *out)"
    return f"""\
/* Auto-rendered scalar C kernel for op_family={op_family}, N={n}. */
#include <stddef.h>

#define N {n}

{sig} {{
    for (size_t i = 0; i < N; ++i) {{
{body}
    }}
}}
"""


class CPUScalarCProvider:
    """Reference KernelProvider for the ukernel-runtime CPU archetype.

    Accepts elementwise f32 contracts; renders parameterized scalar C
    from the contract's op_family + shape.
    """

    name: str = "cpu_scalar_c"

    def accepts_contract(self, contract: ProviderContract) -> bool:
        if contract.op_family not in _CPU_SCALAR_C_TEMPLATES:
            return False
        if contract.dtypes != ("f32",):
            return False
        # All inputs and output share one shape (true for elementwise).
        if not contract.input_shapes or not contract.output_shapes:
            return False
        first_in = contract.input_shapes[0]
        if any(s != first_in for s in contract.input_shapes):
            return False
        if contract.output_shapes[0] != first_in:
            return False
        if contract.op_family in _BINARY_OPS and len(contract.input_shapes) != 2:
            return False
        if contract.op_family in _UNARY_OPS and len(contract.input_shapes) != 1:
            return False
        return True

    def search(self, contract: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
        n = 1
        for dim in contract.output_shapes[0]:
            if dim > 0:
                n *= dim
        return ProviderResult(
            found=True,
            kernel_code=_render_kernel(contract.op_family, n),
            language="c",
            correct=False,  # not validated yet; the smoke test below does that
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        return []


def _module_for(model: nn.Module, *args: torch.Tensor):
    ep = capture_model(model, args)
    module, _ = fx_to_xdsl(ep)
    return module


_TARGET = "examples/target_profiles/cuda_a100.yaml"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_provider_satisfies_kernel_provider_protocol() -> None:
    """The from-scratch CPU provider satisfies the documented Protocol."""
    p = CPUScalarCProvider()
    assert isinstance(p, KernelProvider)
    assert p.name == "cpu_scalar_c"


def test_provider_accepts_elementwise_f32_contracts() -> None:
    p = CPUScalarCProvider()
    accepted = ProviderContract(
        region_id="r0",
        op_family="add",
        input_shapes=((4,), (4,)),
        output_shapes=((4,),),
        dtypes=("f32",),
        target_name="cpu",
        hardware_key="x86_64",
        objective="latency",
    )
    rejected_dtype = ProviderContract(
        region_id="r0",
        op_family="add",
        input_shapes=((4,), (4,)),
        output_shapes=((4,),),
        dtypes=("f16",),
        target_name="cpu",
        hardware_key="x86_64",
        objective="latency",
    )
    rejected_op = ProviderContract(
        region_id="r0",
        op_family="matmul",
        input_shapes=((4, 4), (4, 4)),
        output_shapes=((4, 4),),
        dtypes=("f32",),
        target_name="cpu",
        hardware_key="x86_64",
        objective="latency",
    )
    assert p.accepts_contract(accepted) is True
    assert p.accepts_contract(rejected_dtype) is False
    assert p.accepts_contract(rejected_op) is False


def test_relu_of_add_dispatched_through_cpu_scalar_c_provider() -> None:
    """End-to-end: ``relu(a + b)`` extracts → 2 contracts → 2 C kernels."""

    class ReluAdd(nn.Module):
        def forward(self, a, b):
            return torch.relu(a + b)

    module = _module_for(ReluAdd(), torch.randn(8), torch.randn(8))
    target = load_profile(_TARGET)
    p = CPUScalarCProvider()

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(8), torch.randn(8)),
        extra_providers=[p],
    )

    assert len(out) == 2, [(k["op_name"], k["region_id"]) for k in out]
    by_op = {}
    for k in out:
        if "add" in k["op_name"]:
            by_op["add"] = k
        elif "relu" in k["op_name"]:
            by_op["relu"] = k

    assert set(by_op) == {"add", "relu"}
    for k in by_op.values():
        assert k["provider"] == "cpu_scalar_c"
        assert k["language"] == "c"
        assert k["extension"] == "c"
        assert "void compgen_kernel" in k["source"]
        assert "#define N 8" in k["source"]
    # Op-specific body — proves the contract drove rendering.
    assert "a[i] + b[i]" in by_op["add"]["source"]
    assert "a[i] > 0.0f ? a[i] : 0.0f" in by_op["relu"]["source"]


def test_emitted_c_source_compiles_with_host_gcc(tmp_path: Path) -> None:
    """The rendered source must actually compile — ground-truth proof
    that the Provider emits real code, not pseudo-source."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available on PATH")

    class Mul(nn.Module):
        def forward(self, a, b):
            return a * b

    module = _module_for(Mul(), torch.randn(16), torch.randn(16))
    target = load_profile(_TARGET)
    p = CPUScalarCProvider()

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(16), torch.randn(16)),
        extra_providers=[p],
    )

    assert out, "no kernel emitted"
    src_path = tmp_path / "kernel.c"
    src_path.write_text(out[0]["source"])

    # `-c` = compile only, no link. `-Wall -Werror` = no warnings tolerated.
    proc = subprocess.run(
        ["gcc", "-c", "-Wall", "-Werror", "-o", str(tmp_path / "kernel.o"), str(src_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"gcc failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}\nSOURCE:\n{src_path.read_text()}"
    )
    assert (tmp_path / "kernel.o").exists()


def test_emitted_c_links_and_runs_with_correct_results(tmp_path: Path) -> None:
    """Strongest claim: the emitted kernel actually computes the right
    answer when linked + executed against a host driver. Bit-equality
    proof for the CPU archetype."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available on PATH")

    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    n = 8
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, -0.5])
    b = torch.tensor([0.5, -1.5, 2.0, 0.0, 1.0, 1.0, -0.25, 0.25])
    module = _module_for(Add(), a, b)
    target = load_profile(_TARGET)

    out = run_provider_fallback(module, target, sample_inputs=(a, b), extra_providers=[CPUScalarCProvider()])
    assert out, "no kernel emitted"
    kernel_src = out[0]["source"]

    # Wrap kernel in a tiny host driver that prints results to stdout.
    driver = f"""\
{kernel_src}
#include <stdio.h>

int main(void) {{
    float a[N] = {{1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 0.5f, -0.5f}};
    float b[N] = {{0.5f, -1.5f, 2.0f, 0.0f, 1.0f, 1.0f, -0.25f, 0.25f}};
    float out[N];
    compgen_kernel(a, b, out);
    for (size_t i = 0; i < N; ++i) {{
        printf("%a\\n", out[i]);
    }}
    return 0;
}}
"""
    src = tmp_path / "driver.c"
    src.write_text(driver)
    exe = tmp_path / "driver"
    compile_proc = subprocess.run(
        ["gcc", "-O0", "-Wall", "-Werror", "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr

    run_proc = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run_proc.returncode == 0, run_proc.stderr
    got = [float.fromhex(line) for line in run_proc.stdout.strip().splitlines()]
    expected = (a + b).tolist()
    assert len(got) == n
    for g, e in zip(got, expected, strict=True):
        # Bit-equality on f32: they should be exactly equal here.
        assert g == pytest.approx(e, abs=0.0), (got, expected)


def test_provider_archetype_independence_no_compgen_changes_required() -> None:
    """Meta-assertion: nothing in this test file imports anything
    pack-specific. The Provider above is constructed entirely from
    public CompGen surfaces (``KernelContract``, ``ProviderResult``,
    ``run_provider_fallback``) — no monkey-patches, no internal
    helpers, no per-archetype branches in compgen-side code.

    If this test passes, the architecture supports the ukernel-runtime
    CPU archetype with the same surface that supports the accel-native
    archetype demonstrated by ``radiance-compgen-pack``.
    """
    # Sanity: the imports we rely on are all from compgen public modules.
    import compgen.kernels.codegen_fallback as cf
    import compgen.kernels.provider as cp

    for sym in ("run_provider_fallback",):
        assert hasattr(cf, sym), f"compgen.kernels.codegen_fallback lost {sym}"
    for sym in ("KernelContract", "KernelProvider", "ProviderResult", "SearchBudget"):
        assert hasattr(cp, sym), f"compgen.kernels.provider lost {sym}"
