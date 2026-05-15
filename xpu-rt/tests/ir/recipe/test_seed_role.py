"""P7.1 — semantic role tags propagated from payload _pattern_hint to
``RecipeRegionOp.role``.

A real captured Gemma payload has ``compgen._pattern_hint`` on every
op (matmul / softmax / rmsnorm / silu / view / cat ...). The seed
recipe used to drop that on the floor; with P7.1 it's threaded into
each ``RecipeRegionOp.role``, which agents reach via ``view_recipe``,
``get_dossier.region_map``, and the C codegen comment trail.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.ir.recipe.seed import generate_seed_recipe

EXEMPLAR = Path(__file__).resolve().parents[2] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _seed_recipe():
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _MLP().eval(),
        dev,
        sample_inputs=(torch.randn(1, 32),),
    )
    return generate_seed_recipe(
        compiled.payload_module,
        dev.profile,
        "latency",
    )


def test_recipe_region_op_has_role_property_defined() -> None:
    """Constructing a RecipeRegionOp with role= must succeed + verify."""
    from compgen.ir.recipe.ops_scope import RecipeRegionOp
    from xdsl.dialects.builtin import StringAttr

    op = RecipeRegionOp.build(
        properties={
            "sym_name": StringAttr("r_test"),
            "payload_region_id": StringAttr("MmOp_0"),
            "role": StringAttr("matmul"),
        }
    )
    op.verify()
    role = op.properties.get("role")
    assert isinstance(role, StringAttr)
    assert role.data == "matmul"


def test_seed_recipe_stamps_role_on_regions() -> None:
    """At least one recipe.region in a seed-built recipe must carry role."""
    recipe = _seed_recipe()
    region_ops = [op for op in recipe.body.block.ops if op.name == "recipe.region"]
    assert region_ops, "seed produced zero recipe.region ops"

    with_role = [op for op in region_ops if op.properties.get("role") is not None]
    assert with_role, "no recipe.region carries a role — pattern hints didn't propagate"

    # Every captured op in our toy MLP should land with a useful role.
    # The exact set depends on how torch.export decomposes Linear+ReLU,
    # but at minimum we expect one role string with a recognisable
    # op-family name (matmul / mm / addmm / view / relu / max / ...).
    role_strings = {op.properties["role"].data for op in with_role}
    assert any(
        any(
            family in r
            for family in (
                "matmul",
                "mm",
                "addmm",
                "linear",
                "relu",
                "view",
                "permute",
                "transpose",
            )
        )
        for r in role_strings
    ), f"role strings look unrecognisable: {role_strings}"


def test_role_is_optional_when_no_pattern_hint() -> None:
    """A seed built from a payload with no pattern hints must still
    produce a valid recipe — role is optional."""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    empty = ModuleOp(Region([Block()]))
    # generate_seed_recipe expects a non-None target_profile for some
    # backend inference; reuse the device's.
    dev = _device(EXEMPLAR)
    recipe = generate_seed_recipe(empty, dev.profile, "latency")
    # Empty payload → zero regions; recipe still verifies.
    recipe.verify()


def test_dossier_region_map_carries_role_when_present() -> None:
    """End-to-end via get_dossier."""
    from compgen.agent.invent_slots.registrar import register_invent_slots
    from compgen.agent.llm_driver import LLMDrivenCompiler
    from compgen.llm.mock_client import MockLLMClient
    from compgen.llm.registry import Registry
    from compgen.mcp.session import SessionManager
    from compgen.mcp.tools.inspect import get_dossier

    sm = SessionManager()
    session = sm.open()
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        _MLP().eval(),
        dev,
        sample_inputs=(torch.randn(1, 32),),
    )
    reg = Registry()
    register_invent_slots(reg)
    env = compiled.create_agent_env(budget=2)
    driver = LLMDrivenCompiler(
        env=env,
        target=dev.profile,
        llm_client=MockLLMClient(strict=False),
        budget=2,
        registry=reg,
    )
    session.compiled = compiled
    session.device = dev
    session.driver = driver

    r = get_dossier(sm, session_id=session.session_id)
    rmap = r["region_map"]
    # New shape — every entry is a dict.
    assert all(isinstance(v, dict) for v in rmap.values())
    # At least one entry has a role tag.
    assert any(v.get("role") for v in rmap.values())
    # Reverse index agrees with forward.
    rbr = r["regions_by_role"]
    for role, syms in rbr.items():
        for s in syms:
            assert rmap[s].get("role") == role
