"""Wave 1.13 — MCP target-extensibility tools tests.

The agentic-compilation hook: the agent introspects, describes,
and registers targets via MCP. Tests cover the surface contract
end-to-end.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_registry():
    from compgen.targets.registry import reset

    reset()
    yield
    reset()


class TestToolDescriptors:
    def test_tools_in_registry(self) -> None:
        """All three target tools must be reachable via the
        top-level ``compgen.mcp.tools`` registry."""
        from compgen.mcp.tools import TARGET_TOOLS, get_all_tools

        names = {t["name"] for t in TARGET_TOOLS}
        assert names == {
            "compgen_list_targets",
            "compgen_describe_target",
            "compgen_register_target",
        }
        all_names = {t["name"] for t in get_all_tools()}
        assert names <= all_names

    def test_tool_handlers_callable(self) -> None:
        from compgen.mcp.tools.targets import (
            compgen_describe_target,
            compgen_list_targets,
            compgen_register_target,
        )

        assert callable(compgen_list_targets)
        assert callable(compgen_describe_target)
        assert callable(compgen_register_target)

    def test_input_schemas_present(self) -> None:
        from compgen.mcp.tools.targets import TARGET_TOOLS

        for tool in TARGET_TOOLS:
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema


class TestListAndDescribe:
    def test_list_empty_registry(self) -> None:
        from compgen.mcp.tools.targets import compgen_list_targets

        out = compgen_list_targets()
        assert out["count"] == 0
        assert out["tree"] == {}
        assert out["targets"] == []

    def test_list_after_register(self) -> None:
        from compgen.mcp.tools.targets import compgen_list_targets
        from compgen.targets.registry import register_target

        register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            rationale="paper-faithful target",
        )
        out = compgen_list_targets()
        assert out["count"] == 1
        assert out["tree"] == {"gpu": {"nvidia": ["blackwell"]}}
        assert out["targets"][0]["target_id"] == "gpu.nvidia.blackwell"

    def test_describe_known_target(self) -> None:
        from compgen.mcp.tools.targets import compgen_describe_target
        from compgen.targets.registry import register_target

        register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            rationale="cuBLASDx + cu13 NVRTC + cluster-launch",
            metadata={"sm_count": 132},
        )
        d = compgen_describe_target(target_id="gpu.nvidia.blackwell")
        assert d["status"] == "ok"
        assert d["target_id"] == "gpu.nvidia.blackwell"
        assert d["rationale"].startswith("cuBLASDx")
        assert d["metadata"]["sm_count"] == 132

    def test_describe_unknown_target_returns_status(self) -> None:
        from compgen.mcp.tools.targets import compgen_describe_target

        d = compgen_describe_target(target_id="gpu.tenstorrent.gridx")
        assert d["status"] == "unknown"
        assert d["target_id"] == "gpu.tenstorrent.gridx"


class TestRegister:
    def test_register_with_no_adapters(self) -> None:
        """Registering with all None adapters succeeds — useful for
        placeholder entries the agent fills in later."""
        from compgen.mcp.tools.targets import compgen_register_target

        out = compgen_register_target(
            target_class="gpu",
            vendor="custom",
            arch="experimental",
            rationale="placeholder target",
        )
        assert out["status"] == "ok"
        assert out["target_id"] == "gpu.custom.experimental"
        assert out["errors"] == []

    def test_register_with_invalid_dotted_path(self) -> None:
        """Non-importable adapter path lands in errors but the
        target still registers (with None for that adapter)."""
        from compgen.mcp.tools.targets import compgen_register_target

        out = compgen_register_target(
            target_class="gpu",
            vendor="tenstorrent",
            arch="gridx",
            probe_module="not.a.real.module.MyProbe",
        )
        assert out["status"] == "import_failed"
        assert out["target_id"] == "gpu.tenstorrent.gridx"
        assert any("probe_module" in e for e in out["errors"])

    def test_registered_target_visible_via_list(self) -> None:
        """End-to-end: register via MCP tool, list via MCP tool —
        same surface."""
        from compgen.mcp.tools.targets import (
            compgen_list_targets,
            compgen_register_target,
        )

        compgen_register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            rationale="MCP-registered Blackwell",
        )
        out = compgen_list_targets()
        assert out["count"] == 1
        # The registration_path must reflect that this came from MCP.
        assert out["targets"][0]["registration_path"] == "mcp"

    def test_mcp_register_can_override_in_tree(self) -> None:
        """Re-registering same target_id via MCP replaces an
        in-tree entry. Lets the agent override defaults at session
        scope."""
        from compgen.mcp.tools.targets import (
            compgen_describe_target,
            compgen_register_target,
        )
        from compgen.targets.registry import register_target

        register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            rationale="in-tree default",
            registration_path="in_tree",
        )
        compgen_register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            rationale="agent override",
        )
        d = compgen_describe_target(target_id="gpu.nvidia.blackwell")
        assert d["status"] == "ok"
        assert d["registration_path"] == "mcp"
        assert d["rationale"] == "agent override"


class TestAgenticDiscoveryFlow:
    """The agent's typical discovery sequence — list → describe →
    decide. End-to-end shape."""

    def test_full_flow(self) -> None:
        from compgen.mcp.tools.targets import (
            compgen_describe_target,
            compgen_list_targets,
            compgen_register_target,
        )

        # Step 1: agent lists what's available — empty.
        before = compgen_list_targets()
        assert before["count"] == 0

        # Step 2: agent registers two custom targets.
        compgen_register_target(
            target_class="gpu",
            vendor="nvidia",
            arch="blackwell",
            rationale="paper-faithful",
            metadata={"tier": "A"},
        )
        compgen_register_target(
            target_class="cpu",
            vendor="x86",
            arch="avx512",
            rationale="CPU fallback",
            metadata={"tier": "C"},
        )

        # Step 3: agent re-lists — now has two.
        after = compgen_list_targets()
        assert after["count"] == 2
        assert after["tree"] == {
            "cpu": {"x86": ["avx512"]},
            "gpu": {"nvidia": ["blackwell"]},
        }

        # Step 4: agent inspects one for a decision.
        d = compgen_describe_target(target_id="gpu.nvidia.blackwell")
        assert d["status"] == "ok"
        assert d["metadata"]["tier"] == "A"


# ---------------------------------------------------------------------------
# REQ-027 — handlers accept the SessionManager arg the MCP server passes.
# ---------------------------------------------------------------------------


class _FakeSM:
    """SessionManager stand-in — these tools don't read it."""


class TestMcpDispatchConvention:
    """REQ-027: ``compgen.mcp.server`` dispatches every handler as
    ``handler(sm, **arguments)``. Before the fix, the target tools'
    handlers had no ``sm`` parameter and the dispatcher raised
    ``TypeError: takes 0 positional arguments but 1 was given``."""

    def test_list_targets_accepts_session_arg(self) -> None:
        from compgen.mcp.tools.targets import compgen_list_targets

        out = compgen_list_targets(_FakeSM())
        assert "count" in out

    def test_describe_target_accepts_session_arg(self) -> None:
        from compgen.mcp.tools.targets import compgen_describe_target

        out = compgen_describe_target(_FakeSM(), target_id="x.y.z")
        assert out["status"] == "unknown"

    def test_register_target_accepts_session_arg(self) -> None:
        from compgen.mcp.tools.targets import compgen_register_target

        out = compgen_register_target(
            _FakeSM(),
            target_class="t",
            vendor="v",
            arch="a",
            rationale="REQ-027 dispatch test",
        )
        assert "target_id" in out

    def test_dispatch_via_tool_dict_does_not_raise(self) -> None:
        """Mirrors ``compgen/mcp/server.py:107`` exactly:
        ``tool["handler"](sm, **arguments)``. Each tool dict's
        handler must complete cleanly under that shape."""
        from compgen.mcp.tools.targets import TARGET_TOOLS

        sm = _FakeSM()
        called = 0
        for tool in TARGET_TOOLS:
            if tool["name"] == "compgen_list_targets":
                tool["handler"](sm)
                called += 1
            elif tool["name"] == "compgen_describe_target":
                tool["handler"](sm, target_id="gpu.foo.bar")
                called += 1
            elif tool["name"] == "compgen_register_target":
                tool["handler"](
                    sm,
                    target_class="gpu",
                    vendor="reg027",
                    arch="dispatch_smoke",
                    rationale="dispatch shape probe",
                )
                called += 1
        assert called == 3


class TestToolDictPhase:
    """REQ-027 (secondary): every tool dict must include ``phase`` so
    it satisfies the ``compgen.plugins._validate_mcp_tool`` shape."""

    def test_all_tools_have_phase(self) -> None:
        from compgen.mcp.tools.targets import TARGET_TOOLS

        for tool in TARGET_TOOLS:
            assert "phase" in tool, tool["name"]
            assert isinstance(tool["phase"], str) and tool["phase"]

    def test_validator_accepts_target_tools(self) -> None:
        """The pack-side ``compgen.mcp.tools`` entry-point validator
        accepts the in-tree target tools' shape — anyone re-using
        them as a starter for a pack-owned tool stays compatible."""
        from compgen.mcp.tools.targets import TARGET_TOOLS
        from compgen.plugins import _VALIDATORS

        validator = _VALIDATORS["compgen.mcp.tools"]
        ok, msg = validator(list(TARGET_TOOLS))
        assert ok, msg
