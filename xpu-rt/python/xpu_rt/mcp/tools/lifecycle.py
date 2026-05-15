"""MCP lifecycle tools: open_target, load_model, compile, bundle_export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.api import compile_model
from compgen.api import device as _device
from compgen.api_llm import _resolve_llm, _resolve_model
from compgen.mcp.async_jobs import JobQueue
from compgen.mcp.session import McpSession, SessionManager

# The JobQueue is module-level so every handler shares the same pool.
_JOBS = JobQueue(max_workers=2, inline_threshold_s=5.0)


def _session(sm: SessionManager, session_id: str | None) -> McpSession:
    if session_id is None:
        # Convenience: auto-open a session on the first lifecycle call.
        return sm.open()
    try:
        return sm.get(session_id)
    except KeyError:
        return sm.open(session_id)


def open_target(
    sm: SessionManager,
    *,
    spec_path: str,
    session_id: str | None = None,
    packs: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load a hardware-spec YAML and attach it to the session.

    If ``packs`` is provided, it replaces the session's pack list (paths
    or ``compgen.packs`` entry-point identifiers) before the target
    device is constructed. Packs previously registered via
    :func:`register_pack` are otherwise preserved.

    MCP tool signature: ``open_target(spec_path, packs?) -> {target_id,
    capabilities, num_stages, session_id, active_packs}``.
    """
    session = _session(sm, session_id)
    path = Path(spec_path).expanduser()
    if not path.exists():
        return {
            "ok": False,
            "error": f"spec_path not found: {path}",
            "session_id": session.session_id,
        }
    session.spec_path = path
    if packs is not None:
        session.packs = tuple(packs)

    dev = _device(path, packs=session.packs or None)
    session.device = dev
    return {
        "ok": True,
        "session_id": session.session_id,
        "target_id": dev.profile.name,
        "target_class": dev.capabilities.target_class.value,
        "num_stages": len(dev.dialect_stack.stages),
        "num_devices": len(dev.profile.devices),
        "active_packs": list(session.packs),
    }


def register_pack(
    sm: SessionManager,
    *,
    session_id: str,
    pack: str,
) -> dict[str, Any]:
    """Register an extension pack with the session by path or entry-point value.

    Examples:
        ``register_pack(sid, "/path/to/my_pack")``      — filesystem
        ``register_pack(sid, "my_extensions")``         — entry-point name
        ``register_pack(sid, "my_mod:PACK_ROOT")``      — dotted

    Behavior:
        - Resolves and loads the pack to verify it's well-formed.
        - Appends it to ``session.packs``.
        - If a target is already open, re-opens the device so the
          pack takes effect immediately.
    """
    from compgen.packs import load_pack

    session = sm.get(session_id)
    try:
        loaded = load_pack(pack)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "session_id": session.session_id,
            "error": f"register_pack: {type(e).__name__}: {e}",
        }

    if pack not in session.packs:
        session.packs = session.packs + (pack,)

    device_rebuilt = False
    if session.device is not None and session.spec_path is not None:
        session.device = _device(session.spec_path, packs=session.packs)
        device_rebuilt = True

    return {
        "ok": True,
        "session_id": session.session_id,
        "pack_name": loaded.manifest.name,
        "pack_kinds": list(loaded.manifest.kinds),
        "active_packs": list(session.packs),
        "device_rebuilt": device_rebuilt,
    }


def load_model(
    sm: SessionManager,
    *,
    session_id: str,
    model_path: str | None = None,
    hf_id: str | None = None,
    llm: str = "gemini",
    budget: int = 10,
    objective: str = "latency",
    packs: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load the user's PyTorch model, run the deterministic pipeline, open a driver.

    Exactly one of ``model_path`` (``.py`` file defining ``build_model``)
    or ``hf_id`` (HuggingFace repo) must be supplied.

    If ``packs`` is provided, each pack is registered (equivalent to
    calling :func:`register_pack` for each) before compilation. This is
    a convenience for opening target + packs + model in a single call.
    """
    session = sm.get(session_id)

    if packs:
        for pack in packs:
            result = register_pack(sm, session_id=session_id, pack=pack)
            if not result.get("ok", False):
                return result

    dev = session.require_device()

    if not (bool(model_path) ^ bool(hf_id)):
        return {
            "ok": False,
            "error": "Supply exactly one of model_path (py file) or hf_id.",
        }

    source = model_path or hf_id
    try:
        module, sample_inputs = _resolve_model(
            source if hf_id else Path(source),
            sample_inputs=None,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"load_model: {type(e).__name__}: {e}"}

    session.model_hint = source

    def _compile() -> dict[str, Any]:
        # Anchor the compile output dir under the MCP session scratch
        # so the trace + IR dumps live next to the bundle for this
        # session. ``compile_model`` installs the trace bus + IR-dump
        # writer for this dir.
        compile_out = session.scratch_dir / "compile"
        compile_out.mkdir(parents=True, exist_ok=True)
        # Bind the session's decision registry BEFORE compile_model runs
        # so stage plugins enqueue their sites into the same registry
        # the agent reads/writes via MCP tools.
        session.require_decision_registry()
        compiled = compile_model(
            module,
            dev,
            objective=objective,
            sample_inputs=sample_inputs,
            output_dir=compile_out,
            session_id=session.session_id,
        )
        client, provider = _resolve_llm(llm)
        env = compiled.create_agent_env(budget=budget)
        driver = LLMDrivenCompiler(
            env=env,
            target=dev.profile,
            llm_client=client,
            transcript_dir=session.scratch_dir / "transcripts",
            budget=budget,
        )
        # Stash on the session via closure.
        session.compiled = compiled
        session.driver = driver
        session.llm_client = client
        session.provider = provider
        num_ops = sum(1 for _ in compiled.payload_module.walk())
        return {
            "ok": True,
            "session_id": session.session_id,
            "model_id": source,
            "num_ops": num_ops,
            "pipeline_passed": compiled.pipeline_result.passed,
            "stages_run": compiled.pipeline_result.stages_run,
            "provider": provider,
            "driver_summary": driver.summary(),
            # Expose the compile output dir so clients can read the
            # trace and ir_dumps without guessing the path.
            "output_dir": str(compile_out),
        }

    return _JOBS.run_inline_or_async("load_model", _compile)


def compile_session(
    sm: SessionManager,
    *,
    session_id: str,
    run_llm_loop: bool = True,
) -> dict[str, Any]:
    """Run the agentic loop end-to-end on the session's compiled model.

    Optionally skips the LLM loop (``run_llm_loop=False``) and just
    re-reports the deterministic pipeline outcome from ``load_model``.
    """
    session = sm.get(session_id)
    compiled = session.require_compiled()

    if not run_llm_loop:
        return {
            "ok": True,
            "session_id": session.session_id,
            "pipeline_passed": compiled.pipeline_result.passed,
            "stages_run": compiled.pipeline_result.stages_run,
            "llm_loop": "skipped",
        }

    driver = session.require_driver()

    def _run() -> dict[str, Any]:
        assert session.llm_client is not None
        result = compiled.run_agentic(
            session.llm_client,
            budget=driver.budget,
            with_recipe=True,
        )
        return {
            "ok": True,
            "session_id": session.session_id,
            "pipeline_passed": compiled.pipeline_result.passed,
            "iterations_run": result.iterations_run,
            "improvement_pct": result.total_improvement_pct,
            "initial_cost_us": result.initial_cost_us,
            "final_cost_us": result.final_cost_us,
        }

    return _JOBS.run_inline_or_async("compile", _run)


def bundle_export(
    sm: SessionManager,
    *,
    session_id: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Write the current recipe + payload bundle to ``output_dir``.

    For ``ukernel_runtime`` targets, also emits a buildable C project
    under ``<output_dir>/baremetal/`` via the baremetal bundle plugin.
    Always reads the LIVE payload module from the driver's env (so
    agent mutations via :func:`apply_recipe` are reflected in the
    output) and falls back to the immutable ``compiled.payload_module``
    only when no driver is attached.
    """
    session = sm.get(session_id)
    compiled = session.require_compiled()
    out = Path(output_dir).expanduser() if output_dir else (session.scratch_dir / "bundle")
    out.mkdir(parents=True, exist_ok=True)

    from compgen.ir.recipe.serialize import recipe_to_mlir

    # Prefer the live payload the driver has been mutating (apply_recipe
    # writes to it). Fall back to the compile-time snapshot if we're in
    # a headless context without a driver.
    driver = session.driver
    live_payload = None
    if driver is not None and driver.env.payload_module is not None:
        live_payload = driver.env.payload_module
    payload_module = live_payload if live_payload is not None else compiled.payload_module
    (out / "payload.mlir").write_text(str(payload_module))

    # Recipe IR if tracked.
    if driver is not None and driver.env.recipe is not None:
        (out / "recipe.mlir").write_text(recipe_to_mlir(driver.env.recipe))

    # Target-class-specific artifacts:
    #   ukernel_runtime → Hexagon-style C project under ``baremetal/``
    #   triton_friendly → Triton .py kernel set under ``triton/``
    # Both are purely additive; neither mutates payload.mlir / recipe.mlir.
    target_class = compiled.device.capabilities.target_class.value
    baremetal_info: dict[str, Any] | None = None
    triton_info: dict[str, Any] | None = None
    if target_class == "ukernel_runtime":
        try:
            from compgen.stages.bundle.baremetal_plugin import (
                write_baremetal_bundle,
            )

            baremetal_dir = out / "baremetal"
            result = write_baremetal_bundle(
                payload_module,
                compiled.device.profile,
                baremetal_dir,
            )
            baremetal_info = {
                "ok": True,
                "output_dir": str(result.output_dir),
                "kernel_files": [p.name for p in result.kernel_files],
                "extension_header": result.extension_header.name,
                "extension_source": result.extension_source.name,
                "makefile": result.makefile.name if result.makefile else None,
            }
        except Exception as exc:  # noqa: BLE001
            baremetal_info = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if target_class == "triton_friendly":
        try:
            from compgen.stages.bundle.triton_plugin import write_triton_bundle

            triton_dir = out / "triton"
            tres = write_triton_bundle(payload_module, triton_dir)
            triton_info = {
                "ok": True,
                "output_dir": str(tres.output_dir),
                "kernel_files": [p.name for p in tres.kernel_files],
                "kernels_emitted": tres.kernels_emitted,
                "manifest": tres.manifest_path.name,
                "skipped": tres.skipped,
            }
        except Exception as exc:  # noqa: BLE001
            triton_info = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    manifest = {
        "session_id": session.session_id,
        "target": compiled.device.profile.name,
        "target_class": target_class,
        "objective": compiled.objective,
        "pipeline_passed": compiled.pipeline_result.passed,
        "stages_run": compiled.pipeline_result.stages_run,
        "model_hint": session.model_hint,
        "payload_source": "live_env" if live_payload is not None else "compile_snapshot",
        "baremetal": baremetal_info,
        "triton": triton_info,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    import hashlib

    sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "ok": True,
        "session_id": session.session_id,
        "path": str(out),
        "sha": sha,
        "files": sorted(p.name for p in out.iterdir()),
        "baremetal": baremetal_info,
        "triton": triton_info,
    }


def poll_job(
    sm: SessionManager,
    *,
    job_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Poll an asynchronous job submitted by a lifecycle tool."""
    _ = session_id  # not used — jobs are process-wide
    return _JOBS.poll(job_id)


# ---------------------------------------------------------------------------
# H2 — enter_phase: closed-enum lifecycle transitions
# ---------------------------------------------------------------------------


def enter_phase(
    sm: SessionManager,
    *,
    target_phase: str,
    session_id: str | None = None,
    unsafe: bool = False,
) -> dict[str, Any]:
    """Move a session into ``target_phase`` if the transition is legal.

    Returns a typed dict carrying the prior phase + the new phase. A
    ``status="blocked"`` row fires when:

    * ``target_phase`` is not a known phase;
    * the transition is not legal under
      :func:`compgen.mcp.phase_taxonomy.is_legal_transition` and
      ``unsafe`` is not set;
    * the session does not exist.
    """

    from compgen.mcp.phase_taxonomy import (
        PHASES,
        is_known_phase,
        is_legal_transition,
    )

    if not is_known_phase(target_phase):
        return {
            "ok": False,
            "status": "blocked",
            "blocked_reason": "unknown_phase",
            "target_phase": target_phase,
            "known_phases": list(PHASES),
        }
    session = _session(sm, session_id)
    prior = session.current_phase
    if not is_legal_transition(from_phase=prior, to_phase=target_phase, unsafe=unsafe):
        return {
            "ok": False,
            "status": "blocked",
            "blocked_reason": "illegal_transition",
            "session_id": session.session_id,
            "from_phase": prior,
            "to_phase": target_phase,
            "unsafe_required": True,
        }
    session.current_phase = target_phase
    import structlog as _structlog

    _structlog.get_logger().info(
        "mcp.phase.entered",
        session_id=session.session_id,
        from_phase=prior,
        to_phase=target_phase,
        unsafe=unsafe,
    )
    return {
        "ok": True,
        "status": "ok",
        "session_id": session.session_id,
        "from_phase": prior,
        "to_phase": target_phase,
    }


# ---------------------------------------------------------------------------
# Tool descriptors (consumed by server.py + tests)
# ---------------------------------------------------------------------------

LIFECYCLE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "enter_phase",
        "description": (
            "Move the session into a new phase. Closed-enum transitions; "
            "backward / cross-phase moves require unsafe=true."
        ),
        "phase": "lifecycle",
        "handler": enter_phase,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "target_phase": {"type": "string"},
                "unsafe": {"type": "boolean", "default": False},
            },
            "required": ["target_phase"],
        },
    },
    {
        "name": "open_target",
        "description": "Load a hardware-spec YAML and open a session.",
        "phase": "lifecycle",
        "handler": open_target,
        "input_schema": {
            "type": "object",
            "properties": {
                "spec_path": {"type": "string"},
                "session_id": {"type": "string"},
                "packs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Extension packs to attach (paths or compgen.packs "
                        "entry-point identifiers). Replaces any previously "
                        "registered packs."
                    ),
                },
            },
            "required": ["spec_path"],
        },
    },
    {
        "name": "register_pack",
        "description": (
            "Register a compgen.packs extension with the session "
            "(path or entry-point identifier). Rebuilds the device if a "
            "target is already open."
        ),
        "phase": "lifecycle",
        "handler": register_pack,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "pack": {
                    "type": "string",
                    "description": ("Filesystem path to a pack root, or a 'package[:attr]' entry-point identifier."),
                },
            },
            "required": ["session_id", "pack"],
        },
    },
    {
        "name": "load_model",
        "description": "Run the deterministic pipeline + open an LLM-driven session.",
        "phase": "lifecycle",
        "handler": load_model,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "model_path": {"type": "string"},
                "hf_id": {"type": "string"},
                "llm": {"type": "string", "default": "gemini"},
                "budget": {"type": "integer", "default": 10},
                "objective": {"type": "string", "default": "latency"},
                "packs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Convenience: packs to register before compiling.",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "compile",
        "description": "Run the full LLM-driven agentic loop on the session's model.",
        "phase": "lifecycle",
        "handler": compile_session,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "run_llm_loop": {"type": "boolean", "default": True},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "bundle_export",
        "description": "Write the session's compiled bundle to disk.",
        "phase": "lifecycle",
        "handler": bundle_export,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "output_dir": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "poll_job",
        "description": "Poll the status of a long-running asynchronous tool call.",
        "phase": "job",
        "handler": poll_job,
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "session_id": {"type": "string"},
            },
            "required": ["job_id"],
        },
    },
]


__all__ = [
    "LIFECYCLE_TOOLS",
    "bundle_export",
    "compile_session",
    "load_model",
    "open_target",
    "poll_job",
    "register_pack",
]
