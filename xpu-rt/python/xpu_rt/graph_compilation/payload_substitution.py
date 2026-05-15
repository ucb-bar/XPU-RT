"""IR-level closure: registry-driven FX-graph rewrites.

When ``lower.py`` is invoked with a populated extension registry, this
pre-pass walks each FX graph and replaces every ``call_function`` node
whose target matches a registered extension with the *traced sub-graph*
of the extension's body. The rewritten graph is then handed to
``FXImporter`` as usual, so:

- targets that used to lower to opaque ``func.call @<target>`` now
  lower to the decomposed primitives (``linalg.matmul``, ``linalg.gelu``,
  …) the agent's extension expressed them in.
- the resulting ``payload.mlir`` carries **zero** opaque calls for
  closed targets — closing the agentic loop at the IR level.

Constraints: the extension function must be FX-traceable. For the
canonical ``crgtoy.affine_gelu → F.gelu(F.linear(x, w, b))`` case this
holds trivially. If tracing fails, the substitution is skipped for
that node and recorded in diagnostics — verify already proved
mathematical equivalence, but if we can't represent the extension as
FX we fall back to keeping the opaque call.

The pass is **read-only on the registry** and **does not edit
extension files** — it only rewrites the in-memory FX graph.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.fx

from compgen.graph_compilation.extension_registry import ExtensionRegistry, RegistryEntry

# Same canonicalization as lower.py — keeps the matching consistent.
_HEX_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _canon(s: str) -> str:
    return _HEX_ADDR_RE.sub("", s)


@dataclass
class SubstitutionResult:
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def num_substituted(self) -> int:
        return len(self.substitutions)


# --------------------------------------------------------------------------- #
# Extension function loading
# --------------------------------------------------------------------------- #


def _load_extension_callable(extension_path: Path) -> Any:
    """Load ``extension.py::extension`` from a workspace dir."""
    ep = Path(extension_path) / "extension.py"
    if not ep.exists():
        raise FileNotFoundError(f"extension.py missing: {ep}")
    module_name = f"crg_inline_ext_{ep.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, ep)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension: {ep}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "extension"):
        raise AttributeError(f"{ep} has no `extension` callable")
    return module.extension


# --------------------------------------------------------------------------- #
# Symbolic-trace + splice
# --------------------------------------------------------------------------- #


def _arg_meta_from_parent(node: torch.fx.Node) -> list[tuple[list[int], torch.dtype]]:
    """Extract (shape, dtype) for each arg-bearing predecessor of ``node``.

    Pulls from ``arg.meta['val']`` (populated by ``_normalize_fx_meta``
    in ``lower.py``). Falls back to ``torch.float32`` + 1-D scalar
    when meta is missing — the extension's tracer will surface a
    real failure if the shapes don't make sense.
    """
    out: list[tuple[list[int], torch.dtype]] = []
    for a in node.args:
        if not isinstance(a, torch.fx.Node):
            continue
        val = a.meta.get("val") if hasattr(a, "meta") else None
        if isinstance(val, torch.Tensor):
            shape = [int(s) if isinstance(s, int) else 1 for s in val.shape]
            out.append((shape, val.dtype))
        else:
            out.append(([1], torch.float32))
    return out


def _trace_extension(
    extension_fn: Any,
    arg_meta: list[tuple[list[int], torch.dtype]],
) -> torch.fx.GraphModule:
    """Trace the extension into an FX sub-graph with shape/dtype metadata.

    Uses ``torch.fx.experimental.proxy_tensor.make_fx`` in fake-tensor
    mode so each traced node carries ``meta['val']`` — required by
    ``FXImporter._tensor_type_from_meta``. ``arg_meta`` carries
    ``(shape, dtype)`` per parent-node arg so the fake inputs are
    shape-correct.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    # Build a wrapper with a fixed positional arity (proxy_tensor.make_fx
    # accepts *args; we still build the wrapper for clarity / consistent
    # naming with ``_argN``).
    arg_names = [f"_arg{i}" for i in range(len(arg_meta))]
    arg_list = ", ".join(arg_names)
    src = f"def _wrapped({arg_list}):\n    return extension({arg_list})\n"
    namespace: dict[str, Any] = {"extension": extension_fn}
    exec(src, namespace)
    wrapped = namespace["_wrapped"]

    example_inputs = [
        torch.zeros(shape, dtype=dtype) for shape, dtype in arg_meta
    ]
    gm = make_fx(wrapped, tracing_mode="fake")(*example_inputs)
    return gm


def _splice_extension(
    parent_graph: torch.fx.Graph,
    parent_node: torch.fx.Node,
    extension_gm: torch.fx.GraphModule,
    *,
    inline_prefix: str,
) -> Any:
    """Splice ``extension_gm.graph`` into ``parent_graph`` in place of ``parent_node``.

    Returns the final value-producing node so the caller can wire its
    consumers.
    """
    sub_graph = extension_gm.graph
    placeholders = [n for n in sub_graph.nodes if n.op == "placeholder"]
    output_nodes = [n for n in sub_graph.nodes if n.op == "output"]

    parent_args = list(parent_node.args)
    if len(placeholders) != len(parent_args):
        raise ValueError(
            f"arity mismatch: extension expects {len(placeholders)} args, "
            f"parent node provides {len(parent_args)}"
        )

    # Map sub_graph node → value (in parent_graph context).
    val_map: dict[torch.fx.Node, Any] = {}
    for p, a in zip(placeholders, parent_args):
        val_map[p] = a

    last_value: Any = None
    suffix = 0
    with parent_graph.inserting_before(parent_node):
        for n in sub_graph.nodes:
            if n.op == "placeholder":
                continue
            if n.op == "output":
                # Single-output: args[0] is the produced value (possibly a tuple).
                produced = n.args[0] if n.args else None
                if isinstance(produced, (tuple, list)):
                    last_value = tuple(val_map.get(p, p) for p in produced)
                else:
                    last_value = val_map.get(produced, produced)
                continue

            def _resolve(v: Any) -> Any:
                if isinstance(v, torch.fx.Node):
                    return val_map.get(v, v)
                if isinstance(v, (list, tuple)):
                    return type(v)(_resolve(x) for x in v)
                if isinstance(v, dict):
                    return {k: _resolve(x) for k, x in v.items()}
                return v

            new_args = tuple(_resolve(a) for a in n.args)
            new_kwargs = {k: _resolve(v) for k, v in n.kwargs.items()}
            new_name = f"{inline_prefix}_{n.name}_{suffix}"
            suffix += 1
            # ``get_attr`` nodes refer to the extension GraphModule's
            # parameters; we don't currently splice those, so skip and
            # record (the affine_gelu case is parameter-free since the
            # weights come in through the parent call's args).
            if n.op == "get_attr":
                raise ValueError(
                    f"extension graph references attribute {n.target!r}; "
                    "in-place splice doesn't carry parameters yet"
                )
            new_node = parent_graph.create_node(
                op=n.op,
                target=n.target,
                args=new_args,
                kwargs=new_kwargs,
                name=new_name,
            )
            # Carry FX meta so downstream type inference is reasonable.
            if hasattr(n, "meta"):
                new_node.meta.update(n.meta)
            val_map[n] = new_node
            last_value = new_node
    return last_value


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def apply_extensions(
    graph: torch.fx.Graph,
    registry: ExtensionRegistry,
    *,
    gap_kind_filter: str = "unsupported_op",
) -> SubstitutionResult:
    """Walk ``graph`` and inline registered extensions.

    Mutates ``graph`` in place. Returns a record of every substitution
    + skip reason for diagnostics.
    """
    result = SubstitutionResult()
    # Build a target → entry index for fast lookup. We canonicalize both
    # sides (Dynamo records ``str(node.target)`` which can include hex
    # addresses; the registry stores the canonicalized fx_target).
    by_target: dict[str, RegistryEntry] = {}
    for entry in registry.entries:
        if entry.gap_kind != gap_kind_filter:
            continue
        if entry.verification_status != "pass":
            continue
        by_target[_canon(entry.fx_target)] = entry

    if not by_target:
        return result

    nodes_to_process = [n for n in graph.nodes if n.op == "call_function"]
    for node in nodes_to_process:
        if node not in graph.nodes:
            # Already removed by an earlier pass.
            continue
        canon_target = _canon(str(node.target))
        entry = by_target.get(canon_target)
        if entry is None:
            continue

        try:
            ext_fn = _load_extension_callable(Path(entry.extension_path))
            arg_meta = _arg_meta_from_parent(node)
            ext_gm = _trace_extension(ext_fn, arg_meta)
            inline_prefix = f"ext_{entry.extension_id[-8:]}"
            new_value = _splice_extension(graph, node, ext_gm, inline_prefix=inline_prefix)
        except Exception as exc:
            result.skipped.append(
                {
                    "fx_node": node.name,
                    "fx_target": canon_target,
                    "extension_id": entry.extension_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        # Replace usages of the original node with the new value.
        if new_value is None:
            result.skipped.append(
                {
                    "fx_node": node.name,
                    "fx_target": canon_target,
                    "extension_id": entry.extension_id,
                    "reason": "extension produced no output value",
                }
            )
            continue
        node.replace_all_uses_with(new_value)
        graph.erase_node(node)

        result.substitutions.append(
            {
                "fx_node": node.name,
                "fx_target": canon_target,
                "extension_id": entry.extension_id,
            }
        )

    if result.substitutions:
        graph.lint()
    return result
