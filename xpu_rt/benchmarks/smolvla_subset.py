"""Enumerate unique kernel contracts for the SmolVLA comparison study.

Drives the canonical SmolVLA loader at
:func:`xpu_rt.models.robotics.load_smolvla_bundle`, walks the loaded
``nn.Module`` tree for every :class:`torch.nn.Linear` (and `Conv2d`,
later), synthesises a :class:`KernelContract` per unique
``(in_features, out_features)`` shape per component, and emits the
deduped set as JSON manifests under
``.xpu_rt/benchmarks/smolvla_subset/``.

Phase A consumes :func:`enumerate_unique_contracts` with ``limit=25``;
Phase B passes ``limit=None`` to get every Linear-shape contract
(typically a few dozen post-dedup).

We deliberately use module enumeration rather than FX-graph walking
because TorchDynamo's partition-capture path (the one
:func:`examples.models.smolvla_wrapper.capture_fx_graphs` uses) does
*not* populate ``node.meta['val']`` for SmolVLA's
``vlm_with_expert.lm_expert.layers.N.<submod>.<linear>`` nodes —
shape inference would require running a fake-tensor pass on top, and
the resulting contracts would carry the same dims we already have on
the ``nn.Linear`` instances. Skipping the FX pass keeps this module
cheap and lets us name kernels by their owning module path.

The output contract's ``input_shapes[0]`` uses a configurable seq_len
(default 64 — small enough for Spike to run in seconds; representative
of SmolVLA's per-step action-chunk length). The agent loop only cares
about ``M × K`` and ``K × N`` for matmul region-signature hashing, so
the exact seq_len doesn't affect kernel-search behaviour.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from xpu_rt.kernels.provider import KernelContract
from xpu_rt.promotion.region_signature import (
    RegionSignature,
    hash_region_signature,
    make_region_signature,
)

logger = logging.getLogger(__name__)


# Defaults for sequence length stamped into each emitted matmul contract.
# Plenty for a Phase-A demo; the agent loop uses the (K, N) pair via the
# region_signature, so the exact M doesn't change which kernels match.
DEFAULT_SEQ_LEN = 64


# ---------------------------------------------------------------------------
# Module-tree walker
# ---------------------------------------------------------------------------


def _iter_linear_modules(model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    """Yield (qualified_name, module) for every nn.Linear in the tree."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            yield name, mod


def _action_expert_layer_index_for_path(path: str) -> int | None:
    """Best-effort layer-index extraction from a module path.

    Looks for ``layers.N`` in the path. SmolVLA's expert lives at
    ``...lm_expert.layers.N.<submod>``; PaliGemma backbone uses the same
    naming convention.
    """
    low = path.lower()
    for needle in ("layers.", "layer."):
        idx = low.find(needle)
        if idx < 0:
            continue
        after = low[idx + len(needle):]
        digits = ""
        for ch in after:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return int(digits)
    return None


def _component_for_path(path: str) -> set[str]:
    out: set[str] = set()
    low = path.lower()
    for component, substrings in _COMPONENT_SUBSTRINGS.items():
        if any(s in low for s in substrings):
            out.add(component)
    return out


# ---------------------------------------------------------------------------
# Contract synthesis
# ---------------------------------------------------------------------------


def _synthesize_contract_from_linear(
    *,
    name: str,
    module: nn.Linear,
    target_class: str,
    seq_len: int,
    quantize_to_i8: bool,
) -> KernelContract:
    """Build a KernelContract from one nn.Linear instance."""
    K = int(module.in_features)
    N = int(module.out_features)
    M = int(seq_len)
    if quantize_to_i8:
        dtypes = ("i8", "i8", "i32")
    else:
        weight_dtype = module.weight.dtype
        dtype = _canonical_dtype(str(weight_dtype).removeprefix("torch."))
        dtypes = (dtype, dtype, "fp32" if dtype in ("bf16", "fp16") else dtype)
    return KernelContract(
        region_id=name,
        op_family="matmul",
        input_shapes=((M, K), (K, N)),
        output_shapes=((M, N),),
        dtypes=dtypes,
        layout="row_major",
        target_name=target_class,
    )


def _canonical_dtype(name: str) -> str:
    name = name.lower()
    mapping = {
        "float32": "fp32",
        "float": "fp32",
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "int8": "i8",
        "uint8": "u8",
        "int32": "i32",
        "int64": "i64",
        "bool": "bool",
        "float8_e4m3fn": "f8e4m3",
        "float8_e5m2": "f8e5m2",
    }
    return mapping.get(name, name)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


# Module-path substrings that identify each SmolVLA component.
#
# The PaliGemma/Gemma module tree as Dynamo names it puts:
#   - the language+action joint expert at
#     ``L['model'].vlm_with_expert.lm_expert.layers.N.<submod>``
#   - the action head's projections at ``L['model'].action_in_proj``,
#     ``L['model'].action_out_proj``, ``L['model'].action_time_mlp_*``
#   - the vision tower at ``L['model'].vlm_with_expert.vlm.vision_tower``
#     (or similar). For Phase A we only care about the action-expert +
#     action-head halves.
_COMPONENT_SUBSTRINGS: dict[str, tuple[str, ...]] = {
    "action_expert": (
        # `lm_expert` is the Gemma joint-expert under
        # ``vlm_with_expert.lm_expert.layers.N``. Do NOT include the
        # bare ``vlm_with_expert`` substring — that would also match
        # ``vlm_with_expert.vlm.model.vision_model.…`` (the vision
        # tower) and ``vlm_with_expert.vlm.model.text_model.…`` (the
        # frozen language model), which are NOT the action expert.
        "lm_expert", "gemma_expert",
    ),
    "action_head": (
        "action_in_proj", "action_out_proj",
        "action_time_mlp_in", "action_time_mlp_out", "action_head",
    ),
    "vision": ("vision_tower", "siglip"),
    "language": ("text_model", "language_model", "paligemma"),
}


# (_resolve_components / _action_expert_layer_index removed — we walk
# nn.Module trees instead of FX nodes; see _component_for_path /
# _action_expert_layer_index_for_path above.)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _UniqueContract:
    region_sig_hash: str
    contract: KernelContract
    occurrences: int
    components: tuple[str, ...]
    sample_fx_paths: tuple[str, ...]


@dataclass
class SubsetReport:
    total_linears: int
    total_passing_filter: int
    unique_after_dedup: int
    selected: int
    filter_label: str
    components_seen: dict[str, int] = field(default_factory=dict)


@dataclass
class SubsetSelector:
    """Enumerate unique matmul contracts from SmolVLA's nn.Linear modules.

    Args:
        target_name / target_class: Stamped into every emitted contract
            and into the region signature's ``target_class``.
        action_expert_layers: When set, only Linears whose module path
            contains ``layers.<i>`` for i in this set are kept (e.g.
            ``(0, 1, 2, 3)`` for Phase A). ``None`` means no layer filter.
        components: Substring components to keep. ``()`` means all.
        seq_len: M dimension stamped into each matmul contract. Doesn't
            change region_signature dedup since the per-Linear
            ``(in_features, out_features)`` carries the unique facts.
        quantize_to_i8: When True (default), every contract reports
            ``dtypes=("i8", "i8", "i32")`` — matches the Gemmini study
            assumption that we'll quantize matmuls. Set False to keep
            the model's native dtype.
        device: Torch device for the model loader.
    """

    target_name: str = "gemmini_mx"
    target_class: str = "gemmini_mx"
    action_expert_layers: tuple[int, ...] | None = (0, 1, 2, 3)
    components: tuple[str, ...] = ("action_expert", "action_head")
    seq_len: int = DEFAULT_SEQ_LEN
    quantize_to_i8: bool = True
    device: str = "cpu"

    def load(self) -> Any:
        """Return the SmolVLA wrapper module (no FX capture)."""
        examples_dir = Path(__file__).resolve().parents[1] / "examples"
        if examples_dir.is_dir():
            sys.path.insert(0, str(examples_dir))
        from models.smolvla_wrapper import load_smolvla  # type: ignore[import-not-found]

        wrapper, flat_inputs, num_cams = load_smolvla(device=self.device)
        logger.info(
            "loaded SmolVLA: %d params, %d inputs, %d cams",
            sum(p.numel() for p in wrapper.parameters()),
            len(flat_inputs),
            num_cams,
        )
        return wrapper

    def enumerate_unique_contracts(
        self,
        model: nn.Module,
    ) -> tuple[list[_UniqueContract], SubsetReport]:
        """Walk every nn.Linear under ``model``, synthesize one contract,
        dedup by region signature, return the unique set + a report.
        """
        total_linears = 0
        total_passing = 0
        per_hash: dict[str, _UniqueContract] = {}
        components_seen: Counter[str] = Counter()

        for name, lin in _iter_linear_modules(model):
            total_linears += 1
            mod_components = _component_for_path(name)
            for c in mod_components:
                components_seen[c] += 1

            if self.components:
                if not (mod_components & set(self.components)):
                    continue
            if self.action_expert_layers is not None and "action_expert" in mod_components:
                layer = _action_expert_layer_index_for_path(name)
                if layer is not None and layer not in self.action_expert_layers:
                    continue

            total_passing += 1
            contract = _synthesize_contract_from_linear(
                name=name,
                module=lin,
                target_class=self.target_class,
                seq_len=self.seq_len,
                quantize_to_i8=self.quantize_to_i8,
            )
            sig = self._region_signature(contract)
            h = hash_region_signature(sig)
            if h in per_hash:
                prior = per_hash[h]
                per_hash[h] = _UniqueContract(
                    region_sig_hash=h,
                    contract=prior.contract,
                    occurrences=prior.occurrences + 1,
                    components=prior.components,
                    sample_fx_paths=(
                        prior.sample_fx_paths + (name,)
                        if len(prior.sample_fx_paths) < 4
                        else prior.sample_fx_paths
                    ),
                )
            else:
                per_hash[h] = _UniqueContract(
                    region_sig_hash=h,
                    contract=contract,
                    occurrences=1,
                    components=tuple(sorted(mod_components)),
                    sample_fx_paths=(name,),
                )

        unique = sorted(per_hash.values(), key=lambda u: -u.occurrences)
        report = SubsetReport(
            total_linears=total_linears,
            total_passing_filter=total_passing,
            unique_after_dedup=len(unique),
            selected=0,
            filter_label=self._filter_label(),
            components_seen=dict(components_seen),
        )
        return unique, report

    def select_subset(
        self,
        unique: list[_UniqueContract],
        *,
        limit: int | None,
    ) -> list[_UniqueContract]:
        if limit is None:
            return unique
        return unique[:limit]

    def save(
        self,
        selected: list[_UniqueContract],
        report: SubsetReport,
        *,
        out_dir: Path,
    ) -> Path:
        """Persist contracts + a manifest to disk."""
        out_dir.mkdir(parents=True, exist_ok=True)
        contracts_dir = out_dir / "contracts"
        contracts_dir.mkdir(exist_ok=True)
        manifest_rows = []
        for entry in selected:
            path = contracts_dir / f"{entry.region_sig_hash}.json"
            path.write_text(
                json.dumps(
                    {
                        "region_sig_hash": entry.region_sig_hash,
                        "occurrences": entry.occurrences,
                        "components": list(entry.components),
                        "sample_fx_paths": list(entry.sample_fx_paths),
                        "contract": _serialise_contract(entry.contract),
                    },
                    indent=2,
                )
            )
            manifest_rows.append(
                {
                    "region_sig_hash": entry.region_sig_hash,
                    "occurrences": entry.occurrences,
                    "components": list(entry.components),
                    "op_family": entry.contract.op_family,
                    "dtypes": list(entry.contract.dtypes),
                    "input_shapes": [list(s) for s in entry.contract.input_shapes],
                    "output_shapes": [list(s) for s in entry.contract.output_shapes],
                    "path": str(path.relative_to(out_dir)),
                }
            )
        report = SubsetReport(
            total_linears=report.total_linears,
            total_passing_filter=report.total_passing_filter,
            unique_after_dedup=report.unique_after_dedup,
            selected=len(selected),
            filter_label=report.filter_label,
            components_seen=report.components_seen,
        )
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "report": asdict(report),
                    "target_name": self.target_name,
                    "target_class": self.target_class,
                    "filter": {
                        "components": list(self.components),
                        "action_expert_layers": list(self.action_expert_layers or ()),
                    },
                    "contracts": manifest_rows,
                },
                indent=2,
            )
        )
        return manifest_path

    # ---- helpers ----

    def _region_signature(self, contract: KernelContract) -> RegionSignature:
        # Flatten all input + output dims into the shape_class vector.
        dims: list[int] = []
        for s in (*contract.input_shapes, *contract.output_shapes):
            dims.extend(int(d) for d in s)
        return make_region_signature(
            op_family=contract.op_family,
            dtype=contract.dtypes[0] if contract.dtypes else "unknown",
            layout=contract.layout,
            dims=dims,
            target_class=self.target_class,
        )

    def _filter_label(self) -> str:
        parts: list[str] = []
        if self.components:
            parts.append("components=" + ",".join(self.components))
        if self.action_expert_layers is not None:
            parts.append(
                "action_expert_layers=" + ",".join(str(i) for i in self.action_expert_layers)
            )
        return "; ".join(parts) or "no_filter"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _serialise_contract(c: KernelContract) -> dict[str, Any]:
    return {
        "region_id": c.region_id,
        "op_family": c.op_family,
        "input_shapes": [list(s) for s in c.input_shapes],
        "output_shapes": [list(s) for s in c.output_shapes],
        "dtypes": list(c.dtypes),
        "layout": c.layout,
        "target_name": c.target_name,
        "hardware_key": c.hardware_key,
        "objective": c.objective,
        "constraints": dict(c.constraints),
        "provider_hints": dict(c.provider_hints),
    }


def load_contracts(manifest_path: Path) -> list[KernelContract]:
    """Read back contracts from a saved manifest."""
    body = json.loads(manifest_path.read_text())
    out: list[KernelContract] = []
    root = manifest_path.parent
    for row in body.get("contracts", []):
        path = root / row["path"]
        if not path.exists():
            continue
        entry = json.loads(path.read_text())
        c = entry["contract"]
        out.append(
            KernelContract(
                region_id=c.get("region_id", ""),
                op_family=c["op_family"],
                input_shapes=tuple(tuple(s) for s in c["input_shapes"]),
                output_shapes=tuple(tuple(s) for s in c["output_shapes"]),
                dtypes=tuple(c["dtypes"]),
                layout=c.get("layout", "row_major"),
                target_name=c.get("target_name", ""),
                hardware_key=c.get("hardware_key", ""),
                objective=c.get("objective", "latency"),
                constraints=dict(c.get("constraints", {})),
                provider_hints=dict(c.get("provider_hints", {})),
            )
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_out_dir() -> Path:
    return Path(
        os.environ.get(
            "XPU_RT_SMOLVLA_SUBSET_DIR",
            str(Path(__file__).resolve().parents[0] / ".xpu_rt" / "benchmarks" / "smolvla_subset"),
        )
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Capture SmolVLA + enumerate unique kernel contracts.")
    parser.add_argument("--limit", type=int, default=25, help="Max unique contracts to keep (0 = all).")
    parser.add_argument("--out", type=Path, default=_default_out_dir())
    parser.add_argument("--target", default="gemmini_mx")
    parser.add_argument(
        "--components",
        default="action_expert,action_head",
        help="Comma-separated component substrings; pass '' for no filter.",
    )
    parser.add_argument(
        "--layers",
        default="0,1,2,3",
        help="Action-expert layer indices to keep; pass '' for no filter.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    components = tuple(c for c in args.components.split(",") if c)
    layers = (
        tuple(int(x) for x in args.layers.split(",") if x)
        if args.layers
        else None
    )
    selector = SubsetSelector(
        target_name=args.target,
        target_class=args.target,
        action_expert_layers=layers,
        components=components,
        device=args.device,
    )
    wrapper = selector.load()
    unique, report = selector.enumerate_unique_contracts(wrapper)
    selected = selector.select_subset(unique, limit=args.limit if args.limit > 0 else None)
    final_report = SubsetReport(
        total_linears=report.total_linears,
        total_passing_filter=report.total_passing_filter,
        unique_after_dedup=report.unique_after_dedup,
        selected=len(selected),
        filter_label=report.filter_label,
        components_seen=report.components_seen,
    )
    manifest = selector.save(selected, final_report, out_dir=args.out)

    print(json.dumps({
        "manifest": str(manifest),
        "report": asdict(final_report),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
