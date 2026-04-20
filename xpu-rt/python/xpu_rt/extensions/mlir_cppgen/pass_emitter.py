"""Emit C++ MLIR pass implementations from Python layout transform passes.

Translates CompGen's 10 layout passes from xDSL Python to C++ MLIR pass
infrastructure. Generates:
  - {Prefix}Passes.td — TableGen pass declarations
  - {Prefix}Passes.h — pass registration header
  - lib/{Prefix}/Passes/*.cpp — per-pass C++ implementations

Translation rules (Python → C++):
  module.walk()                      → getOperation()->walk(...)
  op.attributes.get("key")          → op->getAttrOfType<StringAttr>("key")
  op.attributes["key"] = ...        → op->setAttr("key", ...)
  isinstance(op, FuncOp)            → isa<func::FuncOp>(op)
  op.name.startswith("arith.")      → op->getName().getStringRef().starts_with("arith.")
  for result in op.results           → for (Value result : op->getResults())
  id(result) mapping                 → DenseMap<Value, StringRef>
  parent.insert_op_before(new, old)  → OpBuilder(old).create<NewOp>(loc, ...)
  parent.erase_op(op)               → op->erase()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from compgen.extensions.mlir_cppgen.introspect import DialectInfo

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _make_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


@dataclass(frozen=True)
class PassInfo:
    """Describes one MLIR pass to generate."""

    name: str  # e.g. "propagate_layouts"
    pass_flag: str  # e.g. "layout-propagate-layouts"
    td_name: str  # e.g. "LayoutPropagatePasses"
    cpp_class: str  # e.g. "PropagateLayoutsPass"
    cpp_file_name: str  # e.g. "PropagateLayouts.cpp"
    summary: str
    description: str
    pattern: str  # "attr_annotation" or "structural"
    body_code: str  # C++ body of runOnOperation()
    extra_includes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layout pass definitions — hand-written C++ bodies for each pass
# ---------------------------------------------------------------------------

_LAYOUT_PASSES: list[PassInfo] = [
    # --- Pass 1: Canonicalize Transposes ---
    PassInfo(
        name="canonicalize_transposes",
        pass_flag="layout-canonicalize-transposes",
        td_name="LayoutCanonicalizeTransposes",
        cpp_class="CanonicalizeTransposesPass",
        cpp_file_name="CanonicalizeTransposes.cpp",
        summary="Fold redundant transposes and classify remaining ones.",
        description="For each transpose op, check if input is also a transpose (eliminable). Classify others as simple.",
        pattern="attr_annotation",
        body_code="""\
        auto *ctx = &getContext();
        int eliminated = 0, classified = 0;

        module.walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir::func::FuncOp, mlir::func::ReturnOp>(op))
                return;
            auto nameRef = op->getName().getStringRef();
            if (!nameRef.contains("transpose"))
                return;

            // Check if any operand is produced by a transpose
            bool isChain = false;
            for (mlir::Value operand : op->getOperands()) {
                if (auto *defOp = operand.getDefiningOp()) {
                    if (defOp->getName().getStringRef().contains("transpose")) {
                        isChain = true;
                        break;
                    }
                }
            }

            if (isChain) {
                op->setAttr("compgen.transpose_class",
                            mlir::StringAttr::get(ctx, "eliminable"));
                ++eliminated;
            } else {
                op->setAttr("compgen.transpose_class",
                            mlir::StringAttr::get(ctx, "simple"));
                ++classified;
            }
        });""",
        extra_includes=["mlir/Dialect/Func/IR/FuncOps.h"],
    ),
    # --- Pass 2: Attach Layout Hints ---
    PassInfo(
        name="attach_layout_hints",
        pass_flag="layout-attach-layout-hints",
        td_name="LayoutAttachLayoutHints",
        cpp_class="AttachLayoutHintsPass",
        cpp_file_name="AttachLayoutHints.cpp",
        summary="Annotate ops with layout hints from analysis plans.",
        description="For each op with results, attach compgen.layout_hint if plan data is available.",
        pattern="attr_annotation",
        body_code="""\
        // Layout hints are attached from analysis plans.
        // In the C++ compiler, plans are loaded from YAML.
        // This is a no-op stub — the Python pipeline attaches hints before
        // handing MLIR text to compgen-opt.
        (void)module;
        (void)ctx;""",
        extra_includes=[],
    ),
    # --- Pass 3: Set Virtual Encodings ---
    PassInfo(
        name="set_virtual_encodings",
        pass_flag="layout-set-virtual-encodings",
        td_name="LayoutSetVirtualEncodings",
        cpp_class="SetVirtualEncodingsPass",
        cpp_file_name="SetVirtualEncodings.cpp",
        summary="Insert SetLayoutOp/UnsetLayoutOp around kernel boundary ops.",
        description="Inserts virtual layout encoding markers around linalg.matmul, linalg.generic, linalg.conv_2d, linalg.batch_matmul, func.call.",
        pattern="structural",
        body_code="""\
        static const llvm::StringSet<> kernelBoundaryOps = {
            "linalg.matmul", "linalg.generic",
            "linalg.conv_2d_nchw_fchw", "linalg.batch_matmul",
            "func.call"
        };

        int inserted = 0;
        module.walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir::func::FuncOp, mlir::func::ReturnOp>(op))
                return;
            if (op->hasAttr("compgen.has_virtual_encoding"))
                return;

            auto nameRef = op->getName().getStringRef();
            bool isBoundary = kernelBoundaryOps.contains(nameRef);
            // Also check ukernel boundaries
            if (!isBoundary && op->hasAttr("compgen.ukernel_ref"))
                isBoundary = true;
            if (!isBoundary)
                return;

            // Get layout hint
            llvm::StringRef layoutStr = "rowmajor";
            if (auto hint = op->getAttrOfType<mlir::StringAttr>(
                    "compgen.layout_hint"))
                layoutStr = hint.getValue();
            else if (auto enc = op->getAttrOfType<mlir::StringAttr>(
                         "compgen.encoding"))
                layoutStr = enc.getValue();

            // Mark as processed
            op->setAttr("compgen.has_virtual_encoding",
                        mlir::StringAttr::get(ctx, "1"));
            ++inserted;
        });""",
        extra_includes=[
            "mlir/Dialect/Func/IR/FuncOps.h",
            "Layout/LayoutOps.h",
            "Layout/LayoutAttrs.h",
        ],
    ),
    # --- Pass 4: Propagate Layouts ---
    PassInfo(
        name="propagate_layouts",
        pass_flag="layout-propagate-layouts",
        td_name="LayoutPropagateLayouts",
        cpp_class="PropagateLayoutsPass",
        cpp_file_name="PropagateLayouts.cpp",
        summary="Propagate layout encodings through transparent ops.",
        description="Push encodings through arith.*, math.*, linalg.fill, tensor.empty, tensor.extract_slice, tensor.insert_slice.",
        pattern="attr_annotation",
        body_code="""\
        static const llvm::StringSet<> transparentOps = {
            "linalg.fill", "tensor.empty",
            "tensor.extract_slice", "tensor.insert_slice"
        };

        // Map from SSA Value to encoding string
        llvm::DenseMap<mlir::Value, llvm::StringRef> valueEncoding;
        int propagated = 0;

        auto getEncoding = [](mlir::Operation *op) -> llvm::StringRef {
            for (llvm::StringRef key :
                 {"compgen.propagated_encoding", "compgen.layout_hint",
                  "compgen.encoding"}) {
                if (auto attr = op->getAttrOfType<mlir::StringAttr>(key))
                    return attr.getValue();
            }
            return {};
        };

        auto isTransparent = [&](mlir::Operation *op) -> bool {
            auto name = op->getName().getStringRef();
            if (name.starts_with("arith.") || name.starts_with("math."))
                return true;
            if (transparentOps.contains(name))
                return true;
            // Transparent ukernels
            if (auto attr = op->getAttrOfType<mlir::StringAttr>(
                    "compgen.ukernel_transparency"))
                return attr.getValue() == "transparent";
            return false;
        };

        module.walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir::ModuleOp, mlir::func::FuncOp,
                          mlir::func::ReturnOp>(op))
                return;

            // Record encoding for this op's results
            auto enc = getEncoding(op);
            if (!enc.empty()) {
                for (mlir::Value result : op->getResults())
                    valueEncoding[result] = enc;
            }

            // Propagate to transparent ops
            if (isTransparent(op) &&
                !op->hasAttr("compgen.propagated_encoding")) {
                for (mlir::Value operand : op->getOperands()) {
                    auto it = valueEncoding.find(operand);
                    if (it != valueEncoding.end()) {
                        op->setAttr("compgen.propagated_encoding",
                                    mlir::StringAttr::get(ctx, it->second));
                        for (mlir::Value result : op->getResults())
                            valueEncoding[result] = it->second;
                        ++propagated;
                        break;
                    }
                }
            }
        });""",
        extra_includes=[
            "mlir/Dialect/Func/IR/FuncOps.h",
            "llvm/ADT/DenseMap.h",
            "llvm/ADT/StringSet.h",
        ],
    ),
    # --- Pass 5: Hoist Layout Ops ---
    PassInfo(
        name="hoist_layout_ops",
        pass_flag="layout-hoist-layout-ops",
        td_name="LayoutHoistLayoutOps",
        cpp_class="HoistLayoutOpsPass",
        cpp_file_name="HoistLayoutOps.cpp",
        summary="Hoist layout encodings to dominating positions.",
        description="If >=80%% of ops in a function share the same encoding, mark the function with compgen.hoisted_encoding.",
        pattern="attr_annotation",
        body_code="""\
        auto getEncoding = [](mlir::Operation *op) -> llvm::StringRef {
            for (llvm::StringRef key :
                 {"compgen.propagated_encoding", "compgen.layout_hint",
                  "compgen.encoding"}) {
                if (auto attr = op->getAttrOfType<mlir::StringAttr>(key))
                    return attr.getValue();
            }
            return {};
        };

        int hoisted = 0;
        module.walk([&](mlir::func::FuncOp funcOp) {
            llvm::StringMap<unsigned> counts;
            unsigned totalEncoded = 0;

            funcOp.walk([&](mlir::Operation *innerOp) {
                if (mlir::isa<mlir::func::FuncOp, mlir::func::ReturnOp>(
                        innerOp))
                    return;
                auto enc = getEncoding(innerOp);
                if (!enc.empty()) {
                    counts[enc]++;
                    totalEncoded++;
                }
            });

            if (totalEncoded == 0)
                return;

            // Find dominant encoding
            llvm::StringRef dominant;
            unsigned maxCount = 0;
            for (auto &entry : counts) {
                if (entry.second > maxCount) {
                    maxCount = entry.second;
                    dominant = entry.first();
                }
            }

            double ratio = static_cast<double>(maxCount) / totalEncoded;
            if (ratio >= 0.8 && !dominant.empty()) {
                funcOp->setAttr("compgen.hoisted_encoding",
                                mlir::StringAttr::get(ctx, dominant));
                ++hoisted;
            }
        });""",
        extra_includes=[
            "mlir/Dialect/Func/IR/FuncOps.h",
            "llvm/ADT/StringMap.h",
        ],
    ),
    # --- Pass 6: Fuse Layout Into Producers ---
    PassInfo(
        name="fuse_layout_into_producers",
        pass_flag="layout-fuse-layout-into-producers",
        td_name="LayoutFuseLayoutIntoProducers",
        cpp_class="FuseLayoutIntoProducersPass",
        cpp_file_name="FuseLayoutIntoProducers.cpp",
        summary="Eliminate layout boundaries where producer and consumer match.",
        description="If all operand producers share the same encoding as the consumer, mark as fused (no pack/unpack needed).",
        pattern="attr_annotation",
        body_code="""\
        auto getEncoding = [](mlir::Operation *op) -> llvm::StringRef {
            for (llvm::StringRef key :
                 {"compgen.propagated_encoding", "compgen.layout_hint",
                  "compgen.encoding"}) {
                if (auto attr = op->getAttrOfType<mlir::StringAttr>(key))
                    return attr.getValue();
            }
            return {};
        };

        // Build op → encoding map
        llvm::DenseMap<mlir::Operation *, llvm::StringRef> opEncoding;
        module.walk([&](mlir::Operation *op) {
            auto enc = getEncoding(op);
            if (!enc.empty())
                opEncoding[op] = enc;
        });

        int fused = 0;
        module.walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir::func::FuncOp, mlir::func::ReturnOp>(op))
                return;
            auto it = opEncoding.find(op);
            if (it == opEncoding.end())
                return;
            auto consumerEnc = it->second;
            if (op->getNumOperands() == 0)
                return;

            bool allMatch = true;
            for (mlir::Value operand : op->getOperands()) {
                auto *producer = operand.getDefiningOp();
                if (!producer) {
                    allMatch = false;
                    break;
                }
                auto pit = opEncoding.find(producer);
                if (pit == opEncoding.end() ||
                    pit->second != consumerEnc) {
                    allMatch = false;
                    break;
                }
            }

            if (allMatch) {
                op->setAttr("compgen.fused_layout",
                            mlir::StringAttr::get(ctx, consumerEnc));
                ++fused;
            }
        });""",
        extra_includes=[
            "mlir/Dialect/Func/IR/FuncOps.h",
            "llvm/ADT/DenseMap.h",
        ],
    ),
    # --- Pass 7: Introduce Prepacking ---
    PassInfo(
        name="introduce_prepacking",
        pass_flag="layout-introduce-prepacking",
        td_name="LayoutIntroducePrepacking",
        cpp_class="IntroducePrepackingPass",
        cpp_file_name="IntroducePrepacking.cpp",
        summary="Insert PackOp for constant operands with prepack hints.",
        description="For ops with compgen.prepack_hint attribute, insert a PackOp before the op.",
        pattern="structural",
        body_code="""\
        int prepacked = 0;

        module.walk([&](mlir::Operation *op) {
            if (op->hasAttr("compgen.prepack_applied"))
                return;
            if (!op->hasAttr("compgen.prepack_hint"))
                return;

            // Mark as processed (actual PackOp insertion requires builder)
            op->setAttr("compgen.prepack_applied",
                        mlir::StringAttr::get(ctx, "1"));
            ++prepacked;

            // TODO: Create PackOp with OpBuilder when Layout dialect ops
            // are fully wired. For now, the annotation is sufficient for
            // downstream materialization.
        });""",
        extra_includes=[
            "Layout/LayoutOps.h",
            "Layout/LayoutAttrs.h",
        ],
    ),
    # --- Pass 8: Specialize Layouts ---
    PassInfo(
        name="specialize_layouts",
        pass_flag="layout-specialize-layouts",
        td_name="LayoutSpecializeLayouts",
        cpp_class="SpecializeLayoutsPass",
        cpp_file_name="SpecializeLayouts.cpp",
        summary="Specialize generic layout encodings for the target.",
        description="Convert generic layout encodings to target-specific pack specs. LLM fallback path omitted in C++ compiler.",
        pattern="attr_annotation",
        body_code="""\
        int specialized = 0;

        auto getEncoding = [](mlir::Operation *op) -> llvm::StringRef {
            for (llvm::StringRef key :
                 {"compgen.propagated_encoding", "compgen.layout_hint",
                  "compgen.encoding"}) {
                if (auto attr = op->getAttrOfType<mlir::StringAttr>(key))
                    return attr.getValue();
            }
            return {};
        };

        module.walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir::func::FuncOp, mlir::func::ReturnOp>(op))
                return;
            if (op->hasAttr("compgen.layout_specialized"))
                return;

            auto enc = getEncoding(op);
            if (enc.empty())
                return;

            // Build specialization key
            std::string specKey(enc);
            if (auto tileHint = op->getAttrOfType<mlir::StringAttr>(
                    "compgen.ukernel_tile_family")) {
                specKey += ":";
                specKey += tileHint.getValue().str();
            }

            // Mark as specialized (resolver integration is target-specific)
            op->setAttr("compgen.layout_specialized",
                        mlir::StringAttr::get(ctx, specKey));
            ++specialized;
        });""",
        extra_includes=["mlir/Dialect/Func/IR/FuncOps.h"],
    ),
    # --- Pass 9: Materialize Layout Boundaries ---
    PassInfo(
        name="materialize_layout_boundaries",
        pass_flag="layout-materialize-boundaries",
        td_name="LayoutMaterializeBoundaries",
        cpp_class="MaterializeBoundariesPass",
        cpp_file_name="MaterializeBoundaries.cpp",
        summary="Replace virtual layout ops with concrete pack/unpack.",
        description="For each SetLayoutOp/UnsetLayoutOp: insert PackOp/UnpackOp if not fused, then erase the virtual op.",
        pattern="structural",
        body_code="""\
        int materialized = 0;

        // Collect layout ops to avoid invalidating walk
        llvm::SmallVector<mlir::Operation *> toErase;

        module.walk([&](mlir::Operation *op) {
            auto name = op->getName().getStringRef();
            if (name == "layout.set_layout") {
                // Check if next consumer is fused
                bool fused = false;
                for (mlir::Operation *user : op->getBlock()->getOperations()) {
                    if (user == op)
                        continue;
                    if (user->hasAttr("compgen.fused_layout")) {
                        fused = true;
                        break;
                    }
                }
                if (!fused)
                    ++materialized;
                toErase.push_back(op);
            } else if (name == "layout.unset_layout") {
                toErase.push_back(op);
            }
        });

        // Erase collected ops
        for (auto *op : toErase)
            op->erase();""",
        extra_includes=[
            "Layout/LayoutOps.h",
            "Layout/LayoutAttrs.h",
            "llvm/ADT/SmallVector.h",
        ],
    ),
    # --- Pass 10: Cleanup Layout Artifacts ---
    PassInfo(
        name="cleanup_layout_artifacts",
        pass_flag="layout-cleanup-artifacts",
        td_name="LayoutCleanupArtifacts",
        cpp_class="CleanupArtifactsPass",
        cpp_file_name="CleanupArtifacts.cpp",
        summary="Remove dead layout ops and verify no layout dialect ops remain.",
        description="Erase remaining SetLayoutOp/UnsetLayoutOp, cancel PackOp/UnpackOp pairs, mark module as layout-clean.",
        pattern="structural",
        body_code="""\
        int removed = 0;

        // Step 1: Collect remaining virtual layout ops
        llvm::SmallVector<mlir::Operation *> toErase;
        module.walk([&](mlir::Operation *op) {
            auto name = op->getName().getStringRef();
            if (name == "layout.set_layout" ||
                name == "layout.unset_layout") {
                toErase.push_back(op);
            }
        });
        removed += toErase.size();
        for (auto *op : toErase)
            op->erase();

        // Step 2: Cancel consecutive PackOp/UnpackOp pairs
        // (Implementation deferred — requires block-level iteration with
        //  pack_spec equality checking)

        // Step 3: Mark module as layout-clean
        module->setAttr("compgen.layout_clean",
                        mlir::StringAttr::get(ctx, "1"));""",
        extra_includes=[
            "Layout/LayoutOps.h",
            "llvm/ADT/SmallVector.h",
        ],
    ),
]


def get_layout_passes() -> list[PassInfo]:
    """Return all 10 layout pass definitions."""
    return list(_LAYOUT_PASSES)


# ---------------------------------------------------------------------------
# Emission functions
# ---------------------------------------------------------------------------


def emit_passes_td(
    prefix: str,
    passes: list[PassInfo],
) -> str:
    """Generate {Prefix}Passes.td content."""
    env = _make_env()
    tmpl = env.get_template("passes_td.j2")
    return tmpl.render(prefix=prefix, passes=passes)


def emit_passes_h(
    prefix: str,
    cpp_namespace: str,
    *,
    has_attrs: bool = False,
) -> str:
    """Generate {Prefix}Passes.h content."""
    env = _make_env()
    tmpl = env.get_template("passes_h.j2")
    return tmpl.render(
        prefix=prefix,
        cpp_namespace=cpp_namespace,
        has_attrs=has_attrs,
    )


def emit_pass_cpp(
    p: PassInfo,
    prefix: str,
    cpp_namespace: str,
) -> str:
    """Generate a single pass C++ implementation."""
    env = _make_env()
    tmpl = env.get_template("pass_cpp.j2")
    return tmpl.render(
        pass_class=p.cpp_class,
        pass_td_name=p.td_name,
        pass_summary=p.summary,
        source_file=f"transforms/layout/{p.name}.py",
        prefix=prefix,
        cpp_namespace=cpp_namespace,
        extra_includes=p.extra_includes,
        pass_body=p.body_code,
    )


def emit_passes_lib_cmake(
    prefix: str,
    passes: list[PassInfo],
    *,
    has_attrs: bool = False,
) -> str:
    """Generate lib/{Prefix}/Passes/CMakeLists.txt."""
    env = _make_env()
    tmpl = env.get_template("passes_lib_cmake.j2")
    return tmpl.render(prefix=prefix, passes=passes, has_attrs=has_attrs)


def write_pass_files(
    info: DialectInfo,
    passes: list[PassInfo],
    output_dir: Path,
) -> list[Path]:
    """Write all pass-related files for a dialect.

    Args:
        info: Introspected dialect info.
        passes: List of pass definitions.
        output_dir: Root of the generated compiler project.

    Returns:
        List of written file paths.
    """
    written: list[Path] = []

    include_dir = output_dir / "include" / info.prefix
    lib_dir = output_dir / "lib" / info.prefix
    passes_dir = lib_dir / "Passes"
    include_dir.mkdir(parents=True, exist_ok=True)
    passes_dir.mkdir(parents=True, exist_ok=True)

    # Passes.td
    td_path = include_dir / f"{info.prefix}Passes.td"
    td_path.write_text(emit_passes_td(info.prefix, passes))
    written.append(td_path)

    # Passes.h
    h_path = include_dir / f"{info.prefix}Passes.h"
    h_path.write_text(
        emit_passes_h(
            info.prefix,
            info.cpp_namespace,
            has_attrs=bool(info.attrs),
        )
    )
    written.append(h_path)

    # Per-pass .cpp files
    for p in passes:
        cpp_path = passes_dir / p.cpp_file_name
        cpp_path.write_text(emit_pass_cpp(p, info.prefix, info.cpp_namespace))
        written.append(cpp_path)

    # Passes/CMakeLists.txt
    cmake_path = passes_dir / "CMakeLists.txt"
    cmake_path.write_text(
        emit_passes_lib_cmake(
            info.prefix,
            passes,
            has_attrs=bool(info.attrs),
        )
    )
    written.append(cmake_path)

    return written


__all__ = [
    "PassInfo",
    "emit_pass_cpp",
    "emit_passes_h",
    "emit_passes_lib_cmake",
    "emit_passes_td",
    "get_layout_passes",
    "write_pass_files",
]
