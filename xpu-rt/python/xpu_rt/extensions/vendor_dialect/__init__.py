"""Third-party MLIR dialect integration for CompGen.

This subpackage lets a user point CompGen at a vendor MLIR repo
(CUDA Tile IR, Hexagon MLIR, ...) and drive end-to-end compilation
through a generated user-space adapter.

The public surface is small by design:

* :func:`scan_repo` — deterministic repo walker (inputs)
* :class:`VendorDialectDescriptor` — frozen integration spec
* :func:`scaffold_package` — render a user-space adapter package
* :class:`VendorDialectAdapter` — base class user packages subclass
* :func:`register_adapter`, :func:`get_adapter` — runtime registry
"""

from __future__ import annotations

from compgen.extensions.vendor_dialect.adapter import LoweringResult, VendorDialectAdapter
from compgen.extensions.vendor_dialect.builtins import (
    list_builtin_adapters,
    make_builtin_adapter,
    register_builtin_adapter,
)
from compgen.extensions.vendor_dialect.descriptor import (
    BundlePlan,
    CompileEntry,
    LoweringStrategy,
    OpEntry,
    VendorDialectDescriptor,
    VerificationPlan,
)
from compgen.extensions.vendor_dialect.registry import (
    VendorAdapterRegistry,
    adapters_for_target,
    available_adapters,
    get_adapter,
    register_adapter,
    reset_registry,
)
from compgen.extensions.vendor_dialect.scaffold import (
    TEMPLATE_PACK_ROOT,
    ScaffoldResult,
    scaffold_package,
)
from compgen.extensions.vendor_dialect.scanner import ScanResult, TdOp, scan_repo

__all__ = [
    "BundlePlan",
    "CompileEntry",
    "LoweringResult",
    "LoweringStrategy",
    "OpEntry",
    "ScaffoldResult",
    "ScanResult",
    "TEMPLATE_PACK_ROOT",
    "TdOp",
    "VendorAdapterRegistry",
    "VendorDialectAdapter",
    "VendorDialectDescriptor",
    "VerificationPlan",
    "adapters_for_target",
    "available_adapters",
    "get_adapter",
    "list_builtin_adapters",
    "make_builtin_adapter",
    "register_adapter",
    "register_builtin_adapter",
    "reset_registry",
    "scaffold_package",
    "scan_repo",
]
