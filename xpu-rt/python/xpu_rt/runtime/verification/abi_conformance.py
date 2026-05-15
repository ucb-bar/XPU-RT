"""ABI-conformance check (D6, Phase G).

Mechanical post-emit gate: the emitted Layer-1 plan executor calls
only ``cg_rt_*`` (libcompgen_rt) and ``compgen_kernel_*`` (
kernel pack) externs.  Anything else — ``cudaMalloc``,
``cuLaunchKernel``, ``hipMalloc``, ``vkCmdDispatch``, etc. — is a
direct vendor primitive bypassing the HAL; the gate rejects it.

The scanner strips C/C++ comments and string literals before
identifying call sites (so prose like ``"foo failed (cuda)"`` and
comments mentioning the ABI don't false-positive). Calls inside
preprocessor macros that expand to ``cg_rt_*`` are accepted; the
gate runs on the raw source, not the pre-processed translation
unit, by design — the emitters never emit user macros.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from compgen.runtime.errors import AbiConformanceError


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_STRING_RE = re.compile(r'"([^"\\]|\\.)*"')
_CALL_RE = re.compile(r"(?<![.>:])\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

# Symbols the emit may invoke that aren't ``cg_rt_*`` / ``compgen_kernel_*``.
# Anything else is a violation. Keep this list short and audited.
_BUILTIN_ALLOW: frozenset[str] = frozenset({
    "sizeof", "if", "for", "while", "return", "switch", "case",
    "compgen_run", "compgen_select_plan_ref",
    "compgen_dispatch_run",
    # C++ casts the emit needs.
    "static_cast", "reinterpret_cast", "const_cast",
})


# Tighter blocklist: if these appear we name them explicitly in the
# error so operators know what to fix.
_VENDOR_BLOCK: frozenset[str] = frozenset({
    "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaLaunchKernel",
    "cuLaunchKernel", "cuMemAlloc", "cuMemcpyHtoD",
    "hipMalloc", "hipFree", "hipMemcpy", "hipLaunchKernel",
    "vkCreateBuffer", "vkCmdDispatch", "vkQueueSubmit",
    "clEnqueueNDRangeKernel", "clCreateBuffer",
})


@dataclass(frozen=True)
class AbiConformanceReport:
    overall: str  # "pass" | "fail"
    emit_path: str
    called_symbols: tuple[str, ...] = field(default_factory=tuple)
    forbidden_symbols: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "runtime_abi_conformance_v1",
            "overall": self.overall,
            "emit_path": self.emit_path,
            "called_symbols": sorted(self.called_symbols),
            "forbidden_symbols": sorted(self.forbidden_symbols),
        }


def _strip(src: str) -> str:
    src = _BLOCK_COMMENT_RE.sub("", src)
    src = _LINE_COMMENT_RE.sub("", src)
    return _STRING_RE.sub('""', src)


def check_abi_conformance(
    emit_path: Path,
    *,
    raise_on_fail: bool = True,
    extra_allowlist: frozenset[str] = frozenset(),
) -> AbiConformanceReport:
    """Scan ``emit_path`` (a .c / .cpp file) for forbidden externs.

    Returns a typed :class:`AbiConformanceReport`. Raises
    :class:`AbiConformanceError` on failure when ``raise_on_fail``.
    """
    emit_path = Path(emit_path).resolve()
    if not emit_path.exists():
        raise FileNotFoundError(f"emit not found: {emit_path}")
    src = _strip(emit_path.read_text())
    called: set[str] = {m.group(1) for m in _CALL_RE.finditer(src)}
    forbidden: list[str] = []
    for name in called:
        if name.startswith("cg_rt_"):
            continue
        if name.startswith("compgen_kernel_"):
            continue
        if name in _BUILTIN_ALLOW:
            continue
        if name in extra_allowlist:
            continue
        # libc-ish names we don't actually call but might appear
        # via macros; tighten as needed.
        if name in ("memset", "memcpy"):
            continue
        forbidden.append(name)

    overall = "pass" if not forbidden else "fail"
    report = AbiConformanceReport(
        overall=overall,
        emit_path=str(emit_path),
        called_symbols=tuple(sorted(called)),
        forbidden_symbols=tuple(sorted(forbidden)),
    )
    if overall == "fail" and raise_on_fail:
        # Prefer the vendor-block list in the message when present.
        blocked = tuple(s for s in forbidden if s in _VENDOR_BLOCK)
        named = blocked or tuple(forbidden)
        raise AbiConformanceError(named, emit_path=str(emit_path))
    return report
