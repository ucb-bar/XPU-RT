"""Deterministic vendor-repo scanner.

Walks a third-party MLIR dialect repo and collects the facts a downstream
LLM pass needs to propose an integration ``VendorDialectDescriptor``:

* README / docs / tutorials
* CMake / build files
* TableGen ``.td`` files (with a light parse for ops)
* Python bindings entry points
* CLI tool names from ``tools/`` subdirectories
* Test examples that show end-to-end usage

The scanner is pure filesystem work — no LLM calls, no shelling out. The
output is stable across runs, so the exploration agent always has the
same ground truth to reason over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TdOp:
    """A single op extracted from a TableGen file (best-effort)."""

    name: str
    source_file: str
    summary: str = ""


@dataclass(frozen=True)
class ScanResult:
    """Everything the scanner discovered about a vendor repo.

    The fields are intentionally verbose; the exploration LLM pass picks
    the subset it cares about and produces a :class:`VendorDialectDescriptor`.
    """

    repo_path: str
    readme_text: str = ""
    readme_path: str = ""
    cmake_files: tuple[str, ...] = ()
    td_files: tuple[str, ...] = ()
    td_ops: tuple[TdOp, ...] = ()
    python_bindings_paths: tuple[str, ...] = ()
    cli_tools: tuple[str, ...] = ()
    test_examples: tuple[str, ...] = ()
    tutorial_docs: tuple[str, ...] = ()
    license_text: str = ""
    license_spdx: str = ""
    dialect_names: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Compact summary (counts + first-line headers) for LLM prompts."""
        return {
            "repo_path": self.repo_path,
            "readme_present": bool(self.readme_text),
            "num_cmake_files": len(self.cmake_files),
            "num_td_files": len(self.td_files),
            "num_td_ops": len(self.td_ops),
            "num_cli_tools": len(self.cli_tools),
            "num_python_bindings": len(self.python_bindings_paths),
            "num_test_examples": len(self.test_examples),
            "num_tutorial_docs": len(self.tutorial_docs),
            "dialect_names": list(self.dialect_names),
            "license_spdx": self.license_spdx,
        }


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #


_MAX_FILES_PER_CATEGORY = 200
_MAX_READ_BYTES = 256 * 1024


def scan_repo(repo_path: str | Path) -> ScanResult:
    """Walk ``repo_path`` and collect vendor-repo facts.

    Args:
        repo_path: Path to a third-party MLIR dialect repository.

    Returns:
        A frozen :class:`ScanResult` summarising what was found.

    Raises:
        FileNotFoundError: If ``repo_path`` does not exist.
        NotADirectoryError: If ``repo_path`` is not a directory.
    """
    root = Path(repo_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"vendor repo not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"vendor repo path is not a directory: {root}")

    log.info("vendor_scan.start", path=str(root))

    readme_text, readme_path = _read_readme(root)
    license_text, license_spdx = _read_license(root)
    cmake_files = _collect_by_glob(root, ["**/CMakeLists.txt", "**/*.cmake"])
    td_files = _collect_by_glob(root, ["**/*.td"])
    td_ops = _parse_td_ops(root, td_files)
    python_bindings = _collect_by_glob(root, ["**/python/**/*.py", "**/bindings/**/*.py"])
    cli_tools = _collect_cli_tools(root)
    test_examples = _collect_by_glob(root, ["test/**/*.py", "tests/**/*.py", "test/**/*.mlir", "tests/**/*.mlir"])
    tutorial_docs = _collect_by_glob(root, ["docs/**/*.md", "tutorials/**/*.md"])
    dialect_names = _extract_dialect_names(root, td_files, td_ops)

    result = ScanResult(
        repo_path=str(root),
        readme_text=readme_text,
        readme_path=readme_path,
        cmake_files=tuple(cmake_files),
        td_files=tuple(td_files),
        td_ops=tuple(td_ops),
        python_bindings_paths=tuple(python_bindings),
        cli_tools=tuple(cli_tools),
        test_examples=tuple(test_examples),
        tutorial_docs=tuple(tutorial_docs),
        license_text=license_text,
        license_spdx=license_spdx,
        dialect_names=tuple(dialect_names),
    )
    log.info("vendor_scan.done", **result.summary())
    return result


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _read_readme(root: Path) -> tuple[str, str]:
    for name in ("README.md", "README.rst", "README", "README.txt"):
        p = root / name
        if p.is_file():
            return _read_bounded(p), str(p.relative_to(root))
    return "", ""


def _read_license(root: Path) -> tuple[str, str]:
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
        p = root / name
        if p.is_file():
            text = _read_bounded(p)
            return text, _infer_spdx(text)
    return "", ""


def _infer_spdx(text: str) -> str:
    lo = text.lower()
    if "apache license" in lo and "2.0" in lo:
        if "llvm exceptions" in lo:
            return "Apache-2.0-WITH-LLVM-exception"
        return "Apache-2.0"
    if "bsd 3-clause" in lo or "redistributions of source code must retain" in lo:
        return "BSD-3-Clause"
    if "mit license" in lo:
        return "MIT"
    if "gnu general public license" in lo:
        return "GPL"
    return ""


def _collect_by_glob(root: Path, patterns: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if not p.is_file():
                continue
            rel = str(p.relative_to(root))
            if rel in seen:
                continue
            seen.add(rel)
            out.append(rel)
            if len(out) >= _MAX_FILES_PER_CATEGORY:
                return out
    return out


def _collect_cli_tools(root: Path) -> list[str]:
    """CLI tools are most reliably named by a top-level ``tools/<name>`` dir
    or by a ``CMakeLists.txt`` in ``bin/`` / ``tools/``. We collect both."""
    names: list[str] = []
    seen: set[str] = set()
    for candidate_dir in ("tools", "bin"):
        d = root / candidate_dir
        if d.is_dir():
            for child in sorted(d.iterdir()):
                if child.is_dir() and (child / "CMakeLists.txt").is_file():
                    if child.name not in seen:
                        seen.add(child.name)
                        names.append(child.name)
                elif child.is_file() and child.suffix in {".cpp", ".cc"}:
                    stem = child.stem
                    if stem not in seen:
                        seen.add(stem)
                        names.append(stem)
    # Heuristic: qcom_hexagon_backend/bin/linalg-hexagon-opt.cpp style.
    for p in root.rglob("*-opt.cpp"):
        stem = p.stem
        if stem not in seen:
            seen.add(stem)
            names.append(stem)
    for p in root.rglob("*-translate.cpp"):
        stem = p.stem
        if stem not in seen:
            seen.add(stem)
            names.append(stem)
    return names


_TD_OP_RE = re.compile(
    r"def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
    r"(?P<kind>\w*Op[A-Za-z0-9_]*)\b[^{]*\{(?P<body>[^}]*)\}",
    re.DOTALL,
)
_TD_SUMMARY_RE = re.compile(r'let\s+summary\s*=\s*"([^"]*)"')


def _parse_td_ops(root: Path, td_files: list[str]) -> list[TdOp]:
    """Best-effort TableGen op extractor.

    TableGen is a real language; we do NOT aim to fully parse it. We
    just pull op-looking ``def Foo : SomeOp<...> { ... }`` blocks and
    an optional ``let summary = "..."`` string. The exploration agent
    uses this as a hint, not an oracle.
    """
    ops: list[TdOp] = []
    for rel in td_files:
        text = _read_bounded(root / rel)
        if not text:
            continue
        for m in _TD_OP_RE.finditer(text):
            body = m.group("body")
            summary_m = _TD_SUMMARY_RE.search(body)
            summary = summary_m.group(1) if summary_m else ""
            ops.append(TdOp(name=m.group("name"), source_file=rel, summary=summary))
            if len(ops) >= _MAX_FILES_PER_CATEGORY * 4:
                return ops
    return ops


_DIALECT_NAME_RE = re.compile(
    r'def\s+\w+Dialect\s*:\s*Dialect\s*\{[^}]*?let\s+name\s*=\s*"([^"]+)"',
    re.DOTALL,
)


def _extract_dialect_names(root: Path, td_files: list[str], ops: list[TdOp]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for rel in td_files:
        text = _read_bounded(root / rel)
        if not text:
            continue
        for m in _DIALECT_NAME_RE.finditer(text):
            nm = m.group(1)
            if nm not in seen:
                seen.add(nm)
                names.append(nm)
    if not names and ops:
        # Fallback: take parent dir of the first op file as a guess.
        names.append(Path(ops[0].source_file).parent.name)
    return names


def _read_bounded(p: Path) -> str:
    try:
        with p.open("rb") as fh:
            return fh.read(_MAX_READ_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


__all__ = ["ScanResult", "TdOp", "scan_repo"]
