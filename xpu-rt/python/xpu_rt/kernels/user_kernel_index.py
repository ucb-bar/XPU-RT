"""M-62 — User-space kernel-provider discovery and indexing.

The dream's Section 7 calls for users to plug their own kernels into
the auction by pointing CompGen at a directory of kernel sources +
manifests. M-62 lands the filesystem indexing + locked-files audit
pattern, mirroring ``compgen.graph_compilation.extension_verify``.

User-supplied directory layout::

    ~/my-kernels/
      matmul_f32_host_cpu/
        kernel_manifest.yaml      # signature declaration
        kernel.c                  # actual kernel source

``kernel_manifest.yaml`` schema (``user_kernel_manifest_v1``)::

    schema_version: user_kernel_manifest_v1
    op_name: linalg.matmul
    archetype: compute_tiled                  # compute_tiled | pointwise | reduce | memory | activation | type_conv_index
    target_name: host_cpu
    language: c                               # c | triton | cuda | cpp
    kernel_source: kernel.c                   # path relative to manifest
    entry_symbol: compgen_matmul_f32
    inputs:
      - {name: lhs, dtype: f32, layout: row_major, dims: [16, 16]}
      - {name: rhs, dtype: f32, layout: row_major, dims: [16, 32]}
    outputs:
      - {name: out, dtype: f32, layout: row_major, dims: [16, 32]}
    numerics:
      accumulator_dtype: f32
      expected_numerics: tolerance_eps
    dispatch_model: sync
    perf_priors:                              # optional
      estimated_us: 5.0
      confidence: 0.8

Indexing reads each manifest, sha256-hashes every locked file
(manifest + source), and writes a derived index entry under
``.compgen/user_kernel_index/<sha8>/manifest.yaml``. The index is the
on-disk surface :class:`UserKernelProvider` consults at bid time.

Tamper detection: every use re-runs the SHA audit; any drift raises
``UserKernelHashDriftError`` (typed) so a kernel edited post-index
is rejected before it can fulfil an auction bid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


_MANIFEST_SCHEMA = "user_kernel_manifest_v1"
_INDEX_SCHEMA = "user_kernel_index_v1"
_REGISTRY_SCHEMA = "user_kernel_registry_v1"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class UserKernelManifestError(ValueError):
    """The user's ``kernel_manifest.yaml`` is malformed."""


class UserKernelHashDriftError(RuntimeError):
    """A locked file's sha256 disagrees with the index — file edited
    after indexing. The provider refuses to fulfil from this kernel."""


# --------------------------------------------------------------------------- #
# Discovery + indexing
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha8(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class UserKernelManifest:
    """Parsed user-supplied manifest (``kernel_manifest.yaml``)."""

    schema_version: str
    op_name: str
    archetype: str
    target_name: str
    language: str
    kernel_source: str
    entry_symbol: str
    inputs: tuple[dict[str, Any], ...]
    outputs: tuple[dict[str, Any], ...]
    numerics: dict[str, Any]
    dispatch_model: str = "sync"
    perf_priors: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "op_name": self.op_name,
            "archetype": self.archetype,
            "target_name": self.target_name,
            "language": self.language,
            "kernel_source": self.kernel_source,
            "entry_symbol": self.entry_symbol,
            "inputs": [dict(t) for t in self.inputs],
            "outputs": [dict(t) for t in self.outputs],
            "numerics": dict(self.numerics),
            "dispatch_model": self.dispatch_model,
            "perf_priors": dict(self.perf_priors),
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "UserKernelManifest":
        # Schema check first so callers see the right error when the
        # body's shape disagrees on the version field.
        if body.get("schema_version") != _MANIFEST_SCHEMA:
            raise UserKernelManifestError(
                f"unknown schema_version: {body.get('schema_version')!r}; "
                f"expected {_MANIFEST_SCHEMA!r}"
            )
        required = (
            "op_name", "archetype", "target_name", "language",
            "kernel_source", "entry_symbol", "inputs", "outputs", "numerics",
        )
        missing = [k for k in required if k not in body]
        if missing:
            raise UserKernelManifestError(
                f"manifest missing required fields: {missing}"
            )
        return cls(
            schema_version=str(body["schema_version"]),
            op_name=str(body["op_name"]),
            archetype=str(body["archetype"]),
            target_name=str(body["target_name"]),
            language=str(body["language"]),
            kernel_source=str(body["kernel_source"]),
            entry_symbol=str(body["entry_symbol"]),
            inputs=tuple(dict(t) for t in body["inputs"]),
            outputs=tuple(dict(t) for t in body["outputs"]),
            numerics=dict(body["numerics"]),
            dispatch_model=str(body.get("dispatch_model", "sync")),
            perf_priors=dict(body.get("perf_priors") or {}),
        )


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    """Best-effort reader: prefers PyYAML when available, falls back
    to JSON. Manifests are typically YAML but JSON works too."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml.safe_load(text) or {}
    except ImportError:
        return json.loads(text)


def _write_yaml_or_json(path: Path, body: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore[import-untyped]

        path.write_text(
            yaml.safe_dump(body, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
    except ImportError:
        path.write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class IndexEntry:
    """One row in ``.compgen/user_kernel_index/registry.yaml``."""

    index_id: str  # sha8 of source manifest path
    source_dir: str  # absolute path to user's kernel directory
    manifest: UserKernelManifest
    locked_files: dict[str, str]  # filename → sha256[:16]
    indexed_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _INDEX_SCHEMA,
            "index_id": self.index_id,
            "source_dir": self.source_dir,
            "indexed_at_utc": self.indexed_at_utc,
            "manifest": self.manifest.to_dict(),
            "locked_files": dict(self.locked_files),
        }


def discover_user_kernels(*, search_path: Path) -> list[Path]:
    """Walk ``search_path`` for ``kernel_manifest.yaml`` files.

    Returns the list of manifest paths in lexicographic order so the
    indexer is byte-deterministic across reruns.
    """
    search_path = Path(search_path).resolve()
    if not search_path.exists():
        return []
    manifests = sorted(search_path.rglob("kernel_manifest.yaml"))
    # Also accept .yml + .json variants.
    manifests.extend(sorted(search_path.rglob("kernel_manifest.yml")))
    manifests.extend(sorted(search_path.rglob("kernel_manifest.json")))
    return manifests


def index_one_manifest(
    *,
    manifest_path: Path,
    index_root: Path,
) -> IndexEntry:
    """Index a single user-supplied manifest.

    Reads + validates the manifest; SHAs the manifest + the kernel
    source it points at; writes the derived index entry under
    ``index_root/<index_id>/manifest.yaml``.
    """
    manifest_path = Path(manifest_path).resolve()
    body = _read_yaml_or_json(manifest_path)
    parsed = UserKernelManifest.from_dict(body)

    source_dir = manifest_path.parent
    kernel_path = (source_dir / parsed.kernel_source).resolve()
    if not kernel_path.exists():
        raise UserKernelManifestError(
            f"kernel_source {parsed.kernel_source!r} not found relative to "
            f"manifest {manifest_path}"
        )
    if source_dir not in kernel_path.parents and source_dir != kernel_path.parent:
        raise UserKernelManifestError(
            f"kernel_source {parsed.kernel_source!r} escapes the manifest's "
            f"directory ({source_dir}); user kernels must live alongside "
            f"their manifest"
        )

    locked_files = {
        manifest_path.name: _sha16(manifest_path),
        parsed.kernel_source: _sha16(kernel_path),
    }

    index_id = _sha8(str(manifest_path).encode("utf-8"))
    entry_dir = Path(index_root).resolve() / index_id
    entry_dir.mkdir(parents=True, exist_ok=True)
    entry = IndexEntry(
        index_id=index_id,
        source_dir=str(source_dir),
        manifest=parsed,
        locked_files=locked_files,
        indexed_at_utc=_utcnow(),
    )
    _write_yaml_or_json(entry_dir / "manifest.yaml", entry.to_dict())
    log.info(
        "user_kernel_index.indexed",
        index_id=index_id,
        op_name=parsed.op_name,
        target=parsed.target_name,
    )
    return entry


def reindex(*, search_path: Path, index_root: Path) -> dict[str, Any]:
    """Walk + index every manifest under ``search_path``, then update
    the ``registry.yaml`` summary at ``index_root/registry.yaml``.

    Returns ``{indexed_count, manifests_written, errors}``.
    """
    search_path = Path(search_path).resolve()
    index_root = Path(index_root).resolve()
    index_root.mkdir(parents=True, exist_ok=True)

    manifests = discover_user_kernels(search_path=search_path)
    written: list[str] = []
    errors: list[dict[str, Any]] = []
    indexed_entries: list[IndexEntry] = []

    for mpath in manifests:
        try:
            entry = index_one_manifest(
                manifest_path=mpath, index_root=index_root,
            )
            indexed_entries.append(entry)
            written.append(str(index_root / entry.index_id / "manifest.yaml"))
        except (UserKernelManifestError, OSError, json.JSONDecodeError) as exc:
            errors.append({
                "manifest_path": str(mpath),
                "error_kind": type(exc).__name__,
                "error_summary": str(exc),
            })

    registry_body = {
        "schema_version": _REGISTRY_SCHEMA,
        "indexed_at_utc": _utcnow(),
        "search_path": str(search_path),
        "entries": [
            {
                "index_id": e.index_id,
                "source_dir": e.source_dir,
                "op_name": e.manifest.op_name,
                "archetype": e.manifest.archetype,
                "target_name": e.manifest.target_name,
                "language": e.manifest.language,
                "kernel_source": e.manifest.kernel_source,
                "entry_symbol": e.manifest.entry_symbol,
                "indexed_at_utc": e.indexed_at_utc,
            }
            for e in indexed_entries
        ],
    }
    _write_yaml_or_json(index_root / "registry.yaml", registry_body)

    return {
        "indexed_count": len(indexed_entries),
        "manifests_written": written,
        "errors": errors,
        "registry_path": str(index_root / "registry.yaml"),
    }


def load_index_entries(*, index_root: Path) -> list[IndexEntry]:
    """Read every ``<sha>/manifest.yaml`` under ``index_root``.

    Skips entries whose body fails to parse (logged + omitted) so a
    single corrupt entry does not poison the entire registry.
    """
    index_root = Path(index_root).resolve()
    if not index_root.exists():
        return []
    out: list[IndexEntry] = []
    for entry_path in sorted(index_root.glob("*/manifest.yaml")):
        try:
            body = _read_yaml_or_json(entry_path)
            manifest_body = body.get("manifest") or {}
            manifest = UserKernelManifest.from_dict(manifest_body)
            out.append(
                IndexEntry(
                    index_id=str(body.get("index_id") or entry_path.parent.name),
                    source_dir=str(body.get("source_dir") or ""),
                    manifest=manifest,
                    locked_files=dict(body.get("locked_files") or {}),
                    indexed_at_utc=str(body.get("indexed_at_utc") or ""),
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "user_kernel_index.entry_parse_failed",
                path=str(entry_path),
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
    return out


def audit_locked_files(entry: IndexEntry) -> None:
    """Re-hash every locked file and raise :class:`UserKernelHashDriftError`
    if any disagrees with the index entry's recorded hash.

    The provider calls this before fulfilling a bid; any drift means
    the kernel was edited after indexing and we refuse to use it.
    """
    source_dir = Path(entry.source_dir)
    drifted: dict[str, tuple[str, str]] = {}
    for filename, expected_sha in entry.locked_files.items():
        path = source_dir / filename
        if not path.exists():
            drifted[filename] = (expected_sha, "missing")
            continue
        actual_sha = _sha16(path)
        if actual_sha != expected_sha:
            drifted[filename] = (expected_sha, actual_sha)
    if drifted:
        details = "; ".join(
            f"{f}: expected {exp!r} got {got!r}"
            for f, (exp, got) in drifted.items()
        )
        raise UserKernelHashDriftError(
            f"user kernel {entry.index_id!r} ({entry.manifest.op_name}) "
            f"failed locked-files audit: {details}"
        )


# --------------------------------------------------------------------------- #
# Discovery — env + flag resolution
# --------------------------------------------------------------------------- #


def resolve_user_kernel_path(
    *,
    cli_path: Path | str | None = None,
    env_value: str | None = None,
) -> Path | None:
    """Pick the user-kernel source path from CLI flag (priority) or env.

    Returns ``None`` when neither is set.
    """
    if cli_path:
        return Path(cli_path).resolve()
    import os

    val = env_value or os.environ.get("COMPGEN_USER_KERNEL_PATH", "")
    if val:
        return Path(val).resolve()
    return None


def default_index_root() -> Path:
    """The canonical index root: ``.compgen/user_kernel_index/`` under
    the current working directory."""
    return (Path.cwd() / ".compgen" / "user_kernel_index").resolve()


__all__ = [
    "IndexEntry",
    "UserKernelHashDriftError",
    "UserKernelManifest",
    "UserKernelManifestError",
    "audit_locked_files",
    "default_index_root",
    "discover_user_kernels",
    "index_one_manifest",
    "load_index_entries",
    "reindex",
    "resolve_user_kernel_path",
]
