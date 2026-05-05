"""CLI for the model admission track.

Three subcommands, matching the user-facing contract:

- ``validate-registry`` - load and cross-validate every YAML config.
- ``run-suite`` - probe every entry in a suite YAML, emit summary + matrices.
- ``torch-compile`` - probe one (model, slice) pair and emit reports.

Exit codes:

- ``0`` - success (suite ran, validation passed).
- ``1`` - validation failure or non-trivial failure of the run-suite contract
  (e.g. a proxy that should pass, didn't).
- ``2`` - internal precondition violated (config file missing, etc.).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from compgen.model_admission.registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    DEFAULT_SLICES_DIR,
    DEFAULT_SUITES_DIR,
    RegistryError,
    load_registry,
)
from compgen.model_admission.report import aggregate_summary, write_suite_summary
from compgen.model_admission.schemas import (
    AdmissionStatus,
    ModelConfig,
    SliceConfig,
    SuiteConfig,
    SuiteSummaryRow,
)
from compgen.model_admission.torch_compile_probe import run_admission

log = structlog.get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m compgen.model_admission",
        description="Model admission registry + torch.compile suite.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    vr = sub.add_parser("validate-registry", help="Load every YAML config and check cross-references.")
    vr.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    vr.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    vr.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES_DIR)
    vr.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES_DIR)

    rs = sub.add_parser("run-suite", help="Run an admission suite end-to-end.")
    rs.add_argument("--suite", required=True, type=Path)
    rs.add_argument("--out", required=True, type=Path)
    rs.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    rs.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    rs.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES_DIR)
    rs.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES_DIR)
    rs.add_argument(
        "--strict-proxies",
        action="store_true",
        default=True,
        help=(
            "Fail (exit 1) if any required-proxy entry doesn't reach status=available. "
            "On by default; pass --no-strict-proxies to disable."
        ),
    )
    rs.add_argument("--no-strict-proxies", dest="strict_proxies", action="store_false")

    tc = sub.add_parser("torch-compile", help="Probe one (model, slice) pair.")
    tc.add_argument("--model", required=True, type=Path)
    tc.add_argument("--slice", dest="slice_path", type=Path, default=None)
    tc.add_argument("--out", required=True, type=Path)

    vs = sub.add_parser(
        "verify-sources",
        help=(
            "One-time HuggingFace source verification. Reads source_candidates.yaml, "
            "calls HfApi().model_info() once per candidate, and writes the canonical "
            "ref + revision SHA + verified_at into each configs/models/<id>.yaml."
        ),
    )
    vs.add_argument(
        "--candidates",
        type=Path,
        default=Path("configs/model_admission/source_candidates.yaml"),
    )
    vs.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    vs.add_argument(
        "--refresh",
        action="store_true",
        help="Re-verify every entry (default: skip those already source_verified=true).",
    )
    vs.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write YAML changes; only print the table.",
    )
    vs.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="MODEL_ID",
        help="Restrict to a subset of model_ids.",
    )

    fc = sub.add_parser(
        "forecast",
        help=(
            "Project per-model admission status onto a target hardware profile. "
            "Reads each model's support.hardware_requirements and tells you which "
            "currently-unavailable rows would flip to available on different hardware."
        ),
    )
    fc.add_argument(
        "--target-cc",
        type=str,
        default="10.0",
        help="Target compute capability (e.g. 7.5 / 8.0 / 8.9 / 10.0 for Blackwell).",
    )
    fc.add_argument(
        "--target-vram-gb",
        type=float,
        default=192.0,
        help="Target VRAM in GB. Default 192 (B200 HBM3e).",
    )
    fc.add_argument(
        "--target-disk-gb",
        type=float,
        default=300.0,
        help="Target free-disk budget for downloads in GB. Default 300.",
    )
    fc.add_argument(
        "--target-runtime-packages",
        nargs="*",
        default=("flash_attn",),
        metavar="PKG",
        help="Runtime packages assumed available on the target host.",
    )
    fc.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    fc.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    fc.add_argument("--slices-dir", type=Path, default=DEFAULT_SLICES_DIR)
    fc.add_argument("--suites-dir", type=Path, default=DEFAULT_SUITES_DIR)
    fc.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/model_admission/always_test_models"),
        help="Directory of an existing run-suite output (used as the baseline status).",
    )
    fc.add_argument("--out", type=Path, default=None,
                    help="Optional CSV path to also write the forecast to.")

    return p


def _cmd_validate_registry(args: argparse.Namespace) -> int:
    try:
        reg = load_registry(
            registry_path=args.registry,
            models_dir=args.models_dir,
            slices_dir=args.slices_dir,
            suites_dir=args.suites_dir,
        )
    except RegistryError as exc:
        print(f"registry validation failed: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"registry input missing: {exc}", file=sys.stderr)
        return 2
    print(
        f"OK: registry={len(reg.entries)} models={len(reg.models)} "
        f"slices={len(reg.slices)} suites={len(reg.suites)}"
    )
    return 0


def _cmd_torch_compile(args: argparse.Namespace) -> int:
    import os  # noqa: PLC0415

    if not args.model.exists():
        print(f"model config missing: {args.model}", file=sys.stderr)
        return 2
    model_cfg = ModelConfig.from_yaml(args.model)

    # If a transformers pin is configured AND we're not already running under
    # one, re-exec via ``uv run --with transformers==<pin>`` so the model's
    # bundled remote_code sees the version it was authored against. This is
    # the real fix for OpenVLA / DeepSeek-OCR / Moondream2 -- not a monkey
    # patch.
    pin = getattr(model_cfg.compile, "transformers_pin", "") or ""
    extra_pins = tuple(getattr(model_cfg.compile, "extra_pins", ()) or ())
    if pin and os.environ.get("COMPGEN_ADMISSION_PINNED") != pin:
        cmd = ["uv", "run", "--no-sync",
               "--with", f"transformers=={pin}",
               "--with", "tokenizers",
               "--with", "accelerate"]
        for spec in extra_pins:
            cmd.extend(["--with", spec])
        cmd.extend(["python", "-m", "compgen.model_admission", "torch-compile",
                    "--model", str(args.model), "--out", str(args.out)])
        if args.slice_path is not None:
            cmd.extend(["--slice", str(args.slice_path)])
        env = dict(os.environ, COMPGEN_ADMISSION_PINNED=pin)
        os.execvpe(cmd[0], cmd, env)
    slice_cfg: SliceConfig | None = None
    if args.slice_path is not None:
        if not args.slice_path.exists():
            print(f"slice config missing: {args.slice_path}", file=sys.stderr)
            return 2
        slice_cfg = SliceConfig.from_yaml(args.slice_path)
        if slice_cfg.parent_model_id != model_cfg.model_id:
            print(
                f"slice {slice_cfg.slice_id!r} parent_model_id={slice_cfg.parent_model_id!r} "
                f"does not match model {model_cfg.model_id!r}",
                file=sys.stderr,
            )
            return 2
    result = run_admission(model_cfg, slice_cfg, args.out)
    print(
        f"{model_cfg.model_id}/{slice_cfg.slice_id if slice_cfg else '-'}: "
        f"{result.admission.status} ({result.admission.reason or 'ok'})"
    )
    return 0


def _detect_transformers_version(model_cfg: ModelConfig) -> str | None:
    """Return the per-model transformers pin if explicitly configured.

    Pinning is opt-in via ``compile.transformers_pin`` in the model YAML.
    We do NOT auto-pin from ``config.json::transformers_version`` because
    older transformers versions are incompatible with the project's torch /
    other deps -- pinning more aggressively than necessary breaks models
    that work fine on the current transformers (TinyLlama, Phi-3, etc.).

    Set ``compile.transformers_pin`` only for models whose bundled
    remote_code legitimately requires an older transformers (OpenVLA,
    DeepSeek-OCR, Moondream2).
    """

    pin = getattr(model_cfg.compile, "transformers_pin", "") or ""
    return pin or None


def _build_probe_cmd(
    model_cfg: ModelConfig,
    slice_cfg: SliceConfig | None,
    out_dir: Path,
    pinned_transformers: str | None,
) -> list[str]:
    """Compose the subprocess command for one probe.

    When ``pinned_transformers`` is set, run via
    ``uv run --with transformers==X.Y.Z --no-sync`` so the model's bundled
    remote_code sees the version it was authored against. When unset, use
    the current interpreter directly -- faster, no env churn.
    """

    import sys

    base = ["-m", "compgen.model_admission", "torch-compile",
            "--model", str(model_cfg.raw_path), "--out", str(out_dir)]
    if slice_cfg is not None:
        base.extend(["--slice", str(slice_cfg.raw_path)])

    if pinned_transformers and _is_compat_pin_needed(pinned_transformers):
        return [
            "uv", "run", "--no-sync",
            "--with", f"transformers=={pinned_transformers}",
            "--with", "tokenizers",
            "--with", "accelerate",
            "python", *base,
        ]
    return [sys.executable, *base]


def _is_compat_pin_needed(version: str) -> bool:
    """Pin only when the model's transformers is a different MAJOR than ours.

    Same-major (e.g. 5.4 vs 5.5) almost always works on the installed
    transformers. Cross-major (e.g. 4.x vs 5.x) is where remote_code breaks.
    """

    try:
        major = int(version.split(".", 1)[0])
    except Exception:
        return False
    import transformers as _t
    try:
        installed_major = int(_t.__version__.split(".", 1)[0])
    except Exception:
        return False
    return major != installed_major


def _row_for(
    model_cfg: ModelConfig,
    slice_cfg: SliceConfig | None,
    out_dir: Path,
    blocking: bool,
) -> SuiteSummaryRow:
    """Probe one (model, slice) pair in a fresh subprocess.

    Subprocess isolation is required when running on GPU: 22+ multi-GB models
    serialised into one process inevitably OOM the CUDA caching allocator
    even with `torch.cuda.empty_cache()` between runs, because PyTorch holds
    onto fragmented blocks. A subprocess per probe gives clean CUDA state.
    """

    import subprocess
    import sys
    import json as _json

    out_dir.mkdir(parents=True, exist_ok=True)
    pinned_transformers = _detect_transformers_version(model_cfg)
    cmd = _build_probe_cmd(model_cfg, slice_cfg, out_dir, pinned_transformers)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        # Synthesise a row from whatever the subprocess wrote before the kill.
        proc = None  # type: ignore[assignment]

    admission_path = out_dir / "admission_report.json"
    eager_path = out_dir / "eager_report.json"
    fx_path = out_dir / "fx_report.json"
    export_path = out_dir / "export_report.json"
    dynamo_path = out_dir / "dynamo_report.json"
    compile_path = out_dir / "torch_compile_report.json"

    def _load(path: Path) -> dict:
        if path.exists():
            try:
                return _json.loads(path.read_text())
            except Exception:
                return {}
        return {}

    a = _load(admission_path)
    e = _load(eager_path)
    fx = _load(fx_path)
    ex = _load(export_path)
    d = _load(dynamo_path)
    c = _load(compile_path)

    if not a:
        # Subprocess crashed before writing; fabricate a typed unavailable row.
        a = {
            "status": AdmissionStatus.FAILED_EAGER.value,
            "reason": (proc.stderr.splitlines()[-1] if (proc and proc.stderr) else "subprocess failure"),
            "recommended_next_step": "See subprocess stderr in suite log.",
        }

    return SuiteSummaryRow(
        model_id=model_cfg.model_id,
        slice_id=slice_cfg.slice_id if slice_cfg else "",
        family=model_cfg.family,
        support_mode=model_cfg.support.mode,
        blocking=blocking,
        source_verified=model_cfg.source.source_verified,
        weights_available=a.get("status")
        in (AdmissionStatus.AVAILABLE.value, AdmissionStatus.AVAILABLE_SLICE_ONLY.value),
        dependency_status=a.get("status", AdmissionStatus.FAILED_EAGER.value),
        eager_status=e.get("status", "skipped"),
        fx_status=fx.get("status", "skipped"),
        export_status=ex.get("status", "skipped"),
        dynamo_status=d.get("status", "skipped"),
        torch_compile_status=c.get("status", "skipped"),
        graph_break_count=int(c.get("graph_break_count", 0)),
        compile_time_s=float(c.get("compile_time_s", 0.0)),
        recommended_next_step=a.get("recommended_next_step", ""),
    )


def _cmd_run_suite(args: argparse.Namespace) -> int:
    try:
        reg = load_registry(
            registry_path=args.registry,
            models_dir=args.models_dir,
            slices_dir=args.slices_dir,
            suites_dir=args.suites_dir,
        )
    except RegistryError as exc:
        print(f"registry validation failed: {exc}", file=sys.stderr)
        return 1

    if not args.suite.exists():
        print(f"suite missing: {args.suite}", file=sys.stderr)
        return 2
    suite = SuiteConfig.from_yaml(args.suite)

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[SuiteSummaryRow] = []
    proxy_failures: list[str] = []

    for entry in suite.required_proxy:
        m = reg.get_model(entry.model_id)
        s = reg.get_slice(entry.slice_id) if entry.slice_id else None
        run_dir = out_root / "proxy" / entry.model_id / (entry.slice_id or "_")
        blocking = reg.entries.get(entry.model_id).blocking if entry.model_id in reg.entries else True
        row = _row_for(m, s, run_dir, blocking=blocking)
        rows.append(row)
        if row.dependency_status != AdmissionStatus.AVAILABLE.value:
            proxy_failures.append(f"{entry.model_id}/{entry.slice_id or '-'} -> {row.dependency_status}")

    for entry in suite.required_real_if_available:
        m = reg.get_model(entry.model_id)
        s = reg.get_slice(entry.slice_id) if entry.slice_id else None
        run_dir = out_root / "real" / entry.model_id / (entry.slice_id or "_")
        blocking = reg.entries.get(entry.model_id).blocking if entry.model_id in reg.entries else True
        rows.append(_row_for(m, s, run_dir, blocking=blocking))

    for entry in suite.slice_only_stress:
        m = reg.get_model(entry.model_id)
        s = reg.get_slice(entry.slice_id) if entry.slice_id else None
        run_dir = out_root / "stress" / entry.model_id / (entry.slice_id or "_")
        blocking = reg.entries.get(entry.model_id).blocking if entry.model_id in reg.entries else False
        rows.append(_row_for(m, s, run_dir, blocking=blocking))

    summary = aggregate_summary(args.suite, out_root, rows)
    write_suite_summary(out_root, summary)

    print(
        f"suite: total={summary.total} available={summary.available} "
        f"slice_only={summary.available_slice_only} unavailable={summary.unavailable} failed={summary.failed}"
    )
    print(f"summary: {out_root / 'admission_summary.json'}")

    if args.strict_proxies and proxy_failures:
        print("strict-proxies: required proxies failed:", file=sys.stderr)
        for line in proxy_failures:
            print(f"  - {line}", file=sys.stderr)
        return 1
    return 0


def _cmd_verify_sources(args: argparse.Namespace) -> int:
    from compgen.model_admission.verify import VerifyStatus, verify_sources

    if not args.candidates.exists():
        print(f"candidates file missing: {args.candidates}", file=sys.stderr)
        return 2
    try:
        run = verify_sources(
            candidates_path=args.candidates,
            models_dir=args.models_dir,
            refresh=args.refresh,
            dry_run=args.dry_run,
            only_model_ids=list(args.only) if args.only else None,
        )
    except Exception as exc:
        print(f"verify-sources failed: {exc}", file=sys.stderr)
        return 1

    cols = ["model_id", "status", "canonical_ref", "revision", "gated", "error"]
    widths = {c: len(c) for c in cols}
    rows = [r.as_summary_row() for r in run.results]
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    fmt = "  ".join(f"{{{c}:<{widths[c]}}}" for c in cols)
    print(fmt.format(**{c: c for c in cols}))
    print(fmt.format(**{c: "-" * widths[c] for c in cols}))
    for r in rows:
        print(fmt.format(**{c: str(r.get(c, "")) for c in cols}))

    counts = run.by_status()
    print()
    print(
        f"summary: passed={counts.get(VerifyStatus.PASSED.value, 0)} "
        f"gated={counts.get(VerifyStatus.GATED.value, 0)} "
        f"not_found={counts.get(VerifyStatus.NOT_FOUND.value, 0)} "
        f"network_error={counts.get(VerifyStatus.NETWORK_ERROR.value, 0)} "
        f"auth_required={counts.get(VerifyStatus.AUTH_REQUIRED.value, 0)} "
        f"skipped={counts.get(VerifyStatus.SKIPPED_NO_CANDIDATE.value, 0)}"
    )
    if not args.dry_run:
        print(f"wrote {len(run.written)} YAML files: {', '.join(run.written) or '-'}")

    bad = counts.get(VerifyStatus.NETWORK_ERROR.value, 0)
    return 0 if bad == 0 else 1


def _parse_cc(cc: str) -> tuple[int, int]:
    try:
        major, minor = cc.split(".", 1)
        return int(major), int(minor)
    except Exception:
        return 0, 0


def _cc_meets(have: str, need: str) -> bool:
    if not need:
        return True
    return _parse_cc(have) >= _parse_cc(need)


def _cmd_forecast(args: argparse.Namespace) -> int:
    """Project each model's admission status onto a target hardware profile.

    Walks every model's ``support.hardware_requirements`` plus the existing
    ``run-suite`` admission_summary.csv and emits a per-row projection:

    - ``would_flip_to_available``: model is currently unavailable_hardware_constraint
      and the target hardware satisfies its requirements.
    - ``stays_available``: already passing today.
    - ``stays_unavailable``: blocked for non-hardware reasons (missing weights,
      lerobot bug, no public weights, license gating, JAX-only).
    - ``stays_failed_compile``: torch.compile would still fail (HF code bug,
      not hardware) but eager+dynamo data is real.

    Output is a forecast table to stdout; optional CSV via --out.
    """

    import csv  # noqa: PLC0415
    import json as _json  # noqa: PLC0415

    try:
        reg = load_registry(
            registry_path=args.registry,
            models_dir=args.models_dir,
            slices_dir=args.slices_dir,
            suites_dir=args.suites_dir,
        )
    except RegistryError as exc:
        print(f"registry validation failed: {exc}", file=sys.stderr)
        return 1

    summary_csv = args.results_dir / "admission_summary.csv"
    if not summary_csv.exists():
        print(
            f"baseline summary missing: {summary_csv}\n"
            f"Run `compgen.model_admission run-suite` first.",
            file=sys.stderr,
        )
        return 2

    with summary_csv.open() as fh:
        rows = list(csv.DictReader(fh))

    target_cc = args.target_cc
    target_vram = float(args.target_vram_gb)
    target_disk = float(args.target_disk_gb)
    target_pkgs = set(args.target_runtime_packages or ())

    out_rows: list[dict[str, str]] = []
    counts: dict[str, int] = {
        "stays_available": 0,
        "would_flip_to_available": 0,
        "stays_failed_compile": 0,
        "stays_unavailable_software": 0,
        "stays_unavailable_disk": 0,
        "stays_unavailable_license": 0,
    }

    for r in rows:
        mid = r["model_id"]
        status = r.get("dependency_status", "")
        cfg = reg.models.get(mid)
        hw = (cfg.support.hardware_requirements if cfg else {}) or {}
        min_cc = str(hw.get("min_compute_capability", "") or "")
        min_vram = float(hw.get("min_vram_gb", 0.0) or 0.0)
        need_pkgs = set(hw.get("required_runtime_packages", []) or [])
        notes = str(hw.get("notes", "")).lower()

        bucket: str
        why: str

        if status == "available":
            bucket = "stays_available"
            why = ""
        elif status == "available_slice_only":
            bucket = "stays_available"
            why = ""
        elif status == "failed_torch_compile":
            bucket = "stays_failed_compile"
            why = "torch.compile error is in HF code, not hardware"
        elif status == "unavailable_hardware_constraint":
            cc_ok = _cc_meets(target_cc, min_cc)
            vram_ok = target_vram >= min_vram
            pkgs_ok = need_pkgs.issubset(target_pkgs)
            blocked_software = "no torch port" in notes or "lerobot" in notes or "jax-only" in notes
            blocked_no_weights = "weights not publicly" in notes or "no public" in notes or "publicly available" in notes
            if blocked_software:
                bucket = "stays_unavailable_software"
                why = "non-hardware blocker (JAX/lerobot/code bug)"
            elif blocked_no_weights:
                bucket = "stays_unavailable_software"
                why = "no public weights identified"
            elif cc_ok and vram_ok and pkgs_ok:
                bucket = "would_flip_to_available"
                why = f"target meets cc>={min_cc or 'any'}, vram>={min_vram}GB, pkgs={sorted(need_pkgs) or 'none'}"
            else:
                bucket = "stays_unavailable_software"
                why = (
                    f"target shortfall: cc_ok={cc_ok}, vram_ok={vram_ok}, "
                    f"pkgs_ok={pkgs_ok} (need {sorted(need_pkgs) or '-'})"
                )
        elif status == "unavailable_too_large":
            # Huge models -- disk-bound, not GPU-bound. Flip if target_disk is large.
            need_disk = float(hw.get("min_disk_gb", 0.0) or 0.0)
            if target_disk >= max(need_disk, 200.0):
                bucket = "would_flip_to_available"
                why = f"target disk={target_disk}GB sufficient for slice"
            else:
                bucket = "stays_unavailable_disk"
                why = f"need >=200GB free disk; target has {target_disk}GB"
        elif status == "unavailable_missing_weights":
            # Either license-gated or huge slice-only with weights not yet downloaded.
            if cfg and cfg.source.source_verified:
                bucket = "stays_unavailable_disk"
                why = "weights verified but not cached on disk"
            else:
                bucket = "stays_unavailable_license"
                why = "license approval / gating pending"
        elif status == "unavailable_missing_dependency":
            bucket = "stays_unavailable_software"
            why = "missing dependency / loader-side issue (not hardware)"
        elif status == "failed_eager":
            bucket = "stays_failed_compile"
            why = "eager forward fails -- code-side bug, not hardware"
        else:
            bucket = "stays_unavailable_software"
            why = f"unknown status: {status}"

        counts[bucket] = counts.get(bucket, 0) + 1
        out_rows.append({
            "model_id": mid,
            "slice_id": r.get("slice_id", ""),
            "current_status": status,
            "forecast": bucket,
            "why": why,
            "hw_min_cc": min_cc,
            "hw_min_vram_gb": str(min_vram),
            "hw_required_pkgs": ",".join(sorted(need_pkgs)),
        })

    print(f"target: cc={target_cc}  vram={target_vram}GB  disk={target_disk}GB  pkgs={sorted(target_pkgs)}")
    print()
    print(f"{'model_id':<32} {'slice_id':<32} {'current':<32} {'forecast':<26} why")
    print("-" * 160)
    for r in out_rows:
        print(f"{r['model_id']:<32} {r['slice_id']:<32} {r['current_status']:<32} {r['forecast']:<26} {r['why']}")
    print()
    print("summary:")
    for k, v in counts.items():
        if v:
            print(f"  {k:<32} {v}")
    print()
    avail_now = counts["stays_available"]
    avail_target = avail_now + counts["would_flip_to_available"]
    total = len(out_rows)
    print(f"on this host: {avail_now}/{total} available")
    print(f"on target:    {avail_target}/{total} available  (+{counts['would_flip_to_available']})")

    if args.out:
        with args.out.open("w", newline="") as fh:
            cols = ["model_id", "slice_id", "current_status", "forecast", "why",
                    "hw_min_cc", "hw_min_vram_gb", "hw_required_pkgs"]
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for r in out_rows:
                writer.writerow(r)
        print(f"\nwrote: {args.out}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "validate-registry":
        return _cmd_validate_registry(args)
    if args.command == "torch-compile":
        return _cmd_torch_compile(args)
    if args.command == "run-suite":
        return _cmd_run_suite(args)
    if args.command == "verify-sources":
        return _cmd_verify_sources(args)
    if args.command == "forecast":
        return _cmd_forecast(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
