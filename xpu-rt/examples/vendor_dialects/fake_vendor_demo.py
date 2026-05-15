"""End-to-end fake-vendor demo — no GPU, no vendor toolchain required.

Exercises the full integration-agent flow against the tiny fixture
under ``tests/fixtures/fake_vendor/``:

1. Scan the fake vendor repo.
2. Inspect the proposed descriptor (skip approval UI — auto-approve).
3. Scaffold a user-space package into a temp dir.
4. Make the package importable.
5. Run the verification harness.

Run with::

    .venv/bin/python examples/vendor_dialects/fake_vendor_demo.py

The point of this demo is to have a regression-proof end-to-end test
that works without external dependencies; real CUDA Tile / Hexagon
integration (Phase D / E) reuses the same flow with real repos.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from compgen.mcp.tools.vendor_dialect import (
    scaffold_vendor_package,
    scan_vendor_repo,
    verify_vendor_package,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fake_vendor"


class _NullSessionManager:
    """Placeholder SessionManager — vendor tools don't need a session."""


def main() -> int:
    parser = argparse.ArgumentParser(description="Drive the fake vendor dialect flow.")
    parser.add_argument("--repo", default=str(FIXTURE), help="Vendor repo path.")
    parser.add_argument("--target", default="toy-target", help="CompGen target name.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for the scaffolded package (default: temp dir).",
    )
    args = parser.parse_args()

    sm = _NullSessionManager()
    print(f"[1/4] Scanning vendor repo at {args.repo}")
    scan_res = scan_vendor_repo(sm, repo_path=args.repo, target=args.target)
    print(f"      detected {scan_res['scan']['num_td_ops']} ops, "
          f"{scan_res['scan']['num_cli_tools']} CLI tools")

    print("[2/4] Proposed descriptor (YAML excerpt):")
    for line in scan_res["descriptor_yaml"].splitlines()[:15]:
        print(f"      {line}")
    print("      ...")

    tmp_ctx = tempfile.TemporaryDirectory() if args.out is None else None
    out_dir = Path(tmp_ctx.name) if tmp_ctx else args.out
    print(f"[3/4] Scaffolding user-space package under {out_dir}")
    scaffold_res = scaffold_vendor_package(
        sm,
        descriptor_yaml=scan_res["descriptor_yaml"],
        out_dir=str(out_dir),
        overwrite=True,
    )
    pkg_dir = Path(scaffold_res["package_dir"])
    print(f"      wrote {len(scaffold_res['files_written'])} files to {pkg_dir}")

    sys.path.insert(0, str(pkg_dir))

    print("[4/4] Running verification harness")
    verify_res = verify_vendor_package(sm, package_dir=str(pkg_dir))
    report = verify_res["report"]
    print(f"      adapter={report['adapter_name']} target={report['target']}")
    for gate in report["gates"]:
        status = "ok" if gate["passed"] else "FAIL"
        print(f"      [{status}] {gate['name']:14s} ({gate['elapsed_s']:.3f}s) {gate['notes']}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()

    if not verify_res["ok"]:
        print("Verification failed.")
        return 1
    print("\nDone. The scaffolded package would be pip-installable at the shown path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
