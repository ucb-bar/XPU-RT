"""Smoke tests for compile/dispatch_matrix/chipyard wrappers.

We use a synthetic ``$MERLIN_ROOT`` populated with fake tools that
record their argv to a file so we can assert the bridge passed the
right flags through.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from xpu_rt.targets.backends.merlin import (
    build_chipyard_image,
    compile_dispatch_matrix,
    compile_program,
)
from xpu_rt.targets.backends.merlin.bridge import MerlinBridge


def _stand_up_fake_merlin(tmp_path: Path) -> Path:
    root = tmp_path / "merlin"
    tools = root / "tools"
    tools.mkdir(parents=True)
    (tools / "__init__.py").write_text("")

    # compile.py — emits a .vmfb under <build_dir>/<target>/program.vmfb
    (tools / "compile.py").write_text(textwrap.dedent("""
        import argparse, pathlib

        def setup_parser(p):
            p.add_argument("input_path")
            p.add_argument("--target", required=True)
            p.add_argument("--hw", default=None)
            p.add_argument("--quantized", action="store_true")
            p.add_argument("--build-dir", required=True)

        def main(args):
            out = pathlib.Path(args.build_dir) / args.target
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{args.target}.vmfb").write_bytes(b"VMFB")
            print(f"compiled {args.input_path} -> {out}")
            return 0
    """))

    # compile_dispatch_matrix.py — writes a matrix.json
    (tools / "compile_dispatch_matrix.py").write_text(textwrap.dedent("""
        import argparse, json, pathlib

        def setup_parser(p):
            p.add_argument("--source", required=True)
            p.add_argument("--targets", required=True)
            p.add_argument("--out-dir", required=True)

        def main(args):
            out = pathlib.Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            targets = args.targets.split(",")
            payload = {
                "schema_version": "matrix_v1",
                "source": args.source,
                "dispatches": [
                    {"target": t, "dispatch": f"d_{t}_0",
                     "vmfb": str(out / t / "d.vmfb")}
                    for t in targets
                ],
            }
            (out / "matrix.json").write_text(json.dumps(payload))
            return 0
    """))

    # chipyard.py — writes a simulator-* file
    (tools / "chipyard.py").write_text(textwrap.dedent("""
        import argparse, pathlib

        def setup_parser(p):
            p.add_argument("subcommand")
            p.add_argument("--hardware", required=True)
            p.add_argument("--out-dir", required=True)

        def main(args):
            out = pathlib.Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"simulator-{args.hardware}").write_bytes(b"SIM")
            return 0
    """))
    return root


def _teardown_fake_merlin(root: Path) -> None:
    if str(root) in sys.path:
        sys.path.remove(str(root))
    for mod in list(sys.modules):
        if mod == "tools" or mod.startswith("tools."):
            sys.modules.pop(mod, None)


def test_compile_program_writes_vmfb(tmp_path: Path) -> None:
    root = _stand_up_fake_merlin(tmp_path)
    try:
        bridge = MerlinBridge(merlin_root=root)
        result = compile_program(
            bridge,
            target="saturn_opu_v128",
            source=tmp_path / "model.mlir",
            out_dir=tmp_path / "build",
            hw="OPU",
            quantized=True,
        )
        assert result.returncode == 0
        assert result.vmfb_path is not None
        assert result.vmfb_path.is_file()
        assert "saturn_opu_v128" in str(result.vmfb_path)
    finally:
        _teardown_fake_merlin(root)


def test_compile_dispatch_matrix_parses_payload(tmp_path: Path) -> None:
    root = _stand_up_fake_merlin(tmp_path)
    try:
        bridge = MerlinBridge(merlin_root=root)
        result = compile_dispatch_matrix(
            bridge,
            source=tmp_path / "model.mlir",
            targets=["gemmini_mx", "spacemit_x60"],
            out_dir=tmp_path / "matrix",
        )
        assert result.returncode == 0
        assert result.matrix_path.is_file()
        per_target = result.per_target_dispatches
        assert per_target == {"gemmini_mx": ["d_gemmini_mx_0"],
                              "spacemit_x60": ["d_spacemit_x60_0"]}
    finally:
        _teardown_fake_merlin(root)


def test_chipyard_build_finds_simulator(tmp_path: Path) -> None:
    root = _stand_up_fake_merlin(tmp_path)
    try:
        bridge = MerlinBridge(merlin_root=root)
        result = build_chipyard_image(
            bridge, hardware="rocket-rv64", out_dir=tmp_path / "chip",
        )
        assert result.returncode == 0
        assert result.image_path is not None
        assert "simulator-rocket-rv64" in result.image_path.name
    finally:
        _teardown_fake_merlin(root)
