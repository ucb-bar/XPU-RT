"""baremetal C-codegen compilation tests.

The audit suggested that ``c_codegen.py``'s 819 LOC was untested.
That was wrong — ``test_c_codegen.py`` exercises the public surface
(8 tests). closes the **compilation** gap: the emitted C must
parse with ``cc -std=c11 -fsyntax-only``. Without this gate a future
regression that breaks the C-text shape would slip past the
existing structural tests.

The check uses a tiny ``npu_driver.h`` stub that declares the
``npu_*`` entry points the emit references (the stable C boundary
into the baremetal ukernel library). This is exactly the shape the
acceptance is asking for: "generated baremetal C compiles".
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Block, Region

from compgen.runtime.baremetal.c_codegen import emit_module


_NPU_STUB_HEADER = """\
/* Tiny npu_driver.h stub for syntactic compilation checks. The real
 * baremetal build picks up the production npu_driver.h from the
 * libcompgen_rt baremetal platform layer. The emit calls the npu_*
 * boundary using raw float pointers; the stub declares the minimal
 * surface those calls need. */
#ifndef NPU_DRIVER_H
#define NPU_DRIVER_H
#include <stddef.h>
#include <stdint.h>
/* The emit's stable signature: one input float buffer in, one float
 * buffer out. This matches what ``emit_function_definition`` writes
 * for a passthrough ``aten_relu`` (1-in / 1-out). */
extern float *npu_call_aten_relu(const float *in_0);
#endif
"""


def _find_cc() -> str | None:
    for cc in ("cc", "gcc", "clang"):
        path = shutil.which(cc)
        if path:
            return path
    return None


def _make_passthrough_module() -> ModuleOp:
    f32 = Float32Type()
    t = TensorType(f32, [4, 4])
    aten_relu = FuncOp("aten_relu", ([t], [t]), Region([]))
    body = Block(arg_types=[t])
    (x,) = body.args
    call = CallOp("aten_relu", [x], [t])
    body.add_op(call)
    body.add_op(ReturnOp(call.results[0]))
    forward = FuncOp("forward", ([t], [t]), Region([body]))
    return ModuleOp([aten_relu, forward])


@pytest.mark.skipif(_find_cc() is None, reason="no C compiler in PATH")
def test_emitted_baremetal_c_passes_cc_fsyntax_only(
    tmp_path: Path,
) -> None:
    module = _make_passthrough_module()
    out = emit_module(module)
    defn = next(g for g in out if g.sym_name == "forward")
    # Stub header + the emitted definition.
    (tmp_path / "npu_driver.h").write_text(_NPU_STUB_HEADER, encoding="utf-8")
    main_c = tmp_path / "main.c"
    main_c.write_text(
        '#include "npu_driver.h"\n\n' + defn.source,
        encoding="utf-8",
    )
    cc = _find_cc()
    assert cc is not None
    proc = subprocess.run(
        [
            cc, "-std=c11", "-Wall", "-Wextra", "-Wno-unused",
            "-fsyntax-only",
            f"-I{tmp_path}",
            str(main_c),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"baremetal codegen failed cc -fsyntax-only:\n"
        f"emit:\n{defn.source}\nstderr={proc.stderr!r}"
    )


@pytest.mark.skipif(_find_cc() is None, reason="no C compiler in PATH")
def test_emitted_declarations_compile(tmp_path: Path) -> None:
    """The aten_* passthrough declarations are extern prototypes; the
    cc compiles them without bodies (linker would resolve to the
    npu_driver implementations)."""
    module = _make_passthrough_module()
    out = emit_module(module)
    decl = next(g for g in out if g.sym_name == "aten_relu")
    (tmp_path / "npu_driver.h").write_text(_NPU_STUB_HEADER, encoding="utf-8")
    main_c = tmp_path / "decl.c"
    main_c.write_text(
        '#include "npu_driver.h"\n\n' + decl.source,
        encoding="utf-8",
    )
    cc = _find_cc()
    assert cc is not None
    proc = subprocess.run(
        [cc, "-std=c11", "-Wall", "-fsyntax-only", f"-I{tmp_path}",
         str(main_c)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"emitted aten declaration failed to compile: {proc.stderr!r}"
    )
