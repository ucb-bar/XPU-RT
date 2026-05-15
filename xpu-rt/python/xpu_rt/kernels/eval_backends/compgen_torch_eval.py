"""CompGen-native autocomp ``EvalBackend`` that doesn't shell out to
KernelBench.

Autocomp's stock ``KBEvalBackend`` requires a full KernelBench
checkout + a reference ``Model`` file pulled by problem id. For
CompGen-driven contracts we already have the reference module in
process (typically a ``torch.nn.functional`` baseline), so we run
the candidate ``ModelNew`` against a CompGen-provided reference
without any external script.

Returned per-candidate dict matches what
``compgen/third_party/autocomp/autocomp/search/search.py`` reads::

    {"correct": bool,
     "<metric>": float,         # populated only when correct
     "stdout": str, "stderr": str}

When the candidate raises during compilation, instantiation, or
inference, ``correct=False`` is returned with the exception
stashed in ``stderr``. No silent success, no silent skip.
"""

from __future__ import annotations

import io
import math
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch always present in CompGen venv
    torch = None  # type: ignore[assignment]


class CompGenTorchEvalBackend:
    """Autocomp-compatible eval backend powered by torch.

    Drop-in replacement for ``KBEvalBackend`` for CompGen contracts
    where the reference is a ``torch.nn`` module rather than a
    KernelBench problem id.

    Attributes:
        ref_source: Reference Python module text. Must define
            ``Model``, ``get_inputs``, ``get_init_inputs``.
        atol / rtol: Correctness tolerance for output comparison.
        warmup_iters: Pre-timed forward calls (default 3).
        timed_iters: Number of timed forward calls (default 20).
        device: Torch device for evaluation; defaults to ``cuda``
            when available, else ``cpu``.
    """

    def __init__(
        self,
        ref_source: str,
        *,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        warmup_iters: int = 3,
        timed_iters: int = 20,
        device: str | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required for CompGenTorchEvalBackend")
        self.ref_source = ref_source
        self.atol = atol
        self.rtol = rtol
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Autocomp EvalBackend interface
    # ------------------------------------------------------------------

    def evaluate_code(
        self,
        prob: Any,
        code_strs: list[str],
        simulator: str,
    ) -> list[dict[str, Any]]:
        """Evaluate each candidate against the reference."""

        out: list[dict[str, Any]] = []
        for code_str in code_strs:
            out.append(self._evaluate_one(code_str))
        return out

    def get_hw_feedback(self, prob: Any, code_strs: list[str]) -> list[list[str]]:
        return [[] for _ in code_strs]

    def get_backend_specific_rules(self) -> list[str]:
        return [
            "Generated code must define `class ModelNew(torch.nn.Module)` "
            "with a `forward` method matching the reference signature.",
            "Only `ModelNew` is imported by the evaluator. Other classes "
            "and helpers may be defined but must be referenced from "
            "`ModelNew`.",
            "Avoid `print()` in steady-state forward — it pollutes stdout.",
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_one(self, code_str: str) -> dict[str, Any]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                return self._run(code_str)
        except Exception:
            return {
                "correct": False,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue() + "\n" + traceback.format_exc(),
            }

    def _run(self, code_str: str) -> dict[str, Any]:
        ref_ns: dict[str, Any] = {}
        exec(compile(self.ref_source, "<ref_source>", "exec"), ref_ns)
        try:
            ref_cls = ref_ns["Model"]
            get_inputs = ref_ns["get_inputs"]
            get_init_inputs = ref_ns["get_init_inputs"]
        except KeyError as exc:
            return {
                "correct": False,
                "stderr": f"reference module missing {exc.args[0]!r}",
            }

        # Triton's @jit decorator calls inspect.getsourcelines on the
        # decorated function, which requires the candidate to live in a
        # real .py file. exec()-from-string fails the source lookup.
        # Materialize the candidate to a temp .py and import it.
        import importlib.util
        import os
        import sys
        import tempfile
        import uuid

        tmp_dir = tempfile.mkdtemp(prefix="compgen_eval_cand_")
        mod_name = f"compgen_cand_{uuid.uuid4().hex[:12]}"
        cand_path = os.path.join(tmp_dir, f"{mod_name}.py")
        with open(cand_path, "w") as f:
            f.write(code_str)
        spec = importlib.util.spec_from_file_location(mod_name, cand_path)
        if spec is None or spec.loader is None:
            return {
                "correct": False,
                "stderr": f"unable to build importlib spec for candidate at {cand_path}",
            }
        cand_mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = cand_mod
        spec.loader.exec_module(cand_mod)

        if not hasattr(cand_mod, "ModelNew"):
            return {
                "correct": False,
                "stderr": "candidate does not define `class ModelNew`",
            }
        cand_cls = cand_mod.ModelNew

        init_inputs = get_init_inputs()
        ref_inst = ref_cls(*init_inputs).to(self.device)
        cand_inst = cand_cls(*init_inputs).to(self.device)

        inputs = tuple(t.to(self.device) for t in get_inputs())

        # Correctness
        with torch.no_grad():
            ref_out = ref_inst(*inputs)
            cand_out = cand_inst(*inputs)

        ref_tensors = ref_out if isinstance(ref_out, (list, tuple)) else (ref_out,)
        cand_tensors = cand_out if isinstance(cand_out, (list, tuple)) else (cand_out,)
        if len(ref_tensors) != len(cand_tensors):
            return {
                "correct": False,
                "stderr": f"reference produced {len(ref_tensors)} outputs; "
                          f"candidate produced {len(cand_tensors)}",
            }
        for i, (r, c) in enumerate(zip(ref_tensors, cand_tensors)):
            if not torch.allclose(r, c, atol=self.atol, rtol=self.rtol):
                max_abs = (r - c).abs().max().item()
                return {
                    "correct": False,
                    "stderr": f"output[{i}] mismatch: max_abs_diff={max_abs:.3e} "
                              f"atol={self.atol} rtol={self.rtol}",
                }

        # Latency
        for _ in range(self.warmup_iters):
            with torch.no_grad():
                cand_inst(*inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(self.timed_iters):
            with torch.no_grad():
                cand_inst(*inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / self.timed_iters
        latency_ms = elapsed * 1000.0

        return {
            "correct": True,
            "latency": latency_ms,
            "stdout": "",
            "stderr": "",
        }
