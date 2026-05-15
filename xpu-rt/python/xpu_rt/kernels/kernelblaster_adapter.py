"""KernelBlaster subprocess adapter.

Bridges a CompGen :class:`~compgen.kernels.provider.KernelContract` to
NVlabs' KernelBlaster (https://github.com/NVlabs/KernelBlaster). Unlike
autocomp, KernelBlaster is not a Python library: it ships as a Docker
image + shell script (``scripts/run_single_kernelblaster.sh``). This
adapter owns the orchestration — input-file staging, subprocess
invocation, output parsing, and graceful-degradation when KB isn't
installed on the host.

Two invocation modes are supported. The adapter picks one at runtime;
callers can force a specific mode via ``mode=`` on
:class:`KernelBlasterAdapter`:

1. **local** — a cloned KernelBlaster source tree. Set
   ``COMPGEN_KERNELBLASTER_ROOT`` to the path, or drop it at
   ``third_party/kernelblaster`` relative to CWD. The adapter runs
   ``bash scripts/run_single_kernelblaster.sh`` from that directory.
2. **docker** — a pre-built Docker image. Set
   ``COMPGEN_KERNELBLASTER_IMAGE`` (e.g. ``kernelblaster:latest``). The
   adapter runs ``docker run --rm --gpus=all …`` with the workdir
   mounted.

Both modes require ``OPENAI_API_KEY`` in the environment. If neither
mode is available, :func:`search_kernel` raises :class:`KernelBlasterUnavailable`
and the provider reports ``ProviderResult(found=False)`` — exactly the
same contract autocomp follows.

Contract input — the caller must pass the CUDA kernel to optimise plus
its validation harness through ``KernelContract.constraints``::

    contract = KernelContract(
        region_id="matmul_0",
        op_family="matmul",
        target_name="cuda",
        hardware_key="H100",
        constraints={
            "kernelblaster": {
                "init_cu": "<contents of init.cu>",
                "driver_cpp": "<contents of driver.cpp>",
                # optional overrides:
                "dataset": "kernelbench-cuda",
                "precision": "fp16",
                "level": "level_1",
                "problem_id": "0",
            }
        },
    )

KernelBlaster's own input schema (see the repo's
``data/kernelbench-cuda/<level>/<problem>/``) is re-created inside a
temp directory by :func:`_stage_inputs` before KB is invoked.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog


class _NullCtx:
    """No-op context manager used when keeping the workdir on disk."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


from compgen.kernels.provider import (
    ContractFeedback,
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)

log = structlog.get_logger()


DEFAULT_DATASET = "kernelbench-cuda"
DEFAULT_PRECISION = "fp16"
DEFAULT_LEVEL = "level1"
DEFAULT_PROBLEM_ID = "1"
DEFAULT_PROBLEM_NAME = "compgen_custom"
DEFAULT_EXPERIMENT = "compgen_run"
DEFAULT_RL_EXPERIMENT = "kernelblaster"
DEFAULT_GPU_TYPE = "H100"
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"


class KernelBlasterUnavailable(RuntimeError):
    """Raised when KernelBlaster cannot be invoked on this host.

    Reasons: no local checkout, no docker image, missing OPENAI_API_KEY,
    or the configured invocation mode isn't actually installed.
    """


@dataclass
class KernelBlasterConfig:
    """Runtime configuration for a KernelBlaster invocation.

    Every field has an env-var fallback; callers usually construct the
    config via :func:`KernelBlasterConfig.from_env`. Explicit values
    override env vars.
    """

    mode: str = ""  # "local" | "docker" | "" (auto)
    repo_root: Path | None = None  # source checkout for "local" mode
    image: str = ""  # docker image for "docker" mode
    openai_api_key: str = ""
    model: str = DEFAULT_MODEL
    gpu_type: str = DEFAULT_GPU_TYPE
    dataset: str = DEFAULT_DATASET
    precision: str = DEFAULT_PRECISION
    experiment_name: str = DEFAULT_EXPERIMENT
    rl_experiment_name: str = DEFAULT_RL_EXPERIMENT
    docker_extra_args: tuple[str, ...] = ()
    pass_through_env: tuple[str, ...] = (
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "WANDB_API_KEY",
        # CompGen integration: allow KB's GPU server to coexist
        # with other CUDA workloads, and provide the Gemini key
        # KB's OpenAI-compat client picks up.
        "KERNELBLASTER_GPU_SERVER_SKIP_PROCESS_CHECK",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "PERFLAB_KEY",
    )

    @classmethod
    def from_env(cls, **overrides: Any) -> KernelBlasterConfig:
        """Build a config from ``COMPGEN_KERNELBLASTER_*`` + KB env vars."""

        def _pick(*names: str, default: str = "") -> str:
            for n in names:
                value = os.environ.get(n)
                if value:
                    return value
            return default

        repo_root_s = _pick("COMPGEN_KERNELBLASTER_ROOT")
        image = _pick("COMPGEN_KERNELBLASTER_IMAGE")
        mode = _pick("COMPGEN_KERNELBLASTER_MODE")

        cfg = cls(
            mode=mode,
            repo_root=Path(repo_root_s).expanduser() if repo_root_s else None,
            image=image,
            # KernelBlaster accepts any OpenAI-compatible key. We also
            # let GOOGLE_API_KEY satisfy this slot because the forked
            # KB ``query.py`` routes gemini-* models to Google's
            # OpenAI-compatible endpoint when GOOGLE_API_KEY is set.
            openai_api_key=_pick("OPENAI_API_KEY", "GOOGLE_API_KEY", "PERFLAB_KEY"),
            model=_pick("COMPGEN_KERNELBLASTER_MODEL", "MODEL", default=DEFAULT_MODEL),
            gpu_type=_pick("COMPGEN_KERNELBLASTER_GPU_TYPE", "GPU_TYPE", default=DEFAULT_GPU_TYPE),
            dataset=_pick("COMPGEN_KERNELBLASTER_DATASET", "DATASET", default=DEFAULT_DATASET),
            precision=_pick("COMPGEN_KERNELBLASTER_PRECISION", "PRECISION", default=DEFAULT_PRECISION),
            experiment_name=_pick("COMPGEN_KERNELBLASTER_EXPERIMENT", "EXPERIMENT_NAME", default=DEFAULT_EXPERIMENT),
            rl_experiment_name=_pick(
                "COMPGEN_KERNELBLASTER_RL_EXPERIMENT", "RL_EXPERIMENT_NAME", default=DEFAULT_RL_EXPERIMENT
            ),
        )
        for key, val in overrides.items():
            setattr(cfg, key, val)
        return cfg

    def resolved_mode(self) -> str:
        """Auto-detect the invocation mode if not explicitly set."""
        if self.mode:
            return self.mode
        if self.repo_root and self.repo_root.exists():
            return "local"
        if self.image:
            return "docker"
        # Fallback: look for a conventional checkout.
        conventional = Path.cwd() / "third_party" / "kernelblaster"
        if conventional.exists():
            return "local"
        return ""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class KernelBlasterAdapter:
    """Orchestrates a single KernelBlaster search.

    One instance = one search; the adapter is stateless across searches
    beyond the config it was constructed with. The provider layer
    (:class:`compgen.kernels.providers.kernelblaster.KernelBlasterProvider`)
    owns accumulated knowledge across multiple searches.
    """

    config: KernelBlasterConfig = field(default_factory=KernelBlasterConfig.from_env)
    max_runtime_seconds: int = 0  # 0 = unbounded, else subprocess timeout
    # Hook for tests: override the subprocess runner.
    _run: Any = field(default=None, repr=False)

    # ---- Public API -------------------------------------------------------

    def is_available(self) -> tuple[bool, str]:
        """Cheap health check. Returns ``(ok, reason)``.

        No network call, no subprocess. Suitable for
        :func:`~compgen.kernels.providers.kernelblaster.KernelBlasterProvider.accepts_contract`.
        """
        mode = self.config.resolved_mode()
        if not mode:
            return (False, "no KernelBlaster source tree or docker image configured")
        if mode == "local":
            root = self._resolve_local_root()
            if root is None:
                return (False, f"COMPGEN_KERNELBLASTER_ROOT={self.config.repo_root!s} does not exist")
            script = root / "scripts" / "run_single_kernelblaster.sh"
            if not script.exists():
                return (False, f"{script} missing")
        elif mode == "docker":
            if not shutil.which("docker"):
                return (False, "docker mode selected but `docker` is not on PATH")
        else:
            return (False, f"unknown KernelBlaster mode: {mode!r}")
        if not self.config.openai_api_key:
            return (False, "no LLM key set (need one of OPENAI_API_KEY / GOOGLE_API_KEY / PERFLAB_KEY)")
        return (True, "")

    def search_kernel(
        self,
        contract: KernelContract,
        budget: SearchBudget,
    ) -> ProviderResult:
        """Run one KernelBlaster search for ``contract`` within ``budget``.

        Raises:
            KernelBlasterUnavailable: host can't invoke KB
                (no repo, no docker, or no API key).
            ValueError: contract lacks the required ``init_cu`` /
                ``driver_cpp`` payloads.
        """
        ok, reason = self.is_available()
        if not ok:
            raise KernelBlasterUnavailable(reason)

        kb_constraints = contract.constraints.get("kernelblaster") or {}
        init_cu = kb_constraints.get("init_cu")
        driver_cpp = kb_constraints.get("driver_cpp")
        if not init_cu or not driver_cpp:
            raise ValueError(
                "KernelBlaster contract requires constraints.kernelblaster.init_cu "
                "and constraints.kernelblaster.driver_cpp (CUDA kernel + C++ harness)."
            )

        # When COMPGEN_KERNELBLASTER_KEEP_WORKDIR=1, keep the workdir on
        # disk for debugging instead of cleaning it up.
        keep_workdir = os.environ.get("COMPGEN_KERNELBLASTER_KEEP_WORKDIR", "").lower() in ("1", "true", "yes")
        if keep_workdir:
            tmp_s = tempfile.mkdtemp(prefix="compgen_kb_kept_")
            workdir = Path(tmp_s)
            log.info("kernelblaster.workdir.keeping", workdir=str(workdir))
            workdir_ctx = _NullCtx()
        else:
            workdir_ctx = tempfile.TemporaryDirectory(prefix="compgen_kb_")
            workdir = Path(workdir_ctx.name)

        with workdir_ctx:
            # Overlay KB's repo root (local mode only) so the shell
            # script resolves its scripts/, src/, utils/ relative to our
            # workdir while reading data/ + writing out/ locally.
            if self.config.resolved_mode() == "local":
                root = self._resolve_local_root()
                assert root is not None  # guarded by is_available()
                self._overlay_kb_root(workdir, root)
            self._stage_inputs(workdir, init_cu, driver_cpp, kb_constraints)
            args, env = self._build_invocation(workdir, kb_constraints, budget)

            log.info(
                "kernelblaster.search.start",
                mode=self.config.resolved_mode(),
                region=contract.region_id,
                workdir=str(workdir),
            )
            completed = self._invoke(args, env=env, cwd=workdir)
            result = self._parse_output(workdir, contract, kb_constraints, completed)
            log.info(
                "kernelblaster.search.done",
                region=contract.region_id,
                found=result.found,
                latency_us=result.latency_us,
                speedup=result.speedup,
            )
            return result

    # ---- Invocation layout -----------------------------------------------

    def _resolve_local_root(self) -> Path | None:
        """Find the KernelBlaster checkout for local mode.

        If the caller pinned ``repo_root`` (via env or explicit config),
        honour it even when it doesn't exist — that's a diagnostic signal
        worth surfacing, not a cue to silently fall back. If no explicit
        root was set, accept the conventional ``third_party/kernelblaster``
        location under the current working directory.
        """
        if self.config.repo_root is not None:
            return self.config.repo_root if self.config.repo_root.exists() else None
        conventional = Path.cwd() / "third_party" / "kernelblaster"
        return conventional if conventional.exists() else None

    def _problem_dirname(self, kb_constraints: dict[str, Any]) -> str:
        """KB problem dirs are ``NNN_<short_name>``; build one that matches."""
        problem_id = int(kb_constraints.get("problem_id", DEFAULT_PROBLEM_ID))
        name = str(kb_constraints.get("problem_name", DEFAULT_PROBLEM_NAME))
        return f"{problem_id:03d}_{name}"

    def _stage_inputs(
        self,
        workdir: Path,
        init_cu: str,
        driver_cpp: str,
        kb_constraints: dict[str, Any],
    ) -> Path:
        """Lay out KB's expected ``data/<dataset>/<level>/<NNN_name>/`` tree.

        KB's own dataset uses directory names like
        ``001_Square_matrix_multiplication``; the ``--problem-numbers``
        flag selects one by leading integer. We stage our inputs into a
        matching ``NNN_compgen_custom`` (or caller-supplied ``problem_name``)
        directory.
        """
        dataset = kb_constraints.get("dataset", self.config.dataset)
        level = kb_constraints.get("level", DEFAULT_LEVEL)
        problem_dirname = self._problem_dirname(kb_constraints)

        problem_dir = workdir / "data" / dataset / level / problem_dirname
        problem_dir.mkdir(parents=True, exist_ok=True)
        (problem_dir / "init.cu").write_text(init_cu)
        (problem_dir / "driver.cpp").write_text(driver_cpp)

        # Propagate any extra files the caller stuffed into the constraints
        # (e.g. reference outputs, test vectors, CMake snippets).
        extras = kb_constraints.get("extra_files") or {}
        for rel_path, contents in extras.items():
            dest = problem_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(contents)

        return problem_dir

    def _overlay_kb_root(self, workdir: Path, kb_root: Path) -> None:
        """Lay out a shadow KB tree under ``workdir``.

        Top-level items become symlinks except for ``data/`` (mixed)
        and ``out/`` (writable). For ``data/``:

        * The Python package files (``__init__.py``, ``dataset.py``,
          ``kernelbench.py``, ``kernelbench_cuda.py``, etc.) are
          symlinked so ``from data import get_dataset`` works.
        * Each dataset subdirectory (e.g.
          ``kernelbench-cuda/level1/001_*``) is symlinked so KB's
          reference problems remain accessible.
        * The staged CompGen problem (written later by
          ``_stage_inputs``) lives alongside, in a fresh
          subdirectory under
          ``data/<dataset>/<level>/<problem_dirname>``.

        ``.git``, ``out``, and the user's previous KB run artifacts
        are excluded so the workdir is self-contained.
        """
        for child in kb_root.iterdir():
            if child.name in {"data", "out", ".git"}:
                continue
            link = workdir / child.name
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(child)

        # Mirror data/ structure: package files symlinked, dataset
        # directories symlinked, problem subdirectories under our
        # configured (dataset, level) symlinked so KB sees a complete
        # data tree.
        kb_data = kb_root / "data"
        workdir_data = workdir / "data"
        workdir_data.mkdir(parents=True, exist_ok=True)

        if not kb_data.exists():
            # Fixture or stripped checkout without a populated data/
            # tree — leave workdir_data empty for the caller to stage
            # into via :meth:`_stage_inputs`.
            return

        for entry in kb_data.iterdir():
            target = workdir_data / entry.name
            if target.exists() or target.is_symlink():
                continue
            if entry.is_file():
                target.symlink_to(entry)
            else:
                # Dataset directory (e.g. kernelbench-cuda). Mirror its
                # structure: README files symlinked, each level
                # directory mirrored, and inside the levels each
                # problem subdir symlinked. This lets the staged
                # CompGen problem land in a fresh sibling without
                # colliding with KB's reference set.
                target.mkdir(parents=True, exist_ok=True)
                for sub in entry.iterdir():
                    sub_target = target / sub.name
                    if sub_target.exists() or sub_target.is_symlink():
                        continue
                    if sub.is_file():
                        sub_target.symlink_to(sub)
                    else:
                        # level directory — mirror so we can add our
                        # own problem subdir alongside.
                        sub_target.mkdir(parents=True, exist_ok=True)
                        for prob in sub.iterdir():
                            prob_target = sub_target / prob.name
                            if prob_target.exists() or prob_target.is_symlink():
                                continue
                            prob_target.symlink_to(prob)

    def _build_invocation(
        self,
        workdir: Path,
        kb_constraints: dict[str, Any],
        budget: SearchBudget,
    ) -> tuple[list[str], dict[str, str]]:
        """Build the argv + env for the chosen invocation mode."""
        mode = self.config.resolved_mode()
        dataset = kb_constraints.get("dataset", self.config.dataset)
        precision = kb_constraints.get("precision", self.config.precision)
        problem_id = str(int(kb_constraints.get("problem_id", DEFAULT_PROBLEM_ID)))
        level = kb_constraints.get("level", DEFAULT_LEVEL)
        # KB's shell script accepts --problem-numbers and --subset;
        # map the budget onto a rollout cap via KB_MAX_ITERATIONS
        # (forwarded as env since the script reads env internally).
        # Point KB at the workdir's data tree (workdir/data/<dataset>)
        # so its optimization_rl_ncu node finds the staged problem
        # files. Without this, KB's ``Path(__file__).resolve().parents[4]``
        # follows the symlinked src/ back to the real kb checkout and
        # looks for ``001_compgen_custom/init.cu`` there.
        curated_data_dir = workdir / "data" / dataset
        kb_env = {
            "OPENAI_API_KEY": self.config.openai_api_key,
            "MODEL": self.config.model,
            "GPU_TYPE": self.config.gpu_type,
            "DATASET": dataset,
            "PRECISION": precision,
            "LEVEL": level,
            "EXPERIMENT_NAME": self.config.experiment_name,
            "RL_EXPERIMENT_NAME": self.config.rl_experiment_name,
            "KB_MAX_ITERATIONS": str(budget.max_iterations),
            "KB_MAX_CANDIDATES": str(budget.max_candidates),
            "KERNELBLASTER_CURATED_DATA_DIR": str(curated_data_dir),
        }
        for name in self.config.pass_through_env:
            value = os.environ.get(name)
            if value:
                kb_env[name] = value
        # Propagate CompGen's google.genai instrumentation into the KB
        # subprocess so its Gemini API calls land in our usage tracker.
        # We prepend a tiny ``sitecustomize.py`` dir onto PYTHONPATH;
        # CPython auto-imports ``sitecustomize`` at startup from the
        # first matching path entry. Best-effort — falls back silently
        # if the bootstrap dir doesn't exist (e.g. truncated install).
        from pathlib import Path as _Path
        bootstrap_dir = (
            _Path(__file__).resolve().parents[1]
            / "observability"
            / "_subprocess_bootstrap"
        )
        if bootstrap_dir.is_dir():
            existing_pp = os.environ.get("PYTHONPATH", "")
            kb_env["PYTHONPATH"] = (
                f"{bootstrap_dir}{os.pathsep}{existing_pp}"
                if existing_pp
                else str(bootstrap_dir)
            )
        env = {**os.environ, **kb_env}

        script_args = [
            "--problem-numbers",
            problem_id,
            "--subset",
            level,
        ]

        if mode == "local":
            root = self._resolve_local_root()
            assert root is not None  # guarded by is_available()
            # Invoke through the *overlay* — the shell script then
            # computes ROOT_DIR as our workdir, and its data/ + out/
            # lookups resolve against our staged tree, not the user's
            # KB checkout.
            overlay_script = workdir / "scripts" / "run_single_kernelblaster.sh"
            argv = ["bash", str(overlay_script), *script_args]
            env["KERNELBLASTER_ROOT"] = str(root)
            return argv, env

        if mode == "docker":
            argv = [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-v",
                f"{workdir}:/workspace",
                "-w",
                "/workspace",
            ]
            for key, value in kb_env.items():
                argv += ["-e", f"{key}={value}"]
            argv += list(self.config.docker_extra_args)
            argv += [
                self.config.image,
                "bash",
                "scripts/run_single_kernelblaster.sh",
                *script_args,
            ]
            return argv, env

        raise KernelBlasterUnavailable(f"unknown KernelBlaster mode: {mode!r}")

    def _invoke(
        self,
        argv: list[str],
        *,
        env: dict[str, str],
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``argv`` and return the completed process.

        Respects ``self._run`` (test hook) and ``max_runtime_seconds``.
        """
        runner = self._run or subprocess.run
        timeout = self.max_runtime_seconds or None
        try:
            return runner(
                argv,
                env=env,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise KernelBlasterUnavailable(f"KernelBlaster invocation timed out after {timeout}s") from exc
        except FileNotFoundError as exc:
            raise KernelBlasterUnavailable(f"cannot invoke KernelBlaster: {exc}") from exc

    # ---- Output parsing --------------------------------------------------

    def _parse_output(
        self,
        workdir: Path,
        contract: KernelContract,
        kb_constraints: dict[str, Any],
        completed: subprocess.CompletedProcess[str],
    ) -> ProviderResult:
        """Read KB's ``out/`` artifacts + build a ``ProviderResult``.

        Locations come from KB's own conventions:

        ``out/<DATASET>/<PRECISION>/<EXPERIMENT_NAME>/final_rl_cuda_perf.cu``
        ``out/<DATASET>/<PRECISION>/<EXPERIMENT_NAME>/optimization_database.json``
        """
        dataset = kb_constraints.get("dataset", self.config.dataset)
        precision = kb_constraints.get("precision", self.config.precision)
        out_dir = workdir / "out" / dataset / precision / self.config.experiment_name

        kernel_path = out_dir / "final_rl_cuda_perf.cu"
        db_path = out_dir / "optimization_database.json"

        if completed.returncode != 0 and not kernel_path.exists():
            log.warning(
                "kernelblaster.nonzero_return",
                returncode=completed.returncode,
                stderr_tail=(completed.stderr or "")[-500:],
            )
            return ProviderResult(
                found=False,
                metadata={
                    "provider": "kernelblaster",
                    "returncode": completed.returncode,
                    "stderr_tail": (completed.stderr or "")[-500:],
                },
            )

        if not kernel_path.exists():
            return ProviderResult(
                found=False,
                metadata={
                    "provider": "kernelblaster",
                    "reason": "kernel artifact missing",
                    "expected_path": str(kernel_path),
                },
            )

        kernel_source = kernel_path.read_text()
        db: dict[str, Any] = {}
        if db_path.exists():
            try:
                db = json.loads(db_path.read_text())
            except json.JSONDecodeError:
                db = {"_parse_error": True}

        latency_us = float(db.get("final_latency_us", 0.0) or 0.0)
        baseline_us = float(db.get("baseline_latency_us", 0.0) or 0.0)
        speedup = (baseline_us / latency_us) if latency_us > 0 else float(db.get("speedup", 0.0) or 0.0)
        iterations = int(db.get("iterations", 0) or 0)
        candidates = int(db.get("candidates_evaluated", 0) or 0)
        correct = bool(db.get("final_correct", bool(kernel_source)))

        knowledge = _extract_knowledge(contract, db)
        feedback = _extract_feedback(db)

        return ProviderResult(
            found=True,
            kernel_code=kernel_source,
            language="cuda",
            latency_us=latency_us,
            correct=correct,
            plan=db.get("plan", ""),
            speedup=speedup,
            iterations_used=iterations,
            total_candidates=candidates,
            knowledge_exports=knowledge,
            contract_feedback=feedback,
            metadata={
                "provider": "kernelblaster",
                "mode": self.config.resolved_mode(),
                "experiment_name": self.config.experiment_name,
                "dataset": dataset,
                "precision": precision,
                "baseline_latency_us": baseline_us,
                "db_path": str(db_path),
                "kernel_path": str(kernel_path),
            },
        )


def _extract_knowledge(
    contract: KernelContract,
    db: dict[str, Any],
) -> list[KnowledgeExport]:
    """Turn KB's optimization_database.json into CompGen knowledge."""
    exports: list[KnowledgeExport] = []

    lessons = db.get("lessons") or db.get("strategies") or []
    for lesson in lessons:
        if isinstance(lesson, dict):
            exports.append(
                KnowledgeExport(
                    kind=str(lesson.get("kind", "optimization_tactic")),
                    scope=str(lesson.get("scope", "operator_family")),
                    scope_key=str(lesson.get("scope_key", contract.op_family)),
                    content=str(lesson.get("content") or lesson.get("summary", "")),
                    metadata={
                        "provider": "kernelblaster",
                        "target": contract.target_name,
                        **{
                            k: v
                            for k, v in lesson.items()
                            if k not in {"kind", "scope", "scope_key", "content", "summary"}
                        },
                    },
                    confidence=float(lesson.get("confidence", 0.6)),
                )
            )
        elif isinstance(lesson, str):
            exports.append(
                KnowledgeExport(
                    kind="optimization_tactic",
                    scope="operator_family",
                    scope_key=contract.op_family,
                    content=lesson,
                    metadata={"provider": "kernelblaster", "target": contract.target_name},
                    confidence=0.5,
                )
            )

    if not exports and contract.op_family:
        exports.append(
            KnowledgeExport(
                kind="optimization_tactic",
                scope="operator_family",
                scope_key=contract.op_family,
                content=(
                    f"KernelBlaster optimised {contract.op_family} on {contract.hardware_key or contract.target_name}"
                ),
                metadata={
                    "provider": "kernelblaster",
                    "speedup": db.get("speedup"),
                    "final_latency_us": db.get("final_latency_us"),
                },
                confidence=0.5,
            )
        )
    return exports


def _extract_feedback(db: dict[str, Any]) -> list[ContractFeedback]:
    """Surface KB-reported contract changes as ``ContractFeedback``."""
    out: list[ContractFeedback] = []
    for fb in db.get("contract_feedback", []) or []:
        if not isinstance(fb, dict):
            continue
        out.append(
            ContractFeedback(
                field=str(fb.get("field", "")),
                current_value=str(fb.get("current_value", "")),
                suggested_value=str(fb.get("suggested_value", "")),
                reason=str(fb.get("reason", "")),
                measured_gain=float(fb.get("measured_gain", 0.0) or 0.0),
            )
        )
    return out


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_GPU_TYPE",
    "DEFAULT_MODEL",
    "DEFAULT_PRECISION",
    "KernelBlasterAdapter",
    "KernelBlasterConfig",
    "KernelBlasterUnavailable",
]
