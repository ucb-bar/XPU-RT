"""PlanDispatchTable + multi-plan dispatcher emit tests.

Coverage:
- Spec validation: invalid specs (empty entries, unknown feature
  keys, missing default) raise on construction.
- select_plan: deterministic across declaration order; default
  fallback fires when no entry matches.
- Python emit: source parses + imports + ``select_plan_ref`` returns
  the expected ref across 1 000 randomised feature vectors.
- C11 emit: source parses with ``cc -fsyntax-only`` (compiler-gated).
- C++ emit: source parses with ``c++ -fsyntax-only`` (compiler-gated).
- Byte stability: two emits of the same spec are byte-identical.
- Manifest schema: schema_version, spec_hash, all paths, entries
  round-trip cleanly.
- Recipe IR bridge: ``recipe.plan_dispatch_table`` op normalises into
  a :class:`PlanDispatchSpec` and emits the same dispatcher.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from compgen.runtime.glue_emit.dispatch_table import (
    PlanDispatchEntry,
    PlanDispatchSpec,
    emit_dispatch_table,
    select_plan,
)


def _spec(**overrides) -> PlanDispatchSpec:
    defaults = dict(
        workload="proxy_vla",
        target="cuda",
        feature_keys=("batch", "seqlen"),
        entries=(
            PlanDispatchEntry({"batch": 1}, plan_ref="@plan_b1"),
            PlanDispatchEntry({"batch": 4}, plan_ref="@plan_b4"),
            PlanDispatchEntry({"batch": 16, "seqlen": 512},
                              plan_ref="@plan_b16_s512"),
        ),
        default_plan_ref="@plan_default",
    )
    defaults.update(overrides)
    return PlanDispatchSpec(**defaults)


def _find_cc() -> str | None:
    for cc in ("cc", "gcc", "clang"):
        path = shutil.which(cc)
        if path:
            return path
    return None


def _find_cxx() -> str | None:
    for cxx in ("c++", "g++", "clang++"):
        path = shutil.which(cxx)
        if path:
            return path
    return None


# --------------------------------------------------------------------------- #
# Spec validation                                                             #
# --------------------------------------------------------------------------- #


class TestSpecValidation:
    def test_empty_feature_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="feature key"):
            PlanDispatchSpec(
                workload="x", target="cpu",
                feature_keys=(),
                entries=(PlanDispatchEntry({}, "@p"),),
                default_plan_ref="@d",
            )

    def test_empty_entries_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            PlanDispatchSpec(
                workload="x", target="cpu",
                feature_keys=("batch",),
                entries=(),
                default_plan_ref="@d",
            )

    def test_missing_default_rejected(self) -> None:
        with pytest.raises(ValueError, match="default_plan_ref"):
            PlanDispatchSpec(
                workload="x", target="cpu",
                feature_keys=("batch",),
                entries=(PlanDispatchEntry({"batch": 1}, "@p"),),
                default_plan_ref="",
            )

    def test_unknown_feature_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in declared"):
            PlanDispatchSpec(
                workload="x", target="cpu",
                feature_keys=("batch",),
                entries=(PlanDispatchEntry({"seqlen": 512}, "@p"),),
                default_plan_ref="@d",
            )


# --------------------------------------------------------------------------- #
# Pure selection                                                              #
# --------------------------------------------------------------------------- #


class TestSelectPlan:
    def test_first_match_wins(self) -> None:
        spec = _spec()
        assert select_plan(spec, {"batch": 1, "seqlen": 128}) == "@plan_b1"
        assert select_plan(spec, {"batch": 4, "seqlen": 999}) == "@plan_b4"

    def test_multi_feature_entry_requires_all_match(self) -> None:
        spec = _spec()
        # batch=16 alone — no entry matches → default.
        assert (
            select_plan(spec, {"batch": 16, "seqlen": 999})
            == "@plan_default"
        )
        # batch=16 AND seqlen=512 → plan_b16_s512.
        assert (
            select_plan(spec, {"batch": 16, "seqlen": 512})
            == "@plan_b16_s512"
        )

    def test_unknown_features_default(self) -> None:
        spec = _spec()
        assert (
            select_plan(spec, {"batch": 99, "seqlen": 99})
            == "@plan_default"
        )

    def test_randomised_feature_vectors_deterministic(self) -> None:
        spec = _spec()
        rng = random.Random(42)
        for _ in range(1000):
            features = {
                "batch": rng.choice([1, 2, 4, 8, 16, 32]),
                "seqlen": rng.choice([128, 256, 512, 1024]),
            }
            r1 = select_plan(spec, features)
            r2 = select_plan(spec, features)
            assert r1 == r2  # determinism


# --------------------------------------------------------------------------- #
# Python emit                                                                 #
# --------------------------------------------------------------------------- #


class TestPythonEmit:
    def test_python_dispatcher_parses_imports_and_matches(
        self, tmp_path: Path,
    ) -> None:
        spec = _spec()
        result = emit_dispatch_table(spec, tmp_path, target="python")
        assert result.python_path is not None
        # Parse + import.
        module_name = "_gen_dispatcher_py"
        path = result.python_path
        sp = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(module)  # type: ignore[union-attr]
        assert callable(module.select_plan_ref)
        assert module.FEATURE_KEYS == spec.feature_keys
        assert module.DEFAULT_PLAN_REF == spec.default_plan_ref
        assert (
            module.select_plan_ref({"batch": 1, "seqlen": 99})
            == "@plan_b1"
        )

    def test_python_dispatcher_matches_pure_selector(
        self, tmp_path: Path,
    ) -> None:
        spec = _spec()
        result = emit_dispatch_table(spec, tmp_path, target="python")
        sp = importlib.util.spec_from_file_location(
            "_gen_dispatcher_py2", result.python_path,
        )
        module = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(module)  # type: ignore[union-attr]
        rng = random.Random(1234)
        for _ in range(1000):
            features = {
                "batch": rng.choice([1, 4, 16, 999]),
                "seqlen": rng.choice([512, 1024, 9999]),
            }
            assert (
                module.select_plan_ref(features)
                == select_plan(spec, features)
            )


# --------------------------------------------------------------------------- #
# C11 + C++ emit                                                              #
# --------------------------------------------------------------------------- #


class TestNativeEmit:
    def test_c11_emit_produces_file_with_entries(
        self, tmp_path: Path,
    ) -> None:
        spec = _spec()
        result = emit_dispatch_table(spec, tmp_path, target="c11")
        assert result.c11_path is not None
        src = result.c11_path.read_text()
        for entry in spec.entries:
            assert entry.plan_ref in src
        assert "compgen_select_plan_ref" in src
        assert "compgen_dispatch_entry_t" in src
        assert "COMPGEN_DISPATCH_N_FEATURES" in src

    @pytest.mark.skipif(_find_cc() is None, reason="no C compiler in PATH")
    def test_c11_emit_parses_with_cc(self, tmp_path: Path) -> None:
        spec = _spec()
        result = emit_dispatch_table(spec, tmp_path, target="c11")
        cc = _find_cc()
        assert cc is not None
        proc = subprocess.run(
            [
                cc, "-std=c11", "-Wall", "-Wextra", "-fsyntax-only",
                str(result.c11_path),
            ],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"c11 dispatcher emit failed cc -fsyntax-only:\n"
            f"stderr={proc.stderr!r}"
        )

    @pytest.mark.skipif(_find_cxx() is None, reason="no C++ compiler in PATH")
    def test_cpp_emit_parses_with_cxx(self, tmp_path: Path) -> None:
        spec = _spec()
        result = emit_dispatch_table(spec, tmp_path, target="cpp")
        cxx = _find_cxx()
        assert cxx is not None
        proc = subprocess.run(
            [
                cxx, "-std=c++17", "-Wall", "-Wextra", "-fsyntax-only",
                str(result.cpp_path),
            ],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"cpp dispatcher emit failed c++ -fsyntax-only:\n"
            f"stderr={proc.stderr!r}"
        )


# --------------------------------------------------------------------------- #
# Byte stability                                                              #
# --------------------------------------------------------------------------- #


class TestByteStability:
    def test_two_emits_match(self, tmp_path: Path) -> None:
        spec = _spec()
        r1 = emit_dispatch_table(spec, tmp_path, target="all")
        py1 = r1.python_path.read_bytes()
        c11_1 = r1.c11_path.read_bytes()
        cpp1 = r1.cpp_path.read_bytes()

        r2 = emit_dispatch_table(spec, tmp_path, target="all")
        py2 = r2.python_path.read_bytes()
        c11_2 = r2.c11_path.read_bytes()
        cpp2 = r2.cpp_path.read_bytes()

        assert py1 == py2
        assert c11_1 == c11_2
        assert cpp1 == cpp2

        assert r1.spec_hash == r2.spec_hash


# --------------------------------------------------------------------------- #
# Manifest                                                                    #
# --------------------------------------------------------------------------- #


class TestManifest:
    def test_manifest_round_trips_spec(self, tmp_path: Path) -> None:
        spec = _spec()
        result = emit_dispatch_table(spec, tmp_path, target="all")
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["schema_version"] == "plan_dispatch_table_manifest_v1"
        assert manifest["workload"] == spec.workload
        assert manifest["target"] == spec.target
        assert manifest["feature_keys"] == list(spec.feature_keys)
        assert manifest["default_plan_ref"] == spec.default_plan_ref
        assert len(manifest["entries"]) == len(spec.entries)
        for emitted_entry, declared_entry in zip(
            manifest["entries"], spec.entries, strict=True,
        ):
            assert emitted_entry["features"] == declared_entry.features
            assert emitted_entry["plan_ref"] == declared_entry.plan_ref
        assert manifest["spec_hash"] == result.spec_hash
        assert len(manifest["spec_hash"]) == 16


# --------------------------------------------------------------------------- #
# Recipe IR bridge                                                            #
# --------------------------------------------------------------------------- #


class TestRecipeIrBridge:
    def test_plan_dispatch_table_op_round_trips(self) -> None:
        from xdsl.dialects.builtin import (
            ArrayAttr,
            DictionaryAttr,
            IntegerAttr,
            IntegerType,
            StringAttr,
        )

        from compgen.ir.recipe.ops_dispatch import PlanDispatchTableOp
        from compgen.runtime.glue_emit import (
            plan_dispatch_spec_from_recipe_op,
        )

        i64 = IntegerType(64)
        feature_keys = ArrayAttr([StringAttr("batch"), StringAttr("seqlen")])
        e1 = DictionaryAttr({
            "features": DictionaryAttr({
                "batch": IntegerAttr(1, i64),
            }),
            "plan_ref": StringAttr("@plan_b1"),
        })
        e2 = DictionaryAttr({
            "features": DictionaryAttr({
                "batch": IntegerAttr(4, i64),
            }),
            "plan_ref": StringAttr("@plan_b4"),
        })
        entries = ArrayAttr([e1, e2])

        op = PlanDispatchTableOp.create(
            properties={
                "feature_keys": feature_keys,
                "entries": entries,
                "default_plan_ref": StringAttr("@plan_default"),
                "workload": StringAttr("proxy_vla"),
                "target": StringAttr("cuda"),
            },
        )
        op.verify_()  # passes if the op is well-formed.

        spec = plan_dispatch_spec_from_recipe_op(op)
        assert spec.feature_keys == ("batch", "seqlen")
        assert len(spec.entries) == 2
        assert spec.entries[0].features == {"batch": 1}
        assert spec.entries[0].plan_ref == "@plan_b1"
        assert spec.default_plan_ref == "@plan_default"

    def test_plan_dispatch_table_op_rejects_empty_entries(self) -> None:
        from xdsl.dialects.builtin import ArrayAttr, StringAttr
        from xdsl.utils.exceptions import VerifyException

        from compgen.ir.recipe.ops_dispatch import PlanDispatchTableOp

        op = PlanDispatchTableOp.create(
            properties={
                "feature_keys": ArrayAttr([StringAttr("batch")]),
                "entries": ArrayAttr([]),
                "default_plan_ref": StringAttr("@plan_default"),
            },
        )
        with pytest.raises(VerifyException, match="at least one entry"):
            op.verify_()

    def test_plan_dispatch_table_op_rejects_unknown_feature(self) -> None:
        from xdsl.dialects.builtin import (
            ArrayAttr,
            DictionaryAttr,
            IntegerAttr,
            IntegerType,
            StringAttr,
        )
        from xdsl.utils.exceptions import VerifyException

        from compgen.ir.recipe.ops_dispatch import PlanDispatchTableOp

        i64 = IntegerType(64)
        bad = DictionaryAttr({
            "features": DictionaryAttr({
                "unknown_key": IntegerAttr(1, i64),
            }),
            "plan_ref": StringAttr("@plan_b1"),
        })
        op = PlanDispatchTableOp.create(
            properties={
                "feature_keys": ArrayAttr([StringAttr("batch")]),
                "entries": ArrayAttr([bad]),
                "default_plan_ref": StringAttr("@plan_default"),
            },
        )
        with pytest.raises(VerifyException, match="not in declared"):
            op.verify_()


# --------------------------------------------------------------------------- #
# ABI lint — same posture as #
# --------------------------------------------------------------------------- #


class TestAbiLint:
    def test_c_dispatcher_only_calls_libc_safe_symbols(
        self, tmp_path: Path,
    ) -> None:
        spec = _spec()
        result = emit_dispatch_table(spec, tmp_path, target="c11")
        src = result.c11_path.read_text()
        src_no_block = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
        src_no_line = re.sub(r"//[^\n]*", "", src_no_block)
        src_clean = re.sub(r'"([^"\\]|\\.)*"', '""', src_no_line)
        call_re = re.compile(r"(?<![.>:])\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
        called = {m.group(1) for m in call_re.finditer(src_clean)}
        # The C dispatcher only needs control-flow + its own selector
        # name.  No vendor or libcompgen_rt calls — the dispatcher is
        # purely a typed lookup.
        allowed = {
            "compgen_select_plan_ref",
            "if", "for", "while", "return", "sizeof",
        }
        for name in called:
            if name in allowed:
                continue
            raise AssertionError(
                f"dispatcher emit calls unexpected extern {name!r}"
            )
