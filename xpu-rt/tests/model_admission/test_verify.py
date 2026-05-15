"""verify-sources: HfApi mocked; assert YAML rewrite is correct in each branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from compgen.model_admission.verify import (
    VerifyStatus,
    apply_to_model_yaml,
    load_candidates,
    verify_one,
    verify_sources,
)


@dataclass
class _FakeModelInfo:
    id: str
    sha: str
    gated: bool = False
    private: bool = False


class _OkApi:
    def model_info(self, repo_id: str, *, token: str | None = None, revision: str | None = None) -> _FakeModelInfo:
        return _FakeModelInfo(id=repo_id, sha="0" * 40)


class _FakeRequest:
    method = "GET"
    url = "https://huggingface.co/api/models/x"


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.url = "https://huggingface.co/api/models/x"
        self.request = _FakeRequest()

    def json(self) -> dict:  # pragma: no cover -- not exercised here.
        return {}


class _GatedApi:
    def model_info(self, repo_id: str, *, token: str | None = None, revision: str | None = None):
        from huggingface_hub.errors import GatedRepoError

        raise GatedRepoError("repo is gated", response=_FakeResponse(403))


class _NotFoundApi:
    def model_info(self, repo_id: str, *, token: str | None = None, revision: str | None = None):
        from huggingface_hub.errors import RepositoryNotFoundError

        raise RepositoryNotFoundError("404 not found", response=_FakeResponse(404))


class _NetworkErrorApi:
    def model_info(self, repo_id: str, *, token: str | None = None, revision: str | None = None):
        raise OSError("connection refused")


def _write_model_yaml(path: Path, model_id: str) -> None:
    path.write_text(
        "schema_version: model_config_v1\n"
        f"model_id: {model_id}\n"
        "family: vlm\n"
        "source:\n"
        "  provider: huggingface\n"
        "  model_ref: TO_BE_VERIFIED_ONLINE\n"
        "  repo_url: TO_BE_VERIFIED_ONLINE\n"
        "  docs_url: TO_BE_VERIFIED_ONLINE\n"
        "  revision: null\n"
        "  source_verified: false\n"
        "loader:\n"
        "  kind: hf_transformers_vlm\n"
        "  device_policy: auto\n"
        "inputs:\n  kind: single_image_qa\n  processor_required: true\n"
        "compile:\n  mode: torch_compile_admission\n  backend: inductor\n  fullgraph: false\n  dynamic: true\n"
        "support:\n  mode: full_or_slice_smoke\n  full_model_blocking: true\n  reason: ''\n"
        "expected:\n"
        "  can_eager_run: true\n  can_torch_compile: true\n  can_dynamo_capture: true\n  can_slice_compile: true\n"
        "notes:\n  - 'Exact model_ref must be verified online before marking source_verified.'\n"
    )


def _write_candidates(path: Path, mapping: dict[str, str]) -> None:
    rows = [{"model_id": k, "candidate_ref": v} for k, v in mapping.items()]
    payload = {"schema_version": "model_admission_source_candidates_v1", "candidates": rows}
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_verify_one_ok():
    res = verify_one("qwen3_vl_8b", "Qwen/Qwen3-VL-8B-Instruct", _OkApi(), token=None)
    assert res.status == VerifyStatus.PASSED
    assert res.canonical_ref == "Qwen/Qwen3-VL-8B-Instruct"
    assert len(res.revision) == 40


def test_verify_one_gated():
    res = verify_one("llama4", "meta-llama/Llama-4", _GatedApi(), token=None)
    assert res.status == VerifyStatus.GATED


def test_verify_one_not_found():
    res = verify_one("unknown", "x/y", _NotFoundApi(), token=None)
    assert res.status == VerifyStatus.NOT_FOUND


def test_verify_one_network_error():
    res = verify_one("x", "x/y", _NetworkErrorApi(), token=None)
    assert res.status == VerifyStatus.NETWORK_ERROR


def test_verify_one_skipped_no_candidate():
    res = verify_one("x", "TO_BE_VERIFIED_ONLINE", _OkApi(), token=None)
    assert res.status == VerifyStatus.SKIPPED_NO_CANDIDATE


def test_apply_to_model_yaml_passed(tmp_path: Path):
    p = tmp_path / "qwen3_vl_8b.yaml"
    _write_model_yaml(p, "qwen3_vl_8b")
    from compgen.model_admission.verify import VerifyResult

    result = VerifyResult(
        model_id="qwen3_vl_8b",
        candidate_ref="Qwen/Qwen3-VL-8B-Instruct",
        canonical_ref="Qwen/Qwen3-VL-8B-Instruct",
        revision="a" * 40,
        status=VerifyStatus.PASSED,
    )
    changed = apply_to_model_yaml(p, result, verified_by="test@host", verified_at="2026-04-30T00:00:00+00:00")
    assert changed is True
    raw = yaml.safe_load(p.read_text())
    assert raw["source"]["source_verified"] is True
    assert raw["source"]["model_ref"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert raw["source"]["revision"] == "a" * 40
    assert raw["source"]["repo_url"].startswith("https://huggingface.co/")
    assert raw["source"]["verified_by"] == "test@host"
    # placeholder reminder note must be stripped.
    assert all("must be verified online" not in n.lower() for n in raw.get("notes") or [])


def test_apply_to_model_yaml_gated(tmp_path: Path):
    p = tmp_path / "qwen3_vl_8b.yaml"
    _write_model_yaml(p, "qwen3_vl_8b")
    from compgen.model_admission.verify import VerifyResult

    result = VerifyResult(
        model_id="qwen3_vl_8b",
        candidate_ref="Qwen/Qwen3-VL-8B-Instruct",
        status=VerifyStatus.GATED,
    )
    apply_to_model_yaml(p, result, verified_by="t", verified_at="2026-04-30T00:00:00+00:00")
    raw = yaml.safe_load(p.read_text())
    assert raw["source"]["source_verified"] is False
    assert raw["source"]["model_ref"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert any("gated" in n.lower() for n in raw["notes"])


def test_apply_to_model_yaml_not_found(tmp_path: Path):
    p = tmp_path / "x.yaml"
    _write_model_yaml(p, "x")
    from compgen.model_admission.verify import VerifyResult

    result = VerifyResult(
        model_id="x",
        candidate_ref="bogus/nonexistent",
        status=VerifyStatus.NOT_FOUND,
    )
    apply_to_model_yaml(p, result, verified_by="t", verified_at="2026-04-30T00:00:00+00:00")
    raw = yaml.safe_load(p.read_text())
    assert raw["source"]["source_verified"] is False
    assert any("not found upstream" in n.lower() for n in raw["notes"])


def test_verify_sources_end_to_end(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    _write_model_yaml(models / "qwen3_vl_8b.yaml", "qwen3_vl_8b")
    _write_model_yaml(models / "missing.yaml", "missing")
    candidates = tmp_path / "candidates.yaml"
    _write_candidates(
        candidates,
        {
            "qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
            "missing": "TO_BE_VERIFIED_ONLINE",
            "no_yaml": "x/y",
        },
    )

    run = verify_sources(candidates_path=candidates, models_dir=models, api=_OkApi(), token=None)
    by = run.by_status()
    assert by[VerifyStatus.PASSED.value] == 1
    assert by[VerifyStatus.SKIPPED_NO_CANDIDATE.value] == 1
    assert by[VerifyStatus.UNKNOWN_MODEL_ID.value] == 1
    assert run.written == ["qwen3_vl_8b"]


def test_verify_sources_skips_already_verified(tmp_path: Path):
    """Without --refresh, already-verified models should not call the API."""

    class _RaisingApi:
        called = 0

        def model_info(self, repo_id, *, token=None, revision=None):
            _RaisingApi.called += 1
            return _FakeModelInfo(id=repo_id, sha="b" * 40)

    models = tmp_path / "models"
    models.mkdir()
    p = models / "qwen3_vl_8b.yaml"
    _write_model_yaml(p, "qwen3_vl_8b")
    # Mark as already verified.
    raw = yaml.safe_load(p.read_text())
    raw["source"]["source_verified"] = True
    raw["source"]["revision"] = "a" * 40
    raw["source"]["model_ref"] = "Qwen/Qwen3-VL-8B-Instruct"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))

    cand = tmp_path / "candidates.yaml"
    _write_candidates(cand, {"qwen3_vl_8b": "Qwen/Qwen3-VL-8B-Instruct"})

    api = _RaisingApi()
    run = verify_sources(candidates_path=cand, models_dir=models, api=api, token=None)
    assert _RaisingApi.called == 0  # skipped the API call
    assert run.results[0].status == VerifyStatus.PASSED


def test_load_candidates_validates_schema(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("schema_version: wrong\ncandidates: []\n")
    with pytest.raises(ValueError):
        load_candidates(p)
