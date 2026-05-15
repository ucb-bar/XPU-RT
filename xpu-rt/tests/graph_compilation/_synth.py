"""Helpers to build synthetic capture/lower run directories for tests.

Every helper writes real files on disk and returns paths/hashes
suitable for assembling a ``run_manifest.json``. The validator must be
exercised against on-disk artifacts (not in-memory dicts) because R005
recomputes hashes from disk — that is the whole point of the contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from compgen.graph_compilation.hashing import sha256_file, sha256_tree


@dataclass
class StageBuild:
    """A pre-assembled stage record + its on-disk side-effects.

    Tests construct one of these per stage, then hand the list to
    :func:`build_well_formed_run` to produce a coherent run.
    """

    stage_id: str
    status: str
    inputs: list[dict[str, object]]
    outputs: list[dict[str, object]]
    report_path: str
    input_hash: str
    output_hash: str
    llm_calls: int = 0
    started_at_utc: str = "2026-04-30T00:00:00Z"
    finished_at_utc: str = "2026-04-30T00:00:01Z"

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "report_path": self.report_path,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "llm_calls": self.llm_calls,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
        }


def _zero_sha() -> str:
    return "0" * 64


def write_file(path: Path, content: bytes | str) -> dict[str, object]:
    """Write a file, return an ArtifactRef-shaped dict (relative path filled by caller)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode("utf-8")
    path.write_bytes(content)
    return {
        "path": "",  # caller fills
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "kind": "file",
    }


def write_tree(root: Path, files: dict[str, bytes | str]) -> dict[str, object]:
    """Write a tree of files; return ArtifactRef-shaped dict for the tree."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content = content.encode("utf-8")
        f.write_bytes(content)
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return {
        "path": "",
        "sha256": sha256_tree(root),
        "size_bytes": size,
        "kind": "tree",
    }


def make_stage0(run_dir: Path) -> StageBuild:
    """Synthesize a graph_capture stage with a small but realistic artifact set."""
    capture_dir = run_dir / "00_graph_capture"
    capture_dir.mkdir(parents=True, exist_ok=True)

    # exported_program.pt2 stand-in (just a non-empty blob)
    ep = capture_dir / "exported_program.pt2"
    ep_ref = write_file(ep, b"\x80\x02fake-exported-program")
    ep_ref["path"] = "00_graph_capture/exported_program.pt2"

    # capture report
    report = capture_dir / "capture_report.json"
    report_obj = {
        "schema_version": "graph_capture_report_v1",
        "status": "pass",
        "model_id": "synth_tiny",
    }
    report.write_text(json.dumps(report_obj), encoding="utf-8")

    # inputs hash chain seed: hash of model+target config
    seed_file = run_dir / ".inputs" / "stage0_inputs.bin"
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    seed_file.write_bytes(b"model+target+seed")
    input_hash = sha256_file(seed_file)
    output_hash = sha256_tree(capture_dir)

    return StageBuild(
        stage_id="graph_capture",
        status="pass",
        inputs=[],
        outputs=[ep_ref],
        report_path="00_graph_capture/capture_report.json",
        input_hash=input_hash,
        output_hash=output_hash,
    )


def make_stage1(run_dir: Path, prev_output_hash: str) -> StageBuild:
    lower_dir = run_dir / "01_payload_lowering"
    lower_dir.mkdir(parents=True, exist_ok=True)

    payload = lower_dir / "payload.mlir"
    payload_ref = write_file(payload, "module { func.func @main() { return } }\n")
    payload_ref["path"] = "01_payload_lowering/payload.mlir"

    report = lower_dir / "lowering_report.json"
    report.write_text(json.dumps({"schema_version": "payload_lowering_report_v1", "status": "pass"}), encoding="utf-8")

    output_hash = sha256_tree(lower_dir)
    return StageBuild(
        stage_id="payload_lowering",
        status="pass",
        inputs=[
            {
                "path": "00_graph_capture/exported_program.pt2",
                "sha256": sha256_file(run_dir / "00_graph_capture" / "exported_program.pt2"),
                "size_bytes": (run_dir / "00_graph_capture" / "exported_program.pt2").stat().st_size,
                "kind": "file",
            }
        ],
        outputs=[payload_ref],
        report_path="01_payload_lowering/lowering_report.json",
        input_hash=prev_output_hash,
        output_hash=output_hash,
    )


def make_stage2(run_dir: Path, prev_output_hash: str) -> StageBuild:
    analyze_dir = run_dir / "02_gap_discovery"
    analyze_dir.mkdir(parents=True, exist_ok=True)

    gap = analyze_dir / "gap_analysis.json"
    gap_ref = write_file(
        gap,
        json.dumps({"schema_version": "gap_analysis_v1", "regions_total": 1, "ops": []}),
    )
    gap_ref["path"] = "02_gap_discovery/gap_analysis.json"

    report = analyze_dir / "gap_report.json"
    report.write_text(
        json.dumps({"schema_version": "gap_discovery_summary_v1", "status": "pass"}),
        encoding="utf-8",
    )

    output_hash = sha256_tree(analyze_dir)
    return StageBuild(
        stage_id="gap_discovery",
        status="pass",
        inputs=[
            {
                "path": "01_payload_lowering/payload.mlir",
                "sha256": sha256_file(run_dir / "01_payload_lowering" / "payload.mlir"),
                "size_bytes": (run_dir / "01_payload_lowering" / "payload.mlir").stat().st_size,
                "kind": "file",
            }
        ],
        outputs=[gap_ref],
        report_path="02_gap_discovery/gap_report.json",
        input_hash=prev_output_hash,
        output_hash=output_hash,
    )


def write_manifest(run_dir: Path, stages: list[StageBuild], *, git_commit: str | None = None) -> Path:
    if git_commit is None:
        git_commit = "a" * 40
    manifest = {
        "schema_version": "run_manifest_v1",
        "run_id": "synth_run_001",
        "created_at_utc": "2026-04-30T00:00:00Z",
        "git_commit": git_commit,
        "model": {
            "config_path": "configs/models/synth.yaml",
            "model_id": "synth_tiny",
            "config_sha256": "1" * 64,
        },
        "target": {
            "config_path": "configs/targets/host_cpu.yaml",
            "target_id": "host_cpu",
            "config_sha256": "2" * 64,
        },
        "seed": 0,
        "stages": [s.to_dict() for s in stages],
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_ledger(run_dir: Path, stage_ids: list[str], *, drop_finish_for: str | None = None,
                 malformed_line: bool = False) -> Path:
    """Write a ``stage_ledger.jsonl`` covering ``stage_ids`` with start+finish events.

    ``drop_finish_for``: skip the ``finish`` event for the given stage (T10).
    ``malformed_line``: append a non-JSON line at the end (T14).
    """
    path = run_dir / "stage_ledger.jsonl"
    lines: list[str] = []
    for sid in stage_ids:
        for ev in ("start", "finish"):
            if ev == "finish" and sid == drop_finish_for:
                continue
            lines.append(
                json.dumps(
                    {
                        "schema_version": "stage_event_v1",
                        "stage_id": sid,
                        "event": ev,
                        "artifact_path": None,
                        "sha256": None,
                        "timestamp_utc": "2026-04-30T00:00:00Z",
                        "note": None,
                    }
                )
            )
    text = "\n".join(lines) + "\n"
    if malformed_line:
        text += "this is not json\n"
    path.write_text(text, encoding="utf-8")
    return path


def build_well_formed_run(run_dir: Path, *, num_stages: int = 3, git_commit: str | None = None) -> None:
    """Produce a complete, valid run with ``num_stages`` stages."""
    if num_stages < 1 or num_stages > 3:
        raise ValueError("num_stages must be in [1, 3]")

    stages: list[StageBuild] = []
    s0 = make_stage0(run_dir)
    stages.append(s0)
    if num_stages >= 2:
        stages.append(make_stage1(run_dir, prev_output_hash=s0.output_hash))
    if num_stages >= 3:
        stages.append(make_stage2(run_dir, prev_output_hash=stages[1].output_hash))

    write_manifest(run_dir, stages, git_commit=git_commit)
    write_ledger(run_dir, [s.stage_id for s in stages])
