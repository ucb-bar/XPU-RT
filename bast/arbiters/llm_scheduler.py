"""
LLM-based scheduler for XPU-RT workloads.

Reads:
  schedules/training.md        — hand-maintained heuristics (authoritative).
  schedules/training_data.md   — append-only log of prior outputs + self-analyses.

Writes:
  schedules/training_data/<YYYY-MM-DD_HHMMSS>_<name>.json — the new schedule.
  schedules/training_data.md                               — appended with a
                                                             post-hoc analysis
                                                             of the new schedule.

Usage:
  python xpu-rt/llm_scheduler.py <workload.json> [--name NAME] [--backend claude|openai]

The workload JSON is passed verbatim to the LLM; the LLM returns schedule JSON
conforming to the format described in training.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

SCHEDULES_DIR = Path("schedules")
TRAINING_MD = SCHEDULES_DIR / "training.md"
TRAINING_DATA_MD = SCHEDULES_DIR / "training_data.md"
TRAINING_DATA_DIR = SCHEDULES_DIR / "training_data"


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #

def read_heuristics() -> str:
    if not TRAINING_MD.exists():
        raise FileNotFoundError(f"Missing heuristics file: {TRAINING_MD}")
    return TRAINING_MD.read_text()


def read_training_data() -> str:
    return TRAINING_DATA_MD.read_text() if TRAINING_DATA_MD.exists() else ""


def append_training_data(entry: str) -> None:
    TRAINING_DATA_MD.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_DATA_MD.open("a") as f:
        f.write(entry)


def write_schedule(schedule: dict, base_name: str) -> Path:
    """Write schedule JSON to schedules/training_data/<datetime>_<base_name>."""
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if not base_name.endswith(".json"):
        base_name += ".json"
    path = TRAINING_DATA_DIR / f"{stamp}_{base_name}"
    path.write_text(json.dumps(schedule, indent=2))
    return path


# --------------------------------------------------------------------------- #
# LLM backends — lazy imports, swap-compatible signature:
#   fn(prompt: str, system: str) -> str
# --------------------------------------------------------------------------- #

def _call_claude(prompt: str, system: str) -> str:
    import anthropic  # lazy

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _call_openai(prompt: str, system: str) -> str:
    import openai  # lazy

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


BACKENDS: dict[str, Callable[[str, str], str]] = {
    "claude": _call_claude,
    "openai": _call_openai,
}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def _system_prompt(heuristics: str, training_data: str) -> str:
    td_section = (
        f"\n# Prior outputs & self-analyses (chronological)\n{training_data}\n"
        if training_data.strip()
        else ""
    )
    return (
        "You are the XPU-RT LLM scheduler. Produce an optimal schedule that "
        "honours dependencies, per-core exclusivity, and periodic deadlines.\n\n"
        f"# Heuristics (authoritative)\n{heuristics}\n"
        f"{td_section}"
        "When asked to produce a schedule, respond with STRICT JSON only — "
        "no prose, no markdown fences."
    )


def _schedule_prompt(workload: str) -> str:
    return (
        "Produce the schedule JSON for the following workload. "
        "Return JSON only, matching the format in the heuristics.\n\n"
        f"Workload:\n{workload}"
    )


def _analysis_prompt(schedule: dict) -> str:
    return (
        "Analyse the schedule below using the self-analysis checklist from "
        "the heuristics. Respond with 3–5 terse bullets (markdown), no JSON.\n\n"
        f"Schedule:\n{json.dumps(schedule, indent=2)[:6000]}"
    )


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> dict:
    """Extract the outer JSON object from an LLM response (tolerates fences)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in response:\n{text[:400]}")
    return json.loads(text[start : end + 1])


def generate_schedule(workload_path: str | os.PathLike, backend: str = "claude") -> dict:
    """Call the LLM backend to produce a schedule dict for the given workload."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Choices: {list(BACKENDS)}")
    workload = Path(workload_path).read_text()
    system = _system_prompt(read_heuristics(), read_training_data())
    raw = BACKENDS[backend](_schedule_prompt(workload), system)
    return _extract_json(raw)


def analyse_and_log(schedule: dict, out_path: Path, backend: str = "claude") -> str:
    """Ask the LLM to critique the schedule, append the critique to training_data.md."""
    system = (
        "You are the XPU-RT scheduler's self-critic. Keep feedback concise, "
        "actionable, and tied to the heuristics."
    )
    analysis = BACKENDS[backend](_analysis_prompt(schedule), system).strip()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n\n## {stamp} — {out_path.name}\n\n{analysis}\n"
    append_training_data(entry)
    return analysis


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM-based XPU-RT scheduler")
    ap.add_argument("workload", help="Path to workload / dispatch-graph JSON")
    ap.add_argument("--name", default="llm_schedule.json",
                    help="Output filename base (datetime is prepended)")
    ap.add_argument("--backend", default="claude", choices=list(BACKENDS),
                    help="LLM backend to use")
    ap.add_argument("--no-analyse", action="store_true",
                    help="Skip the post-hoc self-analysis step")
    args = ap.parse_args(argv)

    schedule = generate_schedule(args.workload, backend=args.backend)
    out_path = write_schedule(schedule, args.name)
    print(f"Schedule saved to: {out_path}")

    if not args.no_analyse:
        analyse_and_log(schedule, out_path, backend=args.backend)
        print(f"Analysis appended to: {TRAINING_DATA_MD}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
