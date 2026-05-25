"""
M14 — LLM-as-ranker for rewrite candidates.

Ranks candidates produced by ``rewrite.generate_candidates`` (M9). Two
backends:

  - ``anthropic`` (default): real LLM call via the Anthropic API
    (claude-haiku-4-5 by default). Requires ANTHROPIC_API_KEY env var.
    Every prompt + response is cached under ``data/llm_cache/<sha256>.json``
    so re-runs are free.

  - ``mock`` (no key needed): heuristic ranking that simulates the kind of
    multi-criteria reasoning an LLM would perform. Scores each candidate
    by weighted combinations of expected_benefit / expected_risk fields
    plus a small perturbation for tie-breaking. Useful for prototyping the
    integration without API spend.

The ranker plugs into ``run_closed_loop.py --ranker llm`` (see M10's
deterministic/random/cost_model ranker choices).

Public API:
  rank_candidates_via_llm(workload_summary, candidates, top_k, backend, model)
      -> ordered list of candidate_ids (best-first)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_cache"


def _summarize_workload(workload) -> Dict[str, Any]:
    from scheduler_ml import _lower_bound_makespan
    return {
        "n_ops": len(workload.operations),
        "n_machines": len(workload.machines),
        "machines": list(workload.machines),
        "n_combos": len(workload.get_machine_combinations()),
        "critical_path_us": float(_lower_bound_makespan(workload)),
        "n_with_deadlines": sum(1 for op in workload.operations
                                if op.deadline_us is not None),
        "n_with_release": sum(1 for op in workload.operations
                              if op.min_start_t is not None),
    }


def _candidate_summary(c) -> Dict[str, Any]:
    """Convert a Candidate (dataclass or dict) into a compact dict for the LLM."""
    if hasattr(c, "to_dict"):
        d = c.to_dict()
    else:
        d = dict(c)
    return {
        "candidate_id": d.get("candidate_id"),
        "type": d.get("type"),
        "affected_ops": d.get("affected_ops"),
        "expected_benefit": d.get("expected_benefit", {}),
        "expected_risk": d.get("expected_risk", {}),
    }


def _build_prompt(workload_summary: Dict[str, Any],
                  candidate_dicts: List[Dict[str, Any]],
                  top_k: int) -> str:
    return f"""You are a heterogeneous-SoC scheduler advisor.

Workload summary:
{json.dumps(workload_summary, indent=2)}

Available rewrite candidates ({len(candidate_dicts)} total):
{json.dumps(candidate_dicts, indent=2)}

Each candidate is a graph rewrite (fuse/split). expected_benefit and
expected_risk are heuristic estimates from the scheduler's analyzer.

Task: rank the top {top_k} candidates that are most likely to reduce
makespan WITHOUT introducing deadline misses or excessive memory pressure.
Consider:
  - candidates that reduce dispatch count when their estimated transfer
    savings exceed the lost parallelism
  - candidates that split heavy ops when the split exposes parallel
    placement on multiple devices
  - candidates that fuse only when the fused unit fits on a single device

Reply with a JSON array of up to {top_k} candidate_id strings, best first.
No explanation, just the JSON array. Example: ["fuse_a__b", "split_c"]
"""


def _cache_get(prompt: str, model: str) -> Optional[List[str]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((prompt + "|" + model).encode("utf-8")).hexdigest()
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get("ranking")
    except Exception:
        return None


def _cache_put(prompt: str, model: str, ranking: List[str]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((prompt + "|" + model).encode("utf-8")).hexdigest()
    with open(CACHE_DIR / f"{key}.json", "w") as f:
        json.dump({"model": model, "ranking": ranking,
                   "prompt_sha": key}, f, indent=2)


def _call_anthropic(prompt: str, model: str) -> List[str]:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # Strip code-fence if present
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back: extract bracketed list
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _mock_rank(workload_summary: Dict[str, Any],
               candidate_dicts: List[Dict[str, Any]],
               top_k: int) -> List[str]:
    """Heuristic that simulates LLM-style multi-criteria reasoning. Scores
    each candidate by a weighted sum of expected_benefit fields, penalizes
    risk fields. Deterministic given the same inputs.
    """
    rng = np.random.default_rng(0)
    scored = []
    for c in candidate_dicts:
        cb = c.get("expected_benefit", {})
        cr = c.get("expected_risk", {})
        # Positive score = wanted improvements; negative = penalties.
        # Note: predicted_makespan_delta is NEGATIVE for improvements.
        score = 0.0
        score += -float(cb.get("predicted_makespan_delta", 0.0))   # subtract negative -> add positive
        score += float(cb.get("saved_transfer_us", 0.0)) * 0.5
        score += float(cb.get("saved_dispatch_overhead_us", 0.0)) * 0.3
        score += -float(cb.get("dispatch_count_delta", 0.0)) * 5    # negative delta = good
        # Risks
        if cr.get("lost_parallelism"):
            score -= 30
        if cr.get("lost_device_flexibility"):
            score -= 10
        score -= float(cr.get("scratchpad_pressure_increase", 0.0)) * 0.1
        # If workload has tight deadlines, penalize risky moves more.
        if workload_summary.get("n_with_deadlines", 0) > 0:
            if cr.get("lost_parallelism"):
                score -= 20
        scored.append((score, c["candidate_id"]))
    scored.sort(reverse=True)
    return [cid for _, cid in scored[:top_k]]


def rank_candidates_via_llm(workload,
                            candidates: List[Any],
                            *,
                            top_k: int = 5,
                            backend: str = "auto",
                            model: str = "claude-haiku-4-5") -> List[str]:
    """Return up to ``top_k`` candidate_ids in best-first order.

    ``backend``:
      - ``anthropic``: real LLM call (requires ANTHROPIC_API_KEY)
      - ``mock``: deterministic heuristic
      - ``auto``: anthropic if key present, else mock
    """
    workload_summary = _summarize_workload(workload)
    candidate_dicts = [_candidate_summary(c) for c in candidates]
    prompt = _build_prompt(workload_summary, candidate_dicts, top_k)

    if backend == "auto":
        backend = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock"

    if backend == "anthropic":
        cached = _cache_get(prompt, model)
        if cached is not None:
            return cached[:top_k]
        try:
            ranking = _call_anthropic(prompt, model)
            _cache_put(prompt, model, ranking)
            return ranking[:top_k]
        except Exception as exc:
            print(f"[llm_ranker] anthropic call failed ({exc}); falling back to mock")
            backend = "mock"

    if backend == "mock":
        ranking = _mock_rank(workload_summary, candidate_dicts, top_k)
        # Still cache so we get reproducibility.
        _cache_put("MOCK:" + prompt, "mock", ranking)
        return ranking

    raise ValueError(f"unknown backend: {backend}")
