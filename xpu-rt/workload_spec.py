"""Reading one network entry out of a `data/toplevel/*.json` workload spec.

Four small questions get asked of a network entry all over this repo -- what
its profile basename is, what model names its profile might be filed under,
whether it is periodic, and whether it is a windowed slice. Each had two or
three identical implementations (`profile_loader`, `profile_metrics`,
`worst_case_nonperiodic_duration`, `worst_case_periodic_window_fraction`),
which is how `_model_candidates` came to exist in two spellings that agree
today and had no reason to keep agreeing.

They are small enough to be worth centralising precisely because they are
small: each is the kind of thing that gets rewritten from memory rather than
looked up, and a profile lookup that silently searches the wrong model name
returns "no profile" rather than an error.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


def basename_from_dispatch_deps_path(path: str) -> str:
    """`gen_mb/vmfb/<m>/<t>/<hw>/<m>.int8/<m>.int8_dispatch_graph.json` -> `<m>.int8`.

    The profile tree's second level is this basename, which is what lets a
    rewritten graph (`dronet.split_x2.int8`) be profiled beside its baseline
    (`dronet.int8`) instead of overwriting it.
    """
    return os.path.basename(os.path.dirname(path)) if path else ""


def model_candidates(net_key: str, net_info: Dict[str, Any],
                     basename_or_path: str = "") -> List[str]:
    """Model names to try when locating this network's profile, in order.

    Accepts either a basename or a full dispatch-deps path -- callers had one
    or the other and grew a variant each. A path is reduced to its basename
    first; anything without a separator is taken as already being one.
    """
    basename = (basename_from_dispatch_deps_path(basename_or_path)
                if os.sep in (basename_or_path or "") else basename_or_path)
    out: List[str] = []
    stem = os.path.basename(basename).split(".")[0] if basename else ""
    for c in (net_key, net_info.get("identifier"), stem):
        if isinstance(c, str) and c and c not in out:
            out.append(c)
    return out or [net_key]


def is_automatic_periodic(net_info: Dict[str, Any]) -> bool:
    """A network the scheduler releases on its own period."""
    return (net_info.get("period") is not None
            and net_info.get("window_duration") is not None)


def is_windowed_slice(net_info: Dict[str, Any]) -> bool:
    """A network pinned to an explicit `[min_start_t, max_end_t]` window."""
    return (net_info.get("min_start_t") is not None
            and net_info.get("max_end_t") is not None)


def window_ms(net_info: Dict[str, Any]) -> float | None:
    """The DEADLINE for this network: its declared window, else its period.

    `trace_metrics` uses `D = windows_ms.get(m, T)`, so omitting the window
    silently scores against the period -- a more forgiving test than the
    workload asked for.
    """
    w = net_info.get("window_duration", net_info.get("period"))
    return float(w) if w is not None else None


def windows_and_names(spec: Dict[str, Any]) -> tuple[Dict[str, float], set]:
    """`(window_ms per network, the set of real network names)` from a spec.

    The name set is what keeps `<network><instance>` from being split in the
    wrong place for a network whose own name ends in a digit -- see
    `job_names`.
    """
    nets = spec.get("networks") or {}
    windows = {}
    for key, info in nets.items():
        w = window_ms(info)
        if w is not None:
            windows[str(key)] = w
    return windows, set(nets)
