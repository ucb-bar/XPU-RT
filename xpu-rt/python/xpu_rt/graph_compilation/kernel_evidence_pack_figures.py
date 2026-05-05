"""M-25 figure renderers (matplotlib only; no seaborn).

Six PNGs under ``results/.../kernel_evidence_pack/figures/``:
- kernel_calibration_status_by_model.png
- register_pressure_distribution.png
- theoretical_occupancy_by_model.png
- bottleneck_classification_agreement.png
- compiled_us_per_iter_by_model.png
- fx_vs_kernel_joint_claim.png

Best-effort: missing data → figure skipped, never raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path: Path) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def render_kernel_calibration_status(
    rows: list[Any], out_path: Path,
) -> None:
    if not rows:
        return
    statuses = [r.m22_kernel_calibration_status for r in rows]
    counts: dict[str, int] = {}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1
    keys = sorted(counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(keys, [counts[k] for k in keys], color="#2a7ec2")
    ax.set_xlabel("kernel_calibration_status")
    ax.set_ylabel("model count")
    ax.set_title("M-22 kernel_calibration_status by model")
    plt.xticks(rotation=15)
    _save(fig, out_path)


def render_register_pressure_distribution(
    rows: list[Any], out_path: Path,
) -> None:
    """Histogram of register_pressure_mean across all models."""
    values = [
        r.m24_1_register_pressure_mean for r in rows
        if r.m24_1_register_pressure_mean is not None
    ]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=10, color="#3aa66e", edgecolor="white")
    ax.set_xlabel("register_pressure (mean across regions)")
    ax.set_ylabel("model count")
    ax.set_title("M-24.1 register_pressure distribution (per model)")
    _save(fig, out_path)


def render_theoretical_occupancy(
    rows: list[Any], out_path: Path,
) -> None:
    pairs = [
        (r.model_id, r.m24_1_theoretical_occupancy_mean)
        for r in rows
        if r.m24_1_theoretical_occupancy_mean is not None
    ]
    if not pairs:
        return
    pairs.sort(key=lambda x: x[1] or 0)
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(pairs))))
    ax.barh(labels, vals, color="#a06ec0")
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("theoretical_occupancy (0..1)")
    ax.set_title("M-24.1 theoretical_occupancy per model")
    _save(fig, out_path)


def render_bottleneck_agreement(
    rows: list[Any], out_path: Path,
) -> None:
    pairs = [
        (r.model_id, r.m22_agreement_count, r.m22_disagreement_count)
        for r in rows
        if (r.m22_agreement_count or r.m22_disagreement_count)
    ]
    if not pairs:
        return
    pairs.sort(key=lambda x: -(x[1] + x[2]))
    labels = [p[0] for p in pairs]
    agree = [p[1] for p in pairs]
    disagree = [p[2] for p in pairs]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(pairs))))
    ax.barh(labels, agree, color="#3aa66e", label="agree")
    ax.barh(labels, disagree, left=agree, color="#c0506e", label="disagree")
    ax.set_xlabel("region count")
    ax.set_title(
        "M-22 measured-vs-analytical bottleneck agreement (per model)"
    )
    ax.legend()
    _save(fig, out_path)


def render_compiled_us_per_iter(
    rows: list[Any], out_path: Path,
) -> None:
    pairs = [
        (
            r.model_id,
            r.m20_gpu_mean_us if r.m20_gpu_mean_us is not None else 0.0,
            r.m22_1_self_cuda_us_mean
            if r.m22_1_self_cuda_us_mean is not None else 0.0,
        )
        for r in rows
        if r.m20_gpu_mean_us is not None
        or r.m22_1_self_cuda_us_mean is not None
    ]
    if not pairs:
        return
    pairs.sort(key=lambda x: -(x[1] + x[2]))
    labels = [p[0] for p in pairs]
    m20 = [p[1] for p in pairs]
    m22_1 = [p[2] for p in pairs]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(pairs))))
    import numpy as np
    y = np.arange(len(labels))
    ax.barh(y - 0.2, m20, height=0.4, color="#2a7ec2", label="M-20 cuda.Event µs/iter")
    ax.barh(y + 0.2, m22_1, height=0.4, color="#3aa66e", label="M-22.1 self_cuda µs/iter")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("µs / iter")
    ax.set_title("Compiled kernel timing: cuda.Event vs torch.profiler")
    ax.legend()
    _save(fig, out_path)


def render_joint_claim_matrix(
    rows: list[Any], agg: dict[str, Any], out_path: Path,
) -> None:
    """6x3 grid: rows=joint claims, columns={fx_ready, kernel_ready, joint}."""
    n_models = max(1, len(rows))
    claims = list(range(1, 7))
    claim_names = {
        1: "precision", 2: "working_set", 3: "lifetime",
        4: "candidate_evidence", 5: "agent_view", 6: "bottleneck",
    }
    from compgen.graph_compilation.kernel_evidence_pack import (
        _M17_ROW_CLAIM, _M24_ROW_CLAIM,
    )
    fx_pass = []
    kr_pass = []
    joint_pass = []
    for row_idx in claims:
        fx_claim = _M17_ROW_CLAIM[row_idx]
        kr_claim = _M24_ROW_CLAIM[row_idx]
        joint_key = f"row_{row_idx}_{fx_claim}__AND__{kr_claim}"
        fx_pass.append(
            agg["m17_1_row_pass_count"].get(fx_claim, 0) / n_models
        )
        kr_pass.append(
            agg["m24_row_pass_count"].get(kr_claim, 0) / n_models
        )
        joint_pass.append(
            agg["joint_ready_count"].get(joint_key, 0) / n_models
        )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    import numpy as np
    x = np.arange(len(claims))
    w = 0.27
    ax.bar(x - w, fx_pass, width=w, color="#2a7ec2", label="FX (M-17.1)")
    ax.bar(x, kr_pass, width=w, color="#3aa66e", label="Kernel (M-24)")
    ax.bar(x + w, joint_pass, width=w, color="#c0506e", label="Joint")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"row {i}\n{claim_names[i]}" for i in claims],
        rotation=0, fontsize=8,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("model fraction (0..1)")
    ax.set_title("Joint FX × Kernel claim readiness")
    ax.legend()
    _save(fig, out_path)


def render_all_figures(
    rows: list[Any], agg: dict[str, Any], out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    render_kernel_calibration_status(
        rows, out_dir / "kernel_calibration_status_by_model.png"
    )
    render_register_pressure_distribution(
        rows, out_dir / "register_pressure_distribution.png"
    )
    render_theoretical_occupancy(
        rows, out_dir / "theoretical_occupancy_by_model.png"
    )
    render_bottleneck_agreement(
        rows, out_dir / "bottleneck_classification_agreement.png"
    )
    render_compiled_us_per_iter(
        rows, out_dir / "compiled_us_per_iter_by_model.png"
    )
    render_joint_claim_matrix(
        rows, agg, out_dir / "fx_vs_kernel_joint_claim.png"
    )
