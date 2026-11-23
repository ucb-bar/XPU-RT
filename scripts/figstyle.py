"""One visual language for every figure in this project.

WHY THIS IS A MODULE. The print rcParams block was copy-pasted verbatim into
five renderers, and the palette was bound independently in each -- so DroNet
was `#0072B2` (blue) in `plot_k1_evolution.py` and `#E69F00` (orange) in
`plot_interleaving.py`, while yolov8_nano was blue in the second. A reader
comparing two figures from the same run has no way to know that the blue bar in
one is not the blue bar in the other. Colour is an identity claim; it has to be
made in one place.

Sizes are Nature's: single column 89 mm, double 183 mm, and figures are built
AT final size rather than made large and shrunk, which is what keeps 6 pt type
6 pt.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import matplotlib as mpl

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))
import job_names as _job_names  # noqa: E402

MM = 1 / 25.4
SINGLE_COL = 89 * MM
DOUBLE_COL = 183 * MM
MAX_HEIGHT = 170 * MM

#: Okabe-Ito, colourblind-safe, in its canonical order.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"

OKABE_ITO = (BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILLION, PURPLE)

#: A model keeps ONE colour across every figure. yolov8_nano carries the
#: workload (97% of its runtime is one op kind, and it is the model every
#: granularity rung has been about) so it takes the strong blue; dronet is the
#: mid-weight co-runner; mlp_control is the small well-behaved one.
MODEL_COLOR = {
    "yolov8_nano": BLUE,
    "dronet": ORANGE,
    "mlp_control": GREEN,
    "fused_full": PURPLE,
    "lstm_tiny": SKY,
    "vitfly_frontend": YELLOW,
    "vitfly_lstm": VERMILLION,
}

#: Input-size variants of one detector. They are the same model at different
#: resolutions and must read as the same series across figures, so they share
#: its colour rather than each drawing a new one from the palette.
MODEL_ALIASES = ("yolov8_nano_64x96", "yolov8_nano_128x192",
                 "yolov8_nano_320", "yolov8_nano_64")
MODEL_COLOR.update({a: BLUE for a in MODEL_ALIASES})

#: Roles, not models. A deadline is always vermillion, whatever it belongs to.
C_DEADLINE = VERMILLION
C_MUTED = "#999999"
C_WARN = PURPLE
C_BASELINE = BLUE
C_REWRITE = VERMILLION

#: Where committed scripts write. Gitignored: a figure is a build product of
#: measured data plus a script, and both of those are in the repo.
FIGURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "out", "figures")


def model_color(name: str, fallback: str = C_MUTED) -> str:
    """The canonical colour for a model, tolerant of instance suffixes.

    THE EXACT NAME IS TRIED FIRST, and that ordering is the whole point. A
    network name can END IN DIGITS -- `yolov8_nano_64x96` is a real one in
    `networks_dense2` -- so stripping trailing digits to remove an instance
    suffix turns it into `yolov8_nano_64x`, which matches nothing. The model
    that carries the workload then renders in the fallback grey and the legend
    names a model that does not exist. Instance suffixes are only stripped
    when the full name is not itself a known model.
    """
    if name in MODEL_COLOR:
        return MODEL_COLOR[name]
    # The palette's own keys are the known network names, so the split needs
    # no extra argument here.
    return MODEL_COLOR.get(_job_names.model_of(name, MODEL_COLOR), fallback)


def model_of(job_name: str, known: "set[str] | None" = None) -> str:
    """`'yolov8_nano_64x960'` -> `'yolov8_nano_64x96'`, given the known set.

    Delegates to `xpu-rt/job_names.py`, which owns this split for the whole
    repo. It had been written independently seven times before that module
    existed, and the copies disagreed.
    """
    return _job_names.model_of(job_name, known)


def use() -> None:
    """Apply the print rcParams. Idempotent; call once at import."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6,
        "axes.labelsize": 6, "axes.titlesize": 7,
        "xtick.labelsize": 5, "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.06) -> None:
    """Lowercase bold panel label, placed consistently."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="right")


def save(fig, stem: str, out_dir: Optional[str] = None) -> str:
    """Write `<stem>.png` and `<stem>.pdf`; return the PNG path.

    Vector for the paper, raster for a terminal or a README. Both, always, so
    nobody has to re-run a script to get the other one.
    """
    out_dir = out_dir or FIGURE_DIR
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"),
                bbox_inches="tight", pad_inches=0.03)
    return png
