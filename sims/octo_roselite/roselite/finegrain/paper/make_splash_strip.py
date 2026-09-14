#!/usr/bin/env python3
"""The four task splashes side by side, as one transparent strip.

Panels are scaled to a COMMON HEIGHT, not a common width: the four scenes have different
aspect ratios (widowx 640x480, google 640x512, and the eggplant frame is cropped
differently again), so matching width would leave them visibly unequal in the row.

No added text, borders or gaps of a different colour -- the strip inherits whatever the
four figures already say, and stays transparent so it drops onto a light page.
"""
from pathlib import Path
import PIL.Image as I

HERE = Path(__file__).parent
PANELS = ["fig_splash.png", "fig_splash_spoon.png",
          "fig_splash_coke.png", "fig_splash_drawer.png"]
GAP = 26                      # transparent gutter, in output px
OUT = HERE / "fig_splash_strip.png"

ims = [I.open(HERE / p).convert("RGBA") for p in PANELS]
h = min(im.height for im in ims)
scaled = [im.resize((round(im.width * h / im.height), h), I.LANCZOS) for im in ims]
W = sum(im.width for im in scaled) + GAP * (len(scaled) - 1)

strip = I.new("RGBA", (W, h), (0, 0, 0, 0))
x = 0
for im in scaled:
    strip.alpha_composite(im, (x, 0))
    x += im.width + GAP
strip.save(OUT)

print(f"[ok] {OUT}")
print(f"  {len(scaled)} panels, common height {h}px -> {W}x{h}")
for p, im in zip(PANELS, scaled):
    print(f"    {p:24s} {im.width}x{im.height}")
