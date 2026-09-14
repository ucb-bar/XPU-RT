#!/usr/bin/env python3
"""Contact sheet of every candidate in this directory, for side-by-side triage."""
from pathlib import Path
from PIL import Image, ImageDraw
import sys

HERE = Path(__file__).resolve().parent
files = sorted(HERE.glob("cand_*.png"))
COLS, CW, PAD, LAB = 4, 470, 10, 22
rows = (len(files) + COLS - 1) // COLS
thumbs = []
for f in files:
    im = Image.open(f).convert("RGB")
    s = CW / im.width
    thumbs.append((f.stem, im.resize((CW, max(1, int(im.height * s))), Image.LANCZOS)))
rh = [max(t[1].height for t in thumbs[r * COLS:(r + 1) * COLS]) for r in range(rows)]
Wd = COLS * (CW + PAD) + PAD
Ht = sum(h + LAB + PAD for h in rh) + PAD
sheet = Image.new("RGB", (Wd, Ht), "#f2f1ec")
d = ImageDraw.Draw(sheet)
y = PAD
for r in range(rows):
    x = PAD
    for name, im in thumbs[r * COLS:(r + 1) * COLS]:
        d.text((x + 2, y + 5), name, fill="#0b0b0b")
        sheet.paste(im, (x, y + LAB))
        d.rectangle([x, y + LAB, x + CW - 1, y + LAB + im.height - 1], outline="#c3c2b7")
        x += CW + PAD
    y += rh[r] + LAB + PAD
out = HERE / "contact_sheet.png"
sheet.save(out)
print(f"[ok] {out}  {sheet.size}  {len(files)} candidates")
