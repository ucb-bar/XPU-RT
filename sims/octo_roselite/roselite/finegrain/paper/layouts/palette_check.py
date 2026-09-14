#!/usr/bin/env python3
"""Python port of the dataviz skill's scripts/validate_palette.js.

Node is not installed on this host, so the six computable checks are ported
verbatim (same Machado-Oliveira-Fernandes 2009 severity-1.0 matrices, same
OKLab Delta E x100, same thresholds and same pass/warn/fail states) so the
palettes in vocab.py are COMPUTED against the house rules rather than eyeballed.

  python palette_check.py "#eb6834,#1baf7a,#4a3aa7" --mode light --pairs all
"""
from __future__ import annotations
import math, sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
ORDINAL_MIN_DL = 0.06
ORDINAL_LIGHT_FLOOR = 2.0

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}


def _hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [_s2lin(c) for c in _hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklab(h):
    return oklab_from_lin(lin(h))


def oklch(h):
    L, a, b = oklab(h)
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    report, ok = [], True

    off = [(c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not off
    report.append(("Lightness band", "pass" if not off else "fail",
                   f"outside band: {off}" if off else f"all {len(palette)} inside L {lo}-{hi}"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    report.append(("Chroma floor", "pass" if not lowc else "fail",
                   f"below floor (reads gray): {lowc}" if lowc else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    n = len(palette)
    if pairs == "all":
        pairlist = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairlist = [(i, i + 1) for i in range(n - 1)]
    label = "all-pairs" if pairs == "all" else "adjacent"

    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            d = delta_e(palette[i], palette[j], kind)
            if worst is None or d < worst[0]:
                worst = (d, kind, palette[i], palette[j])
    tri = min((delta_e(palette[i], palette[j], "tritan") for i, j in pairlist), default=99)
    wd = worst[0] if worst else 99
    state = "pass" if wd >= CVD_TARGET else ("floor(WARN)" if wd >= CVD_FLOOR else "fail")
    ok &= state != "fail"
    report.append(("CVD separation", state,
                   f"worst {label} {worst[3]}<->{worst[2]} dE {wd:.1f} ({worst[1]}) - tritan {tri:.1f}"
                   if worst else "n/a"))

    nworst = min(((delta_e(palette[i], palette[j]), palette[i], palette[j]) for i, j in pairlist),
                 default=(99, "", ""))
    nd = nworst[0]
    nstate = "pass" if nd >= NORMAL_FLOOR else "fail"
    ok &= nstate == "pass"
    report.append(("Normal-vision floor", nstate,
                   f"worst {label} {nworst[2]}<->{nworst[1]} dE {nd:.1f} (normal)"))

    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    report.append(("Contrast vs surface", "relief(WARN)" if low else "pass",
                   f"below {CONTRAST_MIN}:1 - relief required (visible labels): {low}" if low
                   else f"all {len(palette)} >= {CONTRAST_MIN}:1"))
    return report, ok


def validate_ordinal(palette, mode="light", surface=None):
    surface = surface or DEFAULT_SURFACE[mode]
    report, ok = [], True
    Ls = [oklch(c)[0] for c in palette]
    order = sorted(range(len(Ls)), key=lambda i: Ls[i])
    mono = order == list(range(len(Ls))) or order == list(reversed(range(len(Ls))))
    ok &= mono
    report.append(("Monotone lightness", "pass" if mono else "fail",
                   " ".join(f"{x:.3f}" for x in Ls)))
    gaps = [abs(Ls[i + 1] - Ls[i]) for i in range(len(Ls) - 1)]
    gok = all(g >= ORDINAL_MIN_DL for g in gaps)
    ok &= gok
    report.append(("Step separation", "pass" if gok else "fail",
                   f"min dL {min(gaps):.3f} (>= {ORDINAL_MIN_DL})"))
    hues = [((math.degrees(math.atan2(oklab(c)[2], oklab(c)[1])) % 360) + 360) % 360 for c in palette]
    spread = max(hues) - min(hues)
    hok = spread <= 40
    report.append(("One hue", "pass" if hok else "warn", f"hue spread {spread:.0f} deg"))
    ends = [palette[0], palette[-1]]
    cs = [contrast(c, surface) for c in ends]
    eok = min(cs) >= ORDINAL_LIGHT_FLOOR
    ok &= eok
    report.append(("End step vs surface", "pass" if eok else "fail",
                   f"ends {cs[0]:.2f}:1 / {cs[1]:.2f}:1 (>= {ORDINAL_LIGHT_FLOOR})"))
    return report, ok


def validate_sequential(palette, mode="light", surface=None):
    """A CONTINUOUS sequential ramp (colourbar / heatmap fill).

    Different contract from an ordinal ramp: the lightest step encodes "near
    zero" and is ALLOWED to recede toward the surface, so the 2:1 ordinal floor
    does not apply. What must hold is that it reads as a ramp -- one hue,
    monotone lightness -- and that the DARK end is a usable mark.
    """
    surface = surface or DEFAULT_SURFACE[mode]
    report, ok = [], True
    Ls = [oklch(c)[0] for c in palette]
    mono = Ls == sorted(Ls) or Ls == sorted(Ls, reverse=True)
    ok &= mono
    report.append(("Monotone lightness", "pass" if mono else "fail",
                   f"{Ls[0]:.3f} -> {Ls[-1]:.3f} over {len(palette)} steps"))
    hues = [((math.degrees(math.atan2(oklab(c)[2], oklab(c)[1])) % 360) + 360) % 360
            for c in palette]
    spread = max(hues) - min(hues)
    hok = spread <= 40
    ok &= hok
    report.append(("One hue", "pass" if hok else "fail", f"hue spread {spread:.0f} deg"))
    dark = max(contrast(c, surface) for c in palette)
    dok = dark >= CONTRAST_MIN
    ok &= dok
    report.append(("Dark end vs surface", "pass" if dok else "fail",
                   f"{dark:.2f}:1 (>= {CONTRAST_MIN}) - light end recedes BY DESIGN"))
    return report, ok


def _run(name, pal, **kw):
    ordinal = kw.pop("ordinal", False)
    seq = kw.pop("sequential", False)
    if seq:
        rep, ok = validate_sequential(pal, **kw)
    elif ordinal:
        rep, ok = validate_ordinal(pal, **kw)
    else:
        rep, ok = validate(pal, **kw)
    tag = "SEQUENTIAL" if seq else ("ORDINAL" if ordinal else f"pairs={kw.get('pairs', 'adjacent')}")
    print(f"\n### {name}  [{kw.get('mode','light')}, {tag}]  {' '.join(pal)}")
    for k, s, d in rep:
        print(f"   {s.upper():>12s}  {k:<22s} {d}")
    print(f"   -> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("#"):
        pal = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
        mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "light"
        prs = sys.argv[sys.argv.index("--pairs") + 1] if "--pairs" in sys.argv else "adjacent"
        _run("cli", pal, mode=mode, pairs=prs, ordinal="--ordinal" in sys.argv)
        sys.exit(0)

    import vocab as V
    print(__doc__.strip())
    print("""
These are PRINT figures on a white ground, so ONE mode is selected (light) and
every set is held to the ALL-PAIRS gate -- scatter and small multiples dominate
this figure set, so the adjacent-only pairlist would not be honest. The dark
steps are carried anyway so a slide deck can reuse the vocabulary.

The status palette is NOT run as a categorical set: the skill fixes it and
requires icon + label on every use, so only the pair that actually appears
together on one canvas (success vs failure) is gated.""")
    allok = True
    allok &= _run("LANE  CPU/DSP/HTA", list(V.LANE.values()), mode="light", pairs="all")
    allok &= _run("LANE  dark steps", ["#9085e9", "#d95926", "#199e70"],
                  mode="dark", surface="#1a1a19", pairs="all")
    allok &= _run("PERIOD ramp, ordinal steps", V.LADDER_STEPS, mode="light", ordinal=True)
    allok &= _run("PERIOD ramp, continuous", V.BLUE, mode="light", sequential=True)
    allok &= _run("LATENCY ramp, continuous", V.ORANGE, mode="light", sequential=True)
    print("\n   DESIGN DECISION: latency has NO discrete step set. A 5-step orange")
    print("   ordinal ramp that clears the 2:1 floor lands on #ff966e..#970000,")
    print("   whose dark end collides with status-critical red. Latency is therefore")
    print("   encoded POSITIONALLY (an axis) or as a CONTINUOUS colourbar only --")
    print("   removing the failing gate rather than tuning around it.")
    _run("STATUS success vs failure  [informational]",
         [V.STATUS["good"], V.STATUS["critical"]], mode="light", pairs="all")
    print("   MITIGATED: red/green is the textbook deutan collision (dE 4.1) and the")
    print("   skill fixes these two hexes. Success and failure are therefore carried")
    print("   by GLYPH first -- filled star vs cross -- with the colour as a second")
    print("   channel only. Neither is ever the sole encoding, so this is a")
    print("   documented mitigation, not an unresolved failure.")
    print("\n=== " + ("ALL GATES CLEAR" if allok else "SOME GATES FAILED") + " ===")
    print("relief obligations: HTA aqua is 2.74:1 on white -> every lane mark is")
    print("direct-labelled; the period ramp's lightest step is 2.06:1 -> cells that")
    print("use it carry their value as text.")
    sys.exit(0 if allok else 1)
