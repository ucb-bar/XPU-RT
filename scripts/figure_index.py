#!/usr/bin/env python3
"""A browsable contact sheet of every figure in results/codesign_feedback/refined/.

There are more than fifty renders in that directory -- several forms of the same figure, some current,
some superseded -- and a directory listing does not say which is which. This writes an HTML index with
a thumbnail per figure, grouped by family, carrying what the repository already knows about each one:
when it was written and by which script (from its sidecar), whether it has a sidecar at all (only a
figure with one can be machine-checked), whether the artifact's no-hardware run covers it, and which
one the paper carries.

Nothing here judges a figure. `scripts/verify_showdown_figure.py --metrics <sidecar>` is what decides
whether a figure re-derives; `docs/Artifact/artifact_checklist.md` is the record of what is current and what is
backlog. This page links to both.

    scripts/figure_index.py [--out results/codesign_feedback/refined/INDEX.html] [--thumb-px 900]
    scripts/figure_index.py --self-contained --out /tmp/figures.html
    scripts/figure_index.py --markdown --out results/codesign_feedback/refined/FIGURES.md

Thumbnails are regenerable, so they are written beside the page and ignored by git. --self-contained
inlines them as data: URIs instead, so the one file opens anywhere (copy it to a laptop and
double-click it); its links to the full-size PNG and PDF then only resolve beside the originals.
--markdown writes the same contact sheet as a .md, which VS Code renders with its built-in preview
(Ctrl+Shift+V) with no server and no extension -- the one form that works over Remote-SSH by itself.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import html
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.dirname(os.path.abspath(__file__)).rsplit("/scripts", 1)[0] + "/results"
REF = os.path.join(REPO, "results", "codesign_feedback", "refined")

# the figures the artifact's one-command check re-derives; read from the driver so the two cannot drift
def covered_figures():
    p = os.path.join(REPO, "artifact", "verify_no_hardware.sh")
    if not os.path.exists(p):
        return set()
    m = re.search(r'^FIGURES="\$\{FIGURES:-(.*?)\}"', open(p).read(), re.M)
    return set(m.group(1).split()) if m else set()


def paper_figures():
    """Which repo stems the paper's figures are, matched by content.

    The paper carries more than one figure from this tree, and matching by name would call a stem
    "in the paper" when the paper's own bytes are an older render of it -- which is true of two of
    them. So every file in the paper's plot directory is hashed against every render here, in both
    formats, and only an exact match counts.

    $XPURT_PAPER_PLOTS names that directory; $XPURT_PAPER_FIGURE stays supported for one file.
    Returns (stems that the paper carries, paper files that matched nothing).
    """
    import hashlib

    def sha(f):
        return hashlib.sha256(open(f, "rb").read()).hexdigest()

    # What the paper *includes*, not what sits in its plot directory: a figure can be cut from the
    # text and leave its file behind, and counting the file would report a figure the paper does
    # not carry. $XPURT_PAPER_PLOTS may be the plots/ directory or the paper root; the .tex is
    # looked for beside it either way.
    import re
    cands = []
    d = os.environ.get("XPURT_PAPER_PLOTS", "")
    if d and os.path.isdir(d):
        root = os.path.dirname(d.rstrip("/")) if os.path.basename(d.rstrip("/")) == "plots" else d
        tex = sorted(glob.glob(os.path.join(root, "*.tex"))
                     + glob.glob(os.path.join(root, "**", "*.tex"), recursive=True))
        named = set()
        for t in tex:
            try:
                body = open(t, errors="ignore").read()
            except Exception:
                continue
            for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
                named.add(m.strip())
        for g in sorted(named):
            f = os.path.join(root, g) if os.path.isabs(g) or "/" in g else os.path.join(d, g)
            for c in (f, os.path.join(d, os.path.basename(g))):
                if os.path.exists(c):
                    cands.append(c)
                    break
    one = os.environ.get("XPURT_PAPER_FIGURE", "")
    if one and os.path.exists(one):
        cands.append(one)
    if not cands:
        return set(), []

    # the whole results tree, not just refined/: loop_overview lives a level up, so indexing only
    # refined/ reported one of the paper's figures as matching nothing when it matches exactly
    here = {}
    for ext in ("png", "pdf"):
        for f in sorted(glob.glob(os.path.join(RES, "**", f"*.{ext}"), recursive=True)):
            if "/_thumbs/" in f:
                continue
            here.setdefault(sha(f), os.path.splitext(os.path.basename(f))[0])
    stems, unmatched = set(), []
    for c in cands:
        stem = here.get(sha(c))
        (stems.add(stem) if stem else unmatched.append(os.path.basename(c)))
    return stems, unmatched


def family(stem):
    """Group by the figure this render is a form of."""
    for pre, name in (("showdown_", "warehouse showdown — the audited set"),
                      ("warehouse_showdown_paper", "warehouse showdown — paper forms"),
                      ("warehouse_showdown_v3", "warehouse showdown — v3 forms"),
                      ("warehouse_showdown", "warehouse showdown — other forms"),
                      ("hil_envelope", "flight envelope"),
                      ("gantt", "schedules"),
                      ("schedule_evolution", "schedules")):
        if stem.startswith(pre):
            return name
    return "other figures"


def thumb(src, dst, px):
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return True
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(src)
        im.thumbnail((px, px * 4))
        im.convert("RGB").save(dst, quality=82)
        return True
    except Exception as e:                       # a render we cannot open is still listed, without a thumbnail
        print(f"   no thumbnail for {os.path.basename(src)}: {e}", file=sys.stderr)
        return False


def _inline(path):
    """a thumbnail as a data: URI, for a page that has to travel without its _thumbs directory."""
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REF, "INDEX.html"))
    ap.add_argument("--thumb-px", type=int, default=900)
    ap.add_argument("--self-contained", action="store_true",
                    help="inline the thumbnails as data: URIs so the page needs no sibling files")
    ap.add_argument("--markdown", action="store_true",
                    help="write Markdown instead of HTML, for VS Code's built-in preview")
    a = ap.parse_args()

    tdir = os.path.join(os.path.dirname(a.out), "_thumbs")
    os.makedirs(tdir, exist_ok=True)
    covered = covered_figures()
    paper, unmatched = paper_figures()

    rows = []
    for png in sorted(glob.glob(os.path.join(REF, "*.png"))):
        stem = os.path.splitext(os.path.basename(png))[0]
        side = os.path.join(REF, stem + "_metrics.json")
        meta = {}
        if os.path.exists(side):
            try:
                meta = json.load(open(side))
            except Exception:
                meta = {}
            if not isinstance(meta, dict):       # some sidecars are a list of records, not one object
                meta = {}
        t = os.path.join(tdir, stem + ".jpg")
        rows.append({
            "stem": stem, "png": os.path.basename(png),
            "pdf": os.path.basename(png).replace(".png", ".pdf") if os.path.exists(png[:-4] + ".pdf") else None,
            "thumb": (_inline(t) if a.self_contained else os.path.relpath(t, os.path.dirname(a.out))) if thumb(png, t, a.thumb_px) else None,
            "sidecar": os.path.basename(side) if os.path.exists(side) else None,
            "written": meta.get("written"), "script": meta.get("script"),
            "mb": os.path.getsize(png) / 1048576.0,
            "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(png)).strftime("%Y-%m-%d %H:%M"),
            "covered": stem in covered, "paper": stem in paper,
        })

    fams = {}
    for r in rows:
        fams.setdefault(family(r["stem"]), []).append(r)
    for v in fams.values():
        v.sort(key=lambda r: (not r["paper"], not r["covered"], r["written"] or "", r["stem"]), reverse=False)

    css = """body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:#fafafa;color:#181818}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:32px 0 10px;padding-bottom:5px;border-bottom:1px solid #ddd}
.lede{color:#555;max-width:62em;margin:0 0 6px} .lede code{background:#eee;padding:1px 4px;border-radius:3px}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}
.c{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:10px}
.c.paper{border-color:#1f9e5a;border-width:2px} .c img{width:100%;display:block;border:1px solid #eee;border-radius:3px}
.n{font-weight:600;margin:8px 0 3px;word-break:break-all} .m{color:#666;font-size:12px}
.t{display:inline-block;font-size:11px;padding:1px 6px;border-radius:9px;margin:4px 4px 0 0}
.paperT{background:#e6f5ec;color:#136b3c;border:1px solid #1f9e5a}
.ver{background:#eef3fb;color:#1c4e91;border:1px solid #b9cdea}
.side{background:#f3f0fa;color:#553d94;border:1px solid #cfc3ea}
.none{background:#faf0f0;color:#8c2f2f;border:1px solid #e3bcbc}
a{color:#1c4e91}"""

    o = ["<!doctype html><meta charset=utf-8><title>Figures — codesign_feedback/refined</title>",
         f"<style>{css}</style>", "<h1>Figures in <code>results/codesign_feedback/refined/</code></h1>",
         f"<p class=lede>{len(rows)} renders, {sum(1 for r in rows if r['sidecar'])} with a sidecar. "
         "A sidecar records the sha256 of every input a render read, so only a figure that has one can be "
         "re-derived by <code>scripts/verify_showdown_figure.py --metrics &lt;sidecar&gt;</code>. "
         "<b>artifact check</b> marks the figures <code>artifact/verify_no_hardware.sh</code> re-derives in "
         "one command. <code>docs/Artifact/artifact_checklist.md</code> §5 is the record of what is current and what "
         "is backlog — this page does not judge a figure, it only shows what is on disk.</p>",
         f"<p class=lede>Generated by <code>scripts/figure_index.py</code> on "
         f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. Click a thumbnail for the full render.</p>"]

    for fam in sorted(fams, key=lambda f: (f != "warehouse showdown — paper forms", f)):
        o.append(f"<h2>{html.escape(fam)} <span class=m>({len(fams[fam])})</span></h2><div class=g>")
        for r in fams[fam]:
            tags = []
            if r["paper"]:
                tags.append("<span class='t paperT'>in the paper</span>")
            if r["covered"]:
                tags.append("<span class='t ver'>artifact check</span>")
            tags.append("<span class='t side'>sidecar</span>" if r["sidecar"]
                        else "<span class='t none'>no sidecar</span>")
            prov = (f"{r['script']} · {r['written'][:16].replace('T', ' ')}"
                    if r["written"] and r["script"] else f"rendered {r['mtime']}")
            pdf = f" · <a href='{html.escape(r['pdf'])}'>pdf</a>" if r["pdf"] else ""
            img = (f"<a href='{html.escape(r['png'])}'><img loading=lazy src='{html.escape(r['thumb'])}'></a>"
                   if r["thumb"] else "<div class=m>(no thumbnail)</div>")
            o.append(f"<div class='c{' paper' if r['paper'] else ''}'>{img}"
                     f"<div class=n>{html.escape(r['stem'])}</div>"
                     f"<div class=m>{html.escape(prov)} · {r['mb']:.1f} MB"
                     f" · <a href='{html.escape(r['png'])}'>png</a>{pdf}</div>"
                     f"<div>{''.join(tags)}</div></div>")
        o.append("</div>")

    if a.markdown:
        o = [f"# Figures in `results/codesign_feedback/refined/`", "",
             f"{len(rows)} renders, {sum(1 for r in rows if r['sidecar'])} with a sidecar. A sidecar records the",
             "sha256 of every input a render read, so only a figure that has one can be re-derived by",
             "`scripts/verify_showdown_figure.py --metrics <sidecar>`. **artifact check** marks the figures",
             "`artifact/verify_no_hardware.sh` re-derives in one command. `docs/Artifact/artifact_checklist.md` §5 is the",
             "record of what is current and what is backlog — this page does not judge a figure, it only shows",
             "what is on disk.", "",
             f"Generated by `scripts/figure_index.py --markdown` on "
             f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}.", ""]
        for fam in sorted(fams, key=lambda f: (f != "warehouse showdown — paper forms", f)):
            o += [f"## {fam} ({len(fams[fam])})", ""]
            for r in fams[fam]:
                tags = (["**in the paper**"] if r["paper"] else []) \
                     + (["artifact check"] if r["covered"] else []) \
                     + (["sidecar"] if r["sidecar"] else ["no sidecar"])
                prov = (f"{r['script']} · {r['written'][:16].replace('T', ' ')}"
                        if r["written"] and r["script"] else f"rendered {r['mtime']}")
                links = f"[png]({r['png']})" + (f" · [pdf]({r['pdf']})" if r["pdf"] else "")
                o += [f"### {r['stem']}",
                      f"<img src=\"{r['thumb']}\" width=\"760\">" if r["thumb"] else "_(no thumbnail)_",
                      "", f"{prov} · {r['mb']:.1f} MB · {links} · {' · '.join(tags)}", ""]
    open(a.out, "w").write("\n".join(o) + "\n")
    print(f"wrote {a.out}: {len(rows)} figures, {sum(1 for r in rows if r['sidecar'])} with sidecars, "
          f"{sum(1 for r in rows if r['covered'])} in the artifact check"
          + (f", {len(paper)} carried by the paper" if paper else "")
          + (f", {len(unmatched)} paper file(s) matching no render here: "
             f"{', '.join(sorted(unmatched)[:4])}" if unmatched else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
