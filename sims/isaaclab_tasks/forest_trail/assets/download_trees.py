#!/usr/bin/env python3
"""Download Poly Haven tree assets (CC0) for the forest trail scene.

Usage:
    python3 sims/isaaclab_tasks/forest_trail/assets/download_trees.py

Downloads:
    pine_sapling_small (1k resolution, ~45 MB total):
      - pine_sapling_small_1k.usdc  (geometry, PBR-textured pine sapling ~1.5 m)
      - textures/pine_sapling_small_*_1k.png  (bark + twig PBR maps)

The forest_scene.py auto-detects the USDC and uses it in place of the
procedural fallback pine_tree.usda.  Run this once; the files are cached.
"""

import json
import urllib.request
from pathlib import Path

ASSETS_DIR = Path(__file__).parent
ASSET_ID = "pine_sapling_small"
RESOLUTION = "1k"

# Poly Haven's API/CDN rejects urllib's default "Python-urllib/x.y" UA with a
# 403 (curl and browser UAs pass fine) -- without this header every request
# fails before it gets anywhere near the actual asset.
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _download(url: str, dest: Path, *, show: bool = True) -> None:
    if dest.exists():
        if show:
            print(f"  skip (exists): {dest.name}")
        return
    if show:
        print(f"  downloading:  {dest.name}  ...", end=" ", flush=True)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    if show:
        kb = dest.stat().st_size // 1024
        print(f"{kb} KB")


def main() -> None:
    api_url = f"https://api.polyhaven.com/files/{ASSET_ID}"
    req = urllib.request.Request(api_url, headers=_HEADERS)
    with urllib.request.urlopen(req) as resp:
        files = json.load(resp)

    entry = files["usd"][RESOLUTION]["usd"]
    usdc_url = entry["url"]
    includes = entry.get("include", {})

    out_dir = ASSETS_DIR / ASSET_ID
    tex_dir = out_dir / "textures"
    out_dir.mkdir(exist_ok=True)
    tex_dir.mkdir(exist_ok=True)

    print(f"Downloading {ASSET_ID} ({RESOLUTION}) …")
    _download(usdc_url, out_dir / f"{ASSET_ID}_{RESOLUTION}.usdc")
    for rel_path, info in includes.items():
        fname = Path(rel_path).name
        _download(info["url"], tex_dir / fname)

    print("Done.")
    print(f"USDC: {out_dir / f'{ASSET_ID}_{RESOLUTION}.usdc'}")


if __name__ == "__main__":
    main()
