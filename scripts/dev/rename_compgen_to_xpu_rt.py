#!/usr/bin/env python3
"""Bulk rename `xpu_rt` → `xpu-rt` / `xpu_rt`, `CompGen` → `XPU-RT`.

Applies a series of case-sensitive substitutions across every tracked file
in the repo, except inside vendored upstream trees that must not be rewritten:
  - third_party/
  - merlin/
  - zephyr-chipyard-sw/
  - sims/IsaacLab/
  - .git/

The standalone token `xpu_rt` has two destinations depending on context:
  - Python identifier contexts (.py, .toml, .yaml, .yml, .json):
      \\bxpu_rt\\b → xpu_rt
  - Prose contexts (.md, .rst, .txt):
      \\bxpu_rt\\b → xpu-rt

Compound substrings (xpu_rt-mcp, COMPGEN_, libxpu_rt, mcp__xpu_rt__,
xpu_rt.X, xpu_rt/Y, "xpu_rt", 'xpu_rt') are handled with more specific
rules that run BEFORE the standalone-token rule.

After content rewrites, any file or directory whose name contains `xpu_rt`
is renamed via `git mv`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Paths to skip
# ---------------------------------------------------------------------------
SKIP_DIR_PREFIXES = (
    "third_party/",
    "merlin/",
    "zephyr-chipyard-sw/",
    "sims/IsaacLab/",
    ".git/",
)

# Skip binary file extensions (we only rewrite text content)
SKIP_EXTENSIONS = {
    ".pt", ".pt2", ".onnx", ".pth", ".npz", ".npy",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svg",
    ".so", ".dylib", ".dll", ".a", ".o", ".obj",
    ".whl", ".tar", ".gz", ".zip", ".bz2", ".xz",
    ".ico", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".bin", ".dat", ".mlir.bin", ".pyc", ".pyd",
    ".dlc", ".vmfb", ".jsonl",  # jsonl can be huge LLM logs
}

IDENTIFIER_EXTS = {".py", ".pyx", ".pxd", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini"}
PROSE_EXTS = {".md", ".rst", ".txt"}
# For everything else (shell, C/C++, mlir, etc.), default to identifier-style
# substitution — those are typically code contexts where snake_case is correct.


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, check=True, text=True,
                          capture_output=True, **kw)


def tracked_files() -> list[Path]:
    out = run(["git", "ls-files"]).stdout
    files = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(line.startswith(prefix) for prefix in SKIP_DIR_PREFIXES):
            continue
        p = REPO / line
        if p.suffix.lower() in SKIP_EXTENSIONS:
            continue
        files.append(p)
    return files


def is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


# Compiled-once substitutions. Order matters: compound rules first, standalone
# token last.
WORD_BOUND_COMPGEN = re.compile(r"\bxpu_rt\b")

COMPOUND_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bCompGen\b"),                       "XPU-RT"),
    (re.compile(r"\bxpu_rt-mcp\b"),                   "xpu-rt-mcp"),
    (re.compile(r"\bxpu_rt-run-conformance\b"),       "xpu-rt-run-conformance"),
    (re.compile(r"\bxpu_rt-gemini-usage\b"),          "xpu-rt-gemini-usage"),
    (re.compile(r"\bxpu_rt-tool\b"),                  "xpu-rt-tool"),
    (re.compile(r"\bxpu_rt_output\b"),                "xpu-rt-output"),
    (re.compile(r"\blibxpu_rt\b"),                 "libxpu_rt"),
    (re.compile(r"\bCOMPGEN_"),                        "XPU_RT_"),
    (re.compile(r"\bmcp__xpu_rt__"),                  "mcp__xpu_rt__"),
    (re.compile(r"\bxpu_rt\."),                       "xpu_rt."),
    (re.compile(r"\bxpu_rt/"),                        "xpu_rt/"),
    (re.compile(r'"xpu_rt"'),                         '"xpu-rt"'),
    (re.compile(r"'xpu_rt'"),                         "'xpu-rt'"),
]


def rewrite_text(text: str, is_prose: bool) -> tuple[str, int]:
    n = 0
    for pat, repl in COMPOUND_RULES:
        text, k = pat.subn(repl, text)
        n += k
    bare_repl = "xpu-rt" if is_prose else "xpu_rt"
    text, k = WORD_BOUND_COMPGEN.subn(bare_repl, text)
    n += k
    return text, n


def main() -> int:
    files = tracked_files()
    print(f"Scanning {len(files)} tracked files...", file=sys.stderr)

    total_subs = 0
    touched = 0
    for path in files:
        rel = path.relative_to(REPO)
        if is_binary(path):
            continue
        is_prose = path.suffix.lower() in PROSE_EXTS
        try:
            original = path.read_text()
        except UnicodeDecodeError:
            continue
        rewritten, n = rewrite_text(original, is_prose=is_prose)
        if n:
            path.write_text(rewritten)
            total_subs += n
            touched += 1

    print(f"Rewrote {total_subs} substitutions across {touched} files.", file=sys.stderr)

    # Now rename any file/directory whose path contains "xpu_rt".
    out = run(["git", "ls-files"]).stdout
    renames = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(line.startswith(p) for p in SKIP_DIR_PREFIXES):
            continue
        if "xpu_rt" not in line:
            continue
        # Rewrite the path: xpu_rt → xpu_rt for filesystem (Python dirs need
        # underscores even if they're inside the package). For non-Python
        # contexts like scripts/xpu_rt-mcp.sh, we want xpu-rt (kebab).
        new_path = line
        # File-system renames: same logic as standalone token rule, defaulting
        # to underscore for Python-y paths and kebab for shell scripts.
        if line.endswith(".sh") or line.endswith(".md") or line.endswith(".rst"):
            new_path = line.replace("xpu_rt", "xpu-rt")
        else:
            new_path = line.replace("xpu_rt", "xpu_rt")
        if new_path != line:
            renames.append((line, new_path))

    print(f"Renaming {len(renames)} paths via git mv...", file=sys.stderr)
    for old, new in renames:
        Path(REPO, new).parent.mkdir(parents=True, exist_ok=True)
        run(["git", "mv", old, new])

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
