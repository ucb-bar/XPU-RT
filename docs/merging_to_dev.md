# Merging this branch into `dev`

Verified end to end on 2026-08-28 in a throwaway worktree. Three conflicts are
reported and **a fourth is not** — that one is the reason this file exists.

## The conflict that is not reported

`.gitmodules` auto-merges, and the auto-merge is wrong.

Both sides deleted three lines from it, so git takes both deletions and reports
success. But they are *different* three lines: `dev` removed the **ModelBlaster**
entry (Dima's `b66dd8c`), and this branch removed the **merlin** entry. Applying
both leaves `.gitmodules` with no ModelBlaster stanza while the ModelBlaster
*gitlink* — which is separately in conflict, and which you will resolve in
favour of keeping it — stays in the index.

A gitlink with no matching `.gitmodules` entry is a submodule that
`git submodule update --init` cannot resolve: it has a commit and no URL. Nobody
cloning the result can check ModelBlaster out, and nothing in the merge says so.

Reverting `b66dd8c` is a deliberate decision, not an accident of the merge:
ModelBlaster is a top-level submodule of XPU-RT again.

## The four resolutions

| path | resolution | why |
|---|---|---|
| `.gitmodules` | keep merlin's removal, **restore ModelBlaster's entry** | the two deletions are different lines; see above |
| `ModelBlaster` | **ours** | reverts `b66dd8c`; ModelBlaster is a submodule again |
| `pyproject.toml` | **ours** | ours is a strict superset — same 35 lines plus the `[solvers]` extra |
| `zephyr-chipyard-sw` | **theirs** | `b76ad31f` (2026-08-27 16:43) is newer than ours (11:48) and is not our work; the two have diverged, neither is an ancestor |

```bash
git merge origin/dev --no-commit --no-ff        # reports 3 of the 4

printf '[submodule "ModelBlaster"]\n\tpath = ModelBlaster\n\turl = https://github.com/ucb-bar/ModelBlaster.git\n' >> .gitmodules
git add .gitmodules

git checkout HEAD -- ModelBlaster && git add ModelBlaster
git checkout --ours pyproject.toml && git add pyproject.toml

git rm --cached -q zephyr-chipyard-sw
git update-index --add --cacheinfo \
    160000,$(git rev-parse origin/dev:zephyr-chipyard-sw),zephyr-chipyard-sw
```

## Check before committing the merge

Every `.gitmodules` entry must have a gitlink and every gitlink an entry — the
asymmetry is what the silent auto-merge produces, so check it explicitly:

```bash
python3 - <<'PY'
import configparser, subprocess
c = configparser.ConfigParser(); c.read(".gitmodules")
declared = {c[s]["path"] for s in c.sections()}
linked = {l.split("\t")[1] for l in
          subprocess.run(["git","ls-files","-s"],capture_output=True,text=True)
          .stdout.splitlines() if l.startswith("160000")}
assert declared == linked, (sorted(declared ^ linked))
print("submodules consistent:", sorted(declared))
PY

git diff --name-only --diff-filter=U          # must be empty
python -m pytest xpu-rt/tests/ -q
```

The suite on the merged tree gives **646 passed, 11 skipped** against 656/1 on
this branch. The 10 extra skips are not a regression: a fresh worktree has no
ModelBlaster checkout and no gitignored measured artifacts, and those tests skip
on exactly that. `pytest -rs` names each one.

## Order of operations

**Push ModelBlaster first.** The gitlink this merge records
(`ModelBlaster` at its `feat/k1-xpurt` head) does not resolve for anyone else
until that commit exists on the remote. Merging first produces a `dev` that is
broken for every other clone until the second push lands.
