#!/usr/bin/env bash
# Rebuild the audited figure set in a clean clone, from nothing but the repository and the archives.
#
# This is the test the in-place checks cannot do. verify_showdown_figure re-derives a figure against
# its own sidecar in a tree where every input already happens to be present; it says nothing about
# whether a reviewer who clones the branch and unpacks archive_v3/ has what a rebuild needs. Here the
# clone starts empty, only the named tars are unpacked into it, and every figure is re-rendered by
# scripts/render_audited_set.sh -- the same command the reproduction pages give.
#
#   scripts/repro_clean_clone.sh                 # see "Disk" below; removed by --clean
#   DEST=/elsewhere scripts/repro_clean_clone.sh
#   scripts/repro_clean_clone.sh --clean         # delete the clone and stop
#
# A regenerated sidecar is compared field by field against the committed one. Identical is the pass;
# any difference is a finding, printed in full.
#
# Disk. The clone is shallow, so its .git is small; what it costs is the checkout plus the archives
# it unpacks. 16.2 GB of results/ is tracked in this repository -- 190 energy-run .npz alone are
# ~9 GB, which the repository's own policy (docs/Artifact/artifact_checklist.md §1) says should be archived
# rather than tracked; they predate the ignore rule and stay tracked because ignoring does not
# untrack. Until they are archived and untracked, a checkout carries them.
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${DEST:-${TMPDIR:-/tmp}/repro-audit}"
CLONE="$DEST/xpurt"
AR="$SRC/results/codesign_feedback/archive_v3"
say(){ echo "=== $(date +%H:%M:%S) $*"; }

if [ "${1:-}" = "--clean" ]; then rm -rf "$DEST"; say "removed $DEST"; exit 0; fi

BRANCH=$(git -C "$SRC" rev-parse --abbrev-ref HEAD)
HEAD_SHA=$(git -C "$SRC" rev-parse HEAD)
say "cloning $BRANCH ($HEAD_SHA) into $CLONE"
rm -rf "$CLONE"; mkdir -p "$DEST"
# No --recurse-submodules: ModelBlaster's history runs to tens of GB (a run of this script reached
# 27 GB of clone before it was stopped) and nothing the audited set renders reads it. The one check
# that needs it -- pytest collecting tests/test_ime_profile_from_picks.py -- is a known gap already
# (docs/Artifact/artifact_checklist.md §5: the submodule pointer predates the IME work).
# --depth 1: a render test needs the tree, never the history, and this repository's history is 16 GB
# (the pre-squash past carries every large binary ever committed). Without it a clone that renders
# five figures costs 32 GB, half of it commits nobody reads.
# file:// is not decoration: git silently ignores --depth for a local-path clone and copies the whole
# object store instead ("--depth is ignored in local clones; use file:// instead"). With the path form
# this clone kept all 16 GB of history while reporting success.
git clone --quiet --depth 1 --no-tags --branch "$BRANCH" --single-branch "file://$SRC" "$CLONE" || exit 1
git -C "$CLONE" rev-parse HEAD | grep -q "$HEAD_SHA" || { echo "clone is not at $HEAD_SHA"; exit 1; }

# The tars the five figures need, and nothing else: the point is that this list is sufficient.
say "unpacking the archives the audited set reads"
# display_dumps_audited_set.tar holds all ten dumps this set reads, complete (figure_data.npz AND
# frames/, which panels a-d draw) and at repo-relative paths. The older per-campaign tars hold the
# same npz files but either omit frames/ or store relative paths -- see docs/Artifact/artifact_checklist.md.
# energy_dumps_v1.tar is not optional now that the energy .npz are archived rather than tracked:
# panel D's bars are labelled with each arm's control rate and hil_story_figure._cond_rate_hz reads
# that from energy_runs*/<cond>_s*/figure_data.npz at render time. Without it the labels are missing
# and the regenerated sidecar will not match.
for t in display_dumps_audited_set energy_dumps_v1 scene_records_v3 \
         board_traces_2026-09-17 board_traces_2026-09-24; do
  [ -f "$AR/$t.tar" ] || { echo "  MISSING $t.tar"; continue; }
  tar -xf "$AR/$t.tar" -C "$CLONE" && echo "  $t.tar"
done
# the raw traces untracked on 2026-09-29 (docs/Artifact/external_data.md, "Raw traces"): every Gantt
# row and every board constant --verify re-derives is read from these, so they are not optional either
for t in board_traces_ros_traced_2026-09-29 board_traces_xpurt_long_2026-09-29 raw_dumps_misc_2026-09-29; do
  [ -f "$AR/$t.tar.gz" ] || { echo "  MISSING $t.tar.gz"; continue; }
  tar -xzf "$AR/$t.tar.gz" -C "$CLONE" && echo "  $t.tar.gz"
done

# Not for the audited set -- the list above is still exactly what those five figures need. This one
# is for the repo-wide gate run at the end: verify_hil_feedback.py re-runs its producers, and they
# read five int8 graph IR files from ModelBlaster/build/, which that submodule gitignores as build
# output, so neither a clone nor the offline bundle has them. 200 KB.
[ -f "$AR/modelblaster_ir_v1.tar" ] && tar -xf "$AR/modelblaster_ir_v1.tar" -C "$CLONE" \
  && echo "  modelblaster_ir_v1.tar (for the gate's feedback checks, not the audited set)"

# The host environment. Building it needs a package index; when there is none, the existing venv is
# reused and this run tests the DATA rather than the environment -- which the log then says.
if [ -z "${HOST_PY:-}" ]; then
  say "building the host venv from requirements-host.txt"
  if python3 -m venv "$CLONE/.venv" >/dev/null 2>&1 &&
     "$CLONE/.venv/bin/pip" install --quiet -r "$CLONE/requirements-host.txt" >/dev/null 2>&1; then
    HOST_PY="$CLONE/.venv/bin/python"; say "venv built from the recorded requirements"
  else
    HOST_PY="$SRC/.venv/bin/python"
    say "NOTE: no package index reachable; reusing $HOST_PY. This run tests the data and the"
    say "      commands, NOT that the environment rebuilds from requirements-host.txt."
  fi
fi
export HOST_PY
say "python: $HOST_PY"

# The clone checks out the committed figures, so a render that fails would leave them in place and
# the comparison below would read "identical" from files git put there. Removing them first is what
# makes the comparison a test: a stem missing afterwards means the documented command did not produce
# it -- without the removal, six checked-out sidecars compare identical even when every render dies
# (e.g. on a missing frames/ entry).
say "removing the audited stems from the clone, so only a real render can put them back"
( cd "$CLONE" && rm -f results/codesign_feedback/refined/showdown_*.png \
                       results/codesign_feedback/refined/showdown_*.pdf \
                       results/codesign_feedback/refined/showdown_*_metrics.json )

say "rendering the audited set from the documented commands"
( cd "$CLONE" && bash scripts/render_audited_set.sh ) 2>&1 | sed 's/^/  /'
render_rc=${PIPESTATUS[0]}
[ "$render_rc" = 0 ] || say "NOTE: render_audited_set.sh exited $render_rc"

say "comparing every regenerated sidecar against the committed one"
"$HOST_PY" - "$SRC" "$CLONE" <<'PY'
import json, os, sys
src, clone = sys.argv[1], sys.argv[2]
R = "results/codesign_feedback/refined"
stems = ["showdown_36hz_solver_vs_rosallhart_s1006", "showdown_36hz_solver_vs_rosallhart_s1006_ladder",
         "showdown_45hz_pinned_vs_rosdefault_s1000", "showdown_45hz_pinned_vs_rosdefault_s1003",
         "showdown_45hz_pinned_vs_rospinned_s1011",  "showdown_45hz_solver_vs_rospinned_s1007"]
SKIP = {"written"}                    # a timestamp, not a measurement
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in SKIP: continue
            yield from flat(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    else:
        yield p, o
bad = 0
for st in stems:
    a, b = f"{src}/{R}/{st}_metrics.json", f"{clone}/{R}/{st}_metrics.json"
    if not os.path.exists(b):
        print(f"  {st}: NOT RENDERED in the clone"); bad += 1; continue
    da, db = dict(flat(json.load(open(a)))), dict(flat(json.load(open(b))))
    diff = [k for k in set(da) | set(db) if da.get(k) != db.get(k)
            and not (isinstance(da.get(k), str) and isinstance(db.get(k), str)
                     and da[k].replace(src, "") == db[k].replace(clone, ""))]
    if diff:
        print(f"  {st}: {len(diff)} field(s) differ"); bad += 1
        for k in sorted(diff)[:8]:
            print(f"      {k}\n        committed {da.get(k)!r}\n        clone     {db.get(k)!r}")
    else:
        print(f"  {st}: identical ({len(da)} fields)")
sys.exit(1 if bad else 0)
PY
cmp_rc=$?

# The archives are gitignored, so the clone has none of its own: the two checks that ask "is this
# untracked input in an archive" are pointed at the real archive directory, which is what a reviewer
# would have beside the clone. Without that they fail on every tar this run did not unpack, which
# says nothing about the artifact.
say "the archive checks, inside the clone, against $AR"
( cd "$CLONE" && "$HOST_PY" scripts/verify_archived_dumps.py --archives "$AR" 2>&1 | tail -1 )
( cd "$CLONE" && "$HOST_PY" scripts/verify_board_traces.py  --archives "$AR" 2>&1 | tail -1 )

say "the rest of the repo-wide checks, inside the clone"
( cd "$CLONE" && HOST_PY="$HOST_PY" ARCHIVES="$AR" SKIP_FIGURES=1 bash artifact/verify_no_hardware.sh ) 2>&1 | tail -12

say "REPRO_CLEAN_CLONE_DONE render_rc=${render_rc:-?} sidecar_compare_rc=$cmp_rc"
exit $(( cmp_rc || render_rc ))
