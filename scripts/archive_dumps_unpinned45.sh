#!/usr/bin/env bash
# Archive the display dumps this figure set reads, so a clean clone plus the archives can redraw it.
#
# figure_data.npz dumps are 100-200 MB each and are ignored by pattern rather than tracked, so
# scripts/verify_archived_dumps.py is what stands between "the figure was drawn from something" and
# "the figure was drawn from this". It looks up every untracked input a sidecar records and compares
# sha256 with what the render read; an input that is neither tracked nor archived is a FAIL, because
# without it the figure cannot be redrawn.
#
# This packs the dumps added for the 45 Hz solver-placed forms -- the matched display pairs and the
# seed-1000 pair that did not separate the arms, which is kept because it is a result -- and appends
# the tar's own sha256 to the tracked manifest.
#
#   scripts/archive_dumps_unpinned45.sh
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; A=$R/archive_v3
TAR=$A/display_dumps_unpinned45.tar
say(){ echo "=== $(date +%H:%M:%S) $*"; }

say "waiting for the display pairs to finish"
until grep -q "DISPLAY_PAIRS45_DONE" "$R/campaign_free45/display/pairs45/pairs.log" 2>/dev/null; do sleep 60; done

mapfile -t DIRS < <(ls -d $R/campaign_free45/display/pairs45/*_figdata \
                          $R/campaign_free45/display/pin_l1000_e1000/*_figdata 2>/dev/null)
[ "${#DIRS[@]}" -gt 0 ] || { echo "no dumps to archive"; exit 1; }
say "${#DIRS[@]} dump dir(s)"
printf '   %s\n' "${DIRS[@]}"

# only figure_data.npz is an input a sidecar records; the frames/ and mp4 beside it are not read by
# any render, and packing them would multiply the tar for nothing.
say "packing $TAR"
tar -cf "$TAR" $(printf '%s/figure_data.npz ' "${DIRS[@]}")
sz=$(stat -c%s "$TAR"); sha=$(sha256sum "$TAR" | cut -d' ' -f1)
grep -v "  $(basename "$TAR")\$" "$A/MANIFEST.sha256" > "$A/MANIFEST.sha256.new" 2>/dev/null || : > "$A/MANIFEST.sha256.new"
printf '%s  %14d  %s\n' "$sha" "$sz" "$(basename "$TAR")" >> "$A/MANIFEST.sha256.new"
sort -k3 "$A/MANIFEST.sha256.new" -o "$A/MANIFEST.sha256"; rm -f "$A/MANIFEST.sha256.new"
say "$(basename "$TAR") $sz bytes, sha256 ${sha:0:16}"

say "verifying every untracked figure input is archived and matches"
.venv/bin/python scripts/verify_archived_dumps.py 2>&1 | tail -6
say "ARCHIVE_UNPINNED45_DONE"
