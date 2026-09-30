#!/usr/bin/env bash
# Archive the display dumps a figure set reads, so a clean clone plus the archives can redraw it.
#
# figure_data.npz dumps are 100-200 MB each and are ignored by pattern rather than tracked, so
# scripts/verify_archived_dumps.py is what stands between "the figure was drawn from something" and
# "the figure was drawn from this". It looks up every untracked input a sidecar records and compares
# sha256 with what the render read; an input that is neither tracked nor archived is a FAIL, because
# without it the figure cannot be redrawn.
#
# This takes the tar to write and the dump directories to pack, so a new campaign is archived by
# naming it rather than by copying this file. scripts/archive_dumps_unpinned45.sh is the earlier
# per-campaign form, kept because the tar it wrote is in the manifest.
#
#   scripts/archive_dumps.sh <tar-basename> <dump-dir>...
#   scripts/archive_dumps.sh display_dumps_allcores36 results/.../pairs_ac36/*_figdata
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; A=$R/archive_v3
[ $# -ge 2 ] || { sed -n '2,20p' "$0"; exit 2; }
NAME="${1%.tar}"; shift
TAR=$A/$NAME.tar
say(){ echo "=== $(date +%H:%M:%S) $*"; }

DIRS=(); for d in "$@"; do [ -f "$d/figure_data.npz" ] && DIRS+=("$d") || say "no figure_data.npz in $d, skipped"; done
[ "${#DIRS[@]}" -gt 0 ] || { echo "no dumps to archive"; exit 1; }
say "${#DIRS[@]} dump dir(s) -> $(basename "$TAR")"
printf '   %s\n' "${DIRS[@]}"

# The whole dump directory, not figure_data.npz alone. A sidecar records the npz because that is what
# it hashes, but panels a-d draw frames/frame_NNN.npz -- so a tar of npz files alone lets
# verify_archived_dumps pass while a clean clone cannot render the figure at all. Proven by
# scripts/repro_clean_clone.sh, which failed on frames/frame_061.npz.
# Paths are repo-relative, so `tar -xf <tar> -C <repo>` puts every dump back where a render looks.
# Video is excluded: no render reads it, and it is regenerable from the frames.
say "packing (figure_data.npz + frames/, video excluded)"
tar -cf "$TAR" --exclude='*.mp4' --exclude='*.mkv' "${DIRS[@]}"
sz=$(stat -c%s "$TAR"); sha=$(sha256sum "$TAR" | cut -d' ' -f1)
grep -v "  $(basename "$TAR")\$" "$A/MANIFEST.sha256" > "$A/MANIFEST.sha256.new" 2>/dev/null || : > "$A/MANIFEST.sha256.new"
printf '%s  %14d  %s\n' "$sha" "$sz" "$(basename "$TAR")" >> "$A/MANIFEST.sha256.new"
sort -k3 "$A/MANIFEST.sha256.new" -o "$A/MANIFEST.sha256"; rm -f "$A/MANIFEST.sha256.new"
say "$(basename "$TAR") $sz bytes, sha256 ${sha:0:16}"
say "ARCHIVE_DONE $(basename "$TAR")"
