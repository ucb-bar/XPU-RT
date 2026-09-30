#!/usr/bin/env bash
# Rebuild showdown_36hz_solver_vs_rosallhart_s1006 from data already in the repository and the archives.
#
# Delegates to the one renderer rather than restating its arguments, so this cannot document
# a command the renderer no longer uses. Board runs and flights need hardware; see the
# reproduction page named in README.md.
set -eu
cd "$(dirname "$0")/../../../.."           # repository root
ONLY=showdown_36hz_solver_vs_rosallhart_s1006 bash scripts/render_audited_set.sh
