#!/usr/bin/env bash
# Re-render every candidate. Each script is self-contained: it re-derives its
# numbers from the local data at draw time and writes one PNG beside itself.
# Nothing here touches a remote host, a GPU, or the paper's live fig_*.py.
set -u
cd "$(dirname "$0")"
python3 palette_check.py || echo "!! palette gate failed"
for f in cand_*.py; do
  printf '%-40s ' "$f"
  python3 "$f" >/dev/null 2>&1 && echo ok || { echo FAIL; python3 "$f" 2>&1 | tail -3; }
done
python3 make_contact_sheet.py
