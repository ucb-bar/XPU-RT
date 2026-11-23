#!/usr/bin/env bash
# Fetch ONE file from the board over a link that keeps dropping: each attempt asks only for the bytes
# that are still missing (tail -c +offset over ssh, appended to the local file), until the local md5
# equals the wanted one. Read-only on the board. Used by job.sh for the run archive, and by hand when a
# job's own copy step gave up (the archive stays in our board folder).
# usage: fetch_resume.sh <host> <remote file> <local file> <wanted md5> [max attempts=40]
set -u
HOST="$1"; REMOTE="$2"; LOCAL="$3"; WANT="$4"; MAX="${5:-40}"
SSHO=(-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
md5of() { if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }
size() { wc -c < "$1" | tr -d ' '; }
for i in $(seq 1 "$MAX"); do
  have=0; [[ -f "$LOCAL" ]] && have=$(size "$LOCAL")
  if [[ -f "$LOCAL" && "$(md5of "$LOCAL")" == "$WANT" ]]; then echo "fetch_resume: complete after $((i - 1)) attempt(s), $have bytes, md5 $WANT"; exit 0; fi
  ssh -n "${SSHO[@]}" "$HOST" "tail -c +$((have + 1)) '$REMOTE'" >> "$LOCAL" 2>/dev/null
  now=$(size "$LOCAL")
  echo "fetch_resume: attempt $i: $have -> $now bytes"
  sleep 3
done
[[ "$(md5of "$LOCAL")" == "$WANT" ]] && { echo "fetch_resume: complete, md5 $WANT"; exit 0; }
echo "fetch_resume: GAVE UP after $MAX attempts (have $(size "$LOCAL") bytes, md5 $(md5of "$LOCAL"), want $WANT)"; exit 1
