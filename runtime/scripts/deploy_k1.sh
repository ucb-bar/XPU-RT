#!/usr/bin/env bash
# Stage the board-side artifacts on the SpaceMiT K1.
#
# Why this exists: every runbook step that says "ssh k1 ./bin/<something>"
# assumes <something> is already on the board, and nothing in the tree put it
# there. `run_model_k1.sh` and `run_xpurt_k1.sh` each scp *their own* harness as
# a side effect of running it, so a fresh board is missing exactly the pieces
# you need *before* you can run anything -- the capability probe above all --
# and a fresh reader hits that wall on step one.
#
# What it does:
#   * builds ime_probe from the bring-up source if it is not built yet
#     (the only board-side binary with no other producer);
#   * copies an explicit manifest of local -> remote paths;
#   * verifies every file landed by comparing hashes, and says so per file;
#   * is idempotent -- a file whose remote hash already matches is skipped, so
#     re-running is cheap and safe.
#
# It reads and writes no credentials. Board access is whatever `ssh` is already
# configured to do for $MODELBLASTER_K1_HOST.
#
# Usage:
#   runtime/scripts/deploy_k1.sh [options] [extra-file[:remote-relpath] ...]
#
# Options:
#   --host <h>          override MODELBLASTER_K1_HOST
#   --remote-root <p>   override REMOTE_ROOT
#   --list              print the resolved manifest and exit (no ssh)
#   --dry-run           resolve + validate locally, but do not touch the board
#   --force             copy even when the remote hash already matches
#   --no-build          never invoke the cross compiler; skip missing built files
#   --only <pattern>    restrict the manifest to remote paths matching a glob
#   -h | --help         this text
#
# Environment:
#   MODELBLASTER_K1_HOST         ssh host           (default: k1)
#   REMOTE_ROOT                  board staging dir  (default: /root/mb_k1,
#                                or MODELBLASTER_K1_REMOTE_ROOT if set --
#                                the name the ModelBlaster runners already use)
#   CROSS                        cross toolchain prefix, used only to build
#                                ime_probe (default: riscv64-unknown-linux-gnu-)
#   PROBE_MARCH                  -march for ime_probe (default: rv64gcv)
#   DEPLOY_EXTRA                 whitespace-separated extra entries, same
#                                "local[:remote-relpath]" form as the positional
#                                arguments
#
# Exit status: 0 if every REQUIRED manifest entry is present on the board with a
# matching hash; non-zero otherwise.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MB_ROOT="${MB_ROOT:-${REPO_ROOT}/ModelBlaster}"

HOST="${MODELBLASTER_K1_HOST:-k1}"
REMOTE_ROOT="${REMOTE_ROOT:-${MODELBLASTER_K1_REMOTE_ROOT:-/root/mb_k1}}"
CROSS="${CROSS:-riscv64-unknown-linux-gnu-}"
PROBE_BUILD_DIR="${PROBE_BUILD_DIR:-${REPO_ROOT}/build/k1/bin}"

DO_LIST=0
DRY_RUN=0
FORCE=0
NO_BUILD=0
ONLY=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)        HOST="$2"; shift 2 ;;
        --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
        --list)        DO_LIST=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --force)       FORCE=1; shift ;;
        --no-build)    NO_BUILD=1; shift ;;
        --only)        ONLY="$2"; shift 2 ;;
        -h|--help)     sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)            echo "unknown option: $1" >&2; exit 2 ;;
        *)             EXTRA+=("$1"); shift ;;
    esac
done
if [[ -n "${DEPLOY_EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA+=(${DEPLOY_EXTRA})
fi

# ---------------------------------------------------------------------------
# 1. Build the one artifact that has no other producer.
#
# ime_probe is the board's capability oracle: /proc/cpuinfo does not enumerate
# the vendor IME extension, so the only honest test is to execute smt.vmadot
# under a SIGILL handler on each core. Its source lives with the bring-up
# capture rather than in a build tree, so nothing else ever compiles it.
# ---------------------------------------------------------------------------
PROBE_SRC="$(ls -1d "${REPO_ROOT}"/artifacts/k1_bringup/*/ime_probe.c 2>/dev/null | sort | tail -1 || true)"
PROBE_BIN="${PROBE_BUILD_DIR}/ime_probe"

build_probe() {
    [[ -n "${PROBE_SRC}" ]] || { echo "note: no ime_probe.c under artifacts/k1_bringup/*/ -- skipping" >&2; return 1; }
    if [[ -x "${PROBE_BIN}" && "${PROBE_BIN}" -nt "${PROBE_SRC}" ]]; then
        return 0
    fi
    [[ "${NO_BUILD}" == "0" ]] || { echo "note: --no-build and ${PROBE_BIN} is stale/absent" >&2; return 1; }
    if ! command -v "${CROSS}gcc" >/dev/null 2>&1; then
        echo "error: ${CROSS}gcc not found. Set CROSS= to your riscv64 linux-gnu prefix," >&2
        echo "       or pass --no-build to skip ime_probe." >&2
        return 1
    fi
    mkdir -p "${PROBE_BUILD_DIR}"
    # -march must enable V: the probe *assembles* a vsetvli in order to find out
    # at RUNTIME whether the core traps on it, and the toolchain's default
    # rv64gc rejects the mnemonic outright ("extension `v' ... required"). The
    # flag decides what can be encoded, not what the hardware will accept --
    # which is the whole point of the probe.
    # -static: the board runs glibc 2.41 and would resolve this fine either way,
    # but every other binary we ship is static and a mixed rule is a trap.
    if ! "${CROSS}gcc" -O2 -static -march="${PROBE_MARCH:-rv64gcv}" -mabi=lp64d \
            -o "${PROBE_BIN}" "${PROBE_SRC}"; then
        echo "error: failed to build ${PROBE_BIN}" >&2
        rm -f "${PROBE_BIN}"
        return 1
    fi
    echo "built ${PROBE_BIN} from ${PROBE_SRC#"${REPO_ROOT}"/}"
}

# ---------------------------------------------------------------------------
# 2. The manifest. One line per file: <local>|<remote-relpath>|<required>
#
# Remote paths are deliberately the ones the existing runners already use:
#   <root>/bin/<name>     validation/k1_runner.py deploys single-model harnesses
#                         here, so ad-hoc reruns of a harness live beside the
#                         probe.
#   <root>/xpurt/<name>   ModelBlaster/scripts/run_xpurt_k1.sh deploys the
#                         schedule-driven multi-model harness here.
# Keeping to those two directories means this script never fights the runners
# over layout, and a stale binary is always overwritten in place.
# ---------------------------------------------------------------------------
MANIFEST=()
add() { MANIFEST+=("$1|$2|$3"); }

if build_probe; then
    add "${PROBE_BIN}" "bin/ime_probe" "required"
else
    add "${PROBE_BIN}" "bin/ime_probe" "optional"
fi

# Schedule-driven harnesses already cross-built locally. Pre-staging them is
# optional -- run_xpurt_k1.sh copies the one it just built -- but it is what
# lets you re-run a previous schedule without a rebuild.
while IFS= read -r bin; do
    [[ -n "${bin}" ]] || continue
    add "${bin}" "xpurt/$(basename "$(dirname "${bin}")")" "optional"
done < <(ls -1 "${MB_ROOT}"/build/k1_xpurt/_build/*/xpurt_harness 2>/dev/null || true)

# Single-model harnesses (build/k1/<model>_<quant>_<target>_harness).
while IFS= read -r bin; do
    [[ -n "${bin}" ]] || continue
    add "${bin}" "bin/$(basename "${bin}")" "optional"
done < <(ls -1 "${MB_ROOT}"/build/k1/*_harness 2>/dev/null || true)

for e in ${EXTRA+"${EXTRA[@]}"}; do
    local_path="${e%%:*}"
    rel="${e#*:}"
    [[ "${rel}" != "${e}" ]] || rel="bin/$(basename "${local_path}")"
    add "${local_path}" "${rel}" "required"
done

if [[ -n "${ONLY}" ]]; then
    FILTERED=()
    for row in ${MANIFEST+"${MANIFEST[@]}"}; do
        rel="${row#*|}"; rel="${rel%%|*}"
        # shellcheck disable=SC2053
        [[ "${rel}" == ${ONLY} ]] && FILTERED+=("${row}")
    done
    MANIFEST=(${FILTERED+"${FILTERED[@]}"})
fi

if [[ ${#MANIFEST[@]} -eq 0 ]]; then
    echo "manifest is empty -- nothing to deploy." >&2
    echo "Build something first (ModelBlaster/scripts/run_model_k1.sh or run_xpurt_k1.sh)," >&2
    echo "or pass files explicitly: $0 path/to/binary[:bin/name]" >&2
    exit 1
fi

echo "host=${HOST}  remote-root=${REMOTE_ROOT}  entries=${#MANIFEST[@]}"

# ---------------------------------------------------------------------------
# 3. Local validation. A cross build that silently produced an x86 binary is a
#    real failure mode here (the conda envs on this host export their own
#    CC/LDFLAGS), and it is much cheaper to catch before the copy than as a
#    confusing "cannot execute binary file" on the board.
# ---------------------------------------------------------------------------
elf_is_riscv64() {
    local f="$1" magic machine class
    magic=$(od -An -tx1 -N4 "$f" 2>/dev/null | tr -d ' \n')
    [[ "${magic}" == "7f454c46" ]] || return 1          # \x7fELF
    class=$(od -An -tu1 -j4 -N1 "$f" | tr -d ' ')
    [[ "${class}" == "2" ]] || return 1                 # ELFCLASS64
    machine=$(od -An -tu1 -j18 -N1 "$f" | tr -d ' ')
    [[ "${machine}" == "243" ]] || return 1             # EM_RISCV
}

local_hash() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

declare -a ROWS_LOCAL=() ROWS_REL=() ROWS_REQ=() ROWS_HASH=() ROWS_SIZE=()
missing_required=0
for row in "${MANIFEST[@]}"; do
    lp="${row%%|*}"; rest="${row#*|}"; rel="${rest%%|*}"; req="${rest##*|}"
    if [[ ! -f "${lp}" ]]; then
        printf '  %-9s %-46s MISSING locally\n' "[${req}]" "${rel}"
        [[ "${req}" == "required" ]] && missing_required=1
        continue
    fi
    if ! elf_is_riscv64 "${lp}"; then
        printf '  %-9s %-46s NOT a riscv64 ELF: %s\n' "[${req}]" "${rel}" "${lp}"
        [[ "${req}" == "required" ]] && missing_required=1
        continue
    fi
    ROWS_LOCAL+=("${lp}")
    ROWS_REL+=("${rel}")
    ROWS_REQ+=("${req}")
    ROWS_HASH+=("$(local_hash "${lp}")")
    ROWS_SIZE+=("$(wc -c <"${lp}" | tr -d ' ')")
done

if [[ ${DO_LIST} -eq 1 || ${DRY_RUN} -eq 1 ]]; then
    echo "resolved manifest:"
    for i in "${!ROWS_REL[@]}"; do
        printf '  %-9s %-46s <- %s (%s bytes)\n' \
            "[${ROWS_REQ[$i]}]" "${REMOTE_ROOT}/${ROWS_REL[$i]}" \
            "${ROWS_LOCAL[$i]#"${REPO_ROOT}"/}" "${ROWS_SIZE[$i]}"
    done
    [[ ${missing_required} -eq 0 ]] || { echo "one or more REQUIRED entries are unusable" >&2; exit 1; }
    [[ ${DRY_RUN} -eq 1 ]] && echo "(dry run: the board was not contacted)"
    exit 0
fi
[[ ${missing_required} -eq 0 ]] || { echo "one or more REQUIRED entries are unusable" >&2; exit 1; }
[[ ${#ROWS_REL[@]} -gt 0 ]] || { echo "nothing deployable" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 4. One round trip to learn what is already there, so a no-op deploy costs a
#    single ssh and copies nothing.
# ---------------------------------------------------------------------------
remote_dirs=$(for rel in "${ROWS_REL[@]}"; do dirname "${REMOTE_ROOT}/${rel}"; done | sort -u | tr '\n' ' ')
probe_script="mkdir -p ${remote_dirs}
for f in"
for rel in "${ROWS_REL[@]}"; do probe_script+=" '${REMOTE_ROOT}/${rel}'"; done
probe_script+='; do
  if [ -f "$f" ]; then
    h=$(sha256sum "$f" 2>/dev/null | cut -d" " -f1)
    if [ -n "$h" ]; then echo "$f sha:$h"; else echo "$f size:$(wc -c <"$f" | tr -d " ")"; fi
  else
    echo "$f -"
  fi
done'

declare -A REMOTE_STATE=()
while read -r path state; do
    [[ -n "${path}" ]] && REMOTE_STATE["${path}"]="${state}"
done < <(ssh "${HOST}" 'bash -s' <<<"${probe_script}")

# ---------------------------------------------------------------------------
# 5. Copy what differs, then verify what we copied actually landed.
# ---------------------------------------------------------------------------
copied=0; skipped=0; failed=0
to_verify=()
for i in "${!ROWS_REL[@]}"; do
    rel="${ROWS_REL[$i]}"
    dst="${REMOTE_ROOT}/${rel}"
    want_sha="${ROWS_HASH[$i]}"
    want_size="${ROWS_SIZE[$i]}"
    have="${REMOTE_STATE[${dst}]:--}"
    if [[ ${FORCE} -eq 0 ]]; then
        if [[ "${have}" == "sha:${want_sha}" ]]; then
            printf '  = %-46s up to date\n' "${rel}"; skipped=$((skipped + 1)); continue
        fi
        if [[ "${have}" == "size:${want_size}" ]]; then
            # No sha256sum on the board: size match is weaker evidence, so say
            # so rather than silently calling it verified.
            printf '  = %-46s up to date (size only -- no sha256sum on board)\n' "${rel}"
            skipped=$((skipped + 1)); continue
        fi
    fi
    if ! scp -q "${ROWS_LOCAL[$i]}" "${HOST}:${dst}"; then
        printf '  ! %-46s scp FAILED\n' "${rel}"; failed=$((failed + 1)); continue
    fi
    to_verify+=("${i}")
    copied=$((copied + 1))
done

if [[ ${#to_verify[@]} -gt 0 ]]; then
    verify_script="chmod +x"
    for i in "${to_verify[@]}"; do verify_script+=" '${REMOTE_ROOT}/${ROWS_REL[$i]}'"; done
    verify_script+=$'\nfor f in'
    for i in "${to_verify[@]}"; do verify_script+=" '${REMOTE_ROOT}/${ROWS_REL[$i]}'"; done
    verify_script+='; do
  if [ -f "$f" ]; then
    h=$(sha256sum "$f" 2>/dev/null | cut -d" " -f1)
    if [ -n "$h" ]; then echo "$f sha:$h"; else echo "$f size:$(wc -c <"$f" | tr -d " ")"; fi
  else
    echo "$f -"
  fi
done'
    declare -A AFTER=()
    while read -r path state; do
        [[ -n "${path}" ]] && AFTER["${path}"]="${state}"
    done < <(ssh "${HOST}" 'bash -s' <<<"${verify_script}")

    for i in "${to_verify[@]}"; do
        rel="${ROWS_REL[$i]}"
        got="${AFTER[${REMOTE_ROOT}/${rel}]:--}"
        if [[ "${got}" == "sha:${ROWS_HASH[$i]}" ]]; then
            printf '  + %-46s deployed, sha256 verified\n' "${rel}"
        elif [[ "${got}" == "size:${ROWS_SIZE[$i]}" ]]; then
            printf '  + %-46s deployed, size verified (no sha256sum on board)\n' "${rel}"
        else
            printf '  ! %-46s VERIFY FAILED (remote=%s)\n' "${rel}" "${got}"
            failed=$((failed + 1))
            [[ "${ROWS_REQ[$i]}" == "required" ]] && missing_required=1
        fi
    done
fi

echo "deploy: ${copied} copied, ${skipped} already current, ${failed} failed"
if [[ ${failed} -gt 0 ]]; then
    exit 1
fi
echo "next: ssh ${HOST} ${REMOTE_ROOT}/bin/ime_probe    # confirms which cores carry IME"
