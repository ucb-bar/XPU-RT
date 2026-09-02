#!/usr/bin/env bash
# Get the SpaceMiT cross toolchain, and print the CROSS export that uses it.
#
# WHY THIS EXISTS. Every board build needs riscv64-unknown-linux-gnu-gcc 14.3,
# and until now the only copy on this machine lived inside the `merlin`
# submodule -- so XPU-RT carried an entire compiler repo as a git dependency to
# get one tarball. merlin is not used by the live K1 path for anything else
# (its remaining references are the retired IREE runtime and prose), so the
# submodule is gone and this fetches the toolchain directly from the vendor.
#
# WHY THE VERSION MATTERS, and it is not a preference. GCC 13.2 -- which is
# what `CROSS` defaults to via chipyard's riscv-tools, and what you will get if
# you skip this -- MISCOMPILES the RVV intrinsics. It reorders two
# `__riscv_vsetvl_*` calls so a widening instruction executes under the narrow
# vtype:
#
#     vsetvli e32,m4     <- sets SEW=32
#     vsetvli e8,m1      <- clobbers it to SEW=8
#     vle8.v / vle8.v
#     vsext.vf4          <- ILLEGAL: widening 8->32 needs SEW=32
#
# The binary SIGILLs on the board with no stdout at all. It crashes rather than
# computing a wrong answer, so no past measurement is invalid -- but you lose a
# board slot to `dmesg` and a disassembly before you find out.
#
#   Usage:  eval "$(scripts/setup_spacemit_toolchain.sh)"
#           scripts/setup_spacemit_toolchain.sh --path     # just the prefix
#
# Everything it prints on stdout is shell-evaluable; progress goes to stderr.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIRNAME="spacemit-toolchain-linux-glibc-x86_64-v1.1.2"
URL="https://archive.spacemit.com/toolchain/${DIRNAME}.tar.xz"
DEST_ROOT="${SPACEMIT_TOOLCHAIN_ROOT:-${REPO_ROOT}/tools/riscv-tools-spacemit}"

PATH_ONLY=0
[[ "${1:-}" == "--path" ]] && PATH_ONLY=1

say() { echo "$@" >&2; }

# Search order: an explicit install (SPACEMIT_TOOLCHAIN_ROOT), then ours.
# There used to be a third candidate under merlin/build_tools/, from when
# merlin was a submodule here; the 6 GB tree it pointed at now lives at
# DEST_ROOT and merlin is gone, so the fallback would only ever have found
# a stale copy on a machine that predates the move.
CANDIDATES=(
    "${DEST_ROOT}/${DIRNAME}"
)
FOUND=""
for c in "${CANDIDATES[@]}"; do
    if [[ -x "${c}/bin/riscv64-unknown-linux-gnu-gcc" ]]; then FOUND="${c}"; break; fi
done

if [[ -z "${FOUND}" ]]; then
    say "spacemit toolchain not found; fetching ${DIRNAME}"
    say "  from ${URL}"
    mkdir -p "${DEST_ROOT}"
    ARCHIVE="${DEST_ROOT}/${DIRNAME}.tar.xz"
    if [[ ! -s "${ARCHIVE}" ]]; then
        curl -fL --retry 3 -o "${ARCHIVE}.part" "${URL}" >&2
        mv "${ARCHIVE}.part" "${ARCHIVE}"
    fi
    tar -xf "${ARCHIVE}" -C "${DEST_ROOT}" >&2
    FOUND="${DEST_ROOT}/${DIRNAME}"
    [[ -x "${FOUND}/bin/riscv64-unknown-linux-gnu-gcc" ]] || {
        say "FAILED: no gcc at ${FOUND}/bin after extraction"; exit 1; }
fi

VER="$("${FOUND}/bin/riscv64-unknown-linux-gnu-gcc" -dumpversion 2>/dev/null || echo "?")"
case "${VER}" in
    14.*|15.*) : ;;
    *) say "REFUSING: ${FOUND} is gcc ${VER}. 13.x reorders the RVV vsetvl"
       say "          intrinsics and the board binary SIGILLs. Need >= 14."
       exit 1 ;;
esac
say "spacemit toolchain: gcc ${VER} at ${FOUND}"

if [[ "${PATH_ONLY}" == "1" ]]; then
    echo "${FOUND}/bin/riscv64-unknown-linux-gnu-"
else
    echo "export CROSS=${FOUND}/bin/riscv64-unknown-linux-gnu-"
fi
