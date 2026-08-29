#!/usr/bin/env bash
# Reject an "IME" build that silently fell back to RVV.
#
# Selecting --hw IME sets +xsmtvdot, but nothing guarantees the compiler
# actually emitted smt.vmadot. If the op shapes do not match the matrix
# micro-tile, IREE lowers to ordinary RVV and the build succeeds -- producing a
# binary labelled IME that contains no IME instruction. Profiling that and
# calling it "IME performance" is how a scheduler ends up optimising against a
# number that means nothing.
#
# So the label has to be earned: disassemble and look for the instruction.
#
# Usage:
#   runtime/scripts/verify_ime_build.sh <dir-or-object> [...]
#
# Exit 0 only if every argument yields at least one vmadot.
#
# Note MLP legitimately fails this: every one of its matmuls is 1xNxK (M=1,
# i.e. GEMV), and a matrix engine has nothing to bite on. That is a property of
# the model, not a build error -- which is exactly why this check reports the
# count and lets the caller decide, rather than being wired into the compile.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Any llvm-objdump that knows riscv64 will do. Prefer one beside the
# SpaceMiT cross toolchain, then whatever is on PATH.
_TC="$(bash "${REPO_ROOT}/scripts/setup_spacemit_toolchain.sh" --path 2>/dev/null || true)"
OBJDUMP="${OBJDUMP:-${_TC%riscv64-unknown-linux-gnu-}llvm-objdump}"
command -v "${OBJDUMP}" >/dev/null 2>&1 || OBJDUMP="$(command -v llvm-objdump || true)"
MATTR="${MATTR:-+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot}"

if [[ ! -x "${OBJDUMP}" ]]; then
  echo "error: llvm-objdump not found at ${OBJDUMP}" >&2
  echo "       install llvm-objdump, or set OBJDUMP=" >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <dir-or-object> [...]" >&2
  exit 2
fi

rc=0
for arg in "$@"; do
  if [[ -d "${arg}" ]]; then
    mapfile -t objs < <(find "${arg}" -type f \( -name '*.so' -o -name '*.o' \) | sort)
  else
    objs=("${arg}")
  fi
  if [[ ${#objs[@]} -eq 0 ]]; then
    echo "FAIL  ${arg}: no .so/.o found (was --dump-artifacts passed?)"
    rc=1
    continue
  fi
  total=0
  for o in "${objs[@]}"; do
    n=$("${OBJDUMP}" -d --mattr="${MATTR}" "${o}" 2>/dev/null | grep -cE 'vmadot' || true)
    total=$((total + n))
    printf '  %-70s vmadot=%s\n' "$(basename "${o}")" "${n}"
  done
  if [[ ${total} -gt 0 ]]; then
    echo "PASS  ${arg}: ${total} vmadot instruction(s)"
  else
    echo "FAIL  ${arg}: no vmadot -- this build fell back to RVV, do not profile it as IME"
    rc=1
  fi
done
exit "${rc}"
