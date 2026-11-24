# Sourced by job.sh and golden_c.sh: run a build command, here or in a builder container.
#
# Default (K3TL_BUILDER_IMAGE unset or "none"): no container. The command runs on this machine, the way
# ModelBlaster's runners build anyway: CROSS from the environment, else this repository's
# scripts/setup_spacemit_toolchain.sh fetches the SpacemiT cross toolchain and sets it. Needs ModelBlaster's
# and XPU-RT's Python packages on this machine, and ssh to the board under the job's alias (K3TL_HOST).
#
# K3TL_BUILDER_IMAGE=<image>: run in that container image instead (for a machine that cannot run the
# toolchain, e.g. macOS). The image has the SpacemiT cross-compiler and ModelBlaster's Python packages; its
# entrypoint takes K3_IP and K3_USER and gives ssh inside the container a host alias for the board
# (K3TL_BUILDER_BOARD_ALIAS, default k3w), using the private key mounted at /run/host-ssh/id_ed25519.
# The ModelBlaster checkout and any extra folders are mounted at the SAME paths as on the host.
#
#   k3tl_builder <name> <read-write folder or ""> <read-only folder or ""> <command ...>
#   k3tl_builder_exec <name> <command ...>    run a command where <name>'s build runs (watchdog checks)
#   k3tl_builder_stop <name> <pid>            stop <name>'s build (<pid>: the k3tl_builder call put in background)
#   k3tl_board_alias <host>                   the ssh alias the build uses for the board
#
# Env: K3TL_MB (the ModelBlaster checkout, absolute; set by the caller)
#      K3TL_BUILDER_IMAGE (default: none = no container), K3TL_DOCKER (default docker),
#      K3TL_BUILDER_PLATFORM (default linux/amd64), K3_SSH_KEY (default ~/.ssh/id_ed25519),
#      K3_IP and K3_USER (container only; the caller reads them from `ssh -G <alias>`)
K3TL_XPURT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

k3tl_native() { [[ -z "${K3TL_BUILDER_IMAGE:-}" || "${K3TL_BUILDER_IMAGE}" == none ]]; }

k3tl_board_alias() { if k3tl_native; then echo "$1"; else echo "${K3TL_BUILDER_BOARD_ALIAS:-k3w}"; fi; }

k3tl_builder() {
  local name="$1" rw="$2" ro="$3"; shift 3
  if k3tl_native; then
    if [[ -z "${CROSS:-}" ]]; then
      local setup
      setup="$("$K3TL_XPURT_ROOT/scripts/setup_spacemit_toolchain.sh")" || { echo "no CROSS and the toolchain setup failed" >&2; return 9; }
      eval "$setup"; export CROSS
    fi
    "$@"
    return
  fi
  local key="${K3_SSH_KEY:-$HOME/.ssh/id_ed25519}"
  local m=(--mount "type=bind,src=$K3TL_MB,dst=$K3TL_MB")
  [[ -n "$rw" ]] && m+=(--mount "type=bind,src=$rw,dst=$rw")
  [[ -n "$ro" ]] && m+=(--mount "type=bind,src=$ro,dst=$ro,readonly")
  [[ -f "$key" ]] && m+=(--mount "type=bind,src=$key,dst=/run/host-ssh/id_ed25519,readonly")
  [[ -f "$HOME/.ssh/known_hosts" ]] && m+=(--mount "type=bind,src=$HOME/.ssh/known_hosts,dst=/run/host-ssh/known_hosts,readonly")
  "${K3TL_DOCKER:-docker}" run --rm -i --platform "${K3TL_BUILDER_PLATFORM:-linux/amd64}" --name "$name" "${m[@]}" \
      -e K3_IP="${K3_IP:-not-used}" -e K3_USER="${K3_USER:-root}" "$K3TL_BUILDER_IMAGE" "$@"
}

k3tl_builder_exec() {
  local name="$1"; shift
  if k3tl_native; then "$@"; else "${K3TL_DOCKER:-docker}" exec "$name" "$@"; fi
}

k3tl_kill_tree() { local p; for p in $(pgrep -P "$1" 2>/dev/null); do k3tl_kill_tree "$p"; done; kill "$1" 2>/dev/null; }

k3tl_builder_stop() {
  local name="$1" pid="$2"
  if k3tl_native; then k3tl_kill_tree "$pid"; else "${K3TL_DOCKER:-docker}" kill "$name" >/dev/null 2>&1; fi
}
