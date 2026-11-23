# Sourced by job.sh and golden_c.sh: run a command inside the K3 builder image.
#
# The builder is a container image with the SpacemiT cross-compiler and ModelBlaster's Python packages. Its
# entrypoint takes K3_IP and K3_USER and gives ssh inside the container a host alias for the board
# (K3TL_BUILDER_BOARD_ALIAS, default k3w), using the private key mounted at /run/host-ssh/id_ed25519.
# The ModelBlaster checkout and any extra folders are mounted at the SAME paths as on the host, so no path
# needs translating.
#
#   k3tl_builder <container name> <read-write folder or ""> <read-only folder or ""> <command ...>
#
# Env: K3TL_MB (the ModelBlaster checkout, absolute; set by the caller)
#      K3TL_BUILDER_IMAGE (default k3-builder), K3TL_DOCKER (default docker), K3TL_BUILDER_PLATFORM (default linux/amd64)
#      K3_SSH_KEY (default ~/.ssh/id_ed25519), K3_IP and K3_USER (the caller reads them from `ssh -G <alias>`)
k3tl_builder() {
  local name="$1" rw="$2" ro="$3"; shift 3
  local key="${K3_SSH_KEY:-$HOME/.ssh/id_ed25519}"
  local m=(--mount "type=bind,src=$K3TL_MB,dst=$K3TL_MB")
  [[ -n "$rw" ]] && m+=(--mount "type=bind,src=$rw,dst=$rw")
  [[ -n "$ro" ]] && m+=(--mount "type=bind,src=$ro,dst=$ro,readonly")
  [[ -f "$key" ]] && m+=(--mount "type=bind,src=$key,dst=/run/host-ssh/id_ed25519,readonly")
  [[ -f "$HOME/.ssh/known_hosts" ]] && m+=(--mount "type=bind,src=$HOME/.ssh/known_hosts,dst=/run/host-ssh/known_hosts,readonly")
  "${K3TL_DOCKER:-docker}" run --rm -i --platform "${K3TL_BUILDER_PLATFORM:-linux/amd64}" --name "$name" "${m[@]}" \
      -e K3_IP="${K3_IP:-not-used}" -e K3_USER="${K3_USER:-root}" "${K3TL_BUILDER_IMAGE:-k3-builder}" "$@"
}
