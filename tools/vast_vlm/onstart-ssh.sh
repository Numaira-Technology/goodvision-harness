#!/usr/bin/env bash
# Paste this entire file into the Vast SSH template's On-start Script.
# Keep Vast's existing keys and StrictModes; do not start models automatically.
set -euo pipefail

repair_ssh_permissions() {
  local root_home="$1"
  local ssh_dir="$root_home/.ssh"
  local keys="$ssh_dir/authorized_keys"
  # Refuse unexpected layouts rather than following links or creating empty keys.
  [[ -d "$root_home" && ! -L "$root_home" ]] || return 1
  [[ -d "$ssh_dir" && ! -L "$ssh_dir" ]] || return 1
  [[ -f "$keys" && ! -L "$keys" && -s "$keys" ]] || return 1
  chown root:root "$root_home" "$ssh_dir" "$keys"
  chmod go-w "$root_home"
  chmod 700 "$ssh_dir"
  chmod 600 "$keys"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  [[ "$(id -u)" == 0 ]] || { echo 'On-start requires root' >&2; exit 1; }
  repair_ssh_permissions /root
  echo 'SSH permissions repaired; existing authorized keys preserved.'
fi
