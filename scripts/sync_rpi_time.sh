#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <user@rpi-host-or-ip> [ssh_port]" >&2
  echo "Example: $0 ubuntu@192.168.1.42" >&2
  exit 2
fi

TARGET="$1"
PORT="${2:-22}"
LAPTOP_EPOCH="$(date +%s)"
LAPTOP_TIME="$(date -d "@$LAPTOP_EPOCH" '+%Y-%m-%d %H:%M:%S %Z')"

echo "Laptop time: $LAPTOP_TIME"
echo "Setting Raspberry Pi time on $TARGET ..."

ssh -p "$PORT" "$TARGET" \
  "sudo timedatectl set-ntp false && sudo date -u -s '@$LAPTOP_EPOCH' >/dev/null"

echo "Raspberry Pi time after sync:"
ssh -p "$PORT" "$TARGET" "date '+%Y-%m-%d %H:%M:%S %Z'; timedatectl | sed -n '1,8p'"
