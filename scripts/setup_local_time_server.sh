#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <user@rpi-host-or-ip> [ssh_port] [laptop_ip]" >&2
  echo "Example: $0 ubuntu@192.168.1.42" >&2
  echo "Example with explicit laptop IP: $0 ubuntu@192.168.1.42 22 192.168.1.10" >&2
  exit 2
fi

TARGET="$1"
PORT="${2:-22}"
LAPTOP_IP="${3:-}"

if [[ -z "$LAPTOP_IP" ]]; then
  LAPTOP_IP="$(hostname -I | awk '{print $1}')"
fi

if [[ -z "$LAPTOP_IP" ]]; then
  echo "Could not determine laptop IP address." >&2
  echo "Pass it explicitly as the third argument." >&2
  exit 1
fi

echo "Laptop IP selected for NTP: $LAPTOP_IP"
echo "Configuring chrony on the laptop. You may be prompted for sudo."

if ! command -v chronyd >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y chrony
fi

if [[ ! -f /etc/chrony/chrony.conf.codex-backup ]]; then
  sudo install -m 0644 /etc/chrony/chrony.conf /etc/chrony/chrony.conf.codex-backup
fi

sudo tee /etc/chrony/conf.d/robot-laptop-time-server.conf >/dev/null <<'CONF'
# Serve the laptop clock to the robot on local/private networks.
allow 192.168.0.0/16
allow 10.0.0.0/8
allow 172.16.0.0/12

# Keep serving time even when the laptop has no internet NTP source.
local stratum 10
CONF

sudo systemctl restart chrony

echo "Configuring chrony on $TARGET ..."
ssh -p "$PORT" "$TARGET" "
  set -e
  if ! command -v chronyd >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y chrony
  fi
  if [ ! -f /etc/chrony/chrony.conf.codex-backup ]; then
    sudo install -m 0644 /etc/chrony/chrony.conf /etc/chrony/chrony.conf.codex-backup
  fi
  printf '%s\n' 'server $LAPTOP_IP iburst minpoll 4 maxpoll 4' | sudo tee /etc/chrony/conf.d/laptop-time-source.conf >/dev/null
  sudo timedatectl set-ntp true
  sudo systemctl restart chrony
  sudo chronyc -a makestep || true
  sleep 2
  date '+Raspberry Pi time: %Y-%m-%d %H:%M:%S %Z'
  chronyc sources -v
"

echo "Local time sync is configured. Keep the laptop on the robot network."
