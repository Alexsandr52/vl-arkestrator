#!/usr/bin/env bash
# Deploy vl-arkestrator and the pinned GR00T-pickup-bottle checkout to g1-gpu.
#
# Copies committed code only (git archive / git bundle), then links the bottle operator's
# reviewed local configs and the shared lock directory into the pinned checkout, the same way
# ~/pre_kpi/_bottle_localdev links .xr -> ../_bottle/.xr. Sends nothing to the robot,
# does not start any process and does not touch the GPU.
#
# Usage: tools/deploy_g1.sh [ssh_alias]
# Settings (environment, defaults match g1-gpu on 25.11.2026):
#   REMOTE_ROOT     ~/vl-arkestrator
#   BOTTLE_CONFIG   ~/pre_kpi/_bottle_localdev/config   (reviewed thresholds, temperatures, vla, …)
#   BOTTLE_LOCKS    ~/pre_kpi/_bottle/.xr               (shared xr_launcher/robot_motor locks)
set -euo pipefail

HOST="${1:-g1-gpu}"
REMOTE_ROOT="${REMOTE_ROOT:-vl-arkestrator}"
BOTTLE_CONFIG="${BOTTLE_CONFIG:-pre_kpi/_bottle_localdev/config}"
BOTTLE_LOCKS="${BOTTLE_LOCKS:-pre_kpi/_bottle/.xr}"
# Gitignored bottle configs the controller reads from <checkout>/config.
CONFIGS=(local.json thresholds.yaml temperatures.yaml preparation_pose.yaml vla.yaml bottle_grasp_profile.json)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOTTLE="$ROOT/third_party/GR00T-pickup-bottle"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST")

if [[ -n "$(git -C "$ROOT" status --porcelain -- arkestrator tests profiles config pyproject.toml)" ]]; then
  echo "Uncommitted changes in vl-arkestrator; commit first (only committed code is deployed)." >&2
  exit 1
fi
PIN="$(git -C "$ROOT" rev-parse HEAD:third_party/GR00T-pickup-bottle)"
[[ "$(git -C "$BOTTLE" rev-parse HEAD)" == "$PIN" ]] || { echo "Submodule is not at the pinned $PIN" >&2; exit 1; }

echo "vl-arkestrator $(git -C "$ROOT" rev-parse --short HEAD), bottle pin ${PIN:0:7} -> $HOST:~/$REMOTE_ROOT"
git -C "$ROOT" archive --format=tar HEAD -- arkestrator tests profiles config tools pyproject.toml README.md \
  | "${SSH[@]}" "mkdir -p ~/$REMOTE_ROOT && tar -x -C ~/$REMOTE_ROOT"

BUNDLE="$(mktemp -d)/bottle.bundle"
git -C "$BOTTLE" bundle create -q "$BUNDLE" HEAD
scp -q -o BatchMode=yes "$BUNDLE" "$HOST:.cache/arkestrator-bottle.bundle"
rm -f "$BUNDLE"

"${SSH[@]}" bash -s -- "$REMOTE_ROOT" "$PIN" "$BOTTLE_CONFIG" "$BOTTLE_LOCKS" "${CONFIGS[@]}" <<'REMOTE'
set -euo pipefail
root="$HOME/$1"; pin="$2"; config="$HOME/$3"; locks="$HOME/$4"; shift 4
checkout="$root/third_party/GR00T-pickup-bottle"
bundle="$HOME/.cache/arkestrator-bottle.bundle"
if [[ ! -d "$checkout/.git" ]]; then
  git clone -q "$bundle" "$checkout"
else
  git -C "$checkout" fetch -q "$bundle" HEAD
fi
if [[ -n "$(git -C "$checkout" status --porcelain --untracked-files=no)" ]]; then
  echo "Server checkout has local edits; refusing to overwrite" >&2; exit 1
fi
git -C "$checkout" checkout -q --detach "$pin"
rm -f "$bundle"
[[ -d "$locks" ]] || { echo "Lock directory $locks missing" >&2; exit 1; }
if [[ -e "$checkout/.xr" && ! -L "$checkout/.xr" ]]; then
  echo "$checkout/.xr is a real directory; refusing to replace it" >&2; exit 1
fi
ln -sfn "$locks" "$checkout/.xr"
for name in "$@"; do
  if [[ -e "$config/$name" ]]; then
    if [[ -e "$checkout/config/$name" && ! -L "$checkout/config/$name" ]]; then
      echo "$checkout/config/$name is a real file; refusing to replace it" >&2; exit 1
    fi
    ln -sfn "$config/$name" "$checkout/config/$name"
  else
    echo "note: $config/$name absent, not linked"
  fi
done
echo "checkout $(git -C "$checkout" rev-parse --short HEAD); .xr -> $(readlink "$checkout/.xr")"
REMOTE
