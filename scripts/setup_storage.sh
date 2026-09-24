#!/usr/bin/env bash
# Point ./data (the default $STABLEWM_HOME) at a large disk.
#   scripts/setup_storage.sh /data/$USER/multi-agent-wm   # symlink ./data -> that path
#   scripts/setup_storage.sh                              # no big disk: plain local ./data
# Datasets, checkpoints, hydra outputs and eval results all land under ./data.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
target="${1:-${MAWM_STORAGE:-}}"

if [ -z "$target" ]; then
  mkdir -p "$root/data"
  echo "using local $root/data"
  exit 0
fi

if [ -e "$root/data" ] && [ ! -L "$root/data" ]; then
  echo "error: $root/data exists and is not a symlink; move it first" >&2
  exit 1
fi
mkdir -p "$target"
ln -sfn "$(cd "$target" && pwd)" "$root/data"
echo "$root/data -> $(readlink "$root/data")"
