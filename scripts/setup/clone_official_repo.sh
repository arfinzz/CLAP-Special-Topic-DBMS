#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:-$HOME/projects}"
OFFICIAL_ROOT="${2:-$WORKSPACE_ROOT/clap-official}"
OFFICIAL_REPO_URL="${OFFICIAL_REPO_URL:-https://github.com/LAION-AI/CLAP.git}"

mkdir -p "$WORKSPACE_ROOT"

if [[ -d "$OFFICIAL_ROOT/.git" ]]; then
  echo "Official repo already exists at $OFFICIAL_ROOT"
  git -C "$OFFICIAL_ROOT" remote -v
  exit 0
fi

git clone "$OFFICIAL_REPO_URL" "$OFFICIAL_ROOT"
echo "Cloned official repo to $OFFICIAL_ROOT"
