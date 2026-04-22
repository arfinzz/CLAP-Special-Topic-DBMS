#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${1:-$HOME/projects}"
OFFICIAL_ROOT="${2:-$WORKSPACE_ROOT/clap-official}"
OFFICIAL_REPO_URL="${OFFICIAL_REPO_URL:-https://github.com/LAION-AI/CLAP.git}"
OFFICIAL_REPO_REF="${OFFICIAL_REPO_REF:-1fd4c37}"

mkdir -p "$WORKSPACE_ROOT"

if [[ -d "$OFFICIAL_ROOT/.git" ]]; then
  echo "Official repo already exists at $OFFICIAL_ROOT"
  git -C "$OFFICIAL_ROOT" remote -v
  echo "Current official repo commit: $(git -C "$OFFICIAL_ROOT" rev-parse --short HEAD)"
  exit 0
fi

git clone "$OFFICIAL_REPO_URL" "$OFFICIAL_ROOT"
git -C "$OFFICIAL_ROOT" checkout "$OFFICIAL_REPO_REF"
echo "Cloned official repo to $OFFICIAL_ROOT"
echo "Pinned official repo to $OFFICIAL_REPO_REF"
