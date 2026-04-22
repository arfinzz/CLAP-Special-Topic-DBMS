#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${1:-$HOME/projects}"
DATA_ROOT="${2:-$HOME/datasets/clap}"

echo "Preparing WSL workspace"
echo "  repo root      : $ROOT_DIR"
echo "  workspace root : $WORKSPACE_ROOT"
echo "  data root      : $DATA_ROOT"

mkdir -p "$WORKSPACE_ROOT" "$DATA_ROOT"/{archives,checkpoints,esc50,gtzan,urbansound8k,fsdd}

if [[ "${SKIP_APT:-0}" == "1" ]]; then
  echo "Skipping apt installation because SKIP_APT=1"
elif sudo -n true >/dev/null 2>&1; then
  sudo -n apt update
  sudo -n apt install -y git git-lfs ffmpeg libsndfile1 unzip tar wget curl build-essential
else
  echo "Passwordless sudo is not available in this WSL distro."
  echo "Run the apt commands manually with your password, or rerun with SKIP_APT=1"
  echo "after confirming git, curl, wget, tar, unzip, ffmpeg, gcc, and g++ are already installed."
  exit 1
fi

echo
echo "Workspace directories are ready."
echo "Recommended next steps:"
echo "  1. Create the conda env with scripts/setup/setup_reimpl_env.sh"
echo "  2. Download datasets into $DATA_ROOT"
echo "  3. Link datasets with scripts/setup/link_shared_assets.sh"
