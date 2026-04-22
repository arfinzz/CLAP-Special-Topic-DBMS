#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

normalize_path() {
  local raw_path="${1:-}"
  printf '%s' "${raw_path//$'\r'/}"
}

DATA_ROOT="$(normalize_path "${1:-$HOME/datasets/clap}")"
DATASETS_DIR="$(normalize_path "$ROOT_DIR/data/datasets")"

mkdir -p "$DATASETS_DIR"

link_dir() {
  local source_dir
  local target_dir

  source_dir="$(normalize_path "$1")"
  target_dir="$(normalize_path "$2")"

  rm -rf "$target_dir"
  ln -s "$source_dir" "$target_dir"
  echo "Linked $target_dir -> $source_dir"
}

link_dir "$DATA_ROOT/esc50" "$DATASETS_DIR/esc50"
link_dir "$DATA_ROOT/gtzan" "$DATASETS_DIR/gtzan"
link_dir "$DATA_ROOT/urbansound8k" "$DATASETS_DIR/urbansound8k"
link_dir "$DATA_ROOT/fsdd" "$DATASETS_DIR/fsdd"

echo "Dataset links created under $DATASETS_DIR"
