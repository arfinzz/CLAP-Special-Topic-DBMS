#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-auto}"
DATASETS="${DATASETS:-esc50 urbansound8k gtzan}"
VERIFY_DATASETS="${VERIFY_DATASETS:-$DATASETS}"
AUDIO_BATCH_SIZE="${AUDIO_BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-baseline}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$HOME/datasets/clap/checkpoints}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-paperlike-baseline}"
NOTES="${NOTES:-official-style zero-shot reproduction run}"
SKIP_MISSING="${SKIP_MISSING:-0}"
VERIFY_HASHES="${VERIFY_HASHES:-0}"

if [[ -z "$CHECKPOINT_PATH" && -n "$CHECKPOINT_NAME" ]]; then
  CHECKPOINT_PATH="$CHECKPOINTS_ROOT/$CHECKPOINT_NAME"
fi

VERIFY_CMD=(
  python scripts/repro/verify_assets.py
  --datasets ${VERIFY_DATASETS}
)

if [[ -n "$CHECKPOINT_NAME" ]]; then
  VERIFY_CMD+=(--checkpoints-root "$CHECKPOINTS_ROOT" --require-checkpoints "$CHECKPOINT_NAME")
  if [[ "$VERIFY_HASHES" == "1" ]]; then
    VERIFY_CMD+=(--hash-checkpoints)
  fi
fi

echo "Running: ${VERIFY_CMD[*]}"
"${VERIFY_CMD[@]}"

CMD=(
  python metrics_analysis.py
  --datasets ${DATASETS}
  --device "$DEVICE"
  --audio-batch-size "$AUDIO_BATCH_SIZE"
  --run-tag "$RUN_TAG"
  --checkpoint-label "$CHECKPOINT_LABEL"
  --notes "$NOTES"
)

if [[ "$SKIP_MISSING" == "1" ]]; then
  CMD+=(--skip-missing)
fi

if [[ -n "$CHECKPOINT_PATH" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT_PATH")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
