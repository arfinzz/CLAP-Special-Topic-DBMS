#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-auto}"
DATASETS="${DATASETS:-esc50 urbansound8k gtzan fsdd}"
VERIFY_DATASETS="${VERIFY_DATASETS:-$DATASETS}"
AUDIO_BATCH_SIZE="${AUDIO_BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-extensions}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$HOME/datasets/clap/checkpoints}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-630k-audioset-best.pt}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-laion-630k-improved}"
NOTES="${NOTES:-extended evaluation with FSDD, richer metrics, and ensemble prompting}"
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

BASE_CMD=(
  python metrics_analysis.py
  --datasets ${DATASETS}
  --fsdd-split all
  --device "$DEVICE"
  --audio-batch-size "$AUDIO_BATCH_SIZE"
  --run-tag "$RUN_TAG"
  --checkpoint-label "$CHECKPOINT_LABEL"
  --notes "$NOTES"
)

ENS_CMD=(
  python scripts/analysis/run_ensemble_prompting.py
  --datasets ${DATASETS}
  --fsdd-split all
  --device "$DEVICE"
  --audio-batch-size "$AUDIO_BATCH_SIZE"
  --run-tag "${RUN_TAG}-ensemble"
  --checkpoint-label "$CHECKPOINT_LABEL"
  --notes "$NOTES"
)

if [[ "$SKIP_MISSING" == "1" ]]; then
  BASE_CMD+=(--skip-missing)
  ENS_CMD+=(--skip-missing)
fi

if [[ -n "$CHECKPOINT_PATH" ]]; then
  BASE_CMD+=(--checkpoint "$CHECKPOINT_PATH")
  ENS_CMD+=(--checkpoint "$CHECKPOINT_PATH")
fi

echo "Running: ${BASE_CMD[*]}"
"${BASE_CMD[@]}"

echo "Running: ${ENS_CMD[*]}"
"${ENS_CMD[@]}"
