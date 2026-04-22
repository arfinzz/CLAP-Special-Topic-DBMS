#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${1:-$HOME/projects}"
DATA_ROOT="${2:-$HOME/datasets/clap}"
OFFICIAL_ROOT="${3:-$WORKSPACE_ROOT/clap-official}"
OFFICIAL_ENV_NAME="${OFFICIAL_ENV_NAME:-clap-official-env}"
REIMPL_ENV_NAME="${REIMPL_ENV_NAME:-clap-reimpl-env}"

echo "Running CLAP setup through Phase 4"
echo "  repo root         : $ROOT_DIR"
echo "  workspace root    : $WORKSPACE_ROOT"
echo "  data root         : $DATA_ROOT"
echo "  official checkout : $OFFICIAL_ROOT"
echo "  official env      : $OFFICIAL_ENV_NAME"
echo "  reimpl env        : $REIMPL_ENV_NAME"
echo "  skip apt          : ${SKIP_APT:-0}"

bash "$ROOT_DIR/scripts/setup/prepare_wsl_workspace.sh" "$WORKSPACE_ROOT" "$DATA_ROOT"
bash "$ROOT_DIR/scripts/setup/clone_official_repo.sh" "$WORKSPACE_ROOT" "$OFFICIAL_ROOT"
bash "$ROOT_DIR/scripts/setup/setup_official_env.sh" "$OFFICIAL_ENV_NAME" "$OFFICIAL_ROOT"
bash "$ROOT_DIR/scripts/setup/setup_reimpl_env.sh" "$REIMPL_ENV_NAME" "$OFFICIAL_ROOT"

echo
echo "Phase 0-4 bootstrap completed."
echo "Next phases:"
echo "  5. Download datasets into $DATA_ROOT"
echo "  6. Link them with scripts/setup/link_shared_assets.sh"
echo "  7. Verify assets with scripts/repro/verify_assets.py"
