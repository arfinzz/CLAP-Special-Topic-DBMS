#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINTS_ROOT="${1:-$HOME/datasets/clap/checkpoints}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-630k-audioset-best.pt}"
CHECKPOINT_URL="${CHECKPOINT_URL:-https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt}"
OUT_PATH="$CHECKPOINTS_ROOT/$CHECKPOINT_NAME"
TMP_PATH="$OUT_PATH.part"

mkdir -p "$CHECKPOINTS_ROOT"

download_with_wget() {
  wget -c -O "$TMP_PATH" "$CHECKPOINT_URL"
}

download_with_curl() {
  curl -L --fail --continue-at - -o "$TMP_PATH" "$CHECKPOINT_URL"
}

if command -v wget >/dev/null 2>&1; then
  download_with_wget
elif command -v curl >/dev/null 2>&1; then
  download_with_curl
else
  echo "Neither wget nor curl is available." >&2
  exit 1
fi

mv "$TMP_PATH" "$OUT_PATH"

if command -v sha256sum >/dev/null 2>&1; then
  SHA256="$(sha256sum "$OUT_PATH" | awk '{print $1}')"
else
  SHA256="$(python3 - "$OUT_PATH" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
fi

MANIFEST_PATH="$ROOT_DIR/outputs/manifests/checkpoint_${CHECKPOINT_NAME%.pt}.json"
python3 - "$OUT_PATH" "$CHECKPOINT_URL" "$SHA256" "$MANIFEST_PATH" <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

checkpoint_path = Path(sys.argv[1])
checkpoint_url = sys.argv[2]
sha256 = sys.argv[3]
manifest_path = Path(sys.argv[4])
manifest_path.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "checkpoint_name": checkpoint_path.name,
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_url": checkpoint_url,
    "sha256": sha256,
    "size_bytes": checkpoint_path.stat().st_size,
    "exists": checkpoint_path.exists(),
    "repo_root": str(manifest_path.parents[2]),
    "hostname": os.uname().nodename if hasattr(os, "uname") else None,
}

with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")

print(f"Saved checkpoint manifest: {manifest_path}")
PY

printf 'Checkpoint ready: %s\n' "$OUT_PATH"
printf 'SHA256          : %s\n' "$SHA256"
