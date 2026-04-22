#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-clap-reimpl-env}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OFFICIAL_ROOT="${2:-$HOME/projects/clap-official}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required for this setup script." >&2
  exit 1
fi

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"

conda create -n "$ENV_NAME" python=3.10 -y --solver classic
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install -r "$REPO_ROOT/requirements.txt"
pip install --force-reinstall torch==2.4.1+cu121 torchaudio==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install --force-reinstall numpy==1.26.4 transformers==4.30.0 tokenizers==0.13.3 huggingface_hub==0.36.2
pip install --no-deps -e "$OFFICIAL_ROOT"

python -c "import laion_clap, torch; print('laion_clap:', laion_clap.__file__); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
