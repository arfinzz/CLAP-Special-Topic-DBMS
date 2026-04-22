# CLAP Reimplementation and Extended Evaluation

This repository is the experiment and analysis layer for a CLAP reproduction
project. It uses the official LAION CLAP implementation for the model itself,
and this repo provides:

- dataset loading for `ESC-50`, `UrbanSound8K`, `GTZAN`, and `FSDD`
- zero-shot evaluation scripts
- extra metrics beyond top-1 accuracy
- prompt sensitivity analysis
- prompt ensembling experiments
- run manifests, checkpoint manifests, and acceptance checks

The final verified workflow uses:

- `~/projects/clap-official` for the official LAION CLAP code
- `~/projects/clap-reimpl` for this repo
- `~/datasets/clap` for datasets and checkpoints
- `Ubuntu-22.04` in WSL2
- `Python 3.10`
- `torch 2.4.1+cu121`
- checkpoint `630k-audioset-best.pt`

## Repos Used

You should keep two repos in WSL:

- official model repo: `https://github.com/LAION-AI/CLAP.git`
- experiment repo: `https://github.com/Shuvam-Chakraborty/CLAP.git`

The official repo provides `laion_clap`.
This repo provides the benchmark pipeline and reportable experiment outputs.

## Verified Layout

```text
~/projects/
  clap-official/
  clap-reimpl/

~/datasets/clap/
  archives/
  checkpoints/
  esc50/
  gtzan/
  gtzan_raw_box/
  fsdd/
  urbansound8k/
```

## Exact End-to-End Run Guide

This is the exact setup path for the final verified experiment run.

### 1. Create or open the WSL distro

From Windows PowerShell:

```powershell
wsl -l -v
wsl -d Ubuntu-22.04-Clean
```

If you need a new distro:

```powershell
wsl --install -d Ubuntu-22.04
```

### 2. Install base packages in WSL

Run inside WSL:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git git-lfs ffmpeg libsndfile1 unzip tar wget curl build-essential ca-certificates
```

### 3. Create the workspace folders

```bash
mkdir -p "$HOME/projects"
mkdir -p "$HOME/datasets/clap"/{archives,checkpoints,esc50,gtzan,urbansound8k,fsdd}
```

### 4. Clone both repositories inside WSL

```bash
git clone https://github.com/LAION-AI/CLAP.git "$HOME/projects/clap-official"
git clone https://github.com/Shuvam-Chakraborty/CLAP.git "$HOME/projects/clap-reimpl"
cd "$HOME/projects/clap-reimpl"
```

### 5. Install Miniforge

```bash
cd /tmp
curl -L "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -o Miniforge3.sh
bash Miniforge3.sh -b -p "$HOME/miniforge3"

export PATH="$HOME/miniforge3/bin:$PATH"
eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
conda config --set solver classic
conda config --set auto_activate_base false
export CONDA_NO_PLUGINS=true
```

### 6. Create the verified experiment environment

The verified run used `clap-reimpl-env`.

```bash
cd "$HOME/projects/clap-reimpl"
bash scripts/setup/setup_reimpl_env.sh clap-reimpl-env "$HOME/projects/clap-official"
```

Optional isolated official env:

```bash
bash scripts/setup/setup_official_env.sh clap-official-env "$HOME/projects/clap-official"
```

Quick verification:

```bash
conda run -n clap-reimpl-env python -c "import torch, numpy, transformers, laion_clap; print(torch.__version__); print(torch.cuda.is_available()); print(numpy.__version__); print(transformers.__version__); print(laion_clap.__file__)"
```

### 7. Download ESC-50

```bash
ESC_ARCHIVE="$HOME/datasets/clap/archives/ESC-50-master.zip"
ESC_DEST="$HOME/datasets/clap/esc50"

mkdir -p "$ESC_DEST"
wget -O "$ESC_ARCHIVE" https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
unzip -q "$ESC_ARCHIVE" -d "$ESC_DEST"
find "$ESC_DEST/ESC-50-master/audio" -name '*.wav' | wc -l
```

Expected count:

```bash
2000
```

### 8. Download UrbanSound8K

```bash
US8K_ARCHIVE="$HOME/datasets/clap/archives/UrbanSound8K.tar.gz"
US8K_DEST="$HOME/datasets/clap/urbansound8k"

mkdir -p "$US8K_DEST"
wget -c -O "$US8K_ARCHIVE" "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
tar -xzf "$US8K_ARCHIVE" -C "$US8K_DEST"
test -f "$US8K_DEST/UrbanSound8K/metadata/UrbanSound8K.csv" && echo csv_ok
find "$US8K_DEST/UrbanSound8K/audio" -name '*.wav' | wc -l
```

Expected count:

```bash
8732
```

### 9. Download and prepare GTZAN

This verified run used the raw GTZAN archive referenced by the public dataset
script, then converted `.au` files to `.wav` and removed the known corrupted
`jazz.00054.wav`.

```bash
GTZAN_URL='https://ibm.ent.box.com/index.php?rm=box_download_shared_file&shared_name=gvkgmb4n0h6rbeccdwtwa0ujjjfmouof&file_id=f_961929491012'
GTZAN_ARCHIVE="$HOME/datasets/clap/archives/gtzan_raw_box.zip"
GTZAN_RAW="$HOME/datasets/clap/gtzan_raw_box"
GTZAN_DEST="$HOME/datasets/clap/gtzan"

mkdir -p "$HOME/datasets/clap/archives"
wget -c -O "$GTZAN_ARCHIVE" "$GTZAN_URL"

rm -rf "$GTZAN_RAW" "$GTZAN_DEST"
mkdir -p "$GTZAN_RAW" "$GTZAN_DEST"

tar -xzf "$GTZAN_ARCHIVE" -C "$GTZAN_RAW"
cp -R "$GTZAN_RAW/genres" "$GTZAN_DEST/genres"

find "$GTZAN_DEST/genres" -type f -name '*.au' | while IFS= read -r src; do
  out="${src%.au}.wav"
  ffmpeg -nostdin -loglevel error -y -i "$src" "$out"
  rm -f "$src"
done

rm -f "$GTZAN_DEST/genres/jazz/jazz.00054.wav"
find "$GTZAN_DEST/genres" -name '*.wav' | wc -l
```

Expected count:

```bash
999
```

### 10. Download FSDD

```bash
git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git \
  "$HOME/datasets/clap/fsdd/free-spoken-digit-dataset"
find "$HOME/datasets/clap/fsdd/free-spoken-digit-dataset/recordings" -name '*.wav' | wc -l
```

Expected count:

```bash
3000
```

### 11. Link the shared datasets into the repo

```bash
cd "$HOME/projects/clap-reimpl"
bash scripts/setup/link_shared_assets.sh "$HOME/datasets/clap"
ls -l "$HOME/projects/clap-reimpl/data/datasets"
```

### 12. Download the pinned checkpoint

```bash
cd "$HOME/projects/clap-reimpl"
bash scripts/setup/fetch_checkpoint.sh "$HOME/datasets/clap/checkpoints"
```

The verified checkpoint was:

- file: `630k-audioset-best.pt`
- SHA256: `8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037`

### 13. Verify all datasets and the checkpoint

```bash
cd "$HOME/projects/clap-reimpl"
conda run -n clap-reimpl-env python scripts/repro/verify_assets.py \
  --datasets esc50 urbansound8k gtzan fsdd \
  --datasets-root "$HOME/projects/clap-reimpl/data/datasets" \
  --checkpoints-root "$HOME/datasets/clap/checkpoints" \
  --require-checkpoints 630k-audioset-best.pt \
  --hash-checkpoints \
  --output "$HOME/projects/clap-reimpl/outputs/manifests/asset_check_full_with_checkpoint.json"
```

### 14. Run the full baseline benchmark

This is the paper-style dataset suite:

```bash
cd "$HOME/projects/clap-reimpl"
conda run -n clap-reimpl-env bash -lc '
DATASETS="esc50 urbansound8k gtzan" \
VERIFY_DATASETS="esc50 urbansound8k gtzan" \
CHECKPOINT_NAME=630k-audioset-best.pt \
CHECKPOINT_LABEL=laion-630k-full-baseline \
RUN_TAG=full-baseline \
VERIFY_HASHES=1 \
bash scripts/repro/run_reproduction.sh
'
```

### 15. Run the full extension benchmark

This is the improved setup with the extra dataset and prompt-ensemble analysis:

```bash
cd "$HOME/projects/clap-reimpl"
conda run -n clap-reimpl-env bash -lc '
DATASETS="esc50 urbansound8k gtzan fsdd" \
VERIFY_DATASETS="esc50 urbansound8k gtzan fsdd" \
CHECKPOINT_NAME=630k-audioset-best.pt \
CHECKPOINT_LABEL=laion-630k-full-extensions \
RUN_TAG=full-extensions \
VERIFY_HASHES=1 \
bash scripts/repro/run_extensions.sh
'
```

### 16. Run the final acceptance check

```bash
cd "$HOME/projects/clap-reimpl"
conda run -n clap-reimpl-env python scripts/repro/check_acceptance.py \
  --datasets esc50 urbansound8k gtzan fsdd \
  --expect-ensemble \
  --manifests full_baseline.json full_extensions.json full_extensions_ensemble.json \
  --checkpoints-root "$HOME/datasets/clap/checkpoints" \
  --require-checkpoints 630k-audioset-best.pt \
  --output "$HOME/projects/clap-reimpl/outputs/manifests/acceptance_check_full.json"
```

If everything is correct, the final acceptance report will contain:

```json
"all_ok": true
```

## Expected Final Outputs

After the full run, you should have:

- `outputs/metrics/*.json`
- `outputs/figures/<dataset>/*.png`
- `outputs/tables/zeroshot_summary.csv`
- `outputs/tables/ensemble_summary.csv`
- `outputs/manifests/full_baseline.json`
- `outputs/manifests/full_extensions.json`
- `outputs/manifests/full_extensions_ensemble.json`
- `outputs/manifests/acceptance_check_full.json`

## Important Notes

- This repo is the experiment layer. The official model code still comes from
  `clap-official`.
- The verified full run uses the stronger
  `630k-audioset-best.pt` checkpoint, so the final benchmark should be described
  as an improved-checkpoint reproduction rather than an exact same-checkpoint
  replication of the original paper.
- GTZAN preparation is documented in
  [reports/assets/manifests/gtzan_preparation.json](/C:/Users/arfin/Desktop/sdbms-proj/CLAP/reports/assets/manifests/gtzan_preparation.json).

## Additional Docs

- WSL setup guide: [docs/wsl_reproduction_setup.md](/C:/Users/arfin/Desktop/sdbms-proj/CLAP/docs/wsl_reproduction_setup.md)
- phase 0-4 status: [docs/phase_0_to_4_status.md](/C:/Users/arfin/Desktop/sdbms-proj/CLAP/docs/phase_0_to_4_status.md)
- phase 5-12 status: [docs/phase_5_to_12_status.md](/C:/Users/arfin/Desktop/sdbms-proj/CLAP/docs/phase_5_to_12_status.md)
- workflow notes: [docs/reproduction_workflow.md](/C:/Users/arfin/Desktop/sdbms-proj/CLAP/docs/reproduction_workflow.md)
- final report: [reports/CLAP_Final_Report.md](/C:/Users/arfin/Desktop/sdbms-proj/CLAP/reports/CLAP_Final_Report.md)
