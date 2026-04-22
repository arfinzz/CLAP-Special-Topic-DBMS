# WSL Reproduction Setup

This repository is the analysis layer for the project. The authoritative CLAP
implementation should be installed from the official LAION repository in a
separate checkout, while this repo supplies dataset loaders, evaluation
scripts, richer metrics, and extension experiments.

## Recommended layout

```text
~/projects/
  clap-official/     # official LAION CLAP checkout
  clap-reimpl/       # this repository

~/datasets/clap/
  archives/
  checkpoints/
  esc50/
  gtzan/
  urbansound8k/
  fsdd/
```

Clone both repositories inside the WSL filesystem, not under `/mnt/c`, to
avoid Windows metadata and extraction issues.

Use the verified branches and commit:

```bash
git clone https://github.com/LAION-AI/CLAP.git ~/projects/clap-official
git -C ~/projects/clap-official checkout 1fd4c37

git clone --branch main --single-branch \
  https://github.com/arfinzz/CLAP-Special-Topic-DBMS.git \
  ~/projects/clap-reimpl
```

## 1. Prepare the machine

```bash
cd ~/projects/clap-reimpl
bash scripts/setup/prepare_wsl_workspace.sh
```

Then confirm GPU visibility if you want CUDA:

```bash
nvidia-smi
python3 --version
```

## 2. Create environments

Official baseline environment:

```bash
bash scripts/setup/setup_official_env.sh clap-official-env ~/projects/clap-official
```

Reimplementation environment:

```bash
bash scripts/setup/setup_reimpl_env.sh clap-reimpl-env ~/projects/clap-official
```

The YAML files in [envs](/C:/Users/arfin/Desktop/sdbms-proj/CLAP/envs) are
lightweight manifests; the shell scripts above remain the source of truth.

## Complete phases 0-4 in one command

If you want the setup automated through the environment stage, run:

```bash
cd ~/projects/clap-reimpl
bash scripts/setup/bootstrap_phase_0_to_4.sh
```

Phase definitions and success criteria are tracked in
[docs/phase_0_to_4_status.md](/C:/Users/arfin/Desktop/sdbms-proj/CLAP/docs/phase_0_to_4_status.md).

## 3. Download datasets

Place datasets under `~/datasets/clap` using the following structure:

- `~/datasets/clap/esc50/ESC-50-master/audio/*.wav`
- `~/datasets/clap/gtzan/genres/<genre>/*.wav`
- `~/datasets/clap/urbansound8k/UrbanSound8K/audio/fold*/...`
- `~/datasets/clap/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv`
- `~/datasets/clap/fsdd/free-spoken-digit-dataset/recordings/*.wav`

FSDD can be cloned directly:

```bash
git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git \
  ~/datasets/clap/fsdd/free-spoken-digit-dataset
```

For GTZAN, this project expects 999 files after removing the corrupted
`jazz.00054.wav`.

## 4. Link shared data into the repo

```bash
cd ~/projects/clap-reimpl
bash scripts/setup/link_shared_assets.sh ~/datasets/clap
```

## 5. Verify assets before every run

```bash
python scripts/repro/verify_assets.py \
  --datasets esc50 urbansound8k gtzan fsdd \
  --checkpoints-root ~/datasets/clap/checkpoints \
  --require-checkpoints 630k-audioset-best.pt
```

The report is written to `outputs/manifests/asset_check.json` by default.

## 6. Fetch the pinned extension checkpoint

```bash
bash scripts/setup/fetch_checkpoint.sh ~/datasets/clap/checkpoints
```

This writes the checkpoint file and a checkpoint manifest under
`outputs/manifests/`.

## 7. Run the two experiment tracks

Original-dataset baseline with the improved `630k` checkpoint:

```bash
bash scripts/repro/run_reproduction.sh
```

Extended evaluation:

```bash
bash scripts/repro/run_extensions.sh
```

Each run writes summary tables, metrics JSON, figures, and a run manifest under
`outputs/manifests/`.

If `GTZAN` is still unavailable, you can run the supported subset explicitly:

```bash
SKIP_MISSING=1 \
DATASETS="esc50 urbansound8k fsdd" \
VERIFY_DATASETS="esc50 urbansound8k fsdd" \
CHECKPOINT_NAME=630k-audioset-best.pt \
bash scripts/repro/run_extensions.sh
```

## 8. Acceptance check

```bash
python scripts/repro/check_acceptance.py \
  --datasets esc50 urbansound8k gtzan fsdd \
  --expect-ensemble \
  --manifests full_baseline.json full_extensions.json full_extensions_ensemble.json \
  --checkpoints-root ~/datasets/clap/checkpoints \
  --require-checkpoints 630k-audioset-best.pt
```

## 9. What to report in the paper

Separate your results into two tracks:

- `official baseline`: closest setup to the published evaluation path
- `extended analysis`: newer checkpoint, FSDD, extra metrics, and prompt ensembling

Always document:

- official repo commit
- this repo commit
- checkpoint file name
- checkpoint source
- environment name
- Torch version
- dataset file counts
