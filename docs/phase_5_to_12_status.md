# Phase 5-12 Status

This document tracks the remaining reproduction phases after the WSL and
environment bootstrap.

## Phase 5: shared dataset linking

Target:

- point `data/datasets/*` at the shared dataset root under `~/datasets/clap`
- make the relinking step repeatable

Command:

```bash
bash scripts/setup/link_shared_assets.sh ~/datasets/clap
```

Success criteria:

- every dataset entry under `data/datasets/` is a valid symlink
- rerunning the script does not introduce carriage-return corruption

## Phase 6: dataset acquisition and validation

Target:

- acquire `ESC-50`, `UrbanSound8K`, and `FSDD`
- acquire `GTZAN` if you have a defensible source
- verify counts with the asset checker

Command:

```bash
python scripts/repro/verify_assets.py \
  --datasets esc50 urbansound8k gtzan fsdd \
  --datasets-root ~/projects/clap-reimpl/data/datasets
```

Success criteria:

- `ESC-50` reports `2000/2000`
- `UrbanSound8K` reports `8732/8732`
- `FSDD` reports `3000/3000`
- `GTZAN` reports `999/999` after removing `jazz.00054.wav`

## Phase 7: checkpoint readiness

Target:

- make the full dataset suite runnable with a pinned checkpoint

Command:

```bash
bash scripts/setup/fetch_checkpoint.sh ~/datasets/clap/checkpoints
```

Success criteria:

- `~/datasets/clap/checkpoints/630k-audioset-best.pt` exists
- `outputs/manifests/checkpoint_630k-audioset-best.json` exists

## Phase 8: checkpoint verification

Target:

- verify the required checkpoint before launching experiments

Command:

```bash
python scripts/repro/verify_assets.py \
  --datasets esc50 urbansound8k gtzan fsdd \
  --datasets-root ~/projects/clap-reimpl/data/datasets \
  --checkpoints-root ~/datasets/clap/checkpoints \
  --require-checkpoints 630k-audioset-best.pt \
  --hash-checkpoints
```

## Phase 9: baseline run orchestration

Target:

- launch the original-dataset benchmark path with explicit checkpoint metadata

Command:

```bash
DATASETS="esc50 urbansound8k gtzan" \
VERIFY_DATASETS="esc50 urbansound8k gtzan" \
CHECKPOINT_NAME=630k-audioset-best.pt \
CHECKPOINT_LABEL=laion-630k-full-baseline \
RUN_TAG=full-baseline \
VERIFY_HASHES=1 \
bash scripts/repro/run_reproduction.sh
```

## Phase 10: extension run orchestration

Target:

- run the richer metrics path and ensemble prompting path with the pinned checkpoint

Command:

```bash
DATASETS="esc50 urbansound8k gtzan fsdd" \
VERIFY_DATASETS="esc50 urbansound8k gtzan fsdd" \
CHECKPOINT_NAME=630k-audioset-best.pt \
CHECKPOINT_LABEL=laion-630k-full-extensions \
RUN_TAG=full-extensions \
VERIFY_HASHES=1 \
bash scripts/repro/run_extensions.sh
```

## Phase 11: acceptance checks

Target:

- confirm that metrics JSON, summary tables, manifests, and checkpoint files exist

Command:

```bash
python scripts/repro/check_acceptance.py \
  --datasets esc50 urbansound8k gtzan fsdd \
  --expect-ensemble \
  --manifests full_baseline.json full_extensions.json full_extensions_ensemble.json \
  --checkpoints-root ~/datasets/clap/checkpoints \
  --require-checkpoints 630k-audioset-best.pt
```

## Phase 12: final writing readiness

Target:

- capture exact blockers separately from completed work

Current status:

- `GTZAN` has been acquired and normalized to `999` `.wav` files
- the full four-dataset run is available through the exact commands above
