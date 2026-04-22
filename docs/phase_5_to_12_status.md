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

## Phase 7: supported-dataset checkpoint readiness

Target:

- make the supported datasets runnable with a pinned checkpoint
- treat missing `GTZAN` as an external blocker, not a repo bug

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
  --datasets esc50 urbansound8k fsdd \
  --datasets-root ~/projects/clap-reimpl/data/datasets \
  --checkpoints-root ~/datasets/clap/checkpoints \
  --require-checkpoints 630k-audioset-best.pt \
  --hash-checkpoints
```

## Phase 9: baseline run orchestration

Target:

- launch the paper-style path with explicit checkpoint metadata when available
- allow `SKIP_MISSING=1` for partial runs while `GTZAN` is unavailable

Command:

```bash
SKIP_MISSING=1 \
DATASETS="esc50 urbansound8k" \
VERIFY_DATASETS="esc50 urbansound8k" \
CHECKPOINT_NAME=630k-audioset-best.pt \
bash scripts/repro/run_reproduction.sh
```

## Phase 10: extension run orchestration

Target:

- run the richer metrics path and ensemble prompting path with the pinned checkpoint

Command:

```bash
SKIP_MISSING=1 \
DATASETS="esc50 urbansound8k fsdd" \
VERIFY_DATASETS="esc50 urbansound8k fsdd" \
CHECKPOINT_NAME=630k-audioset-best.pt \
bash scripts/repro/run_extensions.sh
```

## Phase 11: acceptance checks

Target:

- confirm that metrics JSON, summary tables, manifests, and checkpoint files exist

Command:

```bash
python scripts/repro/check_acceptance.py \
  --datasets esc50 urbansound8k fsdd \
  --expect-ensemble \
  --manifests extensions.json extensions-ensemble.json \
  --checkpoints-root ~/datasets/clap/checkpoints \
  --require-checkpoints 630k-audioset-best.pt
```

## Phase 12: final writing readiness

Target:

- capture exact blockers separately from completed work

Current expected blocker:

- `GTZAN` may remain unavailable because the historical public download link is
  no longer dependable; this should be documented explicitly in the paper or
  project report if you proceed without it.
