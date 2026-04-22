# Reproduction Workflow

This project should be run as two clearly separated tracks.

## Track A: Baseline reproduction

Goal:

- reproduce the published zero-shot benchmarks as faithfully as possible
- keep datasets to ESC-50, UrbanSound8K, and GTZAN
- treat accuracy as the primary comparison metric

Recommended command:

```bash
bash scripts/repro/run_reproduction.sh
```

Suggested metadata:

- `RUN_TAG=paperlike-baseline`
- `CHECKPOINT_LABEL=paperlike-baseline`

## Track B: Extended analysis

Goal:

- evaluate the stronger LAION checkpoint
- add FSDD
- report Macro-F1, Balanced Accuracy, Top-5 Accuracy, MRR, and ECE
- test ensemble prompting

Recommended command:

```bash
bash scripts/repro/run_extensions.sh
```

Suggested metadata:

- `RUN_TAG=laion-630k-extensions`
- `CHECKPOINT_LABEL=laion-630k-improved`

## Minimum acceptance checklist

- dataset verification passes
- checkpoint files are logged
- both repos have recorded git commits
- the run manifest was written
- metrics JSON files exist for every evaluated dataset
- summary tables exist under `outputs/tables`
