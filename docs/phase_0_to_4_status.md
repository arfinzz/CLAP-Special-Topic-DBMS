# Phase 0-4 Status

This document defines what "complete through Phase 4" means for this project.

## Phase 0: WSL base setup

Success criteria:

- WSL is being used, not `/mnt/c`
- system packages are installed
- optional GPU visibility is checked with `nvidia-smi`

Command:

```bash
bash scripts/setup/prepare_wsl_workspace.sh
```

## Phase 1: Get the code cleanly

Success criteria:

- this repository exists under `~/projects/clap-reimpl`
- the official LAION repo exists under `~/projects/clap-official`

Command:

```bash
bash scripts/setup/clone_official_repo.sh
```

## Phase 2: Separate environments

Success criteria:

- `clap-official-env` is created
- `clap-reimpl-env` is created

These are created by the Phase 3 and Phase 4 scripts below.

## Phase 3: Official baseline environment

Success criteria:

- official repo dependencies install successfully
- `pip install -e ~/projects/clap-official` works
- `python -c "import laion_clap"` succeeds inside `clap-official-env`

Command:

```bash
bash scripts/setup/setup_official_env.sh
```

## Phase 4: Reimplementation environment

Success criteria:

- analysis dependencies install successfully
- official `laion_clap` is importable inside `clap-reimpl-env`
- `python -c "import laion_clap, torch"` succeeds

Command:

```bash
bash scripts/setup/setup_reimpl_env.sh
```

## One-shot bootstrap

To complete all phases together:

```bash
bash scripts/setup/bootstrap_phase_0_to_4.sh
```

## Important boundary

Completing Phase 4 does not mean the project is ready to run experiments yet.
After that you still need:

- dataset downloads
- dataset linking
- checkpoint downloads
- asset verification
- actual experiment runs
