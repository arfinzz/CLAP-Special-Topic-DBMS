PYTHON ?= python

.PHONY: phase-0-4 verify-assets fetch-checkpoint acceptance-check run-baseline run-extensions

phase-0-4:
	bash scripts/setup/bootstrap_phase_0_to_4.sh

verify-assets:
	$(PYTHON) scripts/repro/verify_assets.py --datasets esc50 urbansound8k gtzan fsdd

fetch-checkpoint:
	bash scripts/setup/fetch_checkpoint.sh

acceptance-check:
	$(PYTHON) scripts/repro/check_acceptance.py

run-baseline:
	bash scripts/repro/run_reproduction.sh

run-extensions:
	bash scripts/repro/run_extensions.sh
