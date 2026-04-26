"""Backward-compatible entry point for the multi-dataset evaluation runner."""
from scripts.analysis.run_zeroshot_metrics import main
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    main()
