from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro_utils import now_utc_iso, write_json

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "esc50": {
        "display_name": "ESC-50",
        "paths": [
            ["esc50", "ESC-50-master", "audio"],
        ],
        "glob": "*.wav",
        "expected_count": 2000,
        "required_files": [],
        "notes": "Official extraction should contain 2000 wav files.",
    },
    "gtzan": {
        "display_name": "GTZAN",
        "paths": [
            ["gtzan", "genres"],
        ],
        "glob": "*/*.wav",
        "expected_count": 999,
        "required_files": [],
        "notes": "Project reports 999 files after removing corrupted jazz.00054.wav.",
    },
    "urbansound8k": {
        "display_name": "UrbanSound8K",
        "paths": [
            ["urbansound8k", "UrbanSound8K"],
        ],
        "glob": "audio/fold*/*.wav",
        "expected_count": 8732,
        "required_files": [
            "metadata/UrbanSound8K.csv",
        ],
        "notes": "Official dataset extraction should contain 8732 wav files plus metadata CSV.",
    },
    "fsdd": {
        "display_name": "FSDD",
        "paths": [
            ["fsdd", "free-spoken-digit-dataset", "recordings"],
            ["fsdd", "recordings"],
            ["fsdd", "free-spoken-digit-dataset-master", "recordings"],
            ["fsdd", "Free-Spoken-Digit-Dataset", "recordings"],
        ],
        "glob": "*.wav",
        "expected_count": 3000,
        "required_files": [],
        "notes": "The official FSDD clone should contain 3000 wav files.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify dataset and checkpoint assets for CLAP reproduction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=REPO_ROOT / "data" / "datasets",
        help="Root directory containing dataset folders or symlinks.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=None,
        help="Optional root directory containing downloaded checkpoints.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_SPECS),
        choices=sorted(DATASET_SPECS),
        help="Datasets to validate.",
    )
    parser.add_argument(
        "--require-checkpoints",
        nargs="*",
        default=[],
        help="Checkpoint filenames that must exist under --checkpoints-root.",
    )
    parser.add_argument(
        "--hash-checkpoints",
        action="store_true",
        help="Compute SHA256 for every required checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "manifests" / "asset_check.json",
        help="Path for the machine-readable verification report.",
    )
    return parser.parse_args()


def resolve_existing_path(root: Path, candidates: list[list[str]]) -> Path | None:
    for parts in candidates:
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset_key: str, datasets_root: Path) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset_key]
    resolved = resolve_existing_path(datasets_root, spec["paths"])

    report: dict[str, Any] = {
        "dataset_key": dataset_key,
        "display_name": spec["display_name"],
        "expected_count": spec["expected_count"],
        "notes": spec["notes"],
        "ok": False,
    }

    if resolved is None:
        report["error"] = f"Missing dataset path under {datasets_root}"
        return report

    wav_files = sorted(resolved.glob(spec["glob"]))
    missing_files = []
    for required in spec["required_files"]:
        required_path = resolved / required
        if not required_path.exists():
            missing_files.append(str(required_path))

    report.update(
        {
            "resolved_path": str(resolved),
            "observed_count": len(wav_files),
            "missing_files": missing_files,
            "ok": len(wav_files) == spec["expected_count"] and not missing_files,
        }
    )

    if dataset_key == "gtzan" and len(wav_files) == 1000:
        report["ok"] = False
        report["warning"] = (
            "GTZAN has 1000 wav files; this project expects 999 after removing corrupted jazz.00054.wav."
        )

    if not report["ok"] and "error" not in report:
        report["error"] = (
            f"Expected {spec['expected_count']} wav files, found {len(wav_files)}."
        )
    return report


def validate_checkpoint(path: Path, hash_file: bool) -> dict[str, Any]:
    report = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        report["size_bytes"] = path.stat().st_size
        if hash_file:
            report["sha256"] = sha256_file(path)
    return report


def main() -> None:
    args = parse_args()

    dataset_reports = [validate_dataset(dataset_key, args.datasets_root) for dataset_key in args.datasets]

    checkpoint_reports: list[dict[str, Any]] = []
    if args.require_checkpoints:
        if args.checkpoints_root is None:
            raise ValueError("--checkpoints-root is required when --require-checkpoints is used.")
        for filename in args.require_checkpoints:
            checkpoint_reports.append(
                validate_checkpoint(args.checkpoints_root / filename, hash_file=args.hash_checkpoints)
            )

    report = {
        "generated_at_utc": now_utc_iso(),
        "datasets_root": str(args.datasets_root),
        "checkpoints_root": str(args.checkpoints_root) if args.checkpoints_root else None,
        "dataset_reports": dataset_reports,
        "checkpoint_reports": checkpoint_reports,
        "all_ok": all(item["ok"] for item in dataset_reports)
        and all(item["exists"] for item in checkpoint_reports),
    }

    write_json(args.output, report)
    print(f"Saved asset verification report: {args.output}")

    for item in dataset_reports:
        status = "OK" if item["ok"] else "FAIL"
        observed = item.get("observed_count", 0)
        print(f"[{status}] {item['display_name']}: {observed}/{item['expected_count']} files")
        if item.get("warning"):
            print(f"  warning: {item['warning']}")
        if item.get("error"):
            print(f"  error: {item['error']}")

    for item in checkpoint_reports:
        status = "OK" if item["exists"] else "FAIL"
        print(f"[{status}] checkpoint: {item['path']}")

    if not report["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
