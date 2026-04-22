from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro_utils import now_utc_iso, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that CLAP reproduction outputs satisfy the acceptance checklist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Root directory containing manifests, metrics, figures, and tables.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["esc50", "urbansound8k", "fsdd"],
        help="Datasets that should have metrics outputs.",
    )
    parser.add_argument(
        "--expect-ensemble",
        action="store_true",
        help="Require ensemble metrics JSON files for every dataset.",
    )
    parser.add_argument(
        "--manifests",
        nargs="*",
        default=[],
        help="Specific run-manifest filenames expected under outputs/manifests.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=None,
        help="Optional checkpoint directory to validate.",
    )
    parser.add_argument(
        "--require-checkpoints",
        nargs="*",
        default=[],
        help="Checkpoint filenames expected under --checkpoints-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "manifests" / "acceptance_check.json",
        help="Path for the acceptance report JSON.",
    )
    return parser.parse_args()


def file_report(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def manifest_has_required_fields(path: Path) -> dict[str, Any]:
    report = file_report(path)
    if not report["exists"]:
        report["ok"] = False
        return report

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report["ok"] = False
        report["error"] = f"Invalid JSON: {exc}"
        return report

    git_info = payload.get("git", {})
    python_info = payload.get("python", {})
    torch_info = payload.get("torch", {})

    report["ok"] = all(
        [
            bool(git_info.get("commit")),
            bool(python_info.get("python_version")),
            isinstance(torch_info.get("available"), bool),
            bool(payload.get("evaluated_datasets")),
        ]
    )
    report["evaluated_datasets"] = payload.get("evaluated_datasets", [])
    report["git_commit"] = git_info.get("short_commit") or git_info.get("commit")
    return report


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root
    metrics_root = outputs_root / "metrics"
    manifests_root = outputs_root / "manifests"
    tables_root = outputs_root / "tables"

    dataset_reports = []
    for dataset in args.datasets:
        dataset_reports.append(
            {
                "dataset": dataset,
                "metrics": file_report(metrics_root / f"{dataset}.json"),
                "ensemble_metrics": file_report(metrics_root / f"{dataset}_ensemble.json"),
            }
        )

    summary_reports = {
        "summary_csv": file_report(tables_root / "zeroshot_summary.csv"),
        "summary_md": file_report(tables_root / "zeroshot_summary.md"),
    }

    manifest_reports = []
    for manifest_name in args.manifests:
        manifest_reports.append(manifest_has_required_fields(manifests_root / manifest_name))

    checkpoint_reports = []
    if args.require_checkpoints:
        if args.checkpoints_root is None:
            raise ValueError("--checkpoints-root is required when --require-checkpoints is used.")
        for filename in args.require_checkpoints:
            checkpoint_reports.append(file_report(args.checkpoints_root / filename))

    all_ok = all(item["metrics"]["exists"] for item in dataset_reports)
    if args.expect_ensemble:
        all_ok = all_ok and all(item["ensemble_metrics"]["exists"] for item in dataset_reports)
    all_ok = all_ok and all(item["exists"] for item in summary_reports.values())
    all_ok = all_ok and all(item.get("ok", item["exists"]) for item in manifest_reports)
    all_ok = all_ok and all(item["exists"] for item in checkpoint_reports)

    payload = {
        "generated_at_utc": now_utc_iso(),
        "outputs_root": str(outputs_root),
        "datasets": args.datasets,
        "dataset_reports": dataset_reports,
        "summary_reports": summary_reports,
        "manifest_reports": manifest_reports,
        "checkpoint_reports": checkpoint_reports,
        "all_ok": all_ok,
    }
    write_json(args.output, payload)
    print(f"Saved acceptance report: {args.output}")

    for item in dataset_reports:
        metrics_status = "OK" if item["metrics"]["exists"] else "FAIL"
        print(f"[{metrics_status}] metrics: {item['metrics']['path']}")
        if args.expect_ensemble:
            ensemble_status = "OK" if item["ensemble_metrics"]["exists"] else "FAIL"
            print(f"[{ensemble_status}] ensemble: {item['ensemble_metrics']['path']}")

    for label, report in summary_reports.items():
        status = "OK" if report["exists"] else "FAIL"
        print(f"[{status}] {label}: {report['path']}")

    for report in manifest_reports:
        status = "OK" if report.get("ok") else "FAIL"
        print(f"[{status}] manifest: {report['path']}")

    for report in checkpoint_reports:
        status = "OK" if report["exists"] else "FAIL"
        print(f"[{status}] checkpoint: {report['path']}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
