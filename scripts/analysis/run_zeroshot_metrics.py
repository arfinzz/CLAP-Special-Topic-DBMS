from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import laion_clap
import matplotlib
from repro_utils import build_run_manifest, write_json

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

if not hasattr(np.random, "integers"):
    # Older NumPy exposes randint on the module RNG but not integers.
    np.random.integers = np.random.randint  # type: ignore[attr-defined]


CLASS_LABELS_DIR = REPO_ROOT / "class_labels"
DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

FSDD_FILENAME = re.compile(r"^(?P<label>\d+)_(?P<speaker>[^_]+)_(?P<index>\d+)\.wav$")
FSDD_DIGITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

PROMPT_FAMILIES = {
    "sound": [
        "This is a sound of {label}.",
        "The sound of {label}.",
        "A recording of {label}.",
        "An audio clip of {label}.",
        "I can hear {label}.",
    ],
    "music": [
        "This audio is a {label} song.",
        "A recording of {label} music.",
        "This is {label} music.",
        "An audio clip of a {label} song.",
        "I can hear a {label} song.",
    ],
    "speech": [
        "A person saying {label}.",
        "Someone says the word {label}.",
        "This audio contains the spoken word {label}.",
        "A recording of a speaker saying {label}.",
        "I can hear someone say {label}.",
    ],
}

DATASET_CHOICES = ("esc50", "urbansound8k", "gtzan", "fsdd")


@dataclass
class DatasetBundle:
    key: str
    display_name: str
    root_dir: Path
    prompt_family: str
    class_names: list[str]
    audio_files: list[str]
    labels: np.ndarray
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot CLAP evaluation with extended metrics and table export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["esc50", "urbansound8k", "gtzan"],
        help="Datasets to evaluate. Use 'all' to run every supported dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, CLAP downloads/loads the default checkpoint.",
    )
    parser.add_argument(
        "--amodel",
        type=str,
        default=None,
        help="Optional audio backbone name, for example 'HTSAT-base' for larger checkpoints.",
    )
    parser.add_argument(
        "--enable-fusion",
        action="store_true",
        help="Enable the CLAP fusion model path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device for inference and similarity scoring.",
    )
    parser.add_argument(
        "--audio-batch-size",
        type=int,
        default=256,
        help="Number of audio files to embed per CLAP call.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for Expected Calibration Error.",
    )
    parser.add_argument(
        "--force-recompute-audio",
        action="store_true",
        help="Ignore cached audio embeddings and recompute them.",
    )
    parser.add_argument(
        "--force-recompute-text",
        action="store_true",
        help="Ignore cached text embeddings and recompute them.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets that are not present locally instead of failing the whole run.",
    )
    parser.add_argument(
        "--fsdd-split",
        choices=("test", "train", "all"),
        default="test",
        help="Subset of FSDD to evaluate. The official split uses indices 0-4 for test.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="zeroshot-metrics",
        help="Short label written into the run manifest.",
    )
    parser.add_argument(
        "--checkpoint-label",
        type=str,
        default="auto-default",
        help="Human-readable checkpoint label for reports and manifests.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional free-form notes stored in the run manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")
    ensure_dir(OUTPUTS_ROOT)

    requested_datasets = resolve_datasets(args.datasets)
    device = resolve_device(args.device)
    print("=" * 72)
    print("CLAP multi-dataset zero-shot evaluation")
    print("=" * 72)
    print(f"Repository root : {REPO_ROOT}")
    print(f"Datasets        : {', '.join(requested_datasets)}")
    print(f"Device          : {device}")
    print(f"Checkpoint      : {args.checkpoint or 'default CLAP checkpoint'}")
    print(f"Audio model     : {args.amodel or 'CLAP default'}")
    print()

    model = build_model(args, device)
    summary_rows: list[dict[str, Any]] = []
    evaluated_datasets: list[str] = []
    skipped_datasets: list[dict[str, str]] = []

    for dataset_key in requested_datasets:
        print("=" * 72)
        print(f"Dataset: {dataset_key}")
        print("=" * 72)
        try:
            bundle = load_dataset_bundle(dataset_key, args)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"Skipping {dataset_key}: {exc}")
                print()
                skipped_datasets.append({"dataset": dataset_key, "reason": str(exc)})
                continue
            raise

        print(f"Samples : {len(bundle.audio_files)}")
        print(f"Classes : {len(bundle.class_names)}")
        print(f"Root    : {bundle.root_dir}")

        prompt_templates = PROMPT_FAMILIES[bundle.prompt_family]
        default_prompt = prompt_templates[0]
        embedding_dir = ensure_dir(OUTPUTS_ROOT / "embeddings" / dataset_key)
        figure_dir = ensure_dir(OUTPUTS_ROOT / "figures" / dataset_key)

        audio_embeddings = load_or_compute_audio_embeddings(
            model=model,
            bundle=bundle,
            cache_dir=embedding_dir,
            batch_size=args.audio_batch_size,
            force_recompute=args.force_recompute_audio,
        )

        default_text_embeddings = load_or_compute_text_embeddings(
            model=model,
            class_names=bundle.class_names,
            prompt_template=default_prompt,
            cache_dir=embedding_dir,
            force_recompute=args.force_recompute_text,
        )

        default_results = evaluate_prompt(
            audio_embeddings=audio_embeddings,
            text_embeddings=default_text_embeddings,
            labels=bundle.labels,
            device=device,
            n_bins=args.n_bins,
        )

        per_class_stats = compute_per_class_accuracy(
            labels=bundle.labels,
            predictions=default_results["predictions"],
            class_names=bundle.class_names,
        )
        confusion_pairs = compute_top_confused_pairs(
            cm=default_results["confusion_matrix"],
            class_names=bundle.class_names,
            top_n=10,
        )

        prompt_results, prompt_summary = evaluate_prompt_sensitivity(
            model=model,
            bundle=bundle,
            audio_embeddings=audio_embeddings,
            cache_dir=embedding_dir,
            device=device,
            force_recompute=args.force_recompute_text,
            n_bins=args.n_bins,
        )

        figure_paths = {
            "reliability": figure_dir / "reliability.png",
            "per_class_accuracy": figure_dir / "per_class_accuracy.png",
            "confusion_matrix": figure_dir / "confusion_matrix.png",
            "prompt_sensitivity": figure_dir / "prompt_sensitivity.png",
        }

        save_reliability_figure(
            metrics=default_results["metrics"],
            ece_info=default_results["ece"],
            dataset_name=bundle.display_name,
            output_path=figure_paths["reliability"],
        )
        save_per_class_accuracy_figure(
            per_class_stats=per_class_stats,
            overall_accuracy=default_results["metrics"]["accuracy"],
            dataset_name=bundle.display_name,
            output_path=figure_paths["per_class_accuracy"],
        )
        save_confusion_figure(
            cm=default_results["confusion_matrix"],
            class_names=bundle.class_names,
            confused_pairs=confusion_pairs,
            dataset_name=bundle.display_name,
            output_path=figure_paths["confusion_matrix"],
        )
        save_prompt_sensitivity_figure(
            prompt_results=prompt_results,
            prompt_summary=prompt_summary,
            dataset_name=bundle.display_name,
            output_path=figure_paths["prompt_sensitivity"],
        )

        metrics_path = OUTPUTS_ROOT / "metrics" / f"{dataset_key}.json"
        report = {
            "dataset": bundle.display_name,
            "dataset_key": bundle.key,
            "root_dir": str(bundle.root_dir),
            "num_samples": int(len(bundle.audio_files)),
            "num_classes": int(len(bundle.class_names)),
            "checkpoint_label": args.checkpoint_label,
            "default_prompt": default_prompt,
            "default_prompt_metrics": default_results["metrics"],
            "prompt_sensitivity": prompt_summary,
            "prompt_results": prompt_results,
            "hardest_classes": per_class_stats[:10],
            "easiest_classes": list(reversed(per_class_stats[-10:])),
            "top_confused_pairs": confusion_pairs,
            "metadata": bundle.metadata,
            "output_files": {key: str(path) for key, path in figure_paths.items()},
        }
        write_json(metrics_path, report)
        evaluated_datasets.append(dataset_key)

        summary_rows.append(
            {
                "dataset": bundle.display_name,
                "dataset_key": bundle.key,
                "num_samples": int(len(bundle.audio_files)),
                "num_classes": int(len(bundle.class_names)),
                "default_prompt": default_prompt,
                "accuracy": default_results["metrics"]["accuracy"],
                "macro_f1": default_results["metrics"]["macro_f1"],
                "balanced_accuracy": default_results["metrics"]["balanced_accuracy"],
                "ece": default_results["metrics"]["ece"],
                "top_5_accuracy": default_results["metrics"]["top_5_accuracy"],
                "mrr": default_results["metrics"]["mrr"],
                "pss": prompt_summary["pss"],
                "best_prompt_accuracy": prompt_summary["best_prompt_accuracy"],
                "best_prompt": prompt_summary["best_prompt"],
            }
        )

        print_metrics_summary(
            bundle=bundle,
            default_prompt=default_prompt,
            metrics=default_results["metrics"],
            prompt_summary=prompt_summary,
            confusion_pairs=confusion_pairs,
            per_class_stats=per_class_stats,
        )
        print(f"Saved metrics JSON : {metrics_path}")
        print(f"Saved figures      : {figure_dir}")
        print()

    if not summary_rows:
        raise RuntimeError("No dataset was evaluated. Check dataset availability or your CLI flags.")

    write_summary_tables(summary_rows)
    manifest_path = OUTPUTS_ROOT / "manifests" / f"{slugify(args.run_tag)}.json"
    summary_csv = OUTPUTS_ROOT / "tables" / "zeroshot_summary.csv"
    summary_md = OUTPUTS_ROOT / "tables" / "zeroshot_summary.md"
    manifest = build_run_manifest(
        repo_root=REPO_ROOT,
        run_kind="zeroshot_metrics",
        cli_args=vars(args),
        requested_datasets=requested_datasets,
        evaluated_datasets=evaluated_datasets,
        skipped_datasets=skipped_datasets,
        output_files={
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "metrics_dir": str(OUTPUTS_ROOT / "metrics"),
            "figures_dir": str(OUTPUTS_ROOT / "figures"),
        },
        extra={
            "checkpoint_label": args.checkpoint_label,
            "notes": args.notes,
        },
    )
    write_json(manifest_path, manifest)
    print(f"Saved run manifest: {manifest_path}")
    print("=" * 72)
    print("Completed. Summary tables written to outputs/tables/")
    print("=" * 72)


def resolve_datasets(raw_datasets: list[str]) -> list[str]:
    requested = [item.lower() for item in raw_datasets]
    if "all" in requested:
        return list(DATASET_CHOICES)

    invalid = sorted(set(requested) - set(DATASET_CHOICES))
    if invalid:
        raise ValueError(
            f"Unsupported dataset(s): {', '.join(invalid)}. Choices: {', '.join(DATASET_CHOICES)}"
        )
    return requested


def resolve_device(device_choice: str) -> str:
    if device_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return device_choice


def build_model(args: argparse.Namespace, device: str) -> Any:
    model_kwargs: dict[str, Any] = {"enable_fusion": args.enable_fusion, "device": device}
    if args.amodel:
        model_kwargs["amodel"] = args.amodel

    model = laion_clap.CLAP_Module(**model_kwargs)
    if args.checkpoint:
        model.load_ckpt(args.checkpoint)
    else:
        model.load_ckpt()
    return model


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "prompt"


def load_class_names_from_json(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        label_map = json.load(handle)
    return [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]


def load_dataset_bundle(dataset_key: str, args: argparse.Namespace) -> DatasetBundle:
    loaders = {
        "esc50": load_esc50_bundle,
        "urbansound8k": load_urbansound8k_bundle,
        "gtzan": load_gtzan_bundle,
        "fsdd": load_fsdd_bundle,
    }
    return loaders[dataset_key](args)


def load_esc50_bundle(_: argparse.Namespace) -> DatasetBundle:
    root_dir = DATASETS_ROOT / "esc50" / "ESC-50-master"
    audio_dir = root_dir / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"Expected ESC-50 audio folder at {audio_dir}")

    class_names = load_class_names_from_json(CLASS_LABELS_DIR / "ESC50_class_labels_indices_space.json")
    audio_paths = sorted(audio_dir.glob("*.wav"))
    audio_files = [str(path) for path in audio_paths]
    labels = np.array([int(path.stem.split("-")[-1]) for path in audio_paths], dtype=np.int64)

    return DatasetBundle(
        key="esc50",
        display_name="ESC-50",
        root_dir=root_dir,
        prompt_family="sound",
        class_names=class_names,
        audio_files=audio_files,
        labels=labels,
        metadata={},
    )


def load_urbansound8k_bundle(_: argparse.Namespace) -> DatasetBundle:
    root_dir = DATASETS_ROOT / "urbansound8k" / "UrbanSound8K"
    metadata_path = root_dir / "metadata" / "UrbanSound8K.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Expected UrbanSound8K metadata at {metadata_path}")

    metadata_df = pd.read_csv(metadata_path).sort_values(["fold", "slice_file_name"]).reset_index(drop=True)
    class_names = load_class_names_from_json(CLASS_LABELS_DIR / "UrbanSound8K_class_labels_indices.json")

    audio_files: list[str] = []
    labels: list[int] = []
    missing_files = 0

    for row in metadata_df.itertuples(index=False):
        audio_path = root_dir / "audio" / f"fold{int(row.fold)}" / row.slice_file_name
        if audio_path.exists():
            audio_files.append(str(audio_path))
            labels.append(int(row.classID))
        else:
            missing_files += 1

    if not audio_files:
        raise FileNotFoundError(f"No UrbanSound8K wav files were found under {root_dir / 'audio'}")

    return DatasetBundle(
        key="urbansound8k",
        display_name="UrbanSound8K",
        root_dir=root_dir,
        prompt_family="sound",
        class_names=class_names,
        audio_files=audio_files,
        labels=np.array(labels, dtype=np.int64),
        metadata={"missing_files": missing_files},
    )


def load_gtzan_bundle(_: argparse.Namespace) -> DatasetBundle:
    root_dir = DATASETS_ROOT / "gtzan" / "genres"
    if not root_dir.exists():
        raise FileNotFoundError(f"Expected GTZAN folder at {root_dir}")

    class_names = load_class_names_from_json(CLASS_LABELS_DIR / "GTZAN_class_labels.json")
    label_to_index = {label: index for index, label in enumerate(class_names)}

    audio_paths = sorted(
        path
        for path in root_dir.glob("*/*.wav")
        if path.parent.is_dir() and not path.name.startswith("._")
    )
    if not audio_paths:
        raise FileNotFoundError(f"No GTZAN wav files were found under {root_dir}")

    audio_files = [str(path) for path in audio_paths]
    labels = np.array([label_to_index[path.parent.name] for path in audio_paths], dtype=np.int64)

    return DatasetBundle(
        key="gtzan",
        display_name="GTZAN",
        root_dir=root_dir,
        prompt_family="music",
        class_names=class_names,
        audio_files=audio_files,
        labels=labels,
        metadata={},
    )


def load_fsdd_bundle(args: argparse.Namespace) -> DatasetBundle:
    root_dir = DATASETS_ROOT / "fsdd"
    recordings_dir = resolve_fsdd_recordings_dir(root_dir)

    audio_files: list[str] = []
    labels: list[int] = []
    speakers: set[str] = set()

    for path in sorted(recordings_dir.glob("*.wav")):
        match = FSDD_FILENAME.match(path.name)
        if not match:
            continue

        recording_index = int(match.group("index"))
        if args.fsdd_split == "test" and recording_index > 4:
            continue
        if args.fsdd_split == "train" and recording_index <= 4:
            continue

        label_index = int(match.group("label"))
        audio_files.append(str(path))
        labels.append(label_index)
        speakers.add(match.group("speaker"))

    if not audio_files:
        raise FileNotFoundError(
            f"No FSDD recordings matched split='{args.fsdd_split}' under {recordings_dir}"
        )

    return DatasetBundle(
        key="fsdd",
        display_name="Free Spoken Digit Dataset",
        root_dir=recordings_dir,
        prompt_family="speech",
        class_names=FSDD_DIGITS,
        audio_files=audio_files,
        labels=np.array(labels, dtype=np.int64),
        metadata={"split": args.fsdd_split, "speakers": sorted(speakers)},
    )


def resolve_fsdd_recordings_dir(root_dir: Path) -> Path:
    candidates = [
        root_dir / "recordings",
        root_dir / "free-spoken-digit-dataset" / "recordings",
        root_dir / "free-spoken-digit-dataset-master" / "recordings",
        root_dir / "Free-Spoken-Digit-Dataset" / "recordings",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Expected FSDD recordings under data/datasets/fsdd/. "
        "Clone the official dataset there before running this benchmark."
    )


def load_or_compute_audio_embeddings(
    model: Any,
    bundle: DatasetBundle,
    cache_dir: Path,
    batch_size: int,
    force_recompute: bool,
) -> np.ndarray:
    audio_cache = cache_dir / "audio_embeddings.npy"
    labels_cache = cache_dir / "labels.npy"
    filelist_cache = cache_dir / "audio_files.json"

    if audio_cache.exists() and labels_cache.exists() and not force_recompute:
        cached_audio = np.load(audio_cache)
        cached_labels = np.load(labels_cache)
        cached_filelist = None
        if filelist_cache.exists():
            with filelist_cache.open("r", encoding="utf-8") as handle:
                cached_filelist = json.load(handle)

        lengths_match = len(cached_audio) == len(bundle.audio_files)
        labels_match = np.array_equal(cached_labels, bundle.labels)
        files_match = cached_filelist == bundle.audio_files if cached_filelist is not None else True
        if lengths_match and labels_match and files_match:
            print(f"Using cached audio embeddings from {audio_cache}")
            return cached_audio

    total_batches = math.ceil(len(bundle.audio_files) / batch_size)
    batch_embeddings: list[np.ndarray] = []
    for batch_idx, start in enumerate(range(0, len(bundle.audio_files), batch_size), start=1):
        batch_files = bundle.audio_files[start : start + batch_size]
        print(f"Embedding audio batch {batch_idx}/{total_batches} ({len(batch_files)} files)")
        with torch.no_grad():
            embeddings = model.get_audio_embedding_from_filelist(x=batch_files, use_tensor=False)
        batch_embeddings.append(np.asarray(embeddings))

    audio_embeddings = np.concatenate(batch_embeddings, axis=0)
    np.save(audio_cache, audio_embeddings)
    np.save(labels_cache, bundle.labels)
    with filelist_cache.open("w", encoding="utf-8") as handle:
        json.dump(bundle.audio_files, handle, indent=2)
    return audio_embeddings


def load_or_compute_text_embeddings(
    model: Any,
    class_names: list[str],
    prompt_template: str,
    cache_dir: Path,
    force_recompute: bool,
) -> np.ndarray:
    prompt_slug = slugify(prompt_template)
    text_cache = cache_dir / f"text_embeddings__{prompt_slug}.npy"
    if text_cache.exists() and not force_recompute:
        cached_text = np.load(text_cache)
        if len(cached_text) == len(class_names):
            return cached_text

    texts = [prompt_template.format(label=label) for label in class_names]
    with torch.no_grad():
        text_embeddings = model.get_text_embedding(texts)
    text_embeddings = np.asarray(text_embeddings)
    np.save(text_cache, text_embeddings)
    return text_embeddings


def evaluate_prompt(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: np.ndarray,
    device: str,
    n_bins: int,
) -> dict[str, Any]:
    audio_tensor = torch.as_tensor(audio_embeddings, device=device)
    text_tensor = torch.as_tensor(text_embeddings, device=device)
    logits = audio_tensor @ text_tensor.T
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    rankings = torch.argsort(logits, dim=1, descending=True).cpu().numpy()

    metrics, ece_info = compute_metrics(probs=probs, rankings=rankings, labels=labels, n_bins=n_bins)
    predictions = probs.argmax(axis=1)
    cm = confusion_matrix(labels, predictions, labels=list(range(len(text_embeddings))))

    return {
        "metrics": metrics,
        "ece": ece_info,
        "predictions": predictions,
        "rankings": rankings,
        "probabilities": probs,
        "confusion_matrix": cm,
    }


def evaluate_prompt_sensitivity(
    model: Any,
    bundle: DatasetBundle,
    audio_embeddings: np.ndarray,
    cache_dir: Path,
    device: str,
    force_recompute: bool,
    n_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_results: list[dict[str, Any]] = []

    for prompt_template in PROMPT_FAMILIES[bundle.prompt_family]:
        text_embeddings = load_or_compute_text_embeddings(
            model=model,
            class_names=bundle.class_names,
            prompt_template=prompt_template,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
        )
        prompt_eval = evaluate_prompt(
            audio_embeddings=audio_embeddings,
            text_embeddings=text_embeddings,
            labels=bundle.labels,
            device=device,
            n_bins=n_bins,
        )
        prompt_results.append(
            {
                "prompt": prompt_template,
                "accuracy": prompt_eval["metrics"]["accuracy"],
            }
        )

    accuracies = np.array([item["accuracy"] for item in prompt_results], dtype=np.float64)
    mean_accuracy = float(accuracies.mean())
    std_accuracy = float(accuracies.std())
    pss = float(std_accuracy / mean_accuracy) if mean_accuracy > 0 else 0.0
    best_index = int(accuracies.argmax())
    worst_index = int(accuracies.argmin())

    return prompt_results, {
        "pss": pss,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "best_prompt": prompt_results[best_index]["prompt"],
        "best_prompt_accuracy": float(accuracies[best_index]),
        "worst_prompt": prompt_results[worst_index]["prompt"],
        "worst_prompt_accuracy": float(accuracies[worst_index]),
        "best_worst_gap": float(accuracies[best_index] - accuracies[worst_index]),
    }


def compute_metrics(
    probs: np.ndarray,
    rankings: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    predictions = probs.argmax(axis=1)
    ranks = np.argmax(rankings == labels[:, None], axis=1)
    top_k = min(5, probs.shape[1])

    ece, bin_accs, bin_confs, bin_counts = compute_ece(probs, labels, n_bins=n_bins)
    metrics = {
        "accuracy": float(np.mean(predictions == labels)),
        "r_at_1": float(np.mean(predictions == labels)),
        "top_5_accuracy": float(np.mean(ranks < top_k)),
        "mrr": float(np.mean(1.0 / (ranks + 1))),
        "mean_rank": float(ranks.mean() + 1.0),
        "median_rank": float(np.floor(np.median(ranks)) + 1.0),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "ece": float(ece),
        "mean_confidence": float(np.mean(probs.max(axis=1))),
    }
    return metrics, {
        "ece": float(ece),
        "bin_accuracies": [float(value) for value in bin_accs],
        "bin_confidences": [float(value) for value in bin_confs],
        "bin_counts": [int(value) for value in bin_counts],
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[float, list[float], list[float], list[int]]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bin_accuracies: list[float] = []
    bin_confidences: list[float] = []
    bin_counts: list[int] = []

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if lower == 0.0:
            mask = (confidences >= lower) & (confidences <= upper)

        if mask.any():
            accuracy = float(correctness[mask].mean())
            confidence = float(confidences[mask].mean())
            count = int(mask.sum())
            ece += abs(accuracy - confidence) * (count / len(labels))
            bin_accuracies.append(accuracy)
            bin_confidences.append(confidence)
            bin_counts.append(count)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((lower + upper) / 2.0))
            bin_counts.append(0)

    return float(ece), bin_accuracies, bin_confidences, bin_counts


def compute_per_class_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    per_class_stats: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        if not mask.any():
            continue
        accuracy = float(np.mean(predictions[mask] == class_index))
        per_class_stats.append(
            {
                "class_name": class_name,
                "accuracy": accuracy,
                "support": int(mask.sum()),
            }
        )
    return sorted(per_class_stats, key=lambda item: item["accuracy"])


def compute_top_confused_pairs(
    cm: np.ndarray,
    class_names: list[str],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    pairs: list[dict[str, Any]] = []
    for true_index in range(cm_no_diag.shape[0]):
        for pred_index in range(cm_no_diag.shape[1]):
            count = int(cm_no_diag[true_index, pred_index])
            if count <= 0:
                continue
            pairs.append(
                {
                    "count": count,
                    "true_label": class_names[true_index],
                    "predicted_label": class_names[pred_index],
                }
            )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:top_n]


def save_reliability_figure(
    metrics: dict[str, float],
    ece_info: dict[str, Any],
    dataset_name: str,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    bins_mid = np.linspace(0.05, 0.95, len(ece_info["bin_accuracies"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        bins_mid,
        ece_info["bin_accuracies"],
        width=0.08,
        alpha=0.75,
        color="steelblue",
        edgecolor="black",
        label="Accuracy per bin",
    )
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{dataset_name} reliability\nECE = {metrics['ece']:.4f}")
    axes[0].legend()

    axes[1].bar(
        bins_mid,
        ece_info["bin_counts"],
        width=0.08,
        alpha=0.8,
        color="coral",
        edgecolor="black",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Number of samples")
    axes[1].set_title("Confidence histogram")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_per_class_accuracy_figure(
    per_class_stats: list[dict[str, Any]],
    overall_accuracy: float,
    dataset_name: str,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    class_names = [item["class_name"] for item in per_class_stats]
    accuracies = [item["accuracy"] for item in per_class_stats]
    colors = [
        "#d73027" if accuracy < 0.5 else "#fee08b" if accuracy < 0.8 else "#1a9850"
        for accuracy in accuracies
    ]

    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(class_names))))
    ax.barh(class_names, accuracies, color=colors, edgecolor="grey", linewidth=0.5)
    ax.axvline(overall_accuracy, color="navy", linestyle="--", linewidth=2, label=f"Overall={overall_accuracy:.3f}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title(f"{dataset_name} per-class accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_figure(
    cm: np.ndarray,
    class_names: list[str],
    confused_pairs: list[dict[str, Any]],
    dataset_name: str,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    top_classes = list(
        dict.fromkeys(
            label
            for pair in confused_pairs[:10]
            for label in (pair["true_label"], pair["predicted_label"])
        )
    )
    if not top_classes:
        top_classes = class_names[: min(10, len(class_names))]

    indices = [class_names.index(label) for label in top_classes]
    submatrix = cm[np.ix_(indices, indices)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        submatrix,
        xticklabels=top_classes,
        yticklabels=top_classes,
        cmap="Blues",
        annot=True,
        fmt="d",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"{dataset_name} confusion matrix\n(most-confused classes)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_prompt_sensitivity_figure(
    prompt_results: list[dict[str, Any]],
    prompt_summary: dict[str, Any],
    dataset_name: str,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    labels = [item["prompt"].replace("{label}", "[LABEL]") for item in prompt_results]
    accuracies = [item["accuracy"] for item in prompt_results]
    best_accuracy = prompt_summary["best_prompt_accuracy"]
    worst_accuracy = prompt_summary["worst_prompt_accuracy"]

    colors = []
    for accuracy in accuracies:
        if accuracy == best_accuracy:
            colors.append("#2166ac")
        elif accuracy == worst_accuracy:
            colors.append("#d73027")
        else:
            colors.append("steelblue")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, accuracies, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(
        prompt_summary["mean_accuracy"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean={prompt_summary['mean_accuracy']:.4f}",
    )
    ax.axhline(
        prompt_summary["mean_accuracy"] + prompt_summary["std_accuracy"],
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"+1 std={prompt_summary['mean_accuracy'] + prompt_summary['std_accuracy']:.4f}",
    )
    ax.axhline(
        prompt_summary["mean_accuracy"] - prompt_summary["std_accuracy"],
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"-1 std={prompt_summary['mean_accuracy'] - prompt_summary['std_accuracy']:.4f}",
    )
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset_name} prompt sensitivity\nPSS = {prompt_summary['pss']:.4f}")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
def write_summary_tables(summary_rows: list[dict[str, Any]]) -> None:
    output_dir = ensure_dir(OUTPUTS_ROOT / "tables")
    dataframe = pd.DataFrame(summary_rows)
    csv_path = output_dir / "zeroshot_summary.csv"
    md_path = output_dir / "zeroshot_summary.md"
    dataframe.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(dataframe), encoding="utf-8")
    print(f"Saved summary CSV : {csv_path}")
    print(f"Saved summary MD  : {md_path}")


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    columns = list(dataframe.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]

    for row in dataframe.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows) + "\n"


def print_metrics_summary(
    bundle: DatasetBundle,
    default_prompt: str,
    metrics: dict[str, float],
    prompt_summary: dict[str, Any],
    confusion_pairs: list[dict[str, Any]],
    per_class_stats: list[dict[str, Any]],
) -> None:
    hardest = per_class_stats[0]
    easiest = per_class_stats[-1]

    print(f"Default prompt      : {default_prompt}")
    print(f"Accuracy / R@1      : {metrics['accuracy']:.4f}")
    print(f"Macro-F1            : {metrics['macro_f1']:.4f}")
    print(f"Balanced accuracy   : {metrics['balanced_accuracy']:.4f}")
    print(f"Top-5 accuracy      : {metrics['top_5_accuracy']:.4f}")
    print(f"MRR                 : {metrics['mrr']:.4f}")
    print(f"ECE                 : {metrics['ece']:.4f}")
    print(f"PSS                 : {prompt_summary['pss']:.4f}")
    print(f"Best prompt         : {prompt_summary['best_prompt']} ({prompt_summary['best_prompt_accuracy']:.4f})")
    print(f"Worst prompt        : {prompt_summary['worst_prompt']} ({prompt_summary['worst_prompt_accuracy']:.4f})")
    print(f"Hardest class       : {hardest['class_name']} ({hardest['accuracy']:.4f})")
    print(f"Easiest class       : {easiest['class_name']} ({easiest['accuracy']:.4f})")
    if confusion_pairs:
        top_pair = confusion_pairs[0]
        print(
            "Top confused pair   : "
            f"{top_pair['true_label']} -> {top_pair['predicted_label']} ({top_pair['count']} times)"
        )
    if bundle.metadata:
        print(f"Dataset metadata    : {bundle.metadata}")


if __name__ == "__main__":
    main()
