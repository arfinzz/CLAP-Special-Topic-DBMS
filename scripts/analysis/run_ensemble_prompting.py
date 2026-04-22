"""
Ensemble Prompting for Zero-Shot CLAP Evaluation
=================================================
New idea: instead of selecting the single best prompt template, average the
L2-normalised text embeddings across ALL prompt variants for each class before
computing cosine similarity with the audio embeddings.

This is analogous to prompt ensembling in CLIP (Radford et al., 2021) but has
never been applied to or reported in the CLAP paper (Elizalde et al., 2023).

Hypothesis: averaging over diverse phrasings of the same class label produces
a more robust centroid in the joint audio-text embedding space, reducing the
variance introduced by any single prompt choice and improving accuracy over
even the best individual prompt.

Usage
-----
    python scripts/analysis/run_ensemble_prompting.py --datasets esc50 --device cuda
    python scripts/analysis/run_ensemble_prompting.py --datasets all --device cuda --skip-missing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Re-use all loaders and helpers from the main runner
from analysis.run_zeroshot_metrics import (
    DATASET_CHOICES,
    OUTPUTS_ROOT,
    PROMPT_FAMILIES,
    DatasetBundle,
    build_model,
    compute_metrics,
    ensure_dir,
    load_dataset_bundle,
    load_or_compute_audio_embeddings,
    load_or_compute_text_embeddings,
    resolve_datasets,
    resolve_device,
    slugify,
)
from repro_utils import build_run_manifest, write_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Core new idea: ensemble text embeddings
# ─────────────────────────────────────────────────────────────────────────────

def compute_ensemble_text_embeddings(
    model: Any,
    class_names: list[str],
    prompt_templates: list[str],
    cache_dir: Path,
    force_recompute: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    For each class, compute text embeddings under every prompt template,
    L2-normalise each one, then average and re-normalise.  The result is a
    single (num_classes, embed_dim) array that represents the ensemble centroid
    for each class.

    Also returns the list of per-prompt L2-normalised embeddings so the caller
    can pass them straight into compute_entropy_weighted_ensemble without
    recomputing.
    """
    all_embeddings: list[np.ndarray] = []

    for template in prompt_templates:
        embeddings = load_or_compute_text_embeddings(
            model=model,
            class_names=class_names,
            prompt_template=template,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
        )
        # L2-normalise before averaging so no single prompt dominates by scale
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        all_embeddings.append(embeddings / norms)

    # Stack → (num_prompts, num_classes, embed_dim), mean over prompts
    stacked = np.stack(all_embeddings, axis=0)          # (P, C, D)
    ensemble = stacked.mean(axis=0)                     # (C, D)

    # Re-normalise the ensemble centroid
    norms = np.linalg.norm(ensemble, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    ensemble = ensemble / norms

    # Cache it
    cache_path = cache_dir / "text_embeddings__ensemble.npy"
    np.save(cache_path, ensemble)
    return ensemble, all_embeddings


def compute_entropy_weighted_ensemble(
    audio_embeddings: np.ndarray,
    normalized_text_embeddings: list[np.ndarray],
    class_names: list[str],
    cache_dir: Path,
) -> tuple[np.ndarray, list[float]]:
    """
    Entropy-weighted ensemble: weight each prompt by the inverse of the mean
    Shannon entropy of its softmax score distribution across all audio samples.

    A prompt that produces sharp (confident) predictions gets a higher weight;
    one that spreads probability mass evenly (high entropy, low information)
    gets a lower weight.  This is fully unsupervised — no labels are used —
    so it remains valid in a zero-shot setting.

    Returns the re-normalised weighted centroid and the per-prompt weights.
    """
    eps = 1e-12
    weights: list[float] = []

    for text_emb in normalized_text_embeddings:
        # Cosine similarity scores → softmax probabilities
        logits = audio_embeddings @ text_emb.T          # (N, C)
        probs  = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        # Mean Shannon entropy over all audio samples (nats)
        entropy = -np.sum(probs * np.log(probs + eps), axis=1).mean()
        weights.append(float(1.0 / (entropy + eps)))

    # Normalise weights to sum to 1
    w = np.array(weights)
    w /= w.sum()

    # Weighted average of L2-normalised embeddings, then re-normalise
    stacked  = np.stack(normalized_text_embeddings, axis=0)   # (P, C, D)
    ensemble = (stacked * w[:, None, None]).sum(axis=0)        # (C, D)
    norms    = np.linalg.norm(ensemble, axis=1, keepdims=True)
    norms    = np.where(norms == 0, 1.0, norms)
    ensemble = ensemble / norms

    cache_path = cache_dir / "text_embeddings__ensemble_weighted.npy"
    np.save(cache_path, ensemble)
    return ensemble.astype(np.float32), w.tolist()


def evaluate_with_embeddings(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: np.ndarray,
    device: str,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Run cosine-similarity classification and return full metrics dict."""
    audio_t = torch.as_tensor(audio_embeddings, device=device)
    text_t  = torch.as_tensor(text_embeddings,  device=device)
    logits   = audio_t @ text_t.T
    probs    = torch.softmax(logits, dim=-1).cpu().numpy()
    rankings = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
    metrics, _ = compute_metrics(probs=probs, rankings=rankings,
                                 labels=labels, n_bins=n_bins)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_ensemble_comparison_figure(
    comparison: dict[str, Any],
    dataset_name: str,
    output_path: Path,
) -> None:
    """
    Bar chart: individual prompts | uniform ensemble | weighted ensemble.
    Both ensemble bars are annotated with their delta vs the best single prompt.
    """
    ensure_dir(output_path.parent)

    labels: list[str] = []
    accuracies: list[float] = []
    colors: list[str] = []

    for item in comparison["individual_prompts"]:
        short = item["prompt"].replace("{label}", "[X]")
        labels.append(short)
        accuracies.append(item["accuracy"])
        colors.append("#5b9bd5")

    labels.append("ENSEMBLE\n(uniform)")
    accuracies.append(comparison["ensemble_accuracy"])
    colors.append("#e05c2e")

    labels.append("ENSEMBLE\n(weighted)")
    accuracies.append(comparison["weighted_ensemble_accuracy"])
    colors.append("#2ecc71")

    best_acc  = comparison["best_prompt_accuracy"]
    worst_acc = comparison["worst_prompt_accuracy"]
    ens_acc   = comparison["ensemble_accuracy"]
    wens_acc  = comparison["weighted_ensemble_accuracy"]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", alpha=0.88)

    ax.axhline(best_acc,  color="green", linestyle="--", linewidth=1.5,
               label=f"Best single prompt ({best_acc:.4f})")
    ax.axhline(worst_acc, color="red",   linestyle="--", linewidth=1.5,
               label=f"Worst single prompt ({worst_acc:.4f})")

    for bar, acc, color in [(bars[-2], ens_acc, "#e05c2e"),
                             (bars[-1], wens_acc, "#2ecc71")]:
        delta = acc - best_acc
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.004,
            f"{delta:+.4f} vs best",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=color,
        )

    ax.set_ylim(max(0, min(accuracies) - 0.05), min(1.0, max(accuracies) + 0.06))
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"{dataset_name} — Ensemble vs individual prompts\n"
        f"Uniform: {ens_acc:.4f} ({ens_acc - best_acc:+.4f})  |  "
        f"Weighted: {wens_acc:.4f} ({wens_acc - best_acc:+.4f})  |  "
        f"Best single: {best_acc:.4f}"
    )
    ax.legend(loc="lower right", fontsize=9)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved ensemble figure: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ensemble prompting evaluation for CLAP zero-shot classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", default=["esc50"],
                   help="Datasets to evaluate. Use 'all' for all four.")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--amodel", type=str, default=None)
    p.add_argument("--enable-fusion", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--audio-batch-size", type=int, default=64)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--force-recompute-audio", action="store_true")
    p.add_argument("--force-recompute-text", action="store_true")
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--fsdd-split", choices=("test", "train", "all"), default="all")
    p.add_argument("--run-tag", type=str, default="ensemble-prompting")
    p.add_argument("--checkpoint-label", type=str, default="auto-default")
    p.add_argument("--notes", type=str, default="")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")
    ensure_dir(OUTPUTS_ROOT)

    requested = resolve_datasets(args.datasets)
    device    = resolve_device(args.device)

    print("=" * 72)
    print("CLAP Ensemble Prompting Evaluation  [NEW IDEA]")
    print("=" * 72)
    print(f"Datasets : {', '.join(requested)}")
    print(f"Device   : {device}")
    print()

    model = build_model(args, device)
    summary_rows: list[dict[str, Any]] = []
    evaluated_datasets: list[str] = []
    skipped_datasets: list[dict[str, str]] = []

    for dataset_key in requested:
        print("=" * 72)
        print(f"Dataset: {dataset_key}")
        print("=" * 72)

        try:
            bundle = load_dataset_bundle(dataset_key, args)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"Skipping {dataset_key}: {exc}\n")
                skipped_datasets.append({"dataset": dataset_key, "reason": str(exc)})
                continue
            raise

        prompt_templates = PROMPT_FAMILIES[bundle.prompt_family]
        embedding_dir    = ensure_dir(OUTPUTS_ROOT / "embeddings" / dataset_key)
        figure_dir       = ensure_dir(OUTPUTS_ROOT / "figures"    / dataset_key)

        # ── Audio embeddings (reuse cache from main runner if available) ──────
        audio_embeddings = load_or_compute_audio_embeddings(
            model=model, bundle=bundle, cache_dir=embedding_dir,
            batch_size=args.audio_batch_size,
            force_recompute=args.force_recompute_audio,
        )

        # ── Individual prompt accuracies ──────────────────────────────────────
        individual_results: list[dict[str, Any]] = []
        for template in prompt_templates:
            text_emb = load_or_compute_text_embeddings(
                model=model, class_names=bundle.class_names,
                prompt_template=template, cache_dir=embedding_dir,
                force_recompute=args.force_recompute_text,
            )
            m = evaluate_with_embeddings(audio_embeddings, text_emb,
                                         bundle.labels, device, args.n_bins)
            individual_results.append({"prompt": template, "accuracy": m["accuracy"]})
            print(f"  [{template.replace('{label}','[X]'):50s}]  acc={m['accuracy']:.4f}")

        # ── Ensemble text embeddings ──────────────────────────────────────────
        print(f"\n  Computing ensemble embeddings (averaging {len(prompt_templates)} prompts)...")
        ensemble_text_emb, normed_embs = compute_ensemble_text_embeddings(
            model=model, class_names=bundle.class_names,
            prompt_templates=prompt_templates, cache_dir=embedding_dir,
            force_recompute=args.force_recompute_text,
        )
        ensemble_metrics = evaluate_with_embeddings(
            audio_embeddings, ensemble_text_emb,
            bundle.labels, device, args.n_bins,
        )

        # ── Entropy-weighted ensemble ─────────────────────────────────────────
        weighted_text_emb, prompt_weights = compute_entropy_weighted_ensemble(
            audio_embeddings=audio_embeddings,
            normalized_text_embeddings=normed_embs,
            class_names=bundle.class_names,
            cache_dir=embedding_dir,
        )
        weighted_metrics = evaluate_with_embeddings(
            audio_embeddings, weighted_text_emb,
            bundle.labels, device, args.n_bins,
        )

        # ── Collate comparison ────────────────────────────────────────────────
        accs         = [r["accuracy"] for r in individual_results]
        best_acc     = max(accs)
        worst_acc    = min(accs)
        best_prompt  = individual_results[accs.index(best_acc)]["prompt"]
        ens_acc      = ensemble_metrics["accuracy"]
        wens_acc     = weighted_metrics["accuracy"]
        delta_best   = ens_acc  - best_acc
        wdelta_best  = wens_acc - best_acc
        delta_worst  = ens_acc  - worst_acc

        # Attach per-prompt weights to individual results for the JSON
        for res, w in zip(individual_results, prompt_weights):
            res["entropy_weight"] = round(w, 6)

        comparison = {
            "dataset": bundle.display_name,
            "num_prompts": len(prompt_templates),
            "individual_prompts": individual_results,
            "best_prompt": best_prompt,
            "best_prompt_accuracy": best_acc,
            "worst_prompt_accuracy": worst_acc,
            "mean_individual_accuracy": float(np.mean(accs)),
            "std_individual_accuracy": float(np.std(accs)),
            "ensemble_accuracy": ens_acc,
            "ensemble_vs_best": delta_best,
            "ensemble_vs_worst": delta_worst,
            "ensemble_metrics": ensemble_metrics,
            "weighted_ensemble_accuracy": wens_acc,
            "weighted_ensemble_vs_best": wdelta_best,
            "weighted_ensemble_metrics": weighted_metrics,
        }

        # ── Print summary ─────────────────────────────────────────────────────
        print(f"\n  ── Results: {bundle.display_name} ──")
        print(f"  Worst single prompt        : {worst_acc:.4f}")
        print(f"  Best  single prompt        : {best_acc:.4f}  ({best_prompt})")
        print(f"  ENSEMBLE (uniform)         : {ens_acc:.4f}  ({delta_best:+.4f} vs best)")
        print(f"  ENSEMBLE (entropy-weighted): {wens_acc:.4f}  ({wdelta_best:+.4f} vs best)")
        print(f"  Prompt weights             : { {t: round(w,3) for t,w in zip(prompt_templates, prompt_weights)} }")
        print(f"  Weighted Macro-F1          : {weighted_metrics['macro_f1']:.4f}")
        print(f"  Weighted Top-5 acc         : {weighted_metrics['top_5_accuracy']:.4f}")
        print(f"  Weighted MRR               : {weighted_metrics['mrr']:.4f}")

        # ── Save figure and JSON ──────────────────────────────────────────────
        fig_path = figure_dir / "ensemble_comparison.png"
        save_ensemble_comparison_figure(comparison, bundle.display_name, fig_path)

        metrics_path = OUTPUTS_ROOT / "metrics" / f"{dataset_key}_ensemble.json"
        write_json(metrics_path, comparison)
        print(f"  Saved JSON : {metrics_path}")
        evaluated_datasets.append(dataset_key)

        summary_rows.append({
            "dataset": bundle.display_name,
            "worst_prompt_acc": worst_acc,
            "best_prompt_acc": best_acc,
            "mean_individual_acc": comparison["mean_individual_accuracy"],
            "ensemble_acc": ens_acc,
            "delta_vs_best": delta_best,
            "weighted_ensemble_acc": wens_acc,
            "wdelta_vs_best": wdelta_best,
            "delta_vs_worst": delta_worst,
            "ensemble_macro_f1": ensemble_metrics["macro_f1"],
            "weighted_macro_f1": weighted_metrics["macro_f1"],
            "ensemble_top5": ensemble_metrics["top_5_accuracy"],
            "weighted_top5": weighted_metrics["top_5_accuracy"],
            "ensemble_mrr": ensemble_metrics["mrr"],
            "weighted_mrr": weighted_metrics["mrr"],
        })
        print()

    # ── Cross-dataset summary table ───────────────────────────────────────────
    if not summary_rows:
        print("No datasets evaluated.")
        return

    print("=" * 72)
    print("ENSEMBLE PROMPTING — CROSS-DATASET SUMMARY")
    print("=" * 72)
    header = (f"{'Dataset':<22} {'Worst':>7} {'Best':>7} "
              f"{'Uniform':>9} {'Δ':>7} {'Weighted':>9} {'Δ':>7}")
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['dataset']:<22} "
            f"{row['worst_prompt_acc']:>7.4f} "
            f"{row['best_prompt_acc']:>7.4f} "
            f"{row['ensemble_acc']:>9.4f} "
            f"{row['delta_vs_best']:>+7.4f} "
            f"{row['weighted_ensemble_acc']:>9.4f} "
            f"{row['wdelta_vs_best']:>+7.4f}"
        )

    # Write summary tables (CSV + Markdown)
    df = pd.DataFrame(summary_rows)
    out_dir = ensure_dir(OUTPUTS_ROOT / "tables")
    df.to_csv(out_dir / "ensemble_summary.csv", index=False)

    md_lines = [
        "| Dataset | Worst | Best | Uniform Ens. | Δ | Weighted Ens. | Δ |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['dataset']} "
            f"| {row['worst_prompt_acc']:.4f} "
            f"| {row['best_prompt_acc']:.4f} "
            f"| {row['ensemble_acc']:.4f} "
            f"| {row['delta_vs_best']:+.4f} "
            f"| {row['weighted_ensemble_acc']:.4f} "
            f"| {row['wdelta_vs_best']:+.4f} |"
        )
    (out_dir / "ensemble_summary.md").write_text("\n".join(md_lines) + "\n")

    manifest_path = OUTPUTS_ROOT / "manifests" / f"{slugify(args.run_tag)}.json"
    manifest = build_run_manifest(
        repo_root=REPO_ROOT,
        run_kind="ensemble_prompting",
        cli_args=vars(args),
        requested_datasets=requested,
        evaluated_datasets=evaluated_datasets,
        skipped_datasets=skipped_datasets,
        output_files={
            "summary_csv": str(out_dir / "ensemble_summary.csv"),
            "summary_md": str(out_dir / "ensemble_summary.md"),
            "metrics_dir": str(OUTPUTS_ROOT / "metrics"),
            "figures_dir": str(OUTPUTS_ROOT / "figures"),
        },
        extra={
            "checkpoint_label": args.checkpoint_label,
            "notes": args.notes,
        },
    )
    write_json(manifest_path, manifest)

    print(f"\nSaved ensemble summary: {out_dir / 'ensemble_summary.csv'}")
    print(f"Saved ensemble summary: {out_dir / 'ensemble_summary.md'}")
    print(f"Saved run manifest: {manifest_path}")
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
