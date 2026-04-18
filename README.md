# CLAP — Paper Reimplementation & Extended Evaluation

<p align="center">
  <img src="https://raw.githubusercontent.com/LAION-AI/CLAP/main/assets/logo.PNG" width="55%"/>
</p>

Reimplementation and extension of:

> **CLAP: Learning Audio Concepts from Natural Language Supervision**
> Elizalde et al., ICASSP 2023 — [arXiv:2206.04769](https://arxiv.org/abs/2206.04769)

Using the improved LAION checkpoint from:

> **Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation**
> Wu et al., ICASSP 2023 — [arXiv:2211.06687](https://arxiv.org/abs/2211.06687)

---

## What this repo adds over the original paper

| Contribution | Details |
|---|---|
| Paper reproduction | ESC-50, UrbanSound8K, GTZAN zero-shot evaluation |
| New dataset | FSDD (Free Spoken Digit Dataset) — not evaluated in the paper |
| New metrics | Macro-F1, Balanced Accuracy, Top-5 Accuracy, MRR, ECE, PSS |
| **New idea: Ensemble Prompting** | Two strategies — uniform average and entropy-weighted average of L2-normalised text embeddings across all prompt templates — never done in the CLAP paper |

---

## Repository layout

```
CLAP/
├── src/laion_clap/               # CLAP model source
├── scripts/analysis/
│   ├── run_zeroshot_metrics.py   # main evaluation runner
│   └── run_ensemble_prompting.py # NEW IDEA: ensemble prompting (uniform + weighted)
├── metrics_analysis.py           # entry point for main runner
├── class_labels/                 # JSON label maps per dataset
├── data/datasets/
│   ├── esc50/ESC-50-master/
│   ├── fsdd/free-spoken-digit-dataset/
│   ├── gtzan/genres/
│   └── urbansound8k/UrbanSound8K/
└── outputs/
    ├── embeddings/   # cached .npy embeddings (reused across runs)
    ├── figures/      # all plots
    ├── metrics/      # per-dataset JSON reports
    └── tables/       # CSV and Markdown summary tables
```

---

## One-time fix for WSL / Windows file corruption

Run this before anything else if you cloned on Windows:

```bash
python -c "
import pathlib
null = b'\x00'
for f in pathlib.Path('src').rglob('*.py'):
    raw = f.read_bytes()
    if null in raw:
        f.write_bytes(raw.replace(null, b''))
        print('Fixed:', f)
print('Done')
"
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

python -c "
import pathlib
f = pathlib.Path('scripts/analysis/run_zeroshot_metrics.py')
content = f.read_text()
old = 'REPO_ROOT = Path(__file__).resolve().parents[2]\nSRC_ROOT = REPO_ROOT / \"src\"\nif str(SRC_ROOT) not in sys.path:\n    sys.path.insert(0, str(SRC_ROOT))'
new = 'REPO_ROOT = Path(__file__).resolve().parents[2]'
if old in content:
    f.write_text(content.replace(old, new))
    print('Fixed scripts/analysis/run_zeroshot_metrics.py')
"
```

---

## Known corrupt files (Windows extraction artefacts)

| Dataset | Corrupt file | Error | Fix applied |
|---|---|---|---|
| ESC-50 | `audio/5-147297-A-27.wav` | `Format not recognised` | Re-extracted from `ESC-50-master.zip` on Linux |
| ESC-50 | `audio/5-150409-A-42.wav` | `Format not recognised` | Re-extracted from `ESC-50-master.zip` on Linux |
| GTZAN | `genres/jazz/jazz.00054.wav` | `bad data offset` | File deleted (not recoverable); evaluation runs on 999/1000 files |

### Re-extraction commands (run once after cloning)

```bash
# ESC-50
rm -rf data/datasets/esc50/ESC-50-master
unzip data/archives/ESC-50-master.zip -d data/datasets/esc50/

# GTZAN
rm -rf data/datasets/gtzan/genres
mkdir -p data/datasets/gtzan
tar -xzf data/archives/genres.tar.gz -C data/datasets/gtzan/ 2>/dev/null
rm -f data/datasets/gtzan/genres/jazz/jazz.00054.wav

# UrbanSound8K
rm -rf data/datasets/urbansound8k/UrbanSound8K
mkdir -p data/datasets/urbansound8k
tar -xzf data/archives/UrbanSound8K.tar.gz -C data/datasets/urbansound8k/ 2>/dev/null
```

### Verify file counts after extraction

```bash
echo "ESC-50      :" && ls data/datasets/esc50/ESC-50-master/audio/*.wav | wc -l      # expect 2000
echo "GTZAN       :" && ls data/datasets/gtzan/genres/*/*.wav | wc -l                  # expect 999
echo "UrbanSound8K:" && ls data/datasets/urbansound8k/UrbanSound8K/audio/fold*/*.wav | wc -l  # expect 8732
echo "FSDD        :" && ls data/datasets/fsdd/free-spoken-digit-dataset/recordings/*.wav | wc -l  # expect 3000
```

> **Note on GTZAN:** `jazz.00054.wav` has unrecoverable header corruption inside the original archive. The file was removed; results are reported on 999/1000 samples.

---

## Environment setup

```bash
conda create -n clap python=3.10
conda activate clap

# PyTorch — CUDA 12.1 (adjust for your GPU)
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Verify
python -c "import laion_clap; import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## Dataset setup

### ESC-50 and GTZAN — already in repo (re-extract on Linux, see above)
```
data/datasets/esc50/ESC-50-master/audio/   (2000 files)
data/datasets/gtzan/genres/<genre>/        (999 files — jazz.00054.wav removed)
```

### UrbanSound8K — download required
Download from https://urbansounddataset.weebly.com/urbansound8k.html, extract to:
```
data/datasets/urbansound8k/UrbanSound8K/audio/fold1/ ... fold10/
data/datasets/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv
```

### FSDD — clone once (new dataset, not in original paper)
```bash
git clone --depth 1 \
    https://github.com/Jakobovski/free-spoken-digit-dataset.git \
    data/datasets/fsdd/free-spoken-digit-dataset
```

---

## Running everything

### Step 1 — Paper reproduction + new dataset + new metrics
```bash
python metrics_analysis.py \
    --datasets esc50 urbansound8k gtzan fsdd \
    --fsdd-split all \
    --device cuda \
    --audio-batch-size 64 \
    --skip-missing \
    2>&1 | tee outputs/eval_run.log
```

### Step 2 — New idea: Ensemble Prompting (uniform + entropy-weighted)
```bash
python scripts/analysis/run_ensemble_prompting.py \
    --datasets esc50 urbansound8k gtzan fsdd \
    --fsdd-split all \
    --device cuda \
    --skip-missing \
    2>&1 | tee outputs/ensemble_run.log
```

> Step 2 reuses cached embeddings from Step 1 and finishes in a few minutes.

### View results
```bash
cat outputs/tables/zeroshot_summary.md     # main results
cat outputs/tables/ensemble_summary.md     # ensemble comparison (uniform vs weighted)
find outputs/figures -name "*.png" | sort  # all figures
```

---

## The new idea: Ensemble Prompting

### Motivation
CLAP Table 3 studies prompt sensitivity by testing prompts individually and picking the best one. It never combines them. In the CLIP vision literature, averaging text embeddings across prompt templates (prompt ensembling) reliably improves zero-shot accuracy. This approach has never been applied to or reported in the CLAP paper.

### Strategy 1 — Uniform ensemble
For each class label:
1. Compute text embedding under every prompt template.
2. L2-normalise each embedding so no prompt dominates by scale.
3. Average the normalised embeddings across all templates.
4. Re-normalise the result to get the ensemble centroid.

### Strategy 2 — Entropy-weighted ensemble
Weight each prompt inversely by the mean Shannon entropy of its softmax score distribution across all audio samples. A prompt that produces confident, sharp predictions (low entropy) receives a higher weight; a weak prompt that spreads probability mass evenly (high entropy) receives a lower weight. No labels are used — the weighting is fully unsupervised and valid in a zero-shot setting.

```
weight_i  = 1 / mean_entropy( softmax( audio @ text_i.T ) )
ensemble  = renormalize( Σ weight_i · text_emb_i )
```

```
Standard:         audio ──cosine──► text("A recording of dog")
Uniform ensemble: audio ──cosine──► mean_norm(text(p1), ..., text(pN))
Weighted ensemble:audio ──cosine──► weighted_norm(text(p1), ..., text(pN))
```

### Prompt templates

**Sound domain** (ESC-50, UrbanSound8K)

| # | Template |
|---|---|
| 1 | `This is a sound of {label}.` |
| 2 | `The sound of {label}.` |
| 3 | `A recording of {label}.` |
| 4 | `An audio clip of {label}.` |
| 5 | `I can hear {label}.` |

**Music domain** (GTZAN) and **Speech domain** (FSDD) use separate domain-appropriate families.

### Results

| Dataset | Best single | Uniform Ens. | Δ | Weighted Ens. | Δ |
|---|---|---|---|---|---|
| ESC-50 | 0.9265 | 0.9270 | +0.0005 | 0.9270 | +0.0005 |
| UrbanSound8K | 0.8106 | 0.7989 | -0.0117 | 0.7989 | -0.0117 |
| GTZAN | 0.6767 | 0.6597 | -0.0170 | 0.6587 | -0.0180 |
| FSDD | 0.1057 | 0.0757 | -0.0300 | 0.0757 | -0.0300 |

### Analysis
Prompt ensembling, which reliably helps in CLIP (vision), **does not reliably transfer to CLAP (audio)**. The gain on ESC-50 is marginal (+0.05%), and on lower-accuracy datasets both strategies are dragged down by weaker prompts. The entropy-weighted approach assigns equal weight (0.2) to all prompts on every dataset, indicating that CLAP produces nearly identical softmax confidence distributions regardless of prompt phrasing — there is no entropy signal to exploit. This suggests audio-text alignment is more sensitive to prompt phrasing than vision-text alignment, making best-prompt selection more important than averaging.

---

## Metrics (beyond the original paper)

| Metric | Description | In paper? |
|---|---|---|
| Accuracy / R@1 | Top-1 zero-shot accuracy | ✅ |
| Macro-F1 | Unweighted F1 across all classes | ❌ New |
| Balanced Accuracy | Mean per-class recall | ❌ New |
| Top-5 Accuracy | Correct class in top-5 predictions | ❌ New |
| MRR | Mean Reciprocal Rank | ❌ New |
| ECE | Expected Calibration Error | ❌ New |
| PSS | Prompt Sensitivity Score (std/mean over prompt variants) | ❌ New |

---

## Results

### All datasets — full results

| Dataset | Samples | Classes | Accuracy | Macro-F1 | Bal. Acc | Top-5 Acc | MRR | ECE | PSS | Best prompt |
|---|---|---|---|---|---|---|---|---|---|---|
| ESC-50 | 2000 | 50 | **91.50%** | 91.20% | 91.50% | 99.35% | 0.9499 | 0.8833 | 0.0044 | `A recording of {label}.` (92.65%) |
| UrbanSound8K | 8732 | 10 | **77.47%** | 77.86% | 79.05% | 96.12% | 0.8582 | 0.6382 | 0.0153 | `I can hear {label}.` (81.06%) |
| GTZAN | 999 | 10 | **60.76%** | 58.80% | 60.79% | 96.60% | 0.7545 | 0.4846 | 0.0453 | `An audio clip of a {label} song.` (67.67%) |
| FSDD | 3000 | 10 | **10.53%** | 6.29% | 10.53% | 56.77% | 0.3133 | 0.0003 | 0.1261 | `Someone says the word {label}.` (10.57%) |

### ESC-50 — paper reproduction

| Metric | Paper (ZS) | This repo |
|---|---|---|
| Accuracy | 82.6% | **91.50%** |
| Macro-F1 | — | 91.20% |
| Balanced Accuracy | — | 91.50% |
| Top-5 Accuracy | — | 99.35% |
| MRR | — | 0.9499 |
| Hardest class | — | insects (27.50%) |
| Easiest class | — | hand saw (100%) |
| Top confused pair | — | insects → cow (19 times) |
| Best single prompt | `this is a sound of [label]` | `A recording of [label].` (92.65%) |
| Ensemble accuracy (uniform) | — | **92.70%** (+0.05% vs best single) |
| Ensemble accuracy (weighted) | — | **92.70%** (+0.05% vs best single) |

> Accuracy improvement over the paper (82.6% → 91.5%) is due to the newer LAION checkpoint trained on 630k pairs vs. the original 128k.

### New dataset: FSDD

| Property | Value |
|---|---|
| Task | Spoken digit recognition (digits 0–9) |
| Classes | 10 |
| Samples | 3000 |
| Speakers | george, jackson, lucas, nicolas, theo, yweweler |
| Domain | Speech |
| In original paper | ❌ Not evaluated |
| Accuracy | 10.53% |
| Hardest class | three (0%) |
| Easiest class | zero (66%) |
| Top confused pair | four → zero (245 times) |

> The low FSDD accuracy reflects that CLAP's general-purpose audio-text pretraining was not trained on speech-word recognition tasks and does not align well with individual spoken digit words.

---

## Pretrained checkpoints

From [HuggingFace — lukewys/laion_clap](https://huggingface.co/lukewys/laion_clap/tree/main):

| Checkpoint | Best for | ESC-50 ZS | `--amodel` flag |
|---|---|---|---|
| `630k-audioset-best.pt` *(default, auto-download)* | General audio <10s | ~91% | *(omit)* |
| `630k-audioset-fusion-best.pt` | Variable-length audio | ~91% | *(omit)* |
| `music_audioset_epoch_15_esc_90.14.pt` | Music | 90.14% | `HTSAT-base` |
| `music_speech_audioset_epoch_15_esc_89.98.pt` | Music + Speech | 89.98% | `HTSAT-base` |

---

## Citation

```bibtex
@inproceedings{elizalde2022clap,
  title     = {CLAP: Learning Audio Concepts from Natural Language Supervision},
  author    = {Elizalde, Benjamin and Deshmukh, Soham and Al Ismail, Mahmoud and Wang, Huaming},
  booktitle = {ICASSP},
  year      = {2023}
}
@inproceedings{laionclap2023,
  title     = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion
               and Keyword-to-Caption Augmentation},
  author    = {Wu*, Yusong and Chen*, Ke and Zhang*, Tianyu and Hui*, Yuchen and
               Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle = {ICASSP},
  year      = {2023}
}
```
