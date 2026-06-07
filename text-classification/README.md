# text-classification

**Part of [HELENA](../README.md) — LLM Evaluation for Non-Verifiable Domains**

This directory evaluates the accuracy of LLM-generated natural-language *category descriptions* as zero-shot text classifiers, and investigates whether classification accuracy correlates with the semantic separability of the underlying text data.

---

## Contents

```
text-classification/
├── README.md                        ← This file
├── text_classification.ipynb        ← Main notebook (run this)
└── outputs/                         ← Artefacts created at runtime
    └── <experiment_name>/
        ├── experiment_config.json   ← Snapshot of CONFIG for reproducibility
        ├── cache/                   ← Cached LLM responses (SHA-256 keyed JSON)
        ├── figures/                 ← PNG plots
        └── results/                 ← CSV tables and JSON descriptions
```

---

## Workflow

| Section | Title | Description |
| ------- | ----- | ----------- |
| §0 | Configuration | Set API key, experiment name, model IDs, hyperparameters |
| §1 | Data Loading | Load JSON from Google Drive; 80/20 stratified split |
| §2 | Description Generation | Iteratively refine category descriptions with **Gemini 3.1 Flash Lite** |
| §3 | Classification | Classify test texts in randomised batches of 20 |
| §4 | Performance Evaluation | Global + per-category accuracy, F1, confusion matrix |
| §5 | Semantic Distance Analysis | Embeddings, SPS metric, PCA visualisation |
| §6 | Correlation Analysis | Pearson/Spearman: accuracy vs semantic separation |
| §7 | Ablation Study *(optional)* | **Gemini 3.5 Flash** — all descriptions in one request |

---

## Input Format

The notebook expects a single JSON file on Google Drive with this structure:

```json
{
  "CategoryA": ["text sample 1", "text sample 2", "..."],
  "CategoryB": ["text sample 1", "text sample 2", "..."]
}
```

### Type Annotations

| Field | Python type | Description |
| ----- | ----------- | ----------- |
| Root object | `Dict[str, List[str]]` | Map of category label → list of raw texts |
| Category key | `str` | Unique, non-empty category label |
| Text list | `List[str]` | Raw text samples; each entry is a non-empty `str` |

**Minimum requirements:**
- At least **2 categories**
- At least **5 texts per category** (so the 80/20 split yields ≥1 test example)

---

## Output Files

All artefacts are written to `outputs/<experiment_name>/` (never overwritten).

### `cache/` — LLM Response Cache

| File pattern | Type | Description |
| ------------ | ---- | ----------- |
| `<tag>_<sha256>.json` | `Dict[str, Any]` | Cached LLM call; keys: `model`, `prompt`, `response`, `timestamp`, `tag` |

### `results/` — Tables and Descriptions

| File | Python type | Description |
| ---- | ----------- | ----------- |
| `data_split_summary.csv` | `pd.DataFrame` | Per-category train/test counts |
| `descriptions_main.json` | `Dict[str, str]` | `{category: description}` — main model |
| `descriptions_ablation.json` | `Dict[str, str]` | `{category: description}` — ablation model |
| `per_category_metrics_main.csv` | `pd.DataFrame` | Precision, Recall, F1, Accuracy per category |
| `per_category_metrics_ablation.csv` | `pd.DataFrame` | Same, for ablation model |
| `global_metrics_main.csv` | `pd.DataFrame` | Overall accuracy, macro P/R/F1 |
| `semantic_metrics_main.csv` | `pd.DataFrame` | `IntraSpread`, `SPS` per category |
| `correlation_statistics_main.csv` | `pd.DataFrame` | Pearson/Spearman r and p-values |
| `correlation_table_main.csv` | `pd.DataFrame` | Merged accuracy + SPS table for correlation |
| `comparison_main_vs_ablation.csv` | `pd.DataFrame` | Global metric delta between main and ablation |

### `figures/` — PNG Plots

| File | Description |
| ---- | ----------- |
| `confusion_matrix_main.png` | Normalised confusion matrix (main model) |
| `per_category_performance_main.png` | Per-category F1 and Accuracy bar charts |
| `semantic_distances_main.png` | Inter-centroid heatmap + SPS bar chart |
| `pca_embeddings_main.png` | PCA of training embeddings coloured by category |
| `correlation_scatter_main.png` | Scatter plots: Accuracy/F1 vs SPS |
| `confusion_matrix_comparison.png` | Side-by-side confusion matrices (ablation) |
| `comparison_main_vs_ablation.png` | Side-by-side bar chart: main vs ablation |

### `experiment_config.json`

Snapshot of the `CONFIG` dictionary (API key excluded) with an added `timestamp` field.  
Type: `Dict[str, Any]`

---

## Semantic Separation Metric: SPS

The **Separation Preservation Score (SPS)** is defined as:

$$\text{SPS} = \frac{\bar{d}_{\text{inter}}}{\bar{\sigma}_{\text{intra}}}$$

where:
- $\bar{d}_{\text{inter}}$ — mean cosine distance between all pairs of category centroids
- $\bar{\sigma}_{\text{intra}}$ — mean intra-category cosine spread (average distance from texts to their centroid)

A **per-category SPS** is also computed as the mean distance to all *other* centroids
divided by the category's own spread.  Higher SPS → better semantic separation.

---

## Reproducibility

```python
# Reuse cached LLM responses from a previous run:
CONFIG['recreate'] = False   # (default)

# Start a fresh experiment without deleting prior data:
CONFIG['recreate'] = True    # creates outputs/<experiment_name>_<timestamp>/
```

No experiment directory is ever deleted or overwritten.

---

## Models Used

| Role | Model ID | Notes |
| ---- | -------- | ----- |
| Description generation | `gemini-3.1-flash-lite-latest` | Gemini 3.1 Flash Lite |
| Text classification | `gemini-3.1-flash-lite-latest` | Same model by default |
| Text embeddings | `models/text-embedding-004` | Google embedding model |
| Ablation descriptions | `gemini-3.5-flash-latest` | Gemini 3.5 Flash |

Model IDs are fully configurable in `CONFIG`.

---

## Quick Start

```python
# 1. Set your API key
CONFIG['gemini_api_key'] = 'your-key-here'   # or: export GEMINI_API_KEY=...

# 2. Set your data path
CONFIG['google_drive_path'] = '/content/drive/MyDrive/data/categories.json'

# 3. Name your experiment
CONFIG['experiment_name'] = 'exp_001'

# 4. Run all cells  (Runtime → Run all in Colab)
```

---

## Dependencies

```
google-generativeai
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
tqdm
ipywidgets   (for tqdm notebook progress bars)
```

Install with:
```bash
pip install google-generativeai pandas numpy matplotlib seaborn scikit-learn scipy tqdm ipywidgets
```
