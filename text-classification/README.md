# Text Classification Description Evaluation

This directory contains a reusable notebook for evaluating whether LLM-generated category descriptions are accurate for text classification and whether that accuracy correlates with semantic-distance preservation across categories.

## Notebook

- [`text_description_accuracy_distance_eval.ipynb`](text_description_accuracy_distance_eval.ipynb): end-to-end experiment notebook.

The notebook implements the requested workflow:

1. Accepts a Google Drive JSON path containing category labels and raw texts.
2. Splits every category into 80% training and 20% testing records.
3. Generates one category description per label using `gemini-3.1-flash-lite` and training data only.
4. Classifies randomized batches of 20 texts using the generated descriptions.
5. Reports global and category-level accuracy, macro/weighted F1, confusion matrices, tables, and charts.
6. Computes semantic-distance preservation by embedding training-text category centroids and generated descriptions, then comparing cosine-distance geometry.
7. Gives an absolute `YES`/`NO` answer on whether accuracy is correlated with distance preservation, backed by Spearman/Pearson tests, bootstrap confidence intervals, tables, and charts.
8. Optionally runs an ablation with `gemini-3.5-flash`, generating all category descriptions in one request.

## Expected source dataset types

The preferred input JSON is a `DatasetInput` object:

```python
from typing import Any, NotRequired, TypedDict

class CategoryInput(TypedDict):
    category: str
    texts: list[str]
    metadata: NotRequired[dict[str, Any]]

class DatasetInput(TypedDict):
    categories: list[CategoryInput]
    dataset_name: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]
```

Example:

```json
{
  "dataset_name": "support_intents_v1",
  "metadata": {"owner": "dataset-building"},
  "categories": [
    {
      "category": "billing",
      "texts": [
        "How can I update my payment method?",
        "I was charged twice this month."
      ],
      "metadata": {"source": "tickets"}
    },
    {
      "category": "technical_support",
      "texts": [
        "The app crashes after I log in.",
        "I cannot reset my password."
      ]
    }
  ]
}
```

A shorthand mapping is also accepted for quick experiments:

```json
{
  "billing": ["How can I update my payment method?", "I was charged twice this month."],
  "technical_support": ["The app crashes after I log in.", "I cannot reset my password."]
}
```

Validation rules:

- At least two categories are required.
- Every category name must be non-empty and unique.
- Every category must include at least two non-empty texts so the notebook can keep at least one training and one testing example.
- Texts are normalized to strings and stripped of leading/trailing whitespace.

## Generated artifact types

The notebook writes a new experiment folder under `text-classification/experiments/<EXPERIMENT_ID>/`. These artifacts are intentionally structured for future dataset-building notebooks.

```python
class DescriptionRecord(TypedDict):
    category: str
    description: str
    model: str
    prompt_hash: str
    generated_at_utc: str
    raw_response: dict[str, Any]

class PredictionRecord(TypedDict):
    text_id: str
    split: Literal["train", "test"]
    true_category: str
    predicted_category: str
    confidence: float | None
    rationale: str
    model: str
    batch_id: str
```

Common outputs:

- `raw/dataset_<hash>.json`: normalized source dataset snapshot.
- `splits/splits.parquet`: per-text train/test split with stable `text_id` values.
- `descriptions/*.json`: cached LLM description responses.
- `predictions/*.json`: cached batch-level LLM classification responses.
- `predictions/<split>_predictions.parquet`: flat prediction table.
- `embeddings/*.json`: cached Gemini embedding responses.
- `metrics/*`: CSV summaries for accuracy, confusion matrix, distance preservation, and correlation analysis.
- `figures/*`: saved charts for confusion matrix, category accuracy, distance matrices, and the accuracy-vs-preservation scatter plot.
- `manifest.json`: experiment configuration, model names, parameters, output locations, and dataset hash.

## Cache and quota policy

The notebook is designed to preserve quotas and previous work:

- `REUSE_CACHE=True` reuses existing descriptions, predictions, embeddings, and split files when present.
- `REUSE_CACHE=False` does **not** delete or overwrite prior artifacts. If a cache file already exists, the notebook raises an error and instructs you to choose a new `EXPERIMENT_ID`.
- `write_json_once` writes cache files atomically and refuses to overwrite existing files.
- To recreate outputs, create a new experiment id rather than deleting an old experiment.

## Semantic-distance preservation metric

For each category, the notebook embeds training texts and generated descriptions. It computes:

1. A centroid embedding for each category's training texts.
2. A description embedding for each category description.
3. A text-centroid cosine-distance matrix.
4. A description cosine-distance matrix.
5. Global preservation: Spearman correlation between the upper triangles of both matrices.
6. Category preservation: Spearman correlation between each category's row in the text-distance matrix and its row in the description-distance matrix.

The main analysis joins category preservation to category accuracy/recall. The notebook reports `YES` only if the category-level Spearman correlation is positive, statistically significant at `p < 0.05`, and its bootstrap 95% confidence interval excludes zero. Otherwise it reports `NO`.

## Environment requirements

Install the notebook dependencies with the first notebook cell:

```bash
pip install google-genai pandas numpy scikit-learn matplotlib seaborn scipy tqdm pydantic pyarrow
```

Set credentials before calling Gemini:

```bash
export GOOGLE_API_KEY="..."
```

In Google Colab, set `MOUNT_GOOGLE_DRIVE=True` and point `GOOGLE_DRIVE_JSON_PATH` to a mounted Drive JSON path such as:

```python
GOOGLE_DRIVE_JSON_PATH = "/content/drive/MyDrive/datasets/category_texts.json"
```
