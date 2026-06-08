# Text Classification Evaluation

This directory contains Python notebooks for evaluating text classification accuracy using Gemini models and semantic embeddings.

## Workflow

The evaluation is split into two notebooks:

1. **`classification_pipeline.ipynb`**: Handles the classification pipeline.
   - Loads data from JSON.
   - Splits data into 80% training and 20% testing per category.
   - Generates category descriptions using Gemini-3.1 Flash lite.
   - Classifies testing data in randomized batches.
   - Performs an ablation study using Gemini 3.5 Flash (single request).
   - Saves all raw outputs (descriptions and results) to Google Drive.

2. **`classification_analysis.ipynb`**: Performs hypothesis-driven analysis of the results.
   - Loads raw outputs from the pipeline.
   - **Semantic Projection**: Visualizes category centroids and description embeddings in 2D using t-SNE.
   - **Error Analysis**: Identifies and prints misclassified texts per category.
   - **Centrality vs Performance**: Analyzes if "central" categories (semantically closer to others) are harder to classify.
   - **Semantic Distance vs Confusion**: Correlates the closeness of categories with their confusion rate.
   - **Semantic Volume Analysis**: Evaluates how semantic dispersion of texts within a category impacts accuracy.
   - **Correlation Analysis**: Measures the relationship between semantic distance preservation and accuracy.

## Directory Structure

- `classification_pipeline.ipynb`: The pipeline notebook.
- `classification_analysis.ipynb`: The analysis notebook.
- `outputs/`: Directory for storing intermediate LLM responses and results.

## Reusability and Types

To enable reusability for future dataset building notebooks, the following types and structures are used:

### Input Data Format (JSON)
Path: `/content/drive/MyDrive/Graphiko/exports/base_data/latest/channel_titles_cleaned.json`

```typescript
interface Dataset {
  [categoryName: string]: string[]; // Array of texts belonging to the category
}
```

### Cached Descriptions (JSON)
```typescript
interface DescriptionsCache {
  [categoryName: string]: string; // Generated description for the category
}
```

### Classification Results (JSON)
```typescript
interface ClassificationResult {
  text: string;
  label: string; // True category
  prediction: string; // Predicted category
}
type ClassificationResults = ClassificationResult[];
```

## Setup

1. Open the notebooks in Google Colab.
2. Ensure you have access to the dataset.
3. Set up your Google Gemini API Key in the Colab Secrets panel.
4. Run `classification_pipeline.ipynb` first, then `classification_analysis.ipynb`.
