# Numeric Inference

This directory contains research on predicting the popularity (views) of YouTube videos based on the semantic content of their titles.

## Purpose
The main goal is to evaluate how much information about a video's potential reach is captured by its title's embedding. We use a linear regression model per channel to predict logarithmic views from low-dimensional (PCA) embeddings of video titles.

## Data Format
The input data `channels_structured.json` follows this interface:

```typescript
type channels = {
    "channel_id": string,
    "channel_name": string,
    "videos": {
        "title": string,
        "id": string,
        "views": integer
    }[]
}[]
```

## Notebooks
- `numeric_inference.ipynb`: This notebook performs the following:
    1. Embeds video titles using `sentence-transformers`.
    2. Splits data into 80% training and 20% testing per channel.
    3. Reduces dimensionality to 15 dimensions using PCA.
    4. Trains per-channel Ordinary Least Squares (OLS) models.
    5. Evaluates predictions and analyzes dimension importance.
    6. Identifies the 10 most significant channels and exports an evaluation dataset.
- `llm-pipeline.ipynb`: This notebook uses Gemini 3.1 Flash lite to generate qualitative descriptions:
    1. Analyzes performance drivers for the top/bottom 25 videos per channel.
    2. Defines semantic meanings for the 15 PCA dimensions using the MECE framework.
    3. Generates channel-specific performance explanations based on significant dimensions.
- `feature-evaluation.ipynb`: This notebook evaluates Gemini 3.1 Flash lite's ability to label dimension contributions and predict views:
    1. Identifies significant dimensions and calculates thresholds for labels (e.g., "extremely positively", "neutral").
    2. Prompts Gemini to evaluate contributions and predict logarithmic views in batches of 10.
    3. Analyzes correlation between label accuracy and prediction error.
    4. Compares performance across various segments (Tails vs Center, Above vs Below Average).
    5. Benchmarks qualitative LLM predictions against OLS numeric predictions.
- `guided-inference.ipynb`: This notebook provides Gemini 3.1 Flash lite with qualitative "hints" to improve numeric view prediction:
    1. Maps PCA embedding values to a 7-point qualitative scale (-3 to +3) based on the training distribution.
    2. Constructs a rich prompt containing global success drivers, dimension definitions, and per-video labels for the 5 most significant dimensions.
    3. Instructs Gemini to predict logarithmic views based on these hints and channel benchmarks.
    4. Evaluates predictions against actual views and OLS benchmarks using MAE and R2.

## Exported Evaluation Dataset
The notebook exports a dataset of the top 10 most significant channels (based on F-statistic p-value) to `top_significant_channels_eval.json`.

### Schema
```typescript
type EvaluationDataset = {
    "channel_id": string,
    "channel_name": string,
    "model": {
        "intercept": number,
        "coefficients": number[], // 15 PCA dimensions
        "p_values": number[]      // [intercept_p, dim1_p, ..., dim15_p]
    },
    "train_videos": VideoRecord[],
    "test_videos": VideoRecord[]
}[]

type VideoRecord = {
    "video_id": string,
    "title": string,
    "actual_views": number,
    "predicted_views": number,
    "publishedAt": string,
    "reduced_embedding": number[] // 15 PCA dimensions
}
```

### Model Usage
The models were trained using OLS on `log1p(views)`. To perform inference:
1. Obtain the 15D PCA embedding of a title.
2. Calculate: `prediction = intercept + sum(coefficients[i] * embedding[i])`
3. Convert back to views: `views = exp(prediction) - 1`

## LLM Analysis Results
The `llm-pipeline.ipynb` notebook exports its findings to `llm_analysis_results.json`.

### Schema
```typescript
type LLMAnalysisResults = {
    "global_performance_descriptions": {
        [channel_id: string]: string // Explanation of what makes videos perform
    },
    "dimension_definitions": string[], // 15 MECE descriptions of PCA dimensions
    "channel_significant_dimension_analysis": {
        [channel_id: string]: string // Explanation of performance drivers via significant dims
    },
    "metadata": {
        "generated_at": string,
        "model": string,
        "input_file": string,
        "num_dimensions": number
    }
}
```

## Feature Evaluation Results
The `feature-evaluation.ipynb` notebook exports its results to `feature_evaluation_results.json`.

### Schema
```typescript
type FeatureEvaluationResults = {
    "channel_id": string,
    "channel_name": string,
    "results": {
        "title": string,
        "dimension_evaluations": {
            [dimension_index: string]: string // Label (e.g., "positive", "neutral")
        },
        "predicted_log_views": number,
        "video_id": string,
        "actual_log_views": number,
        "numeric_prediction": number,
        "reduced_embedding": number[]
    }[]
}[]
```

## Guided Inference Results
The `guided-inference.ipynb` notebook exports its results to `guided_inference_results.json`.

### Schema
```typescript
type GuidedInferenceResults = {
    "title": string,
    "predicted_log_views": number,
    "actual_log_views": number,
    "numeric_prediction": number,
    "video_id": string,
    "channel_id": string,
    "channel_name": string
}[]
```

## Google Colab Usage
The notebooks are designed to be run in Google Colab:
1. Upload the notebook to Colab.
2. Ensure your Google Drive is mounted.
3. Place `channels_structured.json` in the expected path: `/content/drive/MyDrive/Graphiko/exports/base_data/latest/`.
4. Ensure you have the necessary libraries installed (handled in the notebook's first cell).
