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
    3. Reduces dimensionality to 20 dimensions using PCA.
    4. Trains per-channel linear regression models.
    5. Evaluates predictions and analyzes dimension importance.

## Google Colab Usage
The notebooks are designed to be run in Google Colab:
1. Upload the notebook to Colab.
2. Ensure your Google Drive is mounted.
3. Place `channels_structured.json` in the expected path: `/content/drive/MyDrive/Graphiko/exports/base_data/latest/`.
4. Ensure you have the necessary libraries installed (handled in the notebook's first cell).
