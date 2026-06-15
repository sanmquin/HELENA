# HELENA
LLM Evaluation for Non-Verifiable Domains

## Evaluation Datasets
### Numeric Inference Evaluation Dataset
Located in `numeric-inference/`, the `numeric_inference.ipynb` notebook generates an evaluation dataset (`top_significant_channels_eval.json`) containing:
- Significant YouTube channels (top 10 by predictive p-value).
- Per-video metadata, PCA title embeddings, and predicted views.
- Per-channel linear models (OLS weights and p-values per dimension).

This dataset serves as ground truth for evaluating LLM capabilities in predicting video popularity and understanding semantic drivers of reach.
