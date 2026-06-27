# Title Generation and Optimization

This directory contains research on iteratively optimizing YouTube video titles using LLMs and numeric inference models.

## Notebooks

- `0.Title_generator.ipynb`: Performs iterative optimization (10 rounds) of video titles for selected channels. It uses a trained OLS model to provide feedback to Gemini 3.1 Flash lite.
- `1.Optimization_analysis.ipynb`: Analyzes the results of the title generation process, testing hypotheses about starting performance, semantic distance, and optimization trajectories.

## Exported Results

### Title Optimization Results
**Path:** `/content/drive/MyDrive/numeric_inference_outputs/title_optimization_results_v2.json`
**Type:** `OptimizationResult[]`

```typescript
type OptimizationResult = {
    "channel": string,
    "video_id": string,
    "original_title": string,
    "original_score": number, // log-views
    "best_optimized_title": string,
    "best_optimized_score": number,
    "original_dist_to_description": number,
    "best_dist_to_description": number,
    "improvement": number,
    "improvement_pct": number,
    "history": {
        "iteration": number,
        "titles": {
            "text": string,
            "score": number
        }[]
    }[]
}
```

## Methodology

The optimization process follows these steps:
1.  **Baseline**: Calculate the predicted performance of the original title using the per-channel OLS model.
2.  **Iterative Prompting**: Prompt the LLM to generate 20 new titles.
3.  **Feedback Loop**: Score each new title using the OLS model.
4.  **Refinement**: Provide the LLM with the best and worst performing titles from previous rounds to guide the next iteration.
5.  **Selection**: After 10 rounds, select the title with the highest predicted performance.
