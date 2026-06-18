# Semantic Topology and Anisotropic Metrics: Improving Numeric Prediction via Qualitative Reasoning in Latent Spaces

## Abstract

Predicting quantitative outcomes from qualitative semantic data is a fundamental challenge in non-verifiable domains. This paper investigates the task of predicting YouTube video popularity (logarithmic views) based on video titles. We frame this problem through the lens of semantic topology, arguing that latent embedding spaces are "anisotropic"—where different dimensions have unequal impacts on the outcome. Our research demonstrates three key results: (1) guided generation, which provides the model with qualitative hints based on statistical significance, achieves the highest predictive accuracy; (2) the accuracy of qualitative "labeling" for latent dimensions correlates strongly with final numeric precision; and (3) while dimension-specific analysis captures the "shape" of statistical models most accurately, it introduces biases that only guided reasoning can mitigate. These findings suggest that the prompt functions as a reasoning bridge, allowing Large Language Models (LLMs) to navigate complex metric spaces by weighting dimensions according to their task-specific importance.

## 1. Introduction: Semantic Topology as a Foundation for Numeric Reasoning

Traditional evaluation metrics for high-dimensional embeddings, such as cosine similarity, often suffer from isotropy—treating all dimensions as equally significant. However, in task-specific evaluations, the semantic space is inherently anisotropic. Some dimensions may be highly predictive of success, while others may be irrelevant or even detrimental to the target outcome. Understanding this "topological pull" is essential for improving model performance in domains where ground-truth labels are scarce or subjective.

We also draw an analogy to "proactive interference" in the human brain—the phenomenon where existing memories interfere with the acquisition of new, similar information. In LLM inference, a model's broad pre-trained knowledge can interfere with its ability to grasp the specific nuances of a narrow distribution (e.g., a specific YouTube channel's performance drivers). By providing explicit guidance and weighting significant dimensions, we mitigate this interference, enabling more precise test-time reasoning.

## 2. Methodology

The research was conducted using a sequential pipeline across several specialized notebooks:

1.  **Data Preparation and Numeric Baseline**: Video titles were embedded using the `all-MiniLM-L6-v2` model and reduced to a 15-dimensional subspace using Principal Component Analysis (PCA). A per-channel Ordinary Least Squares (OLS) regression model was trained to predict logarithmic views, providing a statistical benchmark for dimension significance.
2.  **Qualitative Analysis**: Using the Gemini 3.1 Flash lite model, we generated MECE-compliant (Mutually Exclusive, Collectively Exhaustive) descriptions for each of the 15 PCA dimensions and analyzed global performance drivers for each channel.
3.  **Multi-Methodological Prediction**: We evaluated four distinct LLM-based prediction strategies:
    *   **Global LLM**: Direct prediction from title and channel context.
    *   **Dimension LLM**: Prediction based on an explicit analysis of all 15 PCA dimensions.
    *   **Feature Evaluation LLM**: A two-step process where the model first "labels" the contribution of each dimension (on a 7-point scale) before predicting views.
    *   **Guided Inference LLM**: The model is provided with qualitative "hints" (e.g., "This title is significantly positive in Dimension 3") for the 5 most significant dimensions.

## 3. Results

The evaluation across ten highly significant YouTube channels yielded three primary insights that clarify the relationship between qualitative reasoning and numeric accuracy.

| Model | MAE (Normalized) | R² | Correlation with OLS (r) |
| :--- | :--- | :--- | :--- |
| **Global LLM** | 1.012 | 0.085 | 0.832 |
| **Dimension LLM** | 1.058 | 0.075 | **0.912** |
| **Feature Evaluation** | 1.042 | 0.049 | 0.853 |
| **Guided Inference** | **0.959** | **0.191** | 0.902 |
| *Numeric (OLS)* | *0.924* | *0.264* | *1.000* |

### 3.1 Guided Generation Achieves the Highest Outcome

The primary result is that **Guided Inference** achieved the best performance among all LLM methodologies, with a normalized Mean Absolute Error (MAE) of 0.959 and an R² of 0.191. This demonstrates that numeric inference can be substantially improved by integrating insights from traditional data analysis. By weighting dimensions anisotropically—focusing the model’s attention on the most significant drivers—we allow the LLM to approximate the precision of a statistical model while maintaining its capacity for natural language reasoning.

### 3.2 Labeling Accuracy Predicts Prediction Accuracy

Our analysis revealed a strong correlation between the accuracy of qualitative dimension "labeling" (Mean Absolute Label Error, or MALE) and final prediction error. When the LLM correctly identified the latent qualitative contribution of a dimension (e.g., correctly labeling a dimension associated with "high-stakes storytelling" as "highly positive"), its numeric predictions were significantly more accurate. This is a crucial finding: it suggests that **we can evaluate the accuracy of a model's reasoning even in the absence of a ground-truth** by measuring the fidelity of its latent representations.

### 3.3 The Correlation-Bias Paradox in Dimension Prediction

An interesting secondary result was that the **Dimension LLM** achieved the highest correlation with the numeric OLS model (r = 0.912), even higher than the Guided Inference model. However, its overall error (MAE) was higher, and its R² was lower. This indicates that while analyzing all dimensions allows the LLM to capture the "shape" of the statistical model, the process introduces increased bias or variance in the absolute predictions. This poses a fundamental question: to what extent can LLMs improve upon simple linear models? By leveraging non-linear relationships, managing variance, or incorporating higher-dimensional context (relevant to each specific video), LLMs may eventually surpass simple OLS benchmarks.

## 4. Discussion

### 4.1 Improvement Strategies and Training

The performance gap between LLM predictions and the OLS baseline suggests several avenues for improvement. Beyond prompting, "test-time reasoning" can be formalized through self-distillation or on-policy distillation. By rewarding models that generate descriptions which preserve the underlying semantic topology, we can train models to be "geometrically faithful." This is particularly relevant for long-context generation, where maintaining a coherent relationship between many concepts is more important than simple fact retrieval.

### 4.2 Cross-Domain Applications

While this study focused on entertainment (YouTube), the principles of anisotropic metrics and guided inference apply to any domain where qualitative data drives quantitative results:
*   **Sales**: Predicting deal closure probability based on the semantic "trajectory" of communication.
*   **Marketing and Communication**: Estimating campaign impact or message effectiveness by analyzing the alignment of copy with proven success dimensions.
*   **Education and Personal Self-Development**: Identifying the specific semantic dimensions of a learning material or behavior habit that correlate with engagement and successful outcomes.

### 4.3 Comparison to Topological Projection

In the previous `text-classification` research, we focused on **Topological Projection**—measuring how well a category description preserved the distance between category centroids. In the numeric domain, we extend this to **Anisotropic Importance**. While classification is concerned with the *boundaries* between regions, numeric prediction is concerned with the *gradient* or *pull* of specific dimensions. A model that understands the topology must not only know where a category "is" but where it "leads."

## 5. Conclusion and Future Work

This paper has shown that numeric prediction in latent spaces is a topological challenge that can be addressed through anisotropic metrics and guided reasoning. The prompt acts as a reasoning framework that overcomes proactive interference and focuses the model on the most significant semantic signals.

Our next steps involve developing a metric based on **Trajectory Similarity**. Rather than evaluating static points in space, we will measure the similarity of reasoning paths—the sequence of latent inferences a model makes compared to the actual trajectories of successful data points. This moves the evaluation of AI from static state-matching to dynamic process-verification, providing a more robust foundation for reasoning in non-verifiable domains.
