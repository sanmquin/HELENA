# Zero-Shot Text Classification of Podcast Episode Titles Using Large Language Models: A Semantic Analysis

## Abstract

This paper investigates the capability of large language models (LLMs) to perform zero-shot text classification on a real-world dataset of podcast episode titles. Using the Gemini 3.1 Flash Lite model, we classify 236 episode titles across 24 distinct podcast channels. Our pipeline leverages LLM-generated natural language descriptions as the sole reference for classification, without any fine-tuning. We achieve a global accuracy of 42.4% and conduct a hypothesis-driven semantic analysis to identify the structural factors that govern classification difficulty. We find statistically significant evidence that (1) semantically central categories are harder to classify (r = 0.44, p = 0.031), (2) model errors follow the underlying semantic topology of the data (r = −0.24, p < 0.0001), and (3) the degree to which a category description preserves inter-category semantic distances is the strongest predictor of per-category accuracy (r = 0.56, p = 0.005). These findings provide actionable guidance for improving LLM-based classification systems beyond model selection.

---

## 1. Introduction

The classification of short, ambiguous text—such as social media posts, email subject lines, or video titles—poses a distinct challenge for text classification systems. Unlike long-form documents, short texts provide minimal context per sample, while category boundaries in real-world taxonomies are often subjective and overlapping.

Recent advances in LLMs offer a compelling alternative to supervised classification: by describing each category in natural language and prompting the model to assign labels, it becomes possible to build a multi-class classifier without labelled training data or model fine-tuning. This paradigm, commonly referred to as *zero-shot classification*, shifts the engineering effort from data collection to description quality.

This work applies this paradigm to a dataset of episode titles from 24 prominent financial, technology, and business podcast channels. The categories range from highly niche channels with a distinctive voice (e.g., *Asianometry*, *Principles by Ray Dalio*) to broadly thematic channels that cover overlapping topics (e.g., *All-In Podcast*, *20VC with Harry Stebbings*, *Garry Tan*). This heterogeneity makes the dataset an interesting testbed for understanding what drives classification success or failure.

Beyond measuring raw accuracy, this paper uses semantic embeddings to test four hypotheses about the structural underpinnings of classification performance:

1. **Semantic Projection**: The spatial relationship between a category's actual text distribution and its LLM-generated description reveals a "semantic distortion" that correlates with model errors.
2. **Centrality and Difficulty**: Categories semantically close to the global mean (i.e., using generic language) are harder to classify.
3. **Error Closeness**: Model errors are not random; semantically closer categories are confused more often.
4. **Semantic Volume**: Categories with higher intra-class dispersion are harder to classify accurately.

---

## 2. Dataset and Data Preparation

### 2.1 Dataset

The dataset consists of episode titles scraped from 27 YouTube-based podcast channels. Each channel covers topics in the intersection of business, technology, finance, and entrepreneurship. The raw dataset contains 1,194 titles distributed across these channels.

### 2.2 Data Cleaning

A key data quality concern was the presence of *classification hints*: episode titles that explicitly mention the channel name or creator's name, which would make classification trivially solvable and invalidate the experiment. To address this, we developed an LLM-assisted cleaning pipeline using Gemini 3.1 Flash Lite.

For each category, the model was prompted to:
1. Decide whether the category should be **used** or **skipped** based on inherent bias or unsuitability.
2. For each individual title, decide whether it should be **kept**, **removed** (if hints are irreparable), or **edited** (if hints can be removed while preserving meaning).

**Outcome**:

| Metric | Value |
|---|---|
| Categories in raw data | 27 |
| Categories preserved | 24 |
| Categories skipped | 3 |
| Total texts in raw data | 1,194 |
| Texts removed | 6 |
| Texts modified (hints removed) | 179 |
| Total texts in cleaned dataset | 1,188 |

The 179 modifications (15% of the dataset) highlight how pervasive category-name leakage is in podcast episode titles. An example of an edited title from *The Prof G Pod – Scott Galloway*:

> **Before**: "The Hidden Engine of China's AI Boom | China Decode"  
> **After**: "The Hidden Engine of China's AI Boom"

### 2.3 Train/Test Split

The cleaned dataset was split per category using an 80/20 stratified split (random seed 42), yielding approximately 948 training texts and 236 test texts (≈10 per category).

---

## 3. Methodology

### 3.1 Description Generation

For each of the 24 categories, a natural language description was generated by prompting Gemini 3.1 Flash Lite with the category name and its training texts:

> *"Provide a concise description of the following category based on these examples: {category}\nExamples:\n{training texts}"*

Descriptions were generated iteratively (one per API call) and cached to support reproducibility. This *iterative, per-category* approach is referred to as the **primary configuration** (Gemini 3.1, iterative).

An ablation study was designed with a **single-request** approach using Gemini 3.5 Flash: all training data is submitted in one prompt requesting descriptions for all categories simultaneously. This tests whether a more powerful model using a unified prompt yields better-calibrated descriptions.

### 3.2 Classification Pipeline

Test texts were shuffled randomly (seed 42) and classified in batches of 20 using the following prompt:

> *"Classify the following texts into one of these categories:\n{category: description}\n\nTexts:\n{batch}\n\nReturn only a JSON list of category names in order."*

The model returns a JSON array of predicted category names, one per input text. Results were cached incrementally after each batch to ensure fault tolerance against API quota limits.

### 3.3 Semantic Embedding and Analysis

For the analysis phase, all texts and category descriptions were encoded using the **all-MiniLM-L6-v2** sentence transformer model. Dimensionality was first reduced to 20 dimensions using PCA to capture dominant semantic variance while filtering high-dimensional noise. Further reduction to 2D was done via t-SNE (initialized with PCA) for visualization.

Pearson correlation coefficients were used to test all hypotheses, with p-values reported for statistical significance.

---

## 4. Results

### 4.1 Overall Performance

The primary classification run (Gemini 3.1 Flash Lite, iterative descriptions) achieved a global accuracy of **42.4%** on the 236-item test set.

| Metric | Value |
|---|---|
| Global Accuracy | 42.4% |
| Macro F1-Score | 0.42 |
| Weighted F1-Score | 0.41 |
| Total test samples | 236 |
| Total errors | 136 |

### 4.2 Per-Category Performance

Performance varied dramatically across categories, from 0% to 90% recall:

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| Principles by Ray Dalio | 1.00 | 0.67 | **0.80** |
| Anthony Pompliano | 0.56 | 0.90 | **0.69** |
| This Week in Startups | 0.64 | 0.70 | **0.67** |
| Asianometry | 0.70 | 0.70 | **0.70** |
| Bg2 Pod | 0.50 | 0.88 | **0.64** |
| Real Vision Presents | 0.83 | 0.50 | 0.62 |
| Network State Podcast | 0.67 | 0.40 | 0.50 |
| Tim Ferriss | 0.46 | 0.60 | 0.52 |
| ARK Invest | 0.46 | 0.60 | 0.52 |
| My First Million | 0.42 | 0.50 | 0.45 |
| Patrick Boyle | 0.38 | 0.50 | 0.43 |
| Tony Robbins | 0.60 | 0.30 | 0.40 |
| Alex Hormozi | 0.31 | 0.50 | 0.38 |
| The Prof G Pod – Scott Galloway | 0.38 | 0.30 | 0.33 |
| Dan Martell | 0.33 | 0.30 | 0.32 |
| All-In Podcast | 0.27 | 0.30 | 0.29 |
| Lenny's Podcast | 0.23 | 0.30 | 0.26 |
| a16z | 0.80 | 0.44 | 0.57 |
| Sequoia Capital | 0.40 | 0.20 | 0.27 |
| Bloomberg Originals | 0.31 | 0.40 | 0.35 |
| 20VC with Harry Stebbings | 0.12 | 0.20 | **0.15** |
| Y Combinator | 0.25 | 0.10 | **0.14** |
| Garry Tan | 0.00 | 0.00 | **0.00** |
| Joe Lonsdale | 0.00 | 0.00 | **0.00** |

The top-performing categories share a distinctive characteristic: they cover specific, identifiable topics. *Principles by Ray Dalio* titles follow a thematic pattern tied to Dalio's framework; *Asianometry* focuses on Asian industrial and technology history; *Anthony Pompliano* covers crypto and Bitcoin with a consistent lexicon. Conversely, the worst-performing categories (*Garry Tan*, *Joe Lonsdale*, *Y Combinator*, *20VC*) are general startup and venture capital commentary channels that share extensive thematic and lexical overlap with each other and with channels like *Sequoia Capital*, *a16z*, and *Bg2 Pod*.

### 4.3 Error Analysis

Inspection of misclassified texts reveals that errors are rarely random. They tend to fall into a small number of semantic clusters:

- **Startup/VC cluster confusion**: Titles from *Garry Tan*, *20VC with Harry Stebbings*, *Joe Lonsdale*, *Y Combinator*, *Sequoia Capital*, and *a16z* were frequently confused for one another. For example, a *Garry Tan* title ("The Truth and Lies About Driverless Cars") was predicted as *The Prof G Pod*, and a *20VC* title ("NVIDIA Predicts $1TRN in Revenue...") was predicted as *This Week in Startups*.
- **Business coaching cluster confusion**: *Alex Hormozi* and *Dan Martell* titles were regularly confused. Both channels produce motivational business content with nearly identical title structures (e.g., "If I Wanted to Create a Business That Runs Itself, Here's What I'd Do").
- **Finance/macro cluster confusion**: *ARK Invest*, *Bloomberg Originals*, *Patrick Boyle*, and *Real Vision Presents* titles showed mutual confusion, particularly when covering overlapping macro-economic topics.

---

## 5. Hypothesis Testing

### 5.1 Hypothesis 1: Semantic Projection

**Hypothesis**: The spatial relationship between a category's text centroid and its description embedding reveals a "semantic distortion." Large distortions indicate that the description does not well-represent the average content of the category, which should correlate with poor classification performance.

**Method**: Embeddings for all test texts were computed and reduced to 20D via PCA. Category centroids (mean of text embeddings) and description embeddings were projected to 2D using t-SNE. The Euclidean distance between a category's centroid and its description in the 20D space measures the distortion.

**Finding**: The t-SNE visualization confirms that categories with similar content cluster together in both the text-based and description-based projections. Broad thematic clusters (finance, tech journalism, entrepreneurship coaching) are visible. The "Combined Distortion Map" shows notable misalignments for consistently misclassified categories: their descriptions occupy a different semantic neighborhood than their actual texts, meaning the model's reference anchor is placed in the wrong region of the embedding space.

Color mapping—where a category's color is derived from the first three PCA components of its centroid—provides an interpretable proxy for semantic identity. Categories with similar colors (nearby semantic positions) are the ones most frequently confused by the classifier.

### 5.2 Hypothesis 2: Centrality and Difficulty

**Hypothesis**: Categories semantically close to the global mean of all texts (i.e., "central" categories) employ generic language and will exhibit lower classification performance.

**Method**: For each category, we computed the mean cosine distance from its centroid to every other category's centroid. This "average inter-class distance" serves as the centrality metric—lower values indicate more central (generic) categories. We then correlated this metric with the per-category F1-score.

**Finding**:

| Metric | Value |
|---|---|
| Pearson r | **0.4412** |
| p-value | **0.031** |

The correlation is positive and statistically significant (p < 0.05). This confirms the hypothesis: categories that are **farther** from the semantic center—meaning they occupy a more isolated, distinctive region of the embedding space—are classified more accurately. Conversely, central categories, which share vocabulary and themes with many other channels, are the hardest to distinguish.

This result explains why channels like *Garry Tan* (broad startup commentary) and *Y Combinator* (general startup advice) consistently fail, while *Asianometry* (specialized tech history) and *Principles by Ray Dalio* (idiosyncratic conceptual framework) succeed.

### 5.3 Hypothesis 3: Error Closeness

**Hypothesis**: Model errors follow the semantic topology of the data. Semantically closer categories are confused more often.

**Method**: We constructed the full inter-category distance matrix (cosine distance between category centroids) and the full confusion matrix (from classification results). Both matrices were flattened—excluding diagonal entries (self-classification)—and correlated using Pearson's r.

**Finding**:

| Metric | Value |
|---|---|
| Pearson r | **−0.2381** |
| p-value | **< 0.0001** |

The negative correlation is highly statistically significant. As semantic distance **decreases**, the confusion count **increases**. This confirms that the model's errors are structured: it confuses categories that are genuinely semantically similar, not arbitrary or random ones.

The practical implication is encouraging: the classifier is not making illogical jumps. When it mis-labels a *20VC* title as *This Week in Startups*, this reflects a genuine semantic similarity in the data, not a failure of the model's reasoning. Improving descriptions for these "confusable pairs" would be a targeted and effective intervention.

### 5.4 Hypothesis 4: Semantic Volume and Dispersion

**Hypothesis**: Categories with high intra-class semantic dispersion (texts spread out in embedding space) are harder to classify because a single description cannot adequately represent the full range of content.

**Method**: For each category, we computed the mean cosine distance from each test text to the category's centroid. This "dispersion" metric was then correlated with per-category recall.

**Finding**:

| Metric | Value |
|---|---|
| Pearson r | **−0.3178** |
| p-value | **0.130** |

The direction of the correlation supports the hypothesis—higher dispersion is associated with lower accuracy—but the result does not reach statistical significance (p = 0.13) with 24 data points. A larger study with more categories would be needed to confirm this effect. The trend is nonetheless consistent with the semantic projection analysis: high-dispersion categories benefit most from multi-faceted descriptions rather than a single summary sentence.

### 5.5 Semantic Distance Preservation

**Hypothesis**: If a set of descriptions accurately reflects the relative distances between categories, the classifier should be able to use them as a reliable map of the label space.

**Method**: We computed the inter-category distance matrix twice: once from raw text centroids and once from description embeddings. We then measured the Pearson correlation between the two sets of pairwise distances (upper triangular) as a "preservation score."

**Finding**: The global preservation score is **0.5555**, meaning the LLM-generated descriptions preserve approximately 55.6% of the variance in the true inter-category distance structure. This is a moderate level of fidelity—the descriptions capture broad-strokes thematic groupings but distort finer-grained distinctions.

To test whether per-category preservation quality predicts accuracy, we computed each category's individual preservation score (correlation between its row in the text-based and description-based distance matrices) and correlated these with per-category recall:

| Metric | Value |
|---|---|
| Pearson r | **0.5587** |
| p-value | **0.005** |

This is the **strongest and most statistically significant predictor** of classification performance found in this study. Categories whose descriptions correctly encode their relative position to all other categories—knowing not just "what they are" but "what they are *relative to others*"—are classified most accurately. This result has a direct implication: description generation should optimize for *relational fidelity*, not just individual accuracy.

---

## 6. Discussion

### 6.1 What Makes a Good Description?

Taken together, the four hypotheses paint a coherent picture. The key factors that determine whether a category will be classified accurately are:

1. **Distinctiveness**: A category that occupies a unique semantic region (high average distance from others) is inherently easier to classify, regardless of description quality.
2. **Relational Fidelity**: A description that correctly encodes the category's position *relative to all other categories* is the single most powerful driver of accuracy (r = 0.56, p = 0.005). This suggests that descriptions generated with awareness of all categories simultaneously would outperform those generated in isolation.
3. **Alignment**: The description must be semantically close to the actual text distribution it is meant to represent. A description that accurately characterizes a different facet of the category than the one reflected in the test data will fail.

### 6.2 Ablation Study

The ablation study—where Gemini 3.5 Flash generates all descriptions simultaneously in a single prompt—was designed to test whether a more capable model with global context produces better-preserved relational structure. Unfortunately, the classification results for this condition were not available in the cached outputs at the time of analysis, so a direct comparison could not be made. Based on the theoretical analysis above, we would predict that simultaneously generated descriptions should exhibit a higher preservation score and therefore higher accuracy, as the model would have access to contrast information between categories.

### 6.3 Limitations

- **Dataset size**: With only 24 categories and approximately 10 test samples per category, statistical power for per-category correlations is limited. The dispersion hypothesis (H4), which showed a plausible but insignificant trend, may achieve significance with a larger dataset.
- **Topic domain**: All channels cover a narrow slice of the content landscape (business, technology, finance). The findings may not generalize to more diverse multi-domain classification tasks.
- **Single model evaluation**: Only Gemini 3.1 Flash Lite was systematically evaluated. Different models, especially those with stronger instruction-following or reasoning capabilities, may show different patterns.
- **Batch classification**: Classifying texts in batches of 20 introduces contextual cross-contamination. A text's classification may be influenced by the other texts in the same batch.

---

## 7. Conclusion

This paper demonstrates that LLM-based zero-shot classification of short, ambiguous text is feasible but challenging. A global accuracy of 42.4% on 24-class podcast title classification with no task-specific training data is a meaningful result, given the high semantic overlap between categories.

The semantic analysis reveals that raw accuracy is largely a function of dataset structure, not just model capability. Channels that speak with a distinctive vocabulary are classified reliably; channels that cover general-interest startup and finance topics fail precisely because they are genuinely hard to distinguish—a fact that no model can fully overcome without better descriptions or additional context.

The most actionable finding is that **relational fidelity**—the degree to which a category's description correctly encodes its position relative to all other categories—is the strongest predictor of classification accuracy (r = 0.56, p = 0.005). This suggests a clear path to improvement: description generation should be a *comparative, globally-aware* process rather than an independent per-category summarization task. Generating all descriptions in a single prompt, explicitly asking the model to distinguish each category from its nearest semantic neighbors, or iteratively refining descriptions based on observed confusion patterns are all promising directions.

---

## References

1. Brown, T., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901.
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*, 3982–3992.
3. van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
4. Yin, W., Hay, J., & Roth, D. (2019). Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach. *Proceedings of EMNLP-IJCNLP 2019*, 3914–3923.
5. Google DeepMind. (2025). Gemini 3.1 Flash Lite and Gemini 3.5 Flash technical reports. Google LLC.
