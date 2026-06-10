# Semantic Topology and Failure Modes in LLM-Based Text Classification of Creator Content

## Abstract

This paper studies a 24-class text-classification task in which short creator-content titles are assigned to their source channels or shows. The system under evaluation uses Gemini-generated category descriptions as class definitions and then prompts a Gemini classifier to label held-out titles. The evaluation is deliberately structured around semantic ambiguity rather than only aggregate accuracy: the cleaning notebook removes direct lexical hints, the pipeline notebook produces cleaned train/test splits and model outputs, and the analysis notebook tests whether classification errors follow the semantic geometry of the dataset. The main Gemini 3.1 Flash Lite run achieved 42.37% global accuracy on 236 held-out examples across 24 categories, with macro F1 of 0.42 and weighted F1 of 0.41. More important than the absolute score, the analysis shows that errors are not random. Categories farther from the semantic center perform better (Pearson r = 0.4412, p = 0.0309), semantically closer categories are confused more often (r = -0.2381, p < 0.0001), and categories whose generated descriptions better preserve raw inter-category distances are more accurate (r = 0.5587, p = 0.0045). Semantic dispersion within a category is negatively associated with recall (r = -0.3178), although this relationship is not statistically significant at conventional thresholds (p = 0.1302). These findings suggest that, for non-verifiable creator-content classification, evaluation should measure not only whether labels are correct but also whether errors respect the underlying semantic topology of the domain.

## 1. Introduction

Text classification is often evaluated as if each class were cleanly separable from every other class. That assumption is weak for creator-content datasets. Podcasts, YouTube channels, and thought-leadership shows frequently discuss overlapping topics such as artificial intelligence, startups, investing, geopolitics, self-improvement, business operations, and macroeconomics. A single title may plausibly resemble multiple channels even when only one source label is considered correct.

This project evaluates that problem in the context of HELENA, an LLM evaluation effort for non-verifiable domains. The target task is not factual question answering, where outputs can often be checked against a reference fact. Instead, the task is source-category classification: given a title, classify it into one of 24 creator channels or shows. This makes the domain a useful testbed for non-verifiable evaluation because many errors are matters of semantic overlap rather than obvious factual failure.

The central research question is:

> When an LLM misclassifies creator-content titles, do its errors reflect the semantic structure of the dataset?

The analysis notebook operationalizes this question through four hypotheses:

1. **Semantic projection:** category centroids and category-description embeddings should reveal clusters, overlaps, and description distortions.
2. **Centrality and difficulty:** semantically central categories should be harder to classify than semantically distinctive categories.
3. **Error closeness:** categories that are close in embedding space should be more frequently confused.
4. **Semantic volume:** internally diffuse categories should be harder to classify because they lack a tight semantic core.

The results support the broad claim that classification performance is shaped by semantic topology. The strongest finding is that description quality matters: categories whose generated descriptions preserve their relative semantic distances to other categories show substantially higher accuracy.

## 2. Dataset and Task

The dataset consists of short text titles grouped by creator channel or show. Before classification, the raw dataset contained 27 categories and 1,194 titles. A Gemini-assisted cleaning pass preserved 24 categories, skipped 3 categories judged unsuitable for unbiased evaluation, removed 6 individual titles, and modified 179 titles to remove lexical hints while preserving semantic meaning. The resulting cleaned dataset contained 1,188 titles.

The final categories used in the classification pipeline were:

- 20VC with Harry Stebbings
- ARK Invest
- Alex Hormozi
- All-In Podcast
- Anthony Pompliano
- Asianometry
- Bg2 Pod
- Bloomberg Originals
- Dan Martell
- Garry Tan
- Joe Lonsdale
- Lenny's Podcast
- My First Million
- Network State Podcast
- Patrick Boyle
- Principles by Ray Dalio
- Real Vision Presents
- Sequoia Capital
- The Prof G Pod – Scott Galloway
- This Week in Startups
- Tim Ferriss
- Tony Robbins
- Y Combinator
- a16z

Each category was split independently into 80% training and 20% testing subsets with a fixed random seed. The held-out test set contained 236 examples. Most categories contributed 10 test examples, while a few had 8 or 9 after cleaning and splitting.

## 3. Methodology

### 3.1 Data cleaning

The cleaning notebook uses Gemini 3.1 Flash Lite to identify direct category-name hints and other biasing artifacts. For each category, the cleaner asks whether the category should be used or skipped and then classifies each title as one of three statuses:

- **keep:** the title can be used as written;
- **remove:** the title contains a strong hint or unsuitable artifact that cannot be safely edited out;
- **edit:** the title can be rewritten to remove the hint while preserving meaning.

This step matters because creator-title datasets frequently leak labels through proper names, show names, host names, or recurring branded segments. The goal is not to make classification impossible, but to ensure that classification depends on semantic resemblance rather than simple string matching.

### 3.2 Description generation and classification

The classification pipeline uses the cleaned dataset as follows:

1. Load the cleaned JSON dataset from Google Drive.
2. Split each category into an 80% train subset and a 20% test subset using `random_state=42`.
3. Generate one concise natural-language description per category from that category's training examples using Gemini 3.1 Flash Lite.
4. Shuffle all held-out test examples with `random.seed(42)`.
5. Classify the shuffled held-out titles in batches of 20 by prompting the model with all category names and generated descriptions.
6. Cache train/test splits, generated descriptions, raw classification outputs, and embeddings for reproducibility and quota control.

The classification prompt asks the model to return only a JSON list of category names in the same order as the batch. This design makes the generated descriptions the primary interface between training data and the classifier: the model does not receive the training titles during classification, only the induced category descriptions.

The pipeline also contains an ablation design in which Gemini 3.5 Flash generates all category descriptions in a single request. However, the analysis notebook did not find ablation results in the loaded cache and therefore the paper reports only the completed Gemini 3.1 Flash Lite findings.

### 3.3 Embedding-based semantic analysis

The analysis notebook uses `all-MiniLM-L6-v2` sentence embeddings to represent titles and category descriptions. For each category, the notebook computes a centroid by averaging embeddings of that category's held-out test titles. It also embeds the generated category description.

The notebook then performs several analyses:

- **Projection analysis:** embeddings are reduced with PCA and visualized with t-SNE to compare text centroids and description embeddings.
- **Centrality analysis:** a category's centrality is measured as its average cosine distance from all other category centroids. Higher average distance means the category is more semantically distinctive.
- **Error-distance analysis:** the pairwise centroid-distance matrix is compared with the confusion matrix to test whether nearby categories are confused more often.
- **Dispersion analysis:** each category's semantic volume is measured as the mean cosine distance between its individual test-title embeddings and its centroid.
- **Distance-preservation analysis:** pairwise distances among raw text centroids are correlated with pairwise distances among generated descriptions. This measures how well the description layer preserves the structure of the underlying title space.

Pearson correlations are used to quantify relationships between semantic measures and performance metrics.

## 4. Results

### 4.1 Overall classification performance

The main Gemini 3.1 Flash Lite run achieved the following aggregate performance on 236 held-out examples:

| Metric | Value |
|---|---:|
| Accuracy | 0.4237 |
| Macro precision | 0.44 |
| Macro recall | 0.43 |
| Macro F1 | 0.42 |
| Weighted precision | 0.44 |
| Weighted recall | 0.42 |
| Weighted F1 | 0.41 |
| Errors | 136 / 236 |

Performance varied substantially by category. The highest-recall categories included Anthony Pompliano (0.90), Bg2 Pod (0.88), Asianometry (0.70), This Week in Startups (0.70), and Principles by Ray Dalio (0.67). The lowest-recall categories included Garry Tan (0.00), Joe Lonsdale (0.00), Y Combinator (0.10), 20VC with Harry Stebbings (0.20), and Sequoia Capital (0.20).

This spread indicates that the task is not uniformly difficult. Some categories have distinctive topical or stylistic signatures, while others inhabit heavily shared regions of the creator-content space.

### 4.2 Semantic projection and description distortion

The projection analysis compares two representations for each category:

1. the centroid of that category's actual held-out titles, and
2. the embedding of the generated category description used by the classifier.

The analysis interprets short centroid-to-description displacement as good description alignment and long displacement as semantic distortion. Distortion matters because the classifier is guided by the generated descriptions, not by direct access to the full training distribution. If a description captures only a narrow or shifted version of the category, then titles from the true category distribution may appear closer to a competing label.

Qualitatively, the analysis reports clusters corresponding to broad thematic domains such as finance-heavy content, startup and venture content, AI and technology content, and personal-development/business-operator content. These clusters help explain why mistakes often appear plausible rather than arbitrary.

### 4.3 Centrality and difficulty

The centrality hypothesis predicts that categories near the semantic center of the dataset should be harder to classify because they share vocabulary with many other categories. The analysis computes each category's average distance to all other category centroids and correlates that distance with F1-score.

The result supports the hypothesis:

| Relationship | Pearson r | p-value | Interpretation |
|---|---:|---:|---|
| Average centroid distance vs. F1-score | 0.4412 | 0.0309 | More semantically distinctive categories perform better. |

The positive correlation means that categories farther from the semantic center tend to have higher F1. In practical terms, a narrowly identifiable channel is easier to classify than a generalist channel whose titles resemble many neighboring domains.

### 4.4 Error closeness and semantic distance

The error-closeness hypothesis predicts that semantically close categories should be confused more often. The notebook flattens the off-diagonal pairwise centroid-distance matrix and compares it with off-diagonal confusion counts.

The result again supports the hypothesis:

| Relationship | Pearson r | p-value | Interpretation |
|---|---:|---:|---|
| Semantic distance vs. confusion count | -0.2381 | < 0.0001 | Closer categories are confused more often. |

The correlation is negative because smaller distance corresponds to larger confusion count. This finding is important because it shows that model errors follow the underlying topology of the category space. The classifier is not merely guessing; it is often choosing a semantically adjacent category.

### 4.5 Semantic volume and within-category dispersion

The semantic-volume hypothesis predicts that categories with more internally dispersed titles should be harder to classify. Dispersion is measured as the average cosine distance from each title embedding to its category centroid, and performance is measured as per-category recall.

The observed relationship is directionally consistent with the hypothesis but not statistically significant:

| Relationship | Pearson r | p-value | Interpretation |
|---|---:|---:|---|
| Category dispersion vs. recall | -0.3178 | 0.1302 | More diffuse categories tend to be less accurate, but evidence is inconclusive. |

This suggests that semantic volume may matter, but the current 24-category sample is too small or too noisy to establish the effect confidently. Another possibility is that dispersion interacts with description quality: a broad category can still perform well if its description explicitly captures multiple subtopics, while a compact category can perform poorly if its description is misleading.

### 4.6 Semantic distance preservation

The strongest analysis concerns distance preservation. The notebook compares pairwise distances among raw category centroids with pairwise distances among generated category descriptions. Across all category pairs, the semantic distance preservation score is:

| Metric | Value |
|---|---:|
| Raw-centroid / description-distance correlation | 0.5555 |

This means that the generated descriptions moderately preserve the geometry of the original title space. The notebook then calculates a per-category preservation score and correlates it with category accuracy:

| Relationship | Pearson r | p-value | Interpretation |
|---|---:|---:|---|
| Per-category preservation vs. accuracy | 0.5587 | 0.0045 | Categories whose descriptions better preserve semantic relations are more accurate. |

This result is the paper's most actionable finding. It suggests that improving category descriptions is not merely a matter of making them more detailed or fluent. Good descriptions should preserve the relative semantic position of the category among its competitors. A description that sounds plausible in isolation can still harm classification if it moves the category closer to the wrong neighbors or farther from its own title distribution.

## 5. Error Analysis

The analysis notebook reports 136 errors among 236 held-out examples. The errors reveal several recurring patterns.

### 5.1 AI, startup, and venture overlap

Several categories discuss AI infrastructure, startups, venture capital, founders, and product strategy. For example, 20VC, Sequoia Capital, Y Combinator, a16z, Lenny's Podcast, This Week in Startups, Bg2 Pod, and Garry Tan all contain titles that can plausibly belong to the same startup/AI ecosystem. This creates dense neighborhoods where source identity is hard to infer from title semantics alone.

Examples include Y Combinator titles predicted as Lenny's Podcast, ARK Invest, 20VC, Garry Tan, or This Week in Startups, and Sequoia Capital titles predicted as 20VC, Lenny's Podcast, ARK Invest, Y Combinator, or Garry Tan. These are not obviously irrational predictions; they reflect overlap in the subject matter of venture-backed technology content.

### 5.2 Finance, macroeconomics, and crypto overlap

Anthony Pompliano, Real Vision Presents, Patrick Boyle, ARK Invest, Bloomberg Originals, and The Prof G Pod all touch markets, macroeconomics, crypto, business news, and geopolitics. The model frequently confuses titles in this region, especially when a title lacks a channel-specific stylistic marker.

Examples include Real Vision Presents titles predicted as Anthony Pompliano, Bloomberg Originals, or Patrick Boyle, and Patrick Boyle titles predicted as All-In Podcast, Anthony Pompliano, The Prof G Pod, or Bloomberg Originals. Again, the mistakes often track reasonable topical alternatives.

### 5.3 Business coaching and self-improvement overlap

Alex Hormozi, Dan Martell, Tony Robbins, Tim Ferriss, and My First Million form another semantically overlapping region around business growth, personal decisions, self-improvement, entrepreneurship, and wealth creation. Tony Robbins titles were often predicted as Alex Hormozi, while Alex Hormozi and Dan Martell titles also confused with business-building categories.

This suggests that broad motivational or business-advice language is insufficiently source-specific unless descriptions capture differences in framing, audience, and recurring rhetorical style.

### 5.4 Generalist categories as error attractors

Some categories appear to function as attractors for broad topical content. For instance, 20VC with Harry Stebbings received predictions for AI, founder, venture, and enterprise-startup titles from several other categories. Such classes may have high false-positive pressure because their descriptions cover a broad, central region of the dataset.

This observation aligns with the centrality result: the more semantically central a category is, the harder it is to use as a precise decision boundary.

## 6. Discussion

The results point toward a topology-aware view of LLM classification. In a highly overlapping domain, accuracy alone can obscure whether the classifier is failing in meaningful or meaningless ways. A wrong label that is semantically adjacent to the correct label should be interpreted differently from a wrong label in a distant domain.

Three implications follow.

First, evaluation should include **semantic error diagnostics**. Confusion matrices are useful, but they become more informative when paired with embedding distances. If most errors occur between neighboring categories, the model may be sensitive to the right concepts but underpowered for fine-grained source attribution.

Second, category descriptions should be evaluated as **geometric objects**, not only as natural-language summaries. A generated description can sound accurate while still distorting the category's relative position. The significant correlation between preservation and accuracy indicates that description generation is a major bottleneck.

Third, benchmark design should distinguish between **intrinsic ambiguity** and **model failure**. Some labels may be difficult because their content is genuinely non-distinctive. For example, a title about AI startups may be semantically plausible for multiple venture and technology channels. Penalizing all such errors equally may overstate model failure in a non-verifiable domain.

## 7. Limitations

This study has several limitations.

1. **Small number of categories:** The correlation analyses operate over 24 classes, which limits statistical power for category-level relationships.
2. **Single main classifier configuration:** The reported final results come from the Gemini 3.1 Flash Lite run. Although the pipeline includes a Gemini 3.5 Flash ablation design, the analysis cache did not contain ablation results.
3. **Embedding-model dependence:** Semantic distances are measured with `all-MiniLM-L6-v2`. Different embedding models may produce different distance structures.
4. **Title-only inputs:** The classifier sees only short titles. Additional metadata, transcripts, thumbnails, host identities, or publication context might improve source attribution but would change the nature of the task.
5. **LLM-assisted cleaning:** The cleaning step uses an LLM to remove hints and judge suitability. This is practical but introduces another model-dependent preprocessing layer.
6. **Source label as ground truth:** The source channel is treated as the correct label, even when a title's topic is semantically closer to another channel's usual content.

## 8. Future Work

Future experiments should test whether topology-aware interventions improve classification. Promising directions include:

- generating multi-faceted descriptions for high-dispersion categories;
- adding negative contrasts, such as "this category is unlike...", for commonly confused pairs;
- optimizing descriptions directly for distance preservation;
- weighting errors by semantic distance to distinguish near misses from severe errors;
- comparing multiple embedding models for topology robustness;
- running the Gemini 3.5 ablation and additional classifier models under the same split;
- using pairwise or hierarchical classification for dense topical neighborhoods;
- measuring whether removing or merging highly central categories improves evaluation reliability.

## 9. Conclusion

The study finds that LLM-based text classification of creator-content titles is governed by semantic topology. The main classifier achieved 42.37% accuracy, but the deeper result is that errors are structured: distinctive categories perform better, nearby categories are confused more often, and categories with better description-distance preservation are substantially more accurate.

For non-verifiable domains, these findings argue for evaluation methods that go beyond exact-match accuracy. A robust evaluation should ask whether the model's mistakes are semantically coherent, whether class descriptions preserve the geometry of the underlying data, and whether apparent errors reflect genuine ambiguity in the label space. In this dataset, the answer is largely yes: the classifier struggles, but it struggles in ways that reveal the structure of the domain.
