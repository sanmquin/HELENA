# Semantic Topology as Metrics for Evaluation Quality in Non-Verifiable Domains

## Abstract

Large language models are increasingly deployed in domains where ground-truth labels are absent, contested, or inherently subjective. Evaluating model behavior in such settings demands metrics that go beyond exact-match accuracy. Semantic topology — the geometric structure induced by text embeddings — provides principled metrics for evaluation of text generation quality even when labels cannot be independently verified. We study a description-based text classification task in which LLM-generated category descriptions serve as the sole interface between training data and a classifier. We find that classification errors are not random: semantically distinctive categories perform better (Pearson r = 0.44, p = 0.03), semantically close categories are confused more often (r = −0.24, p < 0.0001), and high-volume categories may be harder to classify (r = -0.32, p = 0.13). More importantly, categories whose descriptions better preserve the geometry of the underlying data are substantially more accurate (r = 0.56, p = 0.004). The implications is that topological metrics alone can evaluate the quality of a description, making them extremely valuable during inference, and training of Large Language Models. Applications of topological metrics is explored in other domains such as summarization, explaining (inverse descriptions), long-context generation, and artificial reasoning. Additionally, other possible topological metrics are discussed, including anisotropic (task-relevant) metrics, and path similarity with applications for information retrieval, and probabilistic inference. Finally, methods for improvement are proposed during inference, post-training, and mid-training. Including guided-generation, on-policy distillation, dense-rewards based on algorithmic problem-solving, predictive world-models, and hierarchical attention.  

## 1. Introduction: Descriptions as Topological Maps

A category description is not merely a fluent paraphrase of examples. In a description-based classifier, it functions as a map from a high-dimensional instance distribution to a compressed linguistic object. The relevant question is therefore not only whether the description sounds correct in isolation, but whether it preserves the relationships that make the category distinguishable from neighboring categories.

Let each category \(C_i\) be represented by a finite point cloud \(X_i = \{x_{i1}, \ldots, x_{in_i}\}\) in an embedding space \(E \subset \mathbb{R}^d\), and let its generated natural-language description be embedded as \(s_i \in E\). The empirical centroid \(\mu_i = n_i^{-1}\sum_j x_{ij}\) summarizes the category's location, while the within-category dispersion summarizes its semantic volume. The full set of centroids \(\{\mu_i\}_{i=1}^k\) induces a finite metric space with pairwise distances \(d(\mu_i, \mu_j)\). The descriptions induce a second finite metric space with distances \(d(s_i, s_j)\). The key hypothesis is that classification quality depends on how faithfully the second metric space represents the first.

This is a topological claim before it is a predictive one. A good description should preserve neighborhoods: categories that are close in the original instance space should remain close in description space, and categories separated by semantic gaps should remain separated. A bad description introduces distortion. It may collapse distinct regions, exaggerate irrelevant differences, or move a category toward the wrong neighbors. In metric terms, description quality can be framed as low distortion between the centroid metric and the description metric, for example through the preservation of pairwise distances:

\[
\rho_i = \operatorname{corr}_{j \neq i}\left(d(\mu_i, \mu_j), d(s_i, s_j)\right).
\]

This paper uses text classification as a controlled setting for that hypothesis. The task is deliberately narrow: short text items must be assigned to one of 24 thematically overlapping categories using only generated category descriptions. Yet the implications are broader. Whenever natural language is used to summarize, describe, retrieve, or explain a distribution of objects, the description creates an induced topology. The quality of that induced topology can be measured even when individual labels are unavailable, subjective, or expensive to verify.

## 2. Results

The evaluation pipeline used 24 categories and 236 held-out items. Category-identifying lexical artifacts were removed, categories were split into training and testing subsets, one natural-language description was generated per category from training items, and held-out items were classified using only the category names and generated descriptions. Sentence embeddings from all-MiniLM-L6-v2 were computed for items and descriptions; category centroids, pairwise cosine distances, confusion counts, dispersion scores, and distance-preservation correlations were then derived from those embeddings.

| Finding | Topological quantity | Downstream quantity | Result | Interpretation |
|---|---|---|---:|---|
| Overall performance | Description-induced decision structure | Held-out classification | Accuracy = 0.4237; macro F1 = 0.42; weighted F1 = 0.41; errors = 136 / 236 | The classifier is informative but far from saturated, leaving substantial variation to explain geometrically. |
| Central categories are harder to quantify | Average centroid distance from other categories | Per-category F1 | r = 0.44; p = 0.03 | Categories farther from the semantic center perform better; categories in dense regions face higher ambiguity. |
| Close categories are more easily confused | Pairwise semantic distance between category centroids | Pairwise confusion count | r = -0.24; p < 0.0001 | Errors concentrate among neighbors rather than appearing uniformly across the category space. |
| Dispersed categories may be harder to classify | Mean distance from items to their category centroid | Per-category recall | r = -0.32; p = 0.13 | Higher semantic volume plausibly weakens the classification signal, though this result remains probabilistic in the present sample. |
| Description distortion predicts classification quality | Correlation between centroid distances and description distances | Per-category accuracy | r = 0.56; p = 0.004 | The main result: descriptions that preserve the domain's metric structure produce better classification outcomes. |
| Global metric preservation | Correlation between all centroid-pair distances and all description-pair distances | Structural validity of the description layer | r = 0.56 | The description space moderately preserves the topology of the original category space. |

### 2.1 The Main Result: Distortion Predicts Classification Quality

The central result is that distance preservation is strongly associated with classification accuracy. A category whose description preserves its relative distances to other categories is easier to classify; a category whose description distorts those distances is more likely to fail. This matters because it converts description quality from a subjective linguistic judgment into a measurable geometric property.

Mathematically, the description set can be treated as a map \(f: \mu_i \mapsto s_i\) between two finite metric spaces. The relevant failure mode is not local fluency but metric distortion: \(d(s_i, s_j)\) ceases to approximate \(d(\mu_i, \mu_j)\). If \(f\) moves a description toward an incorrect neighbor, then a classifier using descriptions as anchors receives a misleading decision geometry. The observed correlation between per-category preservation and accuracy (r = 0.56, p = 0.004) indicates that this distortion is not merely aesthetic; it predicts operational failure.

This result also explains why the metric is useful when labels are absent. Labels are needed to compute accuracy, but they are not needed to compute the distance-preservation score itself. If representative instances and generated descriptions are available, the induced topology can be evaluated directly. A practitioner can therefore identify descriptions that are likely to be poor before collecting labeled test cases, by asking which descriptions warp the point cloud they are supposed to represent.

### 2.2 Central Descriptions Are More Difficult to Quantify

The first complementary result concerns centrality. Categories near the interior of the semantic space have many close neighbors, and therefore smaller margins. Categories located farther from the average of other centroids are more distinctive and achieve better F1 scores (r = 0.44, p = 0.03).

Topologically, central categories occupy high-density regions of the induced metric space. Their neighborhoods overlap with many alternatives, so a small displacement in description space can change the nearest competing category. The practical implication is that central descriptions require greater precision: they must encode subtle relative differences rather than broad topical identity. A description of an isolated category can be somewhat coarse and still preserve separability; a description in the crowded center must preserve fine-grained boundaries.

### 2.3 Close Descriptions Are More Easily Confused

The second complementary result is that confusion follows proximity. Pairwise semantic distance between category centroids is negatively correlated with pairwise confusion count (r = -0.24, p < 0.0001). The negative sign is expected: smaller distances correspond to larger confusion counts.

This result gives the paper its cohesive error narrative. Misclassifications are not uniformly distributed noise. They are concentrated along short edges in the semantic graph induced by the categories. In nearest-neighbor terms, the model fails where the margin between competing descriptions is small; in topological terms, errors occur where neighborhoods overlap. This distinction is important for evaluation. A mistake between adjacent categories has a different meaning than a mistake across distant regions of the space, even if both count equally under accuracy.

### 2.4 Category Volume May Increase Difficulty

The third complementary result is suggestive rather than conclusive. Categories with greater within-category dispersion tend to have lower recall (r = -0.32, p = 0.13). This aligns with the hypothesis that high-volume categories are harder to describe because no single linguistic anchor can represent all regions of the point cloud equally well.

The geometric intuition is straightforward. A compact category can be approximated by a centroid and a concise description with relatively low information loss. A dispersed category behaves more like a union of subregions. Compressing it into one description may erase internal structure, increase overlap with neighbors, and reduce recall. The present sample does not establish this relationship at conventional significance levels, but the direction is theoretically coherent and motivates richer measures of category shape, such as local density, anisotropy, convex-hull approximations, or persistent homology over category point clouds.

## 3. Applications Beyond Labeled Classification

The bridge from this result to applications is not that labels magically become unnecessary for every evaluation question. Rather, the bridge is that many modern AI systems already create descriptions, summaries, explanations, and memory objects that induce a topology over the content they represent. When those induced spaces can be compared to the spaces of the underlying objects, description quality can be measured structurally, even before direct task labels are available.

### 3.1 Summarization and Memory in Large-Scale AI Systems

Summarization is increasingly used as memory infrastructure in coding assistants, chat interfaces, and long-running agentic systems. A summary decides what information is retained, compressed, and made retrievable. That process has an implicit categorization structure: different summaries emphasize different events, goals, entities, constraints, and unresolved tasks.

The metric proposed here can evaluate whether summaries preserve the topology of the source material. Source passages form a point cloud; summaries form a second point cloud. If summaries collapse distinct issues, exaggerate minor distinctions, or move important constraints away from their relevant context, the induced topological space becomes a poor memory. This gives summarization a structural quality metric: not merely whether a summary is fluent, but whether it preserves the neighborhood relationships needed for later retrieval and repair.

### 3.2 Text Generation, Explanation, and Multimodal Description

For text generation, the metric can be used in the opposite direction. Instead of compressing a point cloud into a description, generation often expands an initial semantic point into a coherent distribution of statements. A product description, a list of bullet points, a set of adjectives, an explanation, or an image-to-text translation should elaborate the source while remaining topologically faithful to it.

In this setting, the initial text or multimodal embedding acts as an anchor, and the generated description creates a local cloud around that anchor. A good generation expands the region without drifting into unrelated semantic neighborhoods. A poor generation may remain grammatical while introducing topological drift: the generated adjectives, examples, or explanations imply a different object than the one being described. Distance-preservation metrics therefore provide a way to evaluate descriptive generation as controlled expansion rather than simple similarity matching.

### 3.3 Long-Context Generation and Cohesion

Long-context evaluation has often been dominated by needle-in-the-haystack tests. Those tests are useful for measuring retrieval of isolated facts, but they say little about whether a long output remains cohesive across many categories, themes, or claims. A long text can retrieve a fact correctly and still drift in how it represents the relationships among concepts.

Description topology offers a complementary evaluation. A long document can be segmented into claims, paragraphs, entities, or conceptual roles; those segments induce a point cloud. The generated text can then be evaluated according to whether descriptions across many categories preserve the topology of the source or planning space. This generalizes beyond single-sentence factuality: it asks whether the document maintains coherent relative structure across the full semantic field.

## 4. Next Steps

The immediate next step is to extend description measurement from deterministic classification into probabilistic prediction. In a video-title setting, each title has an associated number of views. After embedding titles into a dimensionally reduced semantic space, some latent dimensions may have positive effects on performance while others have negative effects. A linear regression can predict log views from those latent features. The parallel linguistic experiment is to describe each predictive dimension, provide a model with the channel's behavior in terms of those descriptions, and test how precisely estimating the latent linguistic features improves numerical prediction. In this framing, a description is no longer only a category label; it becomes a reasoning trace or guideline for probabilistic inference.

A second next step is information retrieval under concentration of measure. High-dimensional embedding spaces often produce isotropic distances: many candidates appear similarly distant from a query, even though only a few semantic dimensions are actually relevant. The retrieval challenge becomes twofold: identify which dimensions matter for a query, and weight them appropriately. In graph information retrieval, texts are associated with references, and each reference has a semantic indicator such as a title. The hypothesis is that the way the reference point cloud is described affects retrieval accuracy during graph navigation. Better descriptions should identify the relevant subspace, weight it more effectively, and reduce navigation errors caused by irrelevant dimensions.

Together, these extensions move the framework from asking whether a description names a category to asking whether a description exposes the latent structure needed for prediction, retrieval, and reasoning.

## 5. Improving Descriptions

The final question is how descriptions can be improved once their topology is measurable. One possibility is guided generation at inference time. Multiple candidate descriptions or explanations can be generated, and a topological metric can select the inference path that best preserves the relevant structure. The architecture would be inspired by speculative decoding: several smaller models propose candidate continuations, while a larger evaluator accepts, rejects, or reranks them. Here, the acceptance criterion is not only token likelihood but geometric faithfulness.

A second possibility is self-distillation, or on-policy distillation. Instead of assigning a sparse reward to a completed answer, the teacher model's preferred response under the topological metric can provide a dense token-level signal through its next-token distribution, often called logit distillation. The best generated sample is rewarded across the full sequence, allowing the student to learn which local linguistic choices preserve global semantic structure.

This training setting creates an important evaluation problem: grokking versus memorization. A model may memorize category descriptions without understanding how descriptions must adjust relative to other categories sharing the same semantic space. The stronger test is whether it generalizes to unseen category configurations, analogous to arithmetic generalization tests with larger numbers, more summands, or unseen combinations. The description problem is larger because the relevant configuration space is unordered and scales combinatorially with the number of categories and independent semantic dimensions. This makes it a useful setting for comparing LoRA and full fine-tuning. A plausible hypothesis is that LoRA may reduce local semantic drift while failing to generalize as strongly to new relational configurations.

A third possibility is mid-training that enlarges the model's linguistic space before supervised answer-following. One direction is inspired by work on thinking with visual primitives, where spatial markers help bridge reference gaps in multimodal reasoning. Analogously, long-context generation could benefit from vocabulary or control tokens that explicitly point to previous sentences, paragraphs, concepts, or reference clusters. Another direction resembles world-model training: given a set of category descriptions, the model predicts which categories will fail classification or which pair will be confused most often. A final direction is architectural. Hierarchical or sub-quadratic attention mechanisms, including sparse attention, long convolutional hybrids, and hierarchical positional schemes, suggest that category descriptions could be treated as one level of a semantic hierarchy and description tokens as another. Even if a specialized attention mechanism for relative descriptions is too narrow, the findings may inform mixture-of-experts routing, semantic segmentation, long-context retrieval, and other systems where preserving relational structure is central.

## 6. Limitations

The correlation analyses operate over 24 categories, which limits statistical power for category-level relationships. The distance measures depend on the choice of sentence embedding model; the geometric relationships observed here may shift under different architectures. The classifier operates on short text items without access to additional context such as metadata or publication history. The data preparation stage introduces a model-dependent preprocessing step whose adequacy cannot be fully verified independently. Finally, source attribution is treated as ground truth even in cases where item semantics are genuinely more consistent with an adjacent category, a limitation intrinsic to any dataset defined by provenance rather than topical annotation.

## 7. Conclusion

This paper argues that semantic topology provides a principled basis for evaluating descriptions in non-verifiable domains. The main result is that description distortion predicts classification quality: categories whose descriptions preserve the relative geometry of the underlying instance space are classified more accurately. The complementary results explain the structure around that finding: central descriptions are harder to quantify, close categories are more easily confused, and high-volume categories may be more difficult to classify.

The broader implication is that descriptions should be evaluated as maps between metric spaces. When a description preserves neighborhoods, margins, and relative distances, it can support reliable classification, retrieval, summarization, explanation, and long-context generation. When it distorts those relationships, errors become predictable from the geometry itself. Evaluation in non-verifiable domains therefore need not depend exclusively on labeled answers; it can be reformulated around the structural fidelity of the representations that language creates.
