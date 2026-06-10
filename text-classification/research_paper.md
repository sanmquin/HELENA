# Semantic Topology as a Proxy for Evaluation Quality in Non-Verifiable Domains

## Abstract

Large language models are increasingly deployed in domains where ground-truth labels are absent, contested, or inherently subjective. Evaluating model behavior in such settings demands metrics that go beyond exact-match accuracy. This paper argues that semantic topology — the geometric structure induced by text embeddings — provides principled proxies for evaluation quality even when labels cannot be independently verified. To ground this argument empirically, we study a description-based text classification task in which LLM-generated category descriptions serve as the sole interface between training data and a classifier. We find that classification errors are not random: semantically distinctive categories perform better (Pearson r = 0.44, p = 0.03), semantically close categories are confused more often (r = −0.24, p < 0.0001), and categories whose descriptions better preserve the geometry of the underlying data are substantially more accurate (r = 0.56, p = 0.004). Together, these findings suggest that distance-preservation metrics can serve as description-quality proxies in the absence of labeled test data, and that this principle extends to any domain where outputs are characterized by natural-language descriptions rather than verifiable facts.

## 1. Introduction

The dominant paradigm for evaluating language models relies on held-out labeled data: a reference answer is compared to the model's output, and quality is measured by agreement. This paradigm is well-suited to tasks where reference answers are unambiguous — factual retrieval, structured data extraction, or well-specified multiple-choice problems. It is poorly suited to a growing class of tasks where no such reference exists.

Much of the value that modern language models are expected to deliver falls outside the regime of verifiable correctness. Content characterization, qualitative assessment, creative evaluation, recommendation labeling, and source attribution in semantically overlapping domains all share the same structural feature: the correct answer is not independently verifiable, and reasonable annotators may disagree. Deploying evaluation methodologies designed for verifiable domains in these settings is not merely imprecise — it conflates inherent semantic ambiguity with model failure, and it discards information about whether errors are coherent with the domain's own structure.

This paper uses description-based text classification as a transparent model for this broader problem. In the task studied here, a language model must assign short text items to one of many categories that are defined exclusively through natural-language descriptions. No external authority adjudicates individual assignments, and many errors are matters of genuine topical overlap rather than obvious factual failure. The setting is therefore both a practical classification problem and a controlled testbed for non-verifiable evaluation.

The central hypothesis is that semantic topology — the geometric relationships among embedded representations of categories and their instances — encodes the difficulty structure of the task in a way that is computable without reference to ground-truth labels. If this hypothesis holds, then the same geometric measures can be used to evaluate the quality of a description system before any labeled data is collected, and can be generalized to domains where labeled data will never be available.

The results support this hypothesis across four distinct operationalizations. More importantly, they motivate a framework in which the quality of a set of natural-language descriptions is assessed not by asking whether individual assignments are correct, but by asking whether the descriptions faithfully represent the geometric structure of the domain they purport to characterize.

## 2. Results

### 2.1 Overall Classification Performance

The evaluation pipeline achieves 42.37% accuracy across 24 categories on 236 held-out items, with macro F1 of 0.42. Aggregate performance on this task is not the primary object of interest; it is offered as context for the more informative finding that per-category performance varies substantially, from near-perfect recall on topically distinctive categories to near-zero recall on categories whose content overlaps heavily with several neighbors.

| Metric | Value |
|---|---:|
| Accuracy | 0.4237 |
| Macro precision | 0.44 |
| Macro recall | 0.43 |
| Macro F1 | 0.42 |
| Weighted F1 | 0.41 |
| Errors | 136 / 236 |

This variance in per-category difficulty is itself a meaningful result. Uniform difficulty would be consistent with errors driven primarily by model capacity; concentrated difficulty concentrated in specific regions of the category space is consistent with errors driven by semantic structure. The analyses below test the latter explanation systematically.

### 2.2 Semantic Distinctiveness and Classification Difficulty

The first structural relationship concerns how a category's position in the overall semantic space predicts its classification difficulty. Categories are more semantically distinctive when their embedding centroids lie far from the average centroid of all other categories. Correlating this distinctiveness with per-category F1-score yields a Pearson correlation of 0.44 (p = 0.03).

| Relationship | Pearson r | p-value |
|---|---:|---:|
| Average centroid distance vs. F1-score | 0.44 | 0.03 |

The positive correlation confirms that the classifier succeeds more often where the semantic signal is clearer and struggles most in the crowded interior of the category space. Critically, distinctiveness is computable entirely from embeddings, without labels, and serves as a leading indicator of expected difficulty before any evaluation data is collected.

### 2.3 Error Structure and Semantic Proximity

If classification errors followed a random model, the probability of mistaking one category for another would be uniform across all category pairs. Instead, correlating pairwise semantic distance with pairwise confusion count yields r = −0.24 (p < 0.0001): categories that are closer in embedding space are confused more often.

| Relationship | Pearson r | p-value |
|---|---:|---:|
| Semantic distance vs. confusion count | −0.24 | < 0.0001 |

The direction of the correlation is negative because smaller distance corresponds to larger confusion count. This result is consequential for the evaluation argument: a classifier whose errors are geometrically coherent is not failing randomly. Its mistakes are concentrated in regions where the correct assignment is genuinely ambiguous, a property that is measurable — and therefore diagnosable — through topology alone.

### 2.4 Semantic Volume and Within-Category Dispersion

Categories whose instances span a broad region of embedding space present a weaker classification signal than compact, well-defined categories. The correlation between per-category dispersion and recall is r = −0.32, in the predicted direction, though it falls short of conventional significance (p = 0.13) in the present sample.

| Relationship | Pearson r | p-value |
|---|---:|---:|
| Category dispersion vs. recall | −0.32 | 0.13 |

While inconclusive on its own, dispersion is a computationally accessible proxy for category difficulty, measurable before any labeled evaluation data is available. The non-significance is most plausibly attributable to sample size rather than to the absence of an underlying relationship, and the directional consistency supports the broader framework.

### 2.5 Description Quality and Semantic Distance Preservation

The most actionable finding concerns the relationship between description quality and classification accuracy. The global correlation between pairwise distances among raw category centroids and pairwise distances among generated descriptions is 0.56, indicating that the description layer moderately preserves the geometry of the underlying instance space. At the per-category level, the degree to which a category's description preserves its relative distances to all other categories correlates with that category's classification accuracy at r = 0.56 (p = 0.004).

| Relationship | Pearson r | p-value |
|---|---:|---:|
| Global centroid / description distance correlation | 0.56 | — |
| Per-category preservation vs. accuracy | 0.56 | 0.004 |

This is the central empirical claim of the paper. A description that sounds accurate in isolation is not necessarily a good description. A description is geometrically good — in a measurable, label-free sense — when it places the category in approximately the correct position relative to all other categories. Descriptions that distort this geometry harm classification in predictable and detectable ways.

## 3. Towards Label-Free Evaluation Quality

The empirical results above motivate a broader theoretical claim. In any domain characterized by natural-language descriptions, the quality of those descriptions can be partially assessed through their preservation of semantic geometry, and this assessment requires no labeled outputs.

Consider the general setting. A set of categories is defined by a corresponding set of descriptions. Instances must be assigned to one of these categories. No verified ground truth is available at evaluation time. Classical accuracy metrics cannot be applied. The question is what can still be measured and whether those measurements are informative.

Semantic distinctiveness, as shown in Section 2.2, answers the question of which categories are likely to cause difficulty, and it does so before any evaluation is conducted. Pairwise distance structure among descriptions answers the question of which errors are expected if the system is imperfect, and those expectations can be compared to observed error patterns even without knowing which individual predictions are correct. Distance preservation — the correlation between inter-description distances and inter-instance distances — answers the question of whether the description system faithfully represents the underlying domain, a prerequisite for the descriptions to be useful as a classification interface.

These three measures together constitute a form of structural validation: evidence that the description system represents the domain faithfully, even when no oracle is available to adjudicate individual assignments. Importantly, none of them requires a label. They require only embeddings of the descriptions and embeddings of a representative sample of instances from each category.

This principle extends beyond the classification setting studied here. Any corpus of natural-language descriptions — product characterizations, content summaries, qualitative rubrics, assessment criteria — can be evaluated by asking whether its inter-description geometry mirrors the inter-instance geometry of the objects it purports to describe. A rubric whose dimensions are geometrically orthogonal captures genuinely distinct aspects of quality. A set of content summaries that clusters differently than the content itself has distorted the domain. A collection of assessments whose relative distances do not reflect the relative similarities of the assessed objects is internally inconsistent, regardless of whether any individual assessment is "correct."

This transforms description evaluation from a labeling problem into a geometric one, opening it to principled quantitative analysis in settings where human verification is absent, expensive, or unreliable. The results presented here offer one concrete realization of that transformation and provide evidence that the geometric signals are strong enough to predict real classification outcomes.

## 4. Methodology

The evaluation pipeline proceeds through three stages: data preparation, description-based classification, and embedding analysis.

In the data preparation stage, a corpus of short text items is cleaned to remove lexical artifacts that directly reveal category identity — proper names, host names, and recurring branded phrases. This step ensures that classification performance reflects semantic understanding rather than surface-level string matching. The result is a cleaned dataset distributed across 24 thematically defined categories, spanning overlapping subject-matter domains including technology, finance, entrepreneurship, and self-improvement. Categories were split independently into 80% training and 20% testing subsets, yielding 236 held-out test items.

In the classification stage, a language model generates one natural-language description per category from that category's training items. These descriptions are the sole interface between the training data and the classifier; the classifier receives no training items directly, only the induced descriptions. Held-out test items are classified by prompting the model with all category names and their generated descriptions. This design makes description quality a central, experimentally isolable variable.

In the analysis stage, sentence embeddings from a standard transformer model (all-MiniLM-L6-v2) are computed for all items and for all generated descriptions. Category centroids are derived by averaging item embeddings within each category. Pairwise cosine distances are then used to operationalize four semantic properties: centrality (average distance from each centroid to all other centroids), error closeness (comparison of the pairwise distance matrix with the pairwise confusion matrix), semantic volume (mean distance from items to their category centroid), and distance preservation (correlation between pairwise distances among centroids and pairwise distances among the corresponding generated descriptions). Pearson correlations are used throughout to quantify relationships between semantic properties and classification performance.

## 5. Discussion

The results support a coherent picture: classification errors in semantically overlapping domains are not random but geometric, and the quality of the description layer — measured through distance preservation — is the primary determinant of accuracy differences across categories.

For evaluation practice, this has implications that extend well beyond the specific task studied here. Accuracy alone is insufficient as an evaluation metric in non-verifiable domains. A system that achieves low accuracy may nonetheless be failing in a principled, semantically coherent way, better characterized as proximity confusion than as ignorance. The appropriate diagnostic is not a higher-level benchmark but a geometric analysis of the error distribution — an analysis that is feasible without access to labeled data.

Category descriptions should be evaluated as geometric objects rather than as linguistic summaries. A description that sounds fluent and informative in isolation can still be harmful if it displaces its category toward the wrong neighbors. The preservation score operationalized here — the correlation between pairwise description distances and pairwise centroid distances — provides a description-quality proxy that is computable before any labeled evaluation is conducted. This makes it applicable to the precisely those settings, development-time and deployment-time alike, where labels are not yet available.

The error patterns observed across thematic regions of the category space further illustrate the evaluation argument. In the startup and venture domain, in the finance and macroeconomics domain, and in the business coaching domain, misclassifications are concentrated among semantically adjacent categories. These are not arbitrary errors; they are coherent with the topical overlap of the categories involved. An evaluation framework that treats them as equivalent to errors across distant domains will systematically mischaracterize the model's behavior.

The generalization of this framework is straightforward. Wherever an LLM must interpret a set of natural-language descriptions to make consequential assignments — recommendation systems that use textual item summaries, content moderation policies defined through exemplar text, qualitative scoring rubrics, assessment criteria in education or hiring — the structural validity of the description set can be assessed geometrically. If the descriptions accurately represent the relative similarities and distinctions among the objects they describe, the downstream system can be expected to behave consistently. If the descriptions distort those relationships, errors will be predictable from the geometry alone. In either case, the evaluation does not require ground-truth labels; it requires embeddings and a well-specified notion of what it means for a description to be geometrically faithful.

## 6. Limitations

The correlation analyses operate over 24 categories, which limits statistical power for category-level relationships. The distance measures depend on the choice of sentence embedding model; the geometric relationships observed here may shift under different architectures. The classifier operates on short text items without access to additional context — such as metadata or publication history — that would be available in deployed settings. The data preparation stage introduces a model-dependent preprocessing step whose adequacy cannot be fully verified independently. Finally, source attribution is treated as ground truth even in cases where item semantics are genuinely more consistent with an adjacent category, a limitation intrinsic to any dataset defined by provenance rather than topical annotation.

## 7. Conclusion

This paper argues that semantic topology provides a principled basis for evaluating language model behavior in non-verifiable domains. Empirically, four geometric properties — categorical distinctiveness, proximity-guided error structure, within-category dispersion, and description distance preservation — collectively explain a substantial portion of the variance in classification performance. The strongest individual predictor, distance preservation, is computed entirely from embeddings and requires no labels.

The broader implication is that geometric consistency between a description set and its underlying instance distribution is a measurable proxy for description quality in any domain where human verification is absent or costly. Evaluation in non-verifiable domains need not be abandoned because ground truth is unavailable; it must instead be reformulated around structural properties of the representation itself. The results presented here offer one concrete realization of that reformulation: a framework in which the quality of a natural-language description system is assessed by asking whether it faithfully preserves the geometry of the domain, rather than by asking whether it produces correct labels on a held-out test set.
