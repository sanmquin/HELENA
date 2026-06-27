import json
import os

class NotebookBuilder:
    def __init__(self):
        self.cells = []

    def add_markdown(self, source):
        self.cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [source]
        })

    def add_code(self, source):
        self.cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [source]
        })

    def save(self, filepath):
        notebook = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.10.12"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }
        with open(filepath, 'w') as f:
            json.dump(notebook, f, indent=4)

builder = NotebookBuilder()

# 1. Title
builder.add_markdown("# Humans vs. LLMs: Why AI-Driven Title Optimization Can Outperform Human Intuition\n\nThis research paper explores the comparative performance of human-generated video titles versus those optimized by Large Language Models (LLMs) through an iterative feedback loop.")

# 2. Setup and Data Loading
builder.add_code([
    "# Setup and Dependencies\n",
    "!pip install -q sentence-transformers\n",
    "\n",
    "import json\n",
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy import stats\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.manifold import TSNE\n",
    "from google.colab import drive\n",
    "\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "BASE_PATH = '/content/drive/MyDrive/numeric_inference_outputs/'\n",
    "RESULTS_PATH = os.path.join(BASE_PATH, 'title_optimization_results_v2.json')\n",
    "EVAL_DATA_PATH = os.path.join(BASE_PATH, 'top_significant_channels_eval.json')\n",
    "LLM_RESULTS_PATH = os.path.join(BASE_PATH, 'llm_analysis_results.json')\n",
    "TRAIN_DATA_PATH = os.path.join(BASE_PATH, 'train_structured_latest.json')\n",
    "EMBEDDING_CACHE_PATH = os.path.join(BASE_PATH, 'video_title_embeddings_latest.json')\n",
    "EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'\n",
    "\n",
    "embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)\n",
    "\n",
    "with open(RESULTS_PATH, 'r') as f:\n",
    "    results = json.load(f)\n",
    "\n",
    "with open(EVAL_DATA_PATH, 'r') as f:\n",
    "    eval_dataset = json.load(f)\n",
    "\n",
    "with open(LLM_RESULTS_PATH, 'r') as f:\n",
    "    llm_analysis = json.load(f)\n",
    "\n",
    "with open(TRAIN_DATA_PATH, 'r') as f:\n",
    "    train_data = json.load(f)\n",
    "\n",
    "with open(EMBEDDING_CACHE_PATH, 'r') as f:\n",
    "    embedding_cache = json.load(f)\n",
    "\n",
    "# PCA Reconstruction\n",
    "all_train_embeddings = []\n",
    "for channel in train_data:\n",
    "    for video in channel['videos']:\n",
    "        if video['title'] in embedding_cache:\n",
    "            all_train_embeddings.append(embedding_cache[video['title']])\n",
    "\n",
    "X_train = np.array(all_train_embeddings)\n",
    "pca = PCA(n_components=15, random_state=42)\n",
    "pca.fit(X_train)\n",
    "\n",
    "df = pd.DataFrame(results)\n",
    "print(f\"Loaded {len(df)} optimization results.\")"
])

# 3. Research Paper Structure
builder.add_markdown([
    "## Hypothesis: LLMs Can Discover High-Performance Titles Beyond Human Intuition\n",
    "\n",
    "### Methodology\n",
    "We utilize an iterative optimization loop where an LLM (Gemini 3.1 Flash lite) generates titles, which are then scored by a per-channel OLS model. This feedback is used to refine suggestions over 5 rounds.\n",
    "\n",
    "### Hypotheses\n",
    "- **Null Hypothesis (H0)**: Human-generated titles (original) represent the performance ceiling for a given video topic.\n",
    "- **Alternative Hypothesis (H1)**: Iterative LLM optimization can discover semantic representations that predict higher view counts than human-authored titles by navigating the latent space more effectively."
])

# 4. Scatter Charts
builder.add_code([
    "def plot_improvement_scatter(df, channel_filter=None, title=\"Title Performance Comparison\"):\n",
    "    plot_df = df.copy() if channel_filter is None else df[df['channel'] == channel_filter]\n",
    "    \n",
    "    human_data = []\n",
    "    ai_data = []\n",
    "    \n",
    "    for _, row in plot_df.iterrows():\n",
    "        # Calculate average score of all AI suggestions for this video across all iterations\n",
    "        all_ai_scores = []\n",
    "        for it in row['history']:\n",
    "            for t in it['titles']:\n",
    "                all_ai_scores.append(t['score'])\n",
    "        \n",
    "        avg_ai_score = np.mean(all_ai_scores)\n",
    "        \n",
    "        # Human performance: original_score vs (original_score - avg_ai_score)\n",
    "        human_data.append({\n",
    "            'pred_views': row['original_score'],\n",
    "            'improvement': row['original_score'] - avg_ai_score,\n",
    "            'type': 'Human'\n",
    "        })\n",
    "        \n",
    "        # AI performance: each suggestion's score vs (score - avg_ai_score)\n",
    "        for it in row['history']:\n",
    "            for t in it['titles']:\n",
    "                ai_data.append({\n",
    "                    'pred_views': t['score'],\n",
    "                    'improvement': t['score'] - avg_ai_score,\n",
    "                    'type': 'AI'\n",
    "                })\n",
    "    \n",
    "    h_df = pd.DataFrame(human_data)\n",
    "    a_df = pd.DataFrame(ai_data)\n",
    "    \n",
    "    plt.figure(figsize=(12, 6))\n",
    "    # Note: Using small alpha for AI data to see Human points clearly\n",
    "    sns.scatterplot(data=a_df, x='pred_views', y='improvement', color='green', alpha=0.3, label='Generated')\n",
    "    sns.scatterplot(data=h_df, x='pred_views', y='improvement', color='red', s=100, label='Human (Original)')\n",
    "    \n",
    "    plt.axhline(0, color='black', linestyle='--')\n",
    "    plt.title(title)\n",
    "    plt.xlabel(\"Predicted Views (Log)\")\n",
    "    plt.ylabel(\"Improvement vs. Average AI Suggestion\")\n",
    "    plt.legend()\n",
    "    plt.show()\n",
    "\n",
    "# 1. 20VC Scatter\n",
    "plot_improvement_scatter(df, \"20VC with Harry Stebbings\", \"20VC: Human vs AI Performance Landscape\")\n",
    "\n",
    "# 2. a16z Scatter\n",
    "plot_improvement_scatter(df, \"a16z\", \"a16z: Human vs AI Performance Landscape\")\n",
    "\n",
    "# 3. Combined Scatter\n",
    "plot_improvement_scatter(df, None, \"Combined: Human vs AI Performance Landscape\")"
])

# 5. Success Determinants Tables
builder.add_markdown("## Success Determinants: Dimension Weights and Significance\n\nWhat determines success for a video? We analyze the contribution (weight coefficient) of each dimension, ordered by significance.")

builder.add_code([
    "def show_dimension_importance(channel_name, eval_dataset):\n",
    "    channel_data = next(c for c in eval_dataset if c['channel_name'].lower() == channel_name.lower())\n",
    "    coeffs = channel_data['model']['coefficients']\n",
    "    p_values = channel_data['model']['p_values']\n",
    "    \n",
    "    importance_df = pd.DataFrame({\n",
    "        'Dimension': [f'Dim {i}' for i in range(len(coeffs))],\n",
    "        'Weight': coeffs,\n",
    "        'Significance (p-value)': p_values\n",
    "    })\n",
    "    \n",
    "    importance_df['AbsWeight'] = importance_df['Weight'].abs()\n",
    "    importance_df = importance_df.sort_values('Significance (p-value)', ascending=True).drop(columns=['AbsWeight'])\n",
    "    \n",
    "    print(f\"### Dimension Contribution for {channel_name}\")\n",
    "    display(importance_df.reset_index(drop=True))\n"
])

builder.add_code("show_dimension_importance(\"20VC with Harry Stebbings\", eval_dataset)")
builder.add_code("show_dimension_importance(\"a16z\", eval_dataset)")

# 6. Human vs AI Performance Tables
builder.add_markdown("## Human vs. AI Performance: A Comparative Analysis\n\nThese tables compare human-generated titles against AI optimized ones, including the relative difference to the best and average AI suggestions.")

builder.add_code([
    "def create_human_ai_comparison(df, channel_name, eval_dataset, pca, embedding_model):\n",
    "    channel_df = df[df['channel'] == channel_name]\n",
    "    channel_eval = next(c for c in eval_dataset if c['channel_name'].lower() == channel_name.lower())\n",
    "    coeffs = np.abs(channel_eval['model']['coefficients'])\n",
    "    top_5_dims = np.argsort(coeffs)[-5:][::-1]\n",
    "    \n",
    "    rows = []\n",
    "    for _, row in channel_df.iterrows():\n",
    "        video_eval = next(v for v in channel_eval['test_videos'] if v['video_id'] == row['video_id'])\n",
    "        real_views = video_eval['actual_views']\n",
    "        \n",
    "        all_ai_scores = []\n",
    "        for it in row['history']:\n",
    "            for t in it['titles']:\n",
    "                all_ai_scores.append(t['score'])\n",
    "        \n",
    "        best_ai_score = max(all_ai_scores)\n",
    "        avg_ai_score = np.mean(all_ai_scores)\n",
    "        \n",
    "        # Truncate titles for visibility if needed\n",
    "        # Comment: We use a limit of 100 characters to ensure full titles are mostly visible while preventing overflow\n",
    "        def truncate(t): return (t[:97] + '...') if len(t) > 100 else t\n",
    "        \n",
    "        emb = embedding_model.encode([row['original_title']])\n",
    "        proj = pca.transform(emb)[0]\n",
    "        \n",
    "        res = {\n",
    "            'Human Title': truncate(row['original_title']),\n",
    "            'Real Views': int(real_views),\n",
    "            'Pred Score': f\"{row['original_score']:.4f}\",\n",
    "            'vs. Best AI': f\"{row['original_score'] - best_ai_score:+.4f}\",\n",
    "            'vs. Avg AI': f\"{row['original_score'] - avg_ai_score:+.4f}\"\n",
    "        }\n",
    "        for d in top_5_dims:\n",
    "            res[f'Dim {d}'] = f\"{proj[d]:.4f}\"\n",
    "            \n",
    "        rows.append(res)\n",
    "        \n",
    "    print(f\"### Human vs AI Comparison: {channel_name}\")\n",
    "    display(pd.DataFrame(rows))\n"
])

builder.add_code("create_human_ai_comparison(df, \"20VC with Harry Stebbings\", eval_dataset, pca, embedding_model)")
builder.add_code("create_human_ai_comparison(df, \"a16z\", eval_dataset, pca, embedding_model)")

# 7. Prompt Examples
builder.add_markdown("## Iterative Refinement: The Prompting Strategy\n\nThe optimization process utilizes a 5-round feedback loop. Below are examples of the prompts used.")

builder.add_markdown("### Initial Generation Prompt\nThis prompt is used in the first iteration to set the baseline and explore the semantic space based on global success drivers.")

builder.add_code([
    "initial_prompt_example = \"\"\"You are an expert YouTube title strategist for the channel 'A16z'.\n",
    "Channel success drivers:\n",
    "[Detailed qualitative analysis of what drives views...]\n",
    "\n",
    "Semantic performance analysis:\n",
    "[Analysis of significant PCA dimensions...]\n",
    "\n",
    "Original Title: Udio: From Text to Tune\n",
    "Current best predicted performance (log-views): 8.0034\n",
    "\n",
    "Task: Generate 10 new, improved titles for this video that will maximize views based on the channel's success drivers and semantic analysis. Return ONLY the 10 titles, one per line.\"\"\"\n",
    "print(initial_prompt_example)"
])

builder.add_markdown("### Feedback-Based Refinement Prompt\nIn subsequent rounds, the LLM receives feedback on its previous suggestions, enabling a \"hill-climbing\" optimization.")

builder.add_code([
    "refinement_prompt_example = \"\"\"[... previous context ...]\n",
    "\n",
    "Previous suggestions feedback:\n",
    "Top performing suggestions:\n",
    "- The Future of Sound: Inside the AI Startup Disrupting the $25B Music Industry (Score: 10.4334)\n",
    "- ...\n",
    "\n",
    "Lower performing suggestions (Avoid these patterns):\n",
    "- Why Udio is Changing Everything (Score: 8.5)\n",
    "- ...\n",
    "\n",
    "Task: Generate 10 new, improved titles for this video...\"\"\"\n",
    "print(refinement_prompt_example)"
])

# 8. Individual Dimension Analysis
builder.add_markdown("## Individual Dimension Analysis: Driving Positive Direction\n\nWe analyze how titles correlate with the positive direction of each significant dimension. This section explicitly indicates the intended direction for performance optimization.")

builder.add_code([
    "def analyze_individual_dimensions(df, channel_name, eval_dataset, pca, embedding_model):\n",
    "    channel_df = df[df['channel'] == channel_name]\n",
    "    channel_eval = next(c for c in eval_dataset if c['channel_name'].lower() == channel_name.lower())\n",
    "    coeffs = channel_eval['model']['coefficients']\n",
    "    top_5_indices = np.argsort(np.abs(coeffs))[-5:][::-1]\n",
    "    \n",
    "    print(f\"### Individual Dimension Breakdown for {channel_name}\")\n",
    "    \n",
    "    for dim_idx in top_5_indices:\n",
    "        weight = coeffs[dim_idx]\n",
    "        direction = \"INCREASE\" if weight > 0 else \"DECREASE\"\n",
    "        \n",
    "        # Extract values for this dimension for all optimized titles\n",
    "        all_optimized_titles = []\n",
    "        for _, row in channel_df.iterrows():\n",
    "            for it in row['history']:\n",
    "                for t in it['titles']:\n",
    "                    all_optimized_titles.append(t['text'])\n",
    "        \n",
    "        all_optimized_titles = list(set(all_optimized_titles))\n",
    "        embs = embedding_model.encode(all_optimized_titles)\n",
    "        projs = pca.transform(embs)[:, dim_idx]\n",
    "        \n",
    "        avg_val = np.mean(projs)\n",
    "        \n",
    "        plt.figure(figsize=(10, 4))\n",
    "        sns.histplot(projs, kde=True, color='blue' if weight > 0 else 'orange')\n",
    "        plt.axvline(avg_val, color='red', linestyle='--', label=f'Avg: {avg_val:.4f}')\n",
    "        plt.title(f\"Dimension {dim_idx} Distribution (Weight: {weight:.4f}) - Should {direction}\")\n",
    "        plt.legend()\n",
    "        plt.show()\n",
    "        \n",
    "        # Top 5 titles for this dimension (considering direction)\n",
    "        sorted_indices = np.argsort(projs)\n",
    "        if weight > 0:\n",
    "            top_indices = sorted_indices[-5:][::-1]\n",
    "        else:\n",
    "            top_indices = sorted_indices[:5]\n",
    "            \n",
    "        print(f\"Top titles for Dim {dim_idx} (direction: {direction}):\")\n",
    "        for idx in top_indices:\n",
    "            print(f\"- {all_optimized_titles[idx]} (Value: {projs[idx]:.4f})\")\n",
    "        print(\"\\n\")\n"
])

builder.add_code("analyze_individual_dimensions(df, \"20VC with Harry Stebbings\", eval_dataset, pca, embedding_model)")
builder.add_code("analyze_individual_dimensions(df, \"a16z\", eval_dataset, pca, embedding_model)")

# 9. Multivariate Analysis and Alignment
builder.add_markdown("## Multivariate Alignment: Correlation and Trade-offs\n\nHow do different dimensions interact? We compute the degree of alignment in the top 5 most significant dimensions and visualize their correlations using a confusion matrix of their directions.")

builder.add_code([
    "def analyze_multivariate_alignment(df, channel_name, eval_dataset, pca, embedding_model):\n",
    "    channel_df = df[df['channel'] == channel_name]\n",
    "    channel_eval = next(c for c in eval_dataset if c['channel_name'].lower() == channel_name.lower())\n",
    "    coeffs = channel_eval['model']['coefficients']\n",
    "    top_5_indices = np.argsort(np.abs(coeffs))[-5:][::-1]\n",
    "    \n",
    "    # Get dimension descriptions\n",
    "    print(f\"### Dimension Descriptions for {channel_name}\")\n",
    "    cid = channel_eval['channel_id']\n",
    "    desc = llm_analysis['channel_significant_dimension_analysis'].get(cid, \"Descriptions not found\")\n",
    "    print(desc)\n",
    "    \n",
    "    # Extract values for top 5 dimensions for all optimized titles\n",
    "    all_titles = []\n",
    "    for _, row in channel_df.iterrows():\n",
    "        for it in row['history']:\n",
    "            for t in it['titles']:\n",
    "                all_titles.append(t['text'])\n",
    "    \n",
    "    all_titles = list(set(all_titles))\n",
    "    embs = embedding_model.encode(all_titles)\n",
    "    projs = pca.transform(embs)[:, top_5_indices]\n",
    "    \n",
    "    # Normalize by desired direction (multiply by sign of weight)\n",
    "    signs = np.sign([coeffs[i] for i in top_5_indices])\n",
    "    aligned_projs = projs * signs\n",
    "    \n",
    "    # Confusion Matrix (Correlation) of aligned directions\n",
    "    corr_matrix = pd.DataFrame(aligned_projs, columns=[f'Dim {i}' for i in top_5_indices]).corr()\n",
    "    \n",
    "    plt.figure(figsize=(10, 8))\n",
    "    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=\".2f\")\n",
    "    plt.title(f\"Directional Correlation Matrix for Top 5 Dimensions ({channel_name})\")\n",
    "    plt.show()\n",
    "    \n",
    "    # Compare to Human titles\n",
    "    human_titles = channel_df['original_title'].tolist()\n",
    "    h_embs = embedding_model.encode(human_titles)\n",
    "    h_projs = pca.transform(h_embs)[:, top_5_indices] * signs\n",
    "    \n",
    "    print(\"\\nAverage Alignment (Normalized Score):\")\n",
    "    print(f\"AI Optimized Average: {aligned_projs.mean(axis=0)}\")\n",
    "    print(f\"Human Average:        {h_projs.mean(axis=0)}\")\n",
    "\n",
    "analyze_multivariate_alignment(df, \"20VC with Harry Stebbings\", eval_dataset, pca, embedding_model)\n",
    "analyze_multivariate_alignment(df, \"a16z\", eval_dataset, pca, embedding_model)"
])

# 10. Optimization Landscape
builder.add_markdown("## The Optimization Landscape: Visualizing Trajectories to Success\n\nWhere do the best titles live? We project the high-performance suggestions and original human titles into a 2D space to visualize the semantic clusters associated with success.")

builder.add_code([
    "def analyze_optimization_landscape(df, pca, embedding_model):\n",
    "    all_titles_data = []\n",
    "    \n",
    "    # 1. Collect best AI suggestions (overall, top 20 across all videos)\n",
    "    all_ai_suggestions = []\n",
    "    for _, row in df.iterrows():\n",
    "        for it in row['history']:\n",
    "            for t in it['titles']:\n",
    "                all_ai_suggestions.append({\n",
    "                    'title': t['text'],\n",
    "                    'score': t['score'],\n",
    "                    'iteration': it['iteration'],\n",
    "                    'original_video': row['original_title'],\n",
    "                    'type': 'AI'\n",
    "                })\n",
    "    \n",
    "    top_ai = pd.DataFrame(all_ai_suggestions).sort_values('score', ascending=False).head(20)\n",
    "    \n",
    "    # 2. Collect best Human titles (top 10)\n",
    "    top_human = df.sort_values('original_score', ascending=False).head(10).copy()\n",
    "    top_human['type'] = 'Human'\n",
    "    top_human = top_human.rename(columns={'original_title': 'title', 'original_score': 'score'})\n",
    "    top_human['iteration'] = 0\n",
    "    top_human['original_video'] = top_human['title']\n",
    "    \n",
    "    combined_best = pd.concat([top_ai[['title', 'score', 'type', 'iteration', 'original_video']], \n",
    "                               top_human[['title', 'score', 'type', 'iteration', 'original_video']]])\n",
    "    \n",
    "    embs = embedding_model.encode(combined_best['title'].tolist())\n",
    "    # Use t-SNE for 2D projection of the 15D PCA space\n",
    "    projs_15d = pca.transform(embs)\n",
    "    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_best)-1))\n",
    "    projs_2d = tsne.fit_transform(projs_15d)\n",
    "    \n",
    "    combined_best['x'] = projs_2d[:, 0]\n",
    "    combined_best['y'] = projs_2d[:, 1]\n",
    "    \n",
    "    plt.figure(figsize=(12, 10))\n",
    "    sns.scatterplot(data=combined_best, x='x', y='y', hue='type', size='score', \n",
    "                    sizes=(100, 500), palette={'AI': 'green', 'Human': 'red'}, alpha=0.7)\n",
    "    \n",
    "    for i, row in combined_best.reset_index().iterrows():\n",
    "        plt.text(row['x']+0.1, row['y']+0.1, row['title'][:30], fontsize=8)\n",
    "        \n",
    "    plt.title(\"2D Projection of Top Performing Titles (Human vs AI)\")\n",
    "    plt.show()\n",
    "    \n",
    "    # Distances analysis\n",
    "    # Distance from best AI to closest best Human in 15D space\n",
    "    from sklearn.metrics import euclidean_distances\n",
    "    ai_idx = combined_best['type'] == 'AI'\n",
    "    human_idx = combined_best['type'] == 'Human'\n",
    "    dist_matrix = euclidean_distances(projs_15d[ai_idx], projs_15d[human_idx])\n",
    "    \n",
    "    print(\"### Distance and Iteration Analysis\")\n",
    "    print(f\"Average distance between top AI and top Human titles (15D): {dist_matrix.mean():.4f}\")\n",
    "    \n",
    "    # Table of top performing suggestions\n",
    "    print(\"\\n#### Top 10 Overall Suggestions\")\n",
    "    display(top_ai.head(10)[['title', 'score', 'iteration', 'original_video']])\n",
    "    \n",
    "    # Iteration analysis\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.countplot(data=top_ai, x='iteration', palette='viridis')\n",
    "    plt.title(\"In which iteration were the top 20 AI suggestions found?\")\n",
    "    plt.show()\n",
    "    \n",
    "    # Video membership\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    top_ai['original_video'].value_counts().plot(kind='barh', color='skyblue')\n",
    "    plt.title(\"Which videos do the top AI suggestions belong to?\")\n",
    "    plt.xlabel(\"Count in Top 20 Suggestions\")\n",
    "    plt.show()\n",
    "\n",
    "analyze_optimization_landscape(df, pca, embedding_model)"
])

# 11. Contrastive Linguistic Analysis
builder.add_markdown("## Contrastive Linguistic Analysis: Style and Semantics\n\nWhat makes a title fail or succeed? We contrast the linguistic styles of the top-performing and bottom-performing suggestions discovered during the optimization process.")

builder.add_code([
    "def analyze_linguistic_differences(df):\n",
    "    all_ai = []\n",
    "    for _, row in df.iterrows():\n",
    "        for it in row['history']:\n",
    "            for t in it['titles']:\n",
    "                all_ai.append({'text': t['text'], 'score': t['score']})\n",
    "    \n",
    "    ai_df = pd.DataFrame(all_ai).sort_values('score', ascending=False)\n",
    "    top_titles = ai_df.head(20)['text'].tolist()\n",
    "    bottom_titles = ai_df.tail(20)['text'].tolist()\n",
    "    \n",
    "    print(\"### Top Performing Stylistic Patterns\")\n",
    "    for t in top_titles[:5]: print(f\"[SUCCESS] {t}\")\n",
    "    \n",
    "    print(\"\\n### Bottom Performing Stylistic Patterns\")\n",
    "    for t in bottom_titles[:5]: print(f\"[FAILURE] {t}\")\n",
    "    \n",
    "    # simple feature extraction: length, number of punctuation, capital letters\n",
    "    def get_features(t):\n",
    "        return {\n",
    "            'len': len(t),\n",
    "            'caps': sum(1 for c in t if c.isupper()),\n",
    "            'punc': sum(1 for c in t if c in '!:?.,')\n",
    "        }\n",
    "    \n",
    "    top_feats = pd.DataFrame([get_features(t) for t in top_titles])\n",
    "    bottom_feats = pd.DataFrame([get_features(t) for t in bottom_titles])\n",
    "    \n",
    "    print(\"\\n### Average Structural Differences\")\n",
    "    stats_df = pd.DataFrame({\n",
    "        'Metric': ['Length', 'Capitals', 'Punctuation'],\n",
    "        'Top Performers': [top_feats['len'].mean(), top_feats['caps'].mean(), top_feats['punc'].mean()],\n",
    "        'Bottom Performers': [bottom_feats['len'].mean(), bottom_feats['caps'].mean(), bottom_feats['punc'].mean()]\n",
    "    })\n",
    "    display(stats_df)\n",
    "\n",
    "analyze_linguistic_differences(df)"
])

builder.save('title generation/2.Humans_vs_LLMs.ipynb')
