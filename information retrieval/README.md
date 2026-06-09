# Information Retrieval Dataset Building

This directory contains a notebook for building a citation-based dataset of Arxiv papers for information retrieval experiments.

## Purpose

The `dataset_builder.ipynb` notebook automates the process of collecting paper metadata and their references to create a graph-like dataset. It starts from a seed Arxiv ID and expands by following references, prioritizing papers that are frequently cited within the collected set.

## Dataset Structure

The dataset is stored as a JSON object where each entry represents a paper:

```typescript
interface Paper {
  arxivId: string;
  title: string;
  abstract: string;
  references: string[]; // List of arxivIDs
  metadata: {
    year: number;
    fetched_at: string; // ISO timestamp
    attempted: boolean;
    success: boolean;
    error?: string;
  };
}

interface Dataset {
  [arxivId: string]: Paper;
}
```

## Features

- **Recursive Fetching**: Starts with an initial Arxiv ID, fetches its data and references, then proceeds to references.
- **Citation-based Prioritization**: Incomplete papers (referenced but not yet fetched) are prioritized based on how many times they have been cited by papers already in the dataset.
- **Filtering**: Only papers published in 2020 or later are included.
- **Duplicate Prevention**: Tracks attempted fetches and existing data to avoid redundant API calls.
- **Rate Limiting**: Respects SemanticScholar and Arxiv API quotas.
- **Google Drive Integration**: Saves state periodically to Google Drive to allow for session-resuming.

## Setup

1. Open `dataset_builder.ipynb` in Google Colab.
2. Ensure you have a Google Drive mounted at `/content/drive`.
3. Set up any necessary API keys (if applicable, though SemanticScholar has a public tier).
4. Run the notebook to fetch up to 100 new papers per session.
