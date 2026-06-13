# Information Retrieval Dataset Building

This directory contains a TypeScript script for building a citation-based dataset of Arxiv papers for information retrieval experiments.

## Purpose

The `src/index.ts` script automates the process of collecting paper metadata and their references to create a graph-like dataset. It starts from a seed Arxiv ID and expands by following references, prioritizing papers that are frequently cited within the collected set.

## Dataset Structure

The dataset is stored as a JSON object (`arxiv_dataset.json`) where each entry represents a paper:

```typescript
interface Paper {
  arxivID: string;
  title: string;
  abstract: string;
  references: string[]; // List of arxivIDs
  metadata: {
    year: number;
    fetched_at: string; // ISO timestamp
    attempted: boolean;
    success: boolean;
    error?: string;
    source?: string;
  };
}

interface Dataset {
  [arxivId: string]: Paper;
}
```

Individual API responses are cached in the `cache/` directory to ensure reproducibility and avoid redundant network calls.

## Features

- **Recursive Fetching**: Starts with an initial Arxiv ID, fetches its data and references, then proceeds to references.
- **Citation-based Prioritization**: Incomplete papers (referenced but not yet fetched) are prioritized based on how many times they have been cited by papers already in the dataset.
- **Filtering**: Only papers published in 2020 or later are included.
- **Duplicate Prevention**: Tracks attempted fetches and existing data.
- **Multi-source Fallback**: Uses Semantic Scholar, OpenAlex, and Arxiv API in sequence to maximize data retrieval.
- **Rate Limiting**: Respects API quotas for all used services.
- **Local Caching**: Saves individual request responses to the `cache/` directory.

## Setup

1.  **Install dependencies**:
    ```bash
    npm install
    ```

2.  **Environment Setup**:
    Copy `.env.example` to `.env` and optionally add your Semantic Scholar API key.
    ```bash
    cp .env.example .env
    ```

3.  **Run the script**:
    ```bash
    npm start
    ```
    By default, it will attempt to fetch 100 new papers. You can adjust this in the `.env` file.

## Integration with Further Analysis

Once the dataset is built locally, it should be uploaded to Google Drive for use in the analysis notebooks (e.g., text-classification).

**Google Drive Target Path:** `/content/drive/MyDrive/information_retrieval/arxiv_dataset.json`

Ensure that the uploaded file matches the schema expected by the subsequent steps in the pipeline.
