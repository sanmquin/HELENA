import * as fs from 'fs';
import * as path from 'path';
import { SemanticScholarClient, OpenAlexClient, ArxivClient } from './clients';
import * as dotenv from 'dotenv';

dotenv.config();

const DATASET_PATH = path.join(__dirname, '..', 'arxiv_dataset.json');
const INITIAL_ARXIV_ID = '2303.08774'; // GPT-4 Technical Report
const TARGET_NEW_PAPERS = process.env.TARGET_NEW_PAPERS ? parseInt(process.env.TARGET_NEW_PAPERS) : 100;
const ERROR_LIMIT = 5;
const SAVE_EVERY = 10;

interface Paper {
  arxivID: string;
  title?: string;
  abstract?: string;
  references?: string[];
  metadata: {
    year?: number;
    fetched_at: string;
    attempted: boolean;
    success: boolean;
    error?: string;
    source?: string;
  };
}

interface Dataset {
  [arxivId: string]: Paper;
}

async function processPaper(
  arxivId: string,
  ssClient: SemanticScholarClient,
  oaClient: OpenAlexClient,
  arxivClient: ArxivClient,
  dataset: Dataset
): Promise<{ success: boolean; error: string | null }> {
  if (dataset[arxivId] && (dataset[arxivId].metadata.success || dataset[arxivId].metadata.attempted)) {
    return { success: false, error: 'Already attempted' };
  }

  console.log(`---> Processing paper: ${arxivId}`);

  let paperData: any = null;
  let source = '';
  let references: string[] = [];

  // 1. Try Semantic Scholar
  const ssRes = await ssClient.getPaperMetadata(arxivId);
  if (ssRes.data) {
    paperData = ssRes.data;
    source = 'SemanticScholar';
    const refRes = await ssClient.getReferences(arxivId);
    references = refRes.data;
    if (refRes.error) console.warn(`      [SS References Warning] ${refRes.error}`);
  } else {
    console.log(`      [SS Metadata Failed] ${ssRes.error}. Falling back to OpenAlex...`);
    // 2. Try OpenAlex
    const oaRes = await oaClient.getPaper(arxivId);
    if (oaRes.data) {
      paperData = oaRes.data;
      source = 'OpenAlex';
      // OpenAlex implementation currently returns empty references as they need resolution
    } else {
      console.log(`      [OpenAlex Failed] ${oaRes.error}. Falling back to Arxiv...`);
      // 3. Try Arxiv
      const arxivRes = await arxivClient.getPaper(arxivId);
      if (arxivRes.data) {
        paperData = arxivRes.data;
        source = 'Arxiv';
      } else {
        console.log(`      [Arxiv Failed] ${arxivRes.error}. Giving up.`);
        dataset[arxivId] = {
          arxivID: arxivId,
          metadata: {
            fetched_at: new Date().toISOString(),
            attempted: true,
            success: false,
            error: `SS: ${ssRes.error} | OA: ${oaRes.error} | Arxiv: ${arxivRes.error}`
          }
        };
        return { success: false, error: 'Failed all APIs' };
      }
    }
  }

  const year = paperData.year;
  if (year && year < 2020) {
    console.log(`      [Skipped] Published in ${year} (Source: ${source})`);
    dataset[arxivId] = {
      arxivID: arxivId,
      metadata: {
        year,
        fetched_at: new Date().toISOString(),
        attempted: true,
        success: false,
        error: `Skipped: Published in ${year}`
      }
    };
    return { success: false, error: `Skipped: ${year}` };
  }

  console.log(`      [Success] Found ${references.length} references via ${source}.`);

  dataset[arxivId] = {
    arxivID: arxivId,
    title: paperData.title || '',
    abstract: paperData.abstract || '',
    references: references,
    metadata: {
      year,
      fetched_at: new Date().toISOString(),
      attempted: true,
      success: true,
      source: source
    }
  };

  return { success: true, error: null };
}

function getNextBatch(dataset: Dataset, limit: number): string[] {
  const referencedIds: string[] = [];
  for (const id in dataset) {
    if (dataset[id].metadata.success) {
      referencedIds.push(...(dataset[id].references || []));
    }
  }

  const candidates = referencedIds.filter(id => !dataset[id]);
  const counts: { [id: string]: number } = {};
  for (const id of candidates) {
    counts[id] = (counts[id] || 0) + 1;
  }

  return Object.keys(counts)
    .sort((a, b) => counts[b] - counts[a])
    .slice(0, limit);
}

async function main() {
  const ssClient = new SemanticScholarClient(process.env.SS_API_KEY);
  const oaClient = new OpenAlexClient();
  const arxivClient = new ArxivClient();

  let dataset: Dataset = {};
  if (fs.existsSync(DATASET_PATH)) {
    dataset = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf-8'));
    console.log(`Loaded dataset with ${Object.keys(dataset).length} entries.`);
  } else {
    console.log('Created new dataset.');
  }

  let newPapersCount = 0;
  let errorCount = 0;

  console.log(`=== Starting run starting with seed ${INITIAL_ARXIV_ID} ===`);

  // Step 1: Initial Arxiv ID
  if (!dataset[INITIAL_ARXIV_ID]) {
    const res = await processPaper(INITIAL_ARXIV_ID, ssClient, oaClient, arxivClient, dataset);
    if (res.success) newPapersCount++;
    else if (res.error && !res.error.includes('Skipped')) errorCount++;
  }

  // Step 2 & 3: Expansion
  while (newPapersCount < TARGET_NEW_PAPERS && errorCount < ERROR_LIMIT) {
    const nextBatch = getNextBatch(dataset, TARGET_NEW_PAPERS - newPapersCount);
    if (nextBatch.length === 0) {
      console.log('No more candidates to fetch.');
      break;
    }

    for (const arxivId of nextBatch) {
      if (newPapersCount >= TARGET_NEW_PAPERS || errorCount >= ERROR_LIMIT) break;

      const res = await processPaper(arxivId, ssClient, oaClient, arxivClient, dataset);
      if (res.success) {
        newPapersCount++;
      } else if (res.error && !res.error.includes('Skipped') && res.error !== 'Already attempted') {
        errorCount++;
      }

      if (newPapersCount % SAVE_EVERY === 0) {
        fs.writeFileSync(DATASET_PATH, JSON.stringify(dataset, null, 2));
      }
    }
  }

  fs.writeFileSync(DATASET_PATH, JSON.stringify(dataset, null, 2));
  console.log(`=== Run complete. Added ${newPapersCount} new papers. Total: ${Object.keys(dataset).length}. Errors: ${errorCount}. ===`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
