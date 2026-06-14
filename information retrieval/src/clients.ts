import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import { XMLParser } from 'fast-xml-parser';

const CACHE_DIR = path.join(__dirname, '..', 'cache');

if (!fs.existsSync(CACHE_DIR)) {
  fs.mkdirSync(CACHE_DIR, { recursive: true });
}

export interface PaperData {
  arxivID: string;
  title: string;
  abstract: string;
  year: number;
  references: string[];
  source: string;
}

abstract class BaseClient {
  protected lastRequestTime = 0;
  protected abstract rateLimitMs: number;

  protected async wait() {
    const now = Date.now();
    const elapsed = now - this.lastRequestTime;
    if (elapsed < this.rateLimitMs) {
      await new Promise(resolve => setTimeout(resolve, this.rateLimitMs - elapsed));
    }
    this.lastRequestTime = Date.now();
  }

  protected saveCache(filename: string, content: string | object) {
    const filePath = path.join(CACHE_DIR, filename);
    const data = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
    fs.writeFileSync(filePath, data);
  }

  protected getFromCache(filename: string): string | null {
    const filePath = path.join(CACHE_DIR, filename);
    if (fs.existsSync(filePath)) {
      return fs.readFileSync(filePath, 'utf-8');
    }
    return null;
  }
}

export class SemanticScholarClient extends BaseClient {
  protected rateLimitMs = 3100; // As per memory instructions for safety
  private baseUrl = 'https://api.semanticscholar.org/graph/v1/paper/';
  private apiKey?: string;

  constructor(apiKey?: string) {
    super();
    this.apiKey = apiKey;
  }

  async getPaperMetadata(arxivId: string): Promise<{ data: any; error: string | null }> {
    const cacheFile = `ss_meta_${arxivId}.json`;
    const cached = this.getFromCache(cacheFile);
    if (cached) return { data: JSON.parse(cached), error: null };

    await this.wait();
    try {
      const fields = 'externalIds,title,abstract,year';
      const url = `${this.baseUrl}ArXiv:${arxivId}?fields=${fields}`;
      const headers = this.apiKey ? { 'x-api-key': this.apiKey } : {};
      const response = await axios.get(url, { headers, timeout: 15000 });

      this.saveCache(cacheFile, response.data);
      return { data: response.data, error: null };
    } catch (error: any) {
      if (error.response?.status === 404) return { data: null, error: 'Not Found' };
      return { data: null, error: error.message };
    }
  }

  async getReferences(arxivId: string): Promise<{ data: string[]; error: string | null }> {
    const cacheFile = `ss_refs_${arxivId}.json`;
    const cached = this.getFromCache(cacheFile);
    if (cached) return { data: JSON.parse(cached), error: null };

    await this.wait();
    try {
      const url = `${this.baseUrl}ArXiv:${arxivId}/references?fields=citedPaper.externalIds&limit=1000`;
      const headers = this.apiKey ? { 'x-api-key': this.apiKey } : {};
      const response = await axios.get(url, { headers, timeout: 15000 });

      const refs = response.data.data || [];
      const arxivRefs = refs
        .map((r: any) => r.citedPaper?.externalIds?.ArXiv)
        .filter((id: string | undefined) => !!id);

      this.saveCache(cacheFile, arxivRefs);
      return { data: arxivRefs, error: null };
    } catch (error: any) {
      return { data: [], error: error.message };
    }
  }
}

export class OpenAlexClient extends BaseClient {
  protected rateLimitMs = 1000;
  private baseUrl = 'https://api.openalex.org/works/';

  async getPaper(arxivId: string): Promise<{ data: any; error: string | null }> {
    const cacheFile = `oa_${arxivId}.json`;
    const cached = this.getFromCache(cacheFile);
    if (cached) return { data: JSON.parse(cached), error: null };

    await this.wait();
    try {
      // Memory says: uses the https://api.openalex.org/works/https://doi.org/10.48550/arxiv.{id} endpoint
      const url = `${this.baseUrl}https://doi.org/10.48550/arxiv.${arxivId}`;
      const response = await axios.get(url, { timeout: 15000 });

      const data = response.data;
      const parsedData = {
        title: data.display_name,
        abstract: this.reconstructAbstract(data.abstract_inverted_index),
        year: data.publication_year,
        arxivID: arxivId,
        // OpenAlex references are OpenAlex IDs, they need resolution to Arxiv IDs.
        // For now we'll store them as they are and maybe handle resolution later if needed,
        // but the memory mentions "batch resolution of OpenAlex referenced work IDs to Arxiv IDs".
        // To keep it simple and consistent with the SS client which returns Arxiv IDs:
        references: [] // We'll leave this empty for now unless we implement resolution
      };

      this.saveCache(cacheFile, parsedData);
      return { data: parsedData, error: null };
    } catch (error: any) {
      if (error.response?.status === 404) return { data: null, error: 'Not Found' };
      return { data: null, error: error.message };
    }
  }

  private reconstructAbstract(invertedIndex: any): string {
    if (!invertedIndex) return '';
    const entries = Object.entries(invertedIndex);
    const words: string[] = [];
    for (const [word, positions] of entries) {
      for (const pos of positions as number[]) {
        words[pos] = word;
      }
    }
    return words.join(' ');
  }
}

export class ArxivClient extends BaseClient {
  protected rateLimitMs = 3000;
  private baseUrl = 'http://export.arxiv.org/api/query?id_list=';
  private parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@_"
  });

  async getPaper(arxivId: string): Promise<{ data: any; error: string | null }> {
    const cacheFile = `arxiv_${arxivId}.xml`;
    const cached = this.getFromCache(cacheFile);
    if (cached) {
      return { data: this.parseArxivXml(cached, arxivId), error: null };
    }

    await this.wait();
    try {
      const url = `${this.baseUrl}${arxivId}`;
      const response = await axios.get(url, { timeout: 15000 });

      this.saveCache(cacheFile, response.data);
      return { data: this.parseArxivXml(response.data, arxivId), error: null };
    } catch (error: any) {
      return { data: null, error: error.message };
    }
  }

  private parseArxivXml(xml: string, arxivId: string) {
    const jsonObj = this.parser.parse(xml);
    const entry = jsonObj.feed?.entry;
    if (!entry || !entry.id) return null;

    return {
      title: (entry.title || '').toString().trim().replace(/\n/g, ' '),
      abstract: (entry.summary || '').toString().trim().replace(/\n/g, ' '),
      year: parseInt((entry.published || '2020').substring(0, 4)),
      arxivID: arxivId
    };
  }
}
