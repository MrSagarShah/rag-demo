/**
 * store.ts — the vector store.
 *
 * This is a deliberately tiny, zero-setup implementation: a JSON file on disk,
 * loaded fully into memory, with brute-force cosine similarity for search. That
 * is perfect for a demo (a few hundred chunks) and keeps the project runnable
 * with `npm i` and nothing else — no Docker, no database, no API keys for the
 * store itself.
 *
 * The important part is the `VectorStore` INTERFACE. Everything else in the app
 * (ingest, query, eval) depends only on this interface, never on the JSON file.
 * That is exactly where a production vector database slots in:
 *
 *   // --- WHERE PGVECTOR / PINECONE WOULD GO ---
 *   // Implement `VectorStore` against your backend of choice — e.g. a
 *   // `PgVectorStore` running `SELECT ... ORDER BY embedding <=> $1 LIMIT k`
 *   // against Postgres+pgvector, or a `PineconeStore` calling index.query().
 *   // No caller changes: swap `new JsonVectorStore(path)` for the new class.
 */

export interface ChunkRecord {
  readonly id: string;
  readonly source: string; // filename the chunk came from
  readonly text: string;
  readonly embedding: number[];
}

/** A scored search hit: the stored record plus its cosine similarity to the query. */
export interface ScoredChunk {
  readonly record: ChunkRecord;
  readonly score: number;
}

export interface VectorStore {
  /** Replace all records (ingest writes the full corpus in one shot). */
  add(records: ChunkRecord[]): Promise<void>;
  /** Return the top-`k` records most similar to `queryEmbedding`, best first. */
  search(queryEmbedding: number[], k: number): Promise<ScoredChunk[]>;
  /** Total number of stored chunks. */
  size(): Promise<number>;
}

import { readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";

/**
 * JSON-file-backed VectorStore with in-memory cosine search.
 * The file is loaded lazily on first access and re-read is cheap for demo sizes.
 */
export class JsonVectorStore implements VectorStore {
  private readonly path: string;
  private records: ChunkRecord[] | null = null;

  constructor(path: string) {
    this.path = path;
  }

  private async load(): Promise<ChunkRecord[]> {
    if (this.records !== null) return this.records;
    if (!existsSync(this.path)) {
      this.records = [];
      return this.records;
    }
    const raw = await readFile(this.path, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      throw new Error(`Corrupt store at ${this.path}: expected a JSON array of chunk records.`);
    }
    this.records = parsed as ChunkRecord[];
    return this.records;
  }

  async add(records: ChunkRecord[]): Promise<void> {
    this.records = records;
    await writeFile(this.path, JSON.stringify(records, null, 2), "utf8");
  }

  async search(queryEmbedding: number[], k: number): Promise<ScoredChunk[]> {
    const records = await this.load();
    const scored: ScoredChunk[] = records.map((record) => ({
      record,
      score: cosineSimilarity(queryEmbedding, record.embedding),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  async size(): Promise<number> {
    return (await this.load()).length;
  }
}

/**
 * Cosine similarity of two equal-length vectors. Returns 0 for a zero vector
 * (avoids NaN). OpenAI embeddings are already L2-normalized, so this reduces to
 * a dot product in practice — we compute it in full so the store works with any
 * embedding provider.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`vector length mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] as number;
    const bi = b[i] as number;
    dot += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
