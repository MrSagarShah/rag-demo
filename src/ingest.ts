/**
 * ingest.ts — read docs/ → chunk → embed → write store.json.
 *
 * Run: npm run ingest
 *
 * Reads every .md/.txt file in docs/, splits each into overlapping chunks,
 * embeds them in a batch, and writes the full corpus to store.json. Each record
 * is { id, source (filename), text, embedding }.
 */

import { readdir, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve, dirname, extname, basename } from "node:path";
import { chunkText } from "./lib/chunk.js";
import { embedBatch } from "./lib/embeddings.js";
import { JsonVectorStore, type ChunkRecord } from "./lib/store.js";
import { STORE_PATH } from "./query.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = resolve(__dirname, "..", "docs");

const SUPPORTED_EXTENSIONS = new Set([".md", ".txt"]);

async function main(): Promise<void> {
  const entries = await readdir(DOCS_DIR, { withFileTypes: true });
  const files = entries
    .filter((e) => e.isFile() && SUPPORTED_EXTENSIONS.has(extname(e.name).toLowerCase()))
    .map((e) => e.name)
    .sort();

  if (files.length === 0) {
    console.error(`No .md/.txt files found in ${DOCS_DIR}`);
    process.exit(1);
  }

  // 1) Read + chunk every file, tracking which source each chunk came from.
  const pending: { id: string; source: string; text: string }[] = [];
  for (const file of files) {
    const content = await readFile(resolve(DOCS_DIR, file), "utf8");
    const chunks = chunkText(content);
    const stem = basename(file, extname(file));
    chunks.forEach((text, i) => {
      pending.push({ id: `${stem}-${i}`, source: file, text });
    });
    console.log(`  ${file}: ${chunks.length} chunk(s)`);
  }

  console.log(`\nEmbedding ${pending.length} chunk(s) with text-embedding-3-small…`);

  // 2) Embed all chunks in one batched request (order preserved).
  const embeddings = await embedBatch(pending.map((p) => p.text));

  // 3) Assemble records and persist.
  const records: ChunkRecord[] = pending.map((p, i) => {
    const embedding = embeddings[i];
    if (!embedding) throw new Error(`Missing embedding for chunk ${p.id}`);
    return { id: p.id, source: p.source, text: p.text, embedding };
  });

  const store = new JsonVectorStore(STORE_PATH);
  await store.add(records);

  console.log(`\nWrote ${records.length} chunk(s) from ${files.length} file(s) to ${STORE_PATH}`);
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
