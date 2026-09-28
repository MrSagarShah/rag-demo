/**
 * query.ts — answer a question over the ingested corpus, WITH citations.
 *
 * Pipeline: embed the question → cosine top-k retrieve → build a grounded
 * prompt → ask Claude to answer ONLY from the provided context and cite each
 * source as [source: filename] → print the answer and the retrieved chunks.
 *
 * Run: npm run query -- "what is the return window?"
 *
 * The functions here (`retrieve`, `generateAnswer`, `answerQuestion`) are the
 * public surface the eval harness reuses — the CLI is just a thin wrapper.
 */

import Anthropic from "@anthropic-ai/sdk";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";
import { embedOne } from "./lib/embeddings.js";
import { JsonVectorStore, type ScoredChunk, type VectorStore } from "./lib/store.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

/** Path to the on-disk vector store, at the project root. */
export const STORE_PATH = resolve(__dirname, "..", "store.json");

/** Number of chunks to retrieve per question. */
export const TOP_K = 4;

/**
 * Claude model used for generation. `claude-sonnet-5` is a current, capable
 * model at the time of writing; swap it for any current model id (e.g.
 * `claude-opus-5` for higher quality, `claude-haiku-4-5` for lower cost) —
 * this is the only line you touch to change the generator.
 */
export const GENERATION_MODEL = "claude-sonnet-5";

let anthropic: Anthropic | null = null;

function getAnthropic(): Anthropic {
  if (anthropic) return anthropic;
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.");
  }
  anthropic = new Anthropic();
  return anthropic;
}

/** Embed the question and return the top-k most similar chunks. */
export async function retrieve(
  question: string,
  store: VectorStore,
  k: number = TOP_K,
): Promise<ScoredChunk[]> {
  const queryEmbedding = await embedOne(question);
  return store.search(queryEmbedding, k);
}

const SYSTEM_PROMPT = `You are a precise support assistant. Answer the user's question using ONLY the provided context passages. Follow these rules strictly:

- Use only facts stated in the context. Do not use outside knowledge or make assumptions.
- If the context does not contain the answer, say exactly: "I don't have enough information to answer that." and nothing more.
- Cite every claim inline with the filename it came from, in the form [source: filename]. If several passages support a sentence, cite each.
- Be concise and direct.`;

/**
 * Build the context block the model sees. Each passage is labelled with its
 * source filename so the model can cite it — and so a reader can audit exactly
 * what the answer was grounded in.
 */
export function buildContext(chunks: ScoredChunk[]): string {
  return chunks
    .map((c, i) => `[Passage ${i + 1} | source: ${c.record.source}]\n${c.record.text}`)
    .join("\n\n---\n\n");
}

/** Ask Claude to answer strictly from the retrieved context. */
export async function generateAnswer(question: string, chunks: ScoredChunk[]): Promise<string> {
  const context = buildContext(chunks);
  const userMessage = `Context passages:\n\n${context}\n\n---\n\nQuestion: ${question}`;

  const response = await getAnthropic().messages.create({
    model: GENERATION_MODEL,
    max_tokens: 1024,
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: userMessage }],
  });

  // Concatenate all text blocks in the response (ignoring any non-text blocks).
  return response.content
    .filter((block): block is Anthropic.TextBlock => block.type === "text")
    .map((block) => block.text)
    .join("")
    .trim();
}

export interface AnswerResult {
  readonly answer: string;
  readonly retrieved: ScoredChunk[];
}

/** End-to-end: retrieve then generate. Reused by the eval harness. */
export async function answerQuestion(
  question: string,
  store: VectorStore,
  k: number = TOP_K,
): Promise<AnswerResult> {
  const retrieved = await retrieve(question, store, k);
  const answer = await generateAnswer(question, retrieved);
  return { answer, retrieved };
}

async function main(): Promise<void> {
  const question = process.argv.slice(2).join(" ").trim();
  if (!question) {
    console.error('Usage: npm run query -- "your question here"');
    process.exit(1);
  }

  const store = new JsonVectorStore(STORE_PATH);
  if ((await store.size()) === 0) {
    console.error("Store is empty. Run `npm run ingest` first.");
    process.exit(1);
  }

  const { answer, retrieved } = await answerQuestion(question, store);

  console.log(`\nQ: ${question}\n`);
  console.log("Answer:\n");
  console.log(answer);
  console.log("\n" + "-".repeat(60));
  console.log(`Retrieved ${retrieved.length} chunks (top-${TOP_K}):`);
  for (const { record, score } of retrieved) {
    const preview = record.text.replace(/\s+/g, " ").slice(0, 90);
    console.log(`  • ${record.source}  (score ${score.toFixed(3)})  "${preview}…"`);
  }
  console.log();
}

// Only run the CLI when this file is the entrypoint, not when imported by eval.ts.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
