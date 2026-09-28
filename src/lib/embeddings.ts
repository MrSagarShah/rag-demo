/**
 * embeddings.ts — turn text into vectors with OpenAI `text-embedding-3-small`.
 *
 * We keep this behind a couple of small functions so the rest of the app never
 * touches the OpenAI client directly — the same reasoning as the VectorStore
 * interface. Swapping the embedding provider (Cohere, Voyage, a local model)
 * means changing only this file.
 */

import OpenAI from "openai";

export const EMBEDDING_MODEL = "text-embedding-3-small";

let client: OpenAI | null = null;

/** Lazily construct the client so importing this module never requires a key. */
function getClient(): OpenAI {
  if (client) return client;
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.");
  }
  client = new OpenAI();
  return client;
}

/**
 * Embed a batch of strings in a single request. Order is preserved: the i-th
 * output vector corresponds to the i-th input string.
 */
export async function embedBatch(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const response = await getClient().embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts,
  });
  // The API returns results with an `index` field; sort defensively before mapping.
  return response.data
    .slice()
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);
}

/** Embed a single string (e.g. a user question). */
export async function embedOne(text: string): Promise<number[]> {
  const [vector] = await embedBatch([text]);
  if (!vector) throw new Error("Embedding request returned no vector.");
  return vector;
}
