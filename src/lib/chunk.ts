/**
 * chunk.ts — split a document into overlapping chunks for embedding.
 *
 * Design tradeoff: we chunk by CHARACTER count, not real tokens. A proper
 * tokenizer (e.g. tiktoken / @anthropic-ai/tokenizer) would give exact token
 * boundaries, but it adds a dependency and startup cost for little gain in a
 * demo. We approximate ~4 chars per token, so a ~600-token target ≈ 2400 chars.
 * The cost of the approximation is that a chunk occasionally runs a little over
 * or under the model's ideal window — acceptable here, and trivially swappable
 * for a token-based splitter later.
 *
 * We also prefer to break on paragraph/sentence boundaries when one is nearby,
 * so chunks stay semantically coherent instead of slicing mid-sentence.
 */

export interface ChunkOptions {
  /** Target chunk size in characters (~600 tokens). */
  readonly size: number;
  /** Overlap between consecutive chunks in characters, to preserve context across boundaries. */
  readonly overlap: number;
}

export const DEFAULT_CHUNK_OPTIONS: ChunkOptions = {
  size: 2400, // ~600 tokens
  overlap: 300, // ~75 tokens of context carried across the boundary
};

/**
 * Split `text` into overlapping chunks. Empty/whitespace-only input yields no chunks.
 */
export function chunkText(text: string, options: ChunkOptions = DEFAULT_CHUNK_OPTIONS): string[] {
  const { size, overlap } = options;
  if (size <= 0) throw new Error("chunk size must be > 0");
  if (overlap < 0 || overlap >= size) throw new Error("overlap must be >= 0 and < size");

  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (normalized.length === 0) return [];
  if (normalized.length <= size) return [normalized];

  const chunks: string[] = [];
  let start = 0;

  while (start < normalized.length) {
    let end = Math.min(start + size, normalized.length);

    // If we're not at the very end, try to end on a natural boundary within the
    // last ~20% of the window (paragraph break preferred, then sentence, then space).
    if (end < normalized.length) {
      const windowStart = start + Math.floor(size * 0.8);
      const boundary = findBoundary(normalized, windowStart, end);
      if (boundary > start) end = boundary;
    }

    const piece = normalized.slice(start, end).trim();
    if (piece.length > 0) chunks.push(piece);

    if (end >= normalized.length) break;
    start = end - overlap; // step forward, keeping `overlap` chars of context
  }

  return chunks;
}

/** Find the best break point in [from, to], preferring paragraph > sentence > word breaks. */
function findBoundary(text: string, from: number, to: number): number {
  const slice = text.slice(from, to);

  const paragraph = slice.lastIndexOf("\n\n");
  if (paragraph !== -1) return from + paragraph + 2;

  const sentence = Math.max(slice.lastIndexOf(". "), slice.lastIndexOf("\n"));
  if (sentence !== -1) return from + sentence + 1;

  const space = slice.lastIndexOf(" ");
  if (space !== -1) return from + space + 1;

  return to;
}
