# rag-demo

A small, runnable TypeScript RAG (retrieval-augmented generation) app: it ingests local documents, retrieves the relevant passages for a question, and answers **with inline citations** — plus a tiny eval that checks whether each answer cites the *correct* source.

It intentionally uses two providers — **OpenAI** for embeddings (`text-embedding-3-small`) and **Anthropic Claude** for generation — because that's how real systems are often built: pick the best tool for each job.

## Architecture

```
docs/*.md ──▶ chunk ──▶ embed (OpenAI) ──▶ store.json (vector store)
                                                 │
question ──▶ embed ──▶ cosine top-k retrieve ◀───┘
                          │
                          ▼
              grounded prompt (context + rules)
                          │
                          ▼
              generate (Claude) ──▶ answer + [source: filename] citations

                          ▲
                          │
              eval loop: for each {question, expectedSource}
              → retrieval hit?  (is expectedSource in top-k?)
              → citation hit?   (does the answer cite expectedSource?)
```

Modules:

| File | Responsibility |
| --- | --- |
| `src/ingest.ts` | Read `docs/`, chunk, embed, write `store.json` |
| `src/query.ts` | Embed question → retrieve top-k → generate cited answer (also exports the reusable functions) |
| `src/eval.ts` | Run the eval cases and print a pass/fail table + aggregate score |
| `src/lib/chunk.ts` | Character-based chunker with overlap |
| `src/lib/store.ts` | `VectorStore` interface + JSON-file implementation with cosine search |
| `src/lib/embeddings.ts` | OpenAI embedding calls, behind small functions |

## Design choices (and why)

- **Local JSON vector store behind a `VectorStore` interface.** The demo needs zero setup — no Docker, no database — so `store.json` + in-memory cosine similarity is plenty for a few hundred chunks. But every caller depends only on the `VectorStore` interface, so moving to **pgvector** or **Pinecone** is a single new class and one changed line at construction. `src/lib/store.ts` marks the exact spot.
- **Cite only from context.** The system prompt forbids outside knowledge and requires an inline `[source: filename]` citation for every claim, with a fixed "I don't have enough information" fallback. Grounded, auditable answers are the whole point of RAG; a fluent guess is worse than an honest "I don't know."
- **An eval that separates *correct* from *merely plausible*.** The harness measures two independent things: did we **retrieve** the right document, and did the answer actually **cite** it? An answer can read beautifully and still cite the wrong doc — that's a liability, not a feature. Splitting retrieval hits from citation hits also localizes failures: a retrieval miss is a retriever/chunking problem; a citation miss with a retrieval hit means the context was present but the model didn't ground its answer in it.

## Setup and run

```bash
npm i

# set the two keys (or copy .env.example to .env and export them)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

npm run ingest                                  # build store.json from docs/
npm run query -- "what is the return window?"   # ask a question
npm run eval                                     # run the eval suite
```

Type-check only (no keys needed):

```bash
npm run typecheck   # tsc --noEmit, strict
```

### Swapping the generation model

`query.ts` sets `GENERATION_MODEL = "claude-sonnet-5"`. Change that one constant to any current Claude model id — e.g. `claude-opus-5` for higher quality or `claude-haiku-4-5` for lower cost. (Model ids evolve; use whatever is current for your account.)

## Example

```
Q: what is the return window?

Answer:

You have 30 days from the date of delivery to return most items for a full
refund. [source: returns-policy.md]

------------------------------------------------------------
Retrieved 4 chunks (top-4):
  • returns-policy.md  (score 0.612)  "Return window ..."
  • shipping-policy.md (score 0.331)  "..."
  ...
```

## How I'd productionize this

- **Swap the JSON store for pgvector** (or Pinecone/Qdrant). The `VectorStore` interface already isolates this; add an ANN index and metadata filtering so retrieval scales past in-memory brute force.
- **Add a reranker.** Retrieve a wider top-k (e.g. 20) with the embedding model, then rerank with a cross-encoder / rerank API and keep the best 4. Embedding recall + rerank precision beats either alone.
- **Stream the generation.** Stream tokens to the client for responsiveness on longer answers, and to avoid HTTP timeouts on large outputs.
- **Real tokenization + smarter chunking.** Replace the char-based chunker with a token-aware, structure-aware splitter (respect headings, tables, code blocks).
- **Per-answer eval logging + regression gates.** Log retrieved chunk ids, scores, the model, and citation outcomes for every production answer; expand the eval set from these logs and gate deploys on it in CI (the harness already exits non-zero on failure). Add faithfulness/groundedness scoring (LLM-as-judge) on top of the citation check.
- **Guardrails.** Cache embeddings, add retries/backoff on both providers, and enforce a hard "answer only from context" check before returning.

## License

MIT — see [LICENSE](./LICENSE).
