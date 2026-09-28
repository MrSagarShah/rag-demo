/**
 * eval.ts — a tiny eval harness for the RAG pipeline.
 *
 * Run: npm run eval
 *
 * For each case { question, expectedSource } we measure two things:
 *   1. Retrieval hit  — was `expectedSource` among the top-k retrieved chunks?
 *   2. Citation hit   — did the generated answer actually cite `expectedSource`
 *                       as [source: expectedSource]?
 *
 * This is the difference between an answer that's right and one that's merely
 * plausible: a fluent answer that cites the wrong document (or nothing) is a
 * liability, not a feature. Retrieval hit without citation hit tells you the
 * context was there but the model didn't ground its answer in it — a distinct,
 * fixable failure mode from the retriever simply missing the document.
 */

import { JsonVectorStore } from "./lib/store.js";
import { answerQuestion, STORE_PATH, TOP_K } from "./query.js";

interface EvalCase {
  readonly question: string;
  readonly expectedSource: string;
}

// Questions target the sample docs in docs/. Keep these in sync with that folder.
const CASES: EvalCase[] = [
  { question: "How many days do I have to return an item?", expectedSource: "returns-policy.md" },
  { question: "How long does standard shipping take?", expectedSource: "shipping-policy.md" },
  { question: "What is the API rate limit on the free tier?", expectedSource: "api-rate-limits.md" },
  { question: "Do you offer a free trial of Nimbus Pro?", expectedSource: "faq.md" },
];

interface CaseResult {
  readonly question: string;
  readonly expectedSource: string;
  readonly retrievalHit: boolean;
  readonly citationHit: boolean;
  readonly retrievedSources: string[];
}

/** True if the answer cites the expected source as [source: <filename>] (whitespace-tolerant). */
function citesSource(answer: string, source: string): boolean {
  const pattern = new RegExp(`\\[\\s*source\\s*:\\s*${escapeRegExp(source)}\\s*\\]`, "i");
  return pattern.test(answer);
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function main(): Promise<void> {
  const store = new JsonVectorStore(STORE_PATH);
  if ((await store.size()) === 0) {
    console.error("Store is empty. Run `npm run ingest` first.");
    process.exit(1);
  }

  console.log(`Running ${CASES.length} eval case(s) (top-k = ${TOP_K})…\n`);

  const results: CaseResult[] = [];
  for (const c of CASES) {
    const { answer, retrieved } = await answerQuestion(c.question, store);
    const retrievedSources = retrieved.map((r) => r.record.source);
    results.push({
      question: c.question,
      expectedSource: c.expectedSource,
      retrievalHit: retrievedSources.includes(c.expectedSource),
      citationHit: citesSource(answer, c.expectedSource),
      retrievedSources,
    });
  }

  printTable(results);

  const retrievalScore = results.filter((r) => r.retrievalHit).length;
  const citationScore = results.filter((r) => r.citationHit).length;
  const n = results.length;

  console.log(`\nRetrieval accuracy: ${retrievalScore}/${n} (${pct(retrievalScore, n)})`);
  console.log(`Citation accuracy:  ${citationScore}/${n} (${pct(citationScore, n)})`);

  // A case passes only if it both retrieved AND cited the right source.
  const passed = results.filter((r) => r.retrievalHit && r.citationHit).length;
  console.log(`Overall (retrieval AND citation): ${passed}/${n} (${pct(passed, n)})`);

  // Non-zero exit if any case fails, so this can gate CI.
  process.exit(passed === n ? 0 : 1);
}

function printTable(results: CaseResult[]): void {
  const rows = results.map((r) => ({
    Question: truncate(r.question, 40),
    Expected: r.expectedSource,
    Retrieval: r.retrievalHit ? "PASS" : "FAIL",
    Citation: r.citationHit ? "PASS" : "FAIL",
  }));

  const columns: (keyof (typeof rows)[number])[] = ["Question", "Expected", "Retrieval", "Citation"];
  const widths = columns.map((col) =>
    Math.max(col.length, ...rows.map((row) => row[col].length)),
  );

  const header = columns.map((col, i) => col.padEnd(widths[i] as number)).join("  ");
  console.log(header);
  console.log(widths.map((w) => "-".repeat(w)).join("  "));
  for (const row of rows) {
    console.log(columns.map((col, i) => row[col].padEnd(widths[i] as number)).join("  "));
  }
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max - 1) + "…";
}

function pct(num: number, den: number): string {
  return den === 0 ? "0%" : `${Math.round((num / den) * 100)}%`;
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
