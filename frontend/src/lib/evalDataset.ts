// The eval dataset the user pastes: checked here, before a 422 from
// /eval/run (or a billed run on items the backend reads differently).
import type { EvalItem } from "@/lib/api";

const DIFFICULTIES = ["easy", "medium", "hard"];
const CATEGORIES = ["in_scope", "out_of_scope"];
// Field names the backend does not read, and the one it does.
const RENAMED: Record<string, string> = { expected: "expected_answer", answer: "expected_answer" };

// Four items of eval/golden/queries.jsonl, which the "Eval Golden" seed
// collection answers; expected_sources are its document titles.
export const STARTER_DATASET: EvalItem[] = [
  {
    id: 1,
    question: "What distance metrics does pgvector support?",
    expected_answer:
      "pgvector supports cosine distance (<=>), L2 Euclidean distance (<->), and inner product (<#>).",
    difficulty: "easy",
    category: "in_scope",
    expected_sources: ["pgvector.md"],
  },
  {
    id: 3,
    question: "What is selectinload used for in SQLAlchemy async?",
    expected_answer:
      "selectinload is an eager loading strategy that loads relationships with an additional query using WHERE id IN (...), avoiding N+1 queries. It is the recommended default in async for one-to-many relationships.",
    difficulty: "medium",
    category: "in_scope",
    expected_sources: ["sqlalchemy-async.md"],
  },
  {
    id: 27,
    question: "What is the difference between IVFFlat and HNSW as indexes in pgvector?",
    expected_answer:
      "IVFFlat partitions the space into clusters and requires training the index with existing data. HNSW (since pgvector 0.5) is a hierarchical graph, offers better recall and speed, and does not require retraining.",
    difficulty: "medium",
    category: "in_scope",
    expected_sources: ["pgvector.md"],
  },
  {
    id: 6,
    question: "What is the capital of France?",
    expected_answer:
      "The question is out of scope for the system — the system should indicate that it does not have enough information.",
    difficulty: "easy",
    category: "out_of_scope",
    expected_sources: [],
  },
];

export type ParsedDataset = { items: EvalItem[]; errors: [] } | { items: null; errors: string[] };

function isNonEmptyString(v: unknown): v is string {
  return typeof v === "string" && v.trim().length > 0;
}

/** The items in `text`, or every problem found in it. */
export function parseEvalDataset(text: string): ParsedDataset {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { items: null, errors: [`not valid JSON: ${e instanceof Error ? e.message : e}`] };
  }
  if (!Array.isArray(raw)) return { items: null, errors: ["the dataset must be a JSON array"] };
  if (raw.length === 0) return { items: null, errors: ["the dataset is empty"] };

  const errors: string[] = [];
  const seen = new Set<number>();
  raw.forEach((item, i) => {
    const at = `item ${i + 1}`;
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      errors.push(`${at}: not an object`);
      return;
    }
    const o = item as Record<string, unknown>;
    for (const [wrong, right] of Object.entries(RENAMED)) {
      if (wrong in o && !(right in o)) errors.push(`${at}: "${wrong}" should be "${right}"`);
    }
    if (!Number.isInteger(o.id)) errors.push(`${at}: "id" must be an integer`);
    else if (seen.has(o.id as number)) errors.push(`${at}: id ${o.id} is repeated`);
    else seen.add(o.id as number);
    if (!isNonEmptyString(o.question)) errors.push(`${at}: "question" is missing`);
    if (typeof o.expected_answer !== "string" && !Object.keys(RENAMED).some((k) => k in o))
      errors.push(`${at}: "expected_answer" is missing`);
    if (!DIFFICULTIES.includes(o.difficulty as string))
      errors.push(`${at}: "difficulty" must be ${DIFFICULTIES.join(", ")}`);
    if (!CATEGORIES.includes(o.category as string))
      errors.push(`${at}: "category" must be ${CATEGORIES.join(" or ")}`);
    if (
      o.expected_sources !== undefined &&
      !(Array.isArray(o.expected_sources) && o.expected_sources.every((s) => typeof s === "string"))
    )
      errors.push(`${at}: "expected_sources" must be a list of document titles`);
  });
  return errors.length > 0 ? { items: null, errors } : { items: raw as EvalItem[], errors: [] };
}
