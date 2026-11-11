import { Fragment, useState } from "react";
import { fmtNumber, fmtPct, type EvalResult, type JudgeVerdict } from "@/lib/api";
import { EmptyState } from "@/components/ui-kit/EmptyState";
import { Markdown } from "@/components/chat/Markdown";

type Filter = "all" | "wrong" | "unsupported" | "judge";

const FILTERS: { key: Filter; label: string; match: (r: EvalResult) => boolean }[] = [
  { key: "all", label: "ALL", match: () => true },
  { key: "wrong", label: "WRONG", match: (r) => !r.correctness.passed && !r.correctness.error },
  {
    key: "unsupported",
    label: "UNSUPPORTED",
    match: (r) => !r.hallucination.passed && !r.hallucination.error,
  },
  {
    key: "judge",
    label: "JUDGE ERRORS",
    match: (r) => !!(r.correctness.error || r.hallucination.error),
  },
];

/** Each item's verdicts, and on a click what was expected, what came back and why. */
export function ItemResults({ results }: { results: EvalResult[] }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [open, setOpen] = useState<number | null>(null);
  const counts = Object.fromEntries(
    FILTERS.map((f) => [f.key, results.filter(f.match).length]),
  ) as Record<Filter, number>;
  const shown = results.filter(FILTERS.find((f) => f.key === filter)!.match);

  return (
    <section aria-label="Items">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h3 className="title-sm text-text-primary">ITEMS</h3>
        <div role="group" aria-label="Filter items" className="flex flex-wrap gap-1">
          {FILTERS.filter((f) => f.key !== "judge" || counts.judge > 0).map((f) => (
            <button
              key={f.key}
              type="button"
              aria-pressed={filter === f.key}
              onClick={() => setFilter(f.key)}
              className={`label-xs px-2 h-7 border ${
                filter === f.key
                  ? "border-text-primary text-text-primary"
                  : "border-border-subtle text-text-secondary hover:text-text-primary"
              }`}
            >
              {f.label} · {counts[f.key]}
            </button>
          ))}
        </div>
      </div>

      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {shown.length === 0 ? (
          <EmptyState label={`NO ${FILTERS.find((f) => f.key === filter)!.label} ITEMS`} />
        ) : (
          <table className="w-full min-w-[720px]">
            <thead>
              <tr className="label-sm text-text-secondary border-b border-border-subtle">
                <th className="text-left pl-4 py-2 font-normal w-12">ID</th>
                <th className="text-left py-2 font-normal">QUESTION</th>
                <th className="text-left py-2 font-normal w-24">CORRECT</th>
                <th className="text-left py-2 font-normal w-24">GROUNDED</th>
                <th className="text-right py-2 font-normal w-16">P@K</th>
                <th className="text-right pr-4 py-2 font-normal w-24">LATENCY</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => {
                const expanded = open === r.item_id;
                return (
                  <Fragment key={r.item_id}>
                    <tr className="body-sm border-b border-border-subtle/50 hover:bg-surface-high">
                      <td className="pl-4 py-2 text-text-secondary tabular-nums">{r.item_id}</td>
                      <td className="py-2 pr-4">
                        <button
                          type="button"
                          aria-expanded={expanded}
                          onClick={() => setOpen(expanded ? null : r.item_id)}
                          className="text-left text-text-primary hover:text-primary-blue"
                        >
                          <span className="text-text-tertiary mr-2">{expanded ? "▾" : "▸"}</span>
                          {r.question}
                        </button>
                        <div className="label-xs text-text-tertiary mt-0.5">
                          {r.difficulty.toUpperCase()} ·{" "}
                          {r.category.replace("_", " ").toUpperCase()}
                        </div>
                      </td>
                      <td className="py-2">
                        <Verdict v={r.correctness} pass="PASS" fail="FAIL" />
                      </td>
                      <td className="py-2">
                        <Verdict v={r.hallucination} pass="YES" fail="NO" />
                      </td>
                      <td className="py-2 text-right tabular-nums text-text-secondary">
                        {r.category === "in_scope" ? fmtPct(r.retrieval_precision, 0) : "—"}
                      </td>
                      <td className="py-2 pr-4 text-right tabular-nums text-text-secondary">
                        {fmtNumber(r.latency_ms)}MS
                      </td>
                    </tr>
                    {expanded && (
                      <tr className="border-b border-border-subtle bg-surface-low">
                        <td colSpan={6} className="px-4 py-4">
                          <ItemDetail r={r} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function Verdict({ v, pass, fail }: { v: JudgeVerdict; pass: string; fail: string }) {
  if (v.error) return <span className="label-xs text-warning">JUDGE ERROR</span>;
  return (
    <span className={`label-xs ${v.passed ? "text-success" : "text-error"}`}>
      {v.passed ? `✓ ${pass}` : `✕ ${fail}`}
    </span>
  );
}

function ItemDetail({ r }: { r: EvalResult }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Block
          label={
            r.category === "out_of_scope" ? "EXPECTED · OUT OF SCOPE, SHOULD DECLINE" : "EXPECTED"
          }
          text={r.expected_answer}
        />
        <Block label="SYSTEM ANSWER" text={r.system_answer} markdown />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Reasoning label="CORRECTNESS JUDGE" v={r.correctness} />
        <Reasoning label="GROUNDING JUDGE" v={r.hallucination} />
      </div>
      <div className="label-xs text-text-tertiary">
        RETRIEVED:{" "}
        <span className="text-text-secondary normal-case">
          {r.citations.length > 0 ? r.citations.join(" · ") : "nothing"}
        </span>
        {r.category === "in_scope" && (
          <>
            {" "}
            · PRECISION {fmtPct(r.retrieval_precision, 0)}
            {r.retrieval_recall !== undefined && <> · RECALL {fmtPct(r.retrieval_recall, 0)}</>}
          </>
        )}{" "}
        · {fmtNumber(r.tokens_input)} IN / {fmtNumber(r.tokens_output)} OUT
      </div>
    </div>
  );
}

// `markdown`: the system's answers are markdown, as in the chat.
function Block({ label, text, markdown }: { label: string; text: string; markdown?: boolean }) {
  return (
    <div>
      <div className="label-xs text-text-tertiary mb-1">{label}</div>
      <div
        className={`body-sm text-text-primary break-words border border-border-subtle bg-surface-lowest p-3 max-h-72 overflow-auto ${markdown ? "" : "whitespace-pre-wrap"}`}
      >
        {!text ? "(empty)" : markdown ? <Markdown text={text} /> : text}
      </div>
    </div>
  );
}

function Reasoning({ label, v }: { label: string; v: JudgeVerdict }) {
  const missing = v.facts_missing ?? [];
  const wrong = v.wrong_claims ?? [];
  return (
    <div>
      <div className="label-xs text-text-tertiary mb-1 flex items-center gap-2">
        {label} <Verdict v={v} pass="PASS" fail="FAIL" />
      </div>
      {missing.length > 0 && <Claims title="MISSING FACTS" claims={missing} />}
      {wrong.length > 0 && <Claims title="WRONG OR UNSUPPORTED" claims={wrong} />}
      <div className="body-sm text-text-secondary whitespace-pre-wrap">{v.reasoning}</div>
    </div>
  );
}

function Claims({ title, claims }: { title: string; claims: string[] }) {
  return (
    <div className="mb-2">
      <div className="label-xs text-error mb-1">{title}</div>
      <ul className="body-sm text-text-primary list-disc pl-5">
        {claims.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
    </div>
  );
}
