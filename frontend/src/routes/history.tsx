import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { api, fmtCost, fmtDate, fmtNumber, truncate, type LogSource } from "@/lib/api";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { SourceTag } from "@/components/ui-kit/SourceTag";
import { useDebounced } from "@/lib/useDebounced";
import { toneText } from "@/lib/metricTone";
import { LoadMore } from "@/components/ui-kit/LoadMore";
import { PAGE_SIZE, usePagedList } from "@/lib/paged";
import { traceUrl } from "@/lib/traceUrl";
import { PROMPTS_HINT } from "@/lib/promptVersion";
import {
  EmptyState,
  ErrorBanner,
  LoadingLine,
  emptyActionCls,
} from "@/components/ui-kit/EmptyState";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "History — WhisperDocs" },
      {
        name: "description",
        content: "Query log: every past question, answer, and pipeline metric.",
      },
    ],
  }),
  component: HistoryPage,
});

const fieldCls =
  "bg-surface-lowest border border-border-subtle px-2 h-9 text-text-primary body-sm focus:outline-none focus:border-border-focus";

// The placeholders the backend writes in place of personal data
// (app/services/pii.py).
const REDACTED = /\[(email|phone|card|cbu|cuit|dni)\]/;

/** The turn's trace id, and a link to it when VITE_TRACE_URL is set. */
function Trace({ id }: { id: string }) {
  const url = traceUrl(id);
  return (
    <>
      <div className="label-sm text-text-secondary mt-4 mb-2">TRACE</div>
      <div className="body-sm text-text-primary break-all select-all">{id}</div>
      {url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-block mt-2 label-sm text-text-primary hover:text-primary-blue"
        >
          OPEN TRACE ↗
        </a>
      )}
    </>
  );
}

function HistoryPage() {
  const [collFilter, setCollFilter] = useState<number | null>(null);
  const [source, setSource] = useState<LogSource | null>(null);
  // The search asks the server once the typing pauses.
  const [search, setSearch] = useState("");
  const searchQ = useDebounced(search.trim(), 300);
  const [expanded, setExpanded] = useState<number | null>(null);
  const filtering = collFilter !== null || source !== null || searchQ !== "";

  const collections = useQuery({ queryKey: ["collections"], queryFn: api.listCollections });
  // Filtered by the server: a filter over the loaded pages would miss older rows.
  const history = usePagedList(
    ["history", "pages", { collection: collFilter, source, q: searchQ }],
    (cursor) =>
      api.historyPage({
        collection_id: collFilter ?? undefined,
        source: source ?? undefined,
        q: searchQ || undefined,
        limit: PAGE_SIZE,
        cursor,
      }),
  );
  const rows = history.items;

  function clearFilters() {
    setCollFilter(null);
    setSource(null);
    setSearch("");
  }

  const collectionName = (id: number | null) =>
    id === null ? "—" : (collections.data?.find((c) => c.id === id)?.name ?? `#${id}`);

  return (
    <div>
      <PageHeader
        title="HISTORY"
        subtitle="QUERY LOG"
        actions={
          <>
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              maxLength={200}
              placeholder="SEARCH QUESTIONS"
              aria-label="Search questions"
              className={`${fieldCls} w-48 placeholder:text-text-tertiary placeholder:label-xs`}
            />
            <label className="flex items-center gap-2 label-sm text-text-secondary">
              SOURCE:
              <select
                value={source ?? ""}
                onChange={(e) => setSource((e.target.value || null) as LogSource | null)}
                className={fieldCls}
              >
                <option value="">ALL</option>
                <option value="chat">CHAT</option>
                <option value="query">QUERY</option>
              </select>
            </label>
            <label className="flex items-center gap-2 label-sm text-text-secondary">
              COLLECTION:
              <select
                value={collFilter ?? ""}
                onChange={(e) => setCollFilter(e.target.value ? Number(e.target.value) : null)}
                className={fieldCls}
              >
                <option value="">ALL</option>
                {collections.data?.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            </label>
          </>
        }
      />

      {history.query.error && (
        <ErrorBanner
          message="failed to load query history"
          onRetry={() => void history.query.refetch()}
          retrying={history.query.isFetching}
        />
      )}

      <div className="border border-border-subtle bg-surface-container overflow-x-auto">
        {history.query.isLoading ? (
          <div className="p-4">
            <LoadingLine />
          </div>
        ) : rows.length === 0 ? (
          filtering ? (
            <EmptyState label="NO QUERIES MATCH THE FILTERS">
              <button type="button" onClick={clearFilters} className={emptyActionCls}>
                CLEAR FILTERS
              </button>
            </EmptyState>
          ) : (
            <EmptyState label="NO QUERIES LOGGED">
              <Link to="/chat" className={emptyActionCls}>
                ASK A QUESTION →
              </Link>
            </EmptyState>
          )
        ) : (
          <table className="w-full min-w-[840px]">
            <thead>
              <tr className="label-sm text-text-secondary border-b border-border-subtle">
                <th className="text-left pl-6 py-3 font-normal">TIMESTAMP</th>
                <th className="text-left py-3 font-normal">SOURCE</th>
                <th className="text-left py-3 font-normal">QUERY</th>
                <th className="text-left py-3 font-normal">MODEL</th>
                <th className="text-right py-3 font-normal">TOKENS IN/OUT</th>
                <th className="text-right py-3 font-normal">LATENCY</th>
                <th className="text-right pr-6 py-3 font-normal">COST</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((q) => {
                const open = expanded === q.id;
                return (
                  <Fragment key={q.id}>
                    <tr
                      onClick={() => setExpanded(open ? null : q.id)}
                      className="body-sm h-9 border-b border-border-subtle/50 hover:bg-surface-high cursor-pointer"
                    >
                      <td className="pl-6 text-text-secondary tabular-nums">
                        {fmtDate(q.created_at)}
                      </td>
                      <td>
                        <SourceTag action={q.agent_action} />
                      </td>
                      <td className="text-text-primary">
                        {/* The row opens on a click; the button, by keyboard. */}
                        <button
                          type="button"
                          aria-expanded={open}
                          onClick={(e) => {
                            e.stopPropagation();
                            setExpanded(open ? null : q.id);
                          }}
                          className="text-left hover:text-primary-blue"
                        >
                          {truncate(q.question, 70)}
                        </button>
                      </td>
                      <td className="text-text-secondary">{q.model_used}</td>
                      <td className="text-right tabular-nums text-text-primary">
                        {fmtNumber(q.tokens_input)} / {fmtNumber(q.tokens_output)}
                      </td>
                      <td
                        className={`text-right tabular-nums ${toneText("latency", q.latency_ms)}`}
                      >
                        {fmtNumber(q.latency_ms)}MS
                      </td>
                      <td className="text-right pr-6 tabular-nums text-text-primary">
                        {fmtCost(q.cost_usd)}
                      </td>
                    </tr>
                    {open && (
                      <tr className="bg-surface-low">
                        <td colSpan={7} className="p-6 border-b border-border-subtle">
                          <div className="grid grid-cols-2 gap-6">
                            <div>
                              <div className="label-sm text-text-secondary mb-2">QUERY</div>
                              <div className="body-sm text-text-primary whitespace-pre-wrap">
                                {q.question}
                              </div>
                              {(REDACTED.test(q.question) || REDACTED.test(q.answer)) && (
                                <div className="label-xs text-text-tertiary mt-2">
                                  THE LOG REPLACES EMAILS, PHONE NUMBERS AND ID NUMBERS WITH
                                  PLACEHOLDERS, SUCH AS <span className="normal-case">[email]</span>
                                  .
                                </div>
                              )}
                              <div className="label-sm text-text-secondary mt-4 mb-2">
                                COLLECTION
                              </div>
                              <div className="body-sm text-text-primary">
                                {collectionName(q.collection_id)}
                              </div>
                              {q.agent_action && q.session_id && (
                                <Link
                                  to="/chat"
                                  search={{ thread: q.session_id }}
                                  className="inline-block mt-4 label-sm text-text-primary hover:text-primary-blue"
                                >
                                  OPEN THREAD {q.session_id.slice(0, 8)} →
                                </Link>
                              )}
                              <div className="label-sm text-text-secondary mt-4 mb-2">
                                CHUNKS RETRIEVED
                              </div>
                              <div className="body-sm text-text-primary tabular-nums">
                                {q.chunks_retrieved}
                              </div>
                              {q.prompt_version && (
                                <>
                                  <div
                                    className="label-sm text-text-secondary mt-4 mb-2"
                                    title={PROMPTS_HINT}
                                  >
                                    PROMPTS
                                  </div>
                                  <div className="body-sm text-text-primary">
                                    {q.prompt_version}
                                  </div>
                                </>
                              )}
                              {q.trace_id && <Trace id={q.trace_id} />}
                            </div>
                            <div>
                              <div className="label-sm text-text-secondary mb-2">RESPONSE</div>
                              <pre className="body-sm text-text-primary whitespace-pre-wrap bg-surface-lowest p-3 border border-border-subtle max-h-80 overflow-auto">
                                {q.answer}
                              </pre>
                            </div>
                          </div>
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
      <LoadMore
        shown={rows.length}
        total={history.total}
        hasMore={history.query.hasNextPage}
        loading={history.query.isFetchingNextPage}
        onLoad={() => void history.query.fetchNextPage()}
      />
    </div>
  );
}
