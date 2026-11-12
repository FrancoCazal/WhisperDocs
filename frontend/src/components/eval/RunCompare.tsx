import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  api,
  fmtCost,
  fmtDate,
  fmtNumber,
  fmtPct,
  type EvalReport,
  type EvalRunSummary,
  type RunDiff,
} from "@/lib/api";
import {
  checkLabel,
  fmtP,
  fmtSetting,
  metricDeltas,
  splitTicks,
  trend,
  versionChanged,
  type MetricDelta,
  type VersionPair,
} from "@/lib/evalDiff";
import { PROMPTS_HINT, fmtVersion } from "@/lib/promptVersion";
import { ErrorBanner, LoadingLine } from "@/components/ui-kit/EmptyState";

const short = (runId: string) => runId.slice(0, 8);
// The list says "+00:00" and a full report "Z": compare instants, not strings.
const at = (iso: string) => Date.parse(iso);

const MCNEMAR_HINT =
  "McNemar's exact test reads only the items whose verdict flipped. Below 0.05, the flips lean one way more than chance would; with few flips, it can say little.";

/** The shown run against another one, as `runner diff` compares them: what
 * makes the difference hard to read, what changed, the metrics, and the
 * items that flipped. */
export function RunCompare({ report, runs }: { report: EvalReport; runs: EvalRunSummary[] }) {
  const others = runs.filter((r) => r.run_id !== report.run_id);
  // By default, the run before this one.
  const previous =
    others.find((r) => at(r.timestamp) < at(report.timestamp))?.run_id ?? others[0]?.run_id ?? null;
  const [picked, setPicked] = useState<string | null>(null);
  const otherId = picked !== null && others.some((r) => r.run_id === picked) ? picked : previous;
  const other = others.find((r) => r.run_id === otherId);

  // Older first, so a regression is always something that got worse over time.
  const pair =
    other &&
    (at(other.timestamp) <= at(report.timestamp)
      ? { prev: other, curr: report }
      : { prev: report, curr: other });

  const diff = useQuery({
    queryKey: ["eval-diff", pair?.prev.run_id, pair?.curr.run_id],
    queryFn: () => api.evalDiff(pair!.prev.run_id, pair!.curr.run_id),
    enabled: pair !== undefined,
    // Saved reports never change.
    staleTime: Infinity,
  });

  if (others.length === 0) return null;

  return (
    <section aria-label="Compare runs" className="mb-8">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h3 className="title-sm text-text-primary">COMPARE</h3>
        {/* A run's label is long: on a phone the select shrinks to fit. */}
        <label className="flex items-center gap-2 min-w-0 max-w-full label-sm text-text-secondary">
          WITH
          <select
            value={otherId ?? ""}
            onChange={(e) => setPicked(e.target.value)}
            className="min-w-0 bg-surface-lowest border border-border-subtle px-2 h-8 text-text-primary body-sm focus:outline-none focus:border-border-focus"
          >
            {others.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {short(r.run_id)} · {fmtDate(r.timestamp)} · {r.dataset_size} ITEMS
              </option>
            ))}
          </select>
        </label>
      </div>

      {diff.error ? (
        <ErrorBanner
          message={`failed to compare with ${short(otherId ?? "")}: ${diff.error.message}`}
          onRetry={() => void diff.refetch()}
          retrying={diff.isFetching}
        />
      ) : !pair || !diff.data ? (
        <LoadingLine />
      ) : (
        <Comparison
          diff={diff.data}
          prompts={{
            prev: pair.prev.prompt_version ?? null,
            curr: pair.curr.prompt_version ?? null,
          }}
          sizes={{ prev: pair.prev.dataset_size, curr: pair.curr.dataset_size }}
        />
      )}
    </section>
  );
}

function Comparison({
  diff,
  prompts,
  sizes,
}: {
  diff: RunDiff;
  prompts: VersionPair;
  sizes: { prev: number; curr: number };
}) {
  const regressions = diff.flips.filter((f) => f.regression);
  const fixes = diff.flips.filter((f) => !f.regression);
  return (
    <>
      <div className="label-xs text-text-tertiary mb-3">
        {short(diff.previous)} → {short(diff.current)} ·{" "}
        {diff.common_items === 0
          ? "NO ITEMS IN COMMON: ONLY THE TOTALS COMPARE"
          : `${diff.common_items} ITEMS IN COMMON`}
        {diff.common_items > 0 && !diff.same_items && (
          <span className="text-warning">
            {" "}
            · THE TOTALS COVER DIFFERENT ITEMS
            {sizes.prev !== sizes.curr && ` (${sizes.prev} VS ${sizes.curr})`}
          </span>
        )}
      </div>
      <div className="label-xs text-text-tertiary mb-3" title={PROMPTS_HINT}>
        PROMPTS:{" "}
        <span className={versionChanged(prompts) ? "text-text-primary" : ""}>
          <Versions v={prompts} />
        </span>
      </div>

      {diff.warnings.length > 0 && (
        <ul aria-label="Warnings" className="space-y-2 mb-4">
          {diff.warnings.map((w) => (
            <li key={w} className="body-sm text-warning border border-warning/40 px-3 py-2">
              <Ticks text={w} />
            </li>
          ))}
        </ul>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <Changed diff={diff} />
        <Flips diff={diff} />
      </div>

      <div className="border border-border-subtle bg-surface-container overflow-x-auto mb-4">
        <table className="w-full min-w-[480px]">
          <thead>
            <tr className="label-sm text-text-secondary border-b border-border-subtle">
              <th className="text-left pl-4 py-2 font-normal">METRIC</th>
              <th className="text-right py-2 font-normal">{short(diff.previous)}</th>
              <th className="text-right py-2 font-normal">{short(diff.current)}</th>
              <th className="text-right pr-4 py-2 font-normal">CHANGE</th>
            </tr>
          </thead>
          <tbody>
            {metricDeltas(diff).map((m) => (
              <tr key={m.label} className="body-sm border-b border-border-subtle/50">
                <td className="pl-4 py-2 text-text-secondary">{m.label}</td>
                <td className="py-2 text-right tabular-nums text-text-secondary">
                  {fmtValue(m, m.prev)}
                </td>
                <td className="py-2 text-right tabular-nums text-text-primary">
                  {fmtValue(m, m.curr)}
                </td>
                <td className={`py-2 pr-4 text-right tabular-nums ${TREND_TONE[trend(m)]}`}>
                  {fmtChange(m)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {diff.common_items > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Changes title="REGRESSIONS" tone="text-error" items={regressions} />
          <Changes title="FIXES" tone="text-success" items={fixes} />
        </div>
      )}
    </>
  );
}

/** Each recorded setting that differs between the two runs. */
function Changed({ diff }: { diff: RunDiff }) {
  return (
    <div className="border border-border-subtle bg-surface-container overflow-x-auto">
      <div className="label-sm text-text-secondary px-4 pt-3 pb-2">WHAT CHANGED</div>
      {diff.changed.length === 0 ? (
        <div className="label-xs text-text-tertiary px-4 pb-3">NOTHING RECORDED DIFFERS</div>
      ) : (
        <table className="w-full">
          <thead>
            <tr className="label-xs text-text-tertiary">
              <th className="text-left pl-4 py-1 font-normal">SETTING</th>
              <th className="text-right px-2 py-1 font-normal">{short(diff.previous)}</th>
              <th className="text-right pr-4 py-1 font-normal">{short(diff.current)}</th>
            </tr>
          </thead>
          <tbody>
            {diff.changed.map((c) => (
              <tr key={c.setting} className="body-sm border-t border-border-subtle/50">
                <td className="pl-4 py-1.5 text-text-secondary">{c.setting}</td>
                <td className="py-1.5 px-2 text-right text-text-tertiary break-all">
                  {fmtSetting(c.previous)}
                </td>
                <td className="py-1.5 pr-4 text-right text-text-primary break-all">
                  {fmtSetting(c.current)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/** Per check, the items that got worse and better, and whether that leans
 * one way more than chance would. */
function Flips({ diff }: { diff: RunDiff }) {
  return (
    <div className="border border-border-subtle bg-surface-container overflow-x-auto">
      <div className="label-sm text-text-secondary px-4 pt-3 pb-2" title={MCNEMAR_HINT}>
        FLIPPED VERDICTS
      </div>
      <table className="w-full">
        <thead>
          <tr className="label-xs text-text-tertiary">
            <th className="text-left pl-4 py-1 font-normal">CHECK</th>
            <th className="text-right py-1 font-normal">WORSE</th>
            <th className="text-right py-1 font-normal">BETTER</th>
            <th className="text-right pr-4 py-1 font-normal" title={MCNEMAR_HINT}>
              MCNEMAR P
            </th>
          </tr>
        </thead>
        <tbody>
          {diff.checks.map((c) => (
            <tr key={c.check} className="body-sm border-t border-border-subtle/50">
              <td className="pl-4 py-1.5 text-text-secondary">{checkLabel(c.check)}</td>
              <td className={`py-1.5 text-right tabular-nums ${c.worse ? "text-error" : ""}`}>
                {c.worse}
              </td>
              <td className={`py-1.5 text-right tabular-nums ${c.better ? "text-success" : ""}`}>
                {c.better}
              </td>
              <td
                className={`py-1.5 pr-4 text-right tabular-nums ${
                  c.p_value < 0.05 ? "text-text-primary" : "text-text-tertiary"
                }`}
              >
                {fmtP(c.p_value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {/* The label's style is upper case; this is a sentence, with a "p". */}
      <div className="label-xs text-text-tertiary px-4 py-2">
        <span className="normal-case">
          Over the items both runs answered. p &lt; 0.05: the flips lean one way more than chance
          would.
        </span>
      </div>
    </div>
  );
}

/** The API's text, with what it wraps in backticks shown as written. */
function Ticks({ text }: { text: string }) {
  return (
    <>
      {splitTicks(text).map((part, i) =>
        part.code ? (
          <code key={i} className="text-text-primary">
            {part.text}
          </code>
        ) : (
          <span key={i}>{part.text}</span>
        ),
      )}
    </>
  );
}

const TREND_TONE = {
  better: "text-success",
  worse: "text-error",
  same: "text-text-tertiary",
} as const;

/** Two runs' versions. A version is lowercase hex, as the CLI and the logs
 * print it, inside text the label styles upper-case. */
function Versions({ v }: { v: VersionPair }) {
  const version = (s: string | null) => <span className="normal-case">{fmtVersion(s)}</span>;
  if (v.prev !== null && v.prev === v.curr) return <>{version(v.curr)} IN BOTH</>;
  return (
    <>
      {version(v.prev)} → {version(v.curr)}
      {(v.prev === null || v.curr === null) && " · A RUN FROM BEFORE THE VERSIONS"}
    </>
  );
}

function fmtValue(m: MetricDelta, v: number | null) {
  if (v === null) return "—";
  if (m.kind === "pct") return fmtPct(v);
  if (m.kind === "cost") return fmtCost(v);
  return `${fmtNumber(Math.round(v))}MS`;
}

function fmtChange(m: MetricDelta) {
  if (m.prev === null || m.curr === null) return "—";
  const d = m.curr - m.prev;
  const sign = d > 0 ? "+" : d < 0 ? "−" : "±";
  const abs = Math.abs(d);
  if (m.kind === "pct") return `${sign}${(abs * 100).toFixed(1)} PTS`;
  if (m.kind === "cost") return `${sign}$${abs.toFixed(4)}`;
  return `${sign}${fmtNumber(Math.round(abs))}MS`;
}

function Changes({ title, tone, items }: { title: string; tone: string; items: RunDiff["flips"] }) {
  return (
    <div className="border border-border-subtle bg-surface-container p-4">
      <div className={`label-sm mb-3 ${tone}`}>
        {title} · {items.length}
      </div>
      {items.length === 0 ? (
        <div className="label-xs text-text-tertiary">NONE</div>
      ) : (
        <ul className="space-y-3">
          {items.map((c) => (
            <li key={`${c.item_id}-${c.check}`}>
              <div className="body-sm text-text-primary">
                <span className="text-text-tertiary tabular-nums mr-2">#{c.item_id}</span>
                {c.question}
              </div>
              <div className="label-xs text-text-tertiary mt-0.5">{checkLabel(c.check)}</div>
              <div className="body-sm text-text-secondary mt-1">{c.reasoning}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
