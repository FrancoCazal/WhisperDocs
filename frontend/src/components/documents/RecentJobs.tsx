import { Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { api, fmtDate, type Collection, type IngestionJob } from "@/lib/api";
import { STAGE_LABEL } from "./stages";

const SHOWN = 10;

const running = (j: IngestionJob) => j.status !== "complete" && j.status !== "failed";

/** The server's latest ingestion jobs: those started from another tab, or
 *  before this page was left, which the upload list above cannot show. */
export function RecentJobs({
  collectionId,
  collections,
}: {
  collectionId: number | null;
  collections: Collection[] | undefined;
}) {
  const jobs = useQuery({
    queryKey: ["ingestion-jobs", collectionId ?? "all"],
    queryFn: () => api.listIngestionJobs(collectionId ?? undefined, SHOWN),
    // Refreshed while a job runs; a finished list stays as it is.
    refetchInterval: (q) => (q.state.data?.some(running) ? 3000 : false),
  });
  const list = jobs.data ?? [];
  if (list.length === 0) return null;
  const active = list.filter(running).length;
  const collectionName = (id: number) => collections?.find((c) => c.id === id)?.name ?? `#${id}`;

  return (
    <details className="mt-3 border border-border-subtle bg-surface-container-low group">
      <summary className="px-4 py-2 label-xs text-text-tertiary cursor-pointer hover:text-text-primary list-none">
        <span className="group-open:hidden">▸</span>
        <span className="hidden group-open:inline">▾</span> RECENT JOBS ON THE SERVER ·{" "}
        {list.length}
        {active > 0 && <span className="text-text-primary"> · {active} RUNNING</span>}
      </summary>
      <ul className="border-t border-border-subtle">
        {list.map((j) => (
          <li
            key={j.id}
            className="px-4 py-2 border-b border-border-subtle/50 last:border-b-0 flex flex-wrap items-center gap-x-3 gap-y-1 body-sm"
          >
            <span className="text-text-primary truncate min-w-0 flex-1">{j.filename}</span>
            <span className="label-xs text-text-tertiary">{collectionName(j.collection_id)}</span>
            <span className="label-xs text-text-tertiary tabular-nums">
              {fmtDate(j.created_at)}
            </span>
            <span
              className={`label-sm ${
                j.status === "complete"
                  ? "text-success"
                  : j.status === "failed"
                    ? "text-error"
                    : "text-text-secondary"
              }`}
              title={j.error_message ?? undefined}
            >
              {STAGE_LABEL[j.status]}
              {running(j) && ` · ${j.progress}%`}
            </span>
            {j.document_id !== null && (
              <Link
                to="/documentos/$id"
                params={{ id: String(j.document_id) }}
                className="label-xs text-text-secondary hover:text-primary-blue"
              >
                OPEN →
              </Link>
            )}
            {j.status === "failed" && j.error_message && (
              <span className="basis-full label-xs text-error truncate">{j.error_message}</span>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}
