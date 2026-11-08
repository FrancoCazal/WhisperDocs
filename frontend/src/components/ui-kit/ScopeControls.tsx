import { useId } from "react";
import { fmtNumber, type Collection } from "@/lib/api";
import { Btn } from "./Button";
import { TOP_K_MAX, TOP_K_MIN } from "@/lib/topK";

const fieldCls =
  "bg-surface-lowest border border-border-subtle px-2 h-7 text-text-primary body-sm focus:outline-none focus:border-border-focus disabled:text-text-tertiary";

/** What a question searches: a collection, every one ("ALL") or one
 *  document, and how many chunks it retrieves. The chip next to the select
 *  says what the scope holds. */
export function ScopeControls({
  collections,
  collectionId,
  onCollection,
  scopeDocument = null,
  onLeaveDocument,
  topK,
  defaultTopK,
  onTopK,
  disabled = false,
}: {
  collections: Collection[] | undefined;
  collectionId: number | null;
  onCollection: (id: number | null) => void;
  // The one document searched in place of the collection; its title once loaded.
  scopeDocument?: { id: number; title: string | null } | null;
  onLeaveDocument?: () => void;
  topK: number;
  defaultTopK: number;
  onTopK: (n: number) => void;
  disabled?: boolean;
}) {
  const hintId = useId();
  const all = collectionId === null;
  const inScope = all
    ? (collections ?? [])
    : (collections ?? []).filter((c) => c.id === collectionId);
  const docs = inScope.reduce((n, c) => n + c.document_count, 0);
  const hint = collections
    ? all
      ? `EVERY COLLECTION (${collections.length}) · ${fmtNumber(docs)} DOCS`
      : `${fmtNumber(docs)} DOCS`
    : null;

  const documentTitle = scopeDocument && (scopeDocument.title ?? `#${scopeDocument.id}`);

  return (
    <>
      {scopeDocument ? (
        <div className="flex items-center gap-2 min-w-0 label-sm text-text-secondary">
          DOCUMENT:
          <span
            className="body-sm normal-case tracking-normal text-text-primary truncate max-w-48"
            title={`Questions search only ${documentTitle}.`}
          >
            {documentTitle}
          </span>
          <Btn
            size="sm"
            variant="quiet"
            onClick={onLeaveDocument}
            disabled={disabled}
            aria-label="Search the whole collection"
            title="The next questions search the document's whole collection."
          >
            ✕
          </Btn>
        </div>
      ) : (
        <>
          <label className="flex items-center gap-2 label-sm text-text-secondary">
            COLLECTION:
            <select
              value={collectionId ?? ""}
              disabled={disabled}
              onChange={(e) => onCollection(e.target.value ? Number(e.target.value) : null)}
              aria-describedby={hint ? hintId : undefined}
              className={fieldCls}
            >
              <option value="">ALL</option>
              {collections?.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
          {hint && (
            <span
              id={hintId}
              title={all ? "ALL searches every collection's documents together." : undefined}
              className="hidden lg:inline label-xs px-2 py-0.5 border border-border-subtle text-text-tertiary tabular-nums"
            >
              {hint}
            </span>
          )}
        </>
      )}
      <label
        className="flex items-center gap-2 label-sm text-text-secondary"
        title={`Chunks retrieved for each question. The backend's default is ${defaultTopK}.`}
      >
        TOP_K:
        <input
          type="number"
          min={TOP_K_MIN}
          max={TOP_K_MAX}
          value={topK}
          disabled={disabled}
          onChange={(e) => onTopK(Number(e.target.value))}
          className={`${fieldCls} w-14 tabular-nums`}
        />
      </label>
    </>
  );
}
