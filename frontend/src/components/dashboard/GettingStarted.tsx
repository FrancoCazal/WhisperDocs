import { Link } from "@tanstack/react-router";

type Step = {
  done: boolean;
  label: string;
  hint: string;
  to: "/colecciones" | "/documentos" | "/chat";
};

/** The three steps to a first answer; gone once all are done. */
export function GettingStarted({
  collections,
  documents,
  questions,
}: {
  collections: number;
  documents: number;
  questions: number;
}) {
  const steps: Step[] = [
    {
      done: collections > 0,
      label: "CREATE A COLLECTION",
      hint: "A namespace for related documents.",
      to: "/colecciones",
    },
    {
      done: documents > 0,
      label: "UPLOAD DOCUMENTS",
      hint: "PDF, Markdown or text: they are chunked and embedded.",
      to: "/documentos",
    },
    {
      done: questions > 0,
      label: "ASK A QUESTION",
      hint: "The agent answers from your documents and cites them.",
      to: "/chat",
    },
  ];
  const next = steps.findIndex((s) => !s.done);
  if (next === -1) return null;
  return (
    <section
      aria-label="Getting started"
      className="mb-6 border border-border-subtle bg-surface-container p-4 sm:p-5"
    >
      <h2 className="title-sm text-text-primary mb-3">GETTING STARTED</h2>
      <ol className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {steps.map((s, i) => (
          <li
            key={s.label}
            aria-current={i === next ? "step" : undefined}
            className={`border p-3 ${i === next ? "border-primary-blue" : "border-border-subtle"}`}
          >
            <div className={`label-sm ${s.done ? "text-success" : "text-text-primary"}`}>
              {s.done ? "✓" : `${i + 1}.`} {s.label}
            </div>
            <div className="body-sm text-text-secondary mt-1">{s.hint}</div>
            {i === next && (
              <Link
                to={s.to}
                className="inline-block mt-2 label-sm text-primary-blue hover:text-text-primary"
              >
                START →
              </Link>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}
