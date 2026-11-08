import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  API_BASE_URL,
  fmtAgo,
  fmtDate,
  getApiKey,
  setApiKey,
  shortModel,
  type ModelRoleState,
} from "@/lib/api";
import { errorKind, useModelStatus } from "@/lib/failures";
import { Btn } from "@/components/ui-kit/Button";
import { inputCls } from "@/components/ui-kit/Modal";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { StatDot } from "@/components/ui-kit/StatDot";
import { LoadingLine } from "@/components/ui-kit/EmptyState";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — WhisperDocs" },
      {
        name: "description",
        content: "System configuration and backend connectivity for WhisperDocs.",
      },
    ],
  }),
  component: SettingsPage,
});

function SettingsPage() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 10_000,
  });

  const adminSettings = useQuery({
    queryKey: ["admin-settings"],
    queryFn: api.adminSettings,
  });
  const healthy = health.data?.status === "ok";

  return (
    <div>
      <PageHeader title="SETTINGS" subtitle="SYSTEM CONFIGURATION" />

      <Section title="BACKEND CONNECTION">
        <Row label="API_BASE_URL" value={API_BASE_URL} />
        <Row
          label="HEALTH"
          value={
            health.isLoading ? (
              <LoadingLine label="PROBING" />
            ) : health.error ? (
              <span className="flex items-center gap-2 text-error">
                <StatDot tone="error" /> UNREACHABLE
              </span>
            ) : (
              <span
                className={`flex items-center gap-2 ${healthy ? "text-success" : "text-warning"}`}
              >
                <StatDot tone={healthy ? "success" : "warning"} />{" "}
                {health.data?.status?.toUpperCase()}
                {health.data?.checks?.database && (
                  // A failed check is "error: <exception>": the full text is the tooltip.
                  <span className="text-text-tertiary" title={health.data.checks.database}>
                    · DB {health.data.checks.database === "ok" ? "OK" : "ERROR"}
                  </span>
                )}
              </span>
            )
          }
        />
        <Row label="ENV_VAR" value="VITE_API_BASE_URL" />
        <Row label="API_KEY" value={<ApiKeyField />} />
      </Section>

      <Section title="AGENT MODELS">
        {adminSettings.isLoading && <LoadingLine />}
        {adminSettings.error && <Row label="ERROR" value={adminSettings.error.message} />}
        {adminSettings.data &&
          Object.entries(adminSettings.data.models).map(([node, cfg]) => (
            <Row
              key={node}
              label={node.toUpperCase()}
              value={
                <span className="flex items-center gap-3">
                  <span className="text-text-primary">{cfg.model}</span>
                  <span className="label-xs text-text-tertiary tabular-nums">
                    T={cfg.temperature.toFixed(2)}
                  </span>
                </span>
              }
            />
          ))}
      </Section>

      <ModelStatusSection />

      {adminSettings.data && (
        <>
          <Section title="RETRIEVAL & COMPACTION">
            <Row label="RETRIEVAL_TOP_K" value={String(adminSettings.data.retrieval.top_k)} />
            <Row
              label="COMPACTION_TOKEN_THRESHOLD"
              value={String(adminSettings.data.compaction.token_threshold)}
            />
            <Row
              label="COMPACTION_COUNT_THRESHOLD"
              value={String(adminSettings.data.compaction.count_threshold)}
            />
            <Row label="K_RECENT" value={String(adminSettings.data.compaction.k_recent)} />
          </Section>

          <Section title="EMBEDDING">
            <Row label="MODEL" value={adminSettings.data.embedding.model} />
            <Row label="DIMENSION" value={String(adminSettings.data.embedding.dimension)} />
          </Section>
        </>
      )}

      <Section title="ENVIRONMENT">
        <Row label="MODE" value={import.meta.env.MODE} />
        <Row label="PROD" value={String(import.meta.env.PROD)} />
        <Row label="UI_VERSION" value="WHISPER_PROTOCOL_V4.0" />
      </Section>

      <div className="mt-8 border border-border-subtle p-4 bg-surface-container-low">
        <div className="label-sm text-text-secondary mb-2">NOTE</div>
        <div className="body-sm text-text-secondary leading-relaxed">
          Model assignments are read-only here. To change them, set the corresponding{" "}
          <code className="text-text-primary">AGENT_MODEL_*</code> env var (e.g.{" "}
          <code className="text-text-primary">AGENT_MODEL_RAG_GENERATION</code>) and restart the
          backend. Runtime mutation would invalidate the compiled LangGraph and is not exposed.
        </div>
      </div>
    </div>
  );
}

/** Each role's last calls, as the API recorded them: polling it calls no model. */
function ModelStatusSection() {
  const status = useModelStatus();
  return (
    <Section title="MODEL STATUS">
      {status.isLoading && <LoadingLine />}
      {status.error && <Row label="ERROR" value={status.error.message} />}
      {status.data && (
        <>
          <Row
            label="RECORDED SINCE"
            value={
              // Words whole: the row breaks anywhere, for URLs.
              <span className="break-normal">
                {fmtDate(status.data.since)}{" "}
                <span className="text-text-tertiary">
                  · FROM THE API&apos;S OWN CALLS; NONE IS MADE TO CHECK
                </span>
              </span>
            }
          />
          {Object.entries(status.data.roles).map(([role, s]) => (
            <Row key={role} label={role.toUpperCase()} value={<RoleState s={s} />} />
          ))}
        </>
      )}
    </Section>
  );
}

// Inline parts with spaces between them, so the row reads as one line of
// text, to a screen reader too. Words stay whole: the row breaks anywhere,
// for URLs.
function RoleState({ s }: { s: ModelRoleState }) {
  const model = <span className="text-text-tertiary">· {shortModel(s.model)}</span>;
  if (s.state === "unknown") {
    return (
      <span className="break-normal">
        <span className="text-text-tertiary">NO CALLS YET</span> {model}
      </span>
    );
  }
  if (s.state === "ok") {
    return (
      <span className="break-normal">
        <span className="inline-flex items-center gap-2 text-success">
          <StatDot tone="success" /> OK
        </span>{" "}
        {s.last_ok_at && <span className="text-text-tertiary">· {fmtAgo(s.last_ok_at)}</span>}{" "}
        {model}
      </span>
    );
  }
  const kind = errorKind(s.last_failure_kind);
  return (
    <span className="break-normal" title={kind.hint}>
      <span className="inline-flex items-center gap-2 text-warning">
        <StatDot tone="warning" /> FAILING · {kind.label}
      </span>{" "}
      <span className="text-text-tertiary">
        · {s.failures_in_a_row} IN A ROW
        {s.last_failure_at && ` · ${fmtAgo(s.last_failure_at)}`}
      </span>{" "}
      {model}
    </span>
  );
}

/** The key the API asks for when it runs with API_KEY set. It is never shown
 *  back: the field says whether one is set, and a new one replaces it. */
function ApiKeyField() {
  const qc = useQueryClient();
  const [draft, setDraft] = useState("");
  const [isSet, setIsSet] = useState(() => getApiKey() !== "");
  const [note, setNote] = useState<string | null>(null);

  function keep(key: string) {
    const stored = setApiKey(key);
    const kept = key.trim() !== "";
    setIsSet(kept);
    setDraft("");
    setNote(kept && !stored ? "KEPT UNTIL RELOAD: THIS BROWSER BLOCKS STORAGE" : null);
    // Every query again, now with the new key.
    void qc.invalidateQueries();
  }

  return (
    <form
      className="flex flex-wrap items-center gap-2 py-2 sm:py-0"
      onSubmit={(e) => {
        e.preventDefault();
        if (draft.trim()) keep(draft);
      }}
    >
      <input
        type="password"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        placeholder={isSet ? "SET · TYPE TO REPLACE" : "NOT SET"}
        title="Needed when the API runs with API_KEY set. Kept in this browser."
        aria-label="API key"
        autoComplete="off"
        spellCheck={false}
        className={`${inputCls} sm:max-w-sm h-8`}
      />
      <Btn type="submit" size="sm" variant="quiet" disabled={!draft.trim()}>
        SAVE
      </Btn>
      {isSet && (
        <Btn size="sm" variant="destructive-ghost" onClick={() => keep("")}>
          CLEAR
        </Btn>
      )}
      {note && <span className="label-xs text-warning">{note}</span>}
    </form>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-8">
      <h2 className="title-sm text-text-primary mb-3">{title}</h2>
      <div className="border border-border-subtle bg-surface-container">{children}</div>
    </section>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-[260px_1fr] items-start sm:items-center gap-1 sm:gap-3 px-3 sm:px-6 py-2 sm:h-11 sm:py-0 border-b border-border-subtle/50 last:border-0">
      <div className="label-sm text-text-secondary">{label}</div>
      <div className="body-sm text-text-primary break-all">{value}</div>
    </div>
  );
}
