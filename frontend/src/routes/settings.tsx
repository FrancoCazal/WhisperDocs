import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { api, API_BASE_URL } from "@/lib/api";
import { PageHeader } from "@/components/ui-kit/PageHeader";
import { StatDot } from "@/components/ui-kit/StatDot";
import { LoadingLine } from "@/components/ui-kit/EmptyState";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — WhisperDocs" },
      { name: "description", content: "System configuration and backend connectivity for WhisperDocs." },
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
                <StatDot ok={false} /> UNREACHABLE
              </span>
            ) : (
              <span className="flex items-center gap-2 text-success">
                <StatDot ok /> {health.data?.status?.toUpperCase()}
                {health.data?.checks?.database && (
                  <span className="text-text-tertiary">
                    · DB {health.data.checks.database.toUpperCase()}
                  </span>
                )}
              </span>
            )
          }
        />
        <Row label="ENV_VAR" value="VITE_API_BASE_URL" />
      </Section>

      <Section title="AGENT MODELS">
        {adminSettings.isLoading && <Row label="LOADING" value="…" />}
        {adminSettings.error && <Row label="ERROR" value="failed to fetch /admin/settings" />}
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
