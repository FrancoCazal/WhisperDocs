// Where a turn's trace opens. VITE_TRACE_URL is a URL with "{trace_id}" in
// place of the id; for the compose file's Langfuse (`--profile tracing`):
//   VITE_TRACE_URL=http://localhost:3000/project/whisperdocs/traces/{trace_id}
// Unset, the history shows the id alone: any OTLP backend can look it up.
const TEMPLATE = (import.meta.env.VITE_TRACE_URL as string | undefined) ?? "";

// An OpenTelemetry trace id: 32 lowercase hex characters (app/telemetry.py).
const TRACE_ID = /^[0-9a-f]{32}$/;

/** The trace's page, or null without a usable template. Only http(s): the
 * template is configuration, but a link is never a script. */
export function traceUrl(traceId: string, template: string = TEMPLATE): string | null {
  if (!TRACE_ID.test(traceId) || !template.includes("{trace_id}")) return null;
  const url = template.replaceAll("{trace_id}", traceId);
  try {
    const { protocol } = new URL(url);
    return protocol === "http:" || protocol === "https:" ? url : null;
  } catch {
    return null;
  }
}
