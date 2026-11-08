import { Link, useRouterState, Outlet } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { API_BASE_URL, api, fmtCost, fmtNumber, shortModel } from "@/lib/api";
import { btnCls } from "@/components/ui-kit/Button";

const NAV_ITEMS = [
  { to: "/", label: "OVERVIEW", glyph: "▦" },
  { to: "/chat", label: "CHAT", glyph: "◆" },
  { to: "/consulta", label: "QUERY", glyph: "◇" },
  { to: "/colecciones", label: "COLLECTIONS", glyph: "▣" },
  { to: "/documentos", label: "DOCUMENTS", glyph: "▤" },
  { to: "/historial", label: "HISTORY", glyph: "◷" },
  { to: "/evaluacion", label: "EVALUATION", glyph: "◫" },
  { to: "/settings", label: "SETTINGS", glyph: "⚙" },
] as const;

const ROUTE_LABELS: Record<string, string> = {
  "/": "SYSTEM_OVERVIEW",
  "/chat": "CHAT // AGENT_THREAD",
  "/consulta": "QUERY // SINGLE_TURN_RAG",
  "/colecciones": "COLLECTIONS // NAMESPACES",
  "/documentos": "DOCUMENTS // CORPUS_INDEX",
  "/historial": "HISTORY // QUERY_LOG",
  "/evaluacion": "EVALUATION // BENCHMARKS",
  "/settings": "SETTINGS // SYSTEM_CONFIG",
};

type ApiState = "checking" | "online" | "degraded" | "offline";

const API_STATE: Record<ApiState, { label: string; dot: string; text: string }> = {
  checking: { label: "CHECKING", dot: "bg-text-tertiary", text: "text-text-tertiary" },
  online: { label: "ONLINE", dot: "bg-success", text: "text-text-primary" },
  degraded: { label: "DEGRADED", dot: "bg-warning", text: "text-warning" },
  offline: { label: "OFFLINE", dot: "bg-error", text: "text-error" },
};

// Polls GET /health. Server and first client render both show "checking", so
// the markup hydrates without a mismatch.
function useApiState(): ApiState {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
    retry: false,
  });
  if (health.isError) return "offline";
  if (!health.data) return "checking";
  return health.data.status === "ok" ? "online" : "degraded";
}

function ApiStatus({ state }: { state: ApiState }) {
  const s = API_STATE[state];
  return (
    <span role="status" className={`flex items-center gap-2 ${s.text}`}>
      <span className={`inline-block w-2 h-2 ${s.dot}`} aria-hidden="true" />
      API {s.label}
    </span>
  );
}

// The last logged answer, from the query log (chat and query pages refresh it).
function LastAnswer() {
  const last = useQuery({ queryKey: ["history", 1], queryFn: () => api.history(1) });
  const q = last.data?.[0];
  if (!q) {
    return (
      <span>LAST ANSWER: {last.isLoading ? "—" : last.isError ? "UNAVAILABLE" : "NONE YET"}</span>
    );
  }
  return (
    <>
      <span>
        LAST ANSWER: <span className="text-text-primary">{shortModel(q.model_used)}</span>
      </span>
      <span className="text-text-tertiary" aria-hidden="true">
        |
      </span>
      <span>
        TOKENS:{" "}
        <span className="text-text-primary">
          {fmtNumber(q.tokens_input)} / {fmtNumber(q.tokens_output)}
        </span>
      </span>
      <span className="text-text-tertiary" aria-hidden="true">
        |
      </span>
      <span>
        LATENCY: <span className="text-text-primary">{fmtNumber(q.latency_ms)}MS</span>
      </span>
      <span className="text-text-tertiary" aria-hidden="true">
        |
      </span>
      <span>
        COST: <span className="text-text-primary">{fmtCost(q.cost_usd)}</span>
      </span>
    </>
  );
}

export function Shell() {
  const apiState = useApiState();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  // The drawer is open on one path: navigating closes it, with no effect to sync.
  const [navOpenAt, setNavOpenAt] = useState<string | null>(null);
  const navOpen = navOpenAt === pathname;
  const setNavOpen = (open: boolean) => setNavOpenAt(open ? pathname : null);
  const breadcrumb =
    ROUTE_LABELS[pathname] ??
    (pathname.startsWith("/documentos/")
      ? "DOCUMENTS // DOCUMENT_DETAIL"
      : pathname.replace("/", "").toUpperCase() || "SYSTEM_OVERVIEW");

  return (
    <div className="min-h-screen flex flex-col bg-surface text-text-primary">
      {/* Skip-to-content for keyboard users */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[100] focus:bg-text-primary focus:text-surface focus:px-3 focus:py-2 focus:label-sm"
      >
        Skip to content
      </a>

      {/* TOP BAR */}
      <header
        className="h-12 flex items-center justify-between px-3 sm:px-6 border-b border-border-subtle bg-surface-low shrink-0"
        role="banner"
      >
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setNavOpen(!navOpen)}
            aria-label={navOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={navOpen}
            aria-controls="primary-nav"
            className={btnCls("quiet", "icon", "md:hidden")}
          >
            <span aria-hidden="true">{navOpen ? "✕" : "☰"}</span>
          </button>
          <span className="title-lg text-text-primary">WHISPERDOCS</span>
        </div>
        <div className="hidden md:block label-sm text-text-secondary">{breadcrumb}</div>
        <div className="flex items-center gap-2 sm:gap-4 label-sm">
          <ApiStatus state={apiState} />
        </div>
      </header>

      <div className="flex flex-1 min-h-0 relative">
        {/* SIDEBAR — desktop static, mobile drawer */}
        {navOpen && (
          <button
            type="button"
            aria-label="Close navigation overlay"
            onClick={() => setNavOpen(false)}
            className="md:hidden fixed inset-0 top-12 z-30 bg-black/50"
          />
        )}
        <aside
          id="primary-nav"
          aria-label="Primary navigation"
          className={[
            "shrink-0 border-r border-border-subtle bg-surface-low flex flex-col z-40",
            "md:static md:w-60 md:translate-x-0",
            "fixed top-12 bottom-0 left-0 w-60 transition-transform",
            navOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
          ].join(" ")}
        >
          <nav className="flex-1 py-2 overflow-y-auto" aria-label="Main">
            {NAV_ITEMS.map((item) => {
              const active =
                item.to === "/"
                  ? pathname === "/"
                  : pathname === item.to || pathname.startsWith(item.to + "/");
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  aria-current={active ? "page" : undefined}
                  className={[
                    "relative flex items-center gap-3 h-10 px-4 label-sm transition-colors",
                    "focus-visible:-outline-offset-2",
                    active
                      ? "text-primary-blue bg-surface-container"
                      : "text-text-secondary hover:text-primary-blue hover:bg-surface-container",
                  ].join(" ")}
                >
                  {active && (
                    <span
                      aria-hidden="true"
                      className="absolute left-0 top-0 bottom-0 w-[2px] bg-primary-blue"
                    />
                  )}
                  <span className="w-4 text-center text-text-tertiary" aria-hidden="true">
                    {item.glyph}
                  </span>
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          <div className="px-4 py-3 border-t border-border-subtle">
            <div className="label-xs text-text-tertiary">API</div>
            <div className="label-xs text-text-secondary mt-1 break-all">{API_BASE_URL}</div>
          </div>
        </aside>

        {/* MAIN */}
        <main
          id="main-content"
          role="main"
          tabIndex={-1}
          className="flex-1 min-w-0 overflow-auto focus:outline-none"
        >
          <div className="p-3 sm:p-6 md:p-8 min-h-full pb-20">
            <Outlet />
          </div>
        </main>
      </div>

      {/* BOTTOM STATUS BAR — hidden on small screens to free vertical space */}
      <footer
        role="contentinfo"
        className="hidden md:flex fixed bottom-0 left-0 right-0 h-7 bg-surface-lowest border-t border-border-subtle items-center justify-between px-4 label-xs text-text-secondary z-50"
      >
        <div className="flex items-center gap-3 min-w-0 truncate">
          <LastAnswer />
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <span>
            DATABASE:{" "}
            <span className={apiState === "online" ? "text-success" : API_STATE[apiState].text}>
              {apiState === "online" ? "OK" : apiState === "degraded" ? "ERROR" : "—"}
            </span>
          </span>
        </div>
      </footer>
    </div>
  );
}
