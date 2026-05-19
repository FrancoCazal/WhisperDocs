import { Link, useRouterState, Outlet } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { fmtTimestamp } from "@/lib/api";

const NAV_ITEMS = [
  { to: "/", label: "OVERVIEW", glyph: "▦" },
  { to: "/chat", label: "CHAT", glyph: "◆" },
  { to: "/colecciones", label: "COLLECTIONS", glyph: "▣" },
  { to: "/documentos", label: "DOCUMENTS", glyph: "▤" },
  { to: "/historial", label: "HISTORY", glyph: "◷" },
  { to: "/evaluacion", label: "EVALUATION", glyph: "◫" },
  { to: "/settings", label: "SETTINGS", glyph: "⚙" },
] as const;

const ROUTE_LABELS: Record<string, string> = {
  "/": "SYSTEM_OVERVIEW",
  "/chat": "CHAT // AGENT_THREAD",
  "/colecciones": "COLLECTIONS // NAMESPACES",
  "/documentos": "DOCUMENTS // CORPUS_INDEX",
  "/historial": "HISTORY // QUERY_LOG",
  "/evaluacion": "EVALUATION // BENCHMARKS",
  "/settings": "SETTINGS // SYSTEM_CONFIG",
};

function useTimestamp() {
  const [ts, setTs] = useState(() => fmtTimestamp());
  useEffect(() => {
    const id = setInterval(() => setTs(fmtTimestamp()), 1000);
    return () => clearInterval(id);
  }, []);
  return ts;
}

export function Shell() {
  const ts = useTimestamp();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const [navOpen, setNavOpen] = useState(false);
  const breadcrumb =
    ROUTE_LABELS[pathname] ??
    (pathname.startsWith("/documentos/")
      ? "DOCUMENTS // DOCUMENT_DETAIL"
      : pathname.replace("/", "").toUpperCase() || "RESUMEN");

  // Close mobile drawer when route changes.
  useEffect(() => {
    setNavOpen(false);
  }, [pathname]);

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
            onClick={() => setNavOpen((v) => !v)}
            aria-label={navOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={navOpen}
            aria-controls="primary-nav"
            className="md:hidden inline-flex items-center justify-center w-8 h-8 border border-border-subtle text-text-primary hover:bg-surface-container focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-blue"
          >
            <span aria-hidden="true">{navOpen ? "✕" : "☰"}</span>
          </button>
          <span className="title-lg text-text-primary">WHISPERDOCS</span>
        </div>
        <div className="hidden md:block label-sm text-text-secondary">{breadcrumb}</div>
        <div className="flex items-center gap-2 sm:gap-4 label-sm">
          <span className="hidden sm:inline text-text-secondary">{ts}</span>
          <span className="flex items-center gap-2 text-text-primary">
            <span className="inline-block w-2 h-2 bg-success" aria-hidden="true" />
            ONLINE
          </span>
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
          <div className="px-4 py-4 border-b border-border-subtle">
            <div className="label-xs text-text-tertiary">ACCESS_LEVEL_01</div>
            <div className="label-xs text-text-secondary mt-1">SESSION_ACCESS_GRANTED</div>
          </div>

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
                    "focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-blue focus-visible:ring-inset",
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

          <div className="px-4 py-3 border-t border-border-subtle space-y-3">
            <button
              type="button"
              className="label-xs text-text-tertiary hover:text-text-primary transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-blue"
            >
              <span aria-hidden="true">⏻ </span>LOGOUT
            </button>
            <div className="label-xs text-text-tertiary">WHISPER_PROTOCOL_V4.0</div>
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
        <div className="flex items-center gap-3">
          <span>TOKENS: <span className="text-text-primary">4,201</span></span>
          <span className="text-text-tertiary" aria-hidden="true">|</span>
          <span>LATENCY: <span className="text-text-primary">14MS</span></span>
          <span className="text-text-tertiary" aria-hidden="true">|</span>
          <span>MODEL: <span className="text-text-primary">gpt-4o-mini</span></span>
          <span className="text-text-tertiary" aria-hidden="true">|</span>
          <span>COST: <span className="text-text-primary">$0.0042</span></span>
        </div>
        <div className="flex items-center gap-3">
          <span>SYSTEM_STATUS: <span className="text-success">NORMAL</span></span>
          <span className="text-text-tertiary" aria-hidden="true">|</span>
          <span>ENCRYPTION_PROTOCOL: <span className="text-success">ACTIVE</span></span>
        </div>
      </footer>
    </div>
  );
}
