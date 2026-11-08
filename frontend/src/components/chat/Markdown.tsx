import {
  createContext,
  memo,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AgentCitation } from "@/lib/api";
import { interleaveCitations } from "./citations";

// The answer is model output, written from retrieved documents. react-markdown
// already escapes raw HTML and drops javascript: URLs; the components below
// cover what it leaves open.

function hostOf(href: string | undefined): string {
  try {
    return href ? new URL(href).host : "";
  } catch {
    return ""; // relative or malformed
  }
}

// In a new tab, so following a link does not drop the conversation, and
// without the referrer. The host is shown, since the link text can say anything.
function SafeLink({ href, children }: { href?: string; children?: ReactNode }) {
  const host = hostOf(href);
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer nofollow"
      className="text-primary-blue underline underline-offset-2"
    >
      {children}
      {host && <span className="label-xs text-text-tertiary ml-1 no-underline">({host} ↗)</span>}
    </a>
  );
}

// Never loaded: a poisoned document can make the model write an image whose
// URL carries conversation text, and loading it would send that text away.
function BlockedImage({ src, alt }: { src?: string; alt?: string }) {
  return (
    <span className="label-xs text-text-tertiary border border-border-subtle px-1 break-all">
      [IMAGE NOT LOADED{alt ? `: ${alt}` : ""}
      {src ? ` · ${src}` : ""}]
    </span>
  );
}

type CitationsValue = { citations: AgentCitation[]; onCite?: (n: number) => void };
const CitationsContext = createContext<CitationsValue>({ citations: [] });

function CitedParagraph({ children }: { children?: ReactNode }) {
  const { citations, onCite } = useContext(CitationsContext);
  return <p>{interleaveCitations(children, citations, onCite)}</p>;
}

function CitedListItem({ children, className }: { children?: ReactNode; className?: string }) {
  const { citations, onCite } = useContext(CitationsContext);
  return <li className={className}>{interleaveCitations(children, citations, onCite)}</li>;
}

// Module constants: a new `components` object on every render made React
// remount every paragraph of every answer on each streamed token.
const COMPONENTS: Components = {
  a: SafeLink,
  img: BlockedImage,
  p: CitedParagraph,
  li: CitedListItem,
};
const REMARK_PLUGINS = [remarkGfm];
const NO_CITATIONS: AgentCitation[] = [];

type RehypePlugins = NonNullable<Parameters<typeof ReactMarkdown>[0]["rehypePlugins"]>;
const PLAIN: RehypePlugins = [];
// The highlighter is its own chunk, fetched the first time an answer has a
// code fence; once loaded, every answer shares it.
let highlighter: RehypePlugins | null = null;
let loading: Promise<RehypePlugins> | null = null;
function loadHighlighter(): Promise<RehypePlugins> {
  loading ??= import("./highlight").then(
    (m) => (highlighter = [m.highlightPlugin]),
    (e: unknown) => {
      loading = null; // the next fence tries again; until then code stays plain
      throw e;
    },
  );
  return loading;
}
const FENCE = /^ {0,3}(```|~~~)/m;

function useRehypePlugins(text: string): RehypePlugins {
  const fenced = FENCE.test(text);
  const [plugins, setPlugins] = useState(highlighter);
  useEffect(() => {
    if (!fenced || plugins) return;
    let live = true;
    loadHighlighter().then(
      (p) => live && setPlugins(p),
      () => {},
    );
    return () => {
      live = false;
    };
  }, [fenced, plugins]);
  return fenced && plugins ? plugins : PLAIN;
}

// Memoized on (text, citations, onCite): a finished answer does not
// re-render while another one streams, so `onCite` must be stable.
export const Markdown = memo(function Markdown({
  text,
  citations = NO_CITATIONS,
  onCite,
}: {
  text: string;
  citations?: AgentCitation[];
  onCite?: (n: number) => void;
}) {
  const value = useMemo(() => ({ citations, onCite }), [citations, onCite]);
  const rehypePlugins = useRehypePlugins(text);
  return (
    <CitationsContext.Provider value={value}>
      <div className="body-read text-text-primary markdown-body">
        <ReactMarkdown
          remarkPlugins={REMARK_PLUGINS}
          rehypePlugins={rehypePlugins}
          components={COMPONENTS}
        >
          {text}
        </ReactMarkdown>
      </div>
    </CitationsContext.Provider>
  );
});
