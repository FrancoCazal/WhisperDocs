/** Where a logged answer came from: the agent's chat (it logs the node that
 *  answered) or POST /query (it logs none). */
export function SourceTag({ action }: { action: string | null }) {
  const chat = action !== null;
  return (
    <span
      className={`label-xs px-1.5 py-0.5 border ${
        chat
          ? "border-primary-blue/50 text-primary-blue"
          : "border-border-subtle text-text-secondary"
      }`}
      title={chat ? `Chat · answered by ${action}` : "Query page"}
    >
      {chat ? "CHAT" : "QUERY"}
    </span>
  );
}
