import { Link } from "@tanstack/react-router";
import { fmtAgo, shortModel } from "@/lib/api";
import { errorKind, failingGroups, ROLES, useModelStatus } from "@/lib/failures";

/** On every page while a model's last calls fail: which roles, why, and what
 *  the app does meanwhile. Gone once a call of each works again. Before
 *  this, an outage the turns survived (a classifier without billing) showed
 *  only inside a turn's trace. */
export function ModelNotice() {
  const status = useModelStatus();
  const groups = status.data ? failingGroups(status.data) : [];
  if (groups.length === 0) return null;

  return (
    <div
      role="status"
      aria-label="Failing models"
      className="shrink-0 border-b border-warning/40 bg-surface-container-low px-3 sm:px-6 py-2 flex flex-col gap-1"
    >
      {groups.map((g) => {
        const kind = errorKind(g.kind);
        const roles = g.roles.map((r) => ROLES[r] ?? { label: r.toUpperCase(), effect: "" });
        const effects = roles.map((r) => r.effect).filter(Boolean);
        return (
          // One line of text, wrapping as text does: spaces between the parts.
          <p key={`${g.model}:${g.kind}`} className="label-sm">
            <span className="text-warning" title={kind.hint}>
              <span aria-hidden="true">⚠ </span>
              {roles.map((r) => r.label).join(", ")} FAILING · {kind.label}
            </span>{" "}
            <span className="text-text-secondary normal-case">· {shortModel(g.model)}</span>{" "}
            {g.lastFailureAt && (
              <span className="text-text-tertiary">· {fmtAgo(g.lastFailureAt)}</span>
            )}{" "}
            {effects.length > 0 && (
              <span className="text-text-tertiary">· {effects.join("; ")}</span>
            )}
          </p>
        );
      })}
      <Link to="/settings" className="label-xs text-text-secondary hover:text-primary-blue w-fit">
        EVERY ROLE'S STATUS IN SETTINGS →
      </Link>
    </div>
  );
}
