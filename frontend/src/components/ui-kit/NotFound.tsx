import { Link } from "@tanstack/react-router";
import { btnCls } from "@/components/ui-kit/Button";

/** A path no route takes. */
export function NotFoundPanel() {
  return (
    <div className="max-w-md mx-auto mt-16 text-center border border-border-subtle p-8 bg-surface-container">
      <div className="display-md text-text-primary">404</div>
      <div className="label-sm text-error mt-2">PATH_NOT_FOUND</div>
      <p className="body-sm text-text-secondary mt-4">
        The requested route is not registered in the protocol map.
      </p>
      <div className="mt-6">
        <Link to="/" className={btnCls("secondary")}>
          RETURN HOME
        </Link>
      </div>
    </div>
  );
}
