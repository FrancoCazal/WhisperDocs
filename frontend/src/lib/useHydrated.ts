import { useSyncExternalStore } from "react";

const subscribe = () => () => {};

/** False while React hydrates the server's HTML, true from then on. The SPA
 *  shell is built once, for "/", and served for every path: what depends on
 *  the URL renders only after hydration, or it would not match the shell. */
export function useHydrated(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}
