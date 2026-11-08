import { useEffect, useState } from "react";

/** `value`, once it has stayed the same for `ms`: a search box asks the
 *  server when the typing pauses, not on every key. */
export function useDebounced<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return settled;
}
