import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";

// The bounds the backend validates (app/schemas/agent.py, query.py).
export const TOP_K_MIN = 1;
export const TOP_K_MAX = 20;
// Until /admin/settings answers.
const FALLBACK_TOP_K = 5;

/** A page's top_k: the backend's RETRIEVAL_TOP_K until the user picks one.
 *  Setting a value that is not a positive number goes back to the default. */
export function useTopK() {
  const settings = useQuery({
    queryKey: ["admin-settings"],
    queryFn: api.adminSettings,
    staleTime: Infinity,
  });
  const [picked, setPicked] = useState<number | null>(null);
  const defaultTopK = settings.data?.retrieval.top_k ?? FALLBACK_TOP_K;
  const setTopK = (n: number) =>
    setPicked(Number.isFinite(n) && n >= TOP_K_MIN ? Math.min(TOP_K_MAX, Math.round(n)) : null);
  return { topK: picked ?? defaultTopK, defaultTopK, setTopK };
}
