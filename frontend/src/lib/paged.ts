import { useMemo } from "react";
import { useInfiniteQuery, type QueryKey } from "@tanstack/react-query";
import type { Page } from "@/lib/api";

export const PAGE_SIZE = 50;

/**
 * A cursor-paged list: the rows loaded so far, the server's total (sent with
 * the first page), and the query to fetch the next page with.
 */
export function usePagedList<T>(
  queryKey: QueryKey,
  fetchPage: (cursor: number | undefined) => Promise<Page<T>>,
) {
  const query = useInfiniteQuery({
    queryKey,
    queryFn: ({ pageParam }) => fetchPage(pageParam),
    initialPageParam: undefined as number | undefined,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
  });
  const items = useMemo(() => query.data?.pages.flatMap((p) => p.items) ?? [], [query.data]);
  const total = query.data?.pages[0]?.total ?? null;
  return { query, items, total };
}
