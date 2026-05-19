# PostgreSQL indexing

Indexes are auxiliary data structures that PostgreSQL maintains alongside tables to speed up search and sorting. Understanding when to create indexes, which type to use, and how to measure their impact is one of the most valued skills in backend engineering. Poorly indexed, a system that ran fine with 10k rows becomes unusable with a million.

## Why indexes exist

Without an index, a query that filters by `email = 'x@y.com'` forces PostgreSQL to do a **sequential scan**: read every row of the table to compare. With a million rows, that's a million comparisons. With a B-tree index on `email`, PostgreSQL can find the row in ~20 comparisons (log₂ of a million).

The price is disk space, slower writes (every insert/update/delete updates all associated indexes), and periodic maintenance (VACUUM, REINDEX). That's why indexing everything indiscriminately is as bad as not indexing anything.

## Index types in PostgreSQL

### B-tree (default)

The most common type and the one PostgreSQL uses if you don't specify another. It works with comparison operators: `=`, `<`, `>`, `<=`, `>=`, `BETWEEN`, `IN`, `IS NULL`. It also supports efficient sorting (`ORDER BY`).

```sql
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_created_at ON orders(created_at);
```

Almost every column you filter by equality or range benefits from B-tree. It's the right choice until you have a specific reason to use something else.

### Hash

Indexes specialized for exact-match lookup. Historically they were faster than B-tree for equality, but B-tree improvements made them irrelevant in most cases. They don't support ranges or sorting.

```sql
CREATE INDEX idx_sessions_token_hash ON sessions USING hash(token);
```

In practice, they're rarely chosen manually. B-tree is the default even for equality.

### GIN (Generalized Inverted Index)

Designed for columns that contain multiple values per row: arrays, JSON, tsvector (full-text search). Each individual value points to the rows that contain it.

```sql
CREATE INDEX idx_posts_tags ON posts USING gin(tags);
-- Enables: WHERE tags @> ARRAY['python', 'backend']

CREATE INDEX idx_users_metadata ON users USING gin(metadata);
-- Enables: WHERE metadata @> '{"plan": "pro"}'::jsonb

CREATE INDEX idx_docs_search ON documents USING gin(to_tsvector('english', content));
-- Enables: WHERE to_tsvector('english', content) @@ to_tsquery('machine & learning')
```

GIN is the choice for native full-text search in Postgres and for filtering JSONB documents by content. It is slower on writes than B-tree, but much faster for array or JSON lookups.

### GiST (Generalized Search Tree)

A generic index that supports multiple operators. Its most common use is for geometric types (`point`, `polygon`), ranges (`daterange`, `int4range`), and nearest-neighbor searches.

```sql
CREATE INDEX idx_events_period ON events USING gist(period);
-- Enables: WHERE period && '[2026-01-01, 2026-06-30)'::daterange

CREATE INDEX idx_locations_geom ON locations USING gist(geometry);
-- PostGIS: WHERE ST_DWithin(geometry, target_point, 1000)
```

pgvector also uses GiST internally when an `ivfflat` or `hnsw` index is created over vectors.

### BRIN (Block Range Index)

Extremely compact. Instead of pointing to each row, it points to ranges of disk blocks and stores the min/max of each range. Useful for columns where values are naturally correlated with the physical order on disk: timestamps in log tables where data is inserted chronologically.

```sql
CREATE INDEX idx_logs_timestamp_brin ON logs USING brin(timestamp);
```

A BRIN index can take up 1000x less space than an equivalent B-tree, at the cost of precision: PostgreSQL ends up reading more blocks than strictly necessary, but with huge tables it's still much faster than a sequential scan.

### Partial indexes

Indexes that only include rows that satisfy a condition. Useful when most queries filter by a specific subset.

```sql
CREATE INDEX idx_users_active_email ON users(email) WHERE deleted_at IS NULL;

CREATE INDEX idx_orders_pending ON orders(created_at) WHERE status = 'pending';
```

The index takes less space and is faster to maintain because it only considers the relevant subset. Any query that includes the same `WHERE` condition can use it.

### Expression indexes

Instead of indexing a column directly, index the result of a function or expression.

```sql
CREATE INDEX idx_users_lower_email ON users(lower(email));
-- Enables: WHERE lower(email) = 'john@example.com'

CREATE INDEX idx_orders_year_month ON orders((date_trunc('month', created_at)));
```

Useful when queries normalize data in the WHERE clause (lowercase, truncate date, cast). Without the expression index, the planner can't use the direct index because the function breaks equality.

### Covering indexes (INCLUDE)

Allow you to add columns that are not part of the index predicate but are needed in the SELECT, avoiding an extra lookup to the table.

```sql
CREATE INDEX idx_orders_user_covering ON orders(user_id) INCLUDE (total, status);
-- SELECT user_id, total, status FROM orders WHERE user_id = ? uses only the index
```

This enables "index-only scans", where PostgreSQL answers the query by reading only the index without touching the table. A valuable optimization when you have repetitive queries with specific columns.

## EXPLAIN and EXPLAIN ANALYZE

You can't optimize what you don't measure. `EXPLAIN` shows the execution plan that the Postgres planner chose for a query. `EXPLAIN ANALYZE` actually runs it and shows real timings.

```sql
EXPLAIN ANALYZE
SELECT * FROM orders WHERE user_id = 42 AND status = 'completed';
```

Typical output:

```
Index Scan using idx_orders_user_status on orders
  (cost=0.43..12.87 rows=3 width=128)
  (actual time=0.034..0.045 rows=2 loops=1)
  Index Cond: (user_id = 42)
  Filter: (status = 'completed')
Planning Time: 0.142 ms
Execution Time: 0.072 ms
```

What to look at:

- **Scan type**: `Index Scan` (good), `Index Only Scan` (ideal), `Bitmap Index Scan` (OK for many rows), `Seq Scan` (bad if the table is large).
- **Estimated rows vs actual**: if they differ by orders of magnitude, statistics are stale — run `ANALYZE`.
- **Execution Time**: actual measurement in milliseconds.
- **Filter vs Index Cond**: what's in Index Cond uses the index; what's in Filter is evaluated row by row over the already-reduced results.

For slow production queries, `EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)` includes cache-hit information, which is much more useful for real debugging.

## When to create an index

Criteria that justify creating an index:

1. **The column appears in WHERE or JOIN** frequently.
2. **The table has more than a few thousand rows** — on small tables, sequential scans are so fast that the index adds nothing.
3. **High selectivity**: the query filters down to a small subset. If `status = 'active'` returns 90% of the rows, the index doesn't help much.
4. **The query is frequent**: it's worth optimizing queries that run 1000 times per minute, not one that runs once a month.

## When NOT to create an index

- **Columns that are written to a lot and read from a little**: every write pays the cost of maintaining the index.
- **Very low-cardinality columns** (e.g. `is_deleted` boolean): a partial index can be better than a B-tree on the whole column.
- **Very small tables**: less than 10k rows typically doesn't need indexes beyond the primary key.
- **Columns rarely used in queries**: creating indexes "just in case" inflates the schema and slows writes with no benefit.

## Composite indexes and column order

A composite index covers multiple columns. The order matters a lot.

```sql
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);
```

This index can be used for:

- `WHERE user_id = ?` — yes
- `WHERE user_id = ? AND created_at > ?` — yes, optimal
- `WHERE user_id = ? ORDER BY created_at` — yes, with sort for free
- `WHERE created_at > ?` — not effectively (requires a full index scan or falls back to a seq scan)

Rule: put first the column that appears in equality conditions; then range columns; finally columns used only for sorting. The first column of the index is the most restrictive one.

## Maintenance

Indexes degrade over time due to fragmentation. Periodic operations:

- **VACUUM**: reclaims space from deleted rows. Postgres does this automatically (autovacuum), but for heavily-written tables you sometimes need to tune parameters.
- **ANALYZE**: updates the statistics used by the planner. Without up-to-date stats, the planner makes bad decisions.
- **REINDEX**: rebuilds the index from scratch. Useful when an index gets significantly fragmented. Recent versions of Postgres offer `REINDEX CONCURRENTLY` to avoid blocking writes.

```sql
VACUUM ANALYZE orders;
REINDEX INDEX CONCURRENTLY idx_orders_user_id;
```

## Detecting unused indexes

Postgres exposes index usage stats in `pg_stat_user_indexes`:

```sql
SELECT schemaname, relname, indexrelname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

Indexes with `idx_scan = 0` since the last stats reset are not being used — candidates for deletion. Caveat: some indexes exist for constraints (UNIQUE, PRIMARY KEY) and don't show as "used" but are still necessary.

## Typical errors

- **Creating indexes preventively with no specific query**: inflates storage and slows writes with no clear benefit.
- **Indexing columns with functions in WHERE without using an expression index**: `WHERE lower(email) = ?` does not use an index on `email`.
- **Not having an index on foreign keys**: PostgreSQL does not create them automatically. Without an index on the FK, deletes on the parent table cause seq scans on the child.
- **Forgetting `ANALYZE` after bulk loads**: the planner keeps using old stats and picks badly.
- **Creating identical indexes with different names**: they take duplicate space. Postgres warns but allows it.

## Known trade-offs

| Dimension | With indexes | Without indexes |
|-----------|--------------|-----------------|
| Read latency | Very low | High on large tables |
| Write latency | Higher (every write updates indexes) | Minimal |
| Disk usage | More (indexes take space) | Only the table |
| Flexibility under query changes | Limited (specific indexes) | Maximum |
| Risk of unexpected slow queries | Lower | Higher |

The right balance is: index what real queries need, measure with EXPLAIN ANALYZE, review `pg_stat_user_indexes` periodically, drop what isn't used.

## References

- PostgreSQL docs - Indexes: https://www.postgresql.org/docs/current/indexes.html
- Use The Index, Luke! — excellent agnostic guide: https://use-the-index-luke.com/
- pgstats documentation: https://www.postgresql.org/docs/current/monitoring-stats.html
- explain.depesz.com — visualizer for execution plans
