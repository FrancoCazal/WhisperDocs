# ADR 0007 — MCP over HTTP, with keys scoped to collections

**Status:** accepted
**Date:** 2026-11

## Context

ADR 0005 put the MCP server on stdio, and left HTTP for when there were
scoped keys (audit M-2). Over stdio the client starts the process, which has
every collection, and the API's rate limits never see its calls. A client
that cannot start a process (a remote agent, a teammate's Cursor), or one
that should see only some of the documents, needs the server over the
network, and a key that says what it reaches.

The 2026-07-28 spec makes Streamable HTTP stateless: each request carries
its protocol version and stands alone. The official SDK (`mcp` 2.3) serves
it from a `StreamableHTTPSessionManager`, and still answers clients that
open with the older `initialize` handshake. Its HTTP app checks the Host
header against DNS rebinding when it is told the hosts to allow.

## Decision

- **Inside the API, at `/mcp`.** The same `build_server()` the stdio entry
  point serves, mounted as a route of the FastAPI app (`app/mcp/http.py`),
  stateless, with JSON answers. The SDK's session manager runs in the API's
  lifespan; it runs once per instance, so each lifespan builds a new one.
  The API's middleware (request id, metrics, CORS) wraps it like any route.
- **Keys of its own, scoped to collections** (`api_keys`, migration 0019):
  a name, the SHA-256 of the key, its first characters to tell it apart,
  and the collections it reaches, null for every one. A CLI makes, lists
  and revokes them (`python -m app.mcp.keys`); `create` needs
  `--collection` or `--all`, so a key reaches everything only when asked
  to, and prints the key once.
- **Who reaches what** (`app/mcp/access.py`). A key, as
  `Authorization: Bearer` or the API's `X-API-Key`:
  - a stored, unrevoked key reaches its collections;
  - the API's own `API_KEY` reaches every one;
  - no key reaches every one while `API_KEY` is unset, as the REST API is
    open then, and is a 401 once it is set. An unknown or revoked key is a
    401 either way.

  `KeyAuth`, an ASGI middleware, checks the key and puts its `Access` in
  the request's state. The SDK hands each tool the HTTP request its message
  came in, so a tool reads its caller from there. Over stdio there is no
  request, and the process reaches every collection, as before.
- **Out of reach is the same as missing.** A collection or document the key
  does not reach gets the error a missing one gets, word for word. The
  lists, `resources/list` and the document resource show only what the key
  reaches. A search without a collection takes the key's collection when it
  has one, and asks for `collection_id` when it has several.
- **The paid tools count against the API's limit.** Over HTTP,
  `search_documents` and `ask_documents` take from the `questions` bucket,
  per key (per IP without one), before any embedding or model call.
- **Host check on.** The SDK's guard against DNS rebinding answers only the
  hosts in `MCP_ALLOWED_HOSTS`, localhost by default; it runs after the key.

## Why

- **One process, one set of limits.** Mounting it in the API shares the
  rate limits, the model status, the logs and the metrics, which ADR 0005
  listed as the cost of stdio.
- **A key per client.** A collection holds one team's or one customer's
  documents; a client gets those and no others. Revoking one key leaves the
  rest working, which the single `API_KEY` cannot do.
- **No oracle.** If out of reach and missing differed, a key could probe
  the ids of other collections.
- **Hash only.** Keys are 32 random bytes, so a plain SHA-256 is enough to
  check them and useless to anyone who reads the table; no password is
  being stretched.

## Trade-offs

- **Not OAuth.** The spec's authorization is OAuth 2.1; a static bearer key
  is what local clients (Claude Code, Cursor) take today. Beyond a local
  network, OAuth would replace the keys.
- **Read-only, still.** No tool writes, so there is no audit log and no
  elicitation yet. Write tools need a key that may write, an `audit_log`
  table and a confirmation before a delete or a move (BE-36).
- **Collections, not documents.** A key reaches whole collections. Finer
  scopes would need the same check in every query, as collections have it
  now.
- **In-process state.** The rate limits live in the API process, as the
  REST API's do.

## Cross-references

- `app/mcp/http.py`: the endpoint, its lifespan and the host check.
- `app/mcp/access.py`: keys, `Access` and `KeyAuth`.
- `app/mcp/keys.py`: the key CLI.
- `app/mcp/server.py`: each tool's scope.
- `tests/test_mcp_http_db.py`: over the SDK's HTTP client against Postgres,
  a key that does not see another collection, 401s, the host check, the
  rate limit per key and both protocol eras.
- `tests/test_mcp_http.py`: the key check, without a database.
- [ADR 0005](0005-mcp-server.md): the server itself.
