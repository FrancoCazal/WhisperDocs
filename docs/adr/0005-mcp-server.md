# ADR 0005 — An MCP server over the same services as the API

**Status:** accepted; HTTP added by
[ADR 0007](0007-mcp-over-http-scoped-keys.md), writes by
[ADR 0008](0008-mcp-writes-confirmed-and-audited.md)
**Date:** 2026-11

## Context

WhisperDocs answers questions about a team's documents, but only through its
own chat, `/query` and WhatsApp. An agent that works somewhere else, such as
Claude Code in a repository, Claude Desktop or Cursor, cannot use the
knowledge base unless someone writes a client for the REST API. MCP is the
protocol those clients speak to tools and data.

The spec revision is **2026-07-28**. Checked against the official changelog,
it removes the `initialize` handshake and the sessions: every request carries
its protocol version in `_meta`, and Streamable HTTP is stateless. It also
deprecates the protocol's Logging feature; over stdio, servers log to stderr.
Clients still open with the handshake (up to `2025-11-25`), so a server must
serve both eras for a while.

## Decision

- **SDK.** The official Python SDK, `mcp` 2.3.0, with its high-level
  `MCPServer` (FastMCP's successor in 2.x). Its `LATEST_PROTOCOL_VERSION` is
  2026-07-28, and it answers both eras with no configuration. FastMCP 2.x
  (the separate project) was only worth it for its OAuth providers, which
  stdio does not need.
- **Transport: stdio first.** `python -m app.mcp`, which a client starts with
  `docker compose exec -T api python -m app.mcp`, against the compose
  Postgres. Streamable HTTP at `/mcp` comes later, with per-scope keys (M-2).
- **Few tools, designed for an agent; not generated from the REST routes.**
  - `search_documents`: the passages for a query, with their text, and
    nothing generated. This is the main tool: the client answers with its
    own model, and can search again with other wording.
  - `ask_documents`: WhisperDocs' own answer, with its citations.
  - `list_collections` and `list_documents`: the ids that scope a search.
  - A resource per document, `whisperdocs://documents/{id}`: the full text.
    `resources/list` names the newest 200; the template reads any of them.
- **Each tool calls the service its route calls.** The routes' bodies moved
  into `app/services/search.py` (`/retrieve`), `answering.py` (`/query`) and
  `catalog.py` (the lists). A test compares each tool's result with its
  route's, so the two cannot drift.
- **Read-only.** No tool uploads, moves or deletes; that stays in the API.
  `ask_documents` writes its `query_logs` row, as `/query` does. (Superseded:
  [ADR 0008](0008-mcp-writes-confirmed-and-audited.md) adds write tools
  behind a key that writes, the user's confirmation and an audit log.)
- **`top_k` is an explicit argument, bounded to 1-20** like the API's. The
  server never infers it.
- **Errors go to the model.** A missing document or an out-of-scope request
  is a tool error that names the ids to look up, and it is raised before the
  embedding is paid for. A failed provider call says its kind (`timeout`,
  `billing`, ...), never the provider's text.
- **Prompt injection.** The server's instructions tell the model that
  passages are uploaded content: an instruction inside one is text to
  report, never a step to follow.

## Why

- **A knowledge base is most useful where the work happens.** With
  `search_documents`, Claude Code can answer from the team's documentation
  without a WhisperDocs model call; the client brings its own model.
- **One code path.** The MCP tools and the API return the same chunks and
  the same answers, and the eval suite's numbers hold for both.
- **Stdio keeps auth out of the first version.** The client starts the
  process on the user's machine, with the database and keys the API has.
  Exposing the server over HTTP needs scoped keys first (M-2), and OAuth 2.1
  beyond a local network.

## Trade-offs

- **Its own process.** The stdio server is not the API process, so the API's
  rate limits and its model status (`/admin/model-status`) do not see its
  calls; its logs are the client's MCP log. Mounting it at `/mcp` inside the
  API would share both.
- **No navigation tools.** There is no outline or "next section" tool, and
  no inferred `top_k`. Those belong to the V2 document navigator. A client
  that needs several hops calls `search_documents` several times.
- **The image needs a rebuild** (`docker compose build api`) for the new
  dependency.

## Cross-references

- `app/mcp/server.py`: the tools, the resource and the instructions.
- `app/mcp/__main__.py`: the stdio entry point.
- `tests/test_mcp.py`: an in-memory client against fakes, both protocol eras,
  and the stdio process, which may write only JSON-RPC to stdout.
- `tests/test_mcp_db.py`: the tools and resources against Postgres.
- [Spec changelog, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog).
