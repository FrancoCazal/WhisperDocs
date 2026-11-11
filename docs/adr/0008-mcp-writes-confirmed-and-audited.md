# ADR 0008 — MCP writes: a key that writes, the user's confirmation, an audit log

**Status:** accepted
**Date:** 2026-11

## Context

ADRs 0005 and 0007 kept the MCP server read-only: uploads, deletions and
moves stayed in the API. An agent that keeps a knowledge base up to date
(adds the runbook it just wrote, retires an old policy, files a document in
the right collection) needs them as tools. Three things were missing for
that (audit M-1 and M-2, BE-36): a key that may write, a way for the user to
approve a change the model asked for, and a record of every change.

The 2026-07-28 spec makes `/mcp` stateless, with no session in which the
server can send the client a request. It asks for input differently: a tool
call may answer with an `InputRequiredResult` holding the questions and an
opaque `request_state`, and the client calls again with the answers. The SDK
(`mcp` 2.3) drives it with `Resolve` and `Elicit`, seals the state against
tampering, and on 2025-11-25 sends the same question mid-call over a
connection that has a channel back. Claude Code, Cursor and VS Code declare
elicitation; Claude Code serves it in interactive sessions.

## Decision

- **Three tools** (`app/mcp/writes.py`): `ingest_document` (a file as text,
  or as base64 for PDF, Word and Excel, through the synchronous upload's
  checks and ingestion), `delete_document` and `move_document`. Each calls
  the service its route calls: `ingestion.ingest_bytes`, and
  `documents.delete_documents` and `move_documents`.
- **A key reads unless made to write** (`api_keys.can_write`, migration
  0023, `python -m app.mcp.keys create ... --write`), and writes only in the
  collections it reaches; a move needs both. The API's own key, an open API
  and stdio write, as the REST API lets them.
- **The user confirms a deletion or a move**, by elicitation. A resolver
  checks the call first (the key, the reach, a target that already holds
  the text) and only then asks, naming the document, its chunks and the
  collections. The tool acts on a yes; a no, a dismissal or an unchecked box
  changes nothing and says so. Adding asks nothing: it changes no existing
  document, and the collection refuses a text it holds.
- **No question, no change.** A client that declares no elicitation is
  refused, and so is a 2025-11-25 client over HTTP, whose stateless exchange
  can neither carry the question nor bring the capabilities it declared at
  `initialize`. The message says to use the Documents page.
- **Every call is logged** (`audit_log`, `app/services/audit.py`): who, the
  channel, the tool, the outcome (done, declined, cancelled, refused,
  failed), the document and collection as plain ids, and a detail with the
  title or the reason. A write that happens commits its row in its own
  transaction; the others commit theirs alone. `GET /admin/audit-log`
  pages it.

## Why

- **The model asks, the user decides.** A delete argument the model fills
  (`confirm: true`) is the model confirming itself. Elicitation puts the
  question in front of the person, from the server, with what will happen.
- **Checks before the question.** Asking "delete this?" and then refusing
  the key would waste the user's attention and teach them to click yes.
- **Refusing beats acting unasked.** Clients that cannot ask are a minority
  and shrinking; the UI and the API still delete. Acting without the
  question on those clients would make the confirmation optional exactly
  where nobody sees it.
- **One commit for a change and its row.** A log written after the fact
  can miss a change, or record one that rolled back.

## Trade-offs

- **The sealed state lives in the process.** The SDK's default key is
  generated at startup, so a question in flight is lost on a restart, and
  several API processes would need a shared key (`RequestStateSecurity`).
  The API runs as one process, as for the rate limits.
- **Moves are one document at a time.** The route moves many; a tool that
  moves many would ask about many at once. One at a time keeps the
  question readable.
- **The REST API's writes are not in the log.** It has its own key and its
  own users (the UI); the log covers what agents do.
- **No OAuth**, still (ADR 0007): the keys are static bearers.

## Cross-references

- `app/mcp/writes.py`: the tools, the resolvers that ask, `why_cannot_ask`.
- `app/services/audit.py`, `app/models/audit_log.py`: the log.
- `app/services/documents.py`, `app/services/ingestion.py`: the services
  the routes share.
- `tests/test_mcp_writes_db.py`: each answer to the question, a client that
  cannot ask, keys that only read or reach one collection, both protocol
  eras over HTTP, the log through the API.
- [ADR 0005](0005-mcp-server.md), [ADR 0007](0007-mcp-over-http-scoped-keys.md).
