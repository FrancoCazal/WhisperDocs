import { afterEach, describe, expect, it, vi } from "vitest";
import {
  api,
  ApiError,
  fmtCost,
  fmtDate,
  fmtNumber,
  fmtTurnTime,
  type AgentStreamEvent,
  type IngestionJobEvent,
} from "./api";

/** A text/event-stream response that sends `chunks` as they are, and then
 *  ends, or with `open` stays open, as a server still answering does. */
function sse(chunks: string[], { open = false, onCancel = () => {} } = {}): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      for (const chunk of chunks) c.enqueue(enc.encode(chunk));
      if (!open) c.close();
    },
    cancel: onCancel,
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

/** `text` cut every `n` characters: a boundary can fall anywhere. */
function cut(text: string, n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += n) out.push(text.slice(i, i + n));
  return out;
}

function stubFetch(res: Response) {
  const fetchMock = vi.fn(async () => res);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function agentEvents(res: Response): Promise<AgentStreamEvent[]> {
  stubFetch(res);
  const events: AgentStreamEvent[] = [];
  await api.agentChatStream({ message: "hi" }, (e) => events.push(e));
  return events;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const STREAM = [
  "event: thread",
  'data: {"thread_id": "t-1"}',
  "",
  ": keep-alive",
  "",
  "event: token",
  'data: {"node": "answer_direct", "delta": "Hel"}',
  "",
  "event: token",
  'data: {"node": "answer_direct", "delta": "lo"}',
  "",
  "event: final",
  'data: {"thread_id": "t-1", "action": "answer_direct", "answer": "Hello",',
  'data:  "citations": [], "metadata": {"model": "m", "tokens_input": 3}, "failed": false}',
  "",
  "",
].join("\r\n");

const PARSED: AgentStreamEvent[] = [
  { type: "thread", thread_id: "t-1" },
  { type: "token", node: "answer_direct", delta: "Hel" },
  { type: "token", node: "answer_direct", delta: "lo" },
  {
    type: "final",
    payload: {
      thread_id: "t-1",
      action: "answer_direct",
      answer: "Hello",
      citations: [],
      metadata: {
        model: "m",
        tokens_input: 3,
        tokens_output: 0,
        latency_ms: 0,
        chunks_retrieved: 0,
        cost_usd: 0,
      },
      failed: false,
      errors: [],
    },
  },
];

describe("the agent's event stream", () => {
  it("reads every event, with CRLF lines and a data field over two lines", async () => {
    expect(await agentEvents(sse([STREAM]))).toEqual(PARSED);
  });

  it("reads the same events whatever the chunk boundaries", async () => {
    // Every 1 to 7 characters: some cuts split a CRLF or a field name.
    for (let n = 1; n <= 7; n++) {
      expect(await agentEvents(sse(cut(STREAM, n)))).toEqual(PARSED);
    }
  });

  it("takes LF and CR line endings too", async () => {
    expect(await agentEvents(sse([STREAM.replaceAll("\r\n", "\n")]))).toEqual(PARSED);
    expect(await agentEvents(sse([STREAM.replaceAll("\r\n", "\r")]))).toEqual(PARSED);
  });

  it("skips data that is not a JSON object, unknown events and invalid ones", async () => {
    const events = await agentEvents(
      sse([
        "event: token\ndata: not json\n\n",
        "event: token\ndata: [1, 2]\n\n",
        'event: surprise\ndata: {"a": 1}\n\n',
        'event: thread\ndata: {"thread_id": 7}\n\n',
        'event: error\ndata: {"message": "boom"}\n\n',
      ]),
    );
    expect(events).toEqual([{ type: "error", message: "boom" }]);
  });

  it("reads the turn's failed calls, and skips entries without a step or a kind", async () => {
    const errors = [
      { step: "classifier", kind: "billing", model: "gemini-3.8-flash" },
      { step: "query_rewrite", kind: "timeout" },
      { kind: "auth" },
      3,
    ];
    const final = { thread_id: "t", answer: "a", citations: [], metadata: {}, errors };
    const [event] = await agentEvents(sse([`event: final\ndata: ${JSON.stringify(final)}\n\n`]));
    expect(event.type === "final" && event.payload.errors).toEqual([
      { step: "classifier", kind: "billing", model: "gemini-3.8-flash" },
      { step: "query_rewrite", kind: "timeout", model: null },
    ]);
  });

  it("drops a message the end of the stream cut off", async () => {
    const events = await agentEvents(
      sse(['event: token\ndata: {"delta": "a"}\n\n', 'event: token\ndata: {"delta": "b"}']),
    );
    expect(events).toEqual([{ type: "token", node: null, delta: "a" }]);
  });

  it("keeps a multi-byte character split across chunks", async () => {
    const bytes = new TextEncoder().encode('event: token\ndata: {"delta": "ñé€"}\n\n');
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        for (const b of bytes) c.enqueue(new Uint8Array([b]));
        c.close();
      },
    });
    const events = await agentEvents(new Response(body, { status: 200 }));
    expect(events).toEqual([{ type: "token", node: null, delta: "ñé€" }]);
  });

  it("sends the question with the defaults the backend expects", async () => {
    const fetchMock = stubFetch(sse([]));
    await api.agentChatStream({ message: "hi" }, () => {});
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/api\/v1\/agent\/chat\/stream$/);
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({
      message: "hi",
      thread_id: null,
      collection_id: null,
      document_id: null,
      top_k: 5,
    });
  });

  it("releases the response when the reader stops early", async () => {
    const onCancel = vi.fn();
    stubFetch(sse([STREAM], { open: true, onCancel }));
    await expect(
      api.agentChatStream({ message: "hi" }, () => {
        throw new Error("stop");
      }),
    ).rejects.toThrow("stop");
    expect(onCancel).toHaveBeenCalled();
  });

  it("turns a failed response into an ApiError that says why", async () => {
    stubFetch(
      new Response(
        JSON.stringify({ detail: [{ loc: ["body", "message"], msg: "Field required" }] }),
        { status: 422 },
      ),
    );
    const error = await api.agentChatStream({ message: "" }, () => {}).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(422);
    expect((error as ApiError).message).toBe("message: Field required");
  });

  it("keeps a proxy's error page as the detail, with the status as the message", async () => {
    stubFetch(new Response("<html>Bad gateway</html>", { status: 502 }));
    const error = (await api
      .agentChatStream({ message: "hi" }, () => {})
      .catch((e: unknown) => e)) as ApiError;
    expect(error.message).toBe("HTTP 502");
    expect(error.detail).toBe("<html>Bad gateway</html>");
  });
});

describe("the ingestion stream", () => {
  it("reads progress, completion and errors, and drops an unknown status", async () => {
    stubFetch(
      sse([
        'event: progress\ndata: {"job_id": 4, "status": "embedding", "progress": 0.5, "processed_chunks": 3, "total_chunks": 6}\n\n',
        'event: progress\ndata: {"job_id": 4, "status": "dreaming"}\n\n',
        'event: complete\ndata: {"job_id": 4, "document_id": 12}\n\n',
        "event: error\ndata: {}\n\n",
      ]),
    );
    const events: IngestionJobEvent[] = [];
    await api.streamIngestionJob(4, (e) => events.push(e));
    expect(events).toEqual([
      {
        type: "progress",
        job_id: 4,
        status: "embedding",
        progress: 0.5,
        processed_chunks: 3,
        total_chunks: 6,
      },
      { type: "complete", job_id: 4, document_id: 12 },
      { type: "error", job_id: undefined, message: "ingestion error" },
    ]);
  });
});

describe("formatting", () => {
  it("writes dates and times in UTC", () => {
    expect(fmtDate("2026-05-16T23:35:00-03:00")).toBe("2026-05-17 02:35 UTC");
  });

  it("gives a turn's date only when it is not today's", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-11-08T12:00:00Z"));
    expect(fmtTurnTime("2026-11-08T09:05:03.250Z")).toBe("09:05:03.250 UTC");
    expect(fmtTurnTime("2026-11-07T23:59:59.000Z")).toBe("2026-11-07 23:59:59.000 UTC");
  });

  it("shows a missing number as a dash", () => {
    expect(fmtNumber(12345)).toBe("12,345");
    expect(fmtNumber(null)).toBe("—");
    expect(fmtNumber(Number.NaN)).toBe("—");
    expect(fmtCost(0.01644)).toBe("$0.0164");
    expect(fmtCost(undefined)).toBe("—");
  });
});
