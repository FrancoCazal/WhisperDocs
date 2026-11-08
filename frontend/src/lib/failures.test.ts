import { describe, expect, it } from "vitest";
import { fmtAgo, type ModelRoleState } from "./api";
import { failingGroups } from "./failures";

function role(over: Partial<ModelRoleState>): ModelRoleState {
  return {
    model: "gemini-3.8-flash",
    state: "unknown",
    calls: 0,
    failures_in_a_row: 0,
    last_ok_at: null,
    last_failure_at: null,
    last_failure_kind: null,
    ...over,
  };
}

const failing = (kind: string, at: string, model = "gemini-3.8-flash") =>
  role({
    model,
    state: "failing",
    calls: 2,
    failures_in_a_row: 2,
    last_failure_at: at,
    last_failure_kind: kind,
  });

describe("the failing models", () => {
  it("are one group per model and kind, in the order of the roles", () => {
    const groups = failingGroups({
      since: "2026-11-08T09:00:00Z",
      roles: {
        classifier: failing("billing", "2026-11-08T10:00:00Z"),
        query_rewrite: failing("billing", "2026-11-08T10:05:00Z"),
        rag_generation: role({ model: "us.anthropic.claude-sonnet-4-5", state: "ok", calls: 3 }),
        clarification: failing("rate_limit", "2026-11-08T10:01:00Z"),
        embedding: failing("billing", "2026-11-08T10:02:00Z", "text-embedding-3-small"),
        summarize: role({}),
      },
    });

    expect(groups).toEqual([
      {
        model: "gemini-3.8-flash",
        kind: "billing",
        roles: ["classifier", "query_rewrite"],
        lastFailureAt: "2026-11-08T10:05:00Z",
      },
      {
        model: "gemini-3.8-flash",
        kind: "rate_limit",
        roles: ["clarification"],
        lastFailureAt: "2026-11-08T10:01:00Z",
      },
      {
        model: "text-embedding-3-small",
        kind: "billing",
        roles: ["embedding"],
        lastFailureAt: "2026-11-08T10:02:00Z",
      },
    ]);
  });

  it("are none while every role works or has no call yet", () => {
    expect(
      failingGroups({
        since: "2026-11-08T09:00:00Z",
        roles: { classifier: role({ state: "ok", calls: 1 }), summarize: role({}) },
      }),
    ).toEqual([]);
  });
});

describe("fmtAgo", () => {
  const now = Date.parse("2026-11-08T12:00:00Z");
  it.each([
    ["2026-11-08T11:59:30Z", "JUST NOW"],
    ["2026-11-08T11:56:00Z", "4 MIN AGO"],
    ["2026-11-08T09:30:00Z", "2 H AGO"],
    ["2026-11-05T12:00:00Z", "3 D AGO"],
    // A clock a little ahead of this one's.
    ["2026-11-08T12:00:05Z", "JUST NOW"],
  ])("%s is %s", (iso, text) => {
    expect(fmtAgo(iso, now)).toBe(text);
  });
});
