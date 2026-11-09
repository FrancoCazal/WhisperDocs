import { describe, expect, it } from "vitest";
import { traceUrl } from "./traceUrl";

const ID = "0af7651916cd43dd8448eb211c80319c";
const LANGFUSE = "http://localhost:3000/project/whisperdocs/traces/{trace_id}";

describe("trace links", () => {
  it("puts the id where the template says", () => {
    expect(traceUrl(ID, LANGFUSE)).toBe(`http://localhost:3000/project/whisperdocs/traces/${ID}`);
    expect(traceUrl(ID, "https://jaeger.example/trace/{trace_id}?uiFind={trace_id}")).toBe(
      `https://jaeger.example/trace/${ID}?uiFind=${ID}`,
    );
  });

  it("gives no link without a template that names the id", () => {
    expect(traceUrl(ID, "")).toBeNull();
    expect(traceUrl(ID, "http://localhost:3000/traces/")).toBeNull();
  });

  it("links only an OpenTelemetry trace id", () => {
    expect(traceUrl("not-a-trace", LANGFUSE)).toBeNull();
    expect(traceUrl(ID.toUpperCase(), LANGFUSE)).toBeNull();
    expect(traceUrl(ID.slice(1), LANGFUSE)).toBeNull();
  });

  it("never links a script or a relative path", () => {
    expect(traceUrl(ID, "javascript:alert('{trace_id}')")).toBeNull();
    expect(traceUrl(ID, "/traces/{trace_id}")).toBeNull();
  });
});
