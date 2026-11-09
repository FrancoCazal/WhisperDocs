import { describe, expect, it } from "vitest";
import { flagRule } from "./instructionFlags";

describe("instruction flags", () => {
  it("names each rule of the backend's scan", () => {
    // app/services/instruction_scan.py: RULES, in its order.
    const rules = ["override", "role", "persona", "prompt_leak", "addressed", "delimiter"];
    expect(rules.map((r) => flagRule(r).label)).toEqual([
      "OVERRIDE",
      "ROLE",
      "PERSONA",
      "PROMPT LEAK",
      "ADDRESSED",
      "DELIMITER",
    ]);
    for (const r of rules) expect(flagRule(r).hint).not.toBe(flagRule("unknown_rule").hint);
  });

  it("shows a rule it does not know under its own name", () => {
    expect(flagRule("tool_call").label).toBe("TOOL CALL");
    expect(flagRule("tool_call").hint).toMatch(/instruction/);
  });
});
