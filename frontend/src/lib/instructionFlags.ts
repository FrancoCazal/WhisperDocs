// The rules of app/services/instruction_scan.py: a chunk that matches one has
// text that reads as an instruction to the model. A flag is a lead for whoever
// keeps the corpus, not a verdict: a guide to prompting quotes injections, and
// retrieval does not read the flags.

export type FlagRule = { label: string; hint: string };

const RULES: Record<string, FlagRule> = {
  override: {
    label: "OVERRIDE",
    hint: 'Asks to drop the instructions so far ("ignore all previous instructions").',
  },
  role: {
    label: "ROLE",
    hint: 'A line that speaks as a role of the conversation ("SYSTEM:", "[ASSISTANT]:").',
  },
  persona: {
    label: "PERSONA",
    hint: 'Gives the model a new mode or rules ("you are now in … mode", "from now on, reply …").',
  },
  prompt_leak: {
    label: "PROMPT LEAK",
    hint: "Asks the model for its system prompt or its instructions.",
  },
  addressed: {
    label: "ADDRESSED",
    hint: 'Text addressed to the model ("NOTE TO AI ASSISTANTS", "Assistant instructions:").',
  },
  delimiter: {
    label: "DELIMITER",
    hint: "Holds a tag of an element the prompts carry data in, such as <documents>.",
  },
};

/** A rule's label and what it matches. A rule this table does not know yet
 * (a newer backend) shows under its own name. */
export function flagRule(name: string): FlagRule {
  return (
    RULES[name] ?? {
      label: name.replace(/_/g, " ").toUpperCase(),
      hint: "Text that reads as an instruction to the model.",
    }
  );
}

// Why a flag is shown, after the line that says what it is.
export const FLAGS_LEAD =
  "Found at ingestion. A lead for whoever keeps the corpus, not a verdict: the prompts hand documents to the model as data, and retrieval does not read the flags.";

export const FLAGS_HINT = `Text that reads as an instruction to the model. ${FLAGS_LEAD}`;
