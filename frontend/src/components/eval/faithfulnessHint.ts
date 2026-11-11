/** What the "unsupported claims" figure measures (audit E5): the report's
 * `hallucination_rate`, which the API and the gates keep under that name. */
export const UNSUPPORTED_CLAIMS_HINT =
  "The share of answers with a claim the retrieved context does not support, true or not: faithfulness to the context, not a fact check. Over the in-scope items; the out-of-scope ones are mostly refusals with nothing to support.";

export const UNSUPPORTED_ALL_ITEMS_HINT =
  "The share of all items whose answer has a claim the retrieved context does not support, true or not. The API and the thresholds call it the hallucination rate.";
