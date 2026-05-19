"""Token counting utilities.

IMPORTANT: tiktoken is OpenAI-specific. Claude and Gemini tokenize differently
(counts can diverge ~10-20% vs tiktoken in Spanish). We use tiktoken as a
cross-provider *estimator* for the compaction trigger only. Exact token counts
for observability come from each provider's response.usage_metadata.
"""

import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens_estimate(messages: list) -> int:
    """Estimate total tokens in a message list for the compaction heuristic.

    TODO: when tool calls are introduced, include `tool_calls` and
    `additional_kwargs` payload size — ignoring them would under-count.
    """
    total_text = "\n".join(getattr(m, "content", "") or "" for m in messages)
    return len(_encoder.encode(total_text))
