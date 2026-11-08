"""Canned-response node for queries the classifier flagged as out-of-scope.

No LLM call. Returns a polite redirect that names the corpus boundary so the
user understands the system did not silently fail. Avoids the hallucination
risk of routing OOS queries through `answer_direct` (which can fabricate
answers without retrieval grounding).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.state import AgentState
from app.agent.utils.fixed_replies import fixed_reply
from app.agent.utils.messages import stamped
from app.agent.utils.turn_errors import errors_so_far


async def out_of_scope_node(state: AgentState) -> dict:
    msg_metadata = {
        "model": None,
        "tokens_input": 0,
        "tokens_output": 0,
        "latency_ms": 0,
        "chunks_retrieved": 0,
    }
    prev = state.get("metadata") or {}
    # No call of its own, but a failed compaction before it still shows.
    errors = errors_so_far(state)
    return {
        "messages": [
            AIMessage(
                content=fixed_reply("out_of_scope", state["messages"]),
                additional_kwargs=stamped(
                    metadata=msg_metadata, action="out_of_scope", errors=errors
                ),
            )
        ],
        "metadata": {**prev, "generation_model": None},
    }
