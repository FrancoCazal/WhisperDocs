"""Canned-response node for queries the classifier flagged as out-of-scope.

No LLM call. Returns a polite redirect that names the corpus boundary so the
user understands the system did not silently fail. Avoids the hallucination
risk of routing OOS queries through `answer_direct` (which can fabricate
answers without retrieval grounding).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agent.state import AgentState

_CANNED_RESPONSE_ES = (
    "Esa pregunta está fuera del alcance de los documentos cargados. "
    "Probá una pregunta sobre el contenido de tu colección."
)


async def out_of_scope_node(state: AgentState) -> dict:
    msg_metadata = {
        "model": None,
        "tokens_input": 0,
        "tokens_output": 0,
        "latency_ms": 0,
        "chunks_retrieved": 0,
    }
    prev = state.get("metadata") or {}
    return {
        "messages": [
            AIMessage(
                content=_CANNED_RESPONSE_ES,
                additional_kwargs={"metadata": msg_metadata},
            )
        ],
        "metadata": {**prev, "generation_model": None},
    }
