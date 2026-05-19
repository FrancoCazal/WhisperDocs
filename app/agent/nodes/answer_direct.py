import logging

from langchain_core.messages import AIMessage

from app.agent.prompts import ANSWER_DIRECT_SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.utils.messages import build_prompt_messages
from app.agent.utils.streaming import stream_generate
from app.services.llm import llm_service_for

logger = logging.getLogger(__name__)


async def answer_direct_node(state: AgentState) -> dict:
    provider, model, temperature = llm_service_for("answer_direct")

    try:
        resp = await stream_generate(
            provider,
            system=ANSWER_DIRECT_SYSTEM_PROMPT,
            messages=build_prompt_messages(state),
            model=model,
            temperature=temperature,
            max_tokens=200,
            node="answer_direct",
        )
        text = resp.text.strip() or "¡Hola! ¿En qué puedo ayudarte con la documentación?"
        md = {
            "generation_model": resp.model,
            "generation_tokens_input": resp.tokens_input,
            "generation_tokens_output": resp.tokens_output,
            "generation_latency_ms": resp.latency_ms,
        }
    except Exception:
        logger.exception("answer_direct_failed")
        text = "Tuve un problema generando la respuesta. ¿Podés intentarlo de nuevo?"
        md = {"generation_model": model}

    msg_metadata = {
        "model": md.get("generation_model"),
        "tokens_input": md.get("generation_tokens_input", 0),
        "tokens_output": md.get("generation_tokens_output", 0),
        "latency_ms": md.get("generation_latency_ms", 0),
        "chunks_retrieved": 0,
    }
    prev = state.get("metadata") or {}
    return {
        "messages": [AIMessage(content=text, additional_kwargs={"metadata": msg_metadata})],
        "metadata": {**prev, **md},
    }
