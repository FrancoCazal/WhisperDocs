"""Ad-hoc local sanity check for Phase 4.1 + 4.2.

Phase 4.1 (default, --checkpointer=memory):
    MemorySaver, 4 canonical cases, no persistence.

Phase 4.2 (--checkpointer=postgres):
    AsyncPostgresSaver via init_checkpointer(). Runs a multi-turn test
    (2 invokes on same thread_id) to verify memory persistence.

Requires live provider credentials (GCP_PROJECT + ADC, AWS_*). For postgres
mode, also DATABASE_URL must be reachable.

Usage:
    python -m scripts.test_agent_local
    python -m scripts.test_agent_local --checkpointer=postgres
"""

import argparse
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.graph import build_agent_graph

CASES = [
    ("Pregunta técnica", "¿Qué es pgvector?", 1),
    ("Saludo", "hola, ¿cómo estás?", 1),
    ("Ambigua", "y eso?", 1),
    ("Fuera de scope", "contame un chiste", 1),
]

MULTI_TURN = [
    "Explicame qué es pgvector en una oración.",
    "¿Y qué métricas de distancia soporta?",
]


def _print_result(label: str, user_msg: str, result: dict) -> None:
    final = result["messages"][-1].content
    action = result.get("pending_action")
    md = result.get("metadata") or {}
    chunks = result.get("retrieved_chunks") or []
    print(f"\n=== {label} ===")
    print(f"USER: {user_msg}")
    print(f"ACTION: {action}")
    print(f"CHUNKS: {len(chunks)}")
    print(f"MODEL: {md.get('generation_model')}")
    print(f"ANSWER: {final}")


async def run_single_turn_cases(agent) -> None:
    for label, msg, collection_id in CASES:
        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content=msg)],
                "collection_id": collection_id,
            },
            config={
                "configurable": {"thread_id": f"test-{label}"},
                "recursion_limit": 10,
            },
        )
        _print_result(label, msg, result)


async def run_multi_turn(agent, thread_id: str) -> None:
    print(f"\n### MULTI-TURN (thread_id={thread_id}) ###")
    for i, msg in enumerate(MULTI_TURN, start=1):
        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content=msg)],
                "collection_id": 1,
            },
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 10,
            },
        )
        _print_result(f"Turn {i}", msg, result)
        print(f"TOTAL MESSAGES IN STATE: {len(result['messages'])}")


async def main(checkpointer_kind: str) -> None:
    if checkpointer_kind == "postgres":
        checkpointer = await init_checkpointer()
        try:
            agent = build_agent_graph().compile(checkpointer=checkpointer)
            await run_single_turn_cases(agent)
            import uuid

            await run_multi_turn(agent, thread_id=f"multi-{uuid.uuid4()}")
        finally:
            await close_checkpointer()
    else:
        agent = build_agent_graph().compile(checkpointer=MemorySaver())
        await run_single_turn_cases(agent)
        await run_multi_turn(agent, thread_id="multi-memory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpointer",
        choices=["memory", "postgres"],
        default="memory",
    )
    args = parser.parse_args()
    asyncio.run(main(args.checkpointer))
