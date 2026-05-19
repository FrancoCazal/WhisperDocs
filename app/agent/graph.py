from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    answer_direct_node,
    ask_clarification_node,
    check_compaction_node,
    decide_action_node,
    out_of_scope_node,
    search_docs_node,
    summarize_node,
)
from app.agent.router import route_action, route_compaction
from app.agent.state import AgentState


def build_agent_graph() -> StateGraph:
    """Factory that returns an *uncompiled* StateGraph.

    The caller compiles with the appropriate checkpointer:
      - AsyncPostgresSaver in production (§3.4)
      - MemorySaver in unit tests (§7.2)
    """
    graph = StateGraph(AgentState)
    graph.add_node("check_compaction", check_compaction_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("decide_action", decide_action_node)
    graph.add_node("search_docs", search_docs_node)
    graph.add_node("ask_clarification", ask_clarification_node)
    graph.add_node("answer_direct", answer_direct_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.add_edge(START, "check_compaction")

    graph.add_conditional_edges(
        "check_compaction",
        route_compaction,
        {
            "summarize": "summarize",
            "decide_action": "decide_action",
        },
    )

    graph.add_edge("summarize", "decide_action")

    graph.add_conditional_edges(
        "decide_action",
        route_action,
        {
            "search_docs": "search_docs",
            "ask_clarification": "ask_clarification",
            "answer_direct": "answer_direct",
            "out_of_scope": "out_of_scope",
        },
    )

    for terminal in (
        "search_docs",
        "ask_clarification",
        "answer_direct",
        "out_of_scope",
    ):
        graph.add_edge(terminal, END)

    return graph
