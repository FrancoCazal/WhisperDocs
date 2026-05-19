# LangGraph: agents and state graphs

LangGraph is a library developed by the LangChain team for building LLM agents as explicit state graphs. Unlike high-level frameworks that hide the agent's logic behind an abstraction ("run this chain and trust me"), LangGraph exposes the graph: each node is a function, each edge is a routing condition, and the state is an object that flows between nodes. This makes it especially well-suited for systems where you need precise control, deep debugging, and persistent memory.

## Why state graphs

LLM agents tend to be cycles of "think → act → observe → think". Implementing that with imperative code (if/else, while loops) works for simple cases but becomes unmanageable when there are multiple branches, error recovery, parallelism, or memory between turns.

A state graph models this naturally:

- **Nodes**: processing steps (call the LLM, run a tool, summarize history).
- **Edges**: transitions, with conditions that decide which one to take based on the current state.
- **State**: an object that is passed between nodes and that each node can update.

The result is declarative, observable code that's easy to test node by node without running the whole graph.

## Typed state

The state is typically a `TypedDict` declaring which fields flow through the graph:

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_id: str
    collection_id: int | None
    pending_action: str | None
```

The `Annotated[..., operator.add]` is a "reducer": it tells the graph how to combine the new value with the existing one when a node returns a change to that field. With `operator.add`, lists are concatenated; without a reducer, the new value replaces the previous one.

This is important for the message history: each node that emits a new message appends it instead of replacing the whole history.

## Defining the graph

```python
from langgraph.graph import StateGraph, END

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decide_action", decide_action_node)
    graph.add_node("search_docs", search_docs_node)
    graph.add_node("ask_clarification", clarification_node)
    graph.add_node("answer_direct", direct_answer_node)

    graph.set_entry_point("decide_action")

    graph.add_conditional_edges(
        "decide_action",
        route_action,
        {
            "search_docs": "search_docs",
            "ask_clarification": "ask_clarification",
            "answer_direct": "answer_direct",
        }
    )

    for node in ["search_docs", "ask_clarification", "answer_direct"]:
        graph.add_edge(node, END)

    return graph.compile()
```

Nodes are functions that receive the state and return a dictionary with the changes:

```python
async def decide_action_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    decision = await llm_classify(last_message)
    return {"pending_action": decision}

def route_action(state: AgentState) -> str:
    return state["pending_action"]

async def search_docs_node(state: AgentState) -> dict:
    query = state["messages"][-1].content
    chunks = await retriever.retrieve(query, state["collection_id"])
    answer = await llm_answer_with_context(query, chunks)
    return {"messages": [AIMessage(content=answer)]}
```

The compiled graph is invocable with an initial state:

```python
from langchain_core.messages import HumanMessage

agent = build_agent_graph()
result = await agent.ainvoke({
    "messages": [HumanMessage(content="What is pgvector?")],
    "user_id": "user-123",
    "collection_id": 1,
    "pending_action": None,
})
```

## Checkpointers: persistent memory

One of LangGraph's most valuable capabilities is the checkpointer: a component that persists the graph's state between invocations, grouped by `thread_id`. This enables natural multi-turn conversations without the agent code having to manage history manually.

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
agent = build_agent_graph().compile(checkpointer=checkpointer)

# First invocation
config = {"configurable": {"thread_id": "user-123"}}
result = await agent.ainvoke(
    {"messages": [HumanMessage("Hi")]},
    config=config,
)

# Second invocation: automatically includes the history
result = await agent.ainvoke(
    {"messages": [HumanMessage("What were we talking about?")]},
    config=config,
)
```

The agent "remembers" the conversation because the checkpointer loads the state associated with `thread_id` before each run, and saves it after. Each thread_id is isolated.

Official checkpointer options:

- **MemorySaver**: in-memory, lost on restart. Useful for tests.
- **SqliteSaver**: in SQLite. Embedded, no external services.
- **PostgresSaver**: persistent in PostgreSQL. The choice for production.
- **Redis**: for extreme low latency at the cost of durability.

Building a custom checkpointer is possible but rarely justified. The official ones cover most cases.

## Conditional edges vs static edges

- **Static edge** (`graph.add_edge(A, B)`): after A, always go to B.
- **Conditional edge** (`graph.add_conditional_edges(A, router, map)`): after A, run the `router` function that returns a string, and use that string to pick the next node according to `map`.

The router is a function that receives the state and returns the name of the branch to take. It can be a simple heuristic, or it can invoke the LLM to decide ("given this state, what's the next action?").

## Cycles and multi-step flows

Graphs can have cycles. A typical pattern:

```
plan → act → observe → (ready? yes → END, no → back to plan)
```

```python
graph.add_edge("plan", "act")
graph.add_edge("act", "observe")
graph.add_conditional_edges("observe", check_done, {
    "continue": "plan",
    "done": END,
})
```

To prevent infinite loops, you can limit the number of iterations with `recursion_limit`:

```python
result = await agent.ainvoke(state, {"configurable": {"thread_id": "x"}, "recursion_limit": 10})
```

## Streaming state

Instead of waiting for the end, you can stream the state as the nodes update it:

```python
async for event in agent.astream(initial_state, config):
    print(event)  # includes each per-node update
```

Useful for UIs that show real-time progress ("Searching documents...", "Generating answer...").

There's also `astream_events` for more granular streaming, including LLM tokens as they're generated.

## Tools and ReAct-style agents

Beyond the general graph pattern, LangGraph provides helpers to build ReAct-style (reason-act) agents with tool calling:

```python
from langgraph.prebuilt import create_react_agent

def search_docs(query: str) -> str:
    """Search the knowledge base."""
    return retrieve(query)

def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)

agent = create_react_agent(
    model=llm,
    tools=[search_docs, calculate],
    checkpointer=checkpointer,
)
```

The agent decides which tool to invoke based on the question, observes the result, decides whether to answer or use another tool, and so on. It's the simplest pattern to get started.

## Testing nodes

Nodes are plain functions, trivially testable:

```python
import pytest

@pytest.mark.asyncio
async def test_decide_action_routes_to_search():
    state = {
        "messages": [HumanMessage("What is pgvector?")],
        "user_id": "test",
        "collection_id": 1,
        "pending_action": None,
    }
    result = await decide_action_node(state)
    assert result["pending_action"] == "search_docs"
```

The whole graph is tested integration-style:

```python
@pytest.mark.asyncio
async def test_agent_answers_with_context():
    agent = build_agent_graph().compile()
    result = await agent.ainvoke({
        "messages": [HumanMessage("What is RAG?")],
        "user_id": "test",
        "collection_id": 1,
        "pending_action": None,
    })
    assert "retrieval" in result["messages"][-1].content.lower()
```

To isolate tests from the real LLM, mock `llm_classify` and `llm_answer_with_context`.

## Observability

LangGraph integrates with LangSmith for tracing executions: every invocation generates a visual trace with each node, its input, its output, tokens consumed, latency. Invaluable for debugging.

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."
```

Without LangSmith, the same effect can be achieved by manually logging in each node with consistent structures.

## Comparison with alternatives

| Framework | Abstraction level | Control | Observable |
|-----------|-------------------|---------|------------|
| LangGraph | Low (explicit graphs) | High | Very good |
| LangChain AgentExecutor | High (pre-built ReAct) | Medium | Medium |
| CrewAI | High (role-based) | Low | Good |
| AutoGen | High (multi-agent conversation) | Low | Good |
| Custom with OpenAI function calling | Very low (everything by hand) | Maximum | Depends on you |

LangGraph sits in the sweet spot: enough abstraction to avoid reinventing memory and routing, but low-level enough that every decision is visible and debuggable.

## Common errors

### Forgetting the reducer on `messages`

```python
messages: list[BaseMessage]  # replaces the whole list on each update
```

Result: history is lost on every node. Use `Annotated[list[BaseMessage], operator.add]`.

### Not passing thread_id

Without `configurable: thread_id`, the checkpointer can't group invocations. Memory doesn't work.

### Cycles with no exit condition

Graphs with cycles and no `recursion_limit` can burn in infinite loops consuming tokens. Always have a deterministic exit.

### Insufficient mocks in tests

Testing an entire graph with a real LLM is expensive, slow, and flaky. Mock the LLM in unit tests; use the real LLM only in integration tests or evals.

## When not to use LangGraph

- Simple cases with a single LLM call (no graph needed).
- Rigid linear flows with no conditional routing (a simple chain is enough).
- Teams without LangChain experience and tight learning curves (the graph abstraction has cognitive overhead).

For systems with multi-step agents, persistent memory, conditional routing, critical observability — LangGraph is one of the best options in the current ecosystem.

## References

- LangGraph docs: https://langchain-ai.github.io/langgraph/
- LangGraph Academy: https://academy.langchain.com/courses/intro-to-langgraph
- Checkpoint implementations: https://langchain-ai.github.io/langgraph/concepts/persistence/
- ReAct paper: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)
