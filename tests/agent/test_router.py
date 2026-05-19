"""Router functions — pure logic, no mocks needed."""

from app.agent.router import route_action, route_compaction


def test_route_compaction_summarize_when_pending_summarize():
    assert route_compaction({"pending_action": "summarize"}) == "summarize"


def test_route_compaction_decide_when_none():
    assert route_compaction({"pending_action": None}) == "decide_action"


def test_route_compaction_decide_when_missing():
    assert route_compaction({}) == "decide_action"


def test_route_action_search_docs():
    assert route_action({"pending_action": "search_docs"}) == "search_docs"


def test_route_action_ask_clarification():
    assert route_action({"pending_action": "ask_clarification"}) == "ask_clarification"


def test_route_action_answer_direct():
    assert route_action({"pending_action": "answer_direct"}) == "answer_direct"


def test_route_action_out_of_scope():
    assert route_action({"pending_action": "out_of_scope"}) == "out_of_scope"


def test_route_action_fallback_on_invalid():
    # Defensive path — invalid state routes to ask_clarification (conservative
    # fallback: better to ask than to silently route to a generative path).
    assert route_action({"pending_action": "bogus"}) == "ask_clarification"
    assert route_action({"pending_action": None}) == "ask_clarification"
    assert route_action({}) == "ask_clarification"
