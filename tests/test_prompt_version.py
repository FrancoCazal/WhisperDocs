"""The prompt version (audit M-4): 12 hex characters that change when any
prompt the agent or /query sends does, and only then."""

from __future__ import annotations

import re

from app.agent import prompts
from app.agent.prompt_version import PROMPT_VERSION, compute_prompt_version, prompt_texts
from app.agent.schemas import AgentAction, RAGAnswer
from app.eval import judges
from app.eval.judges import JUDGE_PROMPT_VERSION
from app.eval.schemas import JudgeResult
from app.services.rag import build_user_prompt
from app.services.retriever import RetrievedChunk

SCHEMAS = (AgentAction, RAGAnswer)


def test_the_version_is_twelve_hex_characters_of_the_prompts_as_they_are():
    assert re.fullmatch(r"[0-9a-f]{12}", PROMPT_VERSION)
    assert compute_prompt_version(vars(prompts), SCHEMAS) == PROMPT_VERSION


def test_a_changed_or_added_prompt_changes_the_version():
    namespace = dict(vars(prompts))
    edited = {**namespace, "RAG_SYSTEM_PROMPT_AGENT": namespace["RAG_SYSTEM_PROMPT_AGENT"] + " "}
    added = {**namespace, "NEW_SYSTEM_PROMPT": "Be brief."}
    sections = {**namespace, "SUMMARY_SECTIONS": (*namespace["SUMMARY_SECTIONS"], "## More")}
    layout = {**namespace, "RAG_CONTEXT_SEPARATOR": "\n\n***\n\n"}

    versions = {
        compute_prompt_version(ns, SCHEMAS) for ns in (namespace, edited, added, sections, layout)
    }

    assert len(versions) == 5


def test_names_that_are_not_prompts_leave_the_version_alone():
    namespace = dict(vars(prompts))
    with_others = {**namespace, "helper": "not a constant", "LIMIT": 3, "PAIR": ("a", 1)}
    # A private fragment that builds a prompt is part of it.
    with_fragment = {**namespace, "_SHARED_RULES": "Answer briefly."}

    assert compute_prompt_version(with_others, SCHEMAS) == PROMPT_VERSION
    assert prompt_texts(with_others) == prompt_texts(namespace)
    assert compute_prompt_version(with_fragment, SCHEMAS) != PROMPT_VERSION


def test_a_structured_outputs_description_is_part_of_the_version():
    class Reworded(AgentAction):
        """Structured output of the decide_action classifier, reworded."""

    assert compute_prompt_version(vars(prompts), (Reworded, RAGAnswer)) != PROMPT_VERSION


def test_the_judges_version_is_its_own_prompts_and_verdict():
    namespace = {k: v for k, v in vars(judges).items() if k != "JUDGE_PROMPT_VERSION"}
    edited = {**namespace, "JUDGE_SYSTEM": namespace["JUDGE_SYSTEM"] + " Be strict."}

    assert re.fullmatch(r"[0-9a-f]{12}", JUDGE_PROMPT_VERSION)
    assert compute_prompt_version(namespace, (JudgeResult,)) == JUDGE_PROMPT_VERSION
    assert compute_prompt_version(edited, (JudgeResult,)) != JUDGE_PROMPT_VERSION
    assert JUDGE_PROMPT_VERSION != PROMPT_VERSION


def _chunk(chunk_id: int, title: str, chunk_index: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        document_title=title,
        chunk_index=chunk_index,
        text=text,
        relevance=0.9,
    )


def test_the_rag_prompt_as_it_is_sent():
    """The documents inside their element (audit A10), the question after it.
    Braces in a question, a title or a chunk stay as they are."""
    chunks = [_chunk(1, 'a "{x}".md', 0, "uno {dos}"), _chunk(2, "b.md", 3, "tres")]

    prompt, included = build_user_prompt("¿Qué {es} esto?", chunks)
    empty, _ = build_user_prompt("q", [])

    assert included == chunks
    assert prompt == (
        '<documents>\n[1] (from "a "{x}".md", chunk 0)\nuno {dos}\n\n---\n\n'
        '[2] (from "b.md", chunk 3)\ntres\n</documents>\n\nQuestion: ¿Qué {es} esto?\n\nAnswer:'
    )
    assert empty == "<documents>\n(no context available)\n</documents>\n\nQuestion: q\n\nAnswer:"
