"""
Cross-session leakage eval (pytest) — memory must not bleed between sessions.

INTEGRATION: real OpenAI embedding calls; needs OPENAI_API_KEY.

Guards the per-session ChromaDB namespacing (collection_name_{session_id}).
Expected to PASS on current code — isolation is already implemented — so this is
a standing guard, not an xfail. A failure here is a serious correctness bug.

Run from repo root (venv active):
    python -m pytest tests/test_leakage_eval.py -v

(Fixture `make_agent` is duplicated from test_retrieval_eval.py for now; a
later tidy can lift it into a shared conftest.py.)
"""

import uuid
from pathlib import Path

import pytest

from memoryos.config import AppConfig
from memoryos.agent import MemoryAgent
from memoryos.memeval import load_dataset

pytestmark = pytest.mark.integration

DATASET = load_dataset(Path(__file__).parent / "memeval_dataset.json")


@pytest.fixture
def make_agent():
    created: list[MemoryAgent] = []

    def _factory(session_id: str, facts) -> MemoryAgent:
        agent = MemoryAgent(AppConfig(), session_id=session_id)
        for f in facts:
            agent.memory.long_term.add("user", f.content, turn_index=0)
        created.append(agent)
        return agent

    yield _factory

    for a in created:
        try:
            a.memory.long_term._chroma.delete_collection(
                a.memory.long_term._collection.name
            )
        except Exception:
            pass


@pytest.mark.parametrize("leak", DATASET.leakage_probes, ids=lambda l: l.id)
def test_no_cross_session_leakage(make_agent, leak):
    """A fact planted in one session must not surface in another session's retrieval."""
    secret = DATASET.fact_by_id(leak.fact)

    # the session that OWNS the secret
    planted = make_agent(
        f"eval-leak-src-{leak.id}-{uuid.uuid4().hex[:8]}", [secret]
    )
    # a DIFFERENT session, with its own unrelated facts
    other_facts = DATASET.facts_for_session(leak.query_session)
    other = make_agent(
        f"eval-leak-other-{leak.id}-{uuid.uuid4().hex[:8]}", other_facts
    )

    # sanity: the query actually CAN find the secret in its own session,
    # otherwise the isolation assertion below would pass vacuously
    src_hits = [m.content for m in planted.memory.long_term.retrieve(leak.query)]
    assert secret.content in src_hits, (
        f"{leak.id}: query can't find the secret in its own session — test is vacuous"
    )

    # isolation: the secret must be absent from the other session's retrieval
    other_hits = [m.content for m in other.memory.long_term.retrieve(leak.query)]
    assert secret.content not in other_hits, (
        f"{leak.id}: LEAK — '{secret.content}' surfaced in session "
        f"{leak.query_session}, which never stored it"
    )