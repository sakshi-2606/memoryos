"""
API-free test: deleting a session PURGES its long-term collection, so "forget"
actually forgets and a later session with the same id cannot resurrect the data.

Stubs embeddings (no network); ChromaDB runs locally.

Run from repo root (venv active):
    python -m pytest tests/test_session_delete_purges.py -v
"""

import uuid

import pytest

from memoryos.config import AppConfig
from memoryos.session_store import InMemorySessionStore


@pytest.fixture
def store():
    s = InMemorySessionStore(AppConfig())
    yield s
    # teardown: purge any surviving collections
    for sid in list(s.list_ids()):
        s.delete(sid)


def _add_fact(agent, text="a durable fact worth keeping"):
    """Plant one memory without hitting the network (stubbed embedding)."""
    dim = agent.memory.config.model.embedding_dim
    agent.memory.long_term._embed = lambda t: [0.1] * dim
    agent.memory.long_term.add("user", text, turn_index=0)


def test_delete_purges_long_term_no_resurrection(store):
    sid = f"purge-{uuid.uuid4().hex[:8]}"

    agent = store.get_or_create(sid)
    _add_fact(agent)
    assert agent.memory.long_term.count() == 1

    assert store.delete(sid) is True

    # a fresh session with the SAME id must start empty — the old vectors are gone
    agent2 = store.get_or_create(sid)
    assert agent2.memory.long_term.count() == 0


def test_delete_unknown_session_returns_false(store):
    assert store.delete("never-created-session") is False