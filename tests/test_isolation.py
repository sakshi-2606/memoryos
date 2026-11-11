"""
API-free test: memory does not bleed across sessions.

Two guarantees, both offline (stubbed embeddings, so CI gates them on every merge):
  1. long-term — a fact stored in session A never surfaces in session B's
     retrieval. Each session is a physically separate ChromaDB collection.
  2. short-term + episodic — these in-memory tiers live on the per-session
     manager, so writing one session's buffers leaves another's untouched.

The integration leakage test (test_leakage_eval.py) is the same guarantee under
REAL embeddings; this is the fast version that runs in CI.

Run from repo root (venv active):
    python -m pytest tests/test_isolation.py -v
"""

import uuid

import pytest

from memoryos.config import AppConfig
from memoryos.memory.manager import MemoryManager


@pytest.fixture
def make_manager():
    created: list[MemoryManager] = []

    def _factory() -> MemoryManager:
        mgr = MemoryManager(AppConfig(), session_id=f"iso-{uuid.uuid4().hex[:8]}")
        # no network: every stored memory gets the same stub embedding
        dim = mgr.config.model.embedding_dim
        mgr.long_term._embed = lambda text: [0.1] * dim
        created.append(mgr)
        return mgr

    yield _factory

    for m in created:
        try:
            m.long_term._chroma.delete_collection(m.long_term._collection.name)
        except Exception:
            pass


def test_long_term_does_not_leak_across_sessions(make_manager):
    a = make_manager()
    b = make_manager()

    secret = "the launch code is purple giraffe seventeen"
    a.long_term.add("user", secret, turn_index=0)
    b.long_term.add("user", "b's own unrelated note about the weather", turn_index=0)

    query = "what is the launch code"

    # non-vacuous: the session that STORED the secret can retrieve it
    a_hits = [m.content for m in a.long_term.retrieve(query)]
    assert secret in a_hits, "setup broken — A can't find its own secret"

    # isolation: the other session never sees it
    b_hits = [m.content for m in b.long_term.retrieve(query)]
    assert secret not in b_hits, f"LEAK — A's secret surfaced in session B: {b_hits}"


def test_in_memory_tiers_are_per_session(make_manager):
    a = make_manager()
    b = make_manager()

    # structurally separate instances — no shared buffers between sessions
    assert a.short_term is not b.short_term
    assert a.episodic is not b.episodic

    # write a full turn into A only
    a.add_message_traced("user", "session A is talking about rockets and orbits", label="user")

    # A's short-term buffer filled; B's stays empty
    assert len(a.short_term) > 0
    assert len(b.short_term) == 0
    assert len(b.episodic) == 0