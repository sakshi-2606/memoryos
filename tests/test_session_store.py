"""
API-free test: the in-memory session store lifecycle.

Uses a fake agent factory so no real MemoryAgent (and no chromadb/OpenAI client)
is constructed.

Run from repo root (venv active):
    python -m pytest tests/test_session_store.py -v
"""

from types import SimpleNamespace

from memoryos.session_store import InMemorySessionStore


def test_in_memory_store_lifecycle():
    calls = []

    def fake_factory(config, session_id):
        calls.append(session_id)
        return SimpleNamespace(session_id=session_id)

    store = InMemorySessionStore(config=None, agent_factory=fake_factory)

    a = store.get_or_create("s1")
    assert a.session_id == "s1"
    assert store.get_or_create("s1") is a      # cached, not recreated
    assert calls == ["s1"]                      # factory called exactly once

    assert "s1" in store
    assert store.get("s1") is a
    assert store.get("missing") is None
    assert store.list_ids() == ["s1"]
    assert len(store) == 1

    assert store.delete("s1") is True
    assert store.delete("s1") is False          # already gone
    assert "s1" not in store
    assert store.list_ids() == []