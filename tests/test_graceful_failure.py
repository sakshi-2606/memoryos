"""
Graceful tier-failure handling (pytest, offline).

Stubs memory-tier methods to raise, and asserts the manager records the failure
in the trace (status=error + error_detail) and DEGRADES the turn instead of
crashing. No real API calls (failing methods are stubbed; the rest is in-memory
or tiktoken).

Run from repo root (venv active):
    python -m pytest tests/test_graceful_failure.py -v
"""

import dataclasses
import uuid

import pytest

from memoryos.config import AppConfig
from memoryos.memory.manager import MemoryManager
from memoryos.memory.trace import TierStatus


@pytest.fixture
def make_manager():
    created: list[MemoryManager] = []

    def _factory(config: AppConfig | None = None) -> MemoryManager:
        mgr = MemoryManager(
            config or AppConfig(),
            session_id=f"graceful-{uuid.uuid4().hex[:8]}",
        )
        created.append(mgr)
        return mgr

    yield _factory

    for m in created:
        try:
            m.long_term._chroma.delete_collection(m.long_term._collection.name)
        except Exception:
            pass


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def test_retrieve_failure_degrades_to_no_context(make_manager):
    mgr = make_manager()
    mgr.long_term.retrieve = _boom   # vector store "down"

    messages, read = mgr.assemble_prompt_traced("what database am I using?")

    assert read.long_term.status == TierStatus.ERROR
    assert "boom" in read.long_term.error_detail
    assert read.long_term.retrieved == []
    # turn still produced a usable prompt (system message present)
    assert messages and messages[0]["role"] == "system"


def test_long_term_add_failure_degrades(make_manager):
    mgr = make_manager()
    mgr.long_term.add = _boom   # embedding/store failure

    wt = mgr.add_message_traced("user", "I am building a RAG system with ChromaDB", label="user")

    assert wt.long_term.status == TierStatus.ERROR
    assert wt.long_term.stored is False
    assert "boom" in wt.long_term.error_detail
    # short-term still succeeded — the turn didn't crash
    assert wt.short_term.status == TierStatus.OK


def test_summarisation_failure_degrades(make_manager):
    # tiny budget so eviction (and thus summarisation) fires quickly
    cfg = dataclasses.replace(
        AppConfig(),
        memory=dataclasses.replace(
            AppConfig().memory,
            short_term_token_budget=5,
            short_term_evict_to_fraction=0.4,
        ),
    )
    mgr = make_manager(cfg)
    mgr.long_term.add = lambda *a, **k: "stub-id"   # avoid real embedding calls
    mgr.episodic.summarise_turns = _boom            # summariser "down"

    wt = None
    for i in range(6):
        wt = mgr.add_message_traced(
            "user", f"this is message number {i} with several words", label="user"
        )
        if wt.evictions:
            break

    assert wt.evictions, "no eviction occurred — budget too high for the test"
    assert wt.episodic.status == TierStatus.ERROR
    assert "boom" in wt.episodic.error_detail
    # evictions are still recorded even though summarisation failed
    assert all(e.tier == "short_term" for e in wt.evictions)