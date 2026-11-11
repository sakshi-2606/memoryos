"""
API-free test: long-term forgetting fires over capacity and is recorded in the
per-turn trace. The embedding call is stubbed (no network); ChromaDB runs locally.

Run from repo root (venv active):
    python -m pytest tests/test_long_term_forgetting.py -v
"""

import dataclasses
import uuid

import pytest

from memoryos.config import AppConfig
from memoryos.memory.manager import MemoryManager
from memoryos.memory.trace import EvictionPolicy


@pytest.fixture
def make_manager():
    created: list[MemoryManager] = []

    def _factory(config: AppConfig | None = None) -> MemoryManager:
        mgr = MemoryManager(
            config or AppConfig(),
            session_id=f"forget-{uuid.uuid4().hex[:8]}",
        )
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


def _capped_config(policy: str = "fifo", cap: int = 3) -> AppConfig:
    """AppConfig with a tiny long-term capacity so forgetting fires quickly."""
    return dataclasses.replace(
        AppConfig(),
        memory=dataclasses.replace(
            AppConfig().memory,
            forgetting_policy=policy,
            long_term_max_episodes=cap,
        ),
    )


def test_forgetting_enforces_capacity_and_is_traced(make_manager):
    # fifo is deterministic (oldest evicted) — enough to prove the wiring fires.
    mgr = make_manager(_capped_config(policy="fifo", cap=3))

    saw_forget = False
    wt = None
    for i in range(5):
        wt = mgr.add_message_traced(
            "user", f"distinct durable fact number {i} worth storing", label="user"
        )
        if any(e.tier == "long_term" for e in wt.evictions):
            saw_forget = True

    # capacity enforced on the store itself
    assert mgr.long_term.count() == 3
    # the forgetting fired and was recorded in the trace with the right policy
    assert saw_forget, "expected long-term forgetting to fire over capacity"
    lt_evictions = [e for e in wt.evictions if e.tier == "long_term"]
    assert lt_evictions
    assert all(e.policy == EvictionPolicy.FORGETTING for e in lt_evictions)


def test_under_capacity_forgets_nothing(make_manager):
    mgr = make_manager(_capped_config(policy="hybrid", cap=100))

    for i in range(4):
        wt = mgr.add_message_traced(
            "user", f"distinct durable fact number {i} worth storing", label="user"
        )
        assert not any(e.tier == "long_term" for e in wt.evictions)

    assert mgr.long_term.count() == 4