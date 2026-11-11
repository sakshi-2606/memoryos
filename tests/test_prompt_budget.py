"""
API-free test: the prompt-context token budget controls which retrieved memories
actually land in the prompt, and the trace's used_in_prompt reflects it.

Stubs long_term.retrieve (no network); token counts come from the real tiktoken
encoder, so the budget boundary is exercised for real.

Run from repo root (venv active):
    python -m pytest tests/test_prompt_budget.py -v
"""

import dataclasses
import uuid

import pytest

from memoryos.config import AppConfig
from memoryos.memory.manager import MemoryManager
from memoryos.memory.long_term import MemoryEntry


@pytest.fixture
def make_manager():
    created: list[MemoryManager] = []

    def _factory(budget: int) -> MemoryManager:
        cfg = dataclasses.replace(
            AppConfig(),
            memory=dataclasses.replace(
                AppConfig().memory, prompt_context_token_budget=budget
            ),
        )
        mgr = MemoryManager(cfg, session_id=f"budget-{uuid.uuid4().hex[:8]}")
        created.append(mgr)
        return mgr

    yield _factory

    for m in created:
        try:
            m.long_term._chroma.delete_collection(m.long_term._collection.name)
        except Exception:
            pass


def _mem(i: int) -> MemoryEntry:
    return MemoryEntry(
        id=f"m{i}",
        content=f"memory number {i} with a handful of distinctive words to spend tokens",
        role="user",
        turn_index=i,
        timestamp="2026-01-01T00:00:00",
        relevance_score=round(0.9 - i * 0.1, 3),   # descending rank
    )


def test_budget_marks_overflow_as_unused(make_manager):
    mgr = make_manager(budget=50)   # small — only the top memory or two will fit
    memories = [_mem(i) for i in range(4)]
    mgr.long_term.retrieve = lambda q: memories   # no network

    messages, read = mgr.assemble_prompt_traced("anything")

    flags = [r.used_in_prompt for r in read.long_term.retrieved]
    assert len(flags) == 4                      # all retrievals are reported
    assert flags[0] is True                     # the top-ranked one fits
    assert flags[-1] is False                   # the lowest-ranked overflow does not
    assert sum(flags) < 4                       # the budget actually excluded something
    assert read.long_term.tokens_contributed <= 50

    # the prompt carries the used memory, not the excluded one
    system_prompt = messages[0]["content"]
    assert memories[0].content in system_prompt
    assert memories[-1].content not in system_prompt


def test_generous_budget_uses_everything(make_manager):
    mgr = make_manager(budget=100_000)   # effectively unlimited
    memories = [_mem(i) for i in range(4)]
    mgr.long_term.retrieve = lambda q: memories

    _, read = mgr.assemble_prompt_traced("anything")

    assert all(r.used_in_prompt for r in read.long_term.retrieved)   # nothing dropped