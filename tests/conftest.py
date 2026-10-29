"""
Shared pytest fixtures for the eval suites.

conftest.py fixtures are auto-available to every test under tests/ with no import.
"""

import pytest

from memoryos.config import AppConfig
from memoryos.agent import MemoryAgent


@pytest.fixture
def make_agent():
    """
    Build a MemoryAgent — optionally with a custom config and pre-planted facts —
    and clean up its Chroma collection when the test finishes.

    Args to the factory:
      session_id: unique session namespace (keeps Chroma collections isolated)
      facts:      optional iterable of Fact objects planted into long-term
                  (embeddings only; use for retrieval/leakage tests)
      config:     optional AppConfig override (e.g. a low summarisation threshold)
    """
    created: list[MemoryAgent] = []

    def _factory(session_id: str, facts=(), config: AppConfig | None = None) -> MemoryAgent:
        agent = MemoryAgent(config or AppConfig(), session_id=session_id)
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