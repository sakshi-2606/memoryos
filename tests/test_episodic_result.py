"""
API-free test: episodic.summarise_turns() compresses a batch into an episode.

The LLM summariser is stubbed, so no OpenAI call is made. Proves summarise_turns
returns a SummarisationResult (episode + the turns + token cost) for a batch, and
None for an empty batch.

Run from repo root (venv active):
    python -m pytest tests/test_episodic_result.py -v
"""

from memoryos.config import ModelConfig
from memoryos.memory.short_term import Message
from memoryos.memory.episodic import EpisodicMemory, SummarisationResult


def test_summarisation_result() -> None:
    epi = EpisodicMemory(ModelConfig())

    # stub the LLM summariser: (summary_text, tokens_used)
    epi._summarise = lambda turns: ("stubbed summary", 123)

    turns = [
        Message(role="user", content=f"fact number {i}", turn_index=i)
        for i in range(3)
    ]

    result = epi.summarise_turns(turns)
    assert isinstance(result, SummarisationResult)
    assert result.episode.summary == "stubbed summary"
    assert result.summarization_tokens == 123
    assert len(result.evicted_turns) == 3
    assert result.episode.turn_range == (0, 2)
    assert result.episode.source_turn_count == 3
    assert len(epi) == 1

    # empty batch -> no episode
    assert epi.summarise_turns([]) is None
    assert len(epi) == 1