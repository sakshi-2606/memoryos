"""
API-free test: episodic summarisation returns episode + evicted turns + tokens.

The real LLM summariser is stubbed out, so no OpenAI call is made and no
chromadb is imported — fast and free. Proves maybe_summarise() now hands back
a SummarisationResult carrying the facts the trace needs.

Run from repo root (venv active):
    python tests/test_episodic_result.py
"""

from memoryos.config import MemoryConfig, ModelConfig
from memoryos.memory.short_term import ShortTermMemory
from memoryos.memory.episodic import EpisodicMemory, SummarisationResult


def main() -> None:
    mem_cfg = MemoryConfig(
        short_term_window=10,
        summarisation_threshold=3,
        turns_to_summarise=3,
    )
    epi = EpisodicMemory(mem_cfg, ModelConfig())

    # stub the LLM summariser: (summary_text, tokens_used)
    epi._summarise = lambda turns: ("stubbed summary", 123)

    # fill short-term to the summarisation threshold
    st = ShortTermMemory(mem_cfg)
    for i in range(3):
        st.add("user", f"fact number {i} about the project")
    assert st.should_summarise()

    result = epi.maybe_summarise(st)
    assert result is not None, "expected a SummarisationResult"
    assert isinstance(result, SummarisationResult)
    assert result.episode.summary == "stubbed summary"
    assert result.summarization_tokens == 123
    assert len(result.evicted_turns) == 3, "should have evicted 3 turns"
    assert result.evicted_turns[0].turn_index == 0
    assert len(st) == 0, "short-term should be drained by the eviction"

    # below threshold -> no summarisation, returns None
    st2 = ShortTermMemory(mem_cfg)
    st2.add("user", "only one message so far")
    assert epi.maybe_summarise(st2) is None

    print("✓ episodic returns SummarisationResult (episode + evicted + tokens)")
    print(f"  tokens={result.summarization_tokens}, "
          f"evicted={len(result.evicted_turns)}, "
          f"episode_id={result.episode.id}")


if __name__ == "__main__":
    main()