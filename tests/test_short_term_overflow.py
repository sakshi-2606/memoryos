"""
API-free test: short-term TOKEN-budget eviction returns the overflow batch.

Uses a simple word-count token counter (no tiktoken, no network) so the budget
arithmetic is easy to reason about. Proves that exceeding the budget evicts the
oldest turns down to the watermark and hands them back (for summarisation),
rather than silently dropping them.

Run from repo root (venv active):
    python -m pytest tests/test_short_term_overflow.py -v
"""

from memoryos.config import MemoryConfig
from memoryos.memory.short_term import ShortTermMemory


def _words(text: str) -> int:
    return len(text.split())


def test_window_overflow_captured() -> None:
    # budget 10 "tokens" (= words), evict down to 50% (watermark = 5)
    cfg = MemoryConfig(short_term_token_budget=10, short_term_evict_to_fraction=0.5)
    st = ShortTermMemory(cfg, token_counter=_words)

    # three 3-word turns: 3 -> 6 -> 9 tokens, all within budget (no eviction)
    for i in range(3):
        _msg, evicted = st.add("user", "one two three")
        assert evicted == [], f"unexpected eviction at turn {i}"
    assert st.total_tokens() == 9
    assert len(st) == 3

    # a 4th 3-word turn -> 12 tokens > budget 10: evict oldest down to <= 5
    _msg, evicted = st.add("user", "four five six")
    # pop turn0 (->9), turn1 (->6), turn2 (->3 <= 5, stop) => evicted 0,1,2
    assert [m.turn_index for m in evicted] == [0, 1, 2]
    assert st.total_tokens() == 3        # only the newest turn remains
    assert len(st) == 1

    # a single over-budget turn is kept, not dropped (never evict the newest)
    st2 = ShortTermMemory(cfg, token_counter=_words)
    _msg, evicted = st2.add("user", "a b c d e f g h i j k l")  # 12 > 10
    assert evicted == []
    assert len(st2) == 1