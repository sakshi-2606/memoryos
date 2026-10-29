"""
API-free test: short-term window overflow is captured and returned.

No OpenAI calls and no chromadb import — this exercises only the in-memory
deque behaviour, so it's fast and free. Proves the silent-drop path now
reports what it dropped.

Run from repo root (venv active):
    python tests/test_short_term_overflow.py
"""

from memoryos.config import MemoryConfig
from memoryos.memory.short_term import ShortTermMemory


def test_window_overflow_captured() -> None:
    # tiny window so we can force overflow in a few adds
    cfg = MemoryConfig(short_term_window=3)
    st = ShortTermMemory(cfg)

    # fill to capacity — nothing should be evicted yet
    for i in range(3):
        _msg, evicted = st.add("user", f"message {i}")
        assert evicted is None, f"unexpected eviction while filling (#{i})"
    assert len(st) == 3

    # next add overflows: the oldest (turn_index 0) should come back as evicted
    _msg, evicted = st.add("user", "message 3")
    assert evicted is not None, "overflow was not captured"
    assert evicted.turn_index == 0, f"wrong item evicted: turn {evicted.turn_index}"
    assert evicted.content == "message 0"
    assert len(st) == 3, "buffer should stay at capacity"

    # one more overflow: now turn_index 1 is the one pushed out
    _msg, evicted = st.add("user", "message 4")
    assert evicted is not None
    assert evicted.turn_index == 1
    assert evicted.content == "message 1"

    print("✓ window overflow captured correctly")
    print(f"  last evicted: turn={evicted.turn_index}, content={evicted.content!r}")


if __name__ == "__main__":
    test_window_overflow_captured()