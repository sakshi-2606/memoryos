"""
API-free test: the ported forgetting policies (pure numpy/BoW, no network).

Run from repo root (venv active):
    python -m pytest tests/test_forgetting.py -v
"""

import pytest

from memoryos.memory.forgetting import (
    ForgettingManager, ForgettingPolicy, EvictionCandidate,
)


def _ep(eid, ts, importance=1.0, content="placeholder content here"):
    return EvictionCandidate(episode_id=eid, content=content, timestamp=ts, importance=importance)


def test_no_eviction_under_capacity():
    m = ForgettingManager(max_episodes=5)
    eps = [_ep(str(i), i) for i in range(3)]
    assert m.candidates_to_evict(eps) == []


def test_fifo_evicts_oldest():
    m = ForgettingManager(policy=ForgettingPolicy.FIFO, max_episodes=2)
    eps = [_ep("a", 1), _ep("b", 2), _ep("c", 3)]
    assert m.candidates_to_evict(eps) == ["a"]            # ts=1 is oldest


def test_importance_evicts_least_important():
    m = ForgettingManager(policy=ForgettingPolicy.IMPORTANCE, max_episodes=2)
    eps = [_ep("a", 1, importance=0.1), _ep("b", 2, importance=2.0), _ep("c", 3, importance=1.0)]
    assert m.candidates_to_evict(eps) == ["a"]            # lowest importance


def test_hybrid_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ForgettingManager(hybrid_weights=(0.5, 0.5, 0.5))


def test_hybrid_protects_distinctive_and_recent():
    # weight recency + surprise only; a distinctive, recent memory should survive
    m = ForgettingManager(
        policy=ForgettingPolicy.HYBRID,
        max_episodes=2,
        hybrid_weights=(0.3, 0.0, 0.7),
    )
    eps = [
        _ep("redundant_1", 1, content="the cat sat on the mat"),
        _ep("redundant_2", 2, content="the cat sat on the mat again"),
        _ep("distinctive", 3, content="quantum entanglement paradox zebra orbit"),
    ]
    evicted = m.candidates_to_evict(eps)
    assert len(evicted) == 1
    assert "distinctive" not in evicted                   # distinctive + recent kept