"""
API-free test: per-session turn locks serialise one session while letting
distinct sessions run concurrently.

Fake agent factory (no MemoryAgent / chromadb / OpenAI); threads prove the
locking behaviour directly on the store primitive the API wraps each turn in.

Run from repo root (venv active):
    python -m pytest tests/test_session_turn_lock.py -v
"""

import threading
import time
from types import SimpleNamespace

from memoryos.session_store import InMemorySessionStore


def _fake_store() -> InMemorySessionStore:
    return InMemorySessionStore(
        config=None,
        agent_factory=lambda config, session_id: SimpleNamespace(session_id=session_id),
    )


def test_session_lock_is_cached_per_id():
    store = _fake_store()
    assert store.session_lock("s1") is store.session_lock("s1")   # same lock reused
    assert store.session_lock("a") is not store.session_lock("b")  # per-session


def test_same_session_serialises_concurrent_turns():
    store = _fake_store()

    active = 0
    max_active = 0
    guard = threading.Lock()
    barrier = threading.Barrier(10)

    def turn():
        nonlocal active, max_active
        barrier.wait()
        with store.session_lock("s1"):
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)          # hold the "turn" open
            with guard:
                active -= 1

    threads = [threading.Thread(target=turn) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_active == 1   # never two turns inside one session's lock at once


def test_distinct_sessions_run_in_parallel():
    store = _fake_store()

    a_entered = threading.Event()
    overlapped = False

    def work_a():
        with store.session_lock("a"):
            a_entered.set()
            time.sleep(0.05)          # hold A's lock open

    def work_b():
        nonlocal overlapped
        with store.session_lock("b"):
            # if B can run while A still holds A's lock, the sessions are independent
            overlapped = a_entered.wait(0.5)

    ta = threading.Thread(target=work_a)
    tb = threading.Thread(target=work_b)
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    assert overlapped   # B proceeded while A held its lock → parallel across sessions


def test_delete_discards_the_session_lock():
    store = _fake_store()
    store.get_or_create("s1")
    first = store.session_lock("s1")

    assert store.delete("s1") is True

    second = store.session_lock("s1")   # a later same-id session gets a fresh lock
    assert second is not first