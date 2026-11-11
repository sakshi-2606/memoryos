"""
API-free test: the session registry is thread-safe under concurrent creation.

Uses a fake agent factory (no real MemoryAgent, no chromadb/OpenAI) and threads
synchronised on a barrier to maximise contention. Deterministic assertions: the
factory runs exactly once per id, and every racing caller gets the same instance.

Run from repo root (venv active):
    python -m pytest tests/test_store_concurrency.py -v
"""

import threading
import time
from types import SimpleNamespace

from memoryos.session_store import InMemorySessionStore


def _counting_factory():
    """A factory that records each creation and is slow enough to widen the race."""
    calls: list[str] = []
    calls_lock = threading.Lock()

    def factory(config, session_id):
        with calls_lock:
            calls.append(session_id)
        time.sleep(0.005)   # simulate construction cost → widens the race window
        return SimpleNamespace(session_id=session_id)

    return factory, calls


def test_concurrent_get_or_create_same_id_creates_once():
    factory, calls = _counting_factory()
    store = InMemorySessionStore(config=None, agent_factory=factory)

    n = 50
    barrier = threading.Barrier(n)
    results: list[object] = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait()                         # release all threads together
        agent = store.get_or_create("same")
        with results_lock:
            results.append(agent)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls.count("same") == 1            # created exactly once
    assert len(results) == n
    assert all(a is results[0] for a in results)   # everyone got the same instance


def test_concurrent_get_or_create_distinct_ids_each_once():
    factory, calls = _counting_factory()
    store = InMemorySessionStore(config=None, agent_factory=factory)

    ids = [f"s{i}" for i in range(30)]
    barrier = threading.Barrier(len(ids))

    def worker(sid):
        barrier.wait()
        store.get_or_create(sid)

    threads = [threading.Thread(target=worker, args=(sid,)) for sid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(calls) == sorted(ids)        # each id created exactly once
    assert len(store) == len(ids)