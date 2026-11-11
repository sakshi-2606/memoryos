"""
API-free test: /chat runs the turn OFF the event loop.

Proves two things with stubbed agents (no OpenAI) by driving the ASGI app through
httpx.AsyncClient wrapped in asyncio.run() — no new test dependency:
  1. distinct sessions run concurrently (the loop isn't blocked);
  2. one session still serialises (the per-session lock holds through the real path).

Run from repo root (venv active):
    python -m pytest tests/test_async_offload.py -v
"""

import asyncio
import threading
import time

import httpx
from httpx import ASGITransport

from memoryos import api
from memoryos.session_store import InMemorySessionStore


def _result(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "turn": 1,
        "response": "ok",
        "memory": {},
        "trace": None,
    }


async def _post(client, sid):
    return await client.post("/chat", json={"session_id": sid, "message": "a few words here"})


async def _two_requests(sid_a, sid_b):
    transport = ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await asyncio.gather(_post(client, sid_a), _post(client, sid_b))


def _use_store(monkeypatch, agent_cls):
    store = InMemorySessionStore(
        config=None, agent_factory=lambda config, session_id: agent_cls(session_id)
    )
    monkeypatch.setattr(api, "_store", store)


def test_distinct_sessions_run_concurrently(monkeypatch):
    # A barrier of 2 only releases when BOTH turns are in-flight at once — which can
    # only happen if the event loop isn't blocked. On the un-offloaded code the 2nd
    # request never starts, the 1st times out → BrokenBarrierError → 500 → fails.
    barrier = threading.Barrier(2, timeout=3.0)

    class _Agent:
        def __init__(self, session_id):
            self.session_id = session_id

        def chat(self, message):
            barrier.wait()
            return _result(self.session_id)

    _use_store(monkeypatch, _Agent)

    results = asyncio.run(_two_requests("a", "b"))
    assert all(r.status_code == 200 for r in results)


def test_same_session_requests_serialise(monkeypatch):
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    class _Agent:
        def __init__(self, session_id):
            self.session_id = session_id

        def chat(self, message):
            with guard:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.02)
            with guard:
                state["active"] -= 1
            return _result(self.session_id)

    _use_store(monkeypatch, _Agent)

    results = asyncio.run(_two_requests("same", "same"))
    assert all(r.status_code == 200 for r in results)
    assert state["max"] == 1   # never two turns at once within one session