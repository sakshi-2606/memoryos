"""
Session store abstraction.

The API keeps a registry of active sessions (session_id -> MemoryAgent). This
puts an interface in front of it so the in-memory default can be swapped for a
Redis/persisted backend later without changing the endpoints.
"""

import threading
from abc import ABC, abstractmethod
from typing import Callable

from memoryos.agent import MemoryAgent
from memoryos.config import AppConfig


class SessionStore(ABC):
    """Registry of active sessions. Implement this to back sessions with Redis etc."""

    @abstractmethod
    def get(self, session_id: str) -> MemoryAgent | None: ...

    @abstractmethod
    def get_or_create(self, session_id: str) -> MemoryAgent: ...

    @abstractmethod
    def delete(self, session_id: str) -> bool: ...

    @abstractmethod
    def list_ids(self) -> list[str]: ...

    def __contains__(self, session_id: str) -> bool:
        return self.get(session_id) is not None

    def __len__(self) -> int:
        return len(self.list_ids())


class InMemorySessionStore(SessionStore):
    """In-process registry (the default). Lost on restart; single-instance only."""

    def __init__(self, config: AppConfig, agent_factory: Callable = MemoryAgent) -> None:
        self._config = config
        self._agent_factory = agent_factory
        self._sessions: dict[str, MemoryAgent] = {}
        self._lock = threading.Lock()   # guards the registry dict

    def get(self, session_id: str) -> MemoryAgent | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> MemoryAgent:
        # Fast path: a dict read is atomic under the GIL, so an already-created
        # session needs no lock.
        agent = self._sessions.get(session_id)
        if agent is not None:
            return agent
        # Slow path: create under the lock, re-checking inside it so two threads
        # racing on the same new id produce exactly one agent (not two, with one
        # silently overwritten).
        with self._lock:
            agent = self._sessions.get(session_id)
            if agent is None:
                agent = self._agent_factory(self._config, session_id=session_id)
                self._sessions[session_id] = agent
            return agent

    def delete(self, session_id: str) -> bool:
        with self._lock:
            agent = self._sessions.pop(session_id, None)
        if agent is None:
            return False
        # purge persisted long-term memory too, so "forget" actually forgets and a
        # later same-id session cannot resurrect it. Done outside the lock (Chroma
        # IO can be slow and must not block the registry). Best-effort: the session
        # is removed from the registry regardless of whether the purge succeeds.
        try:
            agent.memory.long_term.purge()
        except Exception:
            pass
        return True

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())