"""
Session store abstraction.

The API keeps a registry of active sessions (session_id -> MemoryAgent). This
puts an interface in front of it so the in-memory default can be swapped for a
Redis/persisted backend later without changing the endpoints.
"""

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

    def get(self, session_id: str) -> MemoryAgent | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> MemoryAgent:
        if session_id not in self._sessions:
            self._sessions[session_id] = self._agent_factory(
                self._config, session_id=session_id
            )
        return self._sessions[session_id]

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())