from dataclasses import dataclass
from typing import Callable
from memoryos.config import MemoryConfig


@dataclass
class Message:
    """A single turn in a conversation."""
    role: str
    content: str
    turn_index: int


class ShortTermMemory:
    """
    Token-budgeted buffer of recent conversation turns.

    Holds recent turns up to `short_term_token_budget` tokens (measured by an
    injected token counter). When a new turn pushes the buffer over budget, the
    oldest turns are evicted down to a low watermark and RETURNED so the
    coordinator can summarise them into an episode — nothing is silently dropped.
    """

    def __init__(
        self,
        config: MemoryConfig,
        token_counter: Callable[[str], int],
    ) -> None:
        self.token_budget = config.short_term_token_budget
        self.evict_to = int(
            config.short_term_token_budget * config.short_term_evict_to_fraction
        )
        self._count_tokens = token_counter
        self._buffer: list[Message] = []
        self._turn_counter: int = 0

    def total_tokens(self) -> int:
        """Current token footprint of the buffer."""
        return sum(self._count_tokens(m.content) for m in self._buffer)

    def add(self, role: str, content: str) -> tuple[Message, list[Message]]:
        """
        Append a message. If the buffer exceeds the token budget, evict the
        oldest turns down to the low watermark and return them.

        Returns (message, evicted):
          message — the Message just added.
          evicted — oldest Messages removed because the budget was exceeded
                    (empty if the buffer is still within budget). These are
                    handed to the episodic tier for summarisation, not dropped.
        """
        message = Message(role=role, content=content, turn_index=self._turn_counter)
        self._buffer.append(message)
        self._turn_counter += 1

        evicted: list[Message] = []
        if self.total_tokens() > self.token_budget:
            # Evict oldest until at/under the watermark. Never evict the newest
            # turn (a single over-budget turn is kept rather than dropped).
            while len(self._buffer) > 1 and self.total_tokens() > self.evict_to:
                evicted.append(self._buffer.pop(0))

        return message, evicted

    def get_all(self) -> list[Message]:
        """Return all messages currently in the buffer, oldest first."""
        return list(self._buffer)

    def get_recent(self, n: int) -> list[Message]:
        """Return the n most recent messages."""
        return self._buffer[-n:] if n < len(self._buffer) else list(self._buffer)

    def to_prompt_format(self) -> list[dict]:
        """Convert buffer to OpenAI message format for prompt injection."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._buffer
        ]

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory("
            f"budget={self.token_budget} tokens, "
            f"current={self.total_tokens()} tokens / {len(self._buffer)} turns)"
        )