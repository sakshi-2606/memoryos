from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from memoryos.config import MemoryConfig


@dataclass
class Message:
    """A single turn in a conversation."""
    role: str        # "user" or "assistant"
    content: str
    turn_index: int  # position in conversation, used by episodic tier later


class ShortTermMemory:
    """
    Sliding window buffer of recent conversation turns.

    Holds the last `window_size` messages in RAM.
    When full, oldest message is automatically evicted.
    No persistence — resets when the process restarts.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.window_size = config.short_term_window
        self.summarisation_threshold = config.summarisation_threshold
        self._buffer: deque[Message] = deque(maxlen=config.short_term_window)
        self._turn_counter: int = 0

    def add(self, role: str, content: str) -> tuple[Message, Message | None]:
        """
        Add a new message to the buffer.

        Returns (message, evicted):
          message  — the Message just added.
          evicted  — the oldest Message that was silently pushed out because
                     the window was already full (deque maxlen), or None if
                     nothing was dropped.

        Why the peek: a deque(maxlen) drops its leftmost item automatically on
        overflow and tells you nothing. Capturing that item here is what lets
        the coordinator record a window-overflow eviction in the memory trace.
        The buffer's behaviour is otherwise unchanged.
        """
        evicted: Message | None = None
        if self._buffer.maxlen is not None and len(self._buffer) == self._buffer.maxlen:
            # buffer is full — the leftmost item is about to be pushed out
            evicted = self._buffer[0]

        message = Message(
            role=role,
            content=content,
            turn_index=self._turn_counter
        )
        self._buffer.append(message)
        self._turn_counter += 1
        return message, evicted

    def get_all(self) -> list[Message]:
        """Return all messages currently in the buffer, oldest first."""
        return list(self._buffer)

    def get_recent(self, n: int) -> list[Message]:
        """Return the n most recent messages."""
        all_messages = list(self._buffer)
        return all_messages[-n:] if n < len(all_messages) else all_messages

    def should_summarise(self) -> bool:
        """
        Returns True when the buffer has enough turns to trigger
        episodic summarisation. The episodic tier checks this.
        """
        return len(self._buffer) >= self.summarisation_threshold

    def evict_oldest(self, n: int) -> list[Message]:
        """
        Remove and return the n oldest messages from the buffer.
        Called by the episodic tier after summarisation.
        """
        evicted = []
        for _ in range(min(n, len(self._buffer))):
            evicted.append(self._buffer.popleft())
        return evicted

    def to_prompt_format(self) -> list[dict]:
        """
        Convert buffer to OpenAI message format for direct injection
        into the LLM prompt.

        Returns: [{"role": "user", "content": "..."}, ...]
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._buffer
        ]

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory("
            f"window={self.window_size}, "
            f"current={len(self._buffer)} messages)"
        )