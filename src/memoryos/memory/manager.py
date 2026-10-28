from memoryos.config import AppConfig
from memoryos.memory.short_term import ShortTermMemory, Message
from memoryos.memory.long_term import LongTermMemory
from memoryos.memory.episodic import EpisodicMemory, Episode


class MemoryManager:
    """
    Coordinates all three memory tiers.

    Responsibilities:
    - Route incoming messages to the right tiers
    - Assemble context from all tiers for each LLM call
    - Trigger episodic summarisation when needed
    - Resolve conflicts between tiers (recency wins)
    """

    # Messages shorter than this are not stored in long-term
    MIN_CONTENT_LENGTH = 4

    # Filler phrases not worth storing
    NOISE_PHRASES = {
        "ok", "okay", "sure", "yes", "no", "thanks",
        "thank you", "got it", "great", "nice", "cool",
        "alright", "sounds good", "makes sense"
    }

    def __init__(self, config: AppConfig, session_id: str) -> None:
        self.session_id = session_id
        self.config = config

        # Instantiate all three tiers
        self.short_term = ShortTermMemory(config.memory)
        self.long_term = LongTermMemory(
            memory_config=config.memory,
            model_config=config.model,
            storage_config=config.storage,
            session_id=session_id,
        )
        self.episodic = EpisodicMemory(config.memory, config.model)

    def _is_worth_storing(self, content: str) -> bool:
        """Rule-based filter: skip noise before embedding."""
        normalized = content.strip().lower().rstrip("!.")
        if normalized in self.NOISE_PHRASES:
            return False
        if len(content.split()) < self.MIN_CONTENT_LENGTH:
            return False
        return True

    def add_message(self, role: str, content: str) -> dict:
        """
        Process an incoming message through all tiers.

        1. Add to short-term (always)
        2. Add to long-term if worth storing
        3. Trigger episodic summarisation if threshold reached

        Returns a dict with what happened — useful for the UI visualiser.
        """
        result = {
            "added_to_short_term": False,
            "added_to_long_term": False,
            "episode_created": None,
        }

        # 1. Always add to short-term
        message, _evicted = self.short_term.add(role, content)
        result["added_to_short_term"] = True

        # 2. Conditionally add to long-term
        if self._is_worth_storing(content):
            self.long_term.add(role, content, turn_index=message.turn_index)
            result["added_to_long_term"] = True

        # 3. Trigger episodic summarisation if needed
        episode = self.episodic.maybe_summarise(self.short_term)
        if episode:
            result["episode_created"] = episode

        return result

    def get_context(self, current_query: str) -> dict:
        """
        Assemble context from all three tiers for the next LLM call.

        Returns a dict with each tier's contribution — the agent
        uses this to build the final prompt, and the UI uses it
        to show the memory visualiser panel.
        """
        return {
            "episodic": self.episodic.to_prompt_format(),
            "long_term": self.long_term.to_prompt_format(current_query),
            "short_term": self.short_term.to_prompt_format(),
        }

    def build_system_prompt(self, current_query: str) -> str:
        """
        Assemble all memory tiers into one system prompt block.

        Order: episodic (oldest) → long-term (relevant) → short-term (recent)
        Recency wins — short-term is closest to the LLM's attention.
        """
        context = self.get_context(current_query)
        sections = []

        base = (
            "You are a helpful assistant with access to memory "
            "from previous conversations."
        )
        sections.append(base)

        if context["episodic"]:
            sections.append(context["episodic"])

        if context["long_term"]:
            sections.append(context["long_term"])

        return "\n\n".join(sections)

    def get_messages_for_llm(self, current_query: str) -> list[dict]:
        """
        Build the full messages array for the OpenAI API call.

        Structure:
        [
          {"role": "system", "content": <memory context>},
          ... short-term history ...,
          {"role": "user", "content": <current query>}  ← added by agent
        ]
        """
        messages = [
            {"role": "system", "content": self.build_system_prompt(current_query)}
        ]
        messages.extend(self.short_term.to_prompt_format())
        return messages

    def __repr__(self) -> str:
        return (
            f"MemoryManager("
            f"session={self.session_id}, "
            f"short_term={len(self.short_term)}, "
            f"long_term={self.long_term.count()}, "
            f"episodes={len(self.episodic)})"
        )