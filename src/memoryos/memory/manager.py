import time
import tiktoken
from memoryos.config import AppConfig
from memoryos.memory.short_term import ShortTermMemory, Message
from memoryos.memory.long_term import LongTermMemory, MemoryEntry
from memoryos.memory.episodic import EpisodicMemory, Episode, SummarisationResult
from memoryos.memory.trace import (
    ReadTrace, ShortTermRead, LongTermRead, EpisodicRead,
    WriteTrace, ShortTermWrite, LongTermWrite, EpisodicWrite,
    RetrievedItem, EpisodeItem, EvictionRecord,
    TierStatus, EvictionPolicy,
)


def _elapsed_ms(start: float) -> float:
    """Milliseconds since a perf_counter() start, rounded."""
    return round((time.perf_counter() - start) * 1000, 2)


class MemoryManager:
    """
    Coordinates all three memory tiers.

    Responsibilities:
    - Route incoming messages to the right tiers
    - Assemble context from all tiers for each LLM call
    - Trigger episodic summarisation when needed
    - Resolve conflicts between tiers (recency wins)
    - Produce the per-turn memory trace (times each tier call from out here,
      reads each tier's returned domain facts, and records them — the tiers
      stay unaware that tracing exists).
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

        # tiktoken encoder for honest per-tier prompt token counts.
        # Built once here; encoding_for_model knows the common OpenAI models,
        # and we fall back to cl100k_base for anything it doesn't recognise.
        try:
            self._token_encoder = tiktoken.encoding_for_model(config.model.llm_model)
        except KeyError:
            self._token_encoder = tiktoken.get_encoding("cl100k_base")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _is_worth_storing(self, content: str) -> bool:
        """Rule-based filter: skip noise before embedding."""
        normalized = content.strip().lower().rstrip("!.")
        if normalized in self.NOISE_PHRASES:
            return False
        if len(content.split()) < self.MIN_CONTENT_LENGTH:
            return False
        return True

    def _count_tokens(self, text: str) -> int:
        """Token count for a block of prompt text (0 for empty)."""
        if not text:
            return 0
        return len(self._token_encoder.encode(text))

    # ── legacy (untraced) API — kept so existing callers keep working ─────────

    def add_message(self, role: str, content: str) -> dict:
        """
        Process an incoming message through all tiers. Legacy dict return,
        used by the current agent until 2c-ii switches it to the traced path.
        """
        result = {
            "added_to_short_term": False,
            "added_to_long_term": False,
            "episode_created": None,
        }

        message, _evicted = self.short_term.add(role, content)
        result["added_to_short_term"] = True

        if self._is_worth_storing(content):
            self.long_term.add(role, content, turn_index=message.turn_index)
            result["added_to_long_term"] = True

        summ = self.episodic.maybe_summarise(self.short_term)
        if summ:
            result["episode_created"] = summ.episode

        return result

    def get_context(self, current_query: str) -> dict:
        """
        Assemble context from all three tiers (untraced). NOTE: calls
        long_term.to_prompt_format(), which retrieves again — the traced path
        below avoids that double retrieval.
        """
        return {
            "episodic": self.episodic.to_prompt_format(),
            "long_term": self.long_term.to_prompt_format(current_query),
            "short_term": self.short_term.to_prompt_format(),
        }

    def build_system_prompt(self, current_query: str) -> str:
        """Assemble memory tiers into one system prompt block (untraced)."""
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
        """Build the full messages array for the OpenAI API call (untraced)."""
        messages = [
            {"role": "system", "content": self.build_system_prompt(current_query)}
        ]
        messages.extend(self.short_term.to_prompt_format())
        return messages

    # ── traced API — the coordinator layer for the memory trace ──────────────

    _BASE_SYSTEM_PROMPT = (
        "You are a helpful assistant with access to memory "
        "from previous conversations."
    )

    def add_message_traced(self, role: str, content: str, label: str = "") -> WriteTrace:
        """
        Same tier operations as add_message(), but returns a WriteTrace that
        records per-tier status + latency, whether long-term stored the item,
        any summarisation tokens, and every eviction (both paths).

        Error-status population is deferred to the graceful-failure slice (#7):
        for now tier exceptions propagate exactly as they do today.
        """
        wt = WriteTrace(label=label)

        # 1. short-term — always
        t0 = time.perf_counter()
        message, overflow = self.short_term.add(role, content)
        wt.short_term = ShortTermWrite(status=TierStatus.OK, latency_ms=_elapsed_ms(t0))
        if overflow is not None:
            wt.evictions.append(EvictionRecord(
                tier="short_term",
                item_id=str(overflow.turn_index),
                policy=EvictionPolicy.WINDOW_OVERFLOW,
                reason=f"window full (size={self.short_term.window_size})",
            ))

        # 2. long-term — conditional on the noise filter
        t0 = time.perf_counter()
        if self._is_worth_storing(content):
            self.long_term.add(role, content, turn_index=message.turn_index)
            wt.long_term = LongTermWrite(
                status=TierStatus.OK, latency_ms=_elapsed_ms(t0), stored=True
            )
        else:
            wt.long_term = LongTermWrite(
                status=TierStatus.SKIPPED, latency_ms=_elapsed_ms(t0), stored=False
            )

        # 3. episodic — conditional summarisation
        t0 = time.perf_counter()
        summ = self.episodic.maybe_summarise(self.short_term)
        if summ is not None:
            wt.episodic = EpisodicWrite(
                status=TierStatus.OK,
                latency_ms=_elapsed_ms(t0),
                summary_produced=True,
                summarization_tokens=summ.summarization_tokens,
            )
            for turn in summ.evicted_turns:
                wt.evictions.append(EvictionRecord(
                    tier="short_term",
                    item_id=str(turn.turn_index),
                    policy=EvictionPolicy.SUMMARISATION,
                    reason=f"compressed into {summ.episode.id}",
                ))
        else:
            wt.episodic = EpisodicWrite(
                status=TierStatus.SKIPPED, latency_ms=_elapsed_ms(t0)
            )

        return wt

    def assemble_prompt_traced(self, query: str) -> tuple[list[dict], ReadTrace]:
        """
        Build the LLM messages array AND the ReadTrace in a single pass.

        This is the authoritative read for the turn: long-term is retrieved
        ONCE (scores kept for the trace, prompt block formatted from the same
        list), and the prompt is assembled identically to get_messages_for_llm.
        """
        read = ReadTrace()
        cap = self.config.trace.snippet_max_chars

        # episodic read
        t0 = time.perf_counter()
        episodic_block = self.episodic.to_prompt_format()
        ep_latency = _elapsed_ms(t0)
        episodes = self.episodic.get_all_episodes()
        read.episodic = EpisodicRead(
            status=TierStatus.OK if episodes else TierStatus.EMPTY,
            latency_ms=ep_latency,
            episodes=[
                EpisodeItem(id=ep.id, snippet=ep.summary[:cap], used_in_prompt=True)
                for ep in episodes
            ],
            tokens_contributed=self._count_tokens(episodic_block),
        )

        # long-term read — retrieve once, keep scores, then format
        t0 = time.perf_counter()
        memories = self.long_term.retrieve(query)
        lt_latency = _elapsed_ms(t0)
        lt_block = self.long_term.format_memories(memories)
        read.long_term = LongTermRead(
            status=TierStatus.OK if memories else TierStatus.EMPTY,
            latency_ms=lt_latency,
            retrieved=[
                RetrievedItem(
                    id=m.id,
                    score=m.relevance_score,
                    snippet=m.content[:cap],
                    used_in_prompt=True,  # trivially True until token budgets (slice 5)
                )
                for m in memories
            ],
            tokens_contributed=self._count_tokens(lt_block),
        )

        # short-term read
        t0 = time.perf_counter()
        st_messages = self.short_term.to_prompt_format()
        st_latency = _elapsed_ms(t0)
        st_text = "\n".join(m["content"] for m in st_messages)
        read.short_term = ShortTermRead(
            status=TierStatus.OK if st_messages else TierStatus.EMPTY,
            latency_ms=st_latency,
            turns_pulled=len(st_messages),
            tokens_contributed=self._count_tokens(st_text),
        )

        # assemble the prompt — identical structure/order to the legacy path
        sections = [self._BASE_SYSTEM_PROMPT]
        if episodic_block:
            sections.append(episodic_block)
        if lt_block:
            sections.append(lt_block)
        system_prompt = "\n\n".join(sections)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(st_messages)
        return messages, read

    def __repr__(self) -> str:
        return (
            f"MemoryManager("
            f"session={self.session_id}, "
            f"short_term={len(self.short_term)}, "
            f"long_term={self.long_term.count()}, "
            f"episodes={len(self.episodic)})"
        )