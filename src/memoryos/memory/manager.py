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


def _detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


class MemoryManager:
    """
    Coordinates all three memory tiers.

    Memory-tier failures are handled gracefully: a failing tier is recorded in
    the trace (status=error + error_detail) and the turn DEGRADES rather than
    crashing — a down vector store answers without retrieved context, a failed
    summarisation skips the episode. The answer-generation LLM call (in the
    agent) is the only failure that legitimately fails the turn.
    """

    # Messages shorter than this are not stored in long-term
    MIN_CONTENT_LENGTH = 4

    # Filler phrases not worth storing
    NOISE_PHRASES = {
        "ok", "okay", "sure", "yes", "no", "thanks",
        "thank you", "got it", "great", "nice", "cool",
        "alright", "sounds good", "makes sense"
    }

    _BASE_SYSTEM_PROMPT = (
        "You are a helpful assistant with access to memory "
        "from previous conversations."
    )

    def __init__(self, config: AppConfig, session_id: str) -> None:
        self.session_id = session_id
        self.config = config

        # tiktoken encoder first — the short-term tier needs it for budgeting.
        try:
            self._token_encoder = tiktoken.encoding_for_model(config.model.llm_model)
        except KeyError:
            self._token_encoder = tiktoken.get_encoding("cl100k_base")

        self.short_term = ShortTermMemory(config.memory, self._count_tokens)
        self.long_term = LongTermMemory(
            memory_config=config.memory,
            model_config=config.model,
            storage_config=config.storage,
            session_id=session_id,
        )
        self.episodic = EpisodicMemory(config.model)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _is_worth_storing(self, content: str) -> bool:
        normalized = content.strip().lower().rstrip("!.")
        if normalized in self.NOISE_PHRASES:
            return False
        if len(content.split()) < self.MIN_CONTENT_LENGTH:
            return False
        return True

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._token_encoder.encode(text))

    # ── legacy (untraced) API ─────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> dict:
        """Legacy dict return. Degrades on tier failure like the traced path."""
        result = {
            "added_to_short_term": False,
            "added_to_long_term": False,
            "episode_created": None,
        }

        message, evicted = self.short_term.add(role, content)
        result["added_to_short_term"] = True

        if self._is_worth_storing(content):
            try:
                self.long_term.add(role, content, turn_index=message.turn_index)
                result["added_to_long_term"] = True
            except Exception:
                pass  # degrade: not stored

        if evicted:
            try:
                summ = self.episodic.summarise_turns(evicted)
                if summ:
                    result["episode_created"] = summ.episode
            except Exception:
                pass  # degrade: no episode this turn

        return result

    def get_context(self, current_query: str) -> dict:
        return {
            "episodic": self.episodic.to_prompt_format(),
            "long_term": self.long_term.to_prompt_format(current_query),
            "short_term": self.short_term.to_prompt_format(),
        }

    def build_system_prompt(self, current_query: str) -> str:
        context = self.get_context(current_query)
        sections = [self._BASE_SYSTEM_PROMPT]
        if context["episodic"]:
            sections.append(context["episodic"])
        if context["long_term"]:
            sections.append(context["long_term"])
        return "\n\n".join(sections)

    def get_messages_for_llm(self, current_query: str) -> list[dict]:
        messages = [
            {"role": "system", "content": self.build_system_prompt(current_query)}
        ]
        messages.extend(self.short_term.to_prompt_format())
        return messages

    # ── traced API — with graceful tier-failure handling ──────────────────────

    def add_message_traced(self, role: str, content: str, label: str = "") -> WriteTrace:
        wt = WriteTrace(label=label)

        # 1. short-term — in-memory, always succeeds; token-budget eviction
        #    returns the overflow batch
        t0 = time.perf_counter()
        message, evicted = self.short_term.add(role, content)
        wt.short_term = ShortTermWrite(status=TierStatus.OK, latency_ms=_elapsed_ms(t0))

        # 2. long-term — conditional; degrade (don't crash) on store failure
        t0 = time.perf_counter()
        if self._is_worth_storing(content):
            try:
                self.long_term.add(role, content, turn_index=message.turn_index)
                wt.long_term = LongTermWrite(
                    status=TierStatus.OK, latency_ms=_elapsed_ms(t0), stored=True
                )
            except Exception as e:
                wt.long_term = LongTermWrite(
                    status=TierStatus.ERROR, latency_ms=_elapsed_ms(t0),
                    stored=False, error_detail=_detail(e),
                )
        else:
            wt.long_term = LongTermWrite(
                status=TierStatus.SKIPPED, latency_ms=_elapsed_ms(t0), stored=False
            )

        # 3. episodic — summarise the evicted overflow; degrade on failure
        t0 = time.perf_counter()
        if evicted:
            try:
                summ = self.episodic.summarise_turns(evicted)
                wt.episodic = EpisodicWrite(
                    status=TierStatus.OK,
                    latency_ms=_elapsed_ms(t0),
                    summary_produced=True,
                    summarization_tokens=summ.summarization_tokens,
                )
                reason = f"over token budget → {summ.episode.id}"
            except Exception as e:
                wt.episodic = EpisodicWrite(
                    status=TierStatus.ERROR,
                    latency_ms=_elapsed_ms(t0),
                    summary_produced=False,
                    error_detail=_detail(e),
                )
                reason = "over token budget → summarisation failed (turns dropped)"
            for turn in evicted:
                wt.evictions.append(EvictionRecord(
                    tier="short_term",
                    item_id=str(turn.turn_index),
                    policy=EvictionPolicy.TOKEN_BUDGET,
                    reason=reason,
                ))
        else:
            wt.episodic = EpisodicWrite(
                status=TierStatus.SKIPPED, latency_ms=_elapsed_ms(t0)
            )

        return wt

    def assemble_prompt_traced(self, query: str) -> tuple[list[dict], ReadTrace]:
        read = ReadTrace()
        cap = self.config.trace.snippet_max_chars

        # episodic read — in-memory
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

        # long-term read — retrieve once; degrade to no context on failure
        t0 = time.perf_counter()
        try:
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
                        used_in_prompt=True,
                    )
                    for m in memories
                ],
                tokens_contributed=self._count_tokens(lt_block),
            )
        except Exception as e:
            lt_block = ""   # degrade: answer without long-term context
            read.long_term = LongTermRead(
                status=TierStatus.ERROR,
                latency_ms=_elapsed_ms(t0),
                retrieved=[],
                tokens_contributed=0,
                error_detail=_detail(e),
            )

        # short-term read — in-memory
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

        # assemble the prompt
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
            f"short_term={len(self.short_term)} turns, "
            f"long_term={self.long_term.count()}, "
            f"episodes={len(self.episodic)})"
        )