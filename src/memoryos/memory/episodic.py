from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from memoryos.config import MemoryConfig, ModelConfig
from memoryos.memory.short_term import Message


@dataclass
class Episode:
    """A compressed summary of a group of conversation turns."""
    id: str
    summary: str
    turn_range: tuple[int, int]   # (first_turn_index, last_turn_index)
    timestamp: str
    source_turn_count: int        # how many turns were compressed


@dataclass
class SummarisationResult:
    """
    What one summarisation pass produced — the tier's own domain facts,
    handed back to the coordinator so it can record them in the trace.

    - episode: the Episode that was created and stored.
    - evicted_turns: the turns removed from short-term to make this episode
      (the second eviction path — policy = summarisation).
    - summarization_tokens: tokens the LLM summariser call spent.
    """
    episode: Episode
    evicted_turns: list[Message]
    summarization_tokens: int


class EpisodicMemory:
    """
    Summarisation layer that compresses old short-term turns into episodes.

    When short-term memory signals it's ready to summarise, this class:
    1. Takes the oldest N turns from short-term
    2. Sends them to the LLM for summarisation
    3. Stores the resulting Episode
    4. Returns the evicted turns so short-term can remove them

    Episodes are kept in-memory (list) for now.
    In production you'd persist these to a DB alongside long-term memory.
    """

    SUMMARISE_PROMPT = """You are a conversation memory compressor.
Summarise the following conversation turns into 2-3 concise sentences.
Rules:
- Include only facts, preferences, and decisions explicitly stated
- Do not infer or add information not present in the turns
- Write in third person (e.g. "The user mentioned...", "The assistant explained...")
- Be specific, not generic

Conversation turns:
{turns}

Summary:"""

    def __init__(
        self,
        memory_config: MemoryConfig,
        model_config: ModelConfig,
    ) -> None:
        self.turns_to_summarise = memory_config.turns_to_summarise
        self.llm_model = model_config.llm_model
        self._client = OpenAI()
        self._episodes: list[Episode] = []
        self._episode_counter: int = 0

    def _format_turns_for_prompt(self, turns: list[Message]) -> str:
        """Convert Message objects to readable text for the LLM."""
        lines = []
        for turn in turns:
            lines.append(f"{turn.role.capitalize()} (turn {turn.turn_index}): {turn.content}")
        return "\n".join(lines)

    def _summarise(self, turns: list[Message]) -> tuple[str, int]:
        """
        Call LLM to summarise a list of turns into one episode.

        Returns (summary_text, tokens_used). The token count used to be
        discarded on response.usage; we keep it so the trace can report the
        real cost of summarisation.
        """
        formatted = self._format_turns_for_prompt(turns)
        prompt = self.SUMMARISE_PROMPT.format(turns=formatted)

        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,   # low temp = more faithful, less creative
        )
        text = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    def maybe_summarise(self, short_term) -> SummarisationResult | None:
        """
        Check if short-term memory is ready for summarisation.
        If yes: evict oldest turns, summarise, store episode, return a
        SummarisationResult (episode + evicted turns + token cost).
        If no: return None.

        Args:
            short_term: ShortTermMemory instance
        """
        if not short_term.should_summarise():
            return None

        # Evict oldest N turns from short-term
        evicted_turns = short_term.evict_oldest(self.turns_to_summarise)
        if not evicted_turns:
            return None

        # Summarise the evicted turns
        summary_text, summarization_tokens = self._summarise(evicted_turns)

        # Create and store the episode
        episode = Episode(
            id=f"episode_{self._episode_counter:04d}",
            summary=summary_text,
            turn_range=(
                evicted_turns[0].turn_index,
                evicted_turns[-1].turn_index
            ),
            timestamp=datetime.utcnow().isoformat(),
            source_turn_count=len(evicted_turns),
        )
        self._episodes.append(episode)
        self._episode_counter += 1

        return SummarisationResult(
            episode=episode,
            evicted_turns=evicted_turns,
            summarization_tokens=summarization_tokens,
        )

    def get_all_episodes(self) -> list[Episode]:
        """Return all episodes, oldest first."""
        return list(self._episodes)

    def to_prompt_format(self) -> str:
        """
        Format all episodes as a context block for the system prompt.
        Injected before short-term and long-term context.
        """
        if not self._episodes:
            return ""

        lines = ["Summary of earlier conversation:"]
        for ep in self._episodes:
            lines.append(
                f"- [Turns {ep.turn_range[0]}-{ep.turn_range[1]}]: {ep.summary}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"EpisodicMemory(episodes={len(self._episodes)})"