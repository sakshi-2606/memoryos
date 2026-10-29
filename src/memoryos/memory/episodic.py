from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI
from memoryos.config import ModelConfig
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
    """
    episode: Episode
    evicted_turns: list[Message]
    summarization_tokens: int


class EpisodicMemory:
    """
    Summarisation layer that compresses batches of evicted turns into episodes.

    With token-based eviction, the short-term tier decides WHICH turns leave
    (when the token budget is exceeded); this tier takes that evicted batch and
    compresses it into one episode. Episodes are kept in-memory for now.
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

    def __init__(self, model_config: ModelConfig) -> None:
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
        Returns (summary_text, tokens_used).
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

    def summarise_turns(self, turns: list[Message]) -> SummarisationResult | None:
        """
        Summarise a batch of already-evicted turns into one episode.

        Returns a SummarisationResult (episode + the turns + token cost), or
        None for an empty batch. The caller (manager) supplies the batch that
        the short-term tier evicted for exceeding the token budget.
        """
        if not turns:
            return None

        summary_text, summarization_tokens = self._summarise(turns)

        episode = Episode(
            id=f"episode_{self._episode_counter:04d}",
            summary=summary_text,
            turn_range=(turns[0].turn_index, turns[-1].turn_index),
            timestamp=datetime.utcnow().isoformat(),
            source_turn_count=len(turns),
        )
        self._episodes.append(episode)
        self._episode_counter += 1

        return SummarisationResult(
            episode=episode,
            evicted_turns=turns,
            summarization_tokens=summarization_tokens,
        )

    def get_all_episodes(self) -> list[Episode]:
        """Return all episodes, oldest first."""
        return list(self._episodes)

    def to_prompt_format(self) -> str:
        """Format all episodes as a context block for the system prompt."""
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