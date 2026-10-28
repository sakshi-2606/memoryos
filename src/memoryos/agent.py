import time
import uuid
from datetime import datetime
from openai import OpenAI, RateLimitError, APIError
from memoryos.config import AppConfig
from memoryos.memory.manager import MemoryManager
from memoryos.memory.trace import MemoryTrace, Totals


def _elapsed_ms(start: float) -> float:
    """Milliseconds since a perf_counter() start, rounded."""
    return round((time.perf_counter() - start) * 1000, 2)


class MemoryAgent:
    """
    Stateful conversational agent with tiered memory.

    Wraps the MemoryManager and handles:
    - Multi-turn conversation loop
    - LLM API calls with retry logic
    - Memory read/write on every turn
    - Assembling the per-turn MemoryTrace from the manager's reports
    """

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0   # seconds, doubles each retry

    def __init__(self, config: AppConfig, session_id: str | None = None) -> None:
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        self.memory = MemoryManager(config, self.session_id)
        self._client = OpenAI()
        self._turn_count = 0

    def _call_llm(self, messages: list[dict]) -> tuple[str, int]:
        """
        Call OpenAI with exponential backoff retry.

        Returns (text, answer_tokens) where answer_tokens is the completion
        token count for the generated answer — kept for the trace instead of
        being discarded.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model.llm_model,
                    messages=messages,
                    max_tokens=self.config.model.max_tokens,
                    temperature=self.config.model.temperature,
                )
                text = response.choices[0].message.content.strip()
                answer_tokens = (
                    response.usage.completion_tokens if response.usage else 0
                )
                return text, answer_tokens

            except RateLimitError:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"Rate limited. Retrying in {delay}s...")
                time.sleep(delay)

            except APIError as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"API error: {e}. Retrying in {delay}s...")
                time.sleep(delay)

        raise RuntimeError("LLM call failed after all retries")

    def chat(self, user_message: str) -> dict:
        """
        Process one user turn end-to-end, and (when enabled) assemble the
        per-turn MemoryTrace from the manager's traced reports.

        Stages, each timed here at the agent layer:
          user_write -> read (prompt assembly) -> llm -> assistant_write
        """
        self._turn_count += 1
        turn_start = time.perf_counter()

        # 1. user-message write
        t0 = time.perf_counter()
        user_wt = self.memory.add_message_traced("user", user_message, label="user")
        user_write_ms = _elapsed_ms(t0)

        # 2. read: assemble the prompt + the ReadTrace (single authoritative read)
        t0 = time.perf_counter()
        messages, read = self.memory.assemble_prompt_traced(user_message)
        read_ms = _elapsed_ms(t0)

        # 3. append the current user message
        messages.append({"role": "user", "content": user_message})

        # 4. LLM call
        t0 = time.perf_counter()
        response_text, answer_tokens = self._call_llm(messages)
        llm_ms = _elapsed_ms(t0)

        # 5. assistant-message write
        t0 = time.perf_counter()
        assistant_wt = self.memory.add_message_traced(
            "assistant", response_text, label="assistant"
        )
        assistant_write_ms = _elapsed_ms(t0)

        total_ms = _elapsed_ms(turn_start)

        # 6. assemble the trace (gated by config)
        trace = None
        if self.config.trace.enabled:
            trace = MemoryTrace(
                session_id=self.session_id,
                turn_index=self._turn_count,
                turn_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat(),
                query=user_message,
                answer=response_text,
                user_write=user_wt,
                read=read,
                assistant_write=assistant_wt,
                totals=Totals(
                    latency_ms_total=total_ms,
                    latency_ms_by_stage={
                        "user_write": user_write_ms,
                        "read": read_ms,
                        "llm": llm_ms,
                        "assistant_write": assistant_write_ms,
                    },
                    prompt_tokens_by_tier={
                        "short_term": read.short_term.tokens_contributed,
                        "long_term": read.long_term.tokens_contributed,
                        "episodic": read.episodic.tokens_contributed,
                    },
                    summarization_tokens=(
                        user_wt.episodic.summarization_tokens
                        + assistant_wt.episodic.summarization_tokens
                    ),
                    answer_tokens=answer_tokens,
                ),
            )

        return {
            "response": response_text,
            "session_id": self.session_id,
            "turn": self._turn_count,
            # legacy UI metadata — now derived from the traced reports
            # (the redundant second get_context() retrieval is gone)
            "memory": {
                "added_to_long_term": user_wt.long_term.stored,
                "episode_created": (
                    user_wt.episodic.summary_produced
                    or assistant_wt.episodic.summary_produced
                ),
                "short_term_size": len(self.memory.short_term),
                "long_term_size": self.memory.long_term.count(),
                "episode_count": len(self.memory.episodic),
                "context_used": {
                    "episodic": bool(read.episodic.episodes),
                    "long_term": bool(read.long_term.retrieved),
                    "short_term": read.short_term.turns_pulled > 0,
                },
            },
            "trace": trace.to_dict() if trace else None,
        }

    def get_memory_state(self) -> dict:
        """
        Full memory snapshot — used by the Streamlit UI
        to render the memory visualiser panel.
        """
        return {
            "session_id": self.session_id,
            "turn_count": self._turn_count,
            "short_term": [
                {"role": m.role, "content": m.content, "turn": m.turn_index}
                for m in self.memory.short_term.get_all()
            ],
            "long_term_count": self.memory.long_term.count(),
            "episodes": [
                {
                    "id": ep.id,
                    "summary": ep.summary,
                    "turns": ep.turn_range,
                }
                for ep in self.memory.episodic.get_all_episodes()
            ],
        }

    def __repr__(self) -> str:
        return (
            f"MemoryAgent("
            f"session={self.session_id[:8]}..., "
            f"turns={self._turn_count})"
        )