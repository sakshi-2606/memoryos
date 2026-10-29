from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for all three memory tiers."""

    # Short-term: token budget for the recent-turns buffer (tiktoken-measured).
    # When a new turn pushes the buffer over this, oldest turns are evicted.
    short_term_token_budget: int = 2000

    # On overflow, evict oldest turns down to this fraction of the budget
    # (a low watermark) and summarise the evicted batch into one episode.
    short_term_evict_to_fraction: float = 0.5

    # Long-term: how many memories to retrieve per query
    long_term_top_k: int = 5

    # Long-term: minimum similarity score to include a memory (0.0 to 1.0),
    # calibrated via scripts/threshold_sweep.py (F2)
    long_term_similarity_threshold: float = 0.40


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for LLM and embedding models."""

    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = 1536   # text-embedding-3-small output dimension
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass(frozen=True)
class StorageConfig:
    """Configuration for persistence and storage."""

    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    collection_name: str = "long_term_memory"


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for the per-turn memory trace."""

    enabled: bool = True
    include_in_response: bool = True
    snippet_max_chars: int = 200


@dataclass(frozen=True)
class AppConfig:
    """Root config — the one every file imports."""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    def validate(self) -> None:
        """Fail fast if critical config is missing or inconsistent."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        if self.memory.short_term_token_budget <= 0:
            raise ValueError("short_term_token_budget must be positive.")
        if not 0.0 < self.memory.short_term_evict_to_fraction < 1.0:
            raise ValueError(
                "short_term_evict_to_fraction must be between 0 and 1 (exclusive)."
            )


# Module-level singleton — import this everywhere
config = AppConfig()