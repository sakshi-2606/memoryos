from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for all three memory tiers."""

    # Short-term: sliding window size (number of turns to keep in buffer)
    short_term_window: int = 10

    # Long-term: how many memories to retrieve per query
    long_term_top_k: int = 5

    # Long-term: minimum similarity score to include a memory (0.0 to 1.0)
    long_term_similarity_threshold: float = 0.40

    # Episodic: summarise when buffer exceeds this many turns
    summarisation_threshold: int = 8

    # Episodic: how many turns to compress into one summary
    turns_to_summarise: int = 4


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
 
    # Master switch. When False, the turn loop skips building the trace entirely.
    enabled: bool = True
 
    # When True, the full trace is attached to the /chat HTTP response.
    # Turn off if the response gets noisy — the trace can still be logged/dumped.
    include_in_response: bool = True
 
    # Snippets of retrieved memories / episodes are truncated to this many chars
    # so the trace stays compact.
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
        """Fail fast if critical config is missing."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        if self.memory.turns_to_summarise >= self.memory.short_term_window:
            raise ValueError(
                "turns_to_summarise must be less than short_term_window."
            )


# Module-level singleton — import this everywhere
config = AppConfig()