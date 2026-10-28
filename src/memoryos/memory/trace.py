"""
Memory-trace spine for MemoryOS.

ONE MemoryTrace is produced per agent turn. It is a passive record of what
the memory tiers did during that turn — nothing here changes tier behaviour.
The coordinator (manager + agent) fills this object in as the turn runs, then
attaches it to the /chat response and/or dumps it to JSON.

Design notes:
- Every field has a default, so the trace can be created empty at the start of
  a turn and filled in incrementally ("create then fill") by the manager/agent.
- Enums subclass `str`, so `dataclasses.asdict()` + `json.dumps()` serialise
  the whole tree with no custom encoder.
- `schema_version` is the contract. Later slices (eval, observability) load
  these records; when the shape changes, bump this and let loaders assert on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum


SCHEMA_VERSION = "1.0"


# ── Enums ────────────────────────────────────────────────────────────────────

class TierStatus(str, Enum):
    """Outcome of a single tier read or write."""
    OK = "ok"          # ran and produced/returned something
    EMPTY = "empty"    # ran fine but nothing to return/store (e.g. no retrieval hits)
    SKIPPED = "skipped"  # intentionally not run (e.g. content not worth storing)
    ERROR = "error"    # raised — see error_detail


class EvictionPolicy(str, Enum):
    """Which mechanism removed an item from a tier."""
    SUMMARISATION = "summarisation"      # evict_oldest() after an episode was made
    WINDOW_OVERFLOW = "window_overflow"  # deque(maxlen) silently dropped the oldest


# ── Leaf records ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedItem:
    """One memory returned by the long-term (vector) tier during a read."""
    id: str
    score: float
    snippet: str
    used_in_prompt: bool = True   # did it actually land in the assembled prompt?
                                  # trivially True until token budgets (slice 5)


@dataclass
class EpisodeItem:
    """One episode summary pulled in by the episodic tier during a read."""
    id: str
    snippet: str
    used_in_prompt: bool = True


@dataclass
class EvictionRecord:
    """One item that left a tier, and why."""
    tier: str                 # which tier it left (today: always "short_term")
    item_id: str              # identifier of the evicted item (message turn_index)
    policy: EvictionPolicy
    reason: str = ""          # short human-readable note


# ── Read side ────────────────────────────────────────────────────────────────

@dataclass
class ShortTermRead:
    status: TierStatus = TierStatus.EMPTY
    latency_ms: float = 0.0
    turns_pulled: int = 0           # how many window turns went into the prompt
    tokens_contributed: int = 0     # prompt tokens this tier added


@dataclass
class LongTermRead:
    status: TierStatus = TierStatus.EMPTY
    latency_ms: float = 0.0
    retrieved: list[RetrievedItem] = field(default_factory=list)
    tokens_contributed: int = 0
    error_detail: str | None = None


@dataclass
class EpisodicRead:
    status: TierStatus = TierStatus.EMPTY
    latency_ms: float = 0.0
    episodes: list[EpisodeItem] = field(default_factory=list)
    tokens_contributed: int = 0


@dataclass
class ReadTrace:
    """The single authoritative read that assembled the prompt for this turn."""
    short_term: ShortTermRead = field(default_factory=ShortTermRead)
    long_term: LongTermRead = field(default_factory=LongTermRead)
    episodic: EpisodicRead = field(default_factory=EpisodicRead)


# ── Write side (happens twice per turn: user msg, then assistant msg) ─────────

@dataclass
class ShortTermWrite:
    status: TierStatus = TierStatus.OK
    latency_ms: float = 0.0
    error_detail: str | None = None


@dataclass
class LongTermWrite:
    status: TierStatus = TierStatus.SKIPPED
    latency_ms: float = 0.0
    stored: bool = False           # passed the worth-storing filter and embedded?
    error_detail: str | None = None


@dataclass
class EpisodicWrite:
    status: TierStatus = TierStatus.SKIPPED
    latency_ms: float = 0.0
    summary_produced: bool = False
    summarization_tokens: int = 0
    error_detail: str | None = None


@dataclass
class WriteTrace:
    """
    One add_message() pass. Used for both the user-message write and the
    assistant-message write. Evictions are recorded here because both eviction
    paths (summarisation + window overflow) fire during a write.
    """
    label: str = ""               # "user" or "assistant" — which write this is
    short_term: ShortTermWrite = field(default_factory=ShortTermWrite)
    long_term: LongTermWrite = field(default_factory=LongTermWrite)
    episodic: EpisodicWrite = field(default_factory=EpisodicWrite)
    evictions: list[EvictionRecord] = field(default_factory=list)


# ── Totals ───────────────────────────────────────────────────────────────────

@dataclass
class Totals:
    latency_ms_total: float = 0.0
    latency_ms_by_stage: dict = field(default_factory=dict)   # user_write/read/llm/assistant_write
    prompt_tokens_by_tier: dict = field(default_factory=dict)  # short_term/long_term/episodic
    summarization_tokens: int = 0   # summed across both writes
    answer_tokens: int = 0          # tokens the answer-generation call spent


# ── The spine ────────────────────────────────────────────────────────────────

@dataclass
class MemoryTrace:
    """One structured record per agent turn."""
    schema_version: str = SCHEMA_VERSION

    # identity
    session_id: str = ""
    turn_index: int = 0           # conversation turn number (agent._turn_count)
    turn_id: str = ""             # uuid — unique key for this turn
    timestamp: str = ""           # ISO-8601
    query: str = ""               # the user message
    answer: str = ""              # the generated response

    # the three stages
    user_write: WriteTrace = field(default_factory=lambda: WriteTrace(label="user"))
    read: ReadTrace = field(default_factory=ReadTrace)
    assistant_write: WriteTrace = field(default_factory=lambda: WriteTrace(label="assistant"))

    totals: Totals = field(default_factory=Totals)

    def to_dict(self) -> dict:
        """Plain dict, JSON-ready (str-enums serialise as their values)."""
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)