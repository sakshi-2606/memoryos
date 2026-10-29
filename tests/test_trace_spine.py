"""
Standalone smoke test for the memory-trace spine (Batch 1).

This does NOT touch the live turn loop, tiers, or any API. It just proves:
  1. a MemoryTrace can be built with realistic nested data,
  2. it serialises cleanly to JSON (enums included),
  3. it round-trips (json.loads gives back the same structure),
  4. schema_version is present — the contract later slices assert on.

Run from the repo root:
    python test_trace_spine.py
"""

import json
import uuid
from datetime import datetime

from memoryos.memory.trace import (
    MemoryTrace,
    ReadTrace, ShortTermRead, LongTermRead, EpisodicRead,
    WriteTrace, ShortTermWrite, LongTermWrite, EpisodicWrite,
    RetrievedItem, EpisodeItem, EvictionRecord,
    Totals, TierStatus, EvictionPolicy, SCHEMA_VERSION,
)


def build_sample_trace() -> MemoryTrace:
    """Construct a trace that exercises every branch of the schema."""
    return MemoryTrace(
        session_id="demo-session",
        turn_index=3,
        turn_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        query="what database am I using?",
        answer="You're using ChromaDB for local vector storage.",

        user_write=WriteTrace(
            label="user",
            short_term=ShortTermWrite(status=TierStatus.OK, latency_ms=0.4),
            long_term=LongTermWrite(status=TierStatus.OK, latency_ms=120.5, stored=True),
            episodic=EpisodicWrite(status=TierStatus.SKIPPED),
            evictions=[
                EvictionRecord(
                    tier="short_term",
                    item_id="2",
                    policy=EvictionPolicy.WINDOW_OVERFLOW,
                    reason="window full (size=10)",
                )
            ],
        ),

        read=ReadTrace(
            short_term=ShortTermRead(
                status=TierStatus.OK, latency_ms=0.2,
                turns_pulled=6, tokens_contributed=812,
            ),
            long_term=LongTermRead(
                status=TierStatus.OK, latency_ms=95.1, tokens_contributed=140,
                retrieved=[
                    RetrievedItem(id="abc-123", score=0.83,
                                  snippet="I am building a RAG system with ChromaDB",
                                  used_in_prompt=True),
                    RetrievedItem(id="def-456", score=0.61,
                                  snippet="I prefer Python over Java",
                                  used_in_prompt=True),
                ],
            ),
            episodic=EpisodicRead(
                status=TierStatus.EMPTY, latency_ms=0.1, tokens_contributed=0,
            ),
        ),

        assistant_write=WriteTrace(
            label="assistant",
            short_term=ShortTermWrite(status=TierStatus.OK, latency_ms=0.3),
            long_term=LongTermWrite(status=TierStatus.OK, latency_ms=110.0, stored=True),
            episodic=EpisodicWrite(
                status=TierStatus.OK, latency_ms=640.0,
                summary_produced=True, summarization_tokens=187,
            ),
            evictions=[
                EvictionRecord(
                    tier="short_term",
                    item_id="0",
                    policy=EvictionPolicy.SUMMARISATION,
                    reason="compressed into episode_0000",
                )
            ],
        ),

        totals=Totals(
            latency_ms_total=1066.6,
            latency_ms_by_stage={
                "user_write": 120.9, "read": 95.4,
                "llm": 210.3, "assistant_write": 750.3,
            },
            prompt_tokens_by_tier={"short_term": 812, "long_term": 140, "episodic": 0},
            summarization_tokens=187,
            answer_tokens=24,
        ),
    )


def test_trace_spine_serialises() -> None:
    trace = build_sample_trace()

    # 1. serialise
    as_dict = trace.to_dict()
    as_json = trace.to_json()

    # 2. round-trip
    reloaded = json.loads(as_json)

    # 3. assertions
    assert reloaded["schema_version"] == SCHEMA_VERSION, "schema_version missing/wrong"
    assert reloaded["read"]["long_term"]["retrieved"][0]["score"] == 0.83
    # enum serialised as its string value, not "TierStatus.OK"
    assert reloaded["user_write"]["short_term"]["status"] == "ok", \
        f'enum did not serialise to value: {reloaded["user_write"]["short_term"]["status"]!r}'
    assert reloaded["assistant_write"]["evictions"][0]["policy"] == "summarisation"
    assert reloaded["user_write"]["evictions"][0]["policy"] == "window_overflow"
    assert reloaded["totals"]["prompt_tokens_by_tier"]["short_term"] == 812

    print(as_json)
    print("\n✓ trace spine OK — schema_version =", SCHEMA_VERSION)
    print("✓ enums serialise to values, round-trip clean, all fields present")


if __name__ == "__main__":
    test_trace_spine_serialises()