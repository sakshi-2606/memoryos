"""
INTEGRATION test: a live agent turn produces a full MemoryTrace.

Unlike the other tests, this one is NOT stubbed — it makes real OpenAI calls
(embeddings for both turns + the LLM turns), so it costs a few cents and needs
OPENAI_API_KEY set. It's the real proof the spine works end-to-end on a live
turn.

Run from repo root (venv active):
    python tests/test_agent_trace_integration.py
"""

import json
import time
import pytest

from memoryos.config import AppConfig
from memoryos.agent import MemoryAgent

pytestmark = pytest.mark.integration


def test_live_turn_trace() -> None:
    config = AppConfig()
    config.validate()  # fail fast if OPENAI_API_KEY is missing

    agent = MemoryAgent(config, session_id=f"trace-int-{int(time.time())}")

    # turn 1 — plant a fact worth storing
    agent.chat(
        "I'm building a RAG system with ChromaDB and I prefer Python for backend work."
    )

    # turn 2 — should retrieve the planted fact from long-term
    result = agent.chat("What language should I use for my backend?")

    trace = result["trace"]
    assert trace is not None, "no trace attached — is config.trace.enabled True?"

    print(json.dumps(trace, indent=2))

    # ── sanity assertions on the live trace ───────────────────────────────────
    assert trace["schema_version"], "schema_version missing"
    assert trace["turn_index"] == 2
    assert trace["turn_id"], "turn_id missing"
    assert trace["query"] == "What language should I use for my backend?"
    assert trace["answer"], "answer not captured"

    # three stages present
    assert trace["user_write"]["label"] == "user"
    assert trace["assistant_write"]["label"] == "assistant"
    assert "retrieved" in trace["read"]["long_term"]

    # totals populated with real numbers
    totals = trace["totals"]
    assert totals["latency_ms_total"] > 0
    assert set(totals["latency_ms_by_stage"]) == {
        "user_write", "read", "llm", "assistant_write"
    }
    assert set(totals["prompt_tokens_by_tier"]) == {
        "short_term", "long_term", "episodic"
    }
    assert totals["answer_tokens"] >= 0

    retrieved = trace["read"]["long_term"]["retrieved"]
    print("\n✓ live turn produced a MemoryTrace")
    print(f"  total latency: {totals['latency_ms_total']}ms "
          f"(llm={totals['latency_ms_by_stage']['llm']}ms)")
    print(f"  long-term retrieved: {len(retrieved)} item(s)"
          + (f", top score={retrieved[0]['score']}" if retrieved else ""))
    print(f"  prompt tokens by tier: {totals['prompt_tokens_by_tier']}")
    print(f"  answer tokens: {totals['answer_tokens']}")


if __name__ == "__main__":
    test_live_turn_trace()