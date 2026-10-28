"""
API-free test: manager builds ReadTrace + WriteTrace with timings, scores, tokens.

Embeddings are stubbed with a fixed vector, so no OpenAI call is made, and we
stay below the summarisation threshold so no summariser LLM call fires either.
ChromaDB runs locally in an isolated, timestamped session namespace. tiktoken
runs for real (it's local — no network).

Run from repo root (venv active):
    python tests/test_manager_trace.py
"""

import time

from memoryos.config import AppConfig
from memoryos.memory.manager import MemoryManager
from memoryos.memory.trace import TierStatus, EvictionPolicy


def main() -> None:
    config = AppConfig()
    mgr = MemoryManager(config, session_id=f"trace-test-{int(time.time())}")

    # stub embeddings: deterministic fixed vector, no network
    dim = config.model.embedding_dim
    mgr.long_term._embed = lambda text: [0.1] * dim

    # ── write path ──────────────────────────────────────────────────────────
    wt = mgr.add_message_traced(
        "user", "I am building a RAG system with ChromaDB", label="user"
    )
    assert wt.label == "user"
    assert wt.short_term.status == TierStatus.OK
    assert wt.short_term.latency_ms >= 0
    assert wt.long_term.status == TierStatus.OK and wt.long_term.stored is True
    assert wt.episodic.status == TierStatus.SKIPPED      # below threshold
    assert wt.evictions == []                            # window not full

    # a noise message is skipped by the long-term filter
    wt_noise = mgr.add_message_traced("user", "ok", label="user")
    assert wt_noise.long_term.status == TierStatus.SKIPPED
    assert wt_noise.long_term.stored is False

    # ── read path ───────────────────────────────────────────────────────────
    messages, read = mgr.assemble_prompt_traced("what database am I using?")

    # prompt structure preserved: system message first, then history
    assert messages[0]["role"] == "system"
    assert len(messages) >= 2

    # long-term returned a structured hit with a score + snippet
    assert read.long_term.status == TierStatus.OK
    assert len(read.long_term.retrieved) >= 1
    top = read.long_term.retrieved[0]
    assert 0.0 <= top.score <= 1.0
    assert top.snippet
    assert top.used_in_prompt is True

    # tiktoken populated real token counts
    assert read.long_term.tokens_contributed > 0
    assert read.short_term.tokens_contributed > 0
    assert read.short_term.turns_pulled >= 1

    print("✓ manager builds WriteTrace + ReadTrace")
    print(f"  write: long_term stored={wt.long_term.stored}, "
          f"st_latency={wt.short_term.latency_ms}ms, "
          f"lt_latency={wt.long_term.latency_ms}ms")
    print(f"  read:  retrieved={len(read.long_term.retrieved)} "
          f"(top score={read.long_term.retrieved[0].score}), "
          f"lt_tokens={read.long_term.tokens_contributed}, "
          f"st_tokens={read.short_term.tokens_contributed}, "
          f"turns_pulled={read.short_term.turns_pulled}")


if __name__ == "__main__":
    main()