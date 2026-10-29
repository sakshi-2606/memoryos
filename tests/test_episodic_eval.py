"""
Episodic retention eval (pytest) — summarisation must not silently drop facts.

INTEGRATION: real OpenAI calls (embeddings per turn + LLM summarisation on each
eviction). Needs OPENAI_API_KEY.

With token-based eviction, summarisation fires when the short-term token budget
is exceeded. This test uses a tiny budget, plants the facts, then adds filler
turns until every planted fact has been evicted and summarised — robust to exact
token counts — and asserts each fact survives across the resulting episode(s).

Run from repo root (venv active):
    python -m pytest tests/test_episodic_eval.py -v
"""

import dataclasses
import uuid
from pathlib import Path

import pytest

from memoryos.config import AppConfig
from memoryos.memeval import load_dataset

pytestmark = pytest.mark.integration

DATASET = load_dataset(Path(__file__).parent / "memeval_dataset.json")

# a distinctive token from each planted fact — a faithful episode should retain it
RETENTION_KEYWORDS = {
    "f6": "shellfish",
    "f7": "Bengaluru",
    "f8": "Honda",
}

FILLER = "this is an unrelated filler turn used only to advance the buffer"


def _small_budget_config() -> AppConfig:
    """Tiny token budget so eviction + summarisation fire quickly in-test."""
    base = AppConfig()
    return dataclasses.replace(
        base,
        memory=dataclasses.replace(
            base.memory,
            short_term_token_budget=15,
            short_term_evict_to_fraction=0.3,
        ),
    )


@pytest.mark.parametrize("case", DATASET.episodic_cases, ids=lambda c: c.id)
def test_episode_retains_facts(make_agent, case):
    """Every fact compressed into an episode must still be present in the summary."""
    agent = make_agent(
        f"eval-epi-{case.id}-{uuid.uuid4().hex[:8]}",
        config=_small_budget_config(),
    )

    facts = [DATASET.fact_by_id(fid) for fid in case.facts]
    for f in facts:
        agent.memory.add_message_traced("user", f.content, label="user")

    # Flush: add filler turns until every target fact has left the buffer
    # (i.e. been evicted and summarised). Robust to exact token counts.
    target_contents = {f.content for f in facts}
    for _ in range(20):
        still_buffered = any(
            m.content in target_contents for m in agent.memory.short_term.get_all()
        )
        if not still_buffered:
            break
        agent.memory.add_message_traced("user", FILLER, label="user")

    episodes = agent.memory.episodic.get_all_episodes()
    assert episodes, f"{case.id}: no episode created — eviction/summarisation didn't fire"

    summary = " ".join(ep.summary for ep in episodes).lower()
    missing = [
        fid for fid in case.facts
        if RETENTION_KEYWORDS[fid].lower() not in summary
    ]
    assert not missing, (
        f"{case.id}: episode dropped {missing} "
        f"(keywords {[RETENTION_KEYWORDS[m] for m in missing]}); summary: {summary!r}"
    )