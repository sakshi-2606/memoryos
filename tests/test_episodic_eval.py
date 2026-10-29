"""
Episodic retention eval (pytest) — summarisation must not silently drop facts.

INTEGRATION: makes real OpenAI calls — embeddings for each planted turn plus one
LLM summarisation call. Needs OPENAI_API_KEY.

Plants the episodic-case facts into a session whose summarisation threshold is
lowered so compression actually fires, then asserts each planted fact survives in
the episode summary. This is the "episodic silently dropping facts" failure mode.

Run from repo root (venv active):
    python -m pytest tests/test_episodic_eval.py -v

Note: the summariser runs at temperature 0.3, so there's mild nondeterminism; the
keywords chosen are distinctive enough that a faithful summary should keep them.
The retention keywords are expected labels that could later live in the dataset.
"""

import dataclasses
import uuid
from pathlib import Path

import pytest

from memoryos.config import AppConfig
from memoryos.agent import MemoryAgent
from memoryos.memeval import load_dataset

pytestmark = pytest.mark.integration

DATASET = load_dataset(Path(__file__).parent / "memeval_dataset.json")

# a distinctive token from each planted fact — a faithful episode should retain it
RETENTION_KEYWORDS = {
    "f6": "shellfish",
    "f7": "Bengaluru",
    "f8": "Honda",
}


def _low_threshold_config() -> AppConfig:
    """AppConfig that summarises after 3 turns so compression fires in-test."""
    base = AppConfig()
    return dataclasses.replace(
        base,
        memory=dataclasses.replace(
            base.memory,
            summarisation_threshold=3,
            turns_to_summarise=3,
        ),
    )


@pytest.mark.parametrize("case", DATASET.episodic_cases, ids=lambda c: c.id)
def test_episode_retains_facts(make_agent, case):
    """Every fact compressed into an episode must still be present in the summary."""
    agent = make_agent(
    f"eval-epi-{case.id}-{uuid.uuid4().hex[:8]}",
    config=_low_threshold_config(),
    )

    facts = [DATASET.fact_by_id(fid) for fid in case.facts]
    # plant as user turns to drive summarisation (threshold = 3)
    for f in facts:
        agent.memory.add_message_traced("user", f.content, label="user")

    episodes = agent.memory.episodic.get_all_episodes()
    assert episodes, (
        f"{case.id}: no episode created — summarisation didn't trigger"
    )

    summary = " ".join(ep.summary for ep in episodes).lower()
    missing = [
        fid for fid in case.facts
        if RETENTION_KEYWORDS[fid].lower() not in summary
    ]
    assert not missing, (
        f"{case.id}: episode dropped {missing} "
        f"(keywords {[RETENTION_KEYWORDS[m] for m in missing]}); "
        f"summary was: {summary!r}"
    )