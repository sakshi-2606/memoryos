"""
Retrieval eval suite (pytest) — the can't-fail checks for memory retrieval.

INTEGRATION: makes real OpenAI embedding calls (+ one LLM call in the F1 test),
so it needs OPENAI_API_KEY. The dataset is the labeled fixture; each test plants
a session's facts, then probes.

Run from repo root (venv active):
    python -m pytest tests/test_retrieval_eval.py -v

Design note — metrics for a top-k retriever:
  The system admits memories above a similarity threshold, then ranks them. With
  a permissive threshold it returns almost everything, so RAW precision is a poor
  gate. The real quality signal is RANKING: are the relevant facts at the top?
  So the passing gate is precision@k (relevant facts in the top-k slots) + recall.
  Threshold over-admission is a real but separate issue, captured as F2 (xfail).
"""

import uuid
from pathlib import Path

import pytest

from memoryos.config import AppConfig
from memoryos.agent import MemoryAgent
from memoryos.memeval import load_dataset

pytestmark = pytest.mark.integration

DATASET = load_dataset(Path(__file__).parent / "memeval_dataset.json")


@pytest.fixture
def make_agent():
    """
    Factory that builds an agent and plants facts straight into long-term
    (embeddings only — no LLM). Cleans up each session's Chroma collection
    afterwards so the eval doesn't litter the store.
    """
    created: list[MemoryAgent] = []

    def _factory(session_id: str, facts) -> MemoryAgent:
        agent = MemoryAgent(AppConfig(), session_id=session_id)
        for f in facts:
            agent.memory.long_term.add("user", f.content, turn_index=0)
        created.append(agent)
        return agent

    yield _factory

    for a in created:
        try:
            a.memory.long_term._chroma.delete_collection(
                a.memory.long_term._collection.name
            )
        except Exception:
            pass


def _retrieved_fact_ids(memories, facts) -> list[str]:
    """Map retrieved memory contents back to their planted fact ids (rank order preserved)."""
    by_content = {f.content: f.id for f in facts}
    return [by_content[m.content] for m in memories if m.content in by_content]


@pytest.mark.parametrize("probe", DATASET.retrieval_probes, ids=lambda p: p.id)
def test_relevant_ranked_first(make_agent, probe):
    """
    The relevant fact(s) must be retrieved AND ranked at the very top.

    This is the real retrieval-quality gate: precision@k (k = #relevant) must be
    1.0 — the relevant facts occupy the top slots — and recall must be 1.0 (which
    also serves as the right-tier check: the fact is retrievable from long-term).
    """
    facts = DATASET.facts_for_session(probe.session)
    agent = make_agent(f"eval-rank-{probe.id}-{uuid.uuid4().hex[:8]}", facts)

    memories = agent.memory.long_term.retrieve(probe.query)   # sorted by score desc
    retrieved = _retrieved_fact_ids(memories, facts)
    relevant = set(probe.relevant)

    tp = {i for i in retrieved if i in relevant}
    recall = len(tp) / len(relevant)

    k = len(relevant)
    top_k = retrieved[:k]
    precision_at_k = sum(1 for i in top_k if i in relevant) / k

    assert recall == 1.0, f"{probe.id}: missed {relevant - tp}; retrieved {retrieved}"
    assert precision_at_k == 1.0, (
        f"{probe.id}: relevant fact not ranked at top — top{k}={top_k}, "
        f"full order {retrieved}"
    )


@pytest.mark.xfail(
    reason="F2: similarity_threshold (0.5) with similarity=1-(distance/2) admits "
           "unrelated distractors (they clear 0.5), so threshold-filtered precision "
           "is poor — retrieval leans on ranking, not filtering. Tune in a later slice.",
    strict=True,
)
def test_threshold_excludes_distractors(make_agent):
    """Everything the retriever ADMITS (post-threshold) should be mostly relevant."""
    probe = DATASET.retrieval_probes[0]
    facts = DATASET.facts_for_session(probe.session)
    agent = make_agent(f"eval-f2-{uuid.uuid4().hex[:8]}", facts)

    memories = agent.memory.long_term.retrieve(probe.query)
    retrieved = _retrieved_fact_ids(memories, facts)
    relevant = set(probe.relevant)

    precision = sum(1 for i in retrieved if i in relevant) / len(retrieved)
    assert precision >= 0.5, (
        f"distractors cleared the similarity threshold — precision {precision:.2f}, "
        f"retrieved {retrieved}"
    )

def test_turn_does_not_retrieve_current_query(make_agent):
    """A live turn should not surface the current query as its own top memory."""
    probe = DATASET.retrieval_probes[0]
    facts = DATASET.facts_for_session(probe.session)
    agent = make_agent(f"eval-f1-{uuid.uuid4().hex[:8]}", facts)

    result = agent.chat(probe.query)          # real turn: writes query, then reads
    retrieved = result["trace"]["read"]["long_term"]["retrieved"]
    assert retrieved, "nothing retrieved at all"

    top = retrieved[0]
    assert top["snippet"] != probe.query, (
        f"self-retrieval: top hit is the current query itself (score {top['score']})"
    )