"""
Loader + schema for the memory evaluation dataset.

The dataset is a small, versioned, labeled set of planted facts and probes used
by the can't-fail eval suite. This module just loads and validates it; the actual
probe logic lives in the tests that consume it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DATASET_VERSION = "1.0"


# ── dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class Fact:
    """A planted statement, injected into a session as a user turn."""
    id: str
    session: str
    content: str


@dataclass
class RetrievalProbe:
    """A query whose correct long-term hits are `relevant` (the rest are distractors)."""
    id: str
    session: str
    query: str
    relevant: list[str]              # fact ids that count as correct
    expected_tier: str | None = None  # e.g. "long_term"


@dataclass
class LeakageProbe:
    """A fact planted in one session that must NOT surface in another session."""
    id: str
    query: str
    fact: str                 # fact id planted in planted_session
    planted_session: str
    query_session: str        # a different session where the fact must be absent


@dataclass
class EpisodicCase:
    """Facts planted in a session that must survive episodic summarisation."""
    id: str
    session: str
    facts: list[str]          # fact ids that must appear in the episode summary


@dataclass
class EvalDataset:
    dataset_version: str
    description: str
    facts: list[Fact]
    retrieval_probes: list[RetrievalProbe]
    leakage_probes: list[LeakageProbe]
    episodic_cases: list[EpisodicCase]

    # convenience lookups
    def fact_by_id(self, fid: str) -> Fact:
        for f in self.facts:
            if f.id == fid:
                return f
        raise KeyError(f"no fact with id {fid!r}")

    def facts_for_session(self, sid: str) -> list[Fact]:
        return [f for f in self.facts if f.session == sid]


# ── loading + validation ───────────────────────────────────────────────────────

def load_dataset(path: str | Path) -> EvalDataset:
    """Load the dataset JSON, parse into dataclasses, and validate referential integrity."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    ds = EvalDataset(
        dataset_version=data["dataset_version"],
        description=data.get("description", ""),
        facts=[Fact(**f) for f in data["facts"]],
        retrieval_probes=[RetrievalProbe(**p) for p in data["retrieval_probes"]],
        leakage_probes=[LeakageProbe(**lk) for lk in data["leakage_probes"]],
        episodic_cases=[EpisodicCase(**ep) for ep in data["episodic_cases"]],
    )
    _validate(ds)
    return ds


def _validate(ds: EvalDataset) -> None:
    assert ds.dataset_version == DATASET_VERSION, (
        f"dataset_version {ds.dataset_version!r} != loader {DATASET_VERSION!r} "
        "— bump the loader or the dataset"
    )

    fact_ids = [f.id for f in ds.facts]
    assert len(fact_ids) == len(set(fact_ids)), "duplicate fact ids"
    id_set = set(fact_ids)
    session_set = {f.session for f in ds.facts}

    for p in ds.retrieval_probes:
        assert p.session in session_set, f"probe {p.id}: unknown session {p.session!r}"
        assert p.relevant, f"probe {p.id}: no relevant facts labeled"
        for r in p.relevant:
            assert r in id_set, f"probe {p.id}: unknown relevant fact {r!r}"

    for lk in ds.leakage_probes:
        assert lk.fact in id_set, f"leakage {lk.id}: unknown fact {lk.fact!r}"
        assert lk.planted_session != lk.query_session, (
            f"leakage {lk.id}: planted and query sessions must differ"
        )

    for ep in ds.episodic_cases:
        assert ep.facts, f"episodic {ep.id}: no facts to check"
        for fid in ep.facts:
            assert fid in id_set, f"episodic {ep.id}: unknown fact {fid!r}"