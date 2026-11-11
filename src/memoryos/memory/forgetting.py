"""
Forgetting policies for the long-term memory store.

Ported from MemoryBench (github.com/sakshi-2606/memorybench), which benchmarked
compression x forgetting strategies and found **Hybrid forgetting the clear
winner** — FIFO and Surprise were catastrophic (0% accuracy; the wrong policy is
indistinguishable from total memory failure). MemoryOS runs that proven-best
policy on its long-term store.

Hybrid ranks each stored memory by a retention score

    recency_w * recency + importance_w * importance + surprise_w * surprise

(all min-max normalised) and evicts the lowest-retention memories when the store
exceeds capacity. Surprise = bag-of-words L2 distance from the centroid, so
distinctive memories are protected (von Restorff effect); it's a deliberately
cheap proxy — no embedding/API cost during the eviction decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ForgettingPolicy(str, Enum):
    FIFO = "fifo"
    IMPORTANCE = "importance"
    SURPRISE = "surprise"
    HYBRID = "hybrid"


@dataclass
class EvictionCandidate:
    """Minimal view of a stored memory that a forgetting policy ranks."""
    episode_id: str
    content: str
    timestamp: float      # recency signal (higher = more recent)
    importance: float     # 0.0–2.0, default neutral 1.0


class ForgettingManager:
    """
    Decides which stored memories to evict when the store exceeds capacity.

    Args:
        policy: which forgetting heuristic to apply (default HYBRID).
        max_episodes: maximum memories to retain after eviction.
        hybrid_weights: (recency_w, importance_w, surprise_w) for HYBRID;
            must sum to 1.0.
    """

    def __init__(
        self,
        policy: ForgettingPolicy | str = ForgettingPolicy.HYBRID,
        max_episodes: int = 100,
        hybrid_weights: tuple[float, float, float] = (0.2, 0.5, 0.3),
    ) -> None:
        self.policy = ForgettingPolicy(policy)
        self.max_episodes = max_episodes
        self.recency_w, self.importance_w, self.surprise_w = hybrid_weights
        if not abs(sum(hybrid_weights) - 1.0) < 1e-6:
            raise ValueError("hybrid_weights must sum to 1.0")

    def candidates_to_evict(self, episodes: list[EvictionCandidate]) -> list[str]:
        """Return episode_ids to delete (empty if count <= max_episodes)."""
        if len(episodes) <= self.max_episodes:
            return []
        n_to_evict = len(episodes) - self.max_episodes
        return self._rank_by_evictability(episodes)[:n_to_evict]

    # ── ranking (most-evictable first) ────────────────────────────────────────

    def _rank_by_evictability(self, episodes: list[EvictionCandidate]) -> list[str]:
        match self.policy:
            case ForgettingPolicy.FIFO:
                return self._rank_fifo(episodes)
            case ForgettingPolicy.IMPORTANCE:
                return self._rank_importance(episodes)
            case ForgettingPolicy.SURPRISE:
                return self._rank_surprise(episodes)
            case ForgettingPolicy.HYBRID:
                return self._rank_hybrid(episodes)

    def _rank_fifo(self, episodes):
        """Oldest first — the catastrophic baseline."""
        return [e.episode_id for e in sorted(episodes, key=lambda e: e.timestamp)]

    def _rank_importance(self, episodes):
        """Least important first."""
        return [e.episode_id for e in sorted(episodes, key=lambda e: e.importance)]

    def _rank_surprise(self, episodes):
        """Most redundant (lowest surprise) first."""
        scores = self._compute_surprise_scores(episodes)
        return [eid for eid, _ in sorted(scores.items(), key=lambda x: x[1])]

    def _rank_hybrid(self, episodes):
        """Lowest retention score first (recency + importance + surprise)."""
        if not episodes:
            return []
        ts = np.array([e.timestamp for e in episodes], dtype=float)
        imp = np.array([e.importance for e in episodes], dtype=float)
        surprise = self._compute_surprise_scores(episodes)
        surp = np.array([surprise[e.episode_id] for e in episodes], dtype=float)

        def norm(a: np.ndarray) -> np.ndarray:
            rng = a.max() - a.min()
            return (a - a.min()) / rng if rng > 1e-9 else np.ones_like(a) * 0.5

        retention = (
            self.recency_w * norm(ts)
            + self.importance_w * norm(imp)
            + self.surprise_w * norm(surp)
        )
        order = np.argsort(retention)   # ascending → most evictable first
        return [episodes[i].episode_id for i in order]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _compute_surprise_scores(self, episodes) -> dict[str, float]:
        """BoW L2 distance from the centroid; distinctive = high = retained."""
        if not episodes:
            return {}

        vocab: dict[str, int] = {}
        for ep in episodes:
            for word in ep.content.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)

        V = len(vocab)
        if V == 0:
            return {e.episode_id: 1.0 for e in episodes}

        vectors = np.zeros((len(episodes), V))
        for i, ep in enumerate(episodes):
            words = ep.content.lower().split()
            for word in words:
                if word in vocab:
                    vectors[i, vocab[word]] += 1
            if words:
                vectors[i] /= len(words)

        centroid = vectors.mean(axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        return {ep.episode_id: float(d) for ep, d in zip(episodes, distances)}