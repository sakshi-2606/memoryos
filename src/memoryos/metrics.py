"""
Metrics aggregation over persisted memory traces.

Pure functions over the JSONL trace log: load the records, roll them up into
headline metrics (latency percentiles, tokens, retrieval quality, evictions,
episodes). No external dependencies — the dashboard consumes these.
"""

import json
from pathlib import Path


def load_traces(path: str) -> list[dict]:
    """Read the JSONL trace log into a list of trace dicts (empty if absent)."""
    p = Path(path)
    if not p.exists():
        return []
    traces: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            continue   # skip a torn/partial line rather than failing the whole read
    return traces


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100). 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def compute_metrics(traces: list[dict]) -> dict:
    """Roll a list of trace dicts up into headline metrics."""
    n = len(traces)
    if n == 0:
        return {"turn_count": 0}

    def stage(name: str) -> list[float]:
        return [t.get("totals", {}).get("latency_ms_by_stage", {}).get(name, 0.0) for t in traces]

    total_lat = [t.get("totals", {}).get("latency_ms_total", 0.0) for t in traces]

    retrieved = [
        t.get("read", {}).get("long_term", {}).get("retrieved", []) for t in traces
    ]
    retrieved_counts = [len(r) for r in retrieved]
    top_scores = [r[0].get("score", 0.0) for r in retrieved if r]

    def evictions_in(t: dict) -> int:
        return (
            len(t.get("user_write", {}).get("evictions", []))
            + len(t.get("assistant_write", {}).get("evictions", []))
        )

    eviction_counts = [evictions_in(t) for t in traces]

    def episode_made(t: dict) -> bool:
        return (
            t.get("user_write", {}).get("episodic", {}).get("summary_produced", False)
            or t.get("assistant_write", {}).get("episodic", {}).get("summary_produced", False)
        )

    def ptoken(t: dict, tier: str) -> float:
        return t.get("totals", {}).get("prompt_tokens_by_tier", {}).get(tier, 0)

    return {
        "turn_count": n,
        "latency_ms": {
            "total_p50": round(_percentile(total_lat, 50), 1),
            "total_p95": round(_percentile(total_lat, 95), 1),
            "llm_p50": round(_percentile(stage("llm"), 50), 1),
            "llm_p95": round(_percentile(stage("llm"), 95), 1),
            "read_p50": round(_percentile(stage("read"), 50), 1),
        },
        "tokens": {
            "avg_prompt_short_term": _avg([ptoken(t, "short_term") for t in traces]),
            "avg_prompt_long_term": _avg([ptoken(t, "long_term") for t in traces]),
            "avg_prompt_episodic": _avg([ptoken(t, "episodic") for t in traces]),
            "avg_answer": _avg([t.get("totals", {}).get("answer_tokens", 0) for t in traces]),
            "total_summarization": sum(
                t.get("totals", {}).get("summarization_tokens", 0) for t in traces
            ),
        },
        "retrieval": {
            "hit_rate": round(sum(1 for c in retrieved_counts if c > 0) / n, 3),
            "avg_retrieved": _avg(retrieved_counts),
            "avg_top_score": _avg(top_scores),
        },
        "evictions": {
            "total": sum(eviction_counts),
            "turns_with_evictions": sum(1 for e in eviction_counts if e > 0),
        },
        "episodes_created": sum(1 for t in traces if episode_made(t)),
    }