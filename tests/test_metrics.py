"""
API-free test: metrics aggregation over trace dicts + JSONL loading.

Run from repo root (venv active):
    python -m pytest tests/test_metrics.py -v
"""

import json

from memoryos.metrics import compute_metrics, load_traces


def _trace(latency_total, llm, retrieved_scores, *, evictions=0, summary=False,
           answer_tokens=0, summ_tokens=0, prompt=(0, 0, 0)):
    """Minimal but structurally complete trace dict for testing."""
    return {
        "schema_version": "1.0",
        "session_id": "s", "turn_index": 1, "turn_id": "t",
        "user_write": {
            "episodic": {"summary_produced": summary},
            "evictions": [{} for _ in range(evictions)],
        },
        "assistant_write": {
            "episodic": {"summary_produced": False},
            "evictions": [],
        },
        "read": {
            "long_term": {"retrieved": [{"score": s} for s in retrieved_scores]},
        },
        "totals": {
            "latency_ms_total": latency_total,
            "latency_ms_by_stage": {"llm": llm, "read": 0, "user_write": 0, "assistant_write": 0},
            "prompt_tokens_by_tier": {"short_term": prompt[0], "long_term": prompt[1], "episodic": prompt[2]},
            "answer_tokens": answer_tokens,
            "summarization_tokens": summ_tokens,
        },
    }


def test_compute_metrics():
    traces = [
        _trace(100, 60, [0.9], answer_tokens=5),
        _trace(200, 120, [], evictions=1, summary=True, answer_tokens=15, summ_tokens=50),
    ]
    m = compute_metrics(traces)

    assert m["turn_count"] == 2
    assert m["latency_ms"]["total_p50"] == 150.0          # median of 100, 200
    assert m["retrieval"]["hit_rate"] == 0.5              # 1 of 2 turns retrieved
    assert m["retrieval"]["avg_top_score"] == 0.9         # only the hit counts
    assert m["evictions"]["total"] == 1
    assert m["evictions"]["turns_with_evictions"] == 1
    assert m["episodes_created"] == 1
    assert m["tokens"]["avg_answer"] == 10.0
    assert m["tokens"]["total_summarization"] == 50


def test_empty_metrics():
    assert compute_metrics([]) == {"turn_count": 0}


def test_load_traces_roundtrip(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        json.dumps(_trace(100, 60, [0.9])) + "\n"
        + json.dumps(_trace(200, 120, [])) + "\n",
        encoding="utf-8",
    )
    traces = load_traces(str(p))
    assert len(traces) == 2
    # missing file -> empty, not an error
    assert load_traces(str(tmp_path / "missing.jsonl")) == []