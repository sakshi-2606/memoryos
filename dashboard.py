"""
MemoryOS observability dashboard.

A small standalone Streamlit page over the per-turn trace log (traces.jsonl):
headline metrics, latency and token trends over turns, and a recent-turns table.

Run from repo root (venv active):
    streamlit run dashboard.py
"""

import streamlit as st

from memoryos.config import AppConfig
from memoryos.metrics import load_traces, compute_metrics

st.set_page_config(page_title="MemoryOS Observability", layout="wide")
st.title("MemoryOS — Observability")

cfg = AppConfig()
path = st.text_input("Trace log path", cfg.observability.trace_log_path)

traces = load_traces(path)
if not traces:
    st.warning(f"No traces found at `{path}`. Run some chat turns first "
               "(e.g. the API, the Streamlit app, or the agent integration test).")
    st.stop()

m = compute_metrics(traces)

# ── headline tiles ───────────────────────────────────────────────────────────
r1 = st.columns(4)
r1[0].metric("Turns", m["turn_count"])
r1[1].metric("Latency p50 (ms)", m["latency_ms"]["total_p50"])
r1[2].metric("Latency p95 (ms)", m["latency_ms"]["total_p95"])
r1[3].metric("Retrieval hit rate", f'{m["retrieval"]["hit_rate"]:.0%}')

r2 = st.columns(4)
r2[0].metric("Avg top score", m["retrieval"]["avg_top_score"])
r2[1].metric("Evictions", m["evictions"]["total"])
r2[2].metric("Episodes created", m["episodes_created"])
r2[3].metric("Summarisation tokens", m["tokens"]["total_summarization"])

# ── trends over turns ────────────────────────────────────────────────────────
st.subheader("Latency over turns (ms)")
st.line_chart({
    "total": [t["totals"]["latency_ms_total"] for t in traces],
    "llm": [t["totals"]["latency_ms_by_stage"].get("llm", 0) for t in traces],
    "read": [t["totals"]["latency_ms_by_stage"].get("read", 0) for t in traces],
})

st.subheader("Prompt tokens by tier over turns")
st.line_chart({
    "short_term": [t["totals"]["prompt_tokens_by_tier"].get("short_term", 0) for t in traces],
    "long_term": [t["totals"]["prompt_tokens_by_tier"].get("long_term", 0) for t in traces],
    "episodic": [t["totals"]["prompt_tokens_by_tier"].get("episodic", 0) for t in traces],
})

# ── recent turns ─────────────────────────────────────────────────────────────
st.subheader("Recent turns")
rows = []
for t in traces[-25:]:
    lt = t.get("read", {}).get("long_term", {}).get("retrieved", [])
    rows.append({
        "turn": t.get("turn_index"),
        "session": str(t.get("session_id", ""))[:8],
        "query": str(t.get("query", ""))[:60],
        "latency_ms": t.get("totals", {}).get("latency_ms_total"),
        "retrieved": len(lt),
        "top_score": lt[0].get("score") if lt else None,
        "evictions": (len(t.get("user_write", {}).get("evictions", []))
                      + len(t.get("assistant_write", {}).get("evictions", []))),
        "answer_tokens": t.get("totals", {}).get("answer_tokens"),
    })
st.dataframe(rows, use_container_width=True)