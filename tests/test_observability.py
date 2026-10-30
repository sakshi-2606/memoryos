"""
API-free test: the trace logger writes JSONL and honours the enabled flag.

Uses pytest's tmp_path so nothing is written into the repo. Builds a MemoryTrace
directly (no agent, no network).

Run from repo root (venv active):
    python -m pytest tests/test_observability.py -v
"""

import json

from memoryos.memory.trace import MemoryTrace, Totals
from memoryos.observability import TraceLogger


def test_trace_logged_as_jsonl(tmp_path):
    log_path = tmp_path / "traces.jsonl"
    tl = TraceLogger(str(log_path), enabled=True)

    trace = MemoryTrace(
        session_id="sess-1",
        turn_index=1,
        turn_id="turn-abc",
        query="hello",
        answer="hi",
        totals=Totals(latency_ms_total=12.3, answer_tokens=5),
    )
    tl.log(trace)
    tl.log(trace)   # a second turn appends, not overwrites

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    rec = json.loads(lines[0])
    assert rec["session_id"] == "sess-1"
    assert rec["turn_id"] == "turn-abc"
    assert rec["totals"]["latency_ms_total"] == 12.3
    assert rec["schema_version"]   # full trace persisted, not a summary


def test_disabled_writes_nothing(tmp_path):
    log_path = tmp_path / "traces.jsonl"
    tl = TraceLogger(str(log_path), enabled=False)

    tl.log(MemoryTrace(session_id="s", turn_index=1, turn_id="t"))

    assert not log_path.exists()