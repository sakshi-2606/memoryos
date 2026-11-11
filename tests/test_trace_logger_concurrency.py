"""
API-free test: concurrent turns writing to the trace log produce intact JSONL —
no lost or interleaved lines.

Uses tmp_path (nothing written into the repo) and builds MemoryTrace objects
directly (no agent, no network).

Run from repo root (venv active):
    python -m pytest tests/test_trace_logger_concurrency.py -v
"""

import json
import threading

from memoryos.memory.trace import MemoryTrace
from memoryos.observability import TraceLogger


def test_concurrent_logging_writes_complete_lines(tmp_path):
    log_path = tmp_path / "traces.jsonl"
    tl = TraceLogger(str(log_path), enabled=True)

    n = 50
    barrier = threading.Barrier(n)

    def worker(i: int):
        barrier.wait()   # release all writers at once → maximise contention
        tl.log(MemoryTrace(session_id=f"s{i}", turn_index=i, turn_id=f"turn-{i}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == n                       # no lost writes

    seen = set()
    for line in lines:
        rec = json.loads(line)                   # each line is one complete JSON object
        seen.add(rec["turn_id"])
    assert seen == {f"turn-{i}" for i in range(n)}   # every turn present, none torn