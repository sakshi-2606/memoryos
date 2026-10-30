"""
Observability sink for per-turn memory traces.

The MemoryTrace already carries correlation IDs and per-tier metrics; this module
persists each one as a JSON line (the durable structured log and the dashboard's
data source) and emits a one-line summary via the stdlib logger for stdout/ops.
"""

import json
import logging
from pathlib import Path

from memoryos.memory.trace import MemoryTrace

logger = logging.getLogger("memoryos.observability")


class TraceLogger:
    """Append-only JSONL sink for MemoryTrace records, plus a structured log line."""

    def __init__(self, log_path: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.log_path = Path(log_path)

    def log(self, trace: MemoryTrace) -> None:
        """Persist one turn's trace. No-op when disabled; never raises on IO error."""
        if not self.enabled:
            return

        record = trace.to_dict()

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            # Observability must never break the turn — log and move on.
            logger.exception("failed to append trace to %s", self.log_path)

        totals = record.get("totals", {})
        logger.info(
            "turn=%s turn_id=%s session=%s latency=%.1fms answer_tokens=%s",
            record.get("turn_index"),
            str(record.get("turn_id", ""))[:8],
            record.get("session_id"),
            totals.get("latency_ms_total", 0.0),
            totals.get("answer_tokens", 0),
        )