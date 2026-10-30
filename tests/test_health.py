"""
API-level test: liveness + readiness endpoints.

Uses FastAPI's TestClient (no network — readiness checks only the config flag and
a local ChromaDB heartbeat). Needs OPENAI_API_KEY set for the app's startup
validation (a dummy value is fine; no real calls are made).

Run from repo root (venv active):
    python -m pytest tests/test_health.py -v
"""

from fastapi.testclient import TestClient

from memoryos.api import app


def test_health_liveness():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_ready_reports_dependency_checks():
    with TestClient(app) as client:
        r = client.get("/ready")
        assert r.status_code in (200, 503)
        body = r.json()
        assert "ready" in body
        assert "openai_key" in body["checks"]
        assert "chromadb" in body["checks"]