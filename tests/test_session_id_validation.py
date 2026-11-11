"""
API-free test: session_id validation + collision-proof collection naming.

Pure-logic unit tests plus one API-edge check that a bad session_id is rejected
with 422 (not a 500). No network, no OpenAI key: the TestClient is used without a
context manager so the lifespan (and its config.validate()) does not run, and the
422 path returns before any store/embedding call.

Run from repo root (venv active):
    python -m pytest tests/test_session_id_validation.py -v
"""

import pytest
from fastapi.testclient import TestClient

from memoryos.session_id import (
    validate_session_id, collection_name_for, InvalidSessionId,
    MAX_SESSION_ID_LENGTH,
)
from memoryos.api import app


# ── validate_session_id ──────────────────────────────────────────────────────

def test_valid_id_is_returned_stripped():
    assert validate_session_id("  user-42  ") == "user-42"
    assert validate_session_id("abc") == "abc"


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_empty_or_whitespace_rejected(bad):
    with pytest.raises(InvalidSessionId):
        validate_session_id(bad)


def test_overlong_rejected():
    with pytest.raises(InvalidSessionId):
        validate_session_id("x" * (MAX_SESSION_ID_LENGTH + 1))


@pytest.mark.parametrize("bad", ["foo\x00bar", "line\nbreak", "tab\ttab", "bell\x07"])
def test_control_characters_rejected(bad):
    with pytest.raises(InvalidSessionId):
        validate_session_id(bad)


def test_non_string_rejected():
    with pytest.raises(InvalidSessionId):
        validate_session_id(None)            # type: ignore[arg-type]


# ── collection_name_for ──────────────────────────────────────────────────────

def test_collection_name_is_deterministic_and_unique():
    base = "long_term_memory"
    a1 = collection_name_for(base, "session-a")
    a2 = collection_name_for(base, "session-a")
    b = collection_name_for(base, "session-b")
    assert a1 == a2            # same id → same name
    assert a1 != b             # different ids → different names


def test_collection_name_is_chroma_legal_for_hostile_input():
    # ids with characters that would be illegal in a raw collection name still
    # produce a legal name, because the suffix is a hex digest.
    name = collection_name_for("long_term_memory", "a/b c:d😀")
    suffix = name.rsplit("_", 1)[1]
    assert len(suffix) == 16
    assert all(c in "0123456789abcdef" for c in suffix)
    assert 3 <= len(name) <= 63


# ── API edge ─────────────────────────────────────────────────────────────────

def test_chat_rejects_blank_session_id_with_422():
    client = TestClient(app)   # no context manager → lifespan does not run
    resp = client.post("/chat", json={"session_id": "   ", "message": "hello there"})
    assert resp.status_code == 422