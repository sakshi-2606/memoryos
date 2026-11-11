"""
Session identity helpers — validation at the API edge, and collision-proof,
always-legal ChromaDB collection naming.

A session_id is the isolation unit: one session = one tenant, fully walled off
from every other. We validate it at the boundary so bad input is rejected cleanly
(a 422, never a 500), and we derive the physical collection name by HASHING the
id. That guarantees every accepted id maps to a unique, ChromaDB-legal collection
name regardless of its contents — no illegal characters reaching the store, and
no ambiguous separator between one id and the next.
"""

import hashlib
import re

MAX_SESSION_ID_LENGTH = 128

# Control characters (including newlines and tabs) are never allowed in an id.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class InvalidSessionId(ValueError):
    """Raised when a session_id fails validation."""


def validate_session_id(session_id: str) -> str:
    """
    Return the normalised session_id, or raise InvalidSessionId.

    Rules: a non-empty string after stripping surrounding whitespace, at most
    MAX_SESSION_ID_LENGTH characters, with no control characters. Charset is
    otherwise unrestricted — the physical collection name is hashed, so legality
    is guaranteed no matter what the id contains.
    """
    if not isinstance(session_id, str):
        raise InvalidSessionId("session_id must be a string")
    normalised = session_id.strip()
    if not normalised:
        raise InvalidSessionId("session_id must not be empty")
    if len(normalised) > MAX_SESSION_ID_LENGTH:
        raise InvalidSessionId(
            f"session_id must be at most {MAX_SESSION_ID_LENGTH} characters"
        )
    if _CONTROL_CHARS.search(normalised):
        raise InvalidSessionId("session_id must not contain control characters")
    return normalised


def collection_name_for(base: str, session_id: str) -> str:
    """
    Deterministic, collision-proof, always-legal ChromaDB collection name for a
    session — the base name plus a fixed 16-char hex digest of the session_id.
    """
    digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:16]
    return f"{base}_{digest}"