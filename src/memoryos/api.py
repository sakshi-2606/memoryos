import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from memoryos.config import AppConfig
from memoryos.session_store import InMemorySessionStore, SessionStore
from memoryos.session_id import validate_session_id, InvalidSessionId


# ── Request / Response Models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    turn: int
    response: str
    memory_metadata: dict
    trace: dict | None = None   # per-turn memory trace (when enabled + exposed)


class MemoryStateResponse(BaseModel):
    session_id: str
    turn_count: int
    short_term: list[dict]
    long_term_count: int
    episodes: list[dict]


class DeleteResponse(BaseModel):
    session_id: str
    deleted: bool


# ── Session store ────────────────────────────────────────────────────────────

_config: AppConfig = AppConfig()
# In-memory default. Swap for a Redis/persisted SessionStore in production.
_store: SessionStore = InMemorySessionStore(_config)


# ── App Setup ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    _config.validate()
    print(f"MemoryOS API started. Model: {_config.model.llm_model}")
    yield
    print(f"Shutting down. Active sessions: {len(_store)}")


app = FastAPI(
    title="MemoryOS API",
    description="Tiered agent memory system — short-term, long-term, episodic",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _valid_session_id(session_id: str) -> str:
    """Validate a session_id at the edge — reject bad input with 422, not 500."""
    try:
        return validate_session_id(session_id)
    except InvalidSessionId as e:
        raise HTTPException(status_code=422, detail=str(e))


def _run_turn(agent, session_id: str, message: str) -> dict:
    """
    Run one turn on a worker thread. The per-session lock is acquired HERE, on the
    worker thread — never on the event-loop thread across an await (that would
    deadlock two same-session requests). One session serialises; distinct sessions
    run on separate threads in parallel.
    """
    with _store.session_lock(session_id):
        return agent.chat(message)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness — the process is up."""
    return {
        "status": "ok",
        "active_sessions": len(_store),
        "model": _config.model.llm_model,
    }


@app.get("/ready")
async def ready():
    """Readiness — are dependencies actually usable? 503 if not."""
    checks = {"openai_key": bool(_config.openai_api_key)}
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=_config.storage.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        client.heartbeat()
        checks["chromadb"] = True
    except Exception:
        checks["chromadb"] = False

    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"ready": ok, "checks": checks},
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response. Creates a session if needed."""
    session_id = _valid_session_id(request.session_id)
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    agent = _store.get_or_create(session_id)
    # Offload the blocking turn to a worker thread so the event loop stays free —
    # distinct sessions run concurrently; one session serialises (lock is inside
    # _run_turn, on the worker thread).
    result = await asyncio.to_thread(_run_turn, agent, session_id, request.message)

    trace = result.get("trace") if _config.trace.include_in_response else None

    return ChatResponse(
        session_id=result["session_id"],
        turn=result["turn"],
        response=result["response"],
        memory_metadata=result["memory"],
        trace=trace,
    )


@app.get("/memory/{session_id}", response_model=MemoryStateResponse)
async def get_memory(session_id: str):
    """Inspect the full memory state for a session."""
    session_id = _valid_session_id(session_id)
    agent = _store.get(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return MemoryStateResponse(**agent.get_memory_state())


@app.delete("/memory/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: str):
    """Delete a session and all its memory."""
    session_id = _valid_session_id(session_id)
    # the purge does blocking Chroma IO — run it off the event loop
    deleted = await asyncio.to_thread(_store.delete, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return DeleteResponse(session_id=session_id, deleted=True)


@app.get("/sessions")
async def list_sessions():
    """List all active sessions — useful for debugging."""
    ids = _store.list_ids()
    return {"active_sessions": ids, "count": len(ids)}