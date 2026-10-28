from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from memoryos.agent import MemoryAgent
from memoryos.config import AppConfig


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


# ── Session Store ──────────────────────────────────────────────────────────

# In-memory store: session_id → MemoryAgent
# In production: replace with Redis-backed store
_sessions: dict[str, MemoryAgent] = {}
_config: AppConfig = AppConfig()


def get_or_create_session(session_id: str) -> MemoryAgent:
    """Get existing agent or create a new one for this session."""
    if session_id not in _sessions:
        _sessions[session_id] = MemoryAgent(_config, session_id=session_id)
    return _sessions[session_id]


# ── App Setup ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    _config.validate()
    print(f"MemoryOS API started. Model: {_config.model.llm_model}")
    yield
    print(f"Shutting down. Active sessions: {len(_sessions)}")


app = FastAPI(
    title="MemoryOS API",
    description="Tiered agent memory system — short-term, long-term, episodic",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — used by load balancers and monitoring."""
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
        "model": _config.model.llm_model,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get a response.
    Creates a new session if session_id doesn't exist.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    agent = get_or_create_session(request.session_id)

    result = agent.chat(request.message)

    # Expose the trace over HTTP only when configured to. The agent still
    # builds it (for logging/dumps) based on config.trace.enabled; this flag
    # only controls whether it rides along in the response body.
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
    """
    Inspect the full memory state for a session.
    Powers the Streamlit memory visualiser panel.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )

    agent = _sessions[session_id]
    state = agent.get_memory_state()

    return MemoryStateResponse(**state)


@app.delete("/memory/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: str):
    """
    Delete a session and all its memory.
    The user can start fresh with the same session_id after this.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )

    del _sessions[session_id]
    return DeleteResponse(session_id=session_id, deleted=True)


@app.get("/sessions")
async def list_sessions():
    """List all active sessions — useful for debugging."""
    return {
        "active_sessions": list(_sessions.keys()),
        "count": len(_sessions),
    }