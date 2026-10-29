![CI](https://github.com/sakshi-2606/memoryos/actions/workflows/ci.yml/badge.svg)

# MemoryOS

> A tiered agent memory system: short-term buffer · long-term vector store · episodic summarisation

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-orange)

## Overview

Most LLM agents are stateless — every conversation starts from scratch. The naive fix is to stuff the full history into context, which breaks at scale; the next-naive fix is pure vector retrieval, which loses recent context fidelity. MemoryOS uses three coordinated tiers — a sliding-window buffer for recent turns, a ChromaDB vector store for long-term semantic retrieval, and an LLM-based episodic summarization layer that compresses evicted turns instead of losing them. Each tier handles what the others do badly.

## Architecture
```
User message
    │
    ▼
MemoryManager
    ├── ShortTermMemory   (sliding window, in-RAM, last N turns)
    ├── LongTermMemory    (ChromaDB vector store, semantic retrieval)
    └── EpisodicMemory    (LLM summarisation of evicted turns)
    │
    ▼
Prompt assembly (episodic → long-term → short-term)
    │
    ▼
LLM (gpt-4o-mini) → Response
```

## Evaluation

Rather than a single pass/fail score, MemoryOS ships a suite of memory-specific
checks that *can* fail (run with `pytest`):

- **Retrieval ranking** — relevant memories must rank above distractors (precision@k, recall)
- **Cross-session isolation** — a fact in one session must never surface in another
- **Episodic retention** — summarisation must not silently drop facts

The suite caught, and the codebase then fixed, two real bugs: a turn retrieving
its own query (write-before-read ordering), and an over-permissive similarity
threshold admitting unrelated memories. The long-term threshold (0.40) is
calibrated from data — see `scripts/threshold_sweep.py`.

## Features

- Three coordinated memory tiers: sliding window, ChromaDB vector store, episodic summaries
- Per-turn structured memory trace — tiers read/written, what was retrieved (with scores), what was evicted and why, per-tier latency and tokens
- Calibrated semantic retrieval with cross-session isolation
- Noise filter — rule-based gate before embedding to skip filler
- FastAPI REST server with session management; Streamlit memory visualiser
- pytest suite + GitHub Actions CI (offline guards gate every PR)

## Tech Stack

| Component | Tool |
|---|---|
| LLM | OpenAI gpt-4o-mini |
| Embeddings | text-embedding-3-small |
| Vector DB | ChromaDB |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Language | Python 3.11 |

## Getting Started

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation
```bash
git clone https://github.com/sakshi-2606/memoryos.git
cd memoryos
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
cp .env.example .env   # add your OpenAI API key
```

### Run the Streamlit UI
```bash
streamlit run app.py
```

### Run the FastAPI server
```bash
python -m uvicorn src.memoryos.api:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### Run the tests
​```bash
pytest -m "not integration"   # fast offline suite, no API key
pytest                        # full suite incl. live-API eval (needs OPENAI_API_KEY)
​```

## Project Structure
```
memoryos/
├── src/memoryos/
│   ├── config.py           # All tuneable parameters
│   ├── agent.py            # Stateful conversation loop
│   ├── eval.py             # Evaluation harness
│   ├── memory/
│   │   ├── short_term.py   # Sliding window buffer
│   │   ├── long_term.py    # ChromaDB vector store
│   │   ├── episodic.py     # LLM summarisation layer
│   │   └── manager.py      # Tier coordinator
├── app.py                  # Streamlit UI
├── .env.example
└── requirements.txt
```

## Roadmap

- [ ] Token-based eviction (replace turn-count window)
- [ ] Redis-backed session store for multi-instance deployment
- [ ] Local embedding model option (sentence-transformers)
- [ ] Async retrieval to reduce per-turn latency
