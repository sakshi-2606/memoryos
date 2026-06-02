# MemoryOS

> A tiered agent memory system: short-term buffer · long-term vector store · episodic summarisation

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-orange)

## Overview

Most LLM agents are stateless — every conversation starts from scratch. MemoryOS gives an agent persistent, layered memory across three tiers: a sliding window buffer for recent context, a ChromaDB vector store for long-term semantic retrieval, and an LLM-based episodic summarisation layer that compresses old turns rather than losing them. An evaluation harness measures memory hit rate, faithfulness, and response quality lift.

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

## Evaluation Results

| Metric | Score |
|---|---|
| Memory hit rate | 100% |
| Episodic faithfulness | 1.000 |
| Memory lift (with vs without) | +0.100 |

## Features

- Three-tier memory architecture with automatic tier promotion
- Noise filter — rule-based gate before embedding to avoid storing filler
- Token-aware design — built for extension to token-budget eviction
- FastAPI REST server with session management
- Streamlit UI with real-time memory visualiser panel
- Evaluation harness measuring hit rate, faithfulness, and memory lift
- Per-turn memory activity log showing which tiers were read and written

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

### Run evaluation
```bash
python test_eval.py
```

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
